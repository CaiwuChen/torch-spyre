# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IR-level pass to pad y's K (row) dimension to a stick boundary for
BATCH_MATMUL_OP operations.  Runs in CustomPreSchedulingPasses immediately
after insert_restickify, when every ComputedBuffer has a FixedTiledLayout.

Only y is padded; x is left untouched.

For y, the following IR sequence is emitted:
  1. ComputedBuffer - output buffer allocation (FixedLayout)
  2. SpyreConstantFallback - fill constant (FixedLayout)
  3. ComputedBuffer - fill padding region (MutationLayoutSHOULDREMOVE)
  4. ComputedBuffer - copy input data (MutationLayoutSHOULDREMOVE)

y's padded buffer is built at the full K_padded host size by lower_pad_sequence.
reduction_ranges stays at K; the K→K_padded extension happens at SDSC codegen
time: _extend_matmul_k_to_padded in superdsc.py reads K_padded from y's
device_size and widens sdsc_iteration_space[K] to K_padded before
_create_sdsc_tensors runs.

x is left physically untouched.  The hardware masks within-stick elements of x
beyond the true K to zero, so extending the SDSC iteration to K_padded does not
introduce numerical error from x.

Deduplication of identical constants across multiple pad calls happens later
at the IR level via dedup_and_promote_constants.

x and y are identified via identify_matmul_inputs() using the BatchMatmul
generated_dim definition: y is the input whose index contains a symbol
present in the output but absent from x (N).  This handles M==K==N and
M=1 (decode phase) correctly.
"""

import torch
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    Buffer,
    ComputedBuffer,
    Operation,
    Pointwise,
    Reduction,
    TensorBox,
)
from torch._inductor.virtualized import V

from .constants import BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .pass_utils import (
    concretize_expr,
    concretize_index,
    find_reduction_var,
    host_coordinates,
    identify_matmul_inputs,
    is_stick_expr_offset_free,
    lower_pad_sequence,
    replace_computed_buffer_body,
)
from .views import compute_coordinates, matching_dim
from torch_spyre._C import get_elem_in_stick, SpyreTensorLayout

logger = get_inductor_logger("padding")


def compute_padding(cur_size: int, dtype: torch.dtype) -> int:
    stick_size = get_elem_in_stick(dtype)
    pad = (stick_size - (cur_size % stick_size)) % stick_size
    return pad


def _patch_env(graph_lowering) -> None:
    """Add view nodes (ReinterpretView) to env from name_to_users."""
    env: dict = {}
    for tbs in graph_lowering.name_to_users.values():
        for tb in tbs:
            if not tb.data.origins:
                continue
            tb_fx_node = list(tb.data.origins)[0]
            env[tb_fx_node] = tb
    graph_lowering.env.update(env)


def _find_arg_fx_node(arg_name: str) -> torch.fx.Node:
    """Return the FX node whose lowered TensorBox has the given buffer name.

    Buffer names are unique, but a single buffer can be reached through
    multiple FX nodes that present it at different sizes.  For example,
    mm_to_bmm_pass inserts an unsqueeze/reshape so the matmul inner_fn
    indexes x as 3D [1, M, K] even though the underlying buffer is 2D
    [M, K].  Both FX nodes lower to a TensorBox whose get_name() returns
    the same buffer name, but with different get_size() results.

    Returns the first candidate (the base buffer, with no view applied).
    Raises RuntimeError if no candidate exists.
    """
    graph_lowering = V.graph
    _patch_env(graph_lowering)
    candidates = [
        fx_node
        for fx_node, tb in graph_lowering.env.items()
        if isinstance(fx_node, torch.fx.Node)
        and isinstance(tb, TensorBox)
        and tb.get_name() == arg_name
    ]
    if not candidates:
        raise RuntimeError(f"no FX node found for buffer {arg_name!r}")
    return candidates[0]


def _rebuild_matmul(
    op: ComputedBuffer,
    y_padded_buf: Buffer,
    operations: list[Operation],
) -> ComputedBuffer:
    """Rebuild the matmul ComputedBuffer so y's loader reads from the padded buffer.

    Preserves the original inner_fn's x loading unchanged; only replaces y's
    loader with one that reads from the padded buffer.  reduction_ranges stays
    at K; the K→K_padded extension happens at SDSC codegen time via
    _extend_matmul_k_to_padded in superdsc.py.
    """
    reduction = op.data
    assert isinstance(reduction, Reduction)

    orig_inner_fn = reduction.inner_fn
    y_padded_loader = y_padded_buf.make_loader()
    y_ndim = len(y_padded_buf.get_size())
    y_batch_ndim = y_ndim - 2

    def new_inner_fn(
        index,
        reduction_index,
        _orig_inner_fn=orig_inner_fn,
        _y_loader=y_padded_loader,
        _y_batch_ndim=y_batch_ndim,
    ):
        # x_val comes from the original inner_fn; discard its y and replace below.
        x_val, _ = _orig_inner_fn(index, reduction_index)
        y_index = list(index[:_y_batch_ndim]) + list(reduction_index) + [index[-1]]
        y_val = _y_loader(y_index)
        return (x_val, y_val)

    object.__setattr__(reduction, "inner_fn", new_inner_fn)
    # reduction_ranges stays at K; no extension here.

    return replace_computed_buffer_body(op, reduction, operations)


def insert_bmm_padding(graph: GraphLowering) -> None:
    """
    Pad y's K (row) dimension for each BATCH_MATMUL_OP to a stick boundary.

    Mutates ``operations`` in place.  New buffers for y are inserted immediately
    before the matmul that consumes them to preserve topological order.

    x is left entirely untouched.  y's padded buffer is built at K_padded host
    size by lower_pad_sequence; reduction_ranges stays at K so the IR iteration
    space is unchanged.  The K→K_padded widening happens at SDSC codegen time.

    x and y are identified via identify_matmul_inputs() using the BatchMatmul
    generated_dim definition: y is the input whose index contains a symbol
    present in the output but absent from x (N).  This handles M==K==N and
    M=1 (decode phase) correctly.

    Deduplication of identical constants across multiple pad calls happens later
    at the IR level via dedup_and_promote_constants.
    """
    operations = graph.operations
    for op in list(operations):
        if not isinstance(op, ComputedBuffer):
            continue
        reduction = op.data
        if not isinstance(reduction, Reduction):
            continue
        if reduction.reduction_type != BATCH_MATMUL_OP:
            continue

        rw = op.get_read_writes()
        reads = [r for r in rw.reads if hasattr(r, "name")]
        if len(reads) != 2:  # noqa: PLR2004
            continue

        # Skip aligned-K matmuls early before any x/y identification.
        # Aligned-K matmuls need no padding regardless of input layout, and
        # skipping here avoids a spurious warning for e.g. decode-phase SDPA
        # attention where constant-folded dimensions cause identify_matmul_inputs
        # to fail.
        k_val = concretize_expr(reduction.reduction_ranges[0])
        first_buf = next(
            (graph.get_buffer(d.name) for d in reads if graph.get_buffer(d.name)),
            None,
        )
        assert first_buf is not None, (
            f"insert_bmm_padding: no input buffer found for matmul {op.get_name()}"
        )
        dtype = first_buf.get_dtype()
        if compute_padding(k_val, dtype) == 0:
            continue

        write_dep = next(iter(rw.writes))
        x_dep, y_dep = identify_matmul_inputs(reads, write_dep)
        if x_dep is None or y_dep is None:
            logger.warning(
                "insert_bmm_padding: could not identify x/y for %s, skipping",
                op.get_name(),
            )
            continue

        reduction_var = find_reduction_var(x_dep, write_dep)

        # y's K host dim: the dim whose host coordinate contains reduction_var.
        y_buf_tmp = graph.get_buffer(y_dep.name)
        y_host_k_dim: int | None = None
        if y_buf_tmp is not None and isinstance(
            y_buf_tmp.get_layout(), FixedTiledLayout
        ):
            y_h_coords = host_coordinates(y_buf_tmp.get_layout(), y_dep, None)
            y_host_k_dim = next(
                (
                    i
                    for i, c in enumerate(y_h_coords)
                    if reduction_var in c.free_symbols
                ),
                None,
            )

        x_name = x_dep.name
        y_name = y_dep.name
        x_buf = graph.get_buffer(x_name)
        y_buf = graph.get_buffer(y_name)
        if x_buf is None or y_buf is None:
            continue

        device = x_buf.get_device()
        pad = compute_padding(k_val, dtype)

        k_padded = k_val + pad

        logger.debug(
            "insert_bmm_padding: padding %s K=%d -> K=%d (pad=%d)",
            op.get_name(),
            k_val,
            k_padded,
            pad,
        )

        # The FX node for the matmul is used as the insertion anchor so padding
        # nodes are placed immediately before the matmul in the FX graph,
        # minimising their live range.
        matmul_fx_node = next(iter(op.origins))

        # --- Pad y only ---
        # y's K dimension is y's row (mb) dimension.  Padding it to K_padded
        # ensures rows K..K_padded-1 of y are zero-filled so the hardware
        # accumulates no contribution from those rows.
        # lower_pad_sequence builds the padded buffer at K_padded host size;
        # reduction_ranges is NOT changed.  superdsc._extend_matmul_k_to_padded
        # widens sdsc_iteration_space[K] to K_padded at SDSC codegen time,
        # reading K_padded from y's device_layout.device_size.
        y_size = [concretize_expr(s) for s in y_buf.get_size()]
        if y_host_k_dim is None:
            y_k_dim = len(y_size) - 2
        else:
            y_k_dim = y_host_k_dim
        y_padded_size = list(y_size)
        y_padded_size[y_k_dim] = k_padded
        y_fx_node = _find_arg_fx_node(y_name)

        y_orig_stl = y_buf.get_layout().device_layout
        y_padded_buf, y_new_ops = lower_pad_sequence(
            y_fx_node,
            padded_size=y_padded_size,
            device=device,
            dtype=dtype,
            dim=y_k_dim,
            insert_before=matmul_fx_node,
            orig_stl=y_orig_stl,
        )

        # --- Relocate new ops before the matmul ---
        # run_node appended them at the end of operations; move before op.
        for new_op in y_new_ops:
            operations.remove(new_op)
        op_idx = operations.index(op)
        for i, new_op in enumerate(y_new_ops):
            operations.insert(op_idx + i, new_op)

        # --- Rebuild matmul inner_fn to load y from the padded buffer ---
        # x is left entirely untouched: the original inner_fn's x loader is
        # preserved as-is.  Only y's loader is replaced with the padded buffer.
        _rebuild_matmul(op, y_padded_buf, operations)


# --------------------------------------------------------------------------- #
# insert_restickify_padding                                                   #
# --------------------------------------------------------------------------- #


def _project_stick_host_dim(
    host_layout: FixedTiledLayout,
    stick_layout: FixedTiledLayout,
    dep,
    allow_offset: bool = False,
) -> int | None:
    """Return the host_layout host dim carrying stick_layout's within-stick
    coord under dep, or None if no unique canonical match exists.

    When host_layout is stick_layout this is the buffer's own within-stick
    host dim; when they differ, stick_layout's STL is projected through dep.
    Returns None unless the stick coord is valid (offset-free, unless
    allow_offset=True in which case a constant additive offset is permitted).
    """
    host_coords = host_coordinates(host_layout, dep, None)
    stl = stick_layout.device_layout
    device_index = concretize_index(dep.index, set(dep.ranges.keys()))
    device_coords = compute_coordinates(
        stl.device_size, stl.stride_map, dep.ranges, device_index
    )
    if not host_coords or not device_coords:
        return None
    stick_expr = device_coords[-1]
    elems = stl.elems_per_stick()
    if is_stick_expr_offset_free(stick_expr, elems):
        return matching_dim(host_coords, stick_expr)
    if not allow_offset:
        return None
    # Strip the constant offset; the variable part must be a valid stick expression.
    free = stick_expr.free_symbols
    if not free:
        return None
    var_part = stick_expr - stick_expr.subs({s: 0 for s in free})
    if not is_stick_expr_offset_free(var_part, elems):
        return None
    # Strip offsets from host coords so matching_dim sees bare variables.
    stripped_host = [
        (c - c.subs({s: 0 for s in c.free_symbols}) if c.free_symbols else c)
        for c in host_coords
    ]
    return matching_dim(stripped_host, var_part)


def _restickify_input_dep(op: Operation, graph: GraphLowering):
    """Return (in_dep, in_buf, in_layout, host_size, new_stick_dim) when op
    is a single-input pointwise copy whose output STL puts a different host
    dim within the stick than the input's does, else None.

    Both stick dims are recovered in the input's host frame so they are
    directly comparable; the cross-buffer projection makes transpose work
    while reduce drops out as non-Pointwise and flatten drops out via the
    canonical-form filter in _project_stick_host_dim.  Sliced inputs are
    handled by insert_restickify_padding via allow_offset=True.
    """
    if not isinstance(op, ComputedBuffer):
        return None
    out_layout = op.get_layout()
    if not isinstance(out_layout, FixedTiledLayout):
        return None
    if not isinstance(op.data, Pointwise):
        return None

    rw = op.get_read_writes()
    reads = [r for r in rw.reads if hasattr(r, "name")]
    if len(reads) != 1:
        return None
    in_dep = reads[0]

    in_buf = graph.get_buffer(in_dep.name)
    if in_buf is None:
        return None
    in_layout = in_buf.get_layout()
    if not isinstance(in_layout, FixedTiledLayout):
        return None

    in_stick_dim = _project_stick_host_dim(
        in_layout, in_layout, in_dep, allow_offset=True
    )
    new_stick_dim = _project_stick_host_dim(
        in_layout, out_layout, in_dep, allow_offset=True
    )
    if in_stick_dim is None or new_stick_dim is None:
        return None
    if new_stick_dim == in_stick_dim:
        return None

    host_size = [concretize_expr(s) for s in in_layout.size]
    return in_dep, in_buf, in_layout, host_size, new_stick_dim


def _find_slice_fx_node(
    in_dep, iter_extents: list[int], offsets: list[int]
) -> torch.fx.Node:
    """Return the FX node for a sliced view of the buffer named in_dep.name.

    For sliced inputs Inductor lowers the slice to a ReinterpretView sharing
    the same buffer name as the base but with a smaller shape and a non-zero
    storage offset.  We locate it by matching shape and, when multiple
    candidates exist, by matching the expected storage offset derived from the
    host strides and per-dim slice offsets.
    """
    graph_lowering = V.graph
    _patch_env(graph_lowering)
    target_shape = list(iter_extents)
    candidates = [
        fx_node
        for fx_node, tb in graph_lowering.env.items()
        if isinstance(fx_node, torch.fx.Node)
        and isinstance(tb, TensorBox)
        and tb.get_name() == in_dep.name
        and list(fx_node.meta["val"].shape) == target_shape
    ]
    if not candidates:
        raise RuntimeError(
            f"_find_slice_fx_node: no FX node with name={in_dep.name!r} "
            f"and shape={target_shape}"
        )
    if len(candidates) == 1:
        return candidates[0]
    # Break ties by matching storage_offset against the expected linear offset.
    in_buf = graph_lowering.get_buffer(in_dep.name)
    if in_buf is not None:
        host_stride = [int(s) for s in in_buf.get_layout().stride]
        expected_offset = sum(o * s for o, s in zip(offsets, host_stride))
        for fx_node in candidates:
            val = fx_node.meta["val"]
            actual_so = (
                int(val.storage_offset()) if hasattr(val, "storage_offset") else None
            )
            if actual_so is not None and actual_so == expected_offset:
                return fx_node
    return candidates[0]


def insert_restickify_padding(graph: GraphLowering) -> None:
    """Zero-pad a Restickify's input along the dim that becomes the new
    stick dim, when its extent is not a multiple of the stick size.

    Without padding, codegen widens the iteration space to a stick boundary
    and reads past the true extent — those tail elements come from
    uninitialized HBM and end up in the output, producing a value mismatch.

    Strategy: insert a stick-aligned, zero-filled copy of the input ahead of
    the Restickify (lower_pad_sequence) and rewrite the Restickify body to
    load from the padded buffer through a permuted index that maps each
    output iter dim to the corresponding input host dim.  The Restickify's
    ranges, layout, and device_layout are left untouched; codegen's existing
    stick-boundary widening reads from the zero-filled tail of the padded
    buffer instead of uninitialized HBM.

    Sliced inputs are handled by using the slice's iter extents for the
    padded buffer shape, and by finding the already-lowered slice FX node
    so that lower_pad_sequence sees the correct shape and storage offset.
    """
    operations = graph.operations
    for op in list(operations):
        match = _restickify_input_dep(op, graph)
        if match is None:
            continue
        in_dep, in_buf, in_layout, host_size, new_stick_dim = match

        dtype = in_layout.dtype
        device = in_buf.get_device()
        if device is None:
            continue

        in_host_coords = host_coordinates(in_layout, in_dep, None)
        old_iter_syms = list(in_dep.ranges.keys())
        perm: list[int] = []
        iter_extents: list[int] = []
        offsets: list[int] = []
        for i, coord in enumerate(in_host_coords):
            picks = [j for j, s in enumerate(old_iter_syms) if s in coord.free_symbols]
            if len(picks) == 0:
                # size-1 dim: constant coordinate, no loop variable.
                # Use a placeholder perm entry pointing to the first sym; the
                # loader will receive index[0] which equals 0 for a size-1 dim.
                iter_extents.append(1)
                offsets.append(0)
                perm.append(0)
                continue
            assert len(picks) == 1, "_restickify_input_dep should have ensured this"
            sym = old_iter_syms[picks[0]]
            iter_extents.append(concretize_expr(in_dep.ranges[sym]))
            offsets.append(int(coord.subs(sym, 0)) if hasattr(coord, "subs") else 0)
            perm.append(picks[0])

        n = iter_extents[new_stick_dim]
        pad = compute_padding(n, dtype)
        if pad == 0:
            continue

        padded_size = list(iter_extents)
        padded_size[new_stick_dim] = n + pad

        # For contiguous inputs use the base FX node directly.  For sliced
        # inputs find the already-lowered ReinterpretView whose shape matches
        # iter_extents and whose storage offset matches the slice offsets.
        base_fx = _find_arg_fx_node(in_dep.name)
        if list(base_fx.meta["val"].shape) != iter_extents:
            in_fx = _find_slice_fx_node(in_dep, iter_extents, offsets)
        else:
            in_fx = base_fx

        restickify_fx = next(iter(op.origins))

        # For sliced inputs orig_stl's device_size reflects the full backing
        # buffer while in_fx reflects the slice extent.  Patch orig_stl's
        # device_size for each sliced dim so _build_padded_stl can match
        # device dims against the slice extent rather than the full size.
        orig_stl = in_layout.device_layout
        if iter_extents != host_size:
            old_ds = list(orig_stl.device_size)
            old_sm = list(orig_stl.stride_map)
            new_ds = list(old_ds)
            for host_dim, (ie, hs) in enumerate(zip(iter_extents, host_size)):
                if ie != hs:
                    host_stride_val = int(in_layout.stride[host_dim])
                    for k, (sm, ds) in enumerate(zip(old_sm[:-1], old_ds[:-1])):
                        if sm == host_stride_val and ds == hs:
                            new_ds[k] = ie
                            break
            orig_stl = SpyreTensorLayout(new_ds, old_sm, orig_stl.device_dtype)

        padded_buf, new_ops = lower_pad_sequence(
            in_fx,
            padded_size=padded_size,
            device=device,
            dtype=dtype,
            dim=new_stick_dim,
            insert_before=restickify_fx,
            orig_stl=orig_stl,
            fill_value=0.0,
        )

        # Move pad ops to just before the restickify (lower_pad_sequence appends).
        for o in new_ops:
            operations.remove(o)
        idx = operations.index(op)
        for i, o in enumerate(new_ops):
            operations.insert(idx + i, o)

        # Replace the restickify body with a Pointwise that reads padded_buf
        # through a permuted index: input host dim i is addressed by output iter
        # index perm[i], where perm[i] is the lone iter sym in in_host_coords[i].
        # op.data.ranges stays at the logical output extent; the stick-boundary
        # widening happens later in superdsc's _extend_restickify_to_padded
        # (Inductor's _simplify_loops would undo it if done here).
        old_pw = op.data
        padded_loader = padded_buf.make_loader()
        new_pw = Pointwise(
            device=old_pw.device,
            dtype=old_pw.dtype,
            inner_fn=lambda index, _loader=padded_loader, _perm=tuple(perm): _loader(
                [index[p] for p in _perm]
            ),
            ranges=old_pw.ranges,
        )
        replace_computed_buffer_body(op, new_pw, operations)
