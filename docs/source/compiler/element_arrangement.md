Element arrangement
===

Restore EA D2H
---

Restore STANDARD element arrangement in a D2H copy of a tensor whose
SpyreTensorLayout has a staggered element arrangement.

---------------------------------------------------------------------------
1. What the device holds (DL16_TO_FP32)
---------------------------------------------------------------------------
A 16-bit tensor was widened to fp32 on the device.  The device stores the
fp32 values in sticks of 32 elements, and the elements are NOT in logical
order.  Terminology:

  element group   4 consecutive host elements; always stay together.
  stick pair      2 adjacent device sticks that together hold 64 host
                  columns (2 sticks * 32 elements).

Within one stick pair the element groups are dealt out alternately to the
two sticks (even groups -> first stick, odd groups -> second stick):

  Host row, one stick pair = 64 columns = 16 element groups G0..G15:

    col   0..3  4..7  8..11 12..15 16..19 20..23 ...  60..63
        +-----+-----+-----+------+------+------+-----+------+
  host  | G0  | G1  | G2  | G3   | G4   | G5   | ... | G15  |
        +-----+-----+-----+------+------+------+-----+------+

  Device, same stick pair, 2 sticks of 8 element groups (32 elements):

    pos   0..3  4..7  8..11 12..15 16..19 20..23 24..27 28..31
        +-----+-----+-----+------+------+------+------+------+
  stick0| G0  | G2  | G4  | G6   | G8   | G10  | G12  | G14  |
        +-----+-----+-----+------+------+------+------+------+
  stick1| G1  | G3  | G5  | G7   | G9   | G11  | G13  | G15  |
        +-----+-----+-----+------+------+------+------+------+

  Reading D2H: walk the host row left to right; G0 comes from stick0
  pos 0..3, G1 from stick1 pos 0..3, G2 from stick0 pos 4..7, and so on.

  i.e. host group G = 2 * group_in_stick + stick_in_pair.

With more than one stick pair the pattern repeats: stick pair p covers host
columns [64 *p, 64* p + 64) and device sticks 2p and 2p + 1.

---------------------------------------------------------------------------
1. The index mapping
---------------------------------------------------------------------------
Every host column of a row decomposes uniquely as

  host_col = stick_pair     *64     (2 sticks* 32 elements)
           + group_in_stick *8     (skip the other stick's group too)
           + stick_in_pair*  4     (which of the two sticks)
           + elem_in_group           (0..3)

  stick_pair      in [0, num_sticks / 2)
  group_in_stick  in [0, 8)
  stick_in_pair   in [0, 2)
  elem_in_group   in [0, 4)

and the same element lives on the device at

  device_stick    = 2 *stick_pair + stick_in_pair
  device_position = group_in_stick* 4 + elem_in_group      (0..31)

---------------------------------------------------------------------------
1. The strategy: split loops, do not move data
---------------------------------------------------------------------------
A DataConversionStrideInfo (DCSI) is a loop nest, innermost loop first:

  for each index i_k in [0, size_[k]):
    dst[sum_k i_k *stride_dst_[k]] = src[sum_k i_k* stride_src_[k]]

For D2H, src is the device buffer and dst is the host buffer.  The incoming
DCSI assumes the device is in standard order: dimension 0 walks the
32 elements of a stick, and one "stick dimension" walks successive sticks.
That is wrong for a staggered layout.

Instead of adding a shuffling pass we rewrite the loop nest so the same copy
engine un-staggers while it copies.  Two of the original dimensions are each
split into two smaller loops that have *different* src and dst strides:

original                  new loops (size, src step, dst step in elems)
---------------------------------------------------------------------------
  dim 0 (32 elements)  ->   elem_in_group  : 4,   1 elem,    1 col
                            group_in_stick : 8,   4 elems,   8 cols

  stick dim (N sticks) ->   stick_in_pair  : 2,   1 stick,    4 cols
                            stick_pair     : N/2, 2 sticks,  64 cols

  all other dims       ->   unchanged

Read the table as "taking one step in this loop moves the device (src)
pointer by X and the host (dst) pointer by Y".  For example group_in_stick
advances 4 elements within a device stick but jumps 8 columns on the host,
because the 4 columns in between belong to the other stick of the pair;
stick_in_pair does the reverse, moving a whole device stick on the device
but only 4 columns on the host.  The "stick pair" loop then moves two
sticks / 64 columns at a time.  Strides are always expressed in multiples
of the dimension-0 strides, so the element size does not matter.

The stick dimension is identified by its HOST stride (one stick's worth of
host columns), not by its device stride, because the device is free to
order sticks differently (e.g. stick-major).

Requirements (checked): full sticks only, and an even number of sticks along
the stick dimension.
