//! BC7 mode 6 decode: the single-subset, full-`RGBA`, highest-quality opaque
//! and smooth-alpha mode of the BPTC (`BC7`) format.
//!
//! BC7 packs eight modes into a 16-byte block; the active mode is a unary code
//! in the low bits of byte 0 (mode `m` = `m` zero bits then a `1`). The other
//! seven modes add 2/3-subset partitioning (a 64-entry partition table), index
//! rotation, and p-bit sharing variants. **Mode 6** is the one mode with *no*
//! partition table: a single subset, two `RGBA` endpoints at 7 bits/channel
//! each plus a unique p-bit per endpoint (giving 8-bit endpoints), and 4-bit
//! indices for all sixteen texels. That makes it fully reproducible from the
//! block bits alone -- no transcribed partition/anchor tables -- so a CPU
//! golden matches a GPU twin exactly (integer interpolation, no `AI/ML`).
//!
//! Mode 6 is also the mode AAA compressors fall back to for high-fidelity
//! `RGBA` where a single subset already captures the block, so decoding it
//! covers a large fraction of shipped BC7 content. The remaining partitioned
//! modes (0-5, 7) are tracked as a follow-up: they require the Khronos
//! partition and anchor-index tables, which must be validated against a
//! reference decoder before they can be claimed, and are therefore kept out of
//! this module rather than stubbed.
//!
//! # Conventions
//! * The block is little-endian: bit `i` is `block[i / 8] >> (i % 8) & 1`.
//!   Fields are read LSB-first in Khronos field order.
//! * Output is row-major `RGBA8`, texel `t = y * 4 + x`, `t in [0, 16)`.
//! * Endpoint channels expand 7-bit + p-bit to 8-bit as `(v << 1) | p`; this is
//!   exact, so endpoint texels (`index` weight 0 or 64) decode bit-exactly and
//!   interior texels are deterministic (no `+/-1 LSB` ambiguity, unlike the
//!   truncating S3TC palette).
//!
//! # References
//! * Khronos Data Format Specification 1.3, BPTC (`BC7`) block decode.
//! * Microsoft `DXGI_FORMAT_BC7_*` / Vulkan `VK_FORMAT_BC7_*` BPTC spec.

use super::bitio::BitReader;
// Weights and the 2-subset partition/anchor tables are shared with BC6H and
// live in `bptc_tables`; the 3-subset tables below remain BC7-private.
use super::bptc_tables::{BPTC_ANCHORS_2, BPTC_PARTITIONS_2, WEIGHT2, WEIGHT3, WEIGHT4};

/// BPTC 3-subset partition table (Khronos Data Format Spec, "Partition Table
/// for 3 Subsets"). `BPTC_PARTITIONS_3[p][t]` is the subset (`0`, `1`, or `2`)
/// of texel `t = y*4 + x` under partition `p in 0..64`. Shared by BC7 modes
/// 0 and 2. Every row has `partition[0] == 0`, so texel 0 is subset 0's anchor.
#[rustfmt::skip]
const BPTC_PARTITIONS_3: [[u8; 16]; 64] = [
    [0,0,1,1,0,0,1,1,0,2,2,1,2,2,2,2], [0,0,0,1,0,0,1,1,2,2,1,1,2,2,2,1],
    [0,0,0,0,2,0,0,1,2,2,1,1,2,2,1,1], [0,2,2,2,0,0,2,2,0,0,1,1,0,1,1,1],
    [0,0,0,0,0,0,0,0,1,1,2,2,1,1,2,2], [0,0,1,1,0,0,1,1,0,0,2,2,0,0,2,2],
    [0,0,2,2,0,0,2,2,1,1,1,1,1,1,1,1], [0,0,1,1,0,0,1,1,2,2,1,1,2,2,1,1],
    [0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2], [0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2],
    [0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2], [0,0,1,2,0,0,1,2,0,0,1,2,0,0,1,2],
    [0,1,1,2,0,1,1,2,0,1,1,2,0,1,1,2], [0,1,2,2,0,1,2,2,0,1,2,2,0,1,2,2],
    [0,0,1,1,0,1,1,2,1,1,2,2,1,2,2,2], [0,0,1,1,2,0,0,1,2,2,0,0,2,2,2,0],
    [0,0,0,1,0,0,1,1,0,1,1,2,1,1,2,2], [0,1,1,1,0,0,1,1,2,0,0,1,2,2,0,0],
    [0,0,0,0,1,1,2,2,1,1,2,2,1,1,2,2], [0,0,2,2,0,0,2,2,0,0,2,2,1,1,1,1],
    [0,1,1,1,0,1,1,1,0,2,2,2,0,2,2,2], [0,0,0,1,0,0,0,1,2,2,2,1,2,2,2,1],
    [0,0,0,0,0,0,1,1,0,1,2,2,0,1,2,2], [0,0,0,0,1,1,0,0,2,2,1,0,2,2,1,0],
    [0,1,2,2,0,1,2,2,0,0,1,1,0,0,0,0], [0,0,1,2,0,0,1,2,1,1,2,2,2,2,2,2],
    [0,1,1,0,1,2,2,1,1,2,2,1,0,1,1,0], [0,0,0,0,0,1,1,0,1,2,2,1,1,2,2,1],
    [0,0,2,2,1,1,0,2,1,1,0,2,0,0,2,2], [0,1,1,0,0,1,1,0,2,0,0,2,2,2,2,2],
    [0,0,1,1,0,1,2,2,0,1,2,2,0,0,1,1], [0,0,0,0,2,0,0,0,2,2,1,1,2,2,2,1],
    [0,0,0,0,0,0,0,2,1,1,2,2,1,2,2,2], [0,2,2,2,0,0,2,2,0,0,1,2,0,0,1,1],
    [0,0,1,1,0,0,1,2,0,0,2,2,0,2,2,2], [0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,0],
    [0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0], [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0],
    [0,1,2,0,2,0,1,2,1,2,0,1,0,1,2,0], [0,0,1,1,2,2,0,0,1,1,2,2,0,0,1,1],
    [0,0,1,1,1,1,2,2,2,2,0,0,0,0,1,1], [0,1,0,1,0,1,0,1,2,2,2,2,2,2,2,2],
    [0,0,0,0,0,0,0,0,2,1,2,1,2,1,2,1], [0,0,2,2,1,1,2,2,0,0,2,2,1,1,2,2],
    [0,0,2,2,0,0,1,1,0,0,2,2,0,0,1,1], [0,2,2,0,1,2,2,1,0,2,2,0,1,2,2,1],
    [0,1,0,1,2,2,2,2,2,2,2,2,0,1,0,1], [0,0,0,0,2,1,2,1,2,1,2,1,2,1,2,1],
    [0,1,0,1,0,1,0,1,0,1,0,1,2,2,2,2], [0,2,2,2,0,1,1,1,0,2,2,2,0,1,1,1],
    [0,0,0,2,1,1,1,2,0,0,0,2,1,1,1,2], [0,0,0,0,2,1,1,2,2,1,1,2,2,1,1,2],
    [0,2,2,2,0,1,1,1,0,1,1,1,0,2,2,2], [0,0,0,2,1,1,1,2,1,1,1,2,0,0,0,2],
    [0,1,1,0,0,1,1,0,0,1,1,0,2,2,2,2], [0,0,0,0,0,0,0,0,2,1,1,2,2,1,1,2],
    [0,1,1,0,0,1,1,0,2,2,2,2,2,2,2,2], [0,0,2,2,0,0,1,1,0,0,1,1,0,0,2,2],
    [0,0,2,2,1,1,2,2,1,1,2,2,0,0,2,2], [0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,2],
    [0,0,0,2,0,0,0,1,0,0,0,2,0,0,0,1], [0,2,2,2,1,2,2,2,0,2,2,2,1,2,2,2],
    [0,1,0,1,2,2,2,2,2,2,2,2,2,2,2,2], [0,1,1,1,2,0,1,1,2,2,0,1,2,2,2,0],
];

/// Anchor index of the SECOND subset for 3-subset partitioning (Khronos
/// "Fixup" table, subset 1). Subset 0's anchor is always texel 0.
#[rustfmt::skip]
const BPTC_ANCHORS_3_2: [usize; 64] = [
     3, 3,15,15, 8, 3,15,15, 8, 8, 6, 6, 6, 5, 3, 3,
     3, 3, 8,15, 3, 3, 6,10, 5, 8, 8, 6, 8, 5,15,15,
     8,15, 3, 5, 6,10, 8,15,15, 3,15, 5,15,15,15,15,
     3,15, 5, 5, 5, 8, 5,10, 5,10, 8,13,15,12, 3, 3,
];

/// Anchor index of the THIRD subset for 3-subset partitioning (Khronos
/// "Fixup" table, subset 2).
#[rustfmt::skip]
const BPTC_ANCHORS_3_3: [usize; 64] = [
    15, 8, 8, 3,15,15, 3, 8,15,15,15,15,15,15,15, 8,
    15, 8,15, 3,15, 8,15, 8, 3,15, 6,10,15,15,10, 8,
    15, 3,15,10,10, 8, 9,10, 6,15, 8,15, 3, 6, 6, 8,
    15, 3,15,15,15,15,15,15,15,15,15,15, 3,15,15, 8,
];

/// Return the BC7 mode (`0..=7`) encoded in the block's unary prefix, or
/// `None` when byte 0 is zero (a reserved/invalid encoding).
#[inline]
#[must_use]
pub fn bc7_mode(block: &[u8; 16]) -> Option<u8> {
    if block[0] == 0 {
        None
    } else {
        // A nonzero `u8` has `trailing_zeros` in `0..=7`, exactly the mode.
        Some(block[0].trailing_zeros() as u8)
    }
}

/// Expand a 7-bit endpoint channel plus its p-bit to 8 bits (`(v << 1) | p`).
#[inline]
fn expand7(v: u32, p: u32) -> u8 {
    (((v << 1) | (p & 1)) & 0xFF) as u8
}

/// Expand a `bits`-wide endpoint channel (no p-bit) to 8 bits by MSB
/// replication, matching the Khronos BPTC `unquantize` for precisions below
/// 8. For a 7-bit value this is `(v << 1) | (v >> 6)`.
#[inline]
fn expand_rep(v: u32, bits: u32) -> u8 {
    debug_assert!((1..=8).contains(&bits));
    let shifted = v << (8 - bits);
    let replicate = v >> (2 * bits).saturating_sub(8).min(bits);
    ((shifted | replicate) & 0xFF) as u8
}

/// Interpolate one 8-bit channel between endpoints by a 4-bit index weight.
#[inline]
fn interp(e0: u8, e1: u8, weight: u32) -> u8 {
    let r = ((64 - weight) * u32::from(e0) + weight * u32::from(e1) + 32) >> 6;
    (r & 0xFF) as u8
}

/// Decode one 16-byte **BC7 mode 6** block into sixteen `RGBA8` texels.
///
/// The caller must have confirmed the block is mode 6 (see [`bc7_mode`]); the
/// seven mode bits are consumed and ignored here. Field order follows the
/// Khronos spec: `R0 R1 G0 G1 B0 B1 A0 A1` (7 bits each), `P0 P1` (1 bit each),
/// then the index block -- the anchor index (texel 0) is 3 bits with an
/// implicit high `0`, the remaining fifteen are 4 bits each.
#[must_use]
pub fn decode_bc7_mode6(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mut r = BitReader::new(block);
    let _mode = r.read(7); // unary mode-6 marker: six 0s then a 1.

    let r0 = r.read(7);
    let r1 = r.read(7);
    let g0 = r.read(7);
    let g1 = r.read(7);
    let b0 = r.read(7);
    let b1 = r.read(7);
    let a0 = r.read(7);
    let a1 = r.read(7);
    let p0 = r.read(1);
    let p1 = r.read(1);

    let e0 = [
        expand7(r0, p0),
        expand7(g0, p0),
        expand7(b0, p0),
        expand7(a0, p0),
    ];
    let e1 = [
        expand7(r1, p1),
        expand7(g1, p1),
        expand7(b1, p1),
        expand7(a1, p1),
    ];

    let mut indices = [0u8; 16];
    indices[0] = r.read(3) as u8; // anchor: implicit high bit 0.
    for idx in indices.iter_mut().skip(1) {
        *idx = r.read(4) as u8;
    }

    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        let w = WEIGHT4[indices[t] as usize];
        for c in 0..4 {
            texel[c] = interp(e0[c], e1[c], w);
        }
    }
    out
}

/// Decode one 16-byte **BC7 mode 5** block into sixteen `RGBA8` texels.
///
/// Mode 5 is the other single-subset mode without a partition table: `RGB`
/// endpoints at 7 bits/channel (no p-bit, MSB-replicated to 8), a *separate*
/// 8-bit alpha endpoint pair, independent 2-bit colour and 2-bit alpha index
/// blocks (each with a 1-bit anchor at texel 0), and a 2-bit **rotation** that
/// selects which output channel swaps with alpha. It suits content where
/// colour and alpha want different interpolation directions (e.g. masks over a
/// gradient), which a shared index (mode 6) cannot represent.
///
/// The caller must have confirmed the block is mode 5 (see [`bc7_mode`]).
/// Field order (Khronos): rotation(2); `R0 R1 G0 G1 B0 B1` (7 bits each);
/// `A0 A1` (8 bits each); colour indices (anchor 1 bit, rest 2 bits); alpha
/// indices (anchor 1 bit, rest 2 bits).
#[must_use]
pub fn decode_bc7_mode5(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mut r = BitReader::new(block);
    let _mode = r.read(6); // unary mode-5 marker: five 0s then a 1.
    let rotation = r.read(2);

    let r0 = r.read(7);
    let r1 = r.read(7);
    let g0 = r.read(7);
    let g1 = r.read(7);
    let b0 = r.read(7);
    let b1 = r.read(7);
    let a0 = r.read(8);
    let a1 = r.read(8);

    let c0 = [expand_rep(r0, 7), expand_rep(g0, 7), expand_rep(b0, 7)];
    let c1 = [expand_rep(r1, 7), expand_rep(g1, 7), expand_rep(b1, 7)];
    let ae0 = (a0 & 0xFF) as u8;
    let ae1 = (a1 & 0xFF) as u8;

    let mut color_idx = [0u8; 16];
    color_idx[0] = r.read(1) as u8; // colour anchor: implicit high bit 0.
    for idx in color_idx.iter_mut().skip(1) {
        *idx = r.read(2) as u8;
    }
    let mut alpha_idx = [0u8; 16];
    alpha_idx[0] = r.read(1) as u8; // alpha anchor: implicit high bit 0.
    for idx in alpha_idx.iter_mut().skip(1) {
        *idx = r.read(2) as u8;
    }

    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        let cw = WEIGHT2[color_idx[t] as usize];
        let aw = WEIGHT2[alpha_idx[t] as usize];
        let mut rgba = [
            interp(c0[0], c1[0], cw),
            interp(c0[1], c1[1], cw),
            interp(c0[2], c1[2], cw),
            interp(ae0, ae1, aw),
        ];
        // Rotation un-swaps the channel that carried the alpha-index value.
        match rotation {
            1 => rgba.swap(0, 3),
            2 => rgba.swap(1, 3),
            3 => rgba.swap(2, 3),
            _ => {}
        }
        *texel = rgba;
    }
    out
}

/// Decode one 16-byte **BC7 mode 4** block into sixteen `RGBA8` texels.
///
/// Mode 4 is the third partition-free single-subset mode. It carries `RGB`
/// endpoints at 5 bits/channel and a *separate* 6-bit alpha endpoint pair, plus
/// **two** index blocks of different precision -- a 2-bit set and a 3-bit set.
/// A 1-bit *index-selection* flag (`idxMode`) chooses which precision drives
/// colour and which drives alpha: `idxMode = 0` gives colour the 2-bit indices
/// and alpha the 3-bit indices; `idxMode = 1` swaps them. A 2-bit **rotation**
/// then selects which output channel swaps with alpha, exactly as in mode 5.
/// Giving alpha the higher-precision index set suits smooth alpha gradients
/// over blocky colour (or vice-versa), which a shared index cannot represent.
///
/// The caller must have confirmed the block is mode 4 (see [`bc7_mode`]).
/// Field order (Khronos): rotation(2); `idxMode`(1); `R0 R1 G0 G1 B0 B1`
/// (5 bits each); `A0 A1` (6 bits each); the 2-bit index block (anchor 1 bit,
/// rest 2 bits); the 3-bit index block (anchor 2 bits, rest 3 bits).
#[must_use]
pub fn decode_bc7_mode4(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mut r = BitReader::new(block);
    let _mode = r.read(5); // unary mode-4 marker: four 0s then a 1.
    let rotation = r.read(2);
    let idx_mode = r.read(1);

    let r0 = r.read(5);
    let r1 = r.read(5);
    let g0 = r.read(5);
    let g1 = r.read(5);
    let b0 = r.read(5);
    let b1 = r.read(5);
    let a0 = r.read(6);
    let a1 = r.read(6);

    let c0 = [expand_rep(r0, 5), expand_rep(g0, 5), expand_rep(b0, 5)];
    let c1 = [expand_rep(r1, 5), expand_rep(g1, 5), expand_rep(b1, 5)];
    let ae0 = expand_rep(a0, 6);
    let ae1 = expand_rep(a1, 6);

    // Index set 0 (2-bit) is stored first; its anchor (texel 0) is 1 bit.
    let mut idx2 = [0u8; 16];
    idx2[0] = r.read(1) as u8;
    for idx in idx2.iter_mut().skip(1) {
        *idx = r.read(2) as u8;
    }
    // Index set 1 (3-bit) follows; its anchor is 2 bits (implicit high 0).
    let mut idx3 = [0u8; 16];
    idx3[0] = r.read(2) as u8;
    for idx in idx3.iter_mut().skip(1) {
        *idx = r.read(3) as u8;
    }

    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        // idxMode routes which precision drives colour vs alpha.
        let (cw, aw) = if idx_mode == 0 {
            (WEIGHT2[idx2[t] as usize], WEIGHT3[idx3[t] as usize])
        } else {
            (WEIGHT3[idx3[t] as usize], WEIGHT2[idx2[t] as usize])
        };
        let mut rgba = [
            interp(c0[0], c1[0], cw),
            interp(c0[1], c1[1], cw),
            interp(c0[2], c1[2], cw),
            interp(ae0, ae1, aw),
        ];
        // Rotation un-swaps the channel that carried the alpha-index value.
        match rotation {
            1 => rgba.swap(0, 3),
            2 => rgba.swap(1, 3),
            3 => rgba.swap(2, 3),
            _ => {}
        }
        *texel = rgba;
    }
    out
}

/// Decode one 16-byte **BC7 mode 1** block into sixteen `RGBA8` texels.
///
/// Mode 1 is a **two-subset, RGB-only** mode (opaque alpha `255`): a 6-bit
/// partition selects one of 64 texel-to-subset maps, four RGB endpoints (two
/// per subset) are stored at 6 bits/channel with **two shared P-bits** (one per
/// subset, appended as the endpoints' LSB for 7-bit effective precision), and
/// the sixteen texels carry 3-bit interpolation indices. The two anchor texels
/// (texel 0 for subset 0, [`BPTC_ANCHORS_2`]`[partition]` for subset 1) store a
/// 2-bit index with an implicit high `0`. Field order follows the Khronos spec:
/// mode(2), partition(6), R0..R3, G0..G3, B0..B3 (6 bits each), P0 P1, indices.
///
/// The caller must have confirmed the block is mode 1 (see [`bc7_mode`]).
#[must_use]
pub fn decode_bc7_mode1(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mut r = BitReader::new(block);
    let _mode = r.read(2); // unary mode-1 marker: one 0 then a 1.
    let partition = r.read(6) as usize;

    // Four endpoints (subset 0: e0,e1; subset 1: e2,e3), channel-major 6-bit.
    let mut rc = [0u32; 4];
    let mut gc = [0u32; 4];
    let mut bc = [0u32; 4];
    for v in &mut rc {
        *v = r.read(6);
    }
    for v in &mut gc {
        *v = r.read(6);
    }
    for v in &mut bc {
        *v = r.read(6);
    }
    let p0 = r.read(1);
    let p1 = r.read(1);
    let pbit = [p0, p0, p1, p1]; // one shared P-bit per subset.

    // Expand each 6-bit channel + its shared P-bit (7-bit value) to 8 bits by
    // Khronos MSB replication.
    let mut ep = [[0u8; 3]; 4];
    for (e, slot) in ep.iter_mut().enumerate() {
        slot[0] = expand_rep((rc[e] << 1) | pbit[e], 7);
        slot[1] = expand_rep((gc[e] << 1) | pbit[e], 7);
        slot[2] = expand_rep((bc[e] << 1) | pbit[e], 7);
    }

    let partition_map = &BPTC_PARTITIONS_2[partition];
    let anchor1 = BPTC_ANCHORS_2[partition];

    // Indices in texel order: both anchors (texel 0 and `anchor1`) are 2-bit
    // with an implicit high 0; the rest are 3-bit.
    let mut idx = [0u8; 16];
    for (t, slot) in idx.iter_mut().enumerate() {
        let bits = if t == 0 || t == anchor1 { 2 } else { 3 };
        *slot = r.read(bits) as u8;
    }

    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        let subset = usize::from(partition_map[t]);
        let e0 = ep[subset * 2];
        let e1 = ep[subset * 2 + 1];
        let w = WEIGHT3[usize::from(idx[t])];
        *texel = [
            interp(e0[0], e1[0], w),
            interp(e0[1], e1[1], w),
            interp(e0[2], e1[2], w),
            255,
        ];
    }
    out
}

/// Decode one 16-byte **BC7 mode 3** block into sixteen `RGBA8` texels.
///
/// Mode 3 is a **two-subset, RGB-only** mode (opaque alpha `255`) with higher
/// endpoint precision than mode 1: a 6-bit partition selects one of 64
/// texel-to-subset maps, four RGB endpoints (two per subset) are stored at
/// 7 bits/channel with **one P-bit per endpoint** (appended as the channel LSB
/// for 8-bit effective precision), and the sixteen texels carry 2-bit
/// interpolation indices. The two anchor texels (texel 0 for subset 0,
/// [`BPTC_ANCHORS_2`]`[partition]` for subset 1) store a 1-bit index with an
/// implicit high `0`. Field order follows the Khronos spec: mode(4),
/// partition(6), R0..R3, G0..G3, B0..B3 (7 bits each), P0..P3, indices.
///
/// The caller must have confirmed the block is mode 3 (see [`bc7_mode`]).
#[must_use]
pub fn decode_bc7_mode3(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mut r = BitReader::new(block);
    let _mode = r.read(4); // unary mode-3 marker: three 0s then a 1.
    let partition = r.read(6) as usize;

    // Four endpoints (subset 0: e0,e1; subset 1: e2,e3), channel-major 7-bit.
    let mut rc = [0u32; 4];
    let mut gc = [0u32; 4];
    let mut bc = [0u32; 4];
    for v in &mut rc {
        *v = r.read(7);
    }
    for v in &mut gc {
        *v = r.read(7);
    }
    for v in &mut bc {
        *v = r.read(7);
    }
    // One P-bit per endpoint (contrast mode 1, which shares one per subset).
    let mut pbit = [0u32; 4];
    for p in &mut pbit {
        *p = r.read(1);
    }

    // Expand each 7-bit channel + its endpoint P-bit to the full 8-bit value.
    let mut ep = [[0u8; 3]; 4];
    for (e, slot) in ep.iter_mut().enumerate() {
        slot[0] = expand7(rc[e], pbit[e]);
        slot[1] = expand7(gc[e], pbit[e]);
        slot[2] = expand7(bc[e], pbit[e]);
    }

    let partition_map = &BPTC_PARTITIONS_2[partition];
    let anchor1 = BPTC_ANCHORS_2[partition];

    // Indices in texel order: both anchors (texel 0 and `anchor1`) are 1-bit
    // with an implicit high 0; the rest are 2-bit.
    let mut idx = [0u8; 16];
    for (t, slot) in idx.iter_mut().enumerate() {
        let bits = if t == 0 || t == anchor1 { 1 } else { 2 };
        *slot = r.read(bits) as u8;
    }

    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        let subset = usize::from(partition_map[t]);
        let e0 = ep[subset * 2];
        let e1 = ep[subset * 2 + 1];
        let w = WEIGHT2[usize::from(idx[t])];
        *texel = [
            interp(e0[0], e1[0], w),
            interp(e0[1], e1[1], w),
            interp(e0[2], e1[2], w),
            255,
        ];
    }
    out
}

/// Decode one 16-byte **BC7 mode 7** block into sixteen `RGBA8` texels.
///
/// Mode 7 is a **two-subset RGBA** mode: a 6-bit partition selects one of 64
/// texel-to-subset maps, four RGBA endpoints (two per subset) are stored at
/// 5 bits/channel with **one P-bit per endpoint** (appended as the channel LSB
/// for 6-bit effective precision, then MSB-replicated to 8 bits), and the
/// sixteen texels carry 2-bit interpolation indices shared by colour and alpha.
/// The two anchor texels (texel 0 for subset 0,
/// [`BPTC_ANCHORS_2`]`[partition]` for subset 1) store a 1-bit index with an
/// implicit high `0`. Field order follows the Khronos spec: mode(8),
/// partition(6), R0..R3, G0..G3, B0..B3, A0..A3 (5 bits each), P0..P3, indices.
///
/// The caller must have confirmed the block is mode 7 (see [`bc7_mode`]).
#[must_use]
pub fn decode_bc7_mode7(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mut r = BitReader::new(block);
    let _mode = r.read(8); // unary mode-7 marker: seven 0s then a 1.
    let partition = r.read(6) as usize;

    // Four endpoints (subset 0: e0,e1; subset 1: e2,e3), channel-major 5-bit
    // in R,G,B,A order.
    let mut rc = [0u32; 4];
    let mut gc = [0u32; 4];
    let mut bc = [0u32; 4];
    let mut ac = [0u32; 4];
    for v in &mut rc {
        *v = r.read(5);
    }
    for v in &mut gc {
        *v = r.read(5);
    }
    for v in &mut bc {
        *v = r.read(5);
    }
    for v in &mut ac {
        *v = r.read(5);
    }
    // One P-bit per endpoint.
    let mut pbit = [0u32; 4];
    for pb in &mut pbit {
        *pb = r.read(1);
    }

    // Expand each 5-bit channel + its endpoint P-bit (6-bit value) to 8 bits by
    // Khronos MSB replication.
    let mut ep = [[0u8; 4]; 4];
    for (e, slot) in ep.iter_mut().enumerate() {
        slot[0] = expand_rep((rc[e] << 1) | pbit[e], 6);
        slot[1] = expand_rep((gc[e] << 1) | pbit[e], 6);
        slot[2] = expand_rep((bc[e] << 1) | pbit[e], 6);
        slot[3] = expand_rep((ac[e] << 1) | pbit[e], 6);
    }

    let partition_map = &BPTC_PARTITIONS_2[partition];
    let anchor1 = BPTC_ANCHORS_2[partition];

    // Indices in texel order: both anchors (texel 0 and `anchor1`) are 1-bit
    // with an implicit high 0; the rest are 2-bit. Colour and alpha share them.
    let mut idx = [0u8; 16];
    for (t, slot) in idx.iter_mut().enumerate() {
        let bits = if t == 0 || t == anchor1 { 1 } else { 2 };
        *slot = r.read(bits) as u8;
    }

    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        let subset = usize::from(partition_map[t]);
        let e0 = ep[subset * 2];
        let e1 = ep[subset * 2 + 1];
        let w = WEIGHT2[usize::from(idx[t])];
        *texel = [
            interp(e0[0], e1[0], w),
            interp(e0[1], e1[1], w),
            interp(e0[2], e1[2], w),
            interp(e0[3], e1[3], w),
        ];
    }
    out
}

/// Decode one 16-byte **BC7 mode 2** block into sixteen `RGBA8` texels.
///
/// Mode 2 is a **three-subset, RGB-only** mode (opaque alpha `255`). A 6-bit
/// partition selects one of 64 texel-to-subset maps ([`BPTC_PARTITIONS_3`]);
/// six RGB endpoints (two per subset) are stored at 5 bits/channel with **no
/// P-bits** (MSB-replicated to 8 bits), and the sixteen texels carry 2-bit
/// interpolation indices. The three anchor texels -- texel 0 (subset 0),
/// [`BPTC_ANCHORS_3_2`]`[partition]` (subset 1), and
/// [`BPTC_ANCHORS_3_3`]`[partition]` (subset 2) -- store a 1-bit index with an
/// implicit high `0`. Field order follows the Khronos spec: mode(3),
/// partition(6), R0..R5, G0..G5, B0..B5 (5 bits each), indices.
///
/// The caller must have confirmed the block is mode 2 (see [`bc7_mode`]).
#[must_use]
pub fn decode_bc7_mode2(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mut r = BitReader::new(block);
    let _mode = r.read(3); // unary mode-2 marker: two 0s then a 1.
    let partition = r.read(6) as usize;

    // Six endpoints (subset s: e[2s], e[2s+1]), channel-major 5-bit.
    let mut rc = [0u32; 6];
    let mut gc = [0u32; 6];
    let mut bc = [0u32; 6];
    for v in &mut rc {
        *v = r.read(5);
    }
    for v in &mut gc {
        *v = r.read(5);
    }
    for v in &mut bc {
        *v = r.read(5);
    }

    // No P-bits: expand each 5-bit channel to 8 bits by MSB replication.
    let mut ep = [[0u8; 3]; 6];
    for (e, slot) in ep.iter_mut().enumerate() {
        slot[0] = expand_rep(rc[e], 5);
        slot[1] = expand_rep(gc[e], 5);
        slot[2] = expand_rep(bc[e], 5);
    }

    let partition_map = &BPTC_PARTITIONS_3[partition];
    let anchor1 = BPTC_ANCHORS_3_2[partition];
    let anchor2 = BPTC_ANCHORS_3_3[partition];

    // Indices in texel order: the three anchors (texel 0, `anchor1`, `anchor2`)
    // are 1-bit with an implicit high 0; the rest are 2-bit.
    let mut idx = [0u8; 16];
    for (t, slot) in idx.iter_mut().enumerate() {
        let bits = if t == 0 || t == anchor1 || t == anchor2 {
            1
        } else {
            2
        };
        *slot = r.read(bits) as u8;
    }

    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        let subset = usize::from(partition_map[t]);
        let e0 = ep[subset * 2];
        let e1 = ep[subset * 2 + 1];
        let w = WEIGHT2[usize::from(idx[t])];
        *texel = [
            interp(e0[0], e1[0], w),
            interp(e0[1], e1[1], w),
            interp(e0[2], e1[2], w),
            255,
        ];
    }
    out
}

/// Decode one 16-byte **BC7 mode 0** block into sixteen `RGBA8` texels.
///
/// Mode 0 is a **three-subset, RGB-only** mode (opaque alpha `255`) and the
/// only BC7 mode with a **4-bit partition** (just 16 partitions). Six RGB
/// endpoints (two per subset) are stored at 4 bits/channel with **one P-bit per
/// endpoint** (appended as the channel LSB for 5-bit precision, then
/// MSB-replicated to 8 bits), and the sixteen texels carry 3-bit interpolation
/// indices. The three anchor texels -- texel 0 (subset 0),
/// [`BPTC_ANCHORS_3_2`]`[partition]` (subset 1), and
/// [`BPTC_ANCHORS_3_3`]`[partition]` (subset 2) -- store a 2-bit index with an
/// implicit high `0`. Field order follows the Khronos spec: mode(1),
/// partition(4), R0..R5, G0..G5, B0..B5 (4 bits each), P0..P5, indices.
///
/// The caller must have confirmed the block is mode 0 (see [`bc7_mode`]).
#[must_use]
pub fn decode_bc7_mode0(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mut r = BitReader::new(block);
    let _mode = r.read(1); // unary mode-0 marker: a single 1.
    let partition = r.read(4) as usize; // mode 0 uses only 16 partitions.

    // Six endpoints (subset s: e[2s], e[2s+1]), channel-major 4-bit.
    let mut rc = [0u32; 6];
    let mut gc = [0u32; 6];
    let mut bc = [0u32; 6];
    for v in &mut rc {
        *v = r.read(4);
    }
    for v in &mut gc {
        *v = r.read(4);
    }
    for v in &mut bc {
        *v = r.read(4);
    }
    // One P-bit per endpoint.
    let mut pbit = [0u32; 6];
    for p in &mut pbit {
        *p = r.read(1);
    }

    // Expand each 4-bit channel + its endpoint P-bit (5-bit value) to 8 bits by
    // Khronos MSB replication.
    let mut ep = [[0u8; 3]; 6];
    for (e, slot) in ep.iter_mut().enumerate() {
        slot[0] = expand_rep((rc[e] << 1) | pbit[e], 5);
        slot[1] = expand_rep((gc[e] << 1) | pbit[e], 5);
        slot[2] = expand_rep((bc[e] << 1) | pbit[e], 5);
    }

    let partition_map = &BPTC_PARTITIONS_3[partition];
    let anchor1 = BPTC_ANCHORS_3_2[partition];
    let anchor2 = BPTC_ANCHORS_3_3[partition];

    // Indices in texel order: the three anchors (texel 0, `anchor1`, `anchor2`)
    // are 2-bit with an implicit high 0; the rest are 3-bit.
    let mut idx = [0u8; 16];
    for (t, slot) in idx.iter_mut().enumerate() {
        let bits = if t == 0 || t == anchor1 || t == anchor2 {
            2
        } else {
            3
        };
        *slot = r.read(bits) as u8;
    }

    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        let subset = usize::from(partition_map[t]);
        let e0 = ep[subset * 2];
        let e1 = ep[subset * 2 + 1];
        let w = WEIGHT3[usize::from(idx[t])];
        *texel = [
            interp(e0[0], e1[0], w),
            interp(e0[1], e1[1], w),
            interp(e0[2], e1[2], w),
            255,
        ];
    }
    out
}

/// Error returned by [`decode_bc7`] for a BC7 block whose mode is not yet
/// supported by this decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bc7Error {
    /// Byte 0 was zero: a reserved/invalid mode encoding.
    ReservedMode,
    /// A valid but unsupported mode. With all eight BC7 modes (0-7) now
    /// decoded, this variant is unreachable from [`decode_bc7`] and retained
    /// only for forward compatibility / exhaustive matching.
    UnsupportedMode(u8),
}

/// Decode a BC7 block, dispatching on its mode.
///
/// Supported today: the partition-free single-subset modes 4
/// ([`decode_bc7_mode4`]), 5 ([`decode_bc7_mode5`]), 6 ([`decode_bc7_mode6`]),
/// the two-subset RGB modes 1 ([`decode_bc7_mode1`]) and 3 ([`decode_bc7_mode3`]),
/// the two-subset RGBA mode 7 ([`decode_bc7_mode7`]), and the three-subset RGB
/// modes 0 ([`decode_bc7_mode0`]) and 2 ([`decode_bc7_mode2`]) -- all eight
/// modes are now covered and validated against GPU hardware decode (modes with
/// a 6-bit partition across all 64 partitions, mode 0 across its 16). Any byte-0
/// zero block returns [`Bc7Error::ReservedMode`].
pub fn decode_bc7(block: &[u8; 16]) -> Result<[[u8; 4]; 16], Bc7Error> {
    match bc7_mode(block) {
        None => Err(Bc7Error::ReservedMode),
        Some(1) => Ok(decode_bc7_mode1(block)),
        Some(3) => Ok(decode_bc7_mode3(block)),
        Some(7) => Ok(decode_bc7_mode7(block)),
        Some(4) => Ok(decode_bc7_mode4(block)),
        Some(5) => Ok(decode_bc7_mode5(block)),
        Some(6) => Ok(decode_bc7_mode6(block)),
        Some(0) => Ok(decode_bc7_mode0(block)),
        Some(2) => Ok(decode_bc7_mode2(block)),
        Some(m) => Err(Bc7Error::UnsupportedMode(m)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal LSB-first bit writer mirroring [`BitReader`] for test blocks.
    struct BitWriter {
        bytes: [u8; 16],
        pos: usize,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: [0u8; 16],
                pos: 0,
            }
        }

        fn write(&mut self, value: u32, n: u32) {
            for i in 0..n {
                if (value >> i) & 1 == 1 {
                    self.bytes[self.pos / 8] |= 1 << (self.pos % 8);
                }
                self.pos += 1;
            }
        }
    }

    /// Assemble a mode-6 block from field values; `idx` holds the sixteen
    /// 4-bit indices (index 0 must be `<= 7` so its high bit is implicit 0).
    fn make_block(rgba0: [u32; 4], rgba1: [u32; 4], p0: u32, p1: u32, idx: [u8; 16]) -> [u8; 16] {
        let mut w = BitWriter::new();
        w.write(0b100_0000, 7); // mode 6 marker
        w.write(rgba0[0], 7);
        w.write(rgba1[0], 7);
        w.write(rgba0[1], 7);
        w.write(rgba1[1], 7);
        w.write(rgba0[2], 7);
        w.write(rgba1[2], 7);
        w.write(rgba0[3], 7);
        w.write(rgba1[3], 7);
        w.write(p0, 1);
        w.write(p1, 1);
        w.write(u32::from(idx[0]), 3);
        for &i in idx.iter().skip(1) {
            w.write(u32::from(i), 4);
        }
        assert_eq!(w.pos, 128, "mode-6 fields must fill the block exactly");
        w.bytes
    }

    #[test]
    fn mode_detection_reads_unary_prefix() {
        let mut b = [0u8; 16];
        b[0] = 0b0100_0000; // bit 6 set, bits 0-5 clear => mode 6
        assert_eq!(bc7_mode(&b), Some(6));
        b[0] = 0b0000_0001; // bit 0 set => mode 0
        assert_eq!(bc7_mode(&b), Some(0));
        b[0] = 0b1000_0000; // bit 7 set => mode 7
        assert_eq!(bc7_mode(&b), Some(7));
        b[0] = 0; // reserved / invalid
        assert_eq!(bc7_mode(&b), None);
    }

    #[test]
    fn make_block_is_tagged_mode6() {
        let block = make_block([0; 4], [0x7F; 4], 0, 1, [0u8; 16]);
        assert_eq!(bc7_mode(&block), Some(6));
    }

    #[test]
    fn endpoint_indices_decode_exactly() {
        // e0 = (0x7F<<1)|0 = 0xFE on all channels; e1 = (0<<1)|1 = 1.
        let block = make_block(
            [0x7F; 4],
            [0; 4],
            0,
            1,
            [0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15],
        );
        let out = decode_bc7_mode6(&block);
        // Weight 0 -> pure e0; weight 64 -> pure e1 (exact, no rounding slack).
        assert_eq!(out[0], [0xFE, 0xFE, 0xFE, 0xFE]);
        assert_eq!(out[1], [1, 1, 1, 1]);
        assert_eq!(out[2], [0xFE, 0xFE, 0xFE, 0xFE]);
        assert_eq!(out[3], [1, 1, 1, 1]);
    }

    #[test]
    fn constant_block_is_flat_regardless_of_index() {
        // e0 == e1 => every texel equals that colour for any index weight.
        let c = [0x55, 0x2A, 0x7F, 0x10];
        let mut idx = [0u8; 16];
        for (t, i) in idx.iter_mut().enumerate() {
            *i = (t as u8) & 0x0F;
        }
        idx[0] = 5; // anchor must stay <= 7
        let block = make_block(c, c, 1, 1, idx);
        let expected = [
            expand7(c[0], 1),
            expand7(c[1], 1),
            expand7(c[2], 1),
            expand7(c[3], 1),
        ];
        let out = decode_bc7_mode6(&block);
        for texel in &out {
            assert_eq!(*texel, expected);
        }
    }

    #[test]
    fn interior_indices_are_monotonic_between_endpoints() {
        // Red ramps 0 -> 254 as the index weight grows; check monotonicity and
        // bracketing on the red channel across all sixteen weight steps.
        let idx = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let block = make_block([0, 0, 0, 0x7F], [0x7F, 0, 0, 0x7F], 0, 0, idx);
        let out = decode_bc7_mode6(&block);
        let e0r = out[0][0];
        let e1r = out[15][0];
        assert_eq!(e0r, 0);
        assert_eq!(e1r, 0xFE);
        for t in 1..16 {
            assert!(
                out[t][0] >= out[t - 1][0],
                "red must be non-decreasing in index"
            );
            assert!(out[t][0] >= e0r && out[t][0] <= e1r, "red stays bracketed");
        }
        // Alpha endpoints are equal (0xFE) so alpha is flat.
        for texel in &out {
            assert_eq!(texel[3], 0xFE);
        }
    }

    #[test]
    fn pbit_sets_endpoint_low_bit() {
        // Same 7-bit value, differing p-bit, must differ only in the LSB.
        let block = make_block([0x40, 0, 0, 0x7F], [0x40, 0, 0, 0x7F], 0, 1, [0u8; 16]);
        let out = decode_bc7_mode6(&block);
        // index 0 (anchor, weight 0) -> e0 with p=0 => 0x80.
        assert_eq!(out[0][0], 0x80);
        // A weight-64 texel would give e1 with p=1 => 0x81; verify via a block
        // whose anchor still weight-0 but texel 1 is weight 64.
        let mut idx = [0u8; 16];
        idx[1] = 15;
        let block2 = make_block([0x40, 0, 0, 0x7F], [0x40, 0, 0, 0x7F], 0, 1, idx);
        let out2 = decode_bc7_mode6(&block2);
        assert_eq!(out2[1][0], 0x81);
    }

    #[test]
    fn midpoint_index_is_near_average() {
        // Weight 30 (index 7) and 34 (index 8) bracket the true midpoint of a
        // 0..254 ramp (127); check the decode lands within the integer step.
        let block = make_block([0, 0, 0, 0x7F], [0x7F, 0, 0, 0x7F], 0, 0, {
            let mut idx = [0u8; 16];
            idx[1] = 7;
            idx[2] = 8;
            idx
        });
        let out = decode_bc7_mode6(&block);
        assert!(out[1][0] <= 127 && out[2][0] >= 127);
    }

    /// Assemble a mode-5 block: `rgb0/rgb1` are 7-bit, `a0/a1` 8-bit; `cidx`
    /// and `aidx` are the sixteen 2-bit colour/alpha indices (index 0 <= 1).
    fn make_block5(
        rgb0: [u32; 3],
        rgb1: [u32; 3],
        a0: u32,
        a1: u32,
        rotation: u32,
        cidx: [u8; 16],
        aidx: [u8; 16],
    ) -> [u8; 16] {
        let mut w = BitWriter::new();
        w.write(0b10_0000, 6); // mode 5 marker
        w.write(rotation, 2);
        w.write(rgb0[0], 7);
        w.write(rgb1[0], 7);
        w.write(rgb0[1], 7);
        w.write(rgb1[1], 7);
        w.write(rgb0[2], 7);
        w.write(rgb1[2], 7);
        w.write(a0, 8);
        w.write(a1, 8);
        w.write(u32::from(cidx[0]), 1);
        for &i in cidx.iter().skip(1) {
            w.write(u32::from(i), 2);
        }
        w.write(u32::from(aidx[0]), 1);
        for &i in aidx.iter().skip(1) {
            w.write(u32::from(i), 2);
        }
        assert_eq!(w.pos, 128, "mode-5 fields must fill the block exactly");
        w.bytes
    }

    #[test]
    fn mode5_block_is_tagged_mode5() {
        let block = make_block5([0; 3], [0x7F; 3], 0, 255, 0, [0u8; 16], [0u8; 16]);
        assert_eq!(bc7_mode(&block), Some(5));
    }

    #[test]
    fn mode5_color_endpoints_replicate_to_8bit() {
        // 7-bit 0x7F -> (0x7F<<1)|(0x7F>>6) = 0xFF; 0x00 -> 0x00.
        let block = make_block5(
            [0x7F; 3],
            [0; 3],
            200,
            50,
            0,
            [0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3],
            [0u8; 16],
        );
        let out = decode_bc7_mode5(&block);
        // colour weight 0 -> c0 (0xFF); weight 64 (index 3) -> c1 (0x00).
        assert_eq!([out[0][0], out[0][1], out[0][2]], [0xFF, 0xFF, 0xFF]);
        assert_eq!([out[1][0], out[1][1], out[1][2]], [0, 0, 0]);
        // alpha endpoints are exact 8-bit; index 0 everywhere -> a0 = 200.
        assert_eq!(out[0][3], 200);
    }

    #[test]
    fn mode5_alpha_index_is_independent_of_color() {
        // Flat colour, alpha ramps via its own index block.
        let mut aidx = [0u8; 16];
        aidx[1] = 3; // weight 64 -> a1
        let block = make_block5([0x40; 3], [0x40; 3], 0, 255, 0, [0u8; 16], aidx);
        let out = decode_bc7_mode5(&block);
        assert_eq!(out[0][3], 0); // a0
        assert_eq!(out[1][3], 255); // a1
                                    // Colour identical on both texels (flat endpoints).
        assert_eq!(
            [out[0][0], out[0][1], out[0][2]],
            [out[1][0], out[1][1], out[1][2]]
        );
    }

    #[test]
    fn mode5_rotation_swaps_alpha_channel() {
        // rotation 1 swaps R<->A. Colour R=0xFF via c0, alpha=0 via a0.
        let block = make_block5([0x7F, 0, 0], [0x7F, 0, 0], 0, 0, 1, [0u8; 16], [0u8; 16]);
        let out = decode_bc7_mode5(&block);
        // After swap(0,3): channel 0 carries the (0) alpha, channel 3 carries R=0xFF.
        assert_eq!(out[0][0], 0);
        assert_eq!(out[0][3], 0xFF);
    }

    #[test]
    fn dispatch_accepts_all_modes_and_rejects_reserved() {
        let mode6 = make_block([0; 4], [0x7F; 4], 0, 1, [0u8; 16]);
        assert!(decode_bc7(&mode6).is_ok());
        let mode5 = make_block5([0; 3], [0x7F; 3], 0, 255, 0, [0u8; 16], [0u8; 16]);
        assert!(decode_bc7(&mode5).is_ok());

        // All eight modes (0-7) decode; only a byte-0 zero block is reserved.
        let m0 = make_block0(0, [[0; 3]; 6], [0; 6], [0u8; 16]);
        assert_eq!(bc7_mode(&m0), Some(0));
        assert!(decode_bc7(&m0).is_ok());
        let m2 = make_block2(0, [[0; 3]; 6], [0u8; 16]);
        assert_eq!(bc7_mode(&m2), Some(2));
        assert!(decode_bc7(&m2).is_ok());
        let reserved = [0u8; 16];
        assert_eq!(decode_bc7(&reserved), Err(Bc7Error::ReservedMode));
    }

    #[test]
    fn expand_rep_matches_known_replication() {
        assert_eq!(expand_rep(0x7F, 7), 0xFF);
        assert_eq!(expand_rep(0, 7), 0);
        // 0x40 = 0b100_0000; (<<1)=0x80, (>>6)=1, OR = 0x81.
        assert_eq!(expand_rep(0x40, 7), 0x81);
        assert_eq!(expand_rep(0xFF, 8), 0xFF); // 8-bit identity
        assert_eq!(expand_rep(0x5A, 8), 0x5A);
    }

    /// Assemble a mode-4 block: `rgb0/rgb1` are 5-bit, `a0/a1` 6-bit; `idx2`
    /// are the sixteen 2-bit indices (index 0 <= 1), `idx3` the sixteen 3-bit
    /// indices (index 0 <= 3); `idx_mode` selects colour/alpha index routing.
    #[expect(
        clippy::too_many_arguments,
        reason = "test block assembler mirrors the BC7 bit-field layout"
    )]
    fn make_block4(
        rgb0: [u32; 3],
        rgb1: [u32; 3],
        a0: u32,
        a1: u32,
        rotation: u32,
        idx_mode: u32,
        idx2: [u8; 16],
        idx3: [u8; 16],
    ) -> [u8; 16] {
        let mut w = BitWriter::new();
        w.write(0b1_0000, 5); // mode 4 marker
        w.write(rotation, 2);
        w.write(idx_mode, 1);
        w.write(rgb0[0], 5);
        w.write(rgb1[0], 5);
        w.write(rgb0[1], 5);
        w.write(rgb1[1], 5);
        w.write(rgb0[2], 5);
        w.write(rgb1[2], 5);
        w.write(a0, 6);
        w.write(a1, 6);
        w.write(u32::from(idx2[0]), 1);
        for &i in idx2.iter().skip(1) {
            w.write(u32::from(i), 2);
        }
        w.write(u32::from(idx3[0]), 2);
        for &i in idx3.iter().skip(1) {
            w.write(u32::from(i), 3);
        }
        assert_eq!(w.pos, 128, "mode-4 fields must fill the block exactly");
        w.bytes
    }

    #[test]
    fn mode4_block_is_tagged_mode4() {
        let block = make_block4([0; 3], [0x1F; 3], 0, 0x3F, 0, 0, [0u8; 16], [0u8; 16]);
        assert_eq!(bc7_mode(&block), Some(4));
    }

    #[test]
    fn mode4_color_endpoints_replicate_to_8bit() {
        // 5-bit 0x1F -> (0x1F<<3)|(0x1F>>2) = 0xF8|0x07 = 0xFF; 0x00 -> 0x00.
        // idx_mode 0: colour uses the 2-bit index set.
        let mut idx2 = [0u8; 16];
        idx2[1] = 3; // weight 64 -> c1
        let block = make_block4([0x1F; 3], [0; 3], 0x3F, 0, 0, 0, idx2, [0u8; 16]);
        let out = decode_bc7_mode4(&block);
        assert_eq!([out[0][0], out[0][1], out[0][2]], [0xFF, 0xFF, 0xFF]); // c0
        assert_eq!([out[1][0], out[1][1], out[1][2]], [0, 0, 0]); // c1
    }

    #[test]
    fn mode4_alpha_6bit_replicates_and_uses_3bit_index() {
        // 6-bit 0x3F -> (0x3F<<2)|(0x3F>>4) = 0xFC|0x03 = 0xFF.
        // idx_mode 0: alpha uses the 3-bit index set.
        let mut idx3 = [0u8; 16];
        idx3[1] = 7; // weight 64 -> a1
        let block = make_block4([0x10; 3], [0x10; 3], 0, 0x3F, 0, 0, [0u8; 16], idx3);
        let out = decode_bc7_mode4(&block);
        assert_eq!(out[0][3], 0); // a0
        assert_eq!(out[1][3], 0xFF); // a1 replicated
    }

    #[test]
    fn mode4_idx_mode_routes_precision() {
        // idx_mode 1: colour driven by the 3-bit set, alpha by the 2-bit set.
        // colour c0=0, c1=0xFF; alpha a0=0, a1=0xFF.
        let mut idx2 = [0u8; 16];
        let mut idx3 = [0u8; 16];
        idx3[1] = 7; // colour at texel1 -> c1 (0xFF); its 2-bit alpha stays a0.
        idx2[2] = 3; // alpha at texel2 -> a1 (0xFF); its 3-bit colour stays c0.
        let block = make_block4([0; 3], [0x1F; 3], 0, 0x3F, 0, 1, idx2, idx3);
        let out = decode_bc7_mode4(&block);
        assert_eq!([out[1][0], out[1][1], out[1][2]], [0xFF, 0xFF, 0xFF]);
        assert_eq!(out[1][3], 0);
        assert_eq!([out[2][0], out[2][1], out[2][2]], [0, 0, 0]);
        assert_eq!(out[2][3], 0xFF);
    }

    #[test]
    fn mode4_rotation_swaps_alpha_channel() {
        // rotation 1 swaps R<->A. idx_mode 0: colour via 2-bit, alpha via 3-bit.
        // R=0xFF via c0 (idx2 all 0), alpha=0 via a0 (idx3 all 0).
        let block = make_block4([0x1F, 0, 0], [0x1F, 0, 0], 0, 0, 1, 0, [0u8; 16], [0u8; 16]);
        let out = decode_bc7_mode4(&block);
        assert_eq!(out[0][0], 0); // channel 0 now carries the (0) alpha
        assert_eq!(out[0][3], 0xFF); // channel 3 now carries R=0xFF
    }

    #[test]
    fn dispatch_decodes_mode4() {
        let block = make_block4([0; 3], [0x1F; 3], 0, 0x3F, 0, 0, [0u8; 16], [0u8; 16]);
        assert!(decode_bc7(&block).is_ok());
    }

    /// Assemble a two-subset RGB mode-1 block. `rgb[e]` is the 6-bit endpoint
    /// `e in 0..4` (endpoints 0,1 = subset 0; 2,3 = subset 1); `pbit` holds the
    /// two shared P-bits; `idx` the sixteen 3-bit indices (both anchors must be
    /// `<= 3` so their implicit high bit is 0).
    fn make_block1(partition: u32, rgb: [[u32; 3]; 4], pbit: [u32; 2], idx: [u8; 16]) -> [u8; 16] {
        let mut w = BitWriter::new();
        w.write(0b10, 2); // mode-1 unary marker: a 0 then a 1.
        w.write(partition, 6);
        for e in rgb {
            w.write(e[0], 6);
        }
        for e in rgb {
            w.write(e[1], 6);
        }
        for e in rgb {
            w.write(e[2], 6);
        }
        w.write(pbit[0], 1);
        w.write(pbit[1], 1);
        let anchor1 = BPTC_ANCHORS_2[partition as usize];
        for (t, &i) in idx.iter().enumerate() {
            let n = if t == 0 || t == anchor1 { 2 } else { 3 };
            w.write(u32::from(i), n);
        }
        assert_eq!(w.pos, 128, "mode-1 fields must fill the block exactly");
        w.bytes
    }

    #[test]
    fn make_block1_is_tagged_mode1() {
        let block = make_block1(0, [[0; 3]; 4], [0, 0], [0u8; 16]);
        assert_eq!(bc7_mode(&block), Some(1));
    }

    #[test]
    fn mode1_subsets_select_distinct_endpoints() {
        // Partition 0 maps the top-left 2x2 quadrant columns to subsets by
        // `BPTC_PARTITIONS_2[0] = [0,0,1,1, ...]`. Give subset 0 pure black and
        // subset 1 pure white at anchor weight 0, so each texel resolves to its
        // subset's endpoint 0 exactly.
        let black = [0u32; 3];
        let white = [0x3F; 3];
        let rgb = [black, black, white, white];
        let block = make_block1(0, rgb, [0, 1], [0u8; 16]);
        let out = decode_bc7_mode1(&block);
        let map = &BPTC_PARTITIONS_2[0];
        for (t, texel) in out.iter().enumerate() {
            let expect = if map[t] == 0 { 0 } else { 0xFF };
            assert_eq!(
                [texel[0], texel[1], texel[2]],
                [expect; 3],
                "texel {t} subset {}",
                map[t]
            );
            assert_eq!(texel[3], 255, "mode 1 is opaque");
        }
    }

    #[test]
    fn mode1_endpoint_weights_are_exact_at_bounds() {
        // Subset 0: e0 black, e1 white; index 0 -> e0, index 7 -> e1 exactly.
        let rgb = [[0u32; 3], [0x3F; 3], [0u32; 3], [0x3F; 3]];
        let mut idx = [0u8; 16];
        idx[1] = 7; // texel 1 is subset 0 under partition 0, weight 64 -> e1.
        let block = make_block1(0, rgb, [0, 0], idx);
        let out = decode_bc7_mode1(&block);
        // texel 0 (anchor, index 0) -> e0 expanded: (0<<1|0)=0 -> 0.
        assert_eq!([out[0][0], out[0][1], out[0][2]], [0, 0, 0]);
        // texel 1 (index 7) -> e1 expanded: (0x3F<<1|0)=0x7E -> 7-bit MSB
        // replicate to 0xFD ((0x7E<<1)|(0x7E>>6)).
        assert_eq!([out[1][0], out[1][1], out[1][2]], [0xFD, 0xFD, 0xFD]);
    }

    #[test]
    fn dispatch_decodes_mode1() {
        let block = make_block1(13, [[0x20; 3]; 4], [1, 0], [0u8; 16]);
        assert_eq!(bc7_mode(&block), Some(1));
        assert!(decode_bc7(&block).is_ok());
    }

    #[test]
    fn mode1_every_partition_fills_the_block() {
        // All 64 partition rows must assemble to exactly 128 bits (anchor table
        // and partition table agree on index widths); the assembler asserts it.
        for p in 0..64u32 {
            let block = make_block1(
                p,
                [[0x15; 3], [0x2A; 3], [0x3F; 3], [0; 3]],
                [0, 1],
                [1u8; 16],
            );
            assert_eq!(bc7_mode(&block), Some(1), "partition {p}");
            let _ = decode_bc7_mode1(&block);
        }
    }

    /// Assemble a two-subset RGB mode-3 block. `rgb[e]` is the 7-bit endpoint
    /// `e in 0..4` (endpoints 0,1 = subset 0; 2,3 = subset 1); `pbit` holds the
    /// four per-endpoint P-bits; `idx` the sixteen 2-bit indices (both anchors
    /// must be `<= 1` so their implicit high bit is 0).
    fn make_block3(partition: u32, rgb: [[u32; 3]; 4], pbit: [u32; 4], idx: [u8; 16]) -> [u8; 16] {
        let mut w = BitWriter::new();
        w.write(0b1000, 4); // mode-3 unary marker: three 0s then a 1.
        w.write(partition, 6);
        for e in rgb {
            w.write(e[0], 7);
        }
        for e in rgb {
            w.write(e[1], 7);
        }
        for e in rgb {
            w.write(e[2], 7);
        }
        for p in pbit {
            w.write(p, 1);
        }
        let anchor1 = BPTC_ANCHORS_2[partition as usize];
        for (t, &i) in idx.iter().enumerate() {
            let n = if t == 0 || t == anchor1 { 1 } else { 2 };
            w.write(u32::from(i), n);
        }
        assert_eq!(w.pos, 128, "mode-3 fields must fill the block exactly");
        w.bytes
    }

    #[test]
    fn make_block3_is_tagged_mode3() {
        let block = make_block3(0, [[0; 3]; 4], [0, 0, 0, 0], [0u8; 16]);
        assert_eq!(bc7_mode(&block), Some(3));
    }

    #[test]
    fn mode3_subsets_select_distinct_endpoints() {
        // Subset 0 pure black, subset 1 pure white, all texels at index 0 so
        // each resolves to its subset's endpoint 0 exactly.
        let black = [0u32; 3];
        let white = [0x7F; 3];
        let rgb = [black, black, white, white];
        // White endpoint 0 is e2; set its p-bit so (0x7F<<1)|1 = 0xFF.
        let block = make_block3(0, rgb, [0, 0, 1, 0], [0u8; 16]);
        let out = decode_bc7_mode3(&block);
        let map = &BPTC_PARTITIONS_2[0];
        for (t, texel) in out.iter().enumerate() {
            let expect = if map[t] == 0 { 0 } else { 0xFF };
            assert_eq!(
                [texel[0], texel[1], texel[2]],
                [expect; 3],
                "texel {t} subset {}",
                map[t]
            );
            assert_eq!(texel[3], 255, "mode 3 is opaque");
        }
    }

    #[test]
    fn mode3_endpoint_weights_are_exact_at_bounds() {
        // Subset 0: e0 black (p=0), e1 white (p=1); index 0 -> e0, index 3 ->
        // e1 (WEIGHT2[3] = 64) exactly.
        let rgb = [[0u32; 3], [0x7F; 3], [0u32; 3], [0x7F; 3]];
        let mut idx = [0u8; 16];
        idx[1] = 3; // texel 1 is subset 0 under partition 0, weight 64 -> e1.
        let block = make_block3(0, rgb, [0, 1, 0, 1], idx);
        let out = decode_bc7_mode3(&block);
        // texel 0 (anchor, index 0) -> e0 = (0<<1|0) = 0.
        assert_eq!([out[0][0], out[0][1], out[0][2]], [0, 0, 0]);
        // texel 1 (index 3) -> e1 = (0x7F<<1|1) = 0xFF.
        assert_eq!([out[1][0], out[1][1], out[1][2]], [0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn dispatch_decodes_mode3() {
        let block = make_block3(13, [[0x40; 3]; 4], [1, 0, 1, 0], [0u8; 16]);
        assert_eq!(bc7_mode(&block), Some(3));
        assert!(decode_bc7(&block).is_ok());
    }

    #[test]
    fn mode3_every_partition_fills_the_block() {
        // All 64 partition rows must assemble to exactly 128 bits (anchor table
        // and partition table agree on index widths); the assembler asserts it.
        for p in 0..64u32 {
            let block = make_block3(
                p,
                [[0x15; 3], [0x2A; 3], [0x7F; 3], [0; 3]],
                [0, 1, 0, 1],
                [1u8; 16],
            );
            assert_eq!(bc7_mode(&block), Some(3), "partition {p}");
            let _ = decode_bc7_mode3(&block);
        }
    }

    /// Assemble a two-subset RGBA mode-7 block. `rgba[e]` is the 5-bit endpoint
    /// `e in 0..4` (endpoints 0,1 = subset 0; 2,3 = subset 1); `pbit` holds the
    /// four per-endpoint P-bits; `idx` the sixteen 2-bit indices shared by
    /// colour and alpha (both anchors must be `<= 1`).
    fn make_block7(partition: u32, rgba: [[u32; 4]; 4], pbit: [u32; 4], idx: [u8; 16]) -> [u8; 16] {
        let mut w = BitWriter::new();
        w.write(0b1000_0000, 8); // mode-7 unary marker: seven 0s then a 1.
        w.write(partition, 6);
        for e in rgba {
            w.write(e[0], 5);
        }
        for e in rgba {
            w.write(e[1], 5);
        }
        for e in rgba {
            w.write(e[2], 5);
        }
        for e in rgba {
            w.write(e[3], 5);
        }
        for pb in pbit {
            w.write(pb, 1);
        }
        let anchor1 = BPTC_ANCHORS_2[partition as usize];
        for (t, &i) in idx.iter().enumerate() {
            let n = if t == 0 || t == anchor1 { 1 } else { 2 };
            w.write(u32::from(i), n);
        }
        assert_eq!(w.pos, 128, "mode-7 fields must fill the block exactly");
        w.bytes
    }

    #[test]
    fn make_block7_is_tagged_mode7() {
        let block = make_block7(0, [[0; 4]; 4], [0, 0, 0, 0], [0u8; 16]);
        assert_eq!(bc7_mode(&block), Some(7));
    }

    #[test]
    fn mode7_subsets_select_distinct_endpoints() {
        // Subset 0 transparent black, subset 1 opaque white, all texels at
        // index 0 so each resolves to its subset's endpoint 0 exactly.
        let black = [0u32; 4];
        let white = [0x1F; 4];
        let rgba = [black, black, white, white];
        // White endpoint 0 is e2; set its p-bit so (0x1F<<1)|1 = 0x3F -> 0xFF.
        let block = make_block7(0, rgba, [0, 0, 1, 0], [0u8; 16]);
        let out = decode_bc7_mode7(&block);
        let map = &BPTC_PARTITIONS_2[0];
        for (t, texel) in out.iter().enumerate() {
            let expect = if map[t] == 0 { 0 } else { 0xFF };
            assert_eq!(*texel, [expect; 4], "texel {t} subset {}", map[t]);
        }
    }

    #[test]
    fn mode7_alpha_is_interpolated() {
        // Mode 7 carries a real alpha channel (unlike opaque modes 1/3): subset
        // 0 e0 alpha 0, e1 alpha max; index 3 (weight 64) -> e1 alpha exactly.
        let rgba = [[0u32; 4], [0x1F; 4], [0u32; 4], [0x1F; 4]];
        let mut idx = [0u8; 16];
        idx[1] = 3; // texel 1 subset 0 under partition 0, weight 64 -> e1.
        let block = make_block7(0, rgba, [0, 1, 0, 1], idx);
        let out = decode_bc7_mode7(&block);
        assert_eq!(out[0], [0, 0, 0, 0], "anchor index 0 -> e0 (all zero)");
        assert_eq!(out[1], [0xFF, 0xFF, 0xFF, 0xFF], "index 3 -> e1 (all max)");
    }

    #[test]
    fn dispatch_decodes_mode7() {
        let block = make_block7(13, [[0x10; 4]; 4], [1, 0, 1, 0], [0u8; 16]);
        assert_eq!(bc7_mode(&block), Some(7));
        assert!(decode_bc7(&block).is_ok());
    }

    #[test]
    fn mode7_every_partition_fills_the_block() {
        for p in 0..64u32 {
            let block = make_block7(
                p,
                [[0x05; 4], [0x0A; 4], [0x1F; 4], [0; 4]],
                [0, 1, 0, 1],
                [1u8; 16],
            );
            assert_eq!(bc7_mode(&block), Some(7), "partition {p}");
            let _ = decode_bc7_mode7(&block);
        }
    }

    /// Assemble a three-subset RGB mode-2 block. `rgb[e]` is the 5-bit endpoint
    /// `e in 0..6` (endpoints `2s`,`2s+1` belong to subset `s`); `idx` the
    /// sixteen 2-bit indices (the three anchors must be `<= 1`). No P-bits.
    fn make_block2(partition: u32, rgb: [[u32; 3]; 6], idx: [u8; 16]) -> [u8; 16] {
        let mut w = BitWriter::new();
        w.write(0b100, 3); // mode-2 unary marker: two 0s then a 1.
        w.write(partition, 6);
        for e in rgb {
            w.write(e[0], 5);
        }
        for e in rgb {
            w.write(e[1], 5);
        }
        for e in rgb {
            w.write(e[2], 5);
        }
        let a1 = BPTC_ANCHORS_3_2[partition as usize];
        let a2 = BPTC_ANCHORS_3_3[partition as usize];
        for (t, &i) in idx.iter().enumerate() {
            let n = if t == 0 || t == a1 || t == a2 { 1 } else { 2 };
            w.write(u32::from(i), n);
        }
        assert_eq!(w.pos, 128, "mode-2 fields must fill the block exactly");
        w.bytes
    }

    #[test]
    fn make_block2_is_tagged_mode2() {
        let block = make_block2(0, [[0; 3]; 6], [0u8; 16]);
        assert_eq!(bc7_mode(&block), Some(2));
    }

    #[test]
    fn mode2_subsets_select_distinct_endpoints() {
        // Three subsets, three distinct endpoint colours at index 0 (anchor).
        let s0 = [0u32; 3]; // -> 0
        let s1 = [0x1Fu32; 3]; // 5-bit max -> 0xFF
        let s2 = [0x10u32; 3]; // mid -> expand_rep(0x10,5)
        let rgb = [s0, s0, s1, s1, s2, s2];
        let block = make_block2(14, rgb, [0u8; 16]);
        let out = decode_bc7_mode2(&block);
        let map = &BPTC_PARTITIONS_3[14];
        let mid = {
            let v = 0x10u32;
            (((v << 3) | (v >> 2)) & 0xFF) as u8
        };
        for (t, texel) in out.iter().enumerate() {
            let expect = match map[t] {
                0 => 0u8,
                1 => 0xFF,
                _ => mid,
            };
            assert_eq!(
                [texel[0], texel[1], texel[2]],
                [expect; 3],
                "texel {t} subset {}",
                map[t]
            );
            assert_eq!(texel[3], 255, "mode 2 is opaque");
        }
    }

    #[test]
    fn mode2_endpoint_weights_are_exact_at_bounds() {
        // Subset 0: e0 black, e1 white (5-bit max -> 0xFF); index 0 -> e0,
        // index 3 (WEIGHT2[3] = 64) -> e1 exactly.
        let rgb = [
            [0u32; 3], [0x1F; 3], [0u32; 3], [0u32; 3], [0u32; 3], [0u32; 3],
        ];
        let mut idx = [0u8; 16];
        idx[1] = 3; // texel 1 is subset 0 under partition 0, weight 64 -> e1.
        let block = make_block2(0, rgb, idx);
        let out = decode_bc7_mode2(&block);
        assert_eq!([out[0][0], out[0][1], out[0][2]], [0, 0, 0]);
        assert_eq!([out[1][0], out[1][1], out[1][2]], [0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn dispatch_decodes_mode2() {
        let block = make_block2(20, [[0x0A; 3]; 6], [0u8; 16]);
        assert_eq!(bc7_mode(&block), Some(2));
        assert!(decode_bc7(&block).is_ok());
    }

    #[test]
    fn mode2_every_partition_fills_the_block() {
        // All 64 three-subset partition rows must assemble to exactly 128 bits:
        // the two anchor tables and the partition table agree on index widths.
        for p in 0..64u32 {
            let block = make_block2(
                p,
                [
                    [0x05; 3], [0x0A; 3], [0x15; 3], [0x1F; 3], [0x08; 3], [0; 3],
                ],
                [1u8; 16],
            );
            assert_eq!(bc7_mode(&block), Some(2), "partition {p}");
            let _ = decode_bc7_mode2(&block);
        }
    }

    /// Assemble a three-subset RGB mode-0 block. `rgb[e]` is the 4-bit endpoint
    /// `e in 0..6`; `pbit` holds the six per-endpoint P-bits; `idx` the sixteen
    /// 3-bit indices (the three anchors must be `<= 3`). Mode 0 uses only the
    /// low 16 partitions (4-bit partition field).
    fn make_block0(partition: u32, rgb: [[u32; 3]; 6], pbit: [u32; 6], idx: [u8; 16]) -> [u8; 16] {
        let mut w = BitWriter::new();
        w.write(0b1, 1); // mode-0 unary marker: a single 1.
        w.write(partition, 4);
        for e in rgb {
            w.write(e[0], 4);
        }
        for e in rgb {
            w.write(e[1], 4);
        }
        for e in rgb {
            w.write(e[2], 4);
        }
        for p in pbit {
            w.write(p, 1);
        }
        let a1 = BPTC_ANCHORS_3_2[partition as usize];
        let a2 = BPTC_ANCHORS_3_3[partition as usize];
        for (t, &i) in idx.iter().enumerate() {
            let n = if t == 0 || t == a1 || t == a2 { 2 } else { 3 };
            w.write(u32::from(i), n);
        }
        assert_eq!(w.pos, 128, "mode-0 fields must fill the block exactly");
        w.bytes
    }

    #[test]
    fn make_block0_is_tagged_mode0() {
        let block = make_block0(0, [[0; 3]; 6], [0; 6], [0u8; 16]);
        assert_eq!(bc7_mode(&block), Some(0));
    }

    #[test]
    fn mode0_subsets_select_distinct_endpoints() {
        // Three subsets, three distinct endpoint colours at index 0 (anchor).
        // e1 is 4-bit max with its P-bit set -> (0xF<<1|1)=0x1F (5-bit) -> 0xFF.
        let s0 = [0u32; 3];
        let s1 = [0xFu32; 3];
        let s2 = [0x8u32; 3];
        let rgb = [s0, s0, s1, s1, s2, s2];
        let pbit = [0, 0, 1, 1, 0, 0];
        let block = make_block0(5, rgb, pbit, [0u8; 16]);
        let out = decode_bc7_mode0(&block);
        let map = &BPTC_PARTITIONS_3[5];
        let s2v = {
            let v = (0x8u32 << 1) | 0; // 4-bit + p=0 -> 5-bit 0x10
            (((v << 3) | (v >> 2)) & 0xFF) as u8
        };
        for (t, texel) in out.iter().enumerate() {
            let expect = match map[t] {
                0 => 0u8,
                1 => 0xFF,
                _ => s2v,
            };
            assert_eq!(
                [texel[0], texel[1], texel[2]],
                [expect; 3],
                "texel {t} subset {}",
                map[t]
            );
            assert_eq!(texel[3], 255, "mode 0 is opaque");
        }
    }

    #[test]
    fn mode0_endpoint_weights_are_exact_at_bounds() {
        // Subset 0: e0 black (p=0), e1 white (4-bit max + p=1 -> 0xFF);
        // index 0 -> e0, index 7 (WEIGHT3[7] = 64) -> e1 exactly.
        let rgb = [
            [0u32; 3], [0xF; 3], [0u32; 3], [0u32; 3], [0u32; 3], [0u32; 3],
        ];
        let mut idx = [0u8; 16];
        idx[1] = 7; // texel 1 subset 0 under partition 0, weight 64 -> e1.
        let block = make_block0(0, rgb, [0, 1, 0, 0, 0, 0], idx);
        let out = decode_bc7_mode0(&block);
        assert_eq!([out[0][0], out[0][1], out[0][2]], [0, 0, 0]);
        assert_eq!([out[1][0], out[1][1], out[1][2]], [0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn dispatch_decodes_mode0() {
        let block = make_block0(9, [[0x5; 3]; 6], [1, 0, 1, 0, 1, 0], [0u8; 16]);
        assert_eq!(bc7_mode(&block), Some(0));
        assert!(decode_bc7(&block).is_ok());
    }

    #[test]
    fn mode0_every_partition_fills_the_block() {
        // Mode 0 has only 16 partitions (4-bit field); each must fill 128 bits.
        for p in 0..16u32 {
            let block = make_block0(
                p,
                [[0x1; 3], [0x3; 3], [0x7; 3], [0xF; 3], [0x5; 3], [0; 3]],
                [0, 1, 0, 1, 0, 1],
                [2u8; 16],
            );
            assert_eq!(bc7_mode(&block), Some(0), "partition {p}");
            let _ = decode_bc7_mode0(&block);
        }
    }
}
