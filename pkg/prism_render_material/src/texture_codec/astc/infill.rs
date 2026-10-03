//! Weight-grid bilinear infill (Khronos decimation) for arbitrary 2D footprints.
//!
//! ASTC stores a weight *grid* that is frequently smaller than, or a different
//! shape to, the texel footprint. For a `bx` x `by` texel block the decoder
//! resamples the stored `weights_x` x `weights_y` grid up to the `bx * by` texel
//! positions using the bilinear "weight infill" / decimation of the Khronos
//! Data Format Specification 1.3 ("Weight Infill"), transcribed from the ARM
//! `astcenc` reference decoder (Apache-2.0) which every conformant unit
//! implements.
//!
//! The footprint `(bx, by)` comes from the *texture* format (e.g. `ASTC_8x8`),
//! never from the block payload. For the identity `N == M == 4` grid on a 4x4
//! footprint every texel maps exactly onto its own grid point
//! (`fs == ft == 0`), so this reduces to the direct texel->weight mapping the
//! GPU-proven 4x4 path already uses: routing 4x4 blocks through the infill does
//! not regress them.
//!
//! Decode is pure integer arithmetic -- no AI/ML path.

use super::weights::decode_grid_weights_ise;
use super::AstcError;

/// Largest 2D texel footprint handled by the decoder (12x12 = 144 texels).
pub(super) const MAX_TEXELS: usize = 144;

/// ASTC single-plane weight-grid budget: at most 64 stored weights.
const MAX_GRID_WEIGHTS: usize = 64;

/// Decode a single-plane weight grid of `weights_x` x `weights_y` weights
/// (quantised to `levels` BISE levels) from `block`, then bilinearly resample
/// it to the sixteen 4x4 texel positions, each on the `0..=64` scale in
/// row-major order (`texel = y * 4 + x`).
///
/// # Errors
/// Returns [`AstcError::Reserved`] if the stored weight range is not a valid
/// BISE level count, or if the grid exceeds the single-plane weight budget.
#[cfg(test)]
pub(super) fn infill_weights_4x4(
    block: &[u8; 16],
    weights_x: u32,
    weights_y: u32,
    levels: u32,
) -> Result<[u8; 16], AstcError> {
    let mut out = [0u8; 16];
    infill_weights(block, weights_x, weights_y, levels, 4, 4, &mut out)?;
    Ok(out)
}

/// Footprint-generic single-plane infill. Decodes a `weights_x` x `weights_y`
/// grid (quantised to `levels` BISE levels) from `block` and bilinearly
/// resamples it to the `bx * by` texel positions, writing weights on the
/// `0..=64` scale into `out[..bx*by]` in row-major order (`texel = y*bx + x`).
///
/// # Errors
/// Returns [`AstcError::Reserved`] if the stored weight range is not a valid
/// BISE level count, if the grid exceeds the single-plane weight budget, or if
/// the footprint is out of range for `out`.
pub(super) fn infill_weights(
    block: &[u8; 16],
    weights_x: u32,
    weights_y: u32,
    levels: u32,
    bx: u32,
    by: u32,
    out: &mut [u8],
) -> Result<(), AstcError> {
    check_footprint(bx, by, out.len())?;
    let weight_count = (weights_x * weights_y) as usize;
    if weight_count > MAX_GRID_WEIGHTS {
        return Err(AstcError::Reserved);
    }
    let mut grid = [0u8; MAX_GRID_WEIGHTS];
    decode_grid_weights_ise(
        block,
        weight_count as u32,
        levels,
        &mut grid[..weight_count],
    )
    .ok_or(AstcError::Reserved)?;
    expand_into(&grid, weights_x, weights_y, bx, by, out);
    Ok(())
}

/// Footprint-generic dual-plane infill. The ISE stream carries
/// `2 * weights_x * weights_y` interleaved indices (even -> plane 0, odd ->
/// plane 1); each de-interleaved plane is resampled exactly like a single-plane
/// grid into `out0`/`out1` (`bx * by` texels each) on the `0..=64` scale.
///
/// This mirrors the ARM `astcenc` reference de-interleave in
/// `unpack_weights`/`decode_ise` (Apache-2.0).
///
/// # Errors
/// Returns [`AstcError::Reserved`] if the stored weight range is not a valid
/// BISE level count, if the doubled weight count exceeds the ASTC budget, or if
/// the footprint is out of range for the output slices.
#[allow(clippy::too_many_arguments)]
pub(super) fn infill_dual_plane(
    block: &[u8; 16],
    weights_x: u32,
    weights_y: u32,
    levels: u32,
    bx: u32,
    by: u32,
    out0: &mut [u8],
    out1: &mut [u8],
) -> Result<(), AstcError> {
    check_footprint(bx, by, out0.len())?;
    check_footprint(bx, by, out1.len())?;
    let grid_points = (weights_x * weights_y) as usize;
    let total = grid_points * 2;
    if total > MAX_GRID_WEIGHTS {
        return Err(AstcError::Reserved);
    }
    let mut interleaved = [0u8; MAX_GRID_WEIGHTS];
    decode_grid_weights_ise(block, total as u32, levels, &mut interleaved[..total])
        .ok_or(AstcError::Reserved)?;

    let mut grid0 = [0u8; MAX_GRID_WEIGHTS];
    let mut grid1 = [0u8; MAX_GRID_WEIGHTS];
    for i in 0..grid_points {
        grid0[i] = interleaved[2 * i];
        grid1[i] = interleaved[2 * i + 1];
    }
    expand_into(&grid0, weights_x, weights_y, bx, by, out0);
    expand_into(&grid1, weights_x, weights_y, bx, by, out1);
    Ok(())
}

/// Validate a 2D footprint and that `out_len` can hold `bx * by` texels.
fn check_footprint(bx: u32, by: u32, out_len: usize) -> Result<(), AstcError> {
    // 2D ASTC footprints range 4..=12 in each axis (never 0/1), so `bx-1` and
    // `by-1` below are always non-zero. Reject anything out of that envelope.
    if !(2..=12).contains(&bx) || !(2..=12).contains(&by) {
        return Err(AstcError::Reserved);
    }
    let texels = (bx * by) as usize;
    if texels > MAX_TEXELS || out_len < texels {
        return Err(AstcError::Reserved);
    }
    Ok(())
}

/// Bilinearly resample the first `n * m` entries of `grid` (an `n` x `m` weight
/// grid, `n` = weights_x, `m` = weights_y) up to the `bx * by` texel positions,
/// following the Khronos decimation formula. Writes weights on the `0..=64`
/// scale into `out[..bx*by]` in row-major order (`texel = y*bx + x`).
fn expand_into(grid: &[u8], n: u32, m: u32, bx: u32, by: u32, out: &mut [u8]) {
    // Per-texel grid-space step: ds = (1024 + floor(Bs/2)) / (Bs - 1).
    let ds = (1024 + (bx >> 1)) / (bx - 1);
    let dt = (1024 + (by >> 1)) / (by - 1);
    for y in 0..by {
        for x in 0..bx {
            let cs = ds * x;
            let ct = dt * y;
            // Fixed-point grid coordinate (4 fractional bits after >>4).
            let gs = (cs * (n - 1) + 32) >> 6;
            let gt = (ct * (m - 1) + 32) >> 6;
            let js = gs >> 4;
            let fs = gs & 0xF;
            let jt = gt >> 4;
            let ft = gt & 0xF;
            // Bilinear sub-weights on a 16-unit partition of unity.
            let w11 = (fs * ft + 8) >> 4;
            let w10 = ft - w11;
            let w01 = fs - w11;
            let w00 = 16 + w11 - fs - ft;
            // The next grid column/row is only consulted with non-zero weight
            // when fs/ft is non-zero, so clamping the boundary index is safe.
            let js1 = (js + 1).min(n - 1);
            let jt1 = (jt + 1).min(m - 1);
            let p00 = u32::from(grid[(jt * n + js) as usize]);
            let p01 = u32::from(grid[(jt * n + js1) as usize]);
            let p10 = u32::from(grid[(jt1 * n + js) as usize]);
            let p11 = u32::from(grid[(jt1 * n + js1) as usize]);
            let v = (p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11 + 8) >> 4;
            out[(y * bx + x) as usize] = v as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::weights::decode_astc_4x4_weights_ise;
    use super::{infill_weights, infill_weights_4x4};

    /// 4x4 helper mirroring the old `expand` signature for the tests below.
    fn expand(grid: &[u8; 64], n: u32, m: u32) -> [u8; 16] {
        let mut out = [0u8; 16];
        super::expand_into(grid, n, m, 4, 4, &mut out);
        out
    }

    #[test]
    fn infill_4x4_identity_matches_direct_ise() {
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for levels in [2u32, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32] {
            for _ in 0..32 {
                let mut block = [0u8; 16];
                for b in block.iter_mut() {
                    *b = (next() & 0xFF) as u8;
                }
                let direct = decode_astc_4x4_weights_ise(&block, levels).expect("valid range");
                let infilled = infill_weights_4x4(&block, 4, 4, levels).expect("valid range");
                assert_eq!(direct, infilled, "levels {levels}");
            }
        }
    }

    /// An all-zero block decodes to index 0 everywhere, which unquantizes to a
    /// zero weight in every range form regardless of grid shape.
    #[test]
    fn all_zero_grid_is_all_zero() {
        for (n, m, levels) in [
            (8u32, 2u32, 6u32),
            (4, 8, 2),
            (8, 5, 2),
            (2, 7, 4),
            (5, 2, 5),
            (4, 6, 2),
        ] {
            let w = infill_weights_4x4(&[0u8; 16], n, m, levels).expect("valid range");
            assert_eq!(w, [0u8; 16], "grid {n}x{m}");
        }
    }

    /// A constant grid must resample to that same constant: the four bilinear
    /// sub-weights always sum to sixteen, so `(c * 16 + 8) >> 4 == c`.
    #[test]
    fn constant_grid_resamples_to_constant() {
        for (n, m) in [(8u32, 2u32), (4, 8), (8, 5), (2, 7), (5, 2), (4, 6), (4, 3)] {
            let mut grid = [0u8; 64];
            let c = 37u8;
            for g in grid.iter_mut().take((n * m) as usize) {
                *g = c;
            }
            assert_eq!(expand(&grid, n, m), [c; 16], "grid {n}x{m}");
        }
    }

    /// The four grid corners must land exactly on the four texel corners
    /// (`fs == ft == 0` there), with interior entries left at zero.
    #[test]
    fn grid_corners_map_to_texel_corners() {
        for (n, m) in [
            (8u32, 2u32),
            (4, 8),
            (8, 5),
            (2, 7),
            (5, 2),
            (4, 6),
            (4, 3),
            (3, 3),
        ] {
            let mut grid = [0u8; 64];
            grid[0] = 7; // (x=0,   y=0)
            grid[(n - 1) as usize] = 11; // (x=n-1, y=0)
            grid[((m - 1) * n) as usize] = 23; // (x=0,   y=m-1)
            grid[((m - 1) * n + (n - 1)) as usize] = 61; // (x=n-1, y=m-1)
            let out = expand(&grid, n, m);
            assert_eq!(out[0], 7, "grid {n}x{m} top-left");
            assert_eq!(out[3], 11, "grid {n}x{m} top-right");
            assert_eq!(out[12], 23, "grid {n}x{m} bottom-left");
            assert_eq!(out[15], 61, "grid {n}x{m} bottom-right");
        }
    }

    /// On a larger footprint a constant grid still resamples to that constant
    /// everywhere (partition-of-unity holds for any `bx`/`by`).
    #[test]
    fn constant_grid_resamples_to_constant_large_footprint() {
        for (bx, by) in [
            (5u32, 5u32),
            (6, 6),
            (8, 8),
            (10, 10),
            (12, 12),
            (8, 5),
            (10, 6),
        ] {
            let n = bx.min(6);
            let m = by.min(6);
            let mut grid = [0u8; 64];
            let c = 41u8;
            for g in grid.iter_mut().take((n * m) as usize) {
                *g = c;
            }
            let mut out = [0u8; super::MAX_TEXELS];
            super::expand_into(&grid, n, m, bx, by, &mut out[..(bx * by) as usize]);
            for (i, &v) in out[..(bx * by) as usize].iter().enumerate() {
                assert_eq!(v, c, "footprint {bx}x{by} texel {i}");
            }
        }
    }

    /// The four grid corners must land on the four footprint corners for any
    /// `bx`/`by` (grid-to-texel corner correspondence is footprint-independent).
    #[test]
    fn grid_corners_map_to_footprint_corners_large() {
        for (bx, by) in [(5u32, 5u32), (6, 6), (8, 8), (10, 10), (12, 12), (12, 10)] {
            let (n, m) = (4u32, 4u32);
            let mut grid = [0u8; 64];
            grid[0] = 9;
            grid[(n - 1) as usize] = 13;
            grid[((m - 1) * n) as usize] = 29;
            grid[((m - 1) * n + (n - 1)) as usize] = 57;
            let mut out = [0u8; super::MAX_TEXELS];
            let slice = &mut out[..(bx * by) as usize];
            super::expand_into(&grid, n, m, bx, by, slice);
            let tl = 0usize;
            let tr = (bx - 1) as usize;
            let bl = ((by - 1) * bx) as usize;
            let br = ((by - 1) * bx + (bx - 1)) as usize;
            assert_eq!(slice[tl], 9, "{bx}x{by} top-left");
            assert_eq!(slice[tr], 13, "{bx}x{by} top-right");
            assert_eq!(slice[bl], 29, "{bx}x{by} bottom-left");
            assert_eq!(slice[br], 57, "{bx}x{by} bottom-right");
        }
    }

    /// The generic entry point must agree with the 4x4 wrapper on 4x4.
    #[test]
    fn generic_matches_4x4_wrapper() {
        let mut seed = 0x9E37_79B9u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for levels in [2u32, 4, 6, 8, 12, 16] {
            for _ in 0..16 {
                let mut block = [0u8; 16];
                for b in block.iter_mut() {
                    *b = (next() & 0xFF) as u8;
                }
                let wrapped = infill_weights_4x4(&block, 4, 4, levels).expect("valid");
                let mut generic = [0u8; 16];
                infill_weights(&block, 4, 4, levels, 4, 4, &mut generic).expect("valid");
                assert_eq!(wrapped, generic, "levels {levels}");
            }
        }
    }

    /// Out-of-envelope footprints and short output slices are rejected.
    #[test]
    fn rejects_bad_footprint() {
        let block = [0u8; 16];
        let mut out = [0u8; super::MAX_TEXELS];
        assert!(infill_weights(&block, 4, 4, 6, 13, 8, &mut out).is_err());
        assert!(infill_weights(&block, 4, 4, 6, 8, 13, &mut out).is_err());
        let mut tiny = [0u8; 4];
        assert!(infill_weights(&block, 4, 4, 6, 8, 8, &mut tiny).is_err());
    }
}
