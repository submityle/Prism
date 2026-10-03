//! Non-4x4 single-plane weight-grid bilinear infill (Khronos decimation).
//!
//! ASTC stores a weight *grid* that is frequently smaller than, or a different
//! shape to, the texel footprint. For a 4x4 texel block the decoder resamples
//! the stored `weights_x` x `weights_y` grid up to the sixteen texel positions
//! using the bilinear "weight infill" / decimation of the Khronos Data Format
//! Specification 1.3 ("Weight Infill"), transcribed from the ARM `astcenc`
//! reference decoder (Apache-2.0) which every conformant unit implements.
//!
//! For the identity `N == M == 4` grid every texel maps exactly onto its own
//! grid point (`fs == ft == 0`), so this reduces to the direct texel->weight
//! mapping the GPU-proven 4x4 path already uses: routing 4x4 blocks through the
//! infill does not regress them.
//!
//! Decode is pure integer arithmetic -- no AI/ML path.

use super::weights::decode_grid_weights_ise;
use super::AstcError;

/// Texel footprint handled by the LDR decoder (4x4).
const BLOCK_S: u32 = 4;
const BLOCK_T: u32 = 4;

/// Decode a single-plane weight grid of `weights_x` x `weights_y` weights
/// (quantised to `levels` BISE levels) from `block`, then bilinearly resample
/// it to the sixteen 4x4 texel positions, each on the `0..=64` scale in
/// row-major order (`texel = y * 4 + x`).
///
/// # Errors
/// Returns [`AstcError::Reserved`] if the stored weight range is not a valid
/// BISE level count, or if the grid exceeds the single-plane weight budget.
pub(super) fn infill_weights_4x4(
    block: &[u8; 16],
    weights_x: u32,
    weights_y: u32,
    levels: u32,
) -> Result<[u8; 16], AstcError> {
    let weight_count = weights_x * weights_y;
    let mut grid = [0u8; 64];
    if weight_count as usize > grid.len() {
        return Err(AstcError::Reserved);
    }
    decode_grid_weights_ise(
        block,
        weight_count,
        levels,
        &mut grid[..weight_count as usize],
    )
    .ok_or(AstcError::Reserved)?;
    Ok(expand(&grid, weights_x, weights_y))
}

/// Decode a **dual-plane** weight grid of `weights_x` x `weights_y` grid points
/// from `block`. Dual-plane blocks store `2 * weights_x * weights_y` weights,
/// interleaved so that grid point `i` contributes `plane0 = seq[2*i]` and
/// `plane1 = seq[2*i + 1]`. Each plane is then bilinearly resampled to the
/// sixteen 4x4 texel positions independently (same decimation as the
/// single-plane path), and the two resampled planes are returned on the
/// `0..=64` scale in row-major texel order (`texel = y * 4 + x`).
///
/// This mirrors the ARM `astcenc` reference de-interleave in
/// `unpack_weights`/`decode_ise` (Apache-2.0): the ISE sequence is a single
/// stream of `2 * N` indices; even indices feed plane 0 and odd indices feed
/// plane 1, after which each plane decimates exactly like a single-plane grid.
///
/// # Errors
/// Returns [`AstcError::Reserved`] if the stored weight range is not a valid
/// BISE level count, or if the doubled weight count exceeds the ASTC budget.
pub(super) fn infill_dual_plane_4x4(
    block: &[u8; 16],
    weights_x: u32,
    weights_y: u32,
    levels: u32,
) -> Result<([u8; 16], [u8; 16]), AstcError> {
    let grid_points = (weights_x * weights_y) as usize;
    let total = grid_points * 2;
    let mut interleaved = [0u8; 64];
    if total > interleaved.len() {
        return Err(AstcError::Reserved);
    }
    decode_grid_weights_ise(block, total as u32, levels, &mut interleaved[..total])
        .ok_or(AstcError::Reserved)?;

    let mut grid0 = [0u8; 64];
    let mut grid1 = [0u8; 64];
    for i in 0..grid_points {
        grid0[i] = interleaved[2 * i];
        grid1[i] = interleaved[2 * i + 1];
    }
    Ok((
        expand(&grid0, weights_x, weights_y),
        expand(&grid1, weights_x, weights_y),
    ))
}

/// Bilinearly resample the first `n * m` entries of `grid` (an `n` x `m` weight
/// grid, `n` = weights_x, `m` = weights_y) up to the sixteen texel positions of
/// a 4x4 block, following the Khronos decimation formula. Returns weights on
/// the `0..=64` scale in row-major texel order (`texel = y * 4 + x`).
fn expand(grid: &[u8; 64], n: u32, m: u32) -> [u8; 16] {
    // Per-texel grid-space step: ds = (1024 + floor(Bs/2)) / (Bs - 1).
    let ds = (1024 + (BLOCK_S >> 1)) / (BLOCK_S - 1);
    let dt = (1024 + (BLOCK_T >> 1)) / (BLOCK_T - 1);
    let mut out = [0u8; 16];
    for y in 0..BLOCK_T {
        for x in 0..BLOCK_S {
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
            out[(y * BLOCK_S + x) as usize] = v as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::weights::decode_astc_4x4_weights_ise;
    use super::{expand, infill_weights_4x4};

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
}
