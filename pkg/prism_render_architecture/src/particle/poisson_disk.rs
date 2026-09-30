//! Blue-noise point sets via `Bridson`'s Poisson-disk sampling (design §16-§21,
//! §25).
//!
//! Several particle passes want a *random yet minimum-spaced* 2D point set: a
//! surface emitter that scatters spawn slots without visible clumps, a
//! stipple/coverage mask, or a jitter kernel whose taps never collide. A raw
//! uniform draw clumps and leaves holes; a low-discrepancy sequence (see the
//! sibling [`super::halton_sequence`]) is deterministic and *not* random. This
//! module fills the remaining niche: a genuine pseudo-random point set in which
//! every pair of points is at least a radius `r` apart — *blue noise*.
//!
//! The generator is `Bridson`'s algorithm:
//!
//! * A background acceleration grid with cell size `r / sqrt(2)` guarantees each
//!   cell holds at most one point, so a fixed 5x5 cell neighborhood covers every
//!   point that could sit closer than `r`.
//! * An *active list* of not-yet-exhausted points drives the frontier. Each step
//!   pops a random active point and fires `k` candidate darts into the annulus
//!   `[r, 2r]` around it; the first candidate that lands inside the domain and
//!   clears the min-distance test is accepted and pushed onto the frontier.
//! * A point whose `k` candidates all fail is retired from the active list.
//!
//! Randomness comes from a hand-rolled integer-hash `RNG` (a `splitmix32`-style
//! bit-mixer) seeded by [`PoissonConfig::seed`], so a run is fully reproducible
//! and needs no host entropy. Unlike [`super::determinism`]'s stateless hash
//! `RNG`, this is a tiny *stateful stream* local to one sampling pass; this file
//! deliberately imports neither it nor any other particle module.
//!
//! No transcendental functions appear: the annulus darts are drawn by rejection
//! sampling inside the `[-2r, 2r]^2` square (accepting squared radii in
//! `[r^2, 4r^2]`) rather than with `sin`/`cos`, and only `f32::floor`, a single
//! `core::f32::consts::SQRT_2` division, and squared-distance algebra are used.
//! The final `sqrt` lives only in the [`min_pairwise_distance`] verifier.

use alloc::vec::Vec;

/// `2^32` as an `f32`; the exact span used to normalize a `u32` word into the
/// unit interval (`2^32` is representable in an `f32` mantissa).
const U32_SPAN: f32 = 4_294_967_296.0;

/// `2^-32`, the reciprocal of [`U32_SPAN`]; multiplying a `u32` draw by this
/// maps it into `[0, 1)`. Exact, since it is a power of two.
const INV_U32_SPAN: f32 = 1.0 / U32_SPAN;

/// Sentinel stored in an empty acceleration-grid cell (no point index).
const EMPTY_CELL: u32 = u32::MAX;

/// Upper bound on rejection-sampling darts used to draw one annulus offset,
/// guarding the (statistically negligible) tail against an unbounded loop.
const ANNULUS_MAX_TRIES: u32 = 32;

/// A tiny stateful `splitmix32`-style integer-hash `RNG`.
///
/// The generator holds a single `u32` of state, advances it by the golden-ratio
/// odd constant, and finalizes each step with two xor-shift multiplies. It is
/// intentionally self-contained (no `super::determinism` dependency) so a
/// Poisson-disk pass is reproducible from a bare seed.
#[derive(Clone, Copy, Debug)]
pub struct Rng {
    /// The current 32-bit generator state.
    state: u32,
}

impl Rng {
    /// Creates a generator from a 32-bit `seed`.
    #[must_use]
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// Advances the state and returns the next pseudo-random `u32`.
    ///
    /// This is the `splitmix32` finalizer: bump the state by `0x9E3779B9` (the
    /// odd 32-bit fraction of the golden ratio), then mix with two xor-shift
    /// multiplies. Pure integer arithmetic with wrapping multiplies.
    #[must_use]
    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x9E37_79B9);
        let mut z = self.state;
        z = (z ^ (z >> 16)).wrapping_mul(0x21F0_AAAD);
        z = (z ^ (z >> 15)).wrapping_mul(0x735A_2D97);
        z ^ (z >> 15)
    }

    /// Returns the next pseudo-random `f32` in `[0, 1)`.
    ///
    /// The `u32` draw is widened to `f32` and scaled by `2^-32`; the intentional
    /// low-bit rounding is exactly the behavior wanted for a unit-interval draw.
    #[expect(
        clippy::cast_precision_loss,
        reason = "uniform u32 widened for a [0,1) draw; low-bit rounding is intended"
    )]
    #[must_use]
    pub fn next_unit(&mut self) -> f32 {
        let v = self.next_u32();
        (v as f32) * INV_U32_SPAN
    }
}

/// Configuration for one Poisson-disk sampling pass.
#[derive(Clone, Copy, Debug)]
pub struct PoissonConfig {
    /// Minimum permitted spacing `r` between any two accepted points.
    pub radius: f32,
    /// Width of the sampling domain `[0, width)`.
    pub width: f32,
    /// Height of the sampling domain `[0, height)`.
    pub height: f32,
    /// Number of candidate darts fired per active point before it is retired.
    pub k_candidates: u32,
    /// Seed for the internal integer-hash `RNG`.
    pub seed: u32,
}

/// Returns the `Bridson` background-grid cell size for spacing `radius`.
///
/// The cell size is `radius / sqrt(2)`, chosen so a grid cell's diagonal is
/// exactly `radius`; therefore at most one accepted point can occupy a cell and
/// a 5x5 cell window around a candidate contains every possible conflict.
#[must_use]
pub fn cell_size(radius: f32) -> f32 {
    radius / core::f32::consts::SQRT_2
}

/// Maps a non-negative domain coordinate to its 1D grid cell index.
///
/// The coordinate is floored after division by the cell size. Inputs are always
/// non-negative and bounded (candidates are domain-tested first), so the
/// `f32 -> usize` narrowing is exact for the values that reach it.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "coord is a small, non-negative, bounded f32 floor -> exact cell index"
)]
fn cell_index_1d(coord: f32, cell: f32) -> usize {
    let scaled = (coord / cell).floor().max(0.0);
    scaled as usize
}

/// Draws one offset uniformly from the annulus `[radius, 2*radius]` by rejection
/// sampling inside the `[-2r, 2r]^2` square (no `sin`/`cos`).
///
/// Returns `None` only if every dart in the bounded try budget lands outside the
/// annulus, which is astronomically unlikely given the ~0.59 acceptance ratio.
fn annulus_offset(rng: &mut Rng, radius: f32) -> Option<[f32; 2]> {
    let r2 = radius * radius;
    let four_r2 = 4.0 * r2;
    let mut tries = 0u32;
    while tries < ANNULUS_MAX_TRIES {
        tries += 1;
        let ox = (rng.next_unit() * 4.0 - 2.0) * radius;
        let oy = (rng.next_unit() * 4.0 - 2.0) * radius;
        let d2 = ox * ox + oy * oy;
        if (r2..=four_r2).contains(&d2) {
            return Some([ox, oy]);
        }
    }
    None
}

/// Tests whether a candidate at `(x, y)` clears the min-distance `r` (compared as
/// `r^2`) against every point in its 5x5 grid neighborhood.
fn candidate_fits(
    points: &[[f32; 2]],
    grid: &[u32],
    cols: usize,
    rows: usize,
    cell: f32,
    x: f32,
    y: f32,
    r2: f32,
) -> bool {
    let col = cell_index_1d(x, cell);
    let row = cell_index_1d(y, cell);
    let col_lo = col.saturating_sub(2);
    let col_hi = (col + 2).min(cols - 1);
    let row_lo = row.saturating_sub(2);
    let row_hi = (row + 2).min(rows - 1);
    for nr in row_lo..=row_hi {
        for nc in col_lo..=col_hi {
            let slot = grid[nr * cols + nc];
            if slot != EMPTY_CELL {
                let idx = usize::try_from(slot).unwrap_or(0);
                let p = points[idx];
                let dx = p[0] - x;
                let dy = p[1] - y;
                if dx * dx + dy * dy < r2 {
                    return false;
                }
            }
        }
    }
    true
}

/// Registers point `p` in the point list, the active frontier, and the grid.
fn insert_point(
    points: &mut Vec<[f32; 2]>,
    active: &mut Vec<usize>,
    grid: &mut [u32],
    cols: usize,
    cell: f32,
    p: [f32; 2],
) {
    let idx = points.len();
    points.push(p);
    active.push(idx);
    let col = cell_index_1d(p[0], cell);
    let row = cell_index_1d(p[1], cell);
    grid[row * cols + col] = u32::try_from(idx).unwrap_or(EMPTY_CELL);
}

/// Generates a Poisson-disk (blue-noise) point set for the given configuration.
///
/// Returns the accepted points in `[0, width) x [0, height)`; every pair is at
/// least `radius` apart. A non-positive radius or domain extent yields an empty
/// set. The result is fully determined by [`PoissonConfig::seed`].
#[must_use]
pub fn poisson_disk_sample(cfg: PoissonConfig) -> Vec<[f32; 2]> {
    let mut points: Vec<[f32; 2]> = Vec::new();
    if cfg.radius <= 0.0 || cfg.width <= 0.0 || cfg.height <= 0.0 {
        return points;
    }

    let cell = cell_size(cfg.radius);
    let cols = cell_index_1d(cfg.width, cell) + 1;
    let rows = cell_index_1d(cfg.height, cell) + 1;
    let mut grid: Vec<u32> = Vec::new();
    grid.resize(cols * rows, EMPTY_CELL);

    let mut active: Vec<usize> = Vec::new();
    let mut rng = Rng::new(cfg.seed);
    let r2 = cfg.radius * cfg.radius;

    let ix = rng.next_unit() * cfg.width;
    let iy = rng.next_unit() * cfg.height;
    insert_point(&mut points, &mut active, &mut grid, cols, cell, [ix, iy]);

    while !active.is_empty() {
        let slot = usize::try_from(rng.next_u32()).unwrap_or(0) % active.len();
        let base = points[active[slot]];
        let mut found = false;
        let mut c = 0u32;
        while c < cfg.k_candidates {
            c += 1;
            let Some(off) = annulus_offset(&mut rng, cfg.radius) else {
                continue;
            };
            let x = base[0] + off[0];
            let y = base[1] + off[1];
            if !(0.0..cfg.width).contains(&x) || !(0.0..cfg.height).contains(&y) {
                continue;
            }
            if candidate_fits(&points, &grid, cols, rows, cell, x, y, r2) {
                insert_point(&mut points, &mut active, &mut grid, cols, cell, [x, y]);
                found = true;
                break;
            }
        }
        if !found {
            active.swap_remove(slot);
        }
    }

    points
}

/// Returns the smallest Euclidean distance between any two points, or `0.0` when
/// fewer than two points are supplied. Intended as a verification helper.
///
/// The single `sqrt` here is deliberate: all sampling-time comparisons stay in
/// squared space, and only this reporting step converts to a linear distance.
#[must_use]
pub fn min_pairwise_distance(points: &[[f32; 2]]) -> f32 {
    let mut best_sq = f32::INFINITY;
    for (i, a) in points.iter().enumerate() {
        for b in &points[i + 1..] {
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            let d2 = dx * dx + dy * dy;
            if d2 < best_sq {
                best_sq = d2;
            }
        }
    }
    if best_sq.is_infinite() {
        0.0
    } else {
        best_sq.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for exact-ish `f32` comparisons in tests.
    const CMP_EPS: f32 = 1e-6;

    fn cfg(radius: f32, width: f32, height: f32, seed: u32) -> PoissonConfig {
        PoissonConfig {
            radius,
            width,
            height,
            k_candidates: 30,
            seed,
        }
    }

    fn bitwise_same(a: &[[f32; 2]], b: &[[f32; 2]]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(p, q)| p[0].to_bits() == q[0].to_bits() && p[1].to_bits() == q[1].to_bits())
    }

    #[test]
    fn cell_size_is_radius_over_sqrt2() {
        let c = cell_size(2.0);
        assert!((c - 2.0 / core::f32::consts::SQRT_2).abs() < CMP_EPS);
        // sqrt(2) * cell = radius.
        assert!((c * core::f32::consts::SQRT_2 - 2.0).abs() < CMP_EPS);
    }

    #[test]
    fn cell_size_scales_and_is_positive() {
        assert!(cell_size(1.0) > 0.0);
        assert!(cell_size(4.0) > cell_size(2.0));
    }

    #[test]
    fn rng_is_deterministic_for_same_seed() {
        let mut a = Rng::new(12345);
        let mut b = Rng::new(12345);
        for _ in 0..64 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn rng_differs_across_seeds() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let mut any_diff = false;
        for _ in 0..64 {
            if a.next_u32() != b.next_u32() {
                any_diff = true;
            }
        }
        assert!(any_diff);
    }

    #[test]
    fn rng_next_unit_in_unit_interval() {
        let mut rng = Rng::new(7);
        for _ in 0..4096 {
            let u = rng.next_unit();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn rng_uniformity_coarse_buckets() {
        let mut rng = Rng::new(99);
        let mut buckets = [0u32; 10];
        let n = 20_000u32;
        for _ in 0..n {
            let u = rng.next_unit();
            let mut b = cell_index_1d(u * 10.0, 1.0);
            if b > 9 {
                b = 9;
            }
            buckets[b] += 1;
        }
        // Every bucket should be within +/-30% of the ideal n/10 count.
        for &count in &buckets {
            assert!(count > (n / 10) * 7 / 10);
            assert!(count < (n / 10) * 13 / 10);
        }
    }

    #[test]
    fn min_pairwise_distance_empty_is_zero() {
        let pts: [[f32; 2]; 0] = [];
        assert!(min_pairwise_distance(&pts).abs() < CMP_EPS);
    }

    #[test]
    fn min_pairwise_distance_single_is_zero() {
        let pts = [[1.0f32, 2.0]];
        assert!(min_pairwise_distance(&pts).abs() < CMP_EPS);
    }

    #[test]
    fn min_pairwise_distance_known_value() {
        let pts = [[0.0f32, 0.0], [3.0, 4.0], [10.0, 10.0]];
        // Closest pair is (0,0)-(3,4) at distance 5.
        assert!((min_pairwise_distance(&pts) - 5.0).abs() < CMP_EPS);
    }

    #[test]
    fn sample_points_respect_min_distance() {
        let pts = poisson_disk_sample(cfg(4.0, 80.0, 80.0, 2024));
        assert!(pts.len() > 1);
        let dmin = min_pairwise_distance(&pts);
        // Accepted points are at least r apart (allow a tiny f32 slack).
        assert!(dmin + 4.0 * 1e-3 >= 4.0);
    }

    #[test]
    fn sample_points_lie_in_domain() {
        let w = 60.0f32;
        let h = 45.0f32;
        let pts = poisson_disk_sample(cfg(3.0, w, h, 555));
        for p in &pts {
            assert!((0.0..w).contains(&p[0]));
            assert!((0.0..h).contains(&p[1]));
        }
    }

    #[test]
    fn sample_is_deterministic_for_same_seed() {
        let a = poisson_disk_sample(cfg(3.0, 50.0, 50.0, 42));
        let b = poisson_disk_sample(cfg(3.0, 50.0, 50.0, 42));
        assert!(bitwise_same(&a, &b));
    }

    #[test]
    fn sample_differs_across_seeds() {
        let a = poisson_disk_sample(cfg(3.0, 50.0, 50.0, 1));
        let b = poisson_disk_sample(cfg(3.0, 50.0, 50.0, 9999));
        assert!(!bitwise_same(&a, &b));
    }

    #[test]
    fn sample_point_count_grows_with_area() {
        let small = poisson_disk_sample(cfg(3.0, 30.0, 30.0, 77));
        let large = poisson_disk_sample(cfg(3.0, 90.0, 90.0, 77));
        assert!(large.len() > small.len());
    }

    #[test]
    fn sample_point_count_grows_when_radius_shrinks() {
        let coarse = poisson_disk_sample(cfg(8.0, 80.0, 80.0, 321));
        let fine = poisson_disk_sample(cfg(3.0, 80.0, 80.0, 321));
        assert!(fine.len() > coarse.len());
    }

    #[test]
    fn sample_produces_many_points_on_large_domain() {
        let pts = poisson_disk_sample(cfg(2.0, 100.0, 100.0, 3));
        // A 100x100 domain at r=2 comfortably holds many points.
        assert!(pts.len() > 50);
    }

    #[test]
    fn sample_all_points_distinct() {
        let pts = poisson_disk_sample(cfg(3.0, 60.0, 60.0, 88));
        for (i, a) in pts.iter().enumerate() {
            for b in &pts[i + 1..] {
                assert!(a[0].to_bits() != b[0].to_bits() || a[1].to_bits() != b[1].to_bits());
            }
        }
    }

    #[test]
    fn degenerate_tiny_domain_yields_single_point() {
        // Domain smaller than a disk radius: only the seed point fits.
        let pts = poisson_disk_sample(cfg(10.0, 2.0, 2.0, 5));
        assert_eq!(pts.len(), 1);
        assert!((0.0..2.0).contains(&pts[0][0]));
        assert!((0.0..2.0).contains(&pts[0][1]));
    }

    #[test]
    fn zero_radius_returns_empty() {
        assert!(poisson_disk_sample(cfg(0.0, 50.0, 50.0, 1)).is_empty());
    }

    #[test]
    fn zero_width_returns_empty() {
        assert!(poisson_disk_sample(cfg(3.0, 0.0, 50.0, 1)).is_empty());
    }

    #[test]
    fn zero_height_returns_empty() {
        assert!(poisson_disk_sample(cfg(3.0, 50.0, 0.0, 1)).is_empty());
    }

    #[test]
    fn zero_candidates_returns_only_seed() {
        let pts = poisson_disk_sample(PoissonConfig {
            radius: 3.0,
            width: 50.0,
            height: 50.0,
            k_candidates: 0,
            seed: 11,
        });
        assert_eq!(pts.len(), 1);
    }

    #[test]
    fn annulus_offset_lands_in_ring() {
        let mut rng = Rng::new(2718);
        let radius = 5.0f32;
        for _ in 0..2000 {
            if let Some(off) = annulus_offset(&mut rng, radius) {
                let d2 = off[0] * off[0] + off[1] * off[1];
                assert!(d2 + CMP_EPS >= radius * radius);
                assert!(d2 <= 4.0 * radius * radius + CMP_EPS);
            }
        }
    }

    #[test]
    fn cell_index_is_monotone_nondecreasing() {
        let cell = cell_size(4.0);
        let a = cell_index_1d(1.0, cell);
        let b = cell_index_1d(10.0, cell);
        assert!(b >= a);
        assert_eq!(cell_index_1d(0.0, cell), 0);
    }
}
