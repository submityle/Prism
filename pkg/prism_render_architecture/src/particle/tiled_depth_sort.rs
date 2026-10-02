//! Per-`tile` / per-`slice` local transparency depth sort: the blocked variant
//! of the global view-depth sort (design §12).
//!
//! Design §12 lists, among the sort optimizations, *per-`tile` / per-`slice`
//! local sorting*. The core global sort (quantized view-depth key, one-sweep
//! radix / bitonic) is owned by [`super::sort_cull`]; this module is its
//! blocked variant. Instead of ordering every visible alpha-blended particle in
//! one global list, it buckets particles into screen-space `tile`s (optionally
//! sub-divided into depth `slice`s) and orders each bucket *locally*. Two
//! properties motivate the split:
//!
//! 1. **Cache locality** — a `tile`'s particles land in a contiguous output
//!    range, so the compositor streams one `tile` at a time instead of chasing
//!    a globally interleaved order.
//! 2. **Parallelism** — once particles are bucketed, each `tile`'s local sort is
//!    independent, matching how a `GPU` dispatch fans local sorts across
//!    workgroups.
//!
//! It reuses [`super::sort_cull::sort_key`] for the depth key, so the ordering
//! semantics (front-to-back vs. back-to-front, 16-bit quantization) match the
//! global path bit for bit: a single `tile` degenerates to exactly the global
//! sort. No sort key, blend convention, or depth quantization is re-defined
//! here; the key comes straight from [`super::sort_cull`].
//!
//! Determinism: bucketing is a stable counting sort (input order preserved
//! within a `tile`), and each `tile`'s local sort uses a total order
//! `(key, original_index)` whose index tie-break makes equal-depth particles
//! bit-reproducible regardless of the underlying sort's stability. Only
//! ordinary arithmetic, integer casts, and `is_nan` guards are used — no
//! transcendental functions (the crate's determinism lint permits only `sqrt`,
//! which this module does not even need).

use alloc::vec;
use alloc::vec::Vec;

use super::sort_cull::sort_key;

/// A screen-space rectangle the `tile` grid spans, in whatever units the caller
/// uses for the particle screen positions (pixels, normalized device
/// coordinates, …).
///
/// A degenerate axis (`max` not strictly greater than `min`) collapses that
/// axis to a single `tile` column/row so assignment stays defined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenRect {
    /// Left edge (minimum x).
    pub min_x: f32,
    /// Bottom edge (minimum y).
    pub min_y: f32,
    /// Right edge (maximum x).
    pub max_x: f32,
    /// Top edge (maximum y).
    pub max_y: f32,
}

impl ScreenRect {
    /// Builds a rectangle from its two corners.
    #[must_use]
    pub fn new(min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }
}

/// Parameters defining the `tile` grid and the depth key convention.
///
/// The grid has `tiles_x` columns, `tiles_y` rows, and `depth_slices` depth
/// `slice`s, so there are `tiles_x * tiles_y * depth_slices` `tile`s total. A
/// zero count on any axis is treated as one (an empty grid still has one
/// `tile`). `near` / `far` / `back_to_front` are forwarded verbatim to
/// [`super::sort_cull::sort_key`], so the local order matches the global order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileGridParams {
    /// Screen extent the grid covers.
    pub rect: ScreenRect,
    /// Number of `tile` columns across the screen (clamped to at least one).
    pub tiles_x: u32,
    /// Number of `tile` rows down the screen (clamped to at least one).
    pub tiles_y: u32,
    /// Number of depth `slice`s; one disables depth slicing (clamped to at
    /// least one).
    pub depth_slices: u32,
    /// Near plane forwarded to the depth key.
    pub near: f32,
    /// Far plane forwarded to the depth key.
    pub far: f32,
    /// When `true`, the key is inverted so an ascending local sort draws far
    /// particles first (correct for alpha blending); see
    /// [`super::sort_cull::sort_key`].
    pub back_to_front: bool,
}

impl TileGridParams {
    /// Column count, never below one.
    #[must_use]
    pub fn effective_tiles_x(self) -> u32 {
        self.tiles_x.max(1)
    }

    /// Row count, never below one.
    #[must_use]
    pub fn effective_tiles_y(self) -> u32 {
        self.tiles_y.max(1)
    }

    /// Depth `slice` count, never below one.
    #[must_use]
    pub fn effective_depth_slices(self) -> u32 {
        self.depth_slices.max(1)
    }

    /// Total number of `tile`s in the grid.
    #[must_use]
    pub fn tile_count(self) -> u32 {
        self.effective_tiles_x() * self.effective_tiles_y() * self.effective_depth_slices()
    }
}

/// One transparent particle presented to the sort: its screen position (for
/// `tile` assignment) and its view-space depth (for the ordering key).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TiledParticle {
    /// Screen-space x used to pick the `tile` column.
    pub screen_x: f32,
    /// Screen-space y used to pick the `tile` row.
    pub screen_y: f32,
    /// View-space depth fed to [`super::sort_cull::sort_key`].
    pub depth: f32,
}

impl TiledParticle {
    /// Builds a particle record.
    #[must_use]
    pub fn new(screen_x: f32, screen_y: f32, depth: f32) -> Self {
        Self {
            screen_x,
            screen_y,
            depth,
        }
    }
}

/// The integer grid cell a particle maps to.
///
/// `index` is the flattened `tile` id in `0..params.tile_count()`, laid out
/// column-fastest then row then depth `slice`
/// (`(z * tiles_y + y) * tiles_x + x`), so `slice` 0 is nearest and consecutive
/// columns stay adjacent in memory.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TileCoord {
    /// Column (`0..tiles_x`).
    pub x: u32,
    /// Row (`0..tiles_y`).
    pub y: u32,
    /// Depth `slice` (`0..depth_slices`).
    pub z: u32,
    /// Flattened `tile` id.
    pub index: u32,
}

/// The contiguous output range owned by one `tile`.
///
/// `order[start as usize .. (start + count) as usize]` holds that `tile`'s
/// particle indices, already locally sorted. Empty `tile`s have `count == 0`
/// (and a `start` equal to the end of the preceding `tile`).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TileRange {
    /// First position in [`TiledSortResult::order`] for this `tile`.
    pub start: u32,
    /// Number of particles in this `tile`.
    pub count: u32,
}

impl TileRange {
    /// One-past-the-last position for this `tile`.
    #[must_use]
    pub fn end(self) -> u32 {
        self.start + self.count
    }

    /// Returns `true` when the `tile` holds no particles.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.count == 0
    }
}

/// Result of a tiled depth sort: a flat index permutation plus the per-`tile`
/// ranges that carve it up.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TiledSortResult {
    /// Particle indices grouped by `tile` and locally depth-sorted within each
    /// `tile`. Length equals the number of input particles.
    pub order: Vec<u32>,
    /// One [`TileRange`] per `tile`, length `params.tile_count()`.
    pub tiles: Vec<TileRange>,
}

impl TiledSortResult {
    /// Borrows the locally-sorted index slice for `tile`, or an empty slice when
    /// `tile` is out of range.
    #[must_use]
    pub fn tile_indices(&self, tile: u32) -> &[u32] {
        match self.tiles.get(tile as usize) {
            Some(range) => &self.order[range.start as usize..range.end() as usize],
            None => &[],
        }
    }
}

/// Maps a coordinate on one axis to a `tile` bucket in `0..count`.
///
/// A `NaN` coordinate or anything at/below the minimum lands in bucket 0; a
/// coordinate at/above the maximum lands in the last bucket; a degenerate span
/// (`max` not strictly greater than `min`) collapses to bucket 0. This avoids
/// any floating-point equality test and never divides by zero.
fn axis_bucket(coord: f32, min: f32, max: f32, count: u32) -> u32 {
    if count <= 1 {
        return 0;
    }
    let span = max - min;
    if span.is_nan() || span <= 0.0 {
        return 0;
    }
    let t = (coord - min) / span;
    if t.is_nan() || t <= 0.0 {
        return 0;
    }
    if t >= 1.0 {
        return count - 1;
    }
    // `t` is in the open interval (0, 1), so the truncating cast yields a bucket
    // in `0..=count-1`; the guard keeps it in range against rounding at the top.
    let bucket = (t * count as f32) as u32;
    if bucket >= count {
        count - 1
    } else {
        bucket
    }
}

/// Computes the grid cell a particle falls into (design §12).
///
/// Columns and rows come from the screen position within [`ScreenRect`]; the
/// depth `slice` comes from the view-space depth mapped linearly into
/// `[near, far]` (independent of `back_to_front`, since a `slice` is a spatial
/// band, not an ordering). A degenerate near/far range collapses to `slice` 0.
#[must_use]
pub fn tile_of(params: TileGridParams, particle: TiledParticle) -> TileCoord {
    let nx = params.effective_tiles_x();
    let ny = params.effective_tiles_y();
    let nz = params.effective_depth_slices();

    let x = axis_bucket(particle.screen_x, params.rect.min_x, params.rect.max_x, nx);
    let y = axis_bucket(particle.screen_y, params.rect.min_y, params.rect.max_y, ny);
    let z = axis_bucket(particle.depth, params.near, params.far, nz);

    let index = (z * ny + y) * nx + x;
    TileCoord { x, y, z, index }
}

/// Sorts transparent particles into per-`tile` locally-ordered ranges
/// (design §12).
///
/// Two deterministic phases:
///
/// 1. **Bucket** — a stable counting sort places each particle index into its
///    `tile`'s contiguous output range, preserving input order within a `tile`.
/// 2. **Local order** — each `tile`'s range is sorted by the total order
///    `(key, original_index)`, where `key` is [`super::sort_cull::sort_key`] of
///    the particle depth. The unique-index tie-break makes equal-depth
///    particles bit-reproducible.
///
/// With a single `tile` (`tiles_x = tiles_y = depth_slices = 1`) the result is
/// exactly the global view-depth order the reference radix / bitonic sort would
/// produce. Empty input yields an empty `order` and one empty [`TileRange`] per
/// `tile`.
#[must_use]
pub fn tiled_depth_sort(params: TileGridParams, particles: &[TiledParticle]) -> TiledSortResult {
    let num_tiles = params.tile_count() as usize;
    let n = particles.len();

    // Per-particle `tile` id and depth key, computed once.
    let mut tile_ids: Vec<u32> = Vec::with_capacity(n);
    let mut keys: Vec<u16> = Vec::with_capacity(n);
    for particle in particles {
        tile_ids.push(tile_of(params, *particle).index);
        keys.push(sort_key(
            particle.depth,
            params.near,
            params.far,
            params.back_to_front,
        ));
    }

    // Counting sort over `tile` ids: histogram, then exclusive prefix sum into
    // per-`tile` start offsets.
    let mut counts = vec![0u32; num_tiles];
    for &tile in &tile_ids {
        counts[tile as usize] += 1;
    }

    let mut tiles = Vec::with_capacity(num_tiles);
    let mut running = 0u32;
    for &count in &counts {
        tiles.push(TileRange {
            start: running,
            count,
        });
        running += count;
    }

    // Stable scatter: a moving cursor per `tile` preserves input order within
    // each bucket, so the subsequent index tie-break is well-defined.
    let mut cursors: Vec<u32> = tiles.iter().map(|range| range.start).collect();
    let mut order = vec![0u32; n];
    for (index, &tile) in tile_ids.iter().enumerate() {
        let slot = &mut cursors[tile as usize];
        order[*slot as usize] = index as u32;
        *slot += 1;
    }

    // Local depth sort inside each `tile`: total order on `(key, index)`.
    for range in &tiles {
        let span = &mut order[range.start as usize..range.end() as usize];
        span.sort_by(|&a, &b| {
            keys[a as usize]
                .cmp(&keys[b as usize])
                .then_with(|| a.cmp(&b))
        });
    }

    TiledSortResult { order, tiles }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NEAR: f32 = 0.0;
    const FAR: f32 = 100.0;

    fn single_tile(back_to_front: bool) -> TileGridParams {
        TileGridParams {
            rect: ScreenRect::new(0.0, 0.0, 10.0, 10.0),
            tiles_x: 1,
            tiles_y: 1,
            depth_slices: 1,
            near: NEAR,
            far: FAR,
            back_to_front,
        }
    }

    /// Reference global order using the exact `sort_cull` key and the same
    /// `(key, index)` tie-break, standing in for the global radix / bitonic
    /// sort.
    fn global_order(particles: &[TiledParticle], back_to_front: bool) -> Vec<u32> {
        let mut idx: Vec<u32> = (0..particles.len() as u32).collect();
        idx.sort_by(|&a, &b| {
            let ka = sort_key(particles[a as usize].depth, NEAR, FAR, back_to_front);
            let kb = sort_key(particles[b as usize].depth, NEAR, FAR, back_to_front);
            ka.cmp(&kb).then_with(|| a.cmp(&b))
        });
        idx
    }

    #[test]
    fn empty_input_is_safe() {
        let params = TileGridParams {
            tiles_x: 3,
            tiles_y: 2,
            depth_slices: 2,
            ..single_tile(true)
        };
        let result = tiled_depth_sort(params, &[]);
        assert!(result.order.is_empty());
        assert_eq!(result.tiles.len(), params.tile_count() as usize);
        assert!(result.tiles.iter().all(|range| range.is_empty()));
    }

    #[test]
    fn single_element_is_safe() {
        let particles = [TiledParticle::new(5.0, 5.0, 42.0)];
        let result = tiled_depth_sort(single_tile(true), &particles);
        assert_eq!(result.order, vec![0]);
        assert_eq!(result.tiles.len(), 1);
        assert_eq!(result.tiles[0], TileRange { start: 0, count: 1 });
        assert_eq!(result.tile_indices(0), &[0]);
    }

    #[test]
    fn single_tile_matches_global_sort_front_to_back() {
        let particles = [
            TiledParticle::new(1.0, 1.0, 90.0),
            TiledParticle::new(2.0, 2.0, 10.0),
            TiledParticle::new(3.0, 3.0, 50.0),
            TiledParticle::new(4.0, 4.0, 30.0),
        ];
        let result = tiled_depth_sort(single_tile(false), &particles);
        assert_eq!(result.order, global_order(&particles, false));
        // Front-to-back: nearest (depth 10) first.
        assert_eq!(result.order, vec![1, 3, 2, 0]);
    }

    #[test]
    fn single_tile_matches_global_sort_back_to_front() {
        let particles = [
            TiledParticle::new(1.0, 1.0, 90.0),
            TiledParticle::new(2.0, 2.0, 10.0),
            TiledParticle::new(3.0, 3.0, 50.0),
            TiledParticle::new(4.0, 4.0, 30.0),
        ];
        let result = tiled_depth_sort(single_tile(true), &particles);
        assert_eq!(result.order, global_order(&particles, true));
        // Back-to-front: farthest (depth 90) first.
        assert_eq!(result.order, vec![0, 2, 3, 1]);
    }

    #[test]
    fn multiple_tiles_each_locally_sorted() {
        // 2x1 column grid over x in [0, 10]: x < 5 -> column 0, x >= 5 -> 1.
        let params = TileGridParams {
            tiles_x: 2,
            tiles_y: 1,
            depth_slices: 1,
            ..single_tile(false)
        };
        let particles = [
            TiledParticle::new(1.0, 0.0, 80.0), // tile 0
            TiledParticle::new(6.0, 0.0, 20.0), // tile 1
            TiledParticle::new(2.0, 0.0, 30.0), // tile 0
            TiledParticle::new(7.0, 0.0, 70.0), // tile 1
            TiledParticle::new(3.0, 0.0, 50.0), // tile 0
        ];
        let result = tiled_depth_sort(params, &particles);
        assert_eq!(result.tiles.len(), 2);
        assert_eq!(result.tiles[0], TileRange { start: 0, count: 3 });
        assert_eq!(result.tiles[1], TileRange { start: 3, count: 2 });
        // Column 0 front-to-back: depth 30 (idx 2), 50 (idx 4), 80 (idx 0).
        assert_eq!(result.tile_indices(0), &[2, 4, 0]);
        // Column 1 front-to-back: depth 20 (idx 1), 70 (idx 3).
        assert_eq!(result.tile_indices(1), &[1, 3]);
    }

    #[test]
    fn equal_depths_break_ties_by_index() {
        // All equal depth and same tile: order must be ascending input index.
        let particles = [
            TiledParticle::new(5.0, 5.0, 44.0),
            TiledParticle::new(5.0, 5.0, 44.0),
            TiledParticle::new(5.0, 5.0, 44.0),
            TiledParticle::new(5.0, 5.0, 44.0),
        ];
        let result = tiled_depth_sort(single_tile(true), &particles);
        assert_eq!(result.order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn deterministic_bitwise_across_runs() {
        let params = TileGridParams {
            tiles_x: 3,
            tiles_y: 3,
            depth_slices: 2,
            ..single_tile(true)
        };
        let particles = [
            TiledParticle::new(0.5, 9.5, 12.0),
            TiledParticle::new(9.9, 0.1, 88.0),
            TiledParticle::new(4.9, 5.1, 50.0),
            TiledParticle::new(5.0, 5.0, 50.0),
            TiledParticle::new(2.0, 7.0, 50.0),
            TiledParticle::new(8.0, 3.0, 33.0),
        ];
        let a = tiled_depth_sort(params, &particles);
        let b = tiled_depth_sort(params, &particles);
        assert_eq!(a, b);
        // Bit-exact check on the derived keys as well (no float compare).
        let keys_a: Vec<u32> = a
            .order
            .iter()
            .map(|&i| {
                u32::from(sort_key(
                    particles[i as usize].depth,
                    NEAR,
                    FAR,
                    params.back_to_front,
                ))
            })
            .collect();
        let keys_b: Vec<u32> = b
            .order
            .iter()
            .map(|&i| {
                u32::from(sort_key(
                    particles[i as usize].depth,
                    NEAR,
                    FAR,
                    params.back_to_front,
                ))
            })
            .collect();
        assert_eq!(keys_a, keys_b);
        // Within every tile, keys are non-decreasing.
        for range in &a.tiles {
            let span = &a.order[range.start as usize..range.end() as usize];
            for pair in span.windows(2) {
                let k0 = sort_key(particles[pair[0] as usize].depth, NEAR, FAR, true);
                let k1 = sort_key(particles[pair[1] as usize].depth, NEAR, FAR, true);
                assert!(k0 <= k1);
            }
        }
    }

    #[test]
    fn tile_boundaries_assign_correctly() {
        let params = TileGridParams {
            tiles_x: 2,
            tiles_y: 2,
            depth_slices: 2,
            ..single_tile(false)
        };
        // Exactly on the lower edge -> bucket 0 on each axis -> tile 0.
        let low = tile_of(params, TiledParticle::new(0.0, 0.0, 0.0));
        assert_eq!(
            low,
            TileCoord {
                x: 0,
                y: 0,
                z: 0,
                index: 0
            }
        );
        // At/above the upper edge -> last bucket on each axis.
        let high = tile_of(params, TiledParticle::new(10.0, 10.0, 100.0));
        let nx = params.effective_tiles_x();
        let ny = params.effective_tiles_y();
        let expected = (ny + 1) * nx + 1;
        assert_eq!(
            high,
            TileCoord {
                x: 1,
                y: 1,
                z: 1,
                index: expected
            }
        );
        // Just below the midpoint stays in the lower column; at the midpoint
        // crosses into the upper column.
        assert_eq!(tile_of(params, TiledParticle::new(4.999, 0.0, 0.0)).x, 0);
        assert_eq!(tile_of(params, TiledParticle::new(5.0, 0.0, 0.0)).x, 1);
    }

    #[test]
    fn degenerate_rect_and_nan_fall_into_tile_zero() {
        let params = TileGridParams {
            rect: ScreenRect::new(5.0, 5.0, 5.0, 5.0), // zero-area
            tiles_x: 4,
            tiles_y: 4,
            depth_slices: 1,
            near: NEAR,
            far: FAR,
            back_to_front: false,
        };
        assert_eq!(tile_of(params, TiledParticle::new(3.0, 7.0, 10.0)).index, 0);
        // A NaN coordinate is routed to bucket 0 rather than panicking.
        let nan = tile_of(params, TiledParticle::new(f32::NAN, 1.0, 10.0));
        assert_eq!(nan.index, 0);
    }

    #[test]
    fn out_of_range_tile_query_is_empty() {
        let particles = [TiledParticle::new(5.0, 5.0, 10.0)];
        let result = tiled_depth_sort(single_tile(false), &particles);
        assert_eq!(result.tile_indices(99), &[] as &[u32]);
    }

    #[test]
    fn order_is_a_permutation_of_all_indices() {
        let params = TileGridParams {
            tiles_x: 4,
            tiles_y: 3,
            depth_slices: 2,
            ..single_tile(true)
        };
        let particles: Vec<TiledParticle> = (0..37)
            .map(|i| {
                let f = i as f32;
                TiledParticle::new((f * 0.37) % 10.0, (f * 0.71) % 10.0, (f * 2.5) % 100.0)
            })
            .collect();
        let result = tiled_depth_sort(params, &particles);
        let mut seen = result.order.clone();
        seen.sort_unstable();
        let expected: Vec<u32> = (0..particles.len() as u32).collect();
        assert_eq!(seen, expected);
        // Ranges tile the whole output with no gaps or overlaps.
        let total: u32 = result.tiles.iter().map(|r| r.count).sum();
        assert_eq!(total as usize, particles.len());
        for pair in result.tiles.windows(2) {
            assert_eq!(pair[0].end(), pair[1].start);
        }
    }
}
