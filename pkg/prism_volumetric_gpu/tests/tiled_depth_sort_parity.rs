//! Real-device parity for the per-particle `tile`-assignment twin:
//! [`GpuTileOf`](prism_volumetric_gpu::tiled_depth_sort::GpuTileOf) must
//! reproduce the `CPU` golden
//! [`tile_of`](prism_render_architecture::particle::tiled_depth_sort::tile_of)
//! across an empty batch, a fully degenerate single-`tile` grid, a zero-area
//! rect, coordinates at/below the minimum, coordinates at/above the maximum,
//! clear mid-bucket hits on every axis, `NaN` screen and depth coordinates, and
//! a large pseudo-random batch compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every output is a discrete bucket index — there is no continuous value to
//! compare — so the comparison is an *exact* `==` on all four output words
//! (`x`, `y`, `z`, `index`). The only place `CPU` and `GPU` could legally
//! disagree is a query sitting on a bucket boundary, where a `ULP`-scale
//! perturbation of `t * count` can flip the truncation across an integer; the
//! named fixtures are placed clear of those boundaries and the random batch
//! rejection-samples away from them, so an exact match is required
//! unconditionally.
//!
//! Provenance: 孪生自本仓
//! `prism_render_architecture::particle::tiled_depth_sort`；standard
//! screen-space `tile` bucketing; no third-party engine source or derived code.

use prism_render_architecture::particle::tiled_depth_sort::{
    tile_of, ScreenRect, TileCoord, TileGridParams, TiledParticle,
};
use prism_volumetric_gpu::tiled_depth_sort::{GpuTileOf, TileOfQuery, TileOfResult};
use prism_volumetric_gpu::GpuContext;

/// Builds a grid with the given rect, axis counts and depth range, always
/// front-to-back (the ordering flag is irrelevant to `tile_of`).
fn grid(
    rect: ScreenRect,
    tiles_x: u32,
    tiles_y: u32,
    depth_slices: u32,
    near: f32,
    far: f32,
) -> TileGridParams {
    TileGridParams {
        rect,
        tiles_x,
        tiles_y,
        depth_slices,
        near,
        far,
        back_to_front: false,
    }
}

/// Pairs a grid and a particle into one query.
fn query(params: TileGridParams, particle: TiledParticle) -> TileOfQuery {
    TileOfQuery { params, particle }
}

/// Asserts the `GPU` result equals the `CPU` golden [`TileCoord`] field for
/// field.
fn same(lane: usize, got: TileOfResult, want: TileCoord) {
    assert_eq!(got.x, want.x, "lane {lane}: x gpu {} cpu {}", got.x, want.x);
    assert_eq!(got.y, want.y, "lane {lane}: y gpu {} cpu {}", got.y, want.y);
    assert_eq!(got.z, want.z, "lane {lane}: z gpu {} cpu {}", got.z, want.z);
    assert_eq!(
        got.index, want.index,
        "lane {lane}: index gpu {} cpu {}",
        got.index, want.index
    );
}

/// Runs the dispatch and asserts strict lane-for-lane parity against the `CPU`
/// golden `tile_of`. Returns the `GPU` verdicts for extra per-test assertions.
fn check(ctx: &GpuContext, gpu: &GpuTileOf, queries: &[TileOfQuery]) -> Vec<TileOfResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (&g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = tile_of(q.params, q.particle);
        same(lane, g, want);
    }
    got
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Returns whether `coord` on the axis `[lo, hi]` with `count` buckets sits
/// clear of every bucket boundary (both the `0`/`1` edges and the interior
/// integer cuts of `t * count`). Fixtures and random samples pass only when
/// this holds on all three axes, so a legal `ULP` perturbation cannot flip the
/// truncation.
fn clear_of_boundaries(coord: f32, lo: f32, hi: f32, count: u32) -> bool {
    if count <= 1 {
        return true;
    }
    let span = hi - lo;
    let t = (coord - lo) / span;
    // Keep away from the lower and upper edges so the `t <= 0` / `t >= 1`
    // guards are unambiguous.
    if t <= 0.05 || t >= 0.95 {
        return false;
    }
    // Keep away from every interior integer cut.
    let scaled = t * count as f32;
    let frac = scaled - scaled.floor();
    (0.1..=0.9).contains(&frac)
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileOf::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn single_tile_grid_collapses_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileOf::new(&ctx);
    // All three axis counts are one, so every particle maps to tile 0 whatever
    // its coordinates (exercises the `count <= 1` guard on each axis).
    let params = grid(ScreenRect::new(0.0, 0.0, 10.0, 10.0), 1, 1, 1, 0.0, 100.0);
    let queries = [
        query(params, TiledParticle::new(3.3, 7.7, 42.0)),
        query(params, TiledParticle::new(-5.0, 50.0, -1.0)),
        query(params, TiledParticle::new(9.99, 0.01, 99.0)),
    ];
    let got = check(&ctx, &gpu, &queries);
    for g in got {
        assert_eq!(
            g,
            TileOfResult {
                x: 0,
                y: 0,
                z: 0,
                index: 0
            }
        );
    }
}

#[test]
fn zero_count_axis_is_clamped_to_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileOf::new(&ctx);
    // A zero on an axis is clamped to one bucket on both paths; the x axis still
    // splits normally. Place x clear of its single interior boundary.
    let params = grid(ScreenRect::new(0.0, 0.0, 10.0, 10.0), 2, 0, 0, 0.0, 100.0);
    let q = query(params, TiledParticle::new(7.3, 4.0, 40.0));
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].y, 0, "clamped y axis is a single bucket");
    assert_eq!(got[0].z, 0, "clamped z axis is a single bucket");
    assert_eq!(
        got[0].x, 1,
        "x = 7.3 of [0,10] with 2 tiles is the upper half"
    );
}

#[test]
fn degenerate_rect_falls_into_bucket_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileOf::new(&ctx);
    // Zero-area rect (min == max) and an inverted depth range both have a
    // non-positive span, so x/y/z all collapse to bucket 0 (the `span <= 0`
    // guard).
    let params = grid(ScreenRect::new(5.0, 5.0, 5.0, 5.0), 4, 4, 4, 100.0, 100.0);
    let q = query(params, TiledParticle::new(3.0, 7.0, 50.0));
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(
        got[0],
        TileOfResult {
            x: 0,
            y: 0,
            z: 0,
            index: 0
        }
    );

    // Inverted depth range (near > far) is also a non-positive span on z.
    let inv = grid(ScreenRect::new(0.0, 0.0, 10.0, 10.0), 3, 3, 3, 90.0, 10.0);
    let q2 = query(inv, TiledParticle::new(5.3, 5.3, 50.0));
    let got2 = check(&ctx, &gpu, &[q2]);
    assert_eq!(got2[0].z, 0, "inverted near/far collapses the depth axis");
}

#[test]
fn coord_at_or_below_min_is_bucket_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileOf::new(&ctx);
    // Coordinates at or below the minimum give `t <= 0` -> bucket 0 on every
    // axis.
    let params = grid(ScreenRect::new(0.0, 0.0, 10.0, 10.0), 4, 4, 4, 0.0, 100.0);
    let queries = [
        query(params, TiledParticle::new(0.0, 0.0, 0.0)),
        query(params, TiledParticle::new(-3.0, -8.0, -25.0)),
    ];
    let got = check(&ctx, &gpu, &queries);
    for g in got {
        assert_eq!(
            g,
            TileOfResult {
                x: 0,
                y: 0,
                z: 0,
                index: 0
            }
        );
    }
}

#[test]
fn coord_at_or_above_max_is_last_bucket() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileOf::new(&ctx);
    // Coordinates at or above the maximum give `t >= 1` -> the last bucket on
    // every axis.
    let params = grid(ScreenRect::new(0.0, 0.0, 10.0, 10.0), 4, 5, 6, 0.0, 100.0);
    let queries = [
        query(params, TiledParticle::new(10.0, 10.0, 100.0)),
        query(params, TiledParticle::new(25.0, 40.0, 500.0)),
    ];
    let got = check(&ctx, &gpu, &queries);
    for g in got {
        assert_eq!(g.x, 3, "last x bucket");
        assert_eq!(g.y, 4, "last y bucket");
        assert_eq!(g.z, 5, "last z bucket");
        assert_eq!(g.index, (5 * 5 + 4) * 4 + 3);
    }
}

#[test]
fn mid_bucket_hits_on_every_axis() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileOf::new(&ctx);
    // Interior hits placed clear of the bucket cuts: with 10 tiles on [0,100],
    // a coordinate of 35 maps to t = 0.35, bucket 3 (frac 0.5 from the cut).
    let params = grid(
        ScreenRect::new(0.0, 0.0, 100.0, 100.0),
        10,
        10,
        10,
        0.0,
        100.0,
    );
    let cases = [
        (TiledParticle::new(35.0, 35.0, 35.0), 3u32, 3u32, 3u32),
        (TiledParticle::new(5.0, 95.0, 65.0), 0, 9, 6),
        (TiledParticle::new(75.0, 45.0, 15.0), 7, 4, 1),
    ];
    // Each coordinate maps to the exact midpoint of its bucket (`t * count`
    // has fractional part 0.5), the farthest possible point from an integer
    // cut, so the truncation is unambiguous on both paths.
    for (particle, ex, ey, ez) in cases {
        let got = check(&ctx, &gpu, &[query(params, particle)]);
        assert_eq!(got[0].x, ex);
        assert_eq!(got[0].y, ey);
        assert_eq!(got[0].z, ez);
        assert_eq!(got[0].index, (ez * 10 + ey) * 10 + ex);
    }
}

#[test]
fn nan_coordinates_fall_into_bucket_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileOf::new(&ctx);
    // `f32::NAN` is a constant, not a transcendental call. A NaN coordinate
    // yields a NaN `t`, which the bit-pattern guard routes to bucket 0 on that
    // axis, matching the reference `is_nan` guard. Other axes bucket normally.
    let params = grid(ScreenRect::new(0.0, 0.0, 100.0, 100.0), 8, 8, 8, 0.0, 100.0);
    let queries = [
        query(params, TiledParticle::new(f32::NAN, 55.0, 55.0)),
        query(params, TiledParticle::new(55.0, f32::NAN, 55.0)),
        query(params, TiledParticle::new(55.0, 55.0, f32::NAN)),
        query(params, TiledParticle::new(f32::NAN, f32::NAN, f32::NAN)),
    ];
    let got = check(&ctx, &gpu, &queries);
    assert_eq!(got[0].x, 0, "NaN screen_x -> x bucket 0");
    assert_eq!(got[1].y, 0, "NaN screen_y -> y bucket 0");
    assert_eq!(got[2].z, 0, "NaN depth -> z bucket 0");
    assert_eq!(
        got[3],
        TileOfResult {
            x: 0,
            y: 0,
            z: 0,
            index: 0
        }
    );
}

#[test]
fn random_batch_matches_lane_for_lane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileOf::new(&ctx);

    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut queries: Vec<TileOfQuery> = Vec::new();
    let mut saw_interior = false;

    while queries.len() < 400 {
        // Random but modest grid dimensions on each axis.
        let nx = 1 + (lcg(&mut state) * 12.0) as u32;
        let ny = 1 + (lcg(&mut state) * 12.0) as u32;
        let nz = 1 + (lcg(&mut state) * 12.0) as u32;
        let near = -40.0 + lcg(&mut state) * 20.0;
        let far = near + 20.0 + lcg(&mut state) * 160.0;
        let rect = ScreenRect::new(-20.0, -15.0, 30.0, 45.0);
        let params = grid(rect, nx, ny, nz, near, far);

        // Sample positions spanning a bit beyond the rect / depth range so some
        // land in the edge buckets, then reject any that sit on a boundary.
        let screen_x = rect.min_x - 10.0 + lcg(&mut state) * (rect.max_x - rect.min_x + 20.0);
        let screen_y = rect.min_y - 10.0 + lcg(&mut state) * (rect.max_y - rect.min_y + 20.0);
        let depth = near - 10.0 + lcg(&mut state) * (far - near + 20.0);
        let particle = TiledParticle::new(screen_x, screen_y, depth);

        let x_ok = clear_of_boundaries(screen_x, rect.min_x, rect.max_x, nx.max(1));
        let y_ok = clear_of_boundaries(screen_y, rect.min_y, rect.max_y, ny.max(1));
        let z_ok = clear_of_boundaries(depth, near, far, nz.max(1));
        // Edge buckets (coord outside the range) are also unambiguous.
        let x_edge = screen_x <= rect.min_x || screen_x >= rect.max_x;
        let y_edge = screen_y <= rect.min_y || screen_y >= rect.max_y;
        let z_edge = depth <= near || depth >= far;
        if !(x_ok || x_edge) || !(y_ok || y_edge) || !(z_ok || z_edge) {
            continue;
        }
        // Track that at least some interior (non-edge) buckets are exercised.
        if x_ok && y_ok && z_ok {
            saw_interior = true;
        }
        queries.push(query(params, particle));
    }

    let got = gpu.eval(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (&g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = tile_of(q.params, q.particle);
        same(lane, g, want);
    }
    assert!(
        saw_interior,
        "random batch should exercise interior buckets, not only edges"
    );
}
