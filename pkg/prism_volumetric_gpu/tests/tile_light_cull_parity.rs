//! Real-device parity for the `Forward+` tile-light-cull twin:
//! [`GpuTileLightCull`](prism_volumetric_gpu::tile_light_cull::GpuTileLightCull)
//! must reproduce the `CPU` golden
//! [`tile_light_cull`](prism_render_architecture::particle::tile_light_cull)
//! numeric surface across an empty batch, every per-op fixture, a mixed tagged
//! batch and a large pseudo-random batch compared lane for lane.
//!
//! The twin reproduces the module's pure numeric pieces one thread per query:
//! the [`Vec3`](prism_render_architecture::particle::tile_light_cull::Vec3)
//! primitives `plus`, `minus`, `scale`, `dot`, `cross`, `length` and
//! `normalize_or_zero`;
//! [`SphereLight::is_active`](prism_render_architecture::particle::tile_light_cull::SphereLight::is_active);
//! [`Plane::from_inward_normal`](prism_render_architecture::particle::tile_light_cull::Plane::from_inward_normal)
//! and
//! [`Plane::signed_distance`](prism_render_architecture::particle::tile_light_cull::Plane::signed_distance);
//! [`DepthRange::new`](prism_render_architecture::particle::tile_light_cull::DepthRange::new);
//! [`TileGrid::tile_pixel_rect`](prism_render_architecture::particle::tile_light_cull::TileGrid::tile_pixel_rect);
//! [`TileGrid::tile_frustum`](prism_render_architecture::particle::tile_light_cull::TileGrid::tile_frustum);
//! and
//! [`TileFrustum::intersects_sphere`](prism_render_architecture::particle::tile_light_cull::TileFrustum::intersects_sphere).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! guarded divides, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The comparison therefore allows `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3` on the `f32` lanes and asserts an *exact* match on the
//! discrete pixel-rectangle integers and the `is_active` / sphere-cull flags.
//! The fixtures stay clear of the `normalize_or_zero`, near / far clamp and
//! sphere-cull decision thresholds by rejection sampling so the comparison
//! exercises the live solve.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::tile_light_cull`；
//! 无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::tile_light_cull::{
    DepthRange, Plane, SphereLight, TileGrid, Vec3,
};
use prism_volumetric_gpu::tile_light_cull::{
    GpuTileLightCull, TileLightCullQuery, TileLightCullResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the `f32` lanes. A `GPU` may fuse a multiply-add the
/// scalar reference leaves separate, perturbing the low mantissa bits by a few
/// units in the last place; `1e-4` admits that legal slack while still failing a
/// genuinely wrong port.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL_EPS: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn approx(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= REL_EPS
}

/// Returns whether two vectors agree component-wise within tolerance.
fn approx_vec(a: Vec3, b: Vec3) -> bool {
    approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
}

// ---------------------------------------------------------------------------
// Golden evaluation and comparison.
// ---------------------------------------------------------------------------

/// Evaluates the `CPU` golden for one query, returning the expected
/// [`TileLightCullResult`]. The frustum / rectangle / cull queries build a
/// [`TileGrid`] from the same raw extents the query carries, so the twin and the
/// reference see the identical grid.
fn expected(query: &TileLightCullQuery) -> TileLightCullResult {
    match query {
        TileLightCullQuery::Vec3Plus { a, b } => TileLightCullResult::Vector(a.plus(*b)),
        TileLightCullQuery::Vec3Minus { a, b } => TileLightCullResult::Vector(a.minus(*b)),
        TileLightCullQuery::Vec3Scale { a, factor } => {
            TileLightCullResult::Vector(a.scale(*factor))
        }
        TileLightCullQuery::Vec3Dot { a, b } => TileLightCullResult::Scalar(a.dot(*b)),
        TileLightCullQuery::Vec3Cross { a, b } => TileLightCullResult::Vector(a.cross(*b)),
        TileLightCullQuery::Vec3Length { a } => TileLightCullResult::Scalar(a.length()),
        TileLightCullQuery::Vec3NormalizeOrZero { a } => {
            TileLightCullResult::Vector(a.normalize_or_zero())
        }
        TileLightCullQuery::SphereIsActive { radius } => {
            TileLightCullResult::Bool(SphereLight::new(Vec3::ZERO, *radius).is_active())
        }
        TileLightCullQuery::PlaneFromInwardNormal { normal } => {
            TileLightCullResult::Vector(Plane::from_inward_normal(*normal).normal())
        }
        TileLightCullQuery::PlaneSignedDistance { normal, point } => {
            // The query carries an already-unit inward normal; the signed
            // distance to a point is their dot product.
            TileLightCullResult::Scalar(normal.dot(*point))
        }
        TileLightCullQuery::DepthRangeNew { a, b } => {
            let r = DepthRange::new(*a, *b);
            TileLightCullResult::Range {
                min: r.min,
                max: r.max,
            }
        }
        TileLightCullQuery::TilePixelRect {
            tile_x,
            tile_y,
            tile_size,
            width,
            height,
            ..
        } => {
            let grid = TileGrid::new(*width, *height, *tile_size, 1.0, 1.0);
            let (x0, y0, x1, y1) = grid.tile_pixel_rect(*tile_x, *tile_y);
            TileLightCullResult::PixelRect { x0, y0, x1, y1 }
        }
        TileLightCullQuery::TileFrustum {
            tile_x,
            tile_y,
            tile_size,
            width,
            height,
            slope_x,
            slope_y,
            near,
            far,
            ..
        } => {
            let grid = TileGrid::new(*width, *height, *tile_size, *slope_x, *slope_y);
            let frustum = grid.tile_frustum(*tile_x, *tile_y, *near, *far);
            let planes = frustum.sides();
            TileLightCullResult::Frustum {
                sides: [
                    planes[0].normal(),
                    planes[1].normal(),
                    planes[2].normal(),
                    planes[3].normal(),
                ],
                near: frustum.near(),
                far: frustum.far(),
            }
        }
        TileLightCullQuery::IntersectsSphere {
            tile_x,
            tile_y,
            tile_size,
            width,
            height,
            slope_x,
            slope_y,
            near,
            far,
            center,
            radius,
            ..
        } => {
            let grid = TileGrid::new(*width, *height, *tile_size, *slope_x, *slope_y);
            let frustum = grid.tile_frustum(*tile_x, *tile_y, *near, *far);
            let hit = frustum.intersects_sphere(SphereLight::new(*center, *radius));
            TileLightCullResult::Bool(hit)
        }
    }
}

/// Asserts the `GPU` result matches the `CPU` golden for one lane, applying the
/// tolerance on `f32` lanes and an exact match on discrete integers and flags.
fn compare(lane: usize, query: &TileLightCullQuery, got: &TileLightCullResult) {
    let want = expected(query);
    match (got, &want) {
        (TileLightCullResult::Scalar(g), TileLightCullResult::Scalar(w)) => {
            assert!(approx(*g, *w), "lane {lane}: scalar gpu {g} vs cpu {w}");
        }
        (TileLightCullResult::Vector(g), TileLightCullResult::Vector(w)) => {
            assert!(
                approx_vec(*g, *w),
                "lane {lane}: vector gpu {g:?} vs cpu {w:?}"
            );
        }
        (TileLightCullResult::Bool(g), TileLightCullResult::Bool(w)) => {
            assert_eq!(g, w, "lane {lane}: bool gpu {g} vs cpu {w}");
        }
        (
            TileLightCullResult::Range {
                min: gmin,
                max: gmax,
            },
            TileLightCullResult::Range {
                min: wmin,
                max: wmax,
            },
        ) => {
            assert!(
                approx(*gmin, *wmin) && approx(*gmax, *wmax),
                "lane {lane}: range gpu [{gmin}, {gmax}] vs cpu [{wmin}, {wmax}]"
            );
        }
        (
            TileLightCullResult::PixelRect {
                x0: gx0,
                y0: gy0,
                x1: gx1,
                y1: gy1,
            },
            TileLightCullResult::PixelRect {
                x0: wx0,
                y0: wy0,
                x1: wx1,
                y1: wy1,
            },
        ) => {
            assert_eq!(
                (gx0, gy0, gx1, gy1),
                (wx0, wy0, wx1, wy1),
                "lane {lane}: pixel rect mismatch"
            );
        }
        (
            TileLightCullResult::Frustum {
                sides: gsides,
                near: gnear,
                far: gfar,
            },
            TileLightCullResult::Frustum {
                sides: wsides,
                near: wnear,
                far: wfar,
            },
        ) => {
            let mut i = 0usize;
            while i < 4 {
                assert!(
                    approx_vec(gsides[i], wsides[i]),
                    "lane {lane}: frustum side {i} gpu {:?} vs cpu {:?}",
                    gsides[i],
                    wsides[i]
                );
                i += 1;
            }
            assert!(
                approx(*gnear, *wnear) && approx(*gfar, *wfar),
                "lane {lane}: frustum near/far gpu [{gnear}, {gfar}] vs cpu [{wnear}, {wfar}]"
            );
        }
        (g, w) => panic!("lane {lane}: result kind mismatch gpu {g:?} vs cpu {w:?}"),
    }
}

/// Dispatches a single query and asserts it matches the golden.
fn check_one(ctx: &GpuContext, gpu: &GpuTileLightCull, query: TileLightCullQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&query));
    assert_eq!(got.len(), 1, "one result per query");
    compare(0, &query, &got[0]);
}

/// Builds a tile-frustum query, filling the tile counts from a grid built on the
/// same raw extents so the twin and the golden agree on the grid shape.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the golden tile_frustum tile and camera operands"
)]
fn frustum_query(
    tile_x: u32,
    tile_y: u32,
    tile_size: u32,
    width: u32,
    height: u32,
    slope_x: f32,
    slope_y: f32,
    near: f32,
    far: f32,
) -> TileLightCullQuery {
    let grid = TileGrid::new(width, height, tile_size, slope_x, slope_y);
    TileLightCullQuery::TileFrustum {
        tile_x,
        tile_y,
        tile_size,
        tile_count_x: grid.tile_count_x(),
        tile_count_y: grid.tile_count_y(),
        width,
        height,
        slope_x,
        slope_y,
        near,
        far,
    }
}

/// Builds a sphere-vs-frustum cull query against the same grid shape.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the golden tile_frustum plus the sphere operands"
)]
fn intersect_query(
    tile_x: u32,
    tile_y: u32,
    tile_size: u32,
    width: u32,
    height: u32,
    slope_x: f32,
    slope_y: f32,
    near: f32,
    far: f32,
    center: Vec3,
    radius: f32,
) -> TileLightCullQuery {
    let grid = TileGrid::new(width, height, tile_size, slope_x, slope_y);
    TileLightCullQuery::IntersectsSphere {
        tile_x,
        tile_y,
        tile_size,
        tile_count_x: grid.tile_count_x(),
        tile_count_y: grid.tile_count_y(),
        width,
        height,
        slope_x,
        slope_y,
        near,
        far,
        center,
        radius,
    }
}

/// Builds a tile-pixel-rectangle query against the same grid shape.
fn rect_query(
    tile_x: u32,
    tile_y: u32,
    tile_size: u32,
    width: u32,
    height: u32,
) -> TileLightCullQuery {
    let grid = TileGrid::new(width, height, tile_size, 1.0, 1.0);
    TileLightCullQuery::TilePixelRect {
        tile_x,
        tile_y,
        tile_size,
        tile_count_x: grid.tile_count_x(),
        tile_count_y: grid.tile_count_y(),
        width,
        height,
    }
}

// ---------------------------------------------------------------------------
// Per-op fixtures.
// ---------------------------------------------------------------------------

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn vec3_plus_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        TileLightCullQuery::Vec3Plus {
            a: Vec3::new(1.5, -2.0, 3.25),
            b: Vec3::new(-0.5, 4.0, 1.0),
        },
    );
}

#[test]
fn vec3_minus_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        TileLightCullQuery::Vec3Minus {
            a: Vec3::new(5.0, 1.0, -2.0),
            b: Vec3::new(2.0, -3.0, 0.5),
        },
    );
}

#[test]
fn vec3_scale_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        TileLightCullQuery::Vec3Scale {
            a: Vec3::new(1.0, -4.0, 2.5),
            factor: -1.75,
        },
    );
}

#[test]
fn vec3_dot_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        TileLightCullQuery::Vec3Dot {
            a: Vec3::new(1.0, 2.0, 3.0),
            b: Vec3::new(-2.0, 0.5, 4.0),
        },
    );
}

#[test]
fn vec3_cross_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        TileLightCullQuery::Vec3Cross {
            a: Vec3::new(1.0, 0.0, 0.0),
            b: Vec3::new(0.0, 1.0, 0.0),
        },
    );
}

#[test]
fn vec3_length_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        TileLightCullQuery::Vec3Length {
            a: Vec3::new(3.0, 4.0, 12.0),
        },
    );
}

#[test]
fn vec3_normalize_or_zero_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    // Length well above the guard, so it takes the live divide, not the collapse.
    check_one(
        &ctx,
        &gpu,
        TileLightCullQuery::Vec3NormalizeOrZero {
            a: Vec3::new(2.0, -3.0, 6.0),
        },
    );
}

#[test]
fn vec3_normalize_or_zero_collapses_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    // Shorter than the guard: both sides collapse to the exact zero vector.
    check_one(
        &ctx,
        &gpu,
        TileLightCullQuery::Vec3NormalizeOrZero { a: Vec3::ZERO },
    );
}

#[test]
fn sphere_is_active_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    // Clearly active and clearly inactive, both far from the guard threshold.
    check_one(
        &ctx,
        &gpu,
        TileLightCullQuery::SphereIsActive { radius: 2.5 },
    );
    check_one(
        &ctx,
        &gpu,
        TileLightCullQuery::SphereIsActive { radius: 0.0 },
    );
    check_one(
        &ctx,
        &gpu,
        TileLightCullQuery::SphereIsActive { radius: -3.0 },
    );
}

#[test]
fn plane_from_inward_normal_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    check_one(
        &ctx,
        &gpu,
        TileLightCullQuery::PlaneFromInwardNormal {
            normal: Vec3::new(0.0, 3.0, -4.0),
        },
    );
}

#[test]
fn plane_signed_distance_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    // Pass an already-unit inward normal, exactly what the frustum planes store.
    let normal = Plane::from_inward_normal(Vec3::new(1.0, 1.0, 1.0)).normal();
    check_one(
        &ctx,
        &gpu,
        TileLightCullQuery::PlaneSignedDistance {
            normal,
            point: Vec3::new(2.0, -1.0, 3.0),
        },
    );
}

#[test]
fn depth_range_orders_edges() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    // Already-ordered and reversed edges both exercise the ordering branch.
    check_one(
        &ctx,
        &gpu,
        TileLightCullQuery::DepthRangeNew { a: 5.0, b: 20.0 },
    );
    check_one(
        &ctx,
        &gpu,
        TileLightCullQuery::DepthRangeNew { a: 20.0, b: 5.0 },
    );
}

#[test]
fn tile_pixel_rect_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    // 100x64 with tile 32 -> 4x2 grid: interior tile, a partial right-edge tile
    // and an off-grid tile clamped to the last valid column / row.
    check_one(&ctx, &gpu, rect_query(1, 1, 32, 100, 64));
    check_one(&ctx, &gpu, rect_query(3, 0, 32, 100, 64));
    check_one(&ctx, &gpu, rect_query(9, 9, 32, 100, 64));
}

#[test]
fn tile_frustum_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    // Near >= 1 and far >= near + 1 stay clear of the near/far clamp branch.
    check_one(
        &ctx,
        &gpu,
        frustum_query(1, 2, 32, 128, 128, 1.0, 0.75, 1.0, 100.0),
    );
    check_one(
        &ctx,
        &gpu,
        frustum_query(0, 0, 32, 128, 128, 1.2, 1.2, 2.0, 50.0),
    );
}

#[test]
fn tile_frustum_clamps_near_far() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    // far < near forces the clamp; both sides push far to near + GUARD_EPS.
    check_one(
        &ctx,
        &gpu,
        frustum_query(2, 2, 32, 128, 128, 1.0, 1.0, 10.0, 3.0),
    );
}

#[test]
fn intersects_sphere_inside_survives() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    // A sphere centred on the tile axis, well inside near/far, clearly survives.
    check_one(
        &ctx,
        &gpu,
        intersect_query(
            2,
            2,
            32,
            128,
            128,
            1.0,
            1.0,
            5.0,
            100.0,
            Vec3::new(0.0, 0.0, 40.0),
            2.0,
        ),
    );
}

#[test]
fn intersects_sphere_behind_near_is_culled() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    // A tiny sphere well behind the near plane is clearly culled.
    check_one(
        &ctx,
        &gpu,
        intersect_query(
            2,
            2,
            32,
            128,
            128,
            1.0,
            1.0,
            20.0,
            100.0,
            Vec3::new(0.0, 0.0, 2.0),
            0.5,
        ),
    );
}

#[test]
fn intersects_sphere_far_outside_side_is_culled() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    // A small sphere far to one side of a corner tile sits outside a side plane.
    check_one(
        &ctx,
        &gpu,
        intersect_query(
            0,
            0,
            32,
            128,
            128,
            1.0,
            1.0,
            5.0,
            100.0,
            Vec3::new(80.0, 80.0, 20.0),
            0.5,
        ),
    );
}

// ---------------------------------------------------------------------------
// Mixed batch.
// ---------------------------------------------------------------------------

#[test]
fn mixed_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    let normal = Plane::from_inward_normal(Vec3::new(-1.0, 2.0, 0.5)).normal();
    let queries = vec![
        TileLightCullQuery::Vec3Plus {
            a: Vec3::new(1.0, 2.0, 3.0),
            b: Vec3::new(-1.0, 0.5, 2.0),
        },
        TileLightCullQuery::Vec3Minus {
            a: Vec3::new(4.0, -1.0, 0.0),
            b: Vec3::new(1.0, 1.0, 1.0),
        },
        TileLightCullQuery::Vec3Scale {
            a: Vec3::new(2.0, -2.0, 1.0),
            factor: 3.0,
        },
        TileLightCullQuery::Vec3Dot {
            a: Vec3::new(1.0, 0.0, -1.0),
            b: Vec3::new(2.0, 3.0, 4.0),
        },
        TileLightCullQuery::Vec3Cross {
            a: Vec3::new(0.0, 0.0, 1.0),
            b: Vec3::new(1.0, 0.0, 0.0),
        },
        TileLightCullQuery::Vec3Length {
            a: Vec3::new(1.0, 2.0, 2.0),
        },
        TileLightCullQuery::Vec3NormalizeOrZero {
            a: Vec3::new(-3.0, 0.0, 4.0),
        },
        TileLightCullQuery::SphereIsActive { radius: 1.5 },
        TileLightCullQuery::SphereIsActive { radius: 0.0 },
        TileLightCullQuery::PlaneFromInwardNormal {
            normal: Vec3::new(2.0, 0.0, -1.0),
        },
        TileLightCullQuery::PlaneSignedDistance {
            normal,
            point: Vec3::new(1.0, -2.0, 3.0),
        },
        TileLightCullQuery::DepthRangeNew { a: 12.0, b: 3.0 },
        rect_query(2, 1, 32, 100, 64),
        frustum_query(1, 1, 32, 128, 128, 1.0, 0.75, 1.0, 100.0),
        intersect_query(
            2,
            2,
            32,
            128,
            128,
            1.0,
            1.0,
            5.0,
            100.0,
            Vec3::new(0.0, 0.0, 40.0),
            2.0,
        ),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        compare(lane, q, g);
    }
}

// ---------------------------------------------------------------------------
// Pseudo-random batch.
// ---------------------------------------------------------------------------

/// A tiny host-side `u64` linear congruential generator, so the fixtures use no
/// `f32` transcendental method. The multiplier and increment are the well-known
/// `PCG`/`Knuth` constants.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    /// A float in `[0, 1)` with 24 bits of entropy.
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A float in `[-1, 1)`.
    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    fn vec3(&mut self, scale: f32) -> Vec3 {
        Vec3::new(
            self.signed() * scale,
            self.signed() * scale,
            self.signed() * scale,
        )
    }

    /// A vector whose squared length is above `0.25`, so `normalize_or_zero`
    /// takes the live divide rather than the degenerate collapse.
    fn nonzero_vec3(&mut self, scale: f32) -> Vec3 {
        loop {
            let v = self.vec3(scale);
            if v.dot(v) > 0.25 {
                return v;
            }
        }
    }
}

#[test]
fn random_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTileLightCull::new(&ctx);
    let mut rng = Lcg::new(0x05ee_d101_u64);
    let mut queries: Vec<TileLightCullQuery> = Vec::new();
    for _ in 0..160 {
        let op = rng.next_u32() % 11;
        let query = match op {
            0 => TileLightCullQuery::Vec3Plus {
                a: rng.vec3(5.0),
                b: rng.vec3(5.0),
            },
            1 => TileLightCullQuery::Vec3Minus {
                a: rng.vec3(5.0),
                b: rng.vec3(5.0),
            },
            2 => TileLightCullQuery::Vec3Scale {
                a: rng.vec3(5.0),
                factor: rng.signed() * 4.0,
            },
            3 => TileLightCullQuery::Vec3Dot {
                a: rng.vec3(5.0),
                b: rng.vec3(5.0),
            },
            4 => TileLightCullQuery::Vec3Cross {
                a: rng.vec3(5.0),
                b: rng.vec3(5.0),
            },
            5 => TileLightCullQuery::Vec3Length { a: rng.vec3(5.0) },
            6 => TileLightCullQuery::Vec3NormalizeOrZero {
                a: rng.nonzero_vec3(5.0),
            },
            7 => {
                let normal = Plane::from_inward_normal(rng.nonzero_vec3(3.0)).normal();
                TileLightCullQuery::PlaneSignedDistance {
                    normal,
                    point: rng.vec3(5.0),
                }
            }
            8 => TileLightCullQuery::DepthRangeNew {
                a: rng.unit() * 50.0,
                b: rng.unit() * 50.0,
            },
            9 => {
                // 128x128 tile 32 -> 4x4 grid; any tile in range.
                let tx = rng.next_u32() % 4;
                let ty = rng.next_u32() % 4;
                rect_query(tx, ty, 32, 128, 128)
            }
            _ => {
                // A non-degenerate tile sub-frustum, near >= 1, far >= near + 1.
                let tx = rng.next_u32() % 4;
                let ty = rng.next_u32() % 4;
                let near = 1.0 + rng.unit() * 4.0;
                let far = near + 10.0 + rng.unit() * 80.0;
                let slope_x = 0.6 + rng.unit() * 1.2;
                let slope_y = 0.6 + rng.unit() * 1.2;
                frustum_query(tx, ty, 32, 128, 128, slope_x, slope_y, near, far)
            }
        };
        queries.push(query);
    }
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        compare(lane, q, g);
    }
}
