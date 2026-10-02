//! Real-device parity for the clustered-forward light-sampling twin:
//! [`GpuLightClustered`](prism_volumetric_gpu::light_clustered::GpuLightClustered)
//! must reproduce the `CPU` golden
//! [`light_clustered`](prism_render_architecture::particle::light_clustered)
//! across every numeric term it exposes: the unit-interval clamp
//! ([`clamp01`](prism_render_architecture::particle::light_clustered::clamp01)),
//! the Hermite blend
//! ([`smoothstep`](prism_render_architecture::particle::light_clustered::smoothstep)),
//! the windowed inverse-square distance falloff
//! ([`distance_attenuation`](prism_render_architecture::particle::light_clustered::distance_attenuation)),
//! the cone falloff
//! ([`spot_attenuation`](prism_render_architecture::particle::light_clustered::spot_attenuation)),
//! the punctual-light algebra
//! ([`PunctualLight::is_active`](prism_render_architecture::particle::light_clustered::PunctualLight::is_active),
//! [`PunctualLight::range_squared`](prism_render_architecture::particle::light_clustered::PunctualLight::range_squared)),
//! the froxel lookups
//! ([`ClusterGrid::depth_slice`](prism_render_architecture::particle::light_clustered::ClusterGrid::depth_slice),
//! [`ClusterGrid::cluster_coord`](prism_render_architecture::particle::light_clustered::ClusterGrid::cluster_coord),
//! [`ClusterGrid::linear_index`](prism_render_architecture::particle::light_clustered::ClusterGrid::linear_index),
//! [`ClusterGrid::cluster_index`](prism_render_architecture::particle::light_clustered::ClusterGrid::cluster_index)),
//! and the single-light diffuse response
//! ([`shade_point_light`](prism_render_architecture::particle::light_clustered::shade_point_light)).
//!
//! The fixtures exercise one dedicated query per term plus a randomized mixed
//! batch compared element for element. Every scalar is an interior value held
//! well away from its guard branch: ranges are comfortably positive, light and
//! surface are non-coincident with the squared distance clearly inside the
//! squared range, normals point at the light so `n_dot_l` is well above the
//! floor, spot cosines are separated so the Hermite window is non-degenerate,
//! and the view-space probes sit strictly inside the frustum (or decisively
//! outside it for the rejection cases) so no fixture lands on a classification
//! tie.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each continuous term is a polynomial or rational function of its inputs with
//! at most one `sqrt` (the robust normalize inside the shade routine), so `CPU`
//! and `GPU` evaluate the same closed form in the same order. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`) on every continuous `f32` lane; the boolean active flag,
//! the integer slice / tile / cluster indices, and the valid / out-of-frustum
//! flags are compared exactly.
//!
//! # Conditioning
//!
//! Every fixture is kept clear of the guard cracks: ranges are `>= 1` so the
//! falloff never floors to zero on the range test, light / surface separations
//! stay well inside the influence sphere so the distance window is strictly
//! positive, normals face the light so `n_dot_l` is comfortably above `EPS`,
//! spot cosines are separated by a wide margin so the Hermite blend is a true
//! penumbra rather than a hard step, depth probes sit strictly between the near
//! and far planes, and valid screen probes map to normalized coordinates well
//! inside `[-1, 1]` while the rejection probes clear the frustum decisively.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::light_clustered`；
//! 无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::light_clustered::{
    clamp01, distance_attenuation, shade_point_light, smoothstep, spot_attenuation, ClusterCoord,
    ClusterGrid, PunctualLight,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::light_clustered::{
    GpuLightClustered, LightClusteredQuery, LightClusteredResult, MAX_SLICE_BOUNDARIES,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
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

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// Converts a packed component triple into a [`Vec3`].
fn v3(c: [f32; 3]) -> Vec3 {
    Vec3::new(c[0], c[1], c[2])
}

/// A healthy linear froxel grid whose boundaries, tile counts and slopes all sit
/// comfortably away from the constructor's clamps, so the budgeted boundary
/// array round-trips unchanged.
fn healthy_grid() -> ClusterGrid {
    ClusterGrid::linear(16, 12, 16, 0.1, 100.0, 1.0, 0.75)
}

/// Budgets `grid`'s monotonic depth boundaries into the fixed-length device slot
/// and reports how many edges are live.
fn boundaries_of(grid: &ClusterGrid) -> ([f32; MAX_SLICE_BOUNDARIES], u32) {
    let edges = grid.slice_boundaries();
    assert!(
        edges.len() <= MAX_SLICE_BOUNDARIES,
        "grid boundary count {} exceeds the budget {MAX_SLICE_BOUNDARIES}",
        edges.len()
    );
    let mut out = [0.0_f32; MAX_SLICE_BOUNDARIES];
    out[..edges.len()].copy_from_slice(edges);
    (out, edges.len() as u32)
}

/// Recomputes the expected [`LightClusteredResult`] by calling the `CPU` golden
/// directly for `query`. The froxel lookups rebuild the reference grid from the
/// very boundary array the query carries, so the comparison is against exactly
/// the budgeted edges the device sees.
fn golden_result(query: &LightClusteredQuery) -> LightClusteredResult {
    match *query {
        LightClusteredQuery::Clamp01 { x } => LightClusteredResult::Clamp01 { value: clamp01(x) },
        LightClusteredQuery::Smoothstep { edge0, edge1, x } => LightClusteredResult::Smoothstep {
            value: smoothstep(edge0, edge1, x),
        },
        LightClusteredQuery::DistanceAttenuation {
            distance_squared,
            range,
        } => LightClusteredResult::DistanceAttenuation {
            value: distance_attenuation(distance_squared, range),
        },
        LightClusteredQuery::SpotAttenuation {
            cos_angle,
            cos_inner,
            cos_outer,
        } => LightClusteredResult::SpotAttenuation {
            value: spot_attenuation(cos_angle, cos_inner, cos_outer),
        },
        LightClusteredQuery::RangeSquared { range } => LightClusteredResult::RangeSquared {
            value: PunctualLight::point(Vec3::ZERO, Vec3::ZERO, range).range_squared(),
        },
        LightClusteredQuery::IsActive { range } => LightClusteredResult::IsActive {
            active: PunctualLight::point(Vec3::ZERO, Vec3::ZERO, range).is_active(),
        },
        LightClusteredQuery::DepthSlice {
            boundaries,
            boundary_count,
            z,
        } => {
            let grid = ClusterGrid::with_boundaries(
                1,
                1,
                1.0,
                1.0,
                &boundaries[..boundary_count as usize],
            );
            LightClusteredResult::DepthSlice {
                slice: grid.depth_slice(z),
            }
        }
        LightClusteredQuery::ClusterCoord {
            tile_count_x,
            tile_count_y,
            slope_x,
            slope_y,
            boundaries,
            boundary_count,
            view_pos,
        } => {
            let grid = ClusterGrid::with_boundaries(
                tile_count_x,
                tile_count_y,
                slope_x,
                slope_y,
                &boundaries[..boundary_count as usize],
            );
            LightClusteredResult::ClusterCoord {
                coord: grid.cluster_coord(view_pos),
            }
        }
        LightClusteredQuery::LinearIndex {
            tile_count_x,
            tile_count_y,
            slice_count,
            coord,
        } => {
            // `linear_index` reads only the tile counts and the slice count; a
            // linear grid with the same slice count reproduces it exactly.
            let grid = ClusterGrid::linear(
                tile_count_x,
                tile_count_y,
                slice_count,
                0.1,
                100.0,
                1.0,
                1.0,
            );
            LightClusteredResult::LinearIndex {
                index: grid.linear_index(coord),
            }
        }
        LightClusteredQuery::ClusterIndex {
            tile_count_x,
            tile_count_y,
            slope_x,
            slope_y,
            boundaries,
            boundary_count,
            view_pos,
        } => {
            let grid = ClusterGrid::with_boundaries(
                tile_count_x,
                tile_count_y,
                slope_x,
                slope_y,
                &boundaries[..boundary_count as usize],
            );
            LightClusteredResult::ClusterIndex {
                index: grid.cluster_index(view_pos),
            }
        }
        LightClusteredQuery::ShadePointLight {
            light,
            surface_pos,
            normal,
        } => LightClusteredResult::ShadePointLight {
            radiance: {
                let r = shade_point_light(light, surface_pos, normal);
                [r.x, r.y, r.z]
            },
        },
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: each lane the
/// variant carries must agree within the parity bound (continuous lanes within
/// tolerance, boolean / index / validity lanes exactly).
fn pin(idx: usize, query: &LightClusteredQuery, got: &LightClusteredResult) {
    let want = golden_result(query);
    match (got, want) {
        (
            LightClusteredResult::Clamp01 { value: g },
            LightClusteredResult::Clamp01 { value: w },
        )
        | (
            LightClusteredResult::Smoothstep { value: g },
            LightClusteredResult::Smoothstep { value: w },
        )
        | (
            LightClusteredResult::DistanceAttenuation { value: g },
            LightClusteredResult::DistanceAttenuation { value: w },
        )
        | (
            LightClusteredResult::SpotAttenuation { value: g },
            LightClusteredResult::SpotAttenuation { value: w },
        )
        | (
            LightClusteredResult::RangeSquared { value: g },
            LightClusteredResult::RangeSquared { value: w },
        ) => {
            assert!(close(*g, w), "query {idx} scalar lane: gpu {g} vs cpu {w}");
        }
        (
            LightClusteredResult::IsActive { active: g },
            LightClusteredResult::IsActive { active: w },
        ) => {
            assert_eq!(*g, w, "query {idx} is_active: gpu {g} vs cpu {w}");
        }
        (
            LightClusteredResult::DepthSlice { slice: g },
            LightClusteredResult::DepthSlice { slice: w },
        ) => {
            assert_eq!(*g, w, "query {idx} depth_slice: gpu {g:?} vs cpu {w:?}");
        }
        (
            LightClusteredResult::ClusterCoord { coord: g },
            LightClusteredResult::ClusterCoord { coord: w },
        ) => {
            assert_eq!(*g, w, "query {idx} cluster_coord: gpu {g:?} vs cpu {w:?}");
        }
        (
            LightClusteredResult::LinearIndex { index: g },
            LightClusteredResult::LinearIndex { index: w },
        )
        | (
            LightClusteredResult::ClusterIndex { index: g },
            LightClusteredResult::ClusterIndex { index: w },
        ) => {
            assert_eq!(*g, w, "query {idx} cluster index: gpu {g:?} vs cpu {w:?}");
        }
        (
            LightClusteredResult::ShadePointLight { radiance: g },
            LightClusteredResult::ShadePointLight { radiance: w },
        ) => {
            for k in 0..3 {
                assert!(
                    close(g[k], w[k]),
                    "query {idx} radiance[{k}]: gpu {} vs cpu {}",
                    g[k],
                    w[k]
                );
            }
        }
        (g, w) => panic!("query {idx} result variant mismatch: gpu {g:?} vs cpu {w:?}"),
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuLightClustered, queries: &[LightClusteredQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClustered::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn clamp01_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClustered::new(&ctx);
    // One interior value and one on each saturated side, all clear of the knees.
    let queries = vec![
        LightClusteredQuery::Clamp01 { x: 0.37 },
        LightClusteredQuery::Clamp01 { x: -0.8 },
        LightClusteredQuery::Clamp01 { x: 1.6 },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn smoothstep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClustered::new(&ctx);
    // Edges separated by a wide margin so the blend is a true Hermite ramp, with
    // the interpolant below, inside and above the window.
    let queries = vec![
        LightClusteredQuery::Smoothstep {
            edge0: 0.2,
            edge1: 0.8,
            x: 0.5,
        },
        LightClusteredQuery::Smoothstep {
            edge0: 0.2,
            edge1: 0.8,
            x: -0.3,
        },
        LightClusteredQuery::Smoothstep {
            edge0: 0.2,
            edge1: 0.8,
            x: 1.4,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn distance_attenuation_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClustered::new(&ctx);
    // Range comfortably positive, squared distance clearly inside the squared
    // range so the window is strictly between zero and its peak.
    let queries = vec![
        LightClusteredQuery::DistanceAttenuation {
            distance_squared: 4.0,
            range: 10.0,
        },
        LightClusteredQuery::DistanceAttenuation {
            distance_squared: 25.0,
            range: 12.0,
        },
        LightClusteredQuery::DistanceAttenuation {
            distance_squared: 1.0,
            range: 6.0,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn spot_attenuation_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClustered::new(&ctx);
    // Inner / outer cosines separated by a wide margin; alignment cosine inside
    // the cone, in the penumbra, and outside it.
    let queries = vec![
        LightClusteredQuery::SpotAttenuation {
            cos_angle: 0.97,
            cos_inner: 0.9,
            cos_outer: 0.7,
        },
        LightClusteredQuery::SpotAttenuation {
            cos_angle: 0.8,
            cos_inner: 0.9,
            cos_outer: 0.7,
        },
        LightClusteredQuery::SpotAttenuation {
            cos_angle: 0.5,
            cos_inner: 0.9,
            cos_outer: 0.7,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn range_squared_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClustered::new(&ctx);
    let queries = vec![
        LightClusteredQuery::RangeSquared { range: 3.0 },
        LightClusteredQuery::RangeSquared { range: 12.5 },
        LightClusteredQuery::RangeSquared { range: 48.0 },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn is_active_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClustered::new(&ctx);
    // One inactive range (zero, far below the `EPS` floor) and clearly positive
    // ranges, so the boolean is unambiguous on both devices.
    let queries = vec![
        LightClusteredQuery::IsActive { range: 0.0 },
        LightClusteredQuery::IsActive { range: 5.0 },
        LightClusteredQuery::IsActive { range: 30.0 },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn depth_slice_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClustered::new(&ctx);
    let grid = healthy_grid();
    let (boundaries, boundary_count) = boundaries_of(&grid);
    // Depths strictly interior to the near / far range plus two that clear the
    // frustum decisively (in front of near, behind far).
    let queries = vec![
        LightClusteredQuery::DepthSlice {
            boundaries,
            boundary_count,
            z: 10.0,
        },
        LightClusteredQuery::DepthSlice {
            boundaries,
            boundary_count,
            z: 55.0,
        },
        LightClusteredQuery::DepthSlice {
            boundaries,
            boundary_count,
            z: 0.02,
        },
        LightClusteredQuery::DepthSlice {
            boundaries,
            boundary_count,
            z: 180.0,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn cluster_coord_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClustered::new(&ctx);
    let grid = healthy_grid();
    let (boundaries, boundary_count) = boundaries_of(&grid);
    // Two probes land well inside the frustum (normalized coordinates near the
    // center and off-center but clear of the edge); two clear the frustum (one
    // in front of the near plane, one off-screen in X).
    let queries = vec![
        LightClusteredQuery::ClusterCoord {
            tile_count_x: 16,
            tile_count_y: 12,
            slope_x: 1.0,
            slope_y: 0.75,
            boundaries,
            boundary_count,
            view_pos: v3([0.0, 0.0, 10.0]),
        },
        LightClusteredQuery::ClusterCoord {
            tile_count_x: 16,
            tile_count_y: 12,
            slope_x: 1.0,
            slope_y: 0.75,
            boundaries,
            boundary_count,
            view_pos: v3([2.0, 1.5, 10.0]),
        },
        LightClusteredQuery::ClusterCoord {
            tile_count_x: 16,
            tile_count_y: 12,
            slope_x: 1.0,
            slope_y: 0.75,
            boundaries,
            boundary_count,
            view_pos: v3([0.0, 0.0, 0.02]),
        },
        LightClusteredQuery::ClusterCoord {
            tile_count_x: 16,
            tile_count_y: 12,
            slope_x: 1.0,
            slope_y: 0.75,
            boundaries,
            boundary_count,
            view_pos: v3([50.0, 0.0, 10.0]),
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn linear_index_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClustered::new(&ctx);
    // In-range coordinates flatten to a valid index; an out-of-range column
    // yields `None`.
    let queries = vec![
        LightClusteredQuery::LinearIndex {
            tile_count_x: 16,
            tile_count_y: 12,
            slice_count: 16,
            coord: ClusterCoord { x: 3, y: 5, z: 7 },
        },
        LightClusteredQuery::LinearIndex {
            tile_count_x: 16,
            tile_count_y: 12,
            slice_count: 16,
            coord: ClusterCoord { x: 15, y: 11, z: 0 },
        },
        LightClusteredQuery::LinearIndex {
            tile_count_x: 16,
            tile_count_y: 12,
            slice_count: 16,
            coord: ClusterCoord { x: 16, y: 2, z: 1 },
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn cluster_index_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClustered::new(&ctx);
    let grid = healthy_grid();
    let (boundaries, boundary_count) = boundaries_of(&grid);
    // One interior probe (valid flattened index) and one off-screen probe
    // (`None`).
    let queries = vec![
        LightClusteredQuery::ClusterIndex {
            tile_count_x: 16,
            tile_count_y: 12,
            slope_x: 1.0,
            slope_y: 0.75,
            boundaries,
            boundary_count,
            view_pos: v3([1.5, -1.0, 25.0]),
        },
        LightClusteredQuery::ClusterIndex {
            tile_count_x: 16,
            tile_count_y: 12,
            slope_x: 1.0,
            slope_y: 0.75,
            boundaries,
            boundary_count,
            view_pos: v3([0.0, 60.0, 25.0]),
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn shade_point_light_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClustered::new(&ctx);
    // A point light with the surface comfortably inside the range and the normal
    // facing the light; a spot light whose cone axis points at the surface so
    // the alignment cosine is clearly inside the inner cone; and an inactive
    // point light that shades to black.
    let point = PunctualLight::point(v3([0.0, 0.0, 0.0]), v3([3.0, 2.0, 1.0]), 10.0);
    let spot = PunctualLight::spot(
        v3([0.0, 0.0, 0.0]),
        v3([0.0, 0.0, 1.0]),
        v3([1.0, 1.0, 1.0]),
        12.0,
        0.9,
        0.6,
    );
    let inactive = PunctualLight::point(v3([0.0, 0.0, 0.0]), v3([5.0, 5.0, 5.0]), 0.0);
    let queries = vec![
        LightClusteredQuery::ShadePointLight {
            light: point,
            surface_pos: v3([0.0, 0.0, 4.0]),
            normal: v3([0.0, 0.0, -1.0]),
        },
        LightClusteredQuery::ShadePointLight {
            light: spot,
            surface_pos: v3([0.0, 0.0, 5.0]),
            normal: v3([0.0, 0.0, -1.0]),
        },
        LightClusteredQuery::ShadePointLight {
            light: inactive,
            surface_pos: v3([0.0, 0.0, 3.0]),
            normal: v3([0.0, 0.0, -1.0]),
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn randomized_mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLightClustered::new(&ctx);
    let grid = healthy_grid();
    let (boundaries, boundary_count) = boundaries_of(&grid);
    let near = grid.near();
    let far = grid.far();
    let mut state: u64 = 0x1337_c0de_face_b00d_u64 ^ 0x9e37_79b9_7f4a_7c15;
    let mut queries = Vec::new();
    for _ in 0..12 {
        queries.push(LightClusteredQuery::Clamp01 {
            x: ranged(&mut state, -1.5, 1.5),
        });
        // Edges separated by a decisive margin so the blend never collapses.
        let edge0 = ranged(&mut state, 0.0, 0.3);
        let edge1 = edge0 + ranged(&mut state, 0.4, 0.6);
        queries.push(LightClusteredQuery::Smoothstep {
            edge0,
            edge1,
            x: ranged(&mut state, -0.2, 1.2),
        });
        // Range comfortably positive; squared distance kept clearly inside the
        // squared range (at most 70% of it).
        let range = ranged(&mut state, 4.0, 20.0);
        let distance_squared = ranged(&mut state, 0.5, 0.7 * range * range);
        queries.push(LightClusteredQuery::DistanceAttenuation {
            distance_squared,
            range,
        });
        // Spot cosines separated by a wide margin; alignment cosine spans the
        // full window without sitting on an edge.
        let cos_outer = ranged(&mut state, 0.55, 0.7);
        let cos_inner = ranged(&mut state, 0.85, 0.95);
        queries.push(LightClusteredQuery::SpotAttenuation {
            cos_angle: ranged(&mut state, -0.4, 1.0),
            cos_inner,
            cos_outer,
        });
        queries.push(LightClusteredQuery::RangeSquared {
            range: ranged(&mut state, 1.0, 50.0),
        });
        queries.push(LightClusteredQuery::IsActive {
            range: ranged(&mut state, 1.0, 50.0),
        });
        // Depth strictly interior to the near / far range.
        queries.push(LightClusteredQuery::DepthSlice {
            boundaries,
            boundary_count,
            z: ranged(&mut state, near + 2.0, far - 2.0),
        });
        // View probe guaranteed inside the frustum: depth interior, normalized
        // coordinates within +-0.8 of center on both axes.
        let z = ranged(&mut state, near + 5.0, far - 5.0);
        let half_w = z * 1.0;
        let half_h = z * 0.75;
        let view_pos = v3([
            ranged(&mut state, -0.8, 0.8) * half_w,
            ranged(&mut state, -0.8, 0.8) * half_h,
            z,
        ]);
        queries.push(LightClusteredQuery::ClusterCoord {
            tile_count_x: 16,
            tile_count_y: 12,
            slope_x: 1.0,
            slope_y: 0.75,
            boundaries,
            boundary_count,
            view_pos,
        });
        queries.push(LightClusteredQuery::ClusterIndex {
            tile_count_x: 16,
            tile_count_y: 12,
            slope_x: 1.0,
            slope_y: 0.75,
            boundaries,
            boundary_count,
            view_pos,
        });
        // Linear index with in-range coordinates.
        queries.push(LightClusteredQuery::LinearIndex {
            tile_count_x: 16,
            tile_count_y: 12,
            slice_count: 16,
            coord: ClusterCoord {
                x: (lcg(&mut state) * 16.0) as u32 % 16,
                y: (lcg(&mut state) * 12.0) as u32 % 12,
                z: (lcg(&mut state) * 16.0) as u32 % 16,
            },
        });
        // Shade a point or spot light with the surface well inside the range and
        // the normal facing the light. The offset length stays clearly below the
        // range so the distance window is strictly positive.
        let range = ranged(&mut state, 8.0, 20.0);
        let offset = [
            ranged(&mut state, 2.0, 4.0),
            ranged(&mut state, -1.0, 1.0),
            ranged(&mut state, -1.0, 1.0),
        ];
        let surface_pos = v3([
            ranged(&mut state, -3.0, 3.0),
            ranged(&mut state, -3.0, 3.0),
            ranged(&mut state, -3.0, 3.0),
        ]);
        let light_pos = v3([
            surface_pos.x + offset[0],
            surface_pos.y + offset[1],
            surface_pos.z + offset[2],
        ]);
        // Normal points from the surface toward the light.
        let normal = v3(offset);
        let color = v3([
            ranged(&mut state, 0.5, 4.0),
            ranged(&mut state, 0.5, 4.0),
            ranged(&mut state, 0.5, 4.0),
        ]);
        let light = if lcg(&mut state) < 0.5 {
            PunctualLight::point(light_pos, color, range)
        } else {
            // Cone axis points from the light toward the surface, so the
            // alignment cosine is `1` and the point is clearly inside the cone.
            PunctualLight::spot(
                light_pos,
                v3([-offset[0], -offset[1], -offset[2]]),
                color,
                range,
                0.85,
                0.6,
            )
        };
        queries.push(LightClusteredQuery::ShadePointLight {
            light,
            surface_pos,
            normal,
        });
    }
    check(&ctx, &gpu, &queries);
}
