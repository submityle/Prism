//! Real-device parity for the race-free per-vertex cloth aero-gather twin:
//! [`GpuClothAeroGather`](prism_volumetric_gpu::cloth_aero_gather::GpuClothAeroGather)
//! must reproduce the velocity field of the `CPU` golden
//! [`accumulate_aero_gather`](prism_render_architecture::cloth::aero_gather::accumulate_aero_gather)
//! — the `Jacobi` gather over the ascending `CSR` incidence, the shared
//! per-face `triangle_aero_force`, and the integer-hash turbulence jitter —
//! across hand-built fixtures and a randomized sweep, compared vertex-for-vertex.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The reference function is public, so it drives the oracle directly: each
//! scene's particle columns, mesh triangles, pre-resolved adjacency, wind and
//! aero coefficients feed
//! [`accumulate_aero_gather`](prism_render_architecture::cloth::aero_gather::accumulate_aero_gather)
//! on a cloned particle set, and the written-back velocities are compared to the
//! `GPU` result. A passing `GPU == oracle` run is direct evidence the kernel
//! computes the same velocity field, not merely that the shader compiles.
//!
//! # Parity criterion
//!
//! Both sides evaluate the identical arithmetic (`sqrt`, `dot`, `cross`,
//! `+ - * /`) and the identical `u32` wrapping hash, so the velocity field
//! matches to within floating-point tolerance (`abs <= 1e-4` or `rel <= 1e-3`,
//! with a `REL_FLOOR` of `1e-6`). The branch decisions — degenerate triangle,
//! pinned vertex, linear versus quadratic pressure, zero turbulence — are pure
//! ordered comparisons, so the randomized sweep reject-samples positions and
//! snaps the coefficients clear of those crossings.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::aero_gather`；无第三方引擎源码或衍生代码。

use prism_render_architecture::cloth::{
    aero_gather::{accumulate_aero_gather, VertexTriangleAdjacency},
    wind::{AeroParams, WindField},
    ClothParticle, Vec3,
};
use prism_volumetric_gpu::cloth_aero_gather::{
    ClothAeroGatherQuery, ClothAeroGatherResult, GpuClothAeroGather,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute closeness floor for the parity comparison.
const ABS: f32 = 1e-4;
/// Relative closeness bound for the parity comparison.
const REL: f32 = 1e-3;
/// Smallest denominator used in the relative comparison, guarding `0 == 0`.
const REL_FLOOR: f32 = 1e-6;

/// Absolute-or-relative closeness: `true` when `a` and `b` agree to within the
/// shared tolerance.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS || rel <= REL
}

/// One parity scene: the particle columns, the mesh triangles, and the raw wind
/// and aero inputs. The host query constructor and the golden sanitize these
/// identically, so the raw values drive both paths.
struct Scene {
    /// Cloth particles (position, velocity, inverse mass).
    particles: Vec<ClothParticle>,
    /// Mesh faces as vertex-index triples.
    triangles: Vec<[u32; 3]>,
    /// Ambient wind velocity.
    wind_velocity: [f32; 3],
    /// Turbulence strength.
    turbulence: f32,
    /// Drag coefficient.
    drag: f32,
    /// Lift coefficient.
    lift: f32,
    /// Fluid density (`> 0` selects the quadratic model).
    air_density: f32,
    /// Integration step.
    dt: f32,
}

/// Builds a movable particle at `pos` with velocity `vel` and inverse mass `im`.
fn free(pos: [f32; 3], vel: [f32; 3], im: f32) -> ClothParticle {
    let mut p = ClothParticle::new(Vec3::new(pos[0], pos[1], pos[2]), im);
    p.velocity = Vec3::new(vel[0], vel[1], vel[2]);
    p
}

/// The flat position column of `particles`.
fn positions_of(particles: &[ClothParticle]) -> Vec<[f32; 3]> {
    particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z])
        .collect()
}

/// The flat velocity column of `particles`.
fn velocities_of(particles: &[ClothParticle]) -> Vec<[f32; 3]> {
    particles
        .iter()
        .map(|p| [p.velocity.x, p.velocity.y, p.velocity.z])
        .collect()
}

/// Packs a scene into a device query, resolving the `CSR` adjacency host-side.
fn make_query(scene: &Scene) -> ClothAeroGatherQuery {
    let count = scene.particles.len();
    let positions = positions_of(&scene.particles);
    let velocities = velocities_of(&scene.particles);
    let inverse_masses: Vec<f32> = scene.particles.iter().map(|p| p.inverse_mass).collect();
    let adj = VertexTriangleAdjacency::build(count, &scene.triangles);
    ClothAeroGatherQuery::new(
        &positions,
        &velocities,
        &inverse_masses,
        &scene.triangles,
        adj.offsets(),
        adj.entries(),
        adj.vertex_count() as u32,
        scene.wind_velocity,
        scene.turbulence,
        scene.drag,
        scene.lift,
        scene.air_density,
        scene.dt,
    )
}

/// Runs the golden on a clone of the scene's particles and returns the
/// written-back velocity column.
fn oracle(scene: &Scene) -> Vec<[f32; 3]> {
    let mut particles = scene.particles.clone();
    let adj = VertexTriangleAdjacency::build(particles.len(), &scene.triangles);
    let wind = WindField::new(
        Vec3::new(
            scene.wind_velocity[0],
            scene.wind_velocity[1],
            scene.wind_velocity[2],
        ),
        scene.turbulence,
    );
    let aero = AeroParams::new(scene.drag, scene.lift).with_air_density(scene.air_density);
    accumulate_aero_gather(
        &mut particles,
        &scene.triangles,
        &adj,
        &wind,
        aero,
        scene.dt,
    );
    velocities_of(&particles)
}

/// Dispatches every scene on the `GPU` and pins each updated velocity against
/// the oracle, vertex-for-vertex and axis-for-axis.
fn check(ctx: &GpuContext, gpu: &GpuClothAeroGather, scenes: &[Scene]) {
    let queries: Vec<ClothAeroGatherQuery> = scenes.iter().map(make_query).collect();
    let got: Vec<ClothAeroGatherResult> = gpu.evaluate(ctx, &queries);
    assert_eq!(
        got.len(),
        scenes.len(),
        "result count must match the input count"
    );
    for (idx, (scene, result)) in scenes.iter().zip(got.iter()).enumerate() {
        let want = oracle(scene);
        for (v, expected) in want.iter().enumerate() {
            let g = result.velocity(v);
            for (axis, (&gv, &wv)) in g.iter().zip(expected.iter()).enumerate() {
                assert!(
                    close(gv, wv),
                    "scene {idx} vertex {v} axis {axis}: gpu {gv} vs cpu {wv}"
                );
            }
        }
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Maps a raw `u32` into `[lo, hi)` as an `f32`, using only integer and
/// floating-point arithmetic (no transcendental), for the random sweep.
fn uniform(raw: u32, lo: f32, hi: f32) -> f32 {
    let unit = (raw as f32) / (u32::MAX as f32);
    lo + unit * (hi - lo)
}

/// The squared length of the edge cross product, used to reject near-degenerate
/// triangles in the random sweep so no scene straddles the `1e-12` guard.
fn cross_sq(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cx = e1[1] * e2[2] - e1[2] * e2[1];
    let cy = e1[2] * e2[0] - e1[0] * e2[2];
    let cz = e1[0] * e2[1] - e1[1] * e2[0];
    cx * cx + cy * cy + cz * cz
}

/// Builds a random non-degenerate shared-vertex quad scene. Positions are
/// reject-sampled so each face is well clear of the degenerate guard, and the
/// branch-selecting coefficients snap to one side of their crossings.
fn random_scene(state: &mut u64) -> Scene {
    let triangles = vec![[0u32, 1, 2], [2u32, 1, 3]];
    let pts = loop {
        let mut candidate = [[0.0f32; 3]; 4];
        for pt in &mut candidate {
            *pt = [
                uniform(lcg(state), -2.0, 2.0),
                uniform(lcg(state), -2.0, 2.0),
                uniform(lcg(state), -2.0, 2.0),
            ];
        }
        let a = cross_sq(candidate[0], candidate[1], candidate[2]);
        let b = cross_sq(candidate[2], candidate[1], candidate[3]);
        if a > 1e-2 && b > 1e-2 {
            break candidate;
        }
    };
    let mut particles = Vec::with_capacity(4);
    for pt in &pts {
        let im = uniform(lcg(state), 0.5, 2.0);
        let vel = [
            uniform(lcg(state), -1.0, 1.0),
            uniform(lcg(state), -1.0, 1.0),
            uniform(lcg(state), -1.0, 1.0),
        ];
        particles.push(free(*pt, vel, im));
    }
    let wind_velocity = [
        uniform(lcg(state), -3.0, 3.0),
        uniform(lcg(state), -3.0, 3.0),
        uniform(lcg(state), -3.0, 3.0),
    ];
    let turbulence = if lcg(state).is_multiple_of(2) {
        0.0
    } else {
        uniform(lcg(state), 0.1, 1.0)
    };
    let drag = uniform(lcg(state), 0.0, 2.0);
    let lift = uniform(lcg(state), 0.0, 2.0);
    let air_density = if lcg(state).is_multiple_of(2) {
        0.0
    } else {
        uniform(lcg(state), 0.5, 2.0)
    };
    let dt = uniform(lcg(state), 0.1, 2.0);
    Scene {
        particles,
        triangles,
        wind_velocity,
        turbulence,
        drag,
        lift,
        air_density,
        dt,
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_aero_gather parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuClothAeroGather::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn single_triangle_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothAeroGather::new(&ctx);
    // One non-degenerate triangle in the XY plane (normal +Z, area 0.5); all
    // three vertices gather the single face force.
    let scene = Scene {
        particles: vec![
            free([0.0, 0.0, 0.0], [0.1, -0.2, 0.0], 1.0),
            free([1.0, 0.0, 0.0], [0.0, 0.3, -0.1], 1.0),
            free([0.0, 1.0, 0.0], [-0.2, 0.0, 0.2], 1.0),
        ],
        triangles: vec![[0, 1, 2]],
        wind_velocity: [2.0, 0.5, -1.0],
        turbulence: 0.0,
        drag: 1.0,
        lift: 0.5,
        air_density: 0.0,
        dt: 0.5,
    };
    check(&ctx, &gpu, &[scene]);
}

#[test]
fn shared_vertex_quad_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothAeroGather::new(&ctx);
    // A unit quad split into two triangles sharing edge 1-2; vertices 1 and 2
    // each gather both faces.
    let scene = Scene {
        particles: vec![
            free([0.0, 0.0, 0.0], [0.05, 0.0, 0.0], 1.0),
            free([1.0, 0.0, 0.0], [0.0, 0.1, 0.0], 0.5),
            free([0.0, 1.0, 0.0], [0.0, 0.0, 0.2], 1.5),
            free([1.0, 1.0, 0.0], [-0.1, 0.0, 0.0], 2.0),
        ],
        triangles: vec![[0, 1, 2], [2, 1, 3]],
        wind_velocity: [1.5, -0.5, 2.0],
        turbulence: 0.0,
        drag: 0.8,
        lift: 1.2,
        air_density: 0.0,
        dt: 0.75,
    };
    check(&ctx, &gpu, &[scene]);
}

#[test]
fn pinned_vertex_passes_through() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothAeroGather::new(&ctx);
    // Vertex 0 is pinned; the golden skips it and the twin passes its input
    // velocity through unchanged.
    let mut pinned = ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0));
    pinned.velocity = Vec3::new(0.3, -0.2, 0.1);
    let scene = Scene {
        particles: vec![
            pinned,
            free([1.0, 0.0, 0.0], [0.0, 0.1, 0.0], 1.0),
            free([0.0, 1.0, 0.0], [0.0, 0.0, 0.2], 1.0),
        ],
        triangles: vec![[0, 1, 2]],
        wind_velocity: [2.0, 1.0, -1.0],
        turbulence: 0.0,
        drag: 1.0,
        lift: 0.5,
        air_density: 0.0,
        dt: 0.5,
    };
    check(&ctx, &gpu, &[scene]);
}

#[test]
fn nonpositive_dt_is_noop() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothAeroGather::new(&ctx);
    // A zero step is a degenerate no-op: every velocity passes through.
    let scene = Scene {
        particles: vec![
            free([0.0, 0.0, 0.0], [0.1, 0.2, 0.3], 1.0),
            free([1.0, 0.0, 0.0], [0.4, 0.5, 0.6], 1.0),
            free([0.0, 1.0, 0.0], [0.7, 0.8, 0.9], 1.0),
        ],
        triangles: vec![[0, 1, 2]],
        wind_velocity: [3.0, 2.0, 1.0],
        turbulence: 0.0,
        drag: 1.0,
        lift: 1.0,
        air_density: 0.0,
        dt: 0.0,
    };
    check(&ctx, &gpu, &[scene]);
}

#[test]
fn nonfinite_dt_is_noop() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothAeroGather::new(&ctx);
    // A non-finite step is a degenerate no-op, matching the golden guard.
    let scene = Scene {
        particles: vec![
            free([0.0, 0.0, 0.0], [0.1, 0.2, 0.3], 1.0),
            free([1.0, 0.0, 0.0], [0.4, 0.5, 0.6], 1.0),
            free([0.0, 1.0, 0.0], [0.7, 0.8, 0.9], 1.0),
        ],
        triangles: vec![[0, 1, 2]],
        wind_velocity: [3.0, 2.0, 1.0],
        turbulence: 0.0,
        drag: 1.0,
        lift: 1.0,
        air_density: 0.0,
        dt: f32::NAN,
    };
    check(&ctx, &gpu, &[scene]);
}

#[test]
fn out_of_range_triangle_is_dropped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothAeroGather::new(&ctx);
    // The second face references vertex 5 which does not exist (count 3), so the
    // adjacency build drops it; only the first face contributes.
    let scene = Scene {
        particles: vec![
            free([0.0, 0.0, 0.0], [0.1, -0.2, 0.0], 1.0),
            free([1.0, 0.0, 0.0], [0.0, 0.3, -0.1], 1.0),
            free([0.0, 1.0, 0.0], [-0.2, 0.0, 0.2], 1.0),
        ],
        triangles: vec![[0, 1, 2], [5, 1, 2]],
        wind_velocity: [2.0, 0.5, -1.0],
        turbulence: 0.0,
        drag: 1.0,
        lift: 0.5,
        air_density: 0.0,
        dt: 0.5,
    };
    check(&ctx, &gpu, &[scene]);
}

#[test]
fn turbulence_jitter_bit_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothAeroGather::new(&ctx);
    // A positive turbulence engages the integer-hash per-face jitter; the twin's
    // u32 wrapping hash must reproduce the golden offset exactly.
    let scene = Scene {
        particles: vec![
            free([0.0, 0.0, 0.0], [0.05, 0.0, 0.0], 1.0),
            free([1.0, 0.0, 0.0], [0.0, 0.1, 0.0], 0.5),
            free([0.0, 1.0, 0.0], [0.0, 0.0, 0.2], 1.5),
            free([1.0, 1.0, 0.0], [-0.1, 0.0, 0.0], 2.0),
        ],
        triangles: vec![[0, 1, 2], [2, 1, 3]],
        wind_velocity: [1.0, -1.0, 0.5],
        turbulence: 0.7,
        drag: 1.0,
        lift: 0.5,
        air_density: 0.0,
        dt: 0.5,
    };
    check(&ctx, &gpu, &[scene]);
}

#[test]
fn quadratic_model_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothAeroGather::new(&ctx);
    // A positive air density selects the quadratic (airspeed-squared) pressure.
    let scene = Scene {
        particles: vec![
            free([0.0, 0.0, 0.0], [0.1, 0.0, 0.0], 1.0),
            free([2.0, 0.0, 0.0], [0.0, 0.2, 0.0], 1.0),
            free([0.0, 2.0, 0.0], [0.0, 0.0, 0.3], 1.0),
        ],
        triangles: vec![[0, 1, 2]],
        wind_velocity: [4.0, -2.0, 1.0],
        turbulence: 0.0,
        drag: 1.1,
        lift: 0.4,
        air_density: 1.225,
        dt: 0.5,
    };
    check(&ctx, &gpu, &[scene]);
}

#[test]
fn sanitizes_nonfinite_wind_and_negative_coeffs() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothAeroGather::new(&ctx);
    // A non-finite wind component and negative coefficients are sanitized to the
    // same values on both paths (non-finite -> 0, coefficients max(0)).
    let scene = Scene {
        particles: vec![
            free([0.0, 0.0, 0.0], [0.1, -0.2, 0.0], 1.0),
            free([1.0, 0.0, 0.0], [0.0, 0.3, -0.1], 1.0),
            free([0.0, 1.0, 0.0], [-0.2, 0.0, 0.2], 1.0),
        ],
        triangles: vec![[0, 1, 2]],
        wind_velocity: [f32::NAN, 1.0, f32::INFINITY],
        turbulence: -0.5,
        drag: -1.0,
        lift: 0.6,
        air_density: -2.0,
        dt: 0.5,
    };
    check(&ctx, &gpu, &[scene]);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothAeroGather::new(&ctx);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut scenes = Vec::new();
    // A broad sweep over random non-degenerate quads, random velocities, wind,
    // coefficients and step. Every branch is snapped clear of its crossing, so
    // no reject-sampling beyond the triangle-degeneracy guard is needed.
    while scenes.len() < 256 {
        scenes.push(random_scene(&mut state));
    }
    check(&ctx, &gpu, &scenes);
}
