//! Real-device parity for the sphere-traced signed-distance-field ray-marching
//! twin:
//! [`GpuSdfSphereTrace`](prism_volumetric_gpu::mesh_sdf_raymarch::GpuSdfSphereTrace)
//! must reproduce the `CPU` golden `sphere_trace` of
//! `prism_render_architecture::ray_scene::mesh_sdf_raymarch`, which normalizes
//! the ray direction, clips the ray against the field's axis-aligned bounding
//! box with the slab method, and then marches from the entry parameter,
//! sampling the continuous trilinear signed distance at every step and reporting
//! the first crossing of `hit_epsilon`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! guarded direction normalization, the slab clip, trilinear sampling and the
//! sphere-tracing loop — written directly against a flat per-cell
//! signed-distance array so the test does not import `prism_render_architecture`.
//! It operates on the same field layout the twin consumes (`x` fastest,
//! `clamp-to-border` addressing) and tracks every value in `f32` so it follows
//! the `GPU` trajectory step for step, which keeps the discrete `hit` flag and
//! `steps` count identical on both sides.
//!
//! The fixtures cover the shapes the kernel must honor: an axial ray that enters
//! the box and converges on a voxelized sphere, a ray that passes beside the
//! sphere and misses, a ray that points away and leaves the box immediately, a
//! zero-length direction that cannot be normalized (`hit = 0`), a non-unit
//! direction that matches its unit twin (device-side normalization), a sweep
//! over random sphere fields and aimed rays, and an empty query batch (the host
//! short-circuits with no dispatch).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Sampling is a fixed sequence of linear-interpolation adds, the slab clip is a
//! handful of divides and `min`/`max`, and normalization is a single `sqrt`, so
//! `CPU` and `GPU` evaluate the same closed form but need not be bit-exact (a
//! `GPU` may contract a multiply-add). The continuous comparison on the ray
//! parameter, the hit position and the sampled distance is
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`). The `hit` flag
//! and the `steps` count are integer decisions and match exactly; the random
//! sweep rejects rays whose march grazes the hit epsilon on any step or whose
//! advance lands on the exit parameter, so neither discrete decision flips
//! between the two sides.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_raymarch`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mesh_sdf_raymarch::{
    GpuSdfSphereTrace, SdfSphereTraceField, SdfSphereTraceQuery, SdfSphereTraceResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_TOL: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_TOL: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_TOL {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_TOL
}

/// Linear interpolation between `a` and `b` by `s`, matching the golden `lerp`.
fn lerp(a: f32, b: f32, s: f32) -> f32 {
    a + (b - a) * s
}

/// Independent oracle re-implementing the golden trilinear
/// `sample_signed_distance` against a flat per-cell signed-distance array.
fn sample(dims: [u32; 3], origin: [f32; 3], voxel: f32, dist: &[f32], point: [f32; 3]) -> f32 {
    let mut base = [0u32; 3];
    let mut frac = [0f32; 3];
    for (axis, slot) in base.iter_mut().enumerate() {
        if dims[axis] <= 1 {
            frac[axis] = 0.0;
            continue;
        }
        let last = (dims[axis] - 1) as f32;
        let continuous = (point[axis] - origin[axis]) / voxel - 0.5;
        let clamped = continuous.clamp(0.0, last);
        let lower = clamped.floor().clamp(0.0, last - 1.0);
        *slot = lower as u32;
        frac[axis] = clamped - lower;
    }
    let corner = |dx: u32, dy: u32, dz: u32| -> f32 {
        let x = (base[0] + dx).min(dims[0] - 1);
        let y = (base[1] + dy).min(dims[1] - 1);
        let z = (base[2] + dz).min(dims[2] - 1);
        let idx = ((z * dims[1] + y) * dims[0] + x) as usize;
        dist[idx]
    };
    let d000 = corner(0, 0, 0);
    let d100 = corner(1, 0, 0);
    let d010 = corner(0, 1, 0);
    let d110 = corner(1, 1, 0);
    let d001 = corner(0, 0, 1);
    let d101 = corner(1, 0, 1);
    let d011 = corner(0, 1, 1);
    let d111 = corner(1, 1, 1);
    let c00 = lerp(d000, d100, frac[0]);
    let c01 = lerp(d001, d101, frac[0]);
    let c10 = lerp(d010, d110, frac[0]);
    let c11 = lerp(d011, d111, frac[0]);
    let c0 = lerp(c00, c10, frac[1]);
    let c1 = lerp(c01, c11, frac[1]);
    lerp(c0, c1, frac[2])
}

/// Independent oracle re-implementing the golden guarded direction
/// normalization: a direction too short to normalize has no defined heading.
fn normalize(dir: [f32; 3]) -> Option<[f32; 3]> {
    let len2 = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
    if len2 <= f32::MIN_POSITIVE {
        return None;
    }
    let len = len2.sqrt();
    Some([dir[0] / len, dir[1] / len, dir[2] / len])
}

/// Independent oracle re-implementing the golden `ray_aabb` slab clip: the entry
/// parameter starts at zero, the exit at positive infinity, and an axis parallel
/// to a slab misses unless the origin sits between the planes.
fn ray_aabb(o: [f32; 3], dir: [f32; 3], bmin: [f32; 3], bmax: [f32; 3]) -> Option<(f32, f32)> {
    let mut t0 = 0.0f32;
    let mut t1 = f32::INFINITY;
    for axis in 0..3 {
        let d = dir[axis];
        if d.abs() <= f32::MIN_POSITIVE {
            if o[axis] < bmin[axis] || o[axis] > bmax[axis] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / d;
        let mut ta = (bmin[axis] - o[axis]) * inv;
        let mut tb = (bmax[axis] - o[axis]) * inv;
        if ta > tb {
            std::mem::swap(&mut ta, &mut tb);
        }
        t0 = t0.max(ta);
        t1 = t1.min(tb);
        if t0 > t1 {
            return None;
        }
    }
    Some((t0, t1))
}

/// Diagnostics collected along the oracle march, used only by the random sweep
/// to reject rays whose traversal grazes a discrete decision boundary (so
/// neither the `hit` flag nor the `steps` count can flip between the two sides
/// of the parity comparison).
struct MarchDiag {
    /// Minimum over evaluated steps of `|distance - hit_epsilon|`.
    min_hit_margin: f32,
    /// Minimum over advancing steps of `|t - t_exit|`.
    min_exit_margin: f32,
}

/// Independent oracle for the golden `sphere_trace`, mirroring the kernel step
/// for step in `f32`. Returns the trace result plus path diagnostics.
///
/// A zero-length direction, an empty slab interval or a spent step budget all
/// miss with a zeroed result (the golden `None`). A hit returns the step index
/// plus one as the `steps` count, matching the golden `step + 1`.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the golden sphere_trace signature verbatim for a faithful oracle"
)]
fn march(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    dist: &[f32],
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    max_distance: f32,
    hit_epsilon: f32,
    max_steps: u32,
) -> (SdfSphereTraceResult, MarchDiag) {
    let miss = SdfSphereTraceResult {
        hit: 0,
        t: 0.0,
        position: [0.0, 0.0, 0.0],
        distance: 0.0,
        steps: 0,
    };
    let mut diag = MarchDiag {
        min_hit_margin: f32::INFINITY,
        min_exit_margin: f32::INFINITY,
    };

    let Some(dir) = normalize(ray_dir) else {
        return (miss, diag);
    };
    let grid_origin = origin;
    let grid_max = [
        origin[0] + dims[0] as f32 * voxel,
        origin[1] + dims[1] as f32 * voxel,
        origin[2] + dims[2] as f32 * voxel,
    ];
    let Some((t_enter, t_exit_raw)) = ray_aabb(ray_origin, dir, grid_origin, grid_max) else {
        return (miss, diag);
    };
    let t_exit = t_exit_raw.min(max_distance);
    if t_enter > t_exit {
        return (miss, diag);
    }

    let min_step = voxel * 0.125;
    let mut t = t_enter;
    let mut step = 0u32;
    while step < max_steps {
        let pos = [
            ray_origin[0] + t * dir[0],
            ray_origin[1] + t * dir[1],
            ray_origin[2] + t * dir[2],
        ];
        let distance = sample(dims, origin, voxel, dist, pos);
        diag.min_hit_margin = diag.min_hit_margin.min((distance - hit_epsilon).abs());
        if distance <= hit_epsilon {
            return (
                SdfSphereTraceResult {
                    hit: 1,
                    t,
                    position: pos,
                    distance,
                    steps: step + 1,
                },
                diag,
            );
        }
        t += distance.max(min_step);
        diag.min_exit_margin = diag.min_exit_margin.min((t - t_exit).abs());
        if t > t_exit {
            return (miss, diag);
        }
        step += 1;
    }
    (miss, diag)
}

/// Asserts the `GPU` results match the oracle: the `hit` flag and `steps` count
/// exactly, and (on a hit) the ray parameter, position and sampled distance
/// within the continuous tolerance. On a miss both sides are zeroed, so only the
/// discrete fields are compared.
fn assert_parity(
    gpu: &[SdfSphereTraceResult],
    field: &SdfSphereTraceField,
    queries: &[SdfSphereTraceQuery],
    label: &str,
) {
    assert_eq!(gpu.len(), queries.len(), "{label}: result count mismatch");
    for (i, (g, q)) in gpu.iter().zip(queries.iter()).enumerate() {
        let (want, _) = march(
            field.dims(),
            field.origin(),
            field.voxel_size(),
            field.distances(),
            q.origin,
            q.direction,
            q.max_distance,
            q.hit_epsilon,
            q.max_steps,
        );
        assert_eq!(
            g.hit, want.hit,
            "{label}: query {i} hit flag GPU {} vs CPU {}",
            g.hit, want.hit
        );
        assert_eq!(
            g.steps, want.steps,
            "{label}: query {i} step count GPU {} vs CPU {}",
            g.steps, want.steps
        );
        if want.hit == 1 {
            assert!(
                close(g.t, want.t),
                "{label}: query {i} t GPU {} vs CPU {}",
                g.t,
                want.t
            );
            assert!(
                close(g.distance, want.distance),
                "{label}: query {i} distance GPU {} vs CPU {}",
                g.distance,
                want.distance
            );
            for k in 0..3 {
                assert!(
                    close(g.position[k], want.position[k]),
                    "{label}: query {i} position[{k}] GPU {} vs CPU {}",
                    g.position[k],
                    want.position[k]
                );
            }
        }
    }
}

/// A small deterministic linear-congruential generator so the sweep needs no
/// external randomness. Constants are the Numerical Recipes values.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A `[0, 1)` fraction built from the top bits, keeping the fixture pure
    /// integer host-side with no transcendental call.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}

/// Builds a voxelized sphere field: each cell stores the Euclidean distance from
/// its world-space corner to `center` minus `radius`, giving an oriented,
/// well-conditioned signed distance the sphere march converges on quickly.
fn sphere_field(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    center: [f32; 3],
    radius: f32,
) -> SdfSphereTraceField {
    let count = (dims[0] * dims[1] * dims[2]) as usize;
    let mut dist = Vec::with_capacity(count);
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let wx = origin[0] + x as f32 * voxel;
                let wy = origin[1] + y as f32 * voxel;
                let wz = origin[2] + z as f32 * voxel;
                let cx = wx - center[0];
                let cy = wy - center[1];
                let cz = wz - center[2];
                let r = (cx * cx + cy * cy + cz * cz).sqrt();
                dist.push(r - radius);
            }
        }
    }
    SdfSphereTraceField::new(dims, origin, voxel, dist)
}

#[test]
fn axial_ray_hits_sphere() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = sphere_field([16, 16, 16], [-2.0, -2.0, -2.0], 0.25, [0.0, 0.0, 0.0], 1.0);
    let gpu = GpuSdfSphereTrace::new(&ctx);
    // A ray on the +x axis starting outside the sphere marches straight onto its
    // surface at x = -1.
    let queries = vec![
        SdfSphereTraceQuery::new([-1.9, 0.0, 0.0], [1.0, 0.0, 0.0], 8.0, 0.03, 128),
        SdfSphereTraceQuery::new([0.0, -1.9, 0.0], [0.0, 1.0, 0.0], 8.0, 0.03, 128),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "axial_hit");
    for r in &out {
        assert_eq!(r.hit, 1, "axial ray must strike the sphere");
        assert!(r.steps >= 1, "a hit takes at least one step");
    }
}

#[test]
fn grazing_ray_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = sphere_field([16, 16, 16], [-2.0, -2.0, -2.0], 0.25, [0.0, 0.0, 0.0], 1.0);
    let gpu = GpuSdfSphereTrace::new(&ctx);
    // A ray held at y = 1.6 passes well outside the unit sphere, so the signed
    // distance never drops to the hit epsilon.
    let queries = vec![SdfSphereTraceQuery::new(
        [-1.9, 1.6, 0.0],
        [1.0, 0.0, 0.0],
        8.0,
        0.03,
        128,
    )];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "grazing_miss");
    assert_eq!(out[0].hit, 0, "a ray beside the sphere must miss");
}

#[test]
fn ray_pointing_away_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = sphere_field([16, 16, 16], [-2.0, -2.0, -2.0], 0.25, [0.0, 0.0, 0.0], 1.0);
    let gpu = GpuSdfSphereTrace::new(&ctx);
    // A ray starting outside the sphere and pointing further out leaves the box
    // before reaching any surface.
    let queries = vec![SdfSphereTraceQuery::new(
        [-1.9, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        8.0,
        0.03,
        128,
    )];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "pointing_away");
    assert_eq!(out[0].hit, 0, "a ray leaving the box must miss");
}

#[test]
fn degenerate_direction_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = sphere_field([16, 16, 16], [-2.0, -2.0, -2.0], 0.25, [0.0, 0.0, 0.0], 1.0);
    let gpu = GpuSdfSphereTrace::new(&ctx);
    // A zero-length direction cannot be normalized, so the trace misses with a
    // zeroed result (the golden `None`).
    let queries = vec![SdfSphereTraceQuery::new(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        8.0,
        0.03,
        128,
    )];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "degenerate_dir");
    assert_eq!(out[0].hit, 0, "a zero direction must miss");
    assert_eq!(out[0].steps, 0, "a zero direction takes no step");
}

#[test]
fn non_unit_direction_matches_unit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = sphere_field([16, 16, 16], [-2.0, -2.0, -2.0], 0.25, [0.0, 0.0, 0.0], 1.0);
    let gpu = GpuSdfSphereTrace::new(&ctx);
    // The device normalizes the direction, so a scaled direction must produce
    // the same hit as its unit twin.
    let queries = vec![
        SdfSphereTraceQuery::new([-1.9, 0.0, 0.0], [1.0, 0.0, 0.0], 8.0, 0.03, 128),
        SdfSphereTraceQuery::new([-1.9, 0.0, 0.0], [5.0, 0.0, 0.0], 8.0, 0.03, 128),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "non_unit_dir");
    assert_eq!(out[0].hit, out[1].hit, "scaled direction must match unit");
    assert_eq!(out[0].steps, out[1].steps, "step counts must match");
    assert!(close(out[0].t, out[1].t), "ray parameters must match");
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSphereTrace::new(&ctx);
    let mut rng = Lcg::new(0x51A3_C77F);
    // Margins that keep every discrete decision clear of its boundary: the hit
    // epsilon band for the convergence test and the exit parameter for the
    // out-of-box test. Both comfortably exceed any `CPU`/`GPU` drift.
    let hit_margin = 0.01f32;
    let exit_margin = 0.01f32;
    let dims = [20u32, 20, 20];
    let origin = [-2.5f32, -2.5, -2.5];
    let voxel = 0.25f32;
    let mut built = 0u32;
    let mut attempts = 0u32;
    while built < 512 && attempts < 200_000 {
        attempts += 1;
        // A sphere whose surface sits well inside the field extent.
        let center = [
            rng.next_range(-0.3, 0.3),
            rng.next_range(-0.3, 0.3),
            rng.next_range(-0.3, 0.3),
        ];
        let radius = rng.next_range(0.5, 0.9);
        let field = sphere_field(dims, origin, voxel, center, radius);

        // A start point on a shell outside the sphere, built from a random unit
        // heading (normalized with `sqrt` only).
        let shell = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        let shell_len2 = shell[0] * shell[0] + shell[1] * shell[1] + shell[2] * shell[2];
        if shell_len2 < 0.1 {
            continue;
        }
        let shell_len = shell_len2.sqrt();
        let start_dist = radius + rng.next_range(0.6, 1.0);
        let start = [
            center[0] + shell[0] / shell_len * start_dist,
            center[1] + shell[1] / shell_len * start_dist,
            center[2] + shell[2] / shell_len * start_dist,
        ];

        // Aim roughly back at the sphere center with a small jitter so the ray
        // still crosses the surface.
        let aim = [
            center[0] - start[0] + rng.next_range(-0.08, 0.08),
            center[1] - start[1] + rng.next_range(-0.08, 0.08),
            center[2] - start[2] + rng.next_range(-0.08, 0.08),
        ];
        let aim_len2 = aim[0] * aim[0] + aim[1] * aim[1] + aim[2] * aim[2];
        if aim_len2 < 0.1 {
            continue;
        }

        let hit_epsilon = rng.next_range(0.02, 0.04);
        let max_distance = 8.0f32;
        let max_steps = 128u32;
        let (want, diag) = march(
            dims,
            origin,
            voxel,
            field.distances(),
            start,
            aim,
            max_distance,
            hit_epsilon,
            max_steps,
        );
        // Only accept well-conditioned hits that stay clear of both discrete
        // decision boundaries.
        if want.hit != 1 {
            continue;
        }
        if diag.min_hit_margin < hit_margin || diag.min_exit_margin < exit_margin {
            continue;
        }

        let queries = [SdfSphereTraceQuery::new(
            start,
            aim,
            max_distance,
            hit_epsilon,
            max_steps,
        )];
        let out = gpu.evaluate(&ctx, &field, &queries);
        assert_parity(&out, &field, &queries, "sweep");
        built += 1;
    }
    assert_eq!(built, 512, "sweep should build the full sample budget");
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = sphere_field([16, 16, 16], [-2.0, -2.0, -2.0], 0.25, [0.0, 0.0, 0.0], 1.0);
    let gpu = GpuSdfSphereTrace::new(&ctx);
    let out = gpu.evaluate(&ctx, &field, &[]);
    assert!(out.is_empty(), "empty batch must return no results");
}
