//! Real-device parity for the signed-distance-field soft-shadow twin:
//! [`GpuSdfSoftShadow`](prism_volumetric_gpu::mesh_sdf_soft_shadow::GpuSdfSoftShadow)
//! must reproduce the `CPU` golden `sdf_soft_shadow` of
//! `prism_render_architecture::ray_scene::mesh_sdf_soft_shadow`, which marches a
//! penumbra ray through the trilinear sampler `sample_signed_distance` of
//! `prism_render_architecture::ray_scene::mesh_sdf_raymarch`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! trilinear sampling plus the Inigo Quilez improved penumbra sphere trace with
//! the overshoot correction `y = sampled^2 / (2 * prev)` — written directly
//! against a flat per-cell signed-distance array so the test does not import
//! `prism_render_architecture`. It operates on the same field layout the twin
//! consumes (`x` fastest, `clamp-to-border` addressing) and marches in `f32`
//! with a `has_prev` boolean so the first-step correction and the discrete
//! `steps` count align bit-for-bit with the device.
//!
//! The fixtures cover the shapes the kernel must honor: a degenerate direction
//! (`valid = 0`), a ray that starts inside a solid (immediate full occlusion), a
//! clear ray through a far constant field (fully lit), a grazing ray that stays
//! just outside a sphere (true penumbra), a sweep over random fields and rays
//! that never hit and never cross `max_distance` (so `steps` is a fixed count),
//! and an empty query batch (host short-circuits with no dispatch).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The march is a bounded loop of multiplies, a `max`-guarded `sqrt`, a step
//! floor and a trilinear tap, so `CPU` and `GPU` evaluate the same closed form
//! but need not be bit-exact (a `GPU` may contract a multiply-add). The
//! continuous comparison on `visibility` is `abs_diff <= 1e-4 || rel_diff <=
//! 1e-3` (`REL_FLOOR = 1e-6`). The `steps`, `occluded` and `valid` outputs are
//! integer/boolean decisions and match exactly; the fixtures are shaped so the
//! march terminates unambiguously (hits forced or avoided, `max_distance` never
//! crossed, `max_steps` small) so those counts never straddle a floating-point
//! boundary.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_soft_shadow`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mesh_sdf_soft_shadow::{
    GpuSdfSoftShadow, SdfSoftShadowField, SdfSoftShadowQuery, SdfSoftShadowResult,
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

/// Independent oracle re-implementing the golden `sdf_soft_shadow`: a bounded
/// sphere trace of the trilinear field accumulating the improved penumbra
/// estimator. Marches in `f32` with a `has_prev` boolean so the first-step
/// overshoot correction is exactly zero and the discrete `steps` count aligns
/// with the device, which has no portable infinity literal.
fn oracle(field: &SdfSoftShadowField, q: &SdfSoftShadowQuery) -> SdfSoftShadowResult {
    let dims = field.dims();
    let origin = field.origin();
    let voxel = field.voxel_size();
    let dist = field.distances();

    let d = q.direction;
    let len2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    if len2 <= f32::MIN_POSITIVE {
        return SdfSoftShadowResult {
            visibility: 0.0,
            steps: 0,
            occluded: 0,
            valid: 0,
        };
    }
    let inv_len = 1.0 / len2.sqrt();
    let dir = [d[0] * inv_len, d[1] * inv_len, d[2] * inv_len];
    let min_step = voxel * 0.125;

    let mut visibility = 1.0f32;
    let mut t = q.min_distance.max(0.0);
    let mut has_prev = false;
    let mut prev_distance = 0.0f32;
    let mut steps = 0u32;
    let mut occluded = false;

    for i in 0..q.max_steps {
        if t > q.max_distance {
            break;
        }
        steps = i + 1;
        let position = [
            q.origin[0] + t * dir[0],
            q.origin[1] + t * dir[1],
            q.origin[2] + t * dir[2],
        ];
        let sampled = sample(dims, origin, voxel, dist, position);
        if sampled <= q.hit_epsilon {
            visibility = 0.0;
            occluded = true;
            break;
        }
        let y = if has_prev {
            sampled * sampled / (2.0 * prev_distance)
        } else {
            0.0
        };
        let chord = (sampled * sampled - y * y).max(0.0).sqrt();
        let reach = (t - y).max(min_step);
        visibility = visibility.min(q.sharpness * chord / reach);
        prev_distance = sampled;
        has_prev = true;
        t += sampled.max(min_step);
    }

    SdfSoftShadowResult {
        visibility: visibility.clamp(0.0, 1.0),
        steps,
        occluded: u32::from(occluded),
        valid: 1,
    }
}

/// Asserts the `GPU` results match the oracle: `visibility` within the
/// continuous tolerance; `steps`, `occluded` and `valid` exactly.
fn assert_parity(
    gpu: &[SdfSoftShadowResult],
    field: &SdfSoftShadowField,
    queries: &[SdfSoftShadowQuery],
    label: &str,
) {
    assert_eq!(gpu.len(), queries.len(), "{label}: result count mismatch");
    for (i, (g, q)) in gpu.iter().zip(queries.iter()).enumerate() {
        let want = oracle(field, q);
        assert_eq!(g.valid, want.valid, "{label}: query {i} valid flag");
        assert_eq!(
            g.occluded, want.occluded,
            "{label}: query {i} occluded flag"
        );
        assert_eq!(g.steps, want.steps, "{label}: query {i} steps count");
        assert!(
            close(g.visibility, want.visibility),
            "{label}: query {i} visibility GPU {} vs CPU {}",
            g.visibility,
            want.visibility
        );
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

/// Builds a radial (sphere) field: each cell stores the analytic signed
/// distance `|p - center| - radius`, row-major with `x` fastest.
fn radial_field(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    center: [f32; 3],
    radius: f32,
) -> SdfSoftShadowField {
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
    SdfSoftShadowField::new(dims, origin, voxel, dist)
}

/// Builds a field whose every cell holds the same constant signed distance.
fn constant_field(dims: [u32; 3], origin: [f32; 3], voxel: f32, value: f32) -> SdfSoftShadowField {
    let count = (dims[0] * dims[1] * dims[2]) as usize;
    SdfSoftShadowField::new(dims, origin, voxel, vec![value; count])
}

#[test]
fn degenerate_direction_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = constant_field([3, 3, 3], [-1.0, -1.0, -1.0], 1.0, 5.0);
    let gpu = GpuSdfSoftShadow::new(&ctx);
    let queries = vec![
        SdfSoftShadowQuery::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.1, 10.0, 8.0, -1.0, 16),
        SdfSoftShadowQuery::new([0.2, -0.1, 0.3], [0.0, 0.0, 0.0], 0.0, 20.0, 4.0, 0.0, 8),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "degenerate");
    for r in &out {
        assert_eq!(r.valid, 0, "zero direction must be invalid");
        assert_eq!(r.steps, 0);
        assert_eq!(r.occluded, 0);
        assert!(close(r.visibility, 0.0));
    }
}

#[test]
fn ray_starting_inside_solid_is_occluded() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Sphere of radius 0.75 centred at the world origin; the ray starts at the
    // centre so the very first sample is strongly negative and trips the hit.
    let field = radial_field([5, 5, 5], [-1.0, -1.0, -1.0], 0.5, [0.0, 0.0, 0.0], 0.75);
    let gpu = GpuSdfSoftShadow::new(&ctx);
    let queries = vec![SdfSoftShadowQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.0,
        10.0,
        8.0,
        0.0,
        16,
    )];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "occluded");
    assert_eq!(out[0].occluded, 1, "ray inside the solid must be occluded");
    assert_eq!(out[0].steps, 1, "occlusion must trip on the first step");
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].visibility, 0.0));
}

#[test]
fn clear_ray_through_far_field_is_fully_lit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Everything is far away (distance 10), so every step ratio dwarfs one and
    // the running minimum stays at full visibility.
    let field = constant_field([3, 3, 3], [-1.0, -1.0, -1.0], 1.0, 10.0);
    let gpu = GpuSdfSoftShadow::new(&ctx);
    let queries = vec![SdfSoftShadowQuery::new(
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        0.1,
        100.0,
        16.0,
        -0.5,
        2,
    )];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "clear");
    assert_eq!(out[0].occluded, 0, "clear ray must not be occluded");
    assert_eq!(out[0].steps, 2);
    assert_eq!(out[0].valid, 1);
    assert!(
        close(out[0].visibility, 1.0),
        "clear ray must stay fully lit"
    );
}

#[test]
fn grazing_ray_forms_a_penumbra() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Sphere radius 0.75 at the origin; the ray travels along +x on the line
    // y = -0.9, z = 0, so its closest approach is ~0.15 outside the surface:
    // positive everywhere (no hit) yet small enough to darken the light.
    let field = radial_field(
        [17, 17, 17],
        [-2.0, -2.0, -2.0],
        0.25,
        [0.0, 0.0, 0.0],
        0.75,
    );
    let gpu = GpuSdfSoftShadow::new(&ctx);
    let queries = vec![SdfSoftShadowQuery::new(
        [-1.5, -0.9, 0.0],
        [1.0, 0.0, 0.0],
        0.0,
        1000.0,
        4.0,
        -1.0,
        24,
    )];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "penumbra");
    let want = oracle(&field, &queries[0]);
    assert_eq!(out[0].occluded, 0, "grazing ray must not report a hit");
    assert_eq!(
        out[0].steps, 24,
        "march must run every step without crossing the far bound"
    );
    assert_eq!(out[0].valid, 1);
    assert!(
        want.visibility > 0.0 && want.visibility < 1.0,
        "grazing ray must land strictly inside the penumbra, got {}",
        want.visibility
    );
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfSoftShadow::new(&ctx);
    let mut rng = Lcg::new(0x0C7E_51D5);
    let mut built = 0u32;
    while built < 512 {
        let dims = [
            2 + (rng.next_u32() % 4),
            2 + (rng.next_u32() % 4),
            2 + (rng.next_u32() % 4),
        ];
        let voxel = rng.next_range(0.25, 1.5);
        let origin = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        let count = (dims[0] * dims[1] * dims[2]) as usize;
        let mut dist = Vec::with_capacity(count);
        for _ in 0..count {
            // Strictly positive cells so the march never records a hit; combined
            // with a negative `hit_epsilon` the ray always runs the full loop.
            dist.push(rng.next_range(0.3, 3.0));
        }
        let dir = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        let len2 = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
        // Reject a near-zero direction so both sides agree it is valid.
        if len2 <= 1.0e-3 {
            continue;
        }
        let field = SdfSoftShadowField::new(dims, origin, voxel, dist);
        // `max_distance` far beyond the furthest reachable `t` (12 steps of at
        // most 3.0) and a never-tripped `hit_epsilon` pin `steps` to 12 exactly.
        let queries = [SdfSoftShadowQuery::new(
            [
                origin[0] + rng.next_range(0.0, 1.0),
                origin[1] + rng.next_range(0.0, 1.0),
                origin[2] + rng.next_range(0.0, 1.0),
            ],
            dir,
            rng.next_range(0.0, 0.2),
            1000.0,
            rng.next_range(1.0, 16.0),
            -1.0,
            12,
        )];
        let out = gpu.evaluate(&ctx, &field, &queries);
        assert_parity(&out, &field, &queries, "sweep");
        assert_eq!(out[0].steps, 12, "sweep ray must run every step");
        assert_eq!(out[0].occluded, 0, "sweep ray must never hit");
        assert_eq!(out[0].valid, 1);
        built += 1;
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = constant_field([3, 3, 3], [-1.0, -1.0, -1.0], 1.0, 4.0);
    let gpu = GpuSdfSoftShadow::new(&ctx);
    let out = gpu.evaluate(&ctx, &field, &[]);
    assert!(out.is_empty(), "empty batch must return no results");
}
