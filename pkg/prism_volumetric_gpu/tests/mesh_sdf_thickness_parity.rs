//! Real-device parity for the signed-distance-field inward thickness twin:
//! [`GpuSdfThickness`](prism_volumetric_gpu::mesh_sdf_thickness::GpuSdfThickness)
//! must reproduce the `CPU` golden `sdf_thickness` of
//! `prism_render_architecture::ray_scene::mesh_sdf_thickness`, which marches a
//! short ray *into* a surface along the negated outward normal and watches the
//! trilinear sampler `sample_signed_distance` of
//! `prism_render_architecture::ray_scene::mesh_sdf_raymarch` flip sign so the
//! travelled distance at the first re-emergence is the local solid thickness.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! trilinear sampling, a guarded normalize and the fixed-step inward march —
//! written directly against a flat per-cell signed-distance array so the test
//! does not import `prism_render_architecture`. It operates on the same field
//! layout the twin consumes (`x` fastest, `clamp-to-border` addressing).
//!
//! The fixtures cover the shapes the kernel must honor: a slab the probe passes
//! clean through (`reemerged = 1`), free space the probe never enters
//! (`thickness = 0`), an all-solid field whose thickness is capped at the probe
//! budget (`reemerged = 0`), a degenerate zero-length normal (`valid = 0`), a
//! thinner slab, a mixed batch and a `512`-fixture random sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The march is a fixed sequence of adds and ordered comparisons over a
//! trilinear sampler, with one `sqrt` for the normalize, so `CPU` and `GPU`
//! evaluate the same closed form but need not be bit-exact (a `GPU` may
//! contract a multiply-add). The continuous `thickness` comparison is
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`); the `valid`,
//! `entered_solid` and `reemerged` flags are integer decisions and match
//! exactly. The random sweep rejects any fixture whose marched samples skim the
//! sign boundary (`|distance| < 1e-2`) so the per-step solid decision never
//! flips between the two sides of the comparison.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_thickness`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mesh_sdf_thickness::{
    GpuSdfThickness, SdfThicknessField, SdfThicknessQuery, SdfThicknessResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_TOL: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_TOL: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;
/// Rejection margin on the per-step sampled distance for the random sweep, so a
/// fixture never straddles the `distance < 0` solid boundary where `CPU` and
/// `GPU` could disagree on the entry or re-emerge decision.
const SWEEP_MARGIN: f32 = 1.0e-2;

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

/// One probe result plus the smallest sampled `|distance|` seen along the
/// march, used only to reject sweep fixtures that skim the solid boundary.
struct Probe {
    thickness: f32,
    entered: u32,
    reemerged: u32,
    valid: u32,
    min_abs: f32,
}

/// Independent oracle re-implementing the golden `sdf_thickness` branch for
/// branch: the `length_squared <= f32::MIN_POSITIVE` normalize guard, the
/// inward fixed-step march bounded by `max_steps` and `max_distance`, the
/// `distance < 0` entry test, the `else if entered` re-emerge early-out and the
/// loop-exhaustion fallback (full budget when still inside, else zero).
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the golden free-function signature verbatim"
)]
fn march(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    dist: &[f32],
    position: [f32; 3],
    normal: [f32; 3],
    max_distance: f32,
    step: f32,
    max_steps: u32,
) -> Probe {
    let len2 = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    if len2 <= f32::MIN_POSITIVE {
        return Probe {
            thickness: 0.0,
            entered: 0,
            reemerged: 0,
            valid: 0,
            min_abs: f32::INFINITY,
        };
    }
    let len = len2.sqrt();
    let inward = [-normal[0] / len, -normal[1] / len, -normal[2] / len];
    let mut t = 0.0f32;
    let mut entered = 0u32;
    let mut reemerged = 0u32;
    let mut thickness = 0.0f32;
    let mut min_abs = f32::INFINITY;
    let mut i = 0u32;
    while i < max_steps {
        if t > max_distance {
            break;
        }
        let sample_pt = [
            position[0] + t * inward[0],
            position[1] + t * inward[1],
            position[2] + t * inward[2],
        ];
        let distance = sample(dims, origin, voxel, dist, sample_pt);
        let a = distance.abs();
        if a < min_abs {
            min_abs = a;
        }
        if distance < 0.0 {
            entered = 1;
        } else if entered == 1 {
            thickness = t.min(max_distance);
            reemerged = 1;
            break;
        }
        t += step;
        i += 1;
    }
    if reemerged == 0 {
        thickness = if entered == 1 { max_distance } else { 0.0 };
    }
    Probe {
        thickness,
        entered,
        reemerged,
        valid: 1,
        min_abs,
    }
}

/// Asserts the `GPU` results match the oracle: `thickness` within the
/// continuous tolerance, every classification flag exactly. An invalid sample
/// must clear `thickness` to a hard zero.
fn assert_parity(
    gpu: &[SdfThicknessResult],
    field: &SdfThicknessField,
    queries: &[SdfThicknessQuery],
    label: &str,
) {
    assert_eq!(gpu.len(), queries.len(), "{label}: result count mismatch");
    for (i, (g, q)) in gpu.iter().zip(queries.iter()).enumerate() {
        let want = march(
            field.dims(),
            field.origin(),
            field.voxel_size(),
            field.distances(),
            q.position,
            q.normal,
            q.max_distance,
            q.step,
            q.max_steps,
        );
        assert_eq!(g.valid, want.valid, "{label}: query {i} valid flag");
        assert_eq!(
            g.entered_solid, want.entered,
            "{label}: query {i} entered_solid flag"
        );
        assert_eq!(
            g.reemerged, want.reemerged,
            "{label}: query {i} reemerged flag"
        );
        if want.valid == 0 {
            // A degenerate normal clears every output; thickness must be a
            // hard +0.0 (bit comparison avoids a floating-point equality).
            assert_eq!(
                g.thickness.to_bits(),
                0,
                "{label}: query {i} invalid thickness must be zero"
            );
            continue;
        }
        assert!(
            close(g.thickness, want.thickness),
            "{label}: query {i} thickness GPU {} vs CPU {}",
            g.thickness,
            want.thickness
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

/// Builds a slab field centered on `x = 0` with the given `half` width, flat
/// along `y` and `z`: cells store `|x| - half`, negative inside the slab.
fn slab_field(half: f32) -> SdfThicknessField {
    let dims = [17u32, 3, 3];
    let origin = [-2.0, -0.25, -0.25];
    let voxel = 0.25;
    let count = (dims[0] * dims[1] * dims[2]) as usize;
    let mut dist = Vec::with_capacity(count);
    for _z in 0..dims[2] {
        for _y in 0..dims[1] {
            for x in 0..dims[0] {
                let wx = origin[0] + x as f32 * voxel;
                dist.push(wx.abs() - half);
            }
        }
    }
    SdfThicknessField::new(dims, origin, voxel, dist)
}

#[test]
fn penetrates_slab_and_reemerges() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = slab_field(0.5);
    let gpu = GpuSdfThickness::new(&ctx);
    // Start just outside the right face at x = 1.02, march inward (-x) through
    // the slab; the start offset keeps every step clear of the x = +-0.5 faces.
    let queries = vec![SdfThicknessQuery::new(
        [1.02, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        3.0,
        0.05,
        100,
    )];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "penetrate");
    // Sanity: the probe entered, re-emerged and reports a plausible thickness.
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].entered_solid, 1);
    assert_eq!(out[0].reemerged, 1);
    assert!(out[0].thickness > 1.0 && out[0].thickness < 2.0);
}

#[test]
fn thinner_slab_reports_smaller_thickness() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = slab_field(0.25);
    let gpu = GpuSdfThickness::new(&ctx);
    let queries = vec![SdfThicknessQuery::new(
        [1.02, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        3.0,
        0.05,
        100,
    )];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "thin_slab");
    assert_eq!(out[0].reemerged, 1);
}

#[test]
fn free_space_never_enters() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = slab_field(0.5);
    let gpu = GpuSdfThickness::new(&ctx);
    // Start outside the slab and march *away* from it (inward = +x), so every
    // sample stays positive and the probe never enters a solid.
    let queries = vec![SdfThicknessQuery::new(
        [1.5, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        2.0,
        0.1,
        100,
    )];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "free_space");
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].entered_solid, 0);
    assert_eq!(out[0].reemerged, 0);
    assert_eq!(out[0].thickness.to_bits(), 0);
}

#[test]
fn solid_everywhere_caps_at_budget() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Field is solid everywhere, so the probe enters at t = 0 and never
    // re-emerges; the thickness saturates at the probe budget.
    let dims = [4u32, 4, 4];
    let dist = vec![-1.0f32; 64];
    let field = SdfThicknessField::new(dims, [-1.0, -1.0, -1.0], 0.5, dist);
    let gpu = GpuSdfThickness::new(&ctx);
    let max_distance = 2.0;
    let queries = vec![SdfThicknessQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        max_distance,
        0.1,
        100,
    )];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "solid");
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].entered_solid, 1);
    assert_eq!(out[0].reemerged, 0);
    assert!(close(out[0].thickness, max_distance));
}

#[test]
fn degenerate_normal_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = slab_field(0.5);
    let gpu = GpuSdfThickness::new(&ctx);
    let queries = vec![SdfThicknessQuery::new(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        3.0,
        0.05,
        100,
    )];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "degenerate");
    assert_eq!(out[0].valid, 0);
    assert_eq!(out[0].entered_solid, 0);
    assert_eq!(out[0].reemerged, 0);
    assert_eq!(out[0].thickness.to_bits(), 0);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = slab_field(0.5);
    let gpu = GpuSdfThickness::new(&ctx);
    // One dispatch mixing a penetrating probe, a free-space probe and a
    // degenerate normal against the same field.
    let queries = vec![
        SdfThicknessQuery::new([1.02, 0.0, 0.0], [1.0, 0.0, 0.0], 3.0, 0.05, 100),
        SdfThicknessQuery::new([1.5, 0.0, 0.0], [-1.0, 0.0, 0.0], 2.0, 0.1, 100),
        SdfThicknessQuery::new([0.6, 0.0, 0.0], [0.0, 0.0, 0.0], 3.0, 0.05, 100),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "mixed");
    assert_eq!(out[0].reemerged, 1);
    assert_eq!(out[1].entered_solid, 0);
    assert_eq!(out[2].valid, 0);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfThickness::new(&ctx);
    let mut rng = Lcg::new(0x7E57_1234);
    let mut built = 0u32;
    let mut guard = 0u32;
    while built < 512 {
        guard += 1;
        assert!(guard < 2_000_000, "rejection sampling failed to converge");
        let dims = [
            2 + (rng.next_u32() % 4),
            2 + (rng.next_u32() % 4),
            2 + (rng.next_u32() % 4),
        ];
        let voxel = rng.next_range(0.3, 1.0);
        let origin = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        let count = (dims[0] * dims[1] * dims[2]) as usize;
        let mut dist = Vec::with_capacity(count);
        for _ in 0..count {
            dist.push(rng.next_range(-2.0, 2.0));
        }
        // Random start inside the field extent.
        let position = [
            origin[0] + rng.next_range(0.0, (dims[0] - 1) as f32) * voxel,
            origin[1] + rng.next_range(0.0, (dims[1] - 1) as f32) * voxel,
            origin[2] + rng.next_range(0.0, (dims[2] - 1) as f32) * voxel,
        ];
        // Random direction kept firmly non-degenerate (len^2 well above guard).
        let normal = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        let len2 = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
        if len2 < 0.25 {
            continue;
        }
        let max_distance = rng.next_range(0.5, 2.5);
        let step = rng.next_range(0.08, 0.2);
        let max_steps = 64u32;
        let probe = march(
            dims,
            origin,
            voxel,
            &dist,
            position,
            normal,
            max_distance,
            step,
            max_steps,
        );
        // Reject fixtures whose marched samples skim the sign boundary so the
        // per-step solid decision agrees on both sides.
        if probe.min_abs < SWEEP_MARGIN {
            continue;
        }
        let field = SdfThicknessField::new(dims, origin, voxel, dist);
        let queries = [SdfThicknessQuery::new(
            position,
            normal,
            max_distance,
            step,
            max_steps,
        )];
        let out = gpu.evaluate(&ctx, &field, &queries);
        assert_parity(&out, &field, &queries, "sweep");
        built += 1;
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = slab_field(0.5);
    let gpu = GpuSdfThickness::new(&ctx);
    let out = gpu.evaluate(&ctx, &field, &[]);
    assert!(out.is_empty(), "empty batch must return no results");
}
