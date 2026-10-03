//! Real-device parity for the signed-distance-field ambient-occlusion twin:
//! [`GpuSdfAmbientOcclusion`](prism_volumetric_gpu::mesh_sdf_ambient_occlusion::GpuSdfAmbientOcclusion)
//! must reproduce the `CPU` golden `sdf_ambient_occlusion` of
//! `prism_render_architecture::ray_scene::mesh_sdf_ambient_occlusion`, which
//! marches a few taps along the surface normal through the trilinear sampler
//! `sample_signed_distance` of
//! `prism_render_architecture::ray_scene::mesh_sdf_raymarch` and maps the
//! decayed shortfall `(h - d)` to a clamped visibility factor.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! trilinear sampling, guarded normalization and the decayed tap accumulation —
//! written directly against a flat per-cell signed-distance array so the test
//! does not import `prism_render_architecture`. It operates on the same field
//! layout the twin consumes (`x` fastest, `clamp-to-border` addressing).
//!
//! The fixtures cover the degenerate and boundary shapes the kernel must honor:
//! a linear ground-plane field whose on-surface query stays fully visible, a
//! constant field whose point is strongly occluded, a degenerate zero-length
//! normal (`valid = 0` on both sides), a zero `sample_count` (no occlusion, full
//! visibility), a radial bowl field, a sweep over random small fields and
//! queries, and an empty batch (host short-circuits with no dispatch).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The estimator is a fixed sequence of linear-interpolation adds, a decayed
//! accumulation and one final `clamp`, apart from the single `sqrt` the normal
//! normalization performs, so `CPU` and `GPU` evaluate the same closed form but
//! need not be bit-exact (a `GPU` may contract a multiply-add). The continuous
//! comparison is `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`).
//! The `valid` flag is an integer decision and matches exactly; the random
//! sweep rejects degenerate normals so the flag never flips between the sides
//! and rejects clamp-saturated visibilities so the comparison stays in the
//! well-conditioned interior band.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_ambient_occlusion`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mesh_sdf_ambient_occlusion::{
    GpuSdfAmbientOcclusion, SdfAmbientOcclusionField, SdfAmbientOcclusionQuery,
    SdfAmbientOcclusionResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_TOL: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_TOL: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;
/// Minimum normal length squared accepted by the random sweep, kept well above
/// the golden `f32::MIN_POSITIVE` guard so the `valid` flag never flips between
/// the two sides of the parity comparison.
const SWEEP_NORMAL_FLOOR: f32 = 1.0e-2;
/// Interior visibility band accepted by the random sweep so the comparison
/// avoids the saturated ends of the final `clamp`.
const SWEEP_VIS_LO: f32 = 0.05;
/// Upper edge of the interior visibility band accepted by the random sweep.
const SWEEP_VIS_HI: f32 = 0.95;

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

/// Independent oracle re-implementing the golden `sdf_ambient_occlusion`: a
/// guarded normalization, the decayed tap accumulation and the final clamped
/// visibility, or a zero-visibility invalid result when the normal is
/// degenerate.
fn oracle(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    dist: &[f32],
    query: &SdfAmbientOcclusionQuery,
) -> SdfAmbientOcclusionResult {
    let n = query.normal;
    let len2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    if len2 <= f32::MIN_POSITIVE {
        return SdfAmbientOcclusionResult {
            visibility: 0.0,
            valid: 0,
        };
    }
    let len = len2.sqrt();
    let nn = [n[0] / len, n[1] / len, n[2] / len];
    let mut occlusion = 0.0f32;
    let mut weight = 1.0f32;
    for i in 0..query.sample_count {
        let h = query.step * (i as f32 + 1.0);
        let tap = [
            query.position[0] + nn[0] * h,
            query.position[1] + nn[1] * h,
            query.position[2] + nn[2] * h,
        ];
        let d = sample(dims, origin, voxel, dist, tap);
        occlusion += (h - d) * weight;
        weight *= query.decay;
    }
    let visibility = (1.0 - query.strength * occlusion).clamp(0.0, 1.0);
    SdfAmbientOcclusionResult {
        visibility,
        valid: 1,
    }
}

/// Asserts the `GPU` results match the oracle: visibility within the continuous
/// tolerance, `valid` exactly.
fn assert_parity(
    gpu: &[SdfAmbientOcclusionResult],
    field: &SdfAmbientOcclusionField,
    queries: &[SdfAmbientOcclusionQuery],
    label: &str,
) {
    assert_eq!(gpu.len(), queries.len(), "{label}: result count mismatch");
    for (i, (g, q)) in gpu.iter().zip(queries.iter()).enumerate() {
        let want = oracle(
            field.dims(),
            field.origin(),
            field.voxel_size(),
            field.distances(),
            q,
        );
        assert_eq!(g.valid, want.valid, "{label}: query {i} valid flag");
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

/// Builds a field whose stored distance rises linearly along `y`, so the
/// trilinear sample grows exactly one unit of distance per world unit climbed:
/// an on-surface query stepping along `+y` sees `d` track `h` and stays fully
/// visible.
fn plane_y_field() -> SdfAmbientOcclusionField {
    let dims = [3u32, 9, 3];
    let origin = [-0.5, 0.0, -0.5];
    let voxel = 0.5;
    let mut dist = Vec::with_capacity((dims[0] * dims[1] * dims[2]) as usize);
    for _z in 0..dims[2] {
        for y in 0..dims[1] {
            for _x in 0..dims[0] {
                // Stored distance equals the cell's world height, so the sample
                // reconstructs a plane distance (offset by half a voxel).
                dist.push(y as f32 * voxel);
            }
        }
    }
    SdfAmbientOcclusionField::new(dims, origin, voxel, dist)
}

/// Builds a constant field: every tap reports the same distance, so stepping
/// out accumulates the full `(h - d)` shortfall and the point is strongly
/// occluded.
fn constant_field(value: f32) -> SdfAmbientOcclusionField {
    let dims = [3u32, 3, 3];
    let dist = vec![value; (dims[0] * dims[1] * dims[2]) as usize];
    SdfAmbientOcclusionField::new(dims, [-0.5, -0.5, -0.5], 0.5, dist)
}

/// Builds a radial bowl field whose stored distance is the Euclidean distance
/// to a sphere, giving a smoothly varying sample for the parity sweep.
fn radial_field() -> SdfAmbientOcclusionField {
    let dims = [6u32, 6, 6];
    let origin = [-1.25, -1.25, -1.25];
    let voxel = 0.5;
    let center = [0.0f32, 0.0, 0.0];
    let radius = 0.75f32;
    let mut dist = Vec::with_capacity((dims[0] * dims[1] * dims[2]) as usize);
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let wx = origin[0] + x as f32 * voxel - center[0];
                let wy = origin[1] + y as f32 * voxel - center[1];
                let wz = origin[2] + z as f32 * voxel - center[2];
                let r = (wx * wx + wy * wy + wz * wz).sqrt();
                dist.push(r - radius);
            }
        }
    }
    SdfAmbientOcclusionField::new(dims, origin, voxel, dist)
}

#[test]
fn open_space_point_is_fully_visible() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = plane_y_field();
    let gpu = GpuSdfAmbientOcclusion::new(&ctx);
    // On-surface query: `d(tap)` tracks `h` so the shortfall is zero and the
    // visibility stays saturated at one.
    let query = SdfAmbientOcclusionQuery::new([0.0, 0.25, 0.0], [0.0, 1.0, 0.0], 5, 0.3, 0.5, 1.0);
    let queries = vec![query];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "open_space");
    let want = oracle(
        field.dims(),
        field.origin(),
        field.voxel_size(),
        field.distances(),
        &query,
    );
    assert!(
        close(want.visibility, 1.0),
        "open-space oracle visibility should be ~1, got {}",
        want.visibility
    );
    assert_eq!(out[0].valid, 1);
}

#[test]
fn occluded_point_is_darkened() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = constant_field(0.0);
    let gpu = GpuSdfAmbientOcclusion::new(&ctx);
    // A constant (flat) field never grows with `h`, so the shortfall piles up
    // and the point is fully occluded.
    let query = SdfAmbientOcclusionQuery::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 5, 0.3, 0.5, 1.0);
    let queries = vec![query];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "occluded");
    let want = oracle(
        field.dims(),
        field.origin(),
        field.voxel_size(),
        field.distances(),
        &query,
    );
    assert!(
        want.visibility < 0.5,
        "occluded oracle visibility should be strongly darkened, got {}",
        want.visibility
    );
    assert_eq!(out[0].valid, 1);
}

#[test]
fn degenerate_normal_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = radial_field();
    let gpu = GpuSdfAmbientOcclusion::new(&ctx);
    let queries = vec![
        SdfAmbientOcclusionQuery::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 4, 0.25, 0.6, 1.0),
        SdfAmbientOcclusionQuery::new([0.2, -0.1, 0.3], [0.0, 0.0, 0.0], 3, 0.4, 0.5, 0.8),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "degenerate_normal");
    for r in &out {
        assert_eq!(r.valid, 0, "zero normal must be reported invalid");
        assert!(
            close(r.visibility, 0.0),
            "degenerate query must report zero visibility, got {}",
            r.visibility
        );
    }
}

#[test]
fn zero_samples_is_fully_visible() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = radial_field();
    let gpu = GpuSdfAmbientOcclusion::new(&ctx);
    // No taps means no accumulated occlusion, so visibility is one and the
    // normal is still oriented (valid).
    let query = SdfAmbientOcclusionQuery::new([0.1, 0.2, -0.1], [1.0, 0.0, 0.0], 0, 0.3, 0.5, 1.0);
    let queries = vec![query];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "zero_samples");
    assert_eq!(out[0].valid, 1);
    assert!(
        close(out[0].visibility, 1.0),
        "zero-sample query must stay fully visible, got {}",
        out[0].visibility
    );
}

#[test]
fn single_layer_axis_is_handled() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Degenerate x axis (one layer); the sampler must contribute no weight on
    // that axis and the twin must still match the oracle.
    let dims = [1u32, 5, 5];
    let origin = [0.0, 0.0, 0.0];
    let voxel = 0.5;
    let mut dist = Vec::with_capacity((dims[0] * dims[1] * dims[2]) as usize);
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for _x in 0..dims[0] {
                dist.push(y as f32 * 0.4 + z as f32 * 0.2);
            }
        }
    }
    let field = SdfAmbientOcclusionField::new(dims, origin, voxel, dist);
    let gpu = GpuSdfAmbientOcclusion::new(&ctx);
    let queries = vec![
        SdfAmbientOcclusionQuery::new([5.0, 0.6, 0.5], [0.0, 1.0, 0.0], 4, 0.25, 0.6, 0.5),
        SdfAmbientOcclusionQuery::new([0.0, 0.9, 0.8], [0.0, 0.0, 1.0], 3, 0.3, 0.5, 0.4),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "single_layer");
}

#[test]
fn radial_field_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = radial_field();
    let gpu = GpuSdfAmbientOcclusion::new(&ctx);
    let queries = vec![
        SdfAmbientOcclusionQuery::new([0.3, 0.1, -0.2], [0.4, 0.8, 0.1], 5, 0.25, 0.6, 0.5),
        SdfAmbientOcclusionQuery::new([-0.4, 0.25, 0.15], [-0.2, 0.9, 0.3], 4, 0.3, 0.5, 0.4),
        SdfAmbientOcclusionQuery::new([0.2, -0.3, 0.35], [0.1, -0.3, 0.95], 6, 0.2, 0.7, 0.3),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "radial");
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfAmbientOcclusion::new(&ctx);
    let mut rng = Lcg::new(0x0C7A_51D5);
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
            dist.push(rng.next_range(-1.5, 1.5));
        }
        // Interior-ish query point within the field extent.
        let point = [
            origin[0] + rng.next_range(0.0, (dims[0] - 1) as f32) * voxel,
            origin[1] + rng.next_range(0.0, (dims[1] - 1) as f32) * voxel,
            origin[2] + rng.next_range(0.0, (dims[2] - 1) as f32) * voxel,
        ];
        // Non-degenerate normal, kept well clear of the normalization guard so
        // the valid flag never flips between the two sides.
        let normal = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        let len2 = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
        if len2 < SWEEP_NORMAL_FLOOR {
            continue;
        }
        let sample_count = 1 + (rng.next_u32() % 6);
        let step = rng.next_range(0.1, 0.5);
        let decay = rng.next_range(0.3, 0.9);
        let strength = rng.next_range(0.05, 0.5);
        let query =
            SdfAmbientOcclusionQuery::new(point, normal, sample_count, step, decay, strength);
        let want = oracle(dims, origin, voxel, &dist, &query);
        // Keep the comparison in the well-conditioned interior band, away from
        // the saturated ends of the final clamp.
        if want.visibility < SWEEP_VIS_LO || want.visibility > SWEEP_VIS_HI {
            continue;
        }
        let field = SdfAmbientOcclusionField::new(dims, origin, voxel, dist);
        let queries = [query];
        let out = gpu.evaluate(&ctx, &field, &queries);
        assert_parity(&out, &field, &queries, "sweep");
        built += 1;
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = radial_field();
    let gpu = GpuSdfAmbientOcclusion::new(&ctx);
    let out = gpu.evaluate(&ctx, &field, &[]);
    assert!(out.is_empty(), "empty batch must return no results");
}
