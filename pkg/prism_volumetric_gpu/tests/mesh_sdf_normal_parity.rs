//! Real-device parity for the signed-distance-field surface-normal twin:
//! [`GpuSdfNormal`](prism_volumetric_gpu::mesh_sdf_normal::GpuSdfNormal) must
//! reproduce the `CPU` golden `sdf_gradient` and `sdf_normal` of
//! `prism_render_architecture::ray_scene::mesh_sdf_normal`, which differentiate
//! the trilinear sampler `sample_signed_distance` of
//! `prism_render_architecture::ray_scene::mesh_sdf_raymarch` with a symmetric
//! central difference and then guard the normalization.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! trilinear sampling, central-difference gradient and guarded normalization —
//! written directly against a flat per-cell signed-distance array so the test
//! does not import `prism_render_architecture`. It operates on the same field
//! layout the twin consumes (`x` fastest, `clamp-to-border` addressing).
//!
//! The fixtures cover the degenerate and boundary shapes the kernel must honor:
//! an `x`-ramp field (dominant `x` gradient, oriented normal), a constant field
//! (vanishing gradient, zero normal, `valid = 0` on both sides), a single-layer
//! degenerate axis, a radial bowl field, a sweep over random small fields and
//! points, and an empty query batch (host short-circuits with no dispatch).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The gradient and normal are a fixed sequence of linear-interpolation adds, a
//! constant scale and one `sqrt`, so `CPU` and `GPU` evaluate the same closed
//! form but need not be bit-exact (a `GPU` may contract a multiply-add). The
//! continuous comparison is `abs_diff <= 1e-4 || rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`). The `valid` flag is an integer decision and matches
//! exactly; the random sweep rejects points whose gradient length squared sits
//! near the zero-gradient guard so the flag never flips between the sides.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_normal`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mesh_sdf_normal::{
    GpuSdfNormal, SdfNormalField, SdfNormalQuery, SdfNormalResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_TOL: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_TOL: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;
/// Rejection margin on the gradient length squared for the random sweep, so a
/// fixture never straddles the golden `f32::MIN_POSITIVE` zero-gradient guard.
const SWEEP_GUARD: f32 = 1.0e-6;

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

/// Independent oracle re-implementing the golden `sdf_gradient` central
/// difference against a flat per-cell signed-distance array.
fn gradient(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    dist: &[f32],
    point: [f32; 3],
) -> [f32; 3] {
    let inv = 1.0 / (2.0 * voxel);
    let mut g = [0f32; 3];
    for (axis, slot) in g.iter_mut().enumerate() {
        let mut forward = point;
        let mut backward = point;
        forward[axis] += voxel;
        backward[axis] -= voxel;
        let diff = sample(dims, origin, voxel, dist, forward)
            - sample(dims, origin, voxel, dist, backward);
        *slot = diff * inv;
    }
    g
}

/// Independent oracle for the golden `sdf_normal`: the normalized gradient, or a
/// zero normal with `valid = 0` when the gradient is too short to orient.
fn oracle(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    dist: &[f32],
    point: [f32; 3],
) -> SdfNormalResult {
    let g = gradient(dims, origin, voxel, dist, point);
    let len2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    if len2 <= f32::MIN_POSITIVE {
        return SdfNormalResult {
            gradient: g,
            normal: [0.0, 0.0, 0.0],
            valid: 0,
        };
    }
    let len = len2.sqrt();
    SdfNormalResult {
        gradient: g,
        normal: [g[0] / len, g[1] / len, g[2] / len],
        valid: 1,
    }
}

/// Asserts the `GPU` results match the oracle: gradient and normal within the
/// continuous tolerance, `valid` exactly.
fn assert_parity(
    gpu: &[SdfNormalResult],
    field: &SdfNormalField,
    queries: &[SdfNormalQuery],
    label: &str,
) {
    assert_eq!(gpu.len(), queries.len(), "{label}: result count mismatch");
    for (i, (g, q)) in gpu.iter().zip(queries.iter()).enumerate() {
        let want = oracle(
            field.dims(),
            field.origin(),
            field.voxel_size(),
            field.distances(),
            q.point,
        );
        assert_eq!(g.valid, want.valid, "{label}: query {i} valid flag");
        for k in 0..3 {
            assert!(
                close(g.gradient[k], want.gradient[k]),
                "{label}: query {i} gradient[{k}] GPU {} vs CPU {}",
                g.gradient[k],
                want.gradient[k]
            );
            assert!(
                close(g.normal[k], want.normal[k]),
                "{label}: query {i} normal[{k}] GPU {} vs CPU {}",
                g.normal[k],
                want.normal[k]
            );
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

/// Builds a field whose stored distance rises linearly along `x`, giving an
/// analytic gradient of `(slope, 0, 0)` everywhere in the interior.
fn ramp_x_field() -> SdfNormalField {
    let dims = [4u32, 3, 3];
    let origin = [0.0, 0.0, 0.0];
    let voxel = 0.5;
    let slope = 2.0;
    let mut dist = Vec::with_capacity(4 * 3 * 3);
    for _z in 0..dims[2] {
        for _y in 0..dims[1] {
            for x in 0..dims[0] {
                dist.push(x as f32 * slope);
            }
        }
    }
    SdfNormalField::new(dims, origin, voxel, dist)
}

#[test]
fn ramp_field_has_dominant_x_gradient() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = ramp_x_field();
    let gpu = GpuSdfNormal::new(&ctx);
    // Interior points, away from the border where clamp-to-edge flattens the
    // ramp, so the central difference sees the full slope on both taps.
    let queries = vec![
        SdfNormalQuery::new([0.75, 0.5, 0.5]),
        SdfNormalQuery::new([1.0, 0.6, 0.4]),
        SdfNormalQuery::new([0.6, 0.5, 0.5]),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "ramp_x");
    // Sanity: the oracle itself reports an oriented, x-dominant normal.
    for q in &queries {
        let want = oracle(
            field.dims(),
            field.origin(),
            field.voxel_size(),
            field.distances(),
            q.point,
        );
        assert_eq!(want.valid, 1);
        assert!(want.normal[0].abs() > want.normal[1].abs());
        assert!(want.normal[0].abs() > want.normal[2].abs());
    }
}

#[test]
fn constant_field_is_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let dims = [3u32, 3, 3];
    let dist = vec![1.5f32; 27];
    let field = SdfNormalField::new(dims, [-1.0, -1.0, -1.0], 0.5, dist);
    let gpu = GpuSdfNormal::new(&ctx);
    let queries = vec![
        SdfNormalQuery::new([0.0, 0.0, 0.0]),
        SdfNormalQuery::new([-0.3, 0.2, 0.1]),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "constant");
    for r in &out {
        assert_eq!(r.valid, 0, "constant field must yield a zero gradient");
        assert_eq!(r.normal, [0.0, 0.0, 0.0]);
    }
}

#[test]
fn single_layer_axis_is_handled() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Degenerate x axis (one layer), ramp along y so the gradient is oriented.
    let dims = [1u32, 4, 4];
    let origin = [0.0, 0.0, 0.0];
    let voxel = 0.5;
    let mut dist = Vec::with_capacity(16);
    for _z in 0..dims[2] {
        for y in 0..dims[1] {
            for _x in 0..dims[0] {
                dist.push(y as f32 * 1.5);
            }
        }
    }
    let field = SdfNormalField::new(dims, origin, voxel, dist);
    let gpu = GpuSdfNormal::new(&ctx);
    let queries = vec![
        SdfNormalQuery::new([0.0, 0.75, 0.5]),
        SdfNormalQuery::new([5.0, 1.0, 0.6]),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "single_layer");
}

#[test]
fn radial_field_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let dims = [5u32, 5, 5];
    let origin = [-1.0, -1.0, -1.0];
    let voxel = 0.5;
    let center = [0.0f32, 0.0, 0.0];
    let mut dist = Vec::with_capacity(125);
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
                dist.push(r - 0.75);
            }
        }
    }
    let field = SdfNormalField::new(dims, origin, voxel, dist);
    let gpu = GpuSdfNormal::new(&ctx);
    let queries = vec![
        SdfNormalQuery::new([0.3, 0.1, -0.2]),
        SdfNormalQuery::new([-0.4, 0.25, 0.15]),
        SdfNormalQuery::new([0.2, -0.3, 0.35]),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "radial");
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfNormal::new(&ctx);
    let mut rng = Lcg::new(0x51D5_3C01);
    let mut built = 0u32;
    while built < 512 {
        // Random small field.
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
            dist.push(rng.next_range(-2.0, 2.0));
        }
        // Random interior-ish point within the field extent plus a margin.
        let point = [
            origin[0] + rng.next_range(0.0, (dims[0] - 1) as f32) * voxel,
            origin[1] + rng.next_range(0.0, (dims[1] - 1) as f32) * voxel,
            origin[2] + rng.next_range(0.0, (dims[2] - 1) as f32) * voxel,
        ];
        let want = oracle(dims, origin, voxel, &dist, point);
        // Reject points sitting near the zero-gradient guard so the valid flag
        // never flips between the two sides of the parity comparison.
        let len2 = want.gradient[0] * want.gradient[0]
            + want.gradient[1] * want.gradient[1]
            + want.gradient[2] * want.gradient[2];
        if len2 > f32::MIN_POSITIVE && len2 < SWEEP_GUARD {
            continue;
        }
        let field = SdfNormalField::new(dims, origin, voxel, dist);
        let queries = [SdfNormalQuery::new(point)];
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
    let field = ramp_x_field();
    let gpu = GpuSdfNormal::new(&ctx);
    let out = gpu.evaluate(&ctx, &field, &[]);
    assert!(out.is_empty(), "empty batch must return no results");
}
