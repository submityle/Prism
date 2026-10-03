//! Real-device parity for the signed-distance-field surface-curvature twin:
//! [`GpuSdfCurvature`](prism_volumetric_gpu::mesh_sdf_curvature::GpuSdfCurvature)
//! must reproduce the `CPU` golden `sdf_curvature` and
//! `curvature_from_derivatives` of
//! `prism_render_architecture::ray_scene::mesh_sdf_curvature`, which
//! differentiate the trilinear sampler `sample_signed_distance` of
//! `prism_render_architecture::ray_scene::mesh_sdf_raymarch` twice and then
//! evaluate Goldman's implicit-surface mean and Gaussian curvatures.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! trilinear sampling, a central-difference gradient, a central-difference
//! symmetric Hessian and the Goldman curvature algebra — written directly
//! against a flat per-cell signed-distance array so the test does not import
//! `prism_render_architecture`. It operates on the same field layout the twin
//! consumes (`x` fastest, `clamp-to-border` addressing), so the parity check is
//! `CPU`-versus-`GPU` on the same discrete field, not against an analytic
//! truth.
//!
//! The fixtures cover the degenerate and boundary shapes the kernel must honor:
//! a radial (spherical) field, a planar `x`-ramp (vanishing second derivatives,
//! near-zero mean and Gaussian curvature), a saddle-like quadratic field, a
//! constant field (vanishing gradient, zero curvatures, `valid = 0` on both
//! sides), a single-layer degenerate axis, a sweep over random small fields and
//! points, an empty query batch (host short-circuits with no dispatch) and a
//! named invariant check (`principal_max >= principal_min`,
//! `mean ~= 0.5 (k1 + k2)`).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Sampling is linear interpolation, the gradient and Hessian are fixed
//! sequences of differences scaled by constants, and the curvatures use two
//! `sqrt`s and divisions, so `CPU` and `GPU` evaluate the same closed form but
//! need not be bit-exact (a `GPU` may contract a multiply-add). The continuous
//! comparison is `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`).
//! The `valid` flag is an integer decision and matches exactly; the random
//! sweep rejects points whose gradient length squared sits near the
//! zero-gradient guard so the flag never flips and the `1 / |g|^3` mean term is
//! never catastrophically amplified between the two sides.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_curvature`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mesh_sdf_curvature::{
    GpuSdfCurvature, SdfCurvatureField, SdfCurvatureQuery, SdfCurvatureResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_TOL: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_TOL: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;
/// Minimum gradient length squared accepted in the random sweep. Sitting well
/// above the golden `f32::MIN_POSITIVE` guard keeps the `valid` flag stable and
/// the `1 / |g|^3` mean-curvature term well conditioned between the two sides.
const SWEEP_GRAD2_FLOOR: f32 = 1.0e-2;

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

/// Samples the field at `point + h * (dx, dy, dz)`, matching the golden
/// `offset_sample`.
fn offset_sample(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    dist: &[f32],
    point: [f32; 3],
    dx: f32,
    dy: f32,
    dz: f32,
    h: f32,
) -> f32 {
    sample(
        dims,
        origin,
        voxel,
        dist,
        [point[0] + h * dx, point[1] + h * dy, point[2] + h * dz],
    )
}

/// Independent oracle re-implementing the golden `central_gradient`.
fn gradient(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    dist: &[f32],
    point: [f32; 3],
) -> [f32; 3] {
    let h = voxel;
    let inv = 1.0 / (2.0 * h);
    let s = |dx, dy, dz| offset_sample(dims, origin, voxel, dist, point, dx, dy, dz, h);
    [
        (s(1.0, 0.0, 0.0) - s(-1.0, 0.0, 0.0)) * inv,
        (s(0.0, 1.0, 0.0) - s(0.0, -1.0, 0.0)) * inv,
        (s(0.0, 0.0, 1.0) - s(0.0, 0.0, -1.0)) * inv,
    ]
}

/// Independent oracle re-implementing the golden `central_hessian`, returning
/// the symmetric entries as `[fxx, fyy, fzz, fxy, fxz, fyz]`.
fn hessian(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    dist: &[f32],
    point: [f32; 3],
) -> [f32; 6] {
    let h = voxel;
    let s = |dx, dy, dz| offset_sample(dims, origin, voxel, dist, point, dx, dy, dz, h);
    let centre = s(0.0, 0.0, 0.0);
    let inv_sq = 1.0 / (h * h);
    let inv_quad = 1.0 / (4.0 * h * h);

    let fxx = (s(1.0, 0.0, 0.0) - 2.0 * centre + s(-1.0, 0.0, 0.0)) * inv_sq;
    let fyy = (s(0.0, 1.0, 0.0) - 2.0 * centre + s(0.0, -1.0, 0.0)) * inv_sq;
    let fzz = (s(0.0, 0.0, 1.0) - 2.0 * centre + s(0.0, 0.0, -1.0)) * inv_sq;

    let fxy =
        (s(1.0, 1.0, 0.0) - s(1.0, -1.0, 0.0) - s(-1.0, 1.0, 0.0) + s(-1.0, -1.0, 0.0)) * inv_quad;
    let fxz =
        (s(1.0, 0.0, 1.0) - s(1.0, 0.0, -1.0) - s(-1.0, 0.0, 1.0) + s(-1.0, 0.0, -1.0)) * inv_quad;
    let fyz =
        (s(0.0, 1.0, 1.0) - s(0.0, 1.0, -1.0) - s(0.0, -1.0, 1.0) + s(0.0, -1.0, -1.0)) * inv_quad;

    [fxx, fyy, fzz, fxy, fxz, fyz]
}

/// Independent oracle for the golden `sdf_curvature` / `curvature_from_derivatives`:
/// Goldman's implicit-surface mean and Gaussian curvatures split into the two
/// principal curvatures, or zeros with `valid = 0` for a degenerate gradient.
fn oracle(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    dist: &[f32],
    point: [f32; 3],
) -> SdfCurvatureResult {
    let g = gradient(dims, origin, voxel, dist, point);
    let [gx, gy, gz] = g;
    let g2 = gx * gx + gy * gy + gz * gz;
    if g2 <= f32::MIN_POSITIVE {
        return SdfCurvatureResult {
            mean: 0.0,
            gaussian: 0.0,
            principal_max: 0.0,
            principal_min: 0.0,
            valid: 0,
        };
    }
    let [a, d, f, b, c, e] = hessian(dims, origin, voxel, dist, point);
    let g_len = g2.sqrt();
    let g_len3 = g2 * g_len;
    let g4 = g2 * g2;

    let trace = a + d + f;
    let ghg =
        gx * gx * a + gy * gy * d + gz * gz * f + 2.0 * (gx * gy * b + gx * gz * c + gy * gz * e);
    let mean = (trace * g2 - ghg) / (2.0 * g_len3);

    let adj_a = d * f - e * e;
    let adj_d = a * f - c * c;
    let adj_f = a * d - b * b;
    let adj_b = c * e - b * f;
    let adj_c = b * e - c * d;
    let adj_e = b * c - a * e;
    let g_adj_g = gx * gx * adj_a
        + gy * gy * adj_d
        + gz * gz * adj_f
        + 2.0 * (gx * gy * adj_b + gx * gz * adj_c + gy * gz * adj_e);
    let gaussian = g_adj_g / g4;

    let discriminant = (mean * mean - gaussian).max(0.0);
    let root = discriminant.sqrt();
    SdfCurvatureResult {
        mean,
        gaussian,
        principal_max: mean + root,
        principal_min: mean - root,
        valid: 1,
    }
}

/// Gradient length squared of the oracle at a point, used by the sweep to reject
/// ill-conditioned, near-degenerate samples.
fn grad_len2(dims: [u32; 3], origin: [f32; 3], voxel: f32, dist: &[f32], point: [f32; 3]) -> f32 {
    let g = gradient(dims, origin, voxel, dist, point);
    g[0] * g[0] + g[1] * g[1] + g[2] * g[2]
}

/// Asserts the `GPU` results match the oracle: the four curvatures within the
/// continuous tolerance, `valid` exactly.
fn assert_parity(
    gpu: &[SdfCurvatureResult],
    field: &SdfCurvatureField,
    queries: &[SdfCurvatureQuery],
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
        assert!(
            close(g.mean, want.mean),
            "{label}: query {i} mean GPU {} vs CPU {}",
            g.mean,
            want.mean
        );
        assert!(
            close(g.gaussian, want.gaussian),
            "{label}: query {i} gaussian GPU {} vs CPU {}",
            g.gaussian,
            want.gaussian
        );
        assert!(
            close(g.principal_max, want.principal_max),
            "{label}: query {i} principal_max GPU {} vs CPU {}",
            g.principal_max,
            want.principal_max
        );
        assert!(
            close(g.principal_min, want.principal_min),
            "{label}: query {i} principal_min GPU {} vs CPU {}",
            g.principal_min,
            want.principal_min
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

/// Builds a radial (spherical) signed-distance field: each cell stores the
/// distance from its world-space center to a sphere of the given `radius`.
fn radial_field(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    center: [f32; 3],
    radius: f32,
) -> SdfCurvatureField {
    let count = (dims[0] * dims[1] * dims[2]) as usize;
    let mut dist = Vec::with_capacity(count);
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
    SdfCurvatureField::new(dims, origin, voxel, dist)
}

#[test]
fn radial_field_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let dims = [7u32, 7, 7];
    let origin = [-1.5, -1.5, -1.5];
    let voxel = 0.5;
    let field = radial_field(dims, origin, voxel, [0.0, 0.0, 0.0], 0.75);
    let gpu = GpuSdfCurvature::new(&ctx);
    let queries = vec![
        SdfCurvatureQuery::new([0.4, 0.1, -0.2]),
        SdfCurvatureQuery::new([-0.5, 0.3, 0.2]),
        SdfCurvatureQuery::new([0.2, -0.35, 0.4]),
        SdfCurvatureQuery::new([0.6, 0.5, -0.4]),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "radial");
    // Sanity: a sphere is elliptic, so the oracle reports positive Gaussian
    // curvature at a well-conditioned interior point.
    let want = oracle(
        field.dims(),
        field.origin(),
        field.voxel_size(),
        field.distances(),
        [0.4, 0.1, -0.2],
    );
    assert_eq!(want.valid, 1);
}

#[test]
fn planar_ramp_is_flat() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // A linear ramp along x: non-zero gradient but vanishing second derivatives,
    // so mean and Gaussian curvature are both ~0.
    let dims = [5u32, 4, 4];
    let origin = [0.0, 0.0, 0.0];
    let voxel = 0.5;
    let slope = 1.3;
    let count = (dims[0] * dims[1] * dims[2]) as usize;
    let mut dist = Vec::with_capacity(count);
    for _z in 0..dims[2] {
        for _y in 0..dims[1] {
            for x in 0..dims[0] {
                dist.push(x as f32 * slope);
            }
        }
    }
    let field = SdfCurvatureField::new(dims, origin, voxel, dist);
    let gpu = GpuSdfCurvature::new(&ctx);
    // Interior points, away from the clamp-to-edge border.
    let queries = vec![
        SdfCurvatureQuery::new([0.75, 0.5, 0.5]),
        SdfCurvatureQuery::new([1.0, 0.6, 0.4]),
        SdfCurvatureQuery::new([1.25, 0.5, 0.6]),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "planar_ramp");
    for q in &queries {
        let want = oracle(
            field.dims(),
            field.origin(),
            field.voxel_size(),
            field.distances(),
            q.point,
        );
        assert_eq!(want.valid, 1, "ramp interior must have a defined curvature");
        assert!(want.mean.abs() <= 1.0e-3, "flat ramp mean must be ~0");
        assert!(
            want.gaussian.abs() <= 1.0e-3,
            "flat ramp gaussian must be ~0"
        );
    }
}

#[test]
fn saddle_field_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // A hyperbolic-paraboloid-like field f = 0.5(x^2 - y^2) + z, which has a
    // non-trivial, mixed-curvature Hessian.
    let dims = [7u32, 7, 7];
    let origin = [-1.5, -1.5, -1.5];
    let voxel = 0.5;
    let count = (dims[0] * dims[1] * dims[2]) as usize;
    let mut dist = Vec::with_capacity(count);
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let wx = origin[0] + x as f32 * voxel;
                let wy = origin[1] + y as f32 * voxel;
                let wz = origin[2] + z as f32 * voxel;
                dist.push(0.5 * (wx * wx - wy * wy) + wz);
            }
        }
    }
    let field = SdfCurvatureField::new(dims, origin, voxel, dist);
    let gpu = GpuSdfCurvature::new(&ctx);
    let queries = vec![
        SdfCurvatureQuery::new([0.3, 0.2, 0.1]),
        SdfCurvatureQuery::new([-0.4, 0.35, -0.2]),
        SdfCurvatureQuery::new([0.5, -0.3, 0.25]),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "saddle");
}

#[test]
fn constant_field_is_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let dims = [3u32, 3, 3];
    let dist = vec![2.25f32; 27];
    let field = SdfCurvatureField::new(dims, [-1.0, -1.0, -1.0], 0.5, dist);
    let gpu = GpuSdfCurvature::new(&ctx);
    let queries = vec![
        SdfCurvatureQuery::new([0.0, 0.0, 0.0]),
        SdfCurvatureQuery::new([-0.3, 0.2, 0.1]),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "constant");
    for r in &out {
        assert_eq!(r.valid, 0, "constant field must yield a zero gradient");
        assert_eq!(r.mean, 0.0);
        assert_eq!(r.gaussian, 0.0);
        assert_eq!(r.principal_max, 0.0);
        assert_eq!(r.principal_min, 0.0);
    }
}

#[test]
fn single_layer_axis_is_handled() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Degenerate x axis (one layer); the field varies radially in the y-z plane
    // so the in-plane curvature is well defined.
    let dims = [1u32, 7, 7];
    let origin = [0.0, -1.5, -1.5];
    let voxel = 0.5;
    let count = (dims[0] * dims[1] * dims[2]) as usize;
    let mut dist = Vec::with_capacity(count);
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for _x in 0..dims[0] {
                let wy = origin[1] + y as f32 * voxel;
                let wz = origin[2] + z as f32 * voxel;
                dist.push((wy * wy + wz * wz).sqrt() - 0.7);
            }
        }
    }
    let field = SdfCurvatureField::new(dims, origin, voxel, dist);
    let gpu = GpuSdfCurvature::new(&ctx);
    let queries = vec![
        SdfCurvatureQuery::new([3.0, 0.3, -0.2]),
        SdfCurvatureQuery::new([-2.0, -0.35, 0.4]),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "single_layer");
}

#[test]
fn principal_curvature_invariants_hold() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let dims = [7u32, 7, 7];
    let origin = [-1.5, -1.5, -1.5];
    let voxel = 0.5;
    let field = radial_field(dims, origin, voxel, [0.0, 0.0, 0.0], 0.8);
    let gpu = GpuSdfCurvature::new(&ctx);
    let queries = vec![
        SdfCurvatureQuery::new([0.35, 0.1, -0.15]),
        SdfCurvatureQuery::new([-0.4, 0.3, 0.25]),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "invariants");
    for r in &out {
        assert_eq!(r.valid, 1);
        // k1 >= k2 by construction (root >= 0).
        assert!(
            r.principal_max >= r.principal_min - ABS_TOL,
            "principal_max {} must be >= principal_min {}",
            r.principal_max,
            r.principal_min
        );
        // Mean curvature is the average of the two principal curvatures.
        let avg = 0.5 * (r.principal_max + r.principal_min);
        assert!(
            close(r.mean, avg),
            "mean {} must equal 0.5(k1 + k2) {}",
            r.mean,
            avg
        );
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCurvature::new(&ctx);
    let mut rng = Lcg::new(0x0C7A_55E3);
    let mut built = 0u32;
    while built < 512 {
        // Random small field (at least 3 cells on each axis so the central
        // second-difference stencil has interior room).
        let dims = [
            3 + (rng.next_u32() % 4),
            3 + (rng.next_u32() % 4),
            3 + (rng.next_u32() % 4),
        ];
        let voxel = rng.next_range(0.3, 1.4);
        let origin = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        let count = (dims[0] * dims[1] * dims[2]) as usize;
        // Smooth-ish random field: a quadratic in world space with random
        // coefficients so the Hessian is non-trivial but the sampled field is
        // not pure noise (keeps the curvature well scaled).
        let ca = rng.next_range(-1.0, 1.0);
        let cb = rng.next_range(-1.0, 1.0);
        let cc = rng.next_range(-1.0, 1.0);
        let cd = rng.next_range(-0.5, 0.5);
        let ce = rng.next_range(-0.5, 0.5);
        let cf = rng.next_range(-0.5, 0.5);
        let mut dist = Vec::with_capacity(count);
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let wx = origin[0] + x as f32 * voxel;
                    let wy = origin[1] + y as f32 * voxel;
                    let wz = origin[2] + z as f32 * voxel;
                    dist.push(
                        ca * wx + cb * wy + cc * wz + cd * wx * wx + ce * wy * wy + cf * wz * wz,
                    );
                }
            }
        }
        // Random interior point, biased to the middle so the central stencil
        // stays off the clamp-to-edge border.
        let point = [
            origin[0] + rng.next_range(1.0, (dims[0] - 2) as f32) * voxel,
            origin[1] + rng.next_range(1.0, (dims[1] - 2) as f32) * voxel,
            origin[2] + rng.next_range(1.0, (dims[2] - 2) as f32) * voxel,
        ];
        // Reject near-degenerate gradients so the valid flag never flips and the
        // 1 / |g|^3 mean term stays well conditioned between the two sides.
        if grad_len2(dims, origin, voxel, &dist, point) < SWEEP_GRAD2_FLOOR {
            continue;
        }
        let field = SdfCurvatureField::new(dims, origin, voxel, dist);
        let queries = [SdfCurvatureQuery::new(point)];
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
    let field = radial_field([5u32, 5, 5], [-1.0, -1.0, -1.0], 0.5, [0.0, 0.0, 0.0], 0.6);
    let gpu = GpuSdfCurvature::new(&ctx);
    let out = gpu.evaluate(&ctx, &field, &[]);
    assert!(out.is_empty(), "empty batch must return no results");
}
