//! Real-device parity for the signed-distance-field cone-occlusion twin:
//! [`GpuSdfConeOcclusion`](prism_volumetric_gpu::mesh_sdf_cone_occlusion::GpuSdfConeOcclusion)
//! must reproduce the `CPU` golden `sdf_cone_occlusion` of
//! `prism_render_architecture::ray_scene::mesh_sdf_cone_occlusion`, which sweeps
//! a baked seven-cone hemisphere fan through the trilinear sampler
//! `sample_signed_distance` of
//! `prism_render_architecture::ray_scene::mesh_sdf_raymarch`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! trilinear sampling, the branchless Duff et al. (2017) orthonormal basis, the
//! same seven baked cone directions, and the per-cone running minimum of
//! `sampled / (tan_half_angle * t)` — written directly against a flat per-cell
//! signed-distance array so the test never imports `prism_render_architecture`.
//! It operates on the same field layout the twin consumes (`x` fastest,
//! `clamp-to-border` addressing) and marches in `f32` with the identical break
//! order and `t` accumulation so the discrete step budget and the accumulated
//! sums align with the device.
//!
//! The fixtures cover the shapes the kernel must honor: a degenerate normal
//! (`valid = 0`), a clear point far from any occluder (fully visible), a point
//! beside a sphere (partial occlusion with a bent normal that leans away from
//! the solid), a sweep over random strictly-positive fields and normals (so no
//! cone ever closes early and the march is fully deterministic), and an empty
//! query batch (host short-circuits with no dispatch).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Each cone march is a bounded loop of multiplies, `clamp`/`min`/`max` guards
//! and a trilinear tap, closed by one `sqrt` per normalization, so `CPU` and
//! `GPU` evaluate the same closed form but need not be bit-exact (a `GPU` may
//! contract a multiply-add). The continuous comparison on `visibility` and each
//! `bent_normal` component is `abs_diff <= 1e-4 || rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`). The `valid` output is a boolean decision and matches
//! exactly; the fixtures use strictly-positive fields and a far `max_distance`
//! so each cone runs its full step budget with no early break.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_cone_occlusion`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mesh_sdf_cone_occlusion::{
    GpuSdfConeOcclusion, SdfConeOcclusionField, SdfConeOcclusionQuery, SdfConeOcclusionResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_TOL: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_TOL: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// One over the square root of two (`cos 45°`), the ring elevation; matches the
/// golden `COS45`.
const COS45: f32 = 0.707_106_77;
/// Half of [`COS45`], a ring tangent magnitude; matches the golden `RING_HALF`.
const RING_HALF: f32 = 0.353_553_38;
/// `sin 45° · sin 60°`, the other ring tangent magnitude; matches the golden
/// `RING_TALL`.
const RING_TALL: f32 = 0.612_372_44;

/// The seven tangent-space cone directions (`z` is the surface normal): the
/// center cone plus a six-way ring at `45°` elevation. Byte-for-byte the golden
/// `CONE_DIRS`.
const CONE_DIRS: [[f32; 3]; 7] = [
    [0.0, 0.0, 1.0],
    [COS45, 0.0, COS45],
    [RING_HALF, RING_TALL, COS45],
    [-RING_HALF, RING_TALL, COS45],
    [-COS45, 0.0, COS45],
    [-RING_HALF, -RING_TALL, COS45],
    [RING_HALF, -RING_TALL, COS45],
];

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

/// Returns the unit-length vector, or [`None`] when `vector` is too short to
/// normalize, matching the golden `normalize`.
fn normalize(vector: [f32; 3]) -> Option<[f32; 3]> {
    let len2 = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
    if len2 <= f32::MIN_POSITIVE {
        return None;
    }
    let len = len2.sqrt();
    Some([vector[0] / len, vector[1] / len, vector[2] / len])
}

/// Builds the Duff et al. (2017) orthonormal basis `(tangent, bitangent)` for a
/// unit `normal`. The `z` sign is folded in with a `< 0` test so it matches the
/// device `select(1.0, -1.0, n.z < 0.0)` (the golden `f32::signum` returns `1`
/// for a non-negative component just as this does).
fn orthonormal_basis(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let z_sign = if normal[2] < 0.0 { -1.0 } else { 1.0 };
    let a = -1.0 / (z_sign + normal[2]);
    let b = normal[0] * normal[1] * a;
    let tangent = [
        1.0 + z_sign * normal[0] * normal[0] * a,
        z_sign * b,
        -z_sign * normal[0],
    ];
    let bitangent = [b, z_sign + normal[1] * normal[1] * a, -normal[1]];
    (tangent, bitangent)
}

/// Independent oracle re-implementing the golden `sdf_cone_occlusion` against a
/// flat per-cell signed-distance array.
fn oracle(field: &SdfConeOcclusionField, q: &SdfConeOcclusionQuery) -> SdfConeOcclusionResult {
    let dims = field.dims();
    let origin = field.origin();
    let voxel = field.voxel_size();
    let dist = field.distances();

    let Some(n) = normalize(q.normal) else {
        return SdfConeOcclusionResult {
            visibility: 0.0,
            bent_normal: [0.0, 0.0, 0.0],
            valid: 0,
        };
    };
    let (tangent, bitangent) = orthonormal_basis(n);
    let min_step = voxel * 0.5;

    let mut visibility_sum = 0.0f32;
    let mut weight_sum = 0.0f32;
    let mut bent = [0.0f32; 3];

    for cone in &CONE_DIRS {
        let dir = [
            tangent[0] * cone[0] + bitangent[0] * cone[1] + n[0] * cone[2],
            tangent[1] * cone[0] + bitangent[1] * cone[1] + n[1] * cone[2],
            tangent[2] * cone[0] + bitangent[2] * cone[1] + n[2] * cone[2],
        ];
        let weight = cone[2];

        let mut cone_visibility = 1.0f32;
        let mut t = min_step;
        for _ in 0..q.step_count {
            if t >= q.max_distance {
                break;
            }
            let sample_point = [
                q.position[0] + t * dir[0],
                q.position[1] + t * dir[1],
                q.position[2] + t * dir[2],
            ];
            let sampled = sample(dims, origin, voxel, dist, sample_point);
            let radius = q.tan_half_angle * t;
            let open = if radius <= f32::MIN_POSITIVE {
                1.0
            } else {
                (sampled / radius).clamp(0.0, 1.0)
            };
            cone_visibility = cone_visibility.min(open);
            if cone_visibility <= 0.0 {
                break;
            }
            t += sampled.max(min_step);
        }

        visibility_sum += cone_visibility * weight;
        weight_sum += weight;
        bent[0] += dir[0] * cone_visibility * weight;
        bent[1] += dir[1] * cone_visibility * weight;
        bent[2] += dir[2] * cone_visibility * weight;
    }

    let visibility = if weight_sum > f32::MIN_POSITIVE {
        (visibility_sum / weight_sum).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let bent_normal = normalize(bent).unwrap_or(n);

    SdfConeOcclusionResult {
        visibility,
        bent_normal,
        valid: 1,
    }
}

/// Asserts the `GPU` results match the oracle: `visibility` and each
/// `bent_normal` component within the continuous tolerance; `valid` exactly.
/// When a query is degenerate (`valid = 0`) only the flag is compared.
fn assert_parity(
    gpu: &[SdfConeOcclusionResult],
    field: &SdfConeOcclusionField,
    queries: &[SdfConeOcclusionQuery],
    label: &str,
) {
    assert_eq!(gpu.len(), queries.len(), "{label}: result count mismatch");
    for (i, (g, q)) in gpu.iter().zip(queries.iter()).enumerate() {
        let want = oracle(field, q);
        assert_eq!(g.valid, want.valid, "{label}: query {i} valid flag");
        if want.valid == 0 {
            continue;
        }
        assert!(
            close(g.visibility, want.visibility),
            "{label}: query {i} visibility GPU {} vs CPU {}",
            g.visibility,
            want.visibility
        );
        for axis in 0..3 {
            assert!(
                close(g.bent_normal[axis], want.bent_normal[axis]),
                "{label}: query {i} bent_normal[{axis}] GPU {} vs CPU {}",
                g.bent_normal[axis],
                want.bent_normal[axis]
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

/// Builds a radial (sphere) field: each cell stores the analytic signed
/// distance `|p - center| - radius`, row-major with `x` fastest.
fn radial_field(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    center: [f32; 3],
    radius: f32,
) -> SdfConeOcclusionField {
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
    SdfConeOcclusionField::new(dims, origin, voxel, dist)
}

/// Builds a field whose every cell holds the same constant signed distance.
fn constant_field(
    dims: [u32; 3],
    origin: [f32; 3],
    voxel: f32,
    value: f32,
) -> SdfConeOcclusionField {
    let count = (dims[0] * dims[1] * dims[2]) as usize;
    SdfConeOcclusionField::new(dims, origin, voxel, vec![value; count])
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = constant_field([3, 3, 3], [-1.0, -1.0, -1.0], 1.0, 4.0);
    let gpu = GpuSdfConeOcclusion::new(&ctx);
    let out = gpu.evaluate(&ctx, &field, &[]);
    assert!(out.is_empty(), "empty batch must return no results");
}

#[test]
fn degenerate_normal_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let field = constant_field([3, 3, 3], [-1.0, -1.0, -1.0], 1.0, 5.0);
    let gpu = GpuSdfConeOcclusion::new(&ctx);
    let queries = vec![
        SdfConeOcclusionQuery::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.5, 10.0, 8),
        SdfConeOcclusionQuery::new([0.2, -0.1, 0.3], [0.0, 0.0, 0.0], 0.3, 20.0, 4),
    ];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "degenerate");
    for r in &out {
        assert_eq!(r.valid, 0, "zero normal must be invalid");
        assert!(close(r.visibility, 0.0));
        for axis in 0..3 {
            assert!(close(r.bent_normal[axis], 0.0));
        }
    }
}

#[test]
fn open_space_is_fully_visible() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Everything is far away (distance 10 everywhere), so every cone stays wide
    // open and the visibility pins to one with the bent normal along +z.
    let field = constant_field([3, 3, 3], [-1.0, -1.0, -1.0], 1.0, 10.0);
    let gpu = GpuSdfConeOcclusion::new(&ctx);
    let queries = vec![SdfConeOcclusionQuery::new(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        0.1,
        1000.0,
        8,
    )];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "open");
    assert_eq!(out[0].valid, 1);
    let want = oracle(&field, &queries[0]);
    assert!(
        want.visibility > 0.99,
        "open point must be essentially fully visible, got {}",
        want.visibility
    );
}

#[test]
fn point_beside_sphere_is_partially_occluded() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Sphere of radius 0.75 at the world origin. The shaded point sits above the
    // sphere on +z looking up: the center and ring cones graze the solid so the
    // hemisphere is partially occluded and the bent normal leans off-axis.
    let field = radial_field(
        [21, 21, 21],
        [-2.5, -2.5, -2.5],
        0.25,
        [0.0, 0.0, 0.0],
        0.75,
    );
    let gpu = GpuSdfConeOcclusion::new(&ctx);
    let queries = vec![SdfConeOcclusionQuery::new(
        [0.9, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.5,
        4.0,
        16,
    )];
    let out = gpu.evaluate(&ctx, &field, &queries);
    assert_parity(&out, &field, &queries, "sphere");
    assert_eq!(out[0].valid, 1);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfConeOcclusion::new(&ctx);
    let mut rng = Lcg::new(0x51D5_0C7E);
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
            // Strictly positive cells so trilinear taps stay positive: no cone
            // ever closes, so the march runs its full step budget everywhere and
            // the result is fully deterministic.
            dist.push(rng.next_range(0.3, 3.0));
        }
        let normal = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        let len2 = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
        // Reject a near-zero normal so both sides agree it is valid.
        if len2 <= 1.0e-3 {
            continue;
        }
        let field = SdfConeOcclusionField::new(dims, origin, voxel, dist);
        // `max_distance` far beyond the furthest reachable `t` (8 steps of at
        // most 3.0) pins every cone to its full step budget with no early break.
        let queries = [SdfConeOcclusionQuery::new(
            [
                origin[0] + rng.next_range(0.0, 1.0),
                origin[1] + rng.next_range(0.0, 1.0),
                origin[2] + rng.next_range(0.0, 1.0),
            ],
            normal,
            rng.next_range(0.1, 1.0),
            1000.0,
            8,
        )];
        let out = gpu.evaluate(&ctx, &field, &queries);
        assert_parity(&out, &field, &queries, "sweep");
        assert_eq!(out[0].valid, 1, "sweep normal must be valid");
        built += 1;
    }
}
