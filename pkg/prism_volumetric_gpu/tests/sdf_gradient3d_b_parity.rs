//! Real-device parity for the analytic signed-distance *gradient* twin:
//! [`GpuSdfGradient3dB`](prism_volumetric_gpu::sdf_gradient3d_b::GpuSdfGradient3dB)
//! must reproduce the `CPU` closed forms of
//! `prism_render_architecture::ray_scene::sdf_primitives` — the arbitrary
//! segment `capsule_gradient`, the `y`-axis `capped_cylinder_gradient`, the
//! exact `octahedron_gradient`, and the upright `vertical_capsule_gradient` —
//! across interior, exterior and off-crease points for all four shapes plus a
//! randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same four closed forms (the
//! clamped-segment nearest-point normalisation for the capsule and vertical
//! capsule, the planar box gradient mapped back to `3D` for the capped
//! cylinder, and the folded-octant dominant-face normalisation for the
//! octahedron). Because the reference and this oracle are both scalar `f32`, a
//! `GPU == oracle` pass is direct evidence the ported kernel computes the same
//! gradients the reference does.
//!
//! # Parity criterion
//!
//! Each gradient threads through a `sqrt`-based length and a divide, so a `GPU`
//! component may land a few units in the last place from the scalar oracle;
//! each is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with a
//! `rel_diff` floor of `1e-6` so a near-zero expected component does not inflate
//! the relative error.
//!
//! # Conditioning
//!
//! The gradient is discontinuous on the measure-zero skeleton/fold creases (the
//! capsule and vertical-capsule axes, the cylinder central axis and interior
//! medial seam, the octahedron octant seams and coordinate planes). A
//! last-place disagreement there could pick a different branch, so named
//! fixtures stay a safe margin from those creases and the randomized sweep
//! reject-samples every point until it is well clear of all four shapes' seams.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_gradient3d_b::{
    GpuSdfGradient3dB, SdfGradient3dBQuery, SdfGradient3dBResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each gradient component. A `GPU` `sqrt`/divide may land a
/// few units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const DIST_ABS: f32 = 1.0e-4;

/// Relative bound on each gradient component, applied for larger magnitudes
/// where a few units in the last place exceed the absolute floor.
const DIST_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected component
/// does not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Shared capsule endpoint `a`, placed off the coordinate axes so the capsule
/// skeleton stays clear of the sampled points.
const CAP_A: [f32; 3] = [-0.5, -0.3, 0.2];
/// Shared capsule endpoint `b`.
const CAP_B: [f32; 3] = [0.8, 0.6, -0.4];
/// Shared capped-cylinder half height on the `y` axis.
const CYL_HALF: f32 = 1.0;
/// Shared capped-cylinder radius.
const CYL_RADIUS: f32 = 0.7;
/// Shared octahedron vertex radius.
const OCT_RADIUS: f32 = 1.0;
/// Shared vertical-capsule height along `+y`.
const VCAP_HEIGHT: f32 = 1.5;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Euclidean length of a 3-vector, matching the reference `length` helper.
fn length3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Euclidean length of a 2-vector, matching the reference `length2` helper.
fn length2(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

/// Dot product of two 3-vectors, matching the reference `dot` helper.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Component-wise subtraction of two 3-vectors.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Independent reimplementation of the reference `capsule_gradient`: normalise
/// the offset from the clamped nearest point on the segment `a`-`b`, returning
/// the zero vector exactly on the skeleton.
fn capsule_oracle(point: [f32; 3], a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    let pa = sub3(point, a);
    let ba = sub3(b, a);
    let ba_len_sq = dot3(ba, ba);
    let h = if ba_len_sq > f32::MIN_POSITIVE {
        (dot3(pa, ba) / ba_len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let closest = [pa[0] - ba[0] * h, pa[1] - ba[1] * h, pa[2] - ba[2] * h];
    let l = length3(closest);
    if l > 0.0 {
        [closest[0] / l, closest[1] / l, closest[2] / l]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Independent reimplementation of the reference `capped_cylinder_gradient`:
/// the first-quadrant box gradient in the `(radial, |y|)` frame, mapped back
/// along the `xz` unit and `sign(y)`.
fn capped_cylinder_oracle(point: [f32; 3], half_height: f32, radius: f32) -> [f32; 3] {
    let radial = length2(point[0], point[2]);
    let dx = radial - radius;
    let dy = point[1].abs() - half_height;
    let mr = dx.max(0.0);
    let mh = dy.max(0.0);
    let l = length2(mr, mh);
    let (gr, gh) = if l > 0.0 {
        (mr / l, mh / l)
    } else if dx >= dy {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let sy = if point[1] < 0.0 { -1.0 } else { 1.0 };
    if radial > 0.0 {
        [gr * point[0] / radial, gh * sy, gr * point[2] / radial]
    } else {
        [0.0, gh * sy, 0.0]
    }
}

/// Gradient of `length(u)` in the rotated octant frame for a folded triple `q`,
/// matching the reference `grad_q` closure.
fn oct_grad_q(q: [f32; 3], radius: f32) -> [f32; 3] {
    let k = (0.5 * (q[2] - q[1] + radius)).clamp(0.0, radius);
    let u = [q[0], q[1] - radius + k, q[2] - k];
    let l = length3(u);
    if l > 0.0 {
        [u[0] / l, u[1] / l, u[2] / l]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Independent reimplementation of the reference `octahedron_gradient`: fold
/// into the positive octant, pick the dominant face (or the central slab),
/// normalise the folded offset, then un-permute and re-sign.
fn octahedron_oracle(point: [f32; 3], radius: f32) -> [f32; 3] {
    const C: f32 = 0.577_350_26; // 1 / sqrt(3)
    let sign = [
        if point[0] < 0.0 { -1.0 } else { 1.0 },
        if point[1] < 0.0 { -1.0 } else { 1.0 },
        if point[2] < 0.0 { -1.0 } else { 1.0 },
    ];
    let p = [point[0].abs(), point[1].abs(), point[2].abs()];
    let m = p[0] + p[1] + p[2] - radius;
    let gp = if 3.0 * p[0] < m {
        oct_grad_q([p[0], p[1], p[2]], radius)
    } else if 3.0 * p[1] < m {
        let g = oct_grad_q([p[1], p[2], p[0]], radius);
        [g[2], g[0], g[1]]
    } else if 3.0 * p[2] < m {
        let g = oct_grad_q([p[2], p[0], p[1]], radius);
        [g[1], g[2], g[0]]
    } else {
        [C, C, C]
    };
    [gp[0] * sign[0], gp[1] * sign[1], gp[2] * sign[2]]
}

/// Independent reimplementation of the reference `vertical_capsule_gradient`:
/// normalise the offset from the clamped nearest point on the upright `+y`
/// segment, returning the zero vector exactly on the skeleton.
fn vertical_capsule_oracle(point: [f32; 3], height: f32) -> [f32; 3] {
    let qy = point[1] - point[1].clamp(0.0, height);
    let v = [point[0], qy, point[2]];
    let l = length3(v);
    if l > 0.0 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against.
fn oracle(q: &SdfGradient3dBQuery) -> SdfGradient3dBResult {
    SdfGradient3dBResult {
        capsule_grad: capsule_oracle(q.point, q.capsule_a, q.capsule_b),
        capped_cylinder_grad: capped_cylinder_oracle(q.point, q.half_height, q.radius),
        octahedron_grad: octahedron_oracle(q.point, q.oct_radius),
        vertical_capsule_grad: vertical_capsule_oracle(q.point, q.height),
    }
}

/// Pins the three components of one gradient vector against the oracle.
fn check_vec(idx: usize, name: &str, got: [f32; 3], want: [f32; 3]) {
    for (axis, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            close(*g, *w, DIST_ABS, DIST_REL),
            "query {idx} {name} axis {axis}: gpu {g} vs cpu {w}"
        );
    }
}

/// Pins one `GPU` result against the host oracle under the tolerance.
fn check_one(idx: usize, got: &SdfGradient3dBResult, want: &SdfGradient3dBResult) {
    check_vec(idx, "capsule_grad", got.capsule_grad, want.capsule_grad);
    check_vec(
        idx,
        "capped_cylinder_grad",
        got.capped_cylinder_grad,
        want.capped_cylinder_grad,
    );
    check_vec(
        idx,
        "octahedron_grad",
        got.octahedron_grad,
        want.octahedron_grad,
    );
    check_vec(
        idx,
        "vertical_capsule_grad",
        got.vertical_capsule_grad,
        want.vertical_capsule_grad,
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSdfGradient3dB, queries: &[SdfGradient3dBQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_one(idx, result, &want);
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

/// Draws a `f32` in `[lo, hi)` from the generator, using only integer-to-float
/// division (no transcendental).
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let u = lcg(state) as f32 * (1.0 / 4_294_967_296.0);
    lo + (hi - lo) * u
}

/// Builds a query carrying the point plus the four shared, well-conditioned
/// shape parameters.
fn make_query(point: [f32; 3]) -> SdfGradient3dBQuery {
    SdfGradient3dBQuery::new(
        point,
        CAP_A,
        CAP_B,
        CYL_HALF,
        CYL_RADIUS,
        OCT_RADIUS,
        VCAP_HEIGHT,
    )
}

/// Length of the capsule's clamped nearest-point offset, used to keep fixtures
/// off the capsule skeleton where the gradient is discontinuous.
fn capsule_closest_len(point: [f32; 3]) -> f32 {
    let pa = sub3(point, CAP_A);
    let ba = sub3(CAP_B, CAP_A);
    let ba_len_sq = dot3(ba, ba);
    let h = if ba_len_sq > f32::MIN_POSITIVE {
        (dot3(pa, ba) / ba_len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let closest = [pa[0] - ba[0] * h, pa[1] - ba[1] * h, pa[2] - ba[2] * h];
    length3(closest)
}

/// Whether `point` is a safe margin from every shape's gradient crease, so a
/// last-place `GPU` difference never selects a different branch than the
/// oracle.
fn conditioned(point: [f32; 3]) -> bool {
    // Capsule: clear of the segment skeleton.
    if capsule_closest_len(point) <= 0.1 {
        return false;
    }
    // Capped cylinder: off the central axis, off the rim corner, and off the
    // interior medial seam where the inside box picks a different axis.
    let radial = length2(point[0], point[2]);
    if radial <= 0.2 {
        return false;
    }
    let dx = radial - CYL_RADIUS;
    let dy = point[1].abs() - CYL_HALF;
    if dx.abs() < 0.1 && dy.abs() < 0.1 {
        return false;
    }
    if dx < 0.0 && dy < 0.0 && (dx - dy).abs() < 0.15 {
        return false;
    }
    // Octahedron: off the coordinate planes (sign flips) and off the
    // octant-branch seams `3 * p[i] == m`.
    let p = [point[0].abs(), point[1].abs(), point[2].abs()];
    if p[0] <= 0.15 || p[1] <= 0.15 || p[2] <= 0.15 {
        return false;
    }
    let m = p[0] + p[1] + p[2] - OCT_RADIUS;
    if (3.0 * p[0] - m).abs() <= 0.2
        || (3.0 * p[1] - m).abs() <= 0.2
        || (3.0 * p[2] - m).abs() <= 0.2
    {
        return false;
    }
    // Vertical capsule: off the upright `+y` skeleton.
    let qy = point[1] - point[1].clamp(0.0, VCAP_HEIGHT);
    if length2(point[0], point[2]) <= 0.1 || length3([point[0], qy, point[2]]) <= 0.1 {
        return false;
    }
    true
}

/// Draws one well-conditioned random point in `[-3, 3]^3`, reject-sampling
/// until it clears every shape's gradient crease.
fn random_query(state: &mut u64) -> SdfGradient3dBQuery {
    for _ in 0..4096 {
        let point = [
            uniform(state, -3.0, 3.0),
            uniform(state, -3.0, 3.0),
            uniform(state, -3.0, 3.0),
        ];
        if conditioned(point) {
            return make_query(point);
        }
    }
    // Fallback (practically never reached): a known well-conditioned point.
    make_query([1.5, 1.2, 0.9])
}

/// A fixed battery of named points, each well clear of every shape's creases,
/// spanning interior and exterior regions and both the folded faces and the
/// central slab of the octahedron.
fn fixture_queries() -> Vec<SdfGradient3dBQuery> {
    vec![
        // Capsule exterior, off the segment skeleton.
        make_query([1.0, -1.0, 1.0]),
        // Capped-cylinder exterior past the lateral wall.
        make_query([1.5, 0.3, 0.0]),
        // Capped-cylinder interior, inside both the wall and the caps.
        make_query([0.2, 0.0, 0.1]),
        // Octahedron central slab (uniform face normal branch).
        make_query([0.4, 0.4, 0.4]),
        // Octahedron folded face along the dominant `x` axis.
        make_query([2.0, 0.3, 0.3]),
        // Vertical-capsule off the upright skeleton at mid height.
        make_query([0.5, 0.5, 0.0]),
        // Mixed negative-octant point exercising all four sign paths.
        make_query([-1.2, -0.8, -0.9]),
        // Larger-scale point well outside every shape.
        make_query([2.5, 2.0, -1.5]),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_gradient3d_b parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuSdfGradient3dB::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn capsule_gradient_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient3dB::new(&ctx);
    // Two off-skeleton points on opposite sides of the capsule segment.
    let near = make_query([1.0, -1.0, 1.0]);
    let far = make_query([-1.5, 1.2, -0.8]);
    let got = gpu.evaluate(&ctx, &[near, far]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&near));
    check_one(1, &got[1], &oracle(&far));
}

#[test]
fn capped_cylinder_gradient_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient3dB::new(&ctx);
    // Exterior past the wall, interior inside the box, and a point above the cap.
    let wall = make_query([1.5, 0.3, 0.0]);
    let interior = make_query([0.2, 0.0, 0.1]);
    let cap = make_query([0.3, 1.6, 0.3]);
    let got = gpu.evaluate(&ctx, &[wall, interior, cap]);
    assert_eq!(got.len(), 3);
    check_one(0, &got[0], &oracle(&wall));
    check_one(1, &got[1], &oracle(&interior));
    check_one(2, &got[2], &oracle(&cap));
}

#[test]
fn octahedron_gradient_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient3dB::new(&ctx);
    // Central-slab branch, dominant-face branch, and a negative-octant point.
    let slab = make_query([0.4, 0.4, 0.4]);
    let face = make_query([2.0, 0.3, 0.3]);
    let negative = make_query([-0.3, -2.0, -0.4]);
    let got = gpu.evaluate(&ctx, &[slab, face, negative]);
    assert_eq!(got.len(), 3);
    check_one(0, &got[0], &oracle(&slab));
    check_one(1, &got[1], &oracle(&face));
    check_one(2, &got[2], &oracle(&negative));
}

#[test]
fn vertical_capsule_gradient_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient3dB::new(&ctx);
    // Beside the shaft at mid height, below the bottom cap, and above the top cap.
    let shaft = make_query([0.5, 0.5, 0.0]);
    let below = make_query([0.4, -1.0, 0.3]);
    let above = make_query([0.3, 2.5, -0.4]);
    let got = gpu.evaluate(&ctx, &[shaft, below, above]);
    assert_eq!(got.len(), 3);
    check_one(0, &got[0], &oracle(&shaft));
    check_one(1, &got[1], &oracle(&below));
    check_one(2, &got[2], &oracle(&above));
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient3dB::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfGradient3dB::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned points pin all four
    // gradient vectors across a wide span of query positions.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
