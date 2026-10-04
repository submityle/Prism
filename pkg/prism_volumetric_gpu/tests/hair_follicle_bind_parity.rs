//! Real-device parity for the follicle-bind twin:
//! [`GpuHairFollicleBind`](prism_volumetric_gpu::hair_follicle_bind::GpuHairFollicleBind)
//! must reproduce the `CPU` golden
//! `prism_render_architecture::hair::follicle_bind::{compute_barycentric,
//! bind_follicle, transfer_root, transfer_frame}`, which bind a hair root to a
//! rest scalp triangle (barycentric position plus a signed normal offset) and
//! reconstruct the moved root and its orthonormal frame on the deformed
//! triangle.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! Ericson's projected-plane barycentric solve with a centroid fallback, the
//! sanitised non-negative weights, the normal-offset bind, the deformed-surface
//! transfer, and the Gram-Schmidt frame with canonical-axis fallbacks — written
//! out directly so the test never imports `prism_render_architecture`. Because
//! both the reference and this oracle are scalar `f32`, a `GPU == oracle` pass
//! is direct evidence the ported kernel computes the same bind-and-transfer the
//! reference does.
//!
//! The fixtures cover a flat unit triangle whose bind recovers the root exactly,
//! a translated deformation that carries the root rigidly, an orthonormality
//! check on the transferred frame, a degenerate (zero-area) rest triangle that
//! must report `valid = 0` with the reference's centroid-and-canonical
//! fallbacks, and a multi-element batch that exercises the storage stride. A
//! sweep over random well-conditioned triangles and roots follows, plus an
//! empty batch the host short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every continuous output threads through guarded divisions and `sqrt`
//! (barycentric reciprocal, normal re-normalisation), so `CPU` and `GPU`
//! evaluate the same closed form but need not be bit-exact (a `GPU` may contract
//! a multiply-add). The continuous comparison is
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete
//! `valid` flag is compared exactly. The sweep rejects any rest triangle whose
//! barycentric denominator sits near the `1e-12` degeneracy threshold, so parity
//! never balances on that knife edge.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::follicle_bind`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::hair_follicle_bind::{
    GpuHairFollicleBind, HairFollicleBindQuery, HairFollicleBindResult, TriangleFrame,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;
/// Degeneracy threshold mirroring the golden `DEGENERATE_EPS`.
const DEGENERATE_EPS: f32 = 1.0e-12;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Returns `true` when two three-vectors agree component-wise.
fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
    close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
}

/// Difference of two three-vectors.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Sum of two three-vectors.
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales a three-vector by a scalar.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Dot product of two three-vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two three-vectors.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Euclidean length of a three-vector.
fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

/// Independent host `normalize_or`: unit `a`, else `fallback` for a near-zero
/// or non-finite length.
fn normalize_or(a: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = length(a);
    if len > DEGENERATE_EPS && len.is_finite() {
        scale(a, 1.0 / len)
    } else {
        fallback
    }
}

/// Returns `true` when all three components are finite.
fn finite3(a: [f32; 3]) -> bool {
    a[0].is_finite() && a[1].is_finite() && a[2].is_finite()
}

/// Independent host `compute_barycentric`: Ericson's projected-plane solve, with
/// a centroid fallback for a degenerate triangle. Returns `(u, v, w, degenerate)`.
fn compute_barycentric(point: [f32; 3], tri: &TriangleFrame) -> ([f32; 3], bool) {
    let a = tri.positions[0];
    let b = tri.positions[1];
    let c = tri.positions[2];
    let v0 = sub(b, a);
    let v1 = sub(c, a);
    let v2 = sub(point, a);
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() <= DEGENERATE_EPS || !denom.is_finite() {
        return ([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], true);
    }
    let inv = 1.0 / denom;
    let v = (d11 * d20 - d01 * d21) * inv;
    let w = (d00 * d21 - d01 * d20) * inv;
    let u = 1.0 - v - w;
    ([u, v, w], false)
}

/// Independent host `Barycentric::sanitized`: non-negative weights that sum to
/// one, with a centroid fallback for an all-zero set.
fn sanitize_bary(b: [f32; 3]) -> [f32; 3] {
    let clamp0 = |x: f32| if x.is_finite() && x > 0.0 { x } else { 0.0 };
    let u = clamp0(b[0]);
    let v = clamp0(b[1]);
    let w = clamp0(b[2]);
    let sum = u + v + w;
    if sum > DEGENERATE_EPS {
        [u / sum, v / sum, w / sum]
    } else {
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    }
}

/// Interpolates a per-vertex attribute by barycentric weights.
fn bary_mix(bary: [f32; 3], attr: &[[f32; 3]; 3]) -> [f32; 3] {
    add(
        add(scale(attr[0], bary[0]), scale(attr[1], bary[1])),
        scale(attr[2], bary[2]),
    )
}

/// Independent host oracle for one query: binds `root` to `rest` and transfers
/// the binding onto `deformed`, returning the twin's result layout.
fn oracle(q: &HairFollicleBindQuery) -> HairFollicleBindResult {
    // compute_barycentric + sanitize on the rest triangle (bind_follicle).
    let (bary_raw, degenerate) = compute_barycentric(q.root, &q.rest);
    let bary = sanitize_bary(bary_raw);

    let surface_rest = bary_mix(bary, &q.rest.positions);
    let normal_rest = normalize_or(bary_mix(bary, &q.rest.normals), [0.0, 0.0, 1.0]);
    let offset = dot(sub(q.root, surface_rest), normal_rest);
    let normal_offset = if offset.is_finite() { offset } else { 0.0 };

    // transfer_root onto the deformed triangle.
    let surface_def = bary_mix(bary, &q.deformed.positions);
    let normal_def = normalize_or(bary_mix(bary, &q.deformed.normals), [0.0, 0.0, 1.0]);
    let out = add(surface_def, scale(normal_def, normal_offset));
    let out_root = if finite3(out) { out } else { surface_def };

    // transfer_frame: Gram-Schmidt tangent, right-handed bitangent.
    let raw_tangent = bary_mix(bary, &q.deformed.tangents);
    let projected = sub(raw_tangent, scale(normal_def, dot(raw_tangent, normal_def)));
    let fallback = if normal_def[0].abs() < 0.9 {
        normalize_or(cross(normal_def, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
    } else {
        normalize_or(cross(normal_def, [0.0, 1.0, 0.0]), [1.0, 0.0, 0.0])
    };
    let tangent = normalize_or(projected, fallback);
    let bitangent = normalize_or(cross(normal_def, tangent), [0.0, 1.0, 0.0]);

    HairFollicleBindResult {
        root: out_root,
        normal: normal_def,
        tangent,
        bitangent,
        valid: u32::from(!degenerate),
    }
}

/// Pins one `GPU` result against the oracle: `valid` exactly, every continuous
/// quantity within the module tolerance.
fn check_one(idx: usize, got: &HairFollicleBindResult, want: &HairFollicleBindResult) {
    assert_eq!(
        got.valid, want.valid,
        "query {idx} valid: gpu {} vs cpu {}",
        got.valid, want.valid
    );
    assert!(
        close3(got.root, want.root),
        "query {idx} root: gpu {:?} vs cpu {:?}",
        got.root,
        want.root
    );
    assert!(
        close3(got.normal, want.normal),
        "query {idx} normal: gpu {:?} vs cpu {:?}",
        got.normal,
        want.normal
    );
    assert!(
        close3(got.tangent, want.tangent),
        "query {idx} tangent: gpu {:?} vs cpu {:?}",
        got.tangent,
        want.tangent
    );
    assert!(
        close3(got.bitangent, want.bitangent),
        "query {idx} bitangent: gpu {:?} vs cpu {:?}",
        got.bitangent,
        want.bitangent
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuHairFollicleBind, queries: &[HairFollicleBindQuery]) {
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

/// The canonical flat unit triangle in the `z = 0` plane: upward normals and
/// `+x` tangents at every vertex.
fn flat_tri() -> TriangleFrame {
    TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals: [[0.0, 0.0, 1.0]; 3],
        tangents: [[1.0, 0.0, 0.0]; 3],
    }
}

/// Translates every position of a triangle by `delta`, leaving the frame
/// vectors unchanged (a rigid shift).
fn translate(tri: &TriangleFrame, delta: [f32; 3]) -> TriangleFrame {
    TriangleFrame {
        positions: [
            add(tri.positions[0], delta),
            add(tri.positions[1], delta),
            add(tri.positions[2], delta),
        ],
        normals: tri.normals,
        tangents: tri.tangents,
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping hair_follicle_bind parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuHairFollicleBind::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn bind_then_transfer_identity_recovers_root() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairFollicleBind::new(&ctx);
    // Rest == deformed: binding and transferring must recover the root exactly.
    let tri = flat_tri();
    let queries = [
        HairFollicleBindQuery::new([0.25, 0.25, 0.0], tri, tri),
        HairFollicleBindQuery::new([0.25, 0.25, 0.5], tri, tri),
        HairFollicleBindQuery::new([0.1, 0.6, -0.2], tri, tri),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_one(idx, result, &want);
        assert_eq!(result.valid, 1, "query {idx} should be non-degenerate");
        // The round trip reconstructs the original root to the module tolerance.
        assert!(
            close3(result.root, q.root),
            "query {idx} identity transfer should recover root {:?}, got {:?}",
            q.root,
            result.root
        );
    }
}

#[test]
fn translated_triangle_carries_root_rigidly() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairFollicleBind::new(&ctx);
    // A pure translation of the scalp triangle shifts the bound root by the same
    // delta.
    let rest = flat_tri();
    let delta = [3.0, -2.0, 1.5];
    let deformed = translate(&rest, delta);
    let root = [0.3, 0.3, 0.4];
    let queries = [HairFollicleBindQuery::new(root, rest, deformed)];
    let got = gpu.evaluate(&ctx, &queries);
    let want = oracle(&queries[0]);
    check_one(0, &got[0], &want);
    assert!(
        close3(got[0].root, add(root, delta)),
        "translated transfer should shift root by {delta:?}, got {:?}",
        got[0].root
    );
}

#[test]
fn transferred_frame_is_orthonormal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairFollicleBind::new(&ctx);
    // The reconstructed frame must be unit-length and mutually orthogonal,
    // whatever the (non-orthonormal) authored per-vertex tangents are.
    let rest = flat_tri();
    let deformed = TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [2.0, 0.0, 1.0], [0.0, 2.0, -1.0]],
        normals: [[0.1, 0.2, 1.0], [0.0, 0.1, 1.0], [-0.1, 0.0, 1.0]],
        tangents: [[1.0, 0.3, 0.0], [0.9, 0.0, 0.2], [1.0, -0.2, 0.1]],
    };
    let queries = [HairFollicleBindQuery::new([0.3, 0.4, 0.2], rest, deformed)];
    let got = gpu.evaluate(&ctx, &queries);
    let want = oracle(&queries[0]);
    check_one(0, &got[0], &want);
    let r = &got[0];
    assert!(close(length(r.normal), 1.0), "normal must be unit");
    assert!(close(length(r.tangent), 1.0), "tangent must be unit");
    assert!(close(length(r.bitangent), 1.0), "bitangent must be unit");
    assert!(
        close(dot(r.normal, r.tangent), 0.0),
        "normal and tangent must be orthogonal"
    );
    assert!(
        close(dot(r.normal, r.bitangent), 0.0),
        "normal and bitangent must be orthogonal"
    );
    assert!(
        close(dot(r.tangent, r.bitangent), 0.0),
        "tangent and bitangent must be orthogonal"
    );
}

#[test]
fn degenerate_rest_triangle_is_invalid_with_fallbacks() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairFollicleBind::new(&ctx);
    // A zero-area rest triangle (three coincident vertices) drives the centroid
    // fallback: valid = 0, but every continuous output still matches the
    // reference's finite fallbacks.
    let degenerate = TriangleFrame {
        positions: [[1.0, 1.0, 1.0]; 3],
        normals: [[0.0, 0.0, 1.0]; 3],
        tangents: [[1.0, 0.0, 0.0]; 3],
    };
    let deformed = flat_tri();
    let queries = [
        HairFollicleBindQuery::new([2.0, 0.5, 0.3], degenerate, deformed),
        HairFollicleBindQuery::new([0.0, 0.0, 0.0], degenerate, degenerate),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_one(idx, result, &want);
        assert_eq!(
            result.valid, 0,
            "query {idx} degenerate rest triangle must be invalid"
        );
    }
}

#[test]
fn multi_element_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairFollicleBind::new(&ctx);
    // A mixed, >=2 element batch pins the storage stride: identity, translated,
    // sheared and degenerate queries all resolve in one dispatch.
    let rest = flat_tri();
    let sheared = TriangleFrame {
        positions: [[0.0, 0.0, 0.0], [1.5, 0.2, 0.3], [0.1, 1.4, -0.2]],
        normals: [[0.0, 0.1, 1.0], [0.05, 0.0, 1.0], [-0.05, 0.1, 1.0]],
        tangents: [[1.0, 0.0, 0.2], [1.0, 0.1, 0.0], [0.9, -0.1, 0.1]],
    };
    let degenerate = TriangleFrame {
        positions: [[0.5, 0.5, 0.5]; 3],
        normals: [[0.0, 0.0, 1.0]; 3],
        tangents: [[1.0, 0.0, 0.0]; 3],
    };
    let queries = [
        HairFollicleBindQuery::new([0.3, 0.3, 0.1], rest, rest),
        HairFollicleBindQuery::new([0.2, 0.5, 0.4], rest, translate(&rest, [1.0, 2.0, -1.0])),
        HairFollicleBindQuery::new([0.4, 0.3, 0.2], rest, sheared),
        HairFollicleBindQuery::new([1.0, 1.0, 1.0], degenerate, sheared),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairFollicleBind::new(&ctx);
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut queries = Vec::new();
    for _ in 0..512 {
        // Build a well-conditioned rest triangle: a jittered base plus two edge
        // vectors with a large cross-product magnitude, rejecting any triangle
        // whose barycentric denominator sits near the degeneracy threshold.
        let base = [
            uniform(&mut state, -2.0, 2.0),
            uniform(&mut state, -2.0, 2.0),
            uniform(&mut state, -2.0, 2.0),
        ];
        let (edge0, edge1) = loop {
            let e0 = [
                uniform(&mut state, 0.5, 2.0),
                uniform(&mut state, -1.0, 1.0),
                uniform(&mut state, -1.0, 1.0),
            ];
            let e1 = [
                uniform(&mut state, -1.0, 1.0),
                uniform(&mut state, 0.5, 2.0),
                uniform(&mut state, -1.0, 1.0),
            ];
            let d00 = dot(e0, e0);
            let d01 = dot(e0, e1);
            let d11 = dot(e1, e1);
            let denom = d00 * d11 - d01 * d01;
            // Keep the solve far from the zero-area knife edge.
            if denom.abs() > 1.0e-2 {
                break (e0, e1);
            }
        };
        let rest = TriangleFrame {
            positions: [base, add(base, edge0), add(base, edge1)],
            normals: [
                [
                    uniform(&mut state, -0.3, 0.3),
                    uniform(&mut state, -0.3, 0.3),
                    uniform(&mut state, 0.6, 1.4),
                ],
                [
                    uniform(&mut state, -0.3, 0.3),
                    uniform(&mut state, -0.3, 0.3),
                    uniform(&mut state, 0.6, 1.4),
                ],
                [
                    uniform(&mut state, -0.3, 0.3),
                    uniform(&mut state, -0.3, 0.3),
                    uniform(&mut state, 0.6, 1.4),
                ],
            ],
            tangents: [
                [
                    uniform(&mut state, 0.5, 1.5),
                    uniform(&mut state, -0.4, 0.4),
                    uniform(&mut state, -0.4, 0.4),
                ],
                [
                    uniform(&mut state, 0.5, 1.5),
                    uniform(&mut state, -0.4, 0.4),
                    uniform(&mut state, -0.4, 0.4),
                ],
                [
                    uniform(&mut state, 0.5, 1.5),
                    uniform(&mut state, -0.4, 0.4),
                    uniform(&mut state, -0.4, 0.4),
                ],
            ],
        };
        // The deformed triangle is a second, independent well-conditioned frame.
        let dbase = [
            uniform(&mut state, -2.0, 2.0),
            uniform(&mut state, -2.0, 2.0),
            uniform(&mut state, -2.0, 2.0),
        ];
        let deformed = TriangleFrame {
            positions: [
                dbase,
                add(
                    dbase,
                    [
                        uniform(&mut state, 0.5, 2.0),
                        uniform(&mut state, -1.0, 1.0),
                        uniform(&mut state, -1.0, 1.0),
                    ],
                ),
                add(
                    dbase,
                    [
                        uniform(&mut state, -1.0, 1.0),
                        uniform(&mut state, 0.5, 2.0),
                        uniform(&mut state, -1.0, 1.0),
                    ],
                ),
            ],
            normals: rest.normals,
            tangents: rest.tangents,
        };
        // A root near the rest surface, lifted a little along its own axis.
        let root = [
            uniform(&mut state, -2.0, 2.0),
            uniform(&mut state, -2.0, 2.0),
            uniform(&mut state, -2.0, 2.0),
        ];
        queries.push(HairFollicleBindQuery::new(root, rest, deformed));
    }
    check(&ctx, &gpu, &queries);
}
