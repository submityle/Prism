//! Real-device parity for the cloth position-based-dynamics collision-projection
//! twin:
//! [`GpuClothCollisionProject`](prism_volumetric_gpu::cloth_collision_project::GpuClothCollisionProject)
//! must reproduce the `CPU` golden projections of
//! `prism_render_architecture::cloth::collision` (which forward to
//! `prism_physics_core::soft::collision`): `project_out_of_sphere`,
//! `project_out_of_half_space`, `project_out_of_obb` and `apply_backstop`.
//!
//! The oracle here is an independent re-implementation of those four closed
//! forms — the radial sphere push with its coincident-center `+Y` fallback, the
//! unnormalized half-space correction, the oriented-box least-penetration face
//! push through the exact `glam` `Quat::mul_vec3` rotation, and the backstop
//! limiting-plane clamp — written out directly so the test never imports
//! `prism_render_architecture` or `prism_physics_core`. It mirrors the
//! reference branch for branch, including the already-outside, zero-radius,
//! all-non-positive-box and (near) zero-normal degenerate guards.
//!
//! The fixtures cover each projection's interior-push and identity arms plus the
//! documented degenerate paths, a mixed-`kind` batch that exercises the
//! `std430` stride, and an unknown `kind` that must report `valid = 0`. A sweep
//! over random colliders and positions follows, with rejection sampling that
//! keeps every query comfortably inside or outside its collider so parity never
//! sits on the `signed-distance = 0` knife edge. An empty batch the host
//! short-circuits with no dispatch closes the suite.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every coordinate threads through multiplies, adds, guarded divisions and
//! `sqrt`, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact (a `GPU` may contract a multiply-add). The continuous comparison
//! is `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete
//! `valid` flag is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::collision`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cloth_collision_project::{
    ClothCollisionProjectQuery, GpuClothCollisionProject, KIND_BACKSTOP, KIND_HALF_SPACE, KIND_OBB,
    KIND_SPHERE,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;
/// Minimum squared length below which a direction is treated as degenerate,
/// matching the kernel and the reference `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Returns `true` when two three-channel positions agree channel-wise.
fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
    close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
}

/// Dot product of two three-vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two three-vectors.
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Component-wise subtraction.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Component-wise addition.
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales a three-vector by a scalar.
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// `glam` quaternion conjugate: negate the vector part, keep the scalar part.
fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// The exact `glam` `Quat::mul_vec3` form
/// `v*(w*w - b.b) + b*(2 v.b) + (b x v)*(2 w)`, with `b` the vector part.
fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let w = q[3];
    let b = [q[0], q[1], q[2]];
    let b_sq = dot3(b, b);
    let term0 = scale3(v, w * w - b_sq);
    let term1 = scale3(b, dot3(v, b) * 2.0);
    let term2 = scale3(cross3(b, v), w * 2.0);
    add3(add3(term0, term1), term2)
}

/// Normalizes a three-vector, matching `glam::Vec3::normalize_or_zero`: a
/// (near) zero vector returns zero, otherwise `v / sqrt(len_sq)`.
fn normalize_or_zero(v: [f32; 3]) -> [f32; 3] {
    let len_sq = dot3(v, v);
    if len_sq <= EPS_LEN_SQ {
        return [0.0, 0.0, 0.0];
    }
    scale3(v, 1.0 / len_sq.sqrt())
}

/// Host oracle for the sphere projection.
fn project_out_of_sphere(pos: [f32; 3], center: [f32; 3], radius: f32) -> [f32; 3] {
    if radius <= 0.0 {
        return pos;
    }
    let delta = sub3(pos, center);
    let dist_sq = dot3(delta, delta);
    if dist_sq >= radius * radius {
        return pos;
    }
    if dist_sq <= EPS_LEN_SQ {
        return add3(center, [0.0, radius, 0.0]);
    }
    let dir = normalize_or_zero(delta);
    add3(center, scale3(dir, radius))
}

/// Host oracle for the half-space projection.
fn project_out_of_half_space(pos: [f32; 3], normal: [f32; 3], offset: f32) -> [f32; 3] {
    let len_sq = dot3(normal, normal);
    if len_sq <= EPS_LEN_SQ {
        return pos;
    }
    let signed = dot3(normal, pos) - offset;
    if signed >= 0.0 {
        return pos;
    }
    let t = -signed / len_sq;
    add3(pos, scale3(normal, t))
}

/// Host oracle for the oriented-box projection.
fn project_out_of_obb(
    pos: [f32; 3],
    center: [f32; 3],
    orientation: [f32; 4],
    half_extents: [f32; 3],
) -> [f32; 3] {
    if half_extents[0] <= 0.0 && half_extents[1] <= 0.0 && half_extents[2] <= 0.0 {
        return pos;
    }
    let local = quat_rotate(quat_conj(orientation), sub3(pos, center));
    let al = [local[0].abs(), local[1].abs(), local[2].abs()];
    if al[0] >= half_extents[0] || al[1] >= half_extents[1] || al[2] >= half_extents[2] {
        return pos;
    }
    let pen = [
        half_extents[0] - al[0],
        half_extents[1] - al[1],
        half_extents[2] - al[2],
    ];
    let mut local_out = local;
    if pen[0] <= pen[1] && pen[0] <= pen[2] {
        local_out[0] = if local[0] >= 0.0 {
            half_extents[0]
        } else {
            -half_extents[0]
        };
    } else if pen[1] <= pen[2] {
        local_out[1] = if local[1] >= 0.0 {
            half_extents[1]
        } else {
            -half_extents[1]
        };
    } else {
        local_out[2] = if local[2] >= 0.0 {
            half_extents[2]
        } else {
            -half_extents[2]
        };
    }
    add3(center, quat_rotate(orientation, local_out))
}

/// Host oracle for the backstop projection.
fn apply_backstop(pos: [f32; 3], origin: [f32; 3], normal: [f32; 3], distance: f32) -> [f32; 3] {
    let len_sq = dot3(normal, normal);
    if len_sq <= EPS_LEN_SQ {
        return pos;
    }
    let unit = normalize_or_zero(normal);
    let s = dot3(unit, sub3(pos, origin));
    let min_s = -distance;
    if s < min_s {
        add3(pos, scale3(unit, min_s - s))
    } else {
        pos
    }
}

/// Evaluates the host oracle for one query, returning the projected position and
/// the expected `valid` flag.
fn oracle(q: &ClothCollisionProjectQuery) -> ([f32; 3], u32) {
    match q.kind {
        KIND_SPHERE => (project_out_of_sphere(q.pos, q.a, q.s), 1),
        KIND_HALF_SPACE => (project_out_of_half_space(q.pos, q.a, q.s), 1),
        KIND_OBB => (project_out_of_obb(q.pos, q.a, q.b, q.c), 1),
        KIND_BACKSTOP => (apply_backstop(q.pos, q.a, [q.b[0], q.b[1], q.b[2]], q.s), 1),
        _ => (q.pos, 0),
    }
}

/// Asserts a single query matches the oracle through the real device.
fn assert_query(ctx: &GpuContext, gpu: &GpuClothCollisionProject, q: ClothCollisionProjectQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let (pos, valid) = oracle(&q);
    assert_eq!(got[0].valid, valid, "valid mismatch: query={q:?}");
    assert!(
        close3(got[0].pos, pos),
        "pos mismatch: gpu={:?} cpu={pos:?} query={q:?}",
        got[0].pos
    );
}

#[test]
fn sphere_interior_pushes_to_surface() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCollisionProject::new(&ctx);
    // A point halfway between center and surface is pushed radially out.
    assert_query(
        &ctx,
        &gpu,
        ClothCollisionProjectQuery::sphere([1.5, 0.0, 0.0], [0.0, 0.0, 0.0], 2.0),
    );
    // A point already outside is left untouched.
    assert_query(
        &ctx,
        &gpu,
        ClothCollisionProjectQuery::sphere([5.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2.0),
    );
}

#[test]
fn sphere_degenerate_paths() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCollisionProject::new(&ctx);
    // Non-positive radius is inert.
    assert_query(
        &ctx,
        &gpu,
        ClothCollisionProjectQuery::sphere([0.3, -0.2, 0.1], [0.0, 0.0, 0.0], 0.0),
    );
    // Coincident with the center nudges out along +Y.
    assert_query(
        &ctx,
        &gpu,
        ClothCollisionProjectQuery::sphere([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1.5),
    );
}

#[test]
fn half_space_projects_and_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCollisionProject::new(&ctx);
    // A point below an unnormalized plane is lifted onto it.
    assert_query(
        &ctx,
        &gpu,
        ClothCollisionProjectQuery::half_space([0.0, -1.0, 0.0], [0.0, 2.0, 0.0], 0.0),
    );
    // A point already on the feasible side is untouched.
    assert_query(
        &ctx,
        &gpu,
        ClothCollisionProjectQuery::half_space([0.0, 3.0, 0.0], [0.0, 2.0, 0.0], 0.0),
    );
    // A (near) zero normal is inert.
    assert_query(
        &ctx,
        &gpu,
        ClothCollisionProjectQuery::half_space([0.4, -0.5, 0.2], [0.0, 0.0, 0.0], 1.0),
    );
}

#[test]
fn obb_least_penetration_face() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCollisionProject::new(&ctx);
    // 45-degree rotation about Z, interior point pushed to the nearest face.
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let quat_z45 = [0.0, 0.0, s, s];
    assert_query(
        &ctx,
        &gpu,
        ClothCollisionProjectQuery::obb(
            [0.2, 0.1, 0.0],
            [0.0, 0.0, 0.0],
            quat_z45,
            [1.0, 1.0, 1.0],
        ),
    );
    // A point outside the box is untouched.
    assert_query(
        &ctx,
        &gpu,
        ClothCollisionProjectQuery::obb(
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            quat_z45,
            [1.0, 1.0, 1.0],
        ),
    );
    // An all-non-positive box is inert.
    assert_query(
        &ctx,
        &gpu,
        ClothCollisionProjectQuery::obb(
            [0.1, 0.1, 0.1],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ),
    );
}

#[test]
fn backstop_clamps_and_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCollisionProject::new(&ctx);
    // A point sunk well behind the anchor is pushed onto the limiting plane.
    assert_query(
        &ctx,
        &gpu,
        ClothCollisionProjectQuery::backstop(
            [0.0, -3.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
        ),
    );
    // A point in front of the anchor is untouched.
    assert_query(
        &ctx,
        &gpu,
        ClothCollisionProjectQuery::backstop(
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
        ),
    );
    // A (near) zero normal is inert.
    assert_query(
        &ctx,
        &gpu,
        ClothCollisionProjectQuery::backstop(
            [0.3, -2.0, 0.1],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            1.0,
        ),
    );
}

#[test]
fn unknown_kind_passes_through() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCollisionProject::new(&ctx);
    // A kind outside 0..=3 passes the position through with valid = 0.
    let q = ClothCollisionProjectQuery {
        kind: 4,
        pos: [1.0, 2.0, 3.0],
        a: [0.0, 0.0, 0.0],
        b: [0.0, 0.0, 0.0, 1.0],
        c: [0.0, 0.0, 0.0],
        s: 0.0,
    };
    assert_query(&ctx, &gpu, q);
}

#[test]
fn mixed_kind_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCollisionProject::new(&ctx);
    let s = std::f32::consts::FRAC_1_SQRT_2;
    // A batch of distinct kinds exercises the std430 slot stride end to end.
    let queries = [
        ClothCollisionProjectQuery::sphere([0.5, 0.0, 0.0], [0.0, 0.0, 0.0], 2.0),
        ClothCollisionProjectQuery::half_space([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], 0.0),
        ClothCollisionProjectQuery::obb(
            [0.3, 0.2, 0.1],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, s, s],
            [1.0, 1.0, 1.0],
        ),
        ClothCollisionProjectQuery::backstop(
            [0.0, -3.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
        ),
        ClothCollisionProjectQuery {
            kind: 7,
            pos: [9.0, 8.0, 7.0],
            a: [0.0, 0.0, 0.0],
            b: [0.0, 0.0, 0.0, 1.0],
            c: [0.0, 0.0, 0.0],
            s: 0.0,
        },
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (pos, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert!(
            close3(r.pos, pos),
            "batch pos mismatch: gpu={:?} cpu={pos:?} query={q:?}",
            r.pos
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCollisionProject::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "an empty batch returns no results with no dispatch"
    );
}

/// A small deterministic linear-congruential generator for the random sweep,
/// keeping the fixture pure integer host-side with no transcendental call.
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

    /// A `[0, 1)` fraction built from the top bits.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}

/// Builds a random unit quaternion from the generator.
fn random_unit_quat(rng: &mut Lcg) -> [f32; 4] {
    let q = [
        rng.next_range(-1.0, 1.0),
        rng.next_range(-1.0, 1.0),
        rng.next_range(-1.0, 1.0),
        rng.next_range(-1.0, 1.0),
    ];
    let len_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    // Reject the (astronomically unlikely) near-zero draw and fall back to the
    // identity so the orientation is always a well-defined unit quaternion.
    if len_sq <= 1.0e-6 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let inv = 1.0 / len_sq.sqrt();
    [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothCollisionProject::new(&ctx);
    let mut rng = Lcg::new(0x5F_3A_C1_09);
    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        // Cycle through the four projections so the sweep covers every branch,
        // using rejection sampling to keep each sample comfortably on one side
        // of its collider surface (away from the signed-distance = 0 knife
        // edge where a fused multiply-add could flip the discrete arm).
        match rng.next_u32() & 0x3 {
            0 => {
                let center = [
                    rng.next_range(-2.0, 2.0),
                    rng.next_range(-2.0, 2.0),
                    rng.next_range(-2.0, 2.0),
                ];
                let radius = rng.next_range(0.5, 2.0);
                // Place the point either clearly inside or clearly outside.
                let dir = normalize_or_zero([
                    rng.next_range(-1.0, 1.0),
                    rng.next_range(-1.0, 1.0),
                    rng.next_range(-1.0, 1.0),
                ]);
                let dir = if dot3(dir, dir) <= 0.25 {
                    [0.0, 1.0, 0.0]
                } else {
                    dir
                };
                let frac = if rng.next_u32() & 1 == 0 {
                    rng.next_range(0.15, 0.75) * radius
                } else {
                    rng.next_range(1.3, 2.5) * radius
                };
                let pos = add3(center, scale3(dir, frac));
                queries.push(ClothCollisionProjectQuery::sphere(pos, center, radius));
            }
            1 => {
                let normal = [
                    rng.next_range(-1.5, 1.5),
                    rng.next_range(0.5, 1.5),
                    rng.next_range(-1.5, 1.5),
                ];
                let offset = rng.next_range(-1.0, 1.0);
                let len_sq = dot3(normal, normal);
                let base = dot3(normal, [0.0, 0.0, 0.0]);
                let _ = base;
                // Choose a signed distance with a clear margin on either side.
                let signed = if rng.next_u32() & 1 == 0 {
                    rng.next_range(-2.0, -0.3)
                } else {
                    rng.next_range(0.3, 2.0)
                };
                // Solve normal.dot(pos) - offset = signed with pos = t*normal.
                let t = (signed + offset) / len_sq;
                let pos = scale3(normal, t);
                queries.push(ClothCollisionProjectQuery::half_space(pos, normal, offset));
            }
            2 => {
                let center = [
                    rng.next_range(-2.0, 2.0),
                    rng.next_range(-2.0, 2.0),
                    rng.next_range(-2.0, 2.0),
                ];
                let orientation = random_unit_quat(&mut rng);
                let half = [
                    rng.next_range(0.4, 1.5),
                    rng.next_range(0.4, 1.5),
                    rng.next_range(0.4, 1.5),
                ];
                // Build a local point either clearly inside all slabs or clearly
                // outside one, with margins away from each face, then rotate it
                // into world space.
                let inside = rng.next_u32() & 1 == 0;
                let local = if inside {
                    [
                        rng.next_range(-0.7, 0.7) * half[0],
                        rng.next_range(-0.7, 0.7) * half[1],
                        rng.next_range(-0.7, 0.7) * half[2],
                    ]
                } else {
                    [
                        (half[0] + rng.next_range(0.3, 1.5))
                            * if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 },
                        rng.next_range(-0.7, 0.7) * half[1],
                        rng.next_range(-0.7, 0.7) * half[2],
                    ]
                };
                let pos = add3(center, quat_rotate(orientation, local));
                queries.push(ClothCollisionProjectQuery::obb(
                    pos,
                    center,
                    orientation,
                    half,
                ));
            }
            _ => {
                let origin = [
                    rng.next_range(-2.0, 2.0),
                    rng.next_range(-2.0, 2.0),
                    rng.next_range(-2.0, 2.0),
                ];
                let normal = [
                    rng.next_range(-1.5, 1.5),
                    rng.next_range(0.5, 1.5),
                    rng.next_range(-1.5, 1.5),
                ];
                let distance = rng.next_range(0.2, 2.0);
                let unit = normalize_or_zero(normal);
                // Choose a signed distance clearly above or below -distance.
                let s = if rng.next_u32() & 1 == 0 {
                    -distance - rng.next_range(0.3, 2.0)
                } else {
                    -distance + rng.next_range(0.3, 2.0)
                };
                let pos = add3(origin, scale3(unit, s));
                queries.push(ClothCollisionProjectQuery::backstop(
                    pos, origin, normal, distance,
                ));
            }
        }
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (pos, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert!(
            close3(r.pos, pos),
            "sweep pos mismatch: gpu={:?} cpu={pos:?} query={q:?}",
            r.pos
        );
    }
}
