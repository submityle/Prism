//! Real-device parity for the oriented-bounding-box corners twin:
//! [`GpuObbCorners`](prism_volumetric_gpu::obb_corners::GpuObbCorners) must
//! reproduce the `CPU` golden `Obb::corners` of
//! `prism_physics_core::collider::obb`, which expands a compact box — a center,
//! three orthonormal axes and the box half-extents — into its eight world-space
//! corners by scaling each axis by its half-extent and adding the three signed
//! half-axis vectors to the center in a fixed sign pattern.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the three scaled half-axes and the eight signed sums in the golden's order —
//! written out directly in flat `f32` array math so the test never imports
//! `prism_physics_core`, `prism_render_architecture` or `glam`. It mirrors the
//! reference operation for operation and in the same evaluation order.
//!
//! The fixtures cover the regimes the kernel must honor: an axis-aligned unit
//! box at the origin (checked against literal corner values), a box rotated
//! `90` degrees about `Z`, an arbitrary orthonormal rotation at a non-zero
//! center, a multi-element mixed batch that validates the `std430` array stride
//! end to end, plus an empty batch the host short-circuits with no dispatch. A
//! sweep over random orthonormal frames, centers and half-extents follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each corner threads through a few multiply-adds, so `CPU` and `GPU` evaluate
//! the same closed form but need not be bit-exact. The continuous comparison is
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on each of the
//! twenty-four corner components; the discrete `valid` flag is compared
//! exactly. The closed form has no degenerate branch, so `valid` is always `1`
//! and no comparison sits on a branch knife edge.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::obb::Obb::corners`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::obb_corners::{GpuObbCorners, ObbCornersQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Returns `true` when two 3-vectors agree component-wise within tolerance.
fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
    (0..3).all(|k| close(a[k], b[k]))
}

/// Independent host re-implementation of the golden `Obb::corners`, returning
/// the eight world-space corners and the `valid` flag without importing the
/// golden crate or `glam`. The three scaled half-axes and the eight signed sums
/// are evaluated in the same order as the kernel: `hx` sign cycles fastest,
/// `hz` sign slowest.
fn oracle(q: &ObbCornersQuery) -> ([[f32; 3]; 8], u32) {
    let hx = [
        q.axis0[0] * q.half_extents[0],
        q.axis0[1] * q.half_extents[0],
        q.axis0[2] * q.half_extents[0],
    ];
    let hy = [
        q.axis1[0] * q.half_extents[1],
        q.axis1[1] * q.half_extents[1],
        q.axis1[2] * q.half_extents[1],
    ];
    let hz = [
        q.axis2[0] * q.half_extents[2],
        q.axis2[1] * q.half_extents[2],
        q.axis2[2] * q.half_extents[2],
    ];
    let c = q.center;
    // sx, sy, sz are the per-corner signs for hx, hy, hz respectively.
    let corner = |sx: f32, sy: f32, sz: f32| {
        [
            c[0] + sx * hx[0] + sy * hy[0] + sz * hz[0],
            c[1] + sx * hx[1] + sy * hy[1] + sz * hz[1],
            c[2] + sx * hx[2] + sy * hy[2] + sz * hz[2],
        ]
    };
    let corners = [
        corner(-1.0, -1.0, -1.0),
        corner(1.0, -1.0, -1.0),
        corner(-1.0, 1.0, -1.0),
        corner(1.0, 1.0, -1.0),
        corner(-1.0, -1.0, 1.0),
        corner(1.0, -1.0, 1.0),
        corner(-1.0, 1.0, 1.0),
        corner(1.0, 1.0, 1.0),
    ];
    (corners, 1)
}

/// Dispatches one box and asserts the `GPU` result matches the oracle on all
/// eight corners (within tolerance) and the `valid` flag (exactly).
fn assert_parity(ctx: &GpuContext, gpu: &GpuObbCorners, q: ObbCornersQuery) {
    let r = gpu.evaluate(ctx, std::slice::from_ref(&q))[0];
    let (corners, valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    for (i, (g, c)) in r.corners.iter().zip(corners.iter()).enumerate() {
        assert!(
            close3(*g, *c),
            "corner {i} mismatch: gpu={g:?} cpu={c:?} query={q:?}"
        );
    }
}

#[test]
fn axis_aligned_unit_box_matches_literals() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbCorners::new(&ctx);
    // A unit cube centered at the origin with world-aligned axes: the eight
    // corners are the +/-1 sign combinations in the golden's order.
    let q = ObbCornersQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    );
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    let expected = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    for (i, (g, e)) in r.corners.iter().zip(expected.iter()).enumerate() {
        assert!(
            close3(*g, *e),
            "unit box corner {i} mismatch: gpu={g:?} expected={e:?}"
        );
    }
}

#[test]
fn rotated_about_z_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbCorners::new(&ctx);
    // A box rotated 90 degrees about Z: axis0 maps to +Y, axis1 to -X, axis2
    // stays +Z. With half-extents (2, 1, 1) this exercises distinct extents on
    // swapped axes, so a transposed or mis-ordered sign pattern would show up.
    let q = ObbCornersQuery::new(
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
    );
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    // c7 = center + hx + hy + hz = (0,2,0) + (-1,0,0) + (0,0,1) = (-1, 2, 1).
    assert!(
        close3(r.corners[7], [-1.0, 2.0, 1.0]),
        "rotated c7 mismatch: got {:?}",
        r.corners[7]
    );
}

#[test]
fn arbitrary_rotation_nonzero_center_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbCorners::new(&ctx);
    // An arbitrary orthonormal frame (an orthonormal 3x3 rotation) at a
    // non-zero center with distinct half-extents.
    let (axis0, axis1, axis2) = frame_from_quat(0.3, -0.5, 0.4, 0.7);
    let q = ObbCornersQuery::new([2.0, -1.5, 3.0], axis0, axis1, axis2, [1.3, 0.7, 2.1]);
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbCorners::new(&ctx);
    // A multi-element mixed batch (axis-aligned, rotated, arbitrary frame)
    // exercises the std430 array stride: every slot must decode at the right
    // byte offset.
    let (ax0, ax1, ax2) = frame_from_quat(-0.2, 0.6, 0.1, 0.9);
    let queries = [
        ObbCornersQuery::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ),
        ObbCornersQuery::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
        ),
        ObbCornersQuery::new([-4.0, 2.0, 1.0], ax0, ax1, ax2, [0.5, 2.5, 1.2]),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (corners, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        for (i, (g, c)) in r.corners.iter().zip(corners.iter()).enumerate() {
            assert!(
                close3(*g, *c),
                "batch corner {i} mismatch: gpu={g:?} cpu={c:?} query={q:?}"
            );
        }
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbCorners::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
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

/// Builds an orthonormal frame — the three box axes — from a quaternion using
/// the `glam` column-major quaternion-to-matrix basis. The three returned
/// columns are mutually orthonormal when the quaternion is normalized, matching
/// the golden's `Obb` axis contract without importing `glam`.
fn frame_from_quat(x: f32, y: f32, z: f32, w: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let inv = 1.0 / (x * x + y * y + z * z + w * w).sqrt();
    let (x, y, z, w) = (x * inv, y * inv, z * inv, w * inv);
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;
    let col0 = [1.0 - (yy + zz), xy + wz, xz - wy];
    let col1 = [xy - wz, 1.0 - (xx + zz), yz + wx];
    let col2 = [xz + wy, yz - wx, 1.0 - (xx + yy)];
    (col0, col1, col2)
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuObbCorners::new(&ctx);
    let mut rng = Lcg::new(0x3C_9A_71_E2);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // A normalized quaternion away from zero norm yields a well-formed
        // orthonormal frame.
        let a = rng.next_range(-1.0, 1.0);
        let b = rng.next_range(-1.0, 1.0);
        let c = rng.next_range(-1.0, 1.0);
        let d = rng.next_range(-1.0, 1.0);
        if a * a + b * b + c * c + d * d <= 0.04 {
            continue;
        }
        let (axis0, axis1, axis2) = frame_from_quat(a, b, c, d);
        let center = [
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
        ];
        let half_extents = [
            rng.next_range(0.1, 3.0),
            rng.next_range(0.1, 3.0),
            rng.next_range(0.1, 3.0),
        ];
        queries.push(ObbCornersQuery::new(
            center,
            axis0,
            axis1,
            axis2,
            half_extents,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (corners, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        for (i, (g, c)) in r.corners.iter().zip(corners.iter()).enumerate() {
            assert!(
                close3(*g, *c),
                "sweep corner {i} mismatch: gpu={g:?} cpu={c:?} query={q:?}"
            );
        }
    }
}
