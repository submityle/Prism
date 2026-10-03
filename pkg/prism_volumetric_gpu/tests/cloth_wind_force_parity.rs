//! Real-device parity for the cloth wind-force twin:
//! [`GpuClothWindForce`](prism_volumetric_gpu::cloth_wind_force::GpuClothWindForce)
//! must reproduce the `CPU` golden
//! [`triangle_wind_force`](prism_render_architecture::cloth::wind::triangle_wind_force)
//! across the linear and quadratic pressure models, plus a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected force is produced by calling the golden `triangle_wind_force`
//! directly with the hand-rolled render [`Vec3`](prism_render_architecture::cloth::Vec3)
//! and [`AeroParams`](prism_render_architecture::cloth::wind::AeroParams), so
//! the test pins `GPU == golden`, not merely that the shader compiles.
//!
//! # Parity criterion
//!
//! The force is a continuous `f32` quantity, so each component is compared with
//! the tolerance `abs <= 1e-4` or `rel <= 1e-3` (relative floor `1e-6`). The
//! randomized sweep rejects near-degenerate triangles so the squared edge-cross
//! stays well away from the `1e-12` degeneracy threshold.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::wind`；无第三方引擎源码或衍生代码。

use prism_render_architecture::cloth::wind::{triangle_wind_force, AeroParams};
use prism_render_architecture::cloth::Vec3;
use prism_volumetric_gpu::cloth_wind_force::{
    ClothWindForceQuery, ClothWindForceResult, GpuClothWindForce,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for a continuous `f32` parity comparison.
const EPS: f32 = 1e-4;
/// Relative tolerance for a continuous `f32` parity comparison.
const REL: f32 = 1e-3;
/// Floor on the relative-tolerance denominator so tiny magnitudes stay stable.
const REL_FLOOR: f32 = 1e-6;

/// Returns `true` when `a` and `b` agree within the crate's continuous
/// tolerance (`abs <= 1e-4` or `rel <= 1e-3`, floor `1e-6`).
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Computes the in-host golden force for one query by calling the reference
/// `triangle_wind_force` directly.
fn oracle(q: &ClothWindForceQuery) -> [f32; 3] {
    let v = |a: [f32; 3]| Vec3::new(a[0], a[1], a[2]);
    let aero = AeroParams::new(q.drag, q.lift).with_air_density(q.air_density);
    let force = triangle_wind_force(
        v(q.p0),
        v(q.p1),
        v(q.p2),
        v(q.v0),
        v(q.v1),
        v(q.v2),
        v(q.wind),
        aero,
    );
    [force.x, force.y, force.z]
}

/// Pins one `GPU` result against the in-host oracle component-wise.
fn check_query(idx: usize, q: &ClothWindForceQuery, got: &ClothWindForceResult) {
    let want = oracle(q);
    for axis in 0..3 {
        assert!(
            close(got.force[axis], want[axis]),
            "query {idx} axis {axis}: gpu {got:?} golden {want:?}",
        );
    }
}

/// A tiny host-side `LCG` producing a stream of `u32` words; used only to drive
/// the randomized fixture sweep (no `GPU` state depends on it).
struct Lcg {
    state: u64,
}

impl Lcg {
    /// Seeds the generator.
    fn new(seed: u64) -> Lcg {
        Lcg { state: seed }
    }

    /// Advances the state and returns the next `u32` word.
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 33) as u32
    }

    /// Returns the next `f32` uniformly in `[lo, hi)`.
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0);
        lo + (hi - lo) * unit
    }
}

/// Returns the squared length of the edge-cross for a candidate triangle, used
/// by the sweep to reject near-degenerate fixtures.
fn cross_len_sq(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]) -> f32 {
    let a = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let b = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let cx = a[1] * b[2] - a[2] * b[1];
    let cy = a[2] * b[0] - a[0] * b[2];
    let cz = a[0] * b[1] - a[1] * b[0];
    cx * cx + cy * cy + cz * cz
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuClothWindForce::new(&ctx);
    let out = twin.evaluate(&ctx, &[]);
    assert!(out.is_empty(), "empty batch must return an empty vector");
}

#[test]
fn linear_model_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuClothWindForce::new(&ctx);

    // A unit XY triangle (normal along +Z), wind straight down the normal: a
    // pure-drag case under the linear (air_density = 0) model.
    let drag_face = ClothWindForceQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0],
        0.8,
        0.2,
        0.0,
    );
    // Same triangle, wind in-plane: a pure-lift case.
    let lift_face = ClothWindForceQuery::new(
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0],
        [1.5, -0.5, 0.0],
        1.0,
        0.5,
        0.0,
    );
    // A tilted triangle with mixed wind and moving vertices.
    let tilted = ClothWindForceQuery::new(
        [0.3, -0.4, 0.2],
        [1.7, 0.1, -0.5],
        [-0.6, 1.3, 0.9],
        [0.05, -0.02, 0.1],
        [-0.03, 0.04, 0.0],
        [0.01, 0.0, -0.06],
        [2.0, 1.0, -1.5],
        0.6,
        0.9,
        0.0,
    );

    let queries = [drag_face, lift_face, tilted];
    let out = twin.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (idx, (q, got)) in queries.iter().zip(out.iter()).enumerate() {
        check_query(idx, q, got);
    }
}

#[test]
fn quadratic_model_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuClothWindForce::new(&ctx);

    // Quadratic (air_density > 0) model: dynamic pressure scales with airspeed.
    let slow = ClothWindForceQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
        1.0,
        0.3,
        1.225,
    );
    // Doubling the airspeed should quadruple the quadratic force (checked by the
    // oracle, so parity still just compares to the golden).
    let fast = ClothWindForceQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 4.0],
        1.0,
        0.3,
        1.225,
    );
    let tilted = ClothWindForceQuery::new(
        [-0.5, 0.2, 0.4],
        [1.1, -0.3, 0.0],
        [0.2, 1.4, -0.7],
        [0.02, 0.01, -0.03],
        [-0.01, 0.0, 0.02],
        [0.0, -0.02, 0.01],
        [1.0, 2.5, -0.5],
        0.7,
        1.1,
        0.9,
    );

    let queries = [slow, fast, tilted];
    let out = twin.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (idx, (q, got)) in queries.iter().zip(out.iter()).enumerate() {
        check_query(idx, q, got);
    }
}

#[test]
fn randomized_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuClothWindForce::new(&ctx);

    let mut rng = Lcg::new(0x5F37_59DF_1234_ABCD);
    let mut queries: Vec<ClothWindForceQuery> = Vec::with_capacity(512);
    while queries.len() < 512 {
        let p0 = [
            rng.next_f32(-2.0, 2.0),
            rng.next_f32(-2.0, 2.0),
            rng.next_f32(-2.0, 2.0),
        ];
        let p1 = [
            rng.next_f32(-2.0, 2.0),
            rng.next_f32(-2.0, 2.0),
            rng.next_f32(-2.0, 2.0),
        ];
        let p2 = [
            rng.next_f32(-2.0, 2.0),
            rng.next_f32(-2.0, 2.0),
            rng.next_f32(-2.0, 2.0),
        ];
        // Reject near-degenerate triangles so the squared edge-cross stays far
        // from the 1e-12 degeneracy threshold (keeps parity off the branch tie).
        if cross_len_sq(p0, p1, p2) < 0.25 {
            continue;
        }
        let v0 = [
            rng.next_f32(-0.5, 0.5),
            rng.next_f32(-0.5, 0.5),
            rng.next_f32(-0.5, 0.5),
        ];
        let v1 = [
            rng.next_f32(-0.5, 0.5),
            rng.next_f32(-0.5, 0.5),
            rng.next_f32(-0.5, 0.5),
        ];
        let v2 = [
            rng.next_f32(-0.5, 0.5),
            rng.next_f32(-0.5, 0.5),
            rng.next_f32(-0.5, 0.5),
        ];
        let wind = [
            rng.next_f32(-3.0, 3.0),
            rng.next_f32(-3.0, 3.0),
            rng.next_f32(-3.0, 3.0),
        ];
        let drag = rng.next_f32(0.0, 2.0);
        let lift = rng.next_f32(0.0, 2.0);
        // Alternate between the linear (0) and quadratic (> 0) pressure models.
        let air_density = if queries.len().is_multiple_of(2) {
            0.0
        } else {
            rng.next_f32(0.3, 2.0)
        };
        queries.push(ClothWindForceQuery::new(
            p0,
            p1,
            p2,
            v0,
            v1,
            v2,
            wind,
            drag,
            lift,
            air_density,
        ));
    }

    let out = twin.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (idx, (q, got)) in queries.iter().zip(out.iter()).enumerate() {
        check_query(idx, q, got);
    }
}
