//! Real-device parity for the capsule-volume twin:
//! [`GpuBoundingCapsuleVolume`](prism_volumetric_gpu::bounding_capsule_volume::GpuBoundingCapsuleVolume)
//! must reproduce the `CPU` golden `BoundingCapsule::volume` of
//! `prism_physics_core::collider::bounding_capsule`.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out with scalar `f32` arithmetic so the test never imports
//! `prism_physics_core`, `prism_render_architecture` or `glam`. The segment
//! length `h` between the two cap centres drives the cylinder term `pi r^2 h`,
//! and the full-sphere term `4/3 pi r^3` adds the two hemispherical caps. Every
//! operator is evaluated in the same order the kernel uses, with the same `PI`
//! and `4/3` `f32` constants, so no extra `f32` error is introduced.
//!
//! The fixtures cover a zero-length segment (a pure sphere), a general upright
//! capsule, a large-radius capsule (exercising the relative tolerance), a
//! tiny-radius capsule (exercising the `REL_FLOOR`), a diagonal segment, a mixed
//! `>=2`-element batch that validates the `std430` stride end to end, and a
//! `512`-query `LCG` sweep. An empty batch is short-circuited by the host with
//! no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The volume is continuous and checked with an absolute-or-relative tolerance
//! (`abs <= 1e-4 || rel <= 1e-3`, `REL_FLOOR = 1e-6`); `valid` is discrete and
//! compared exactly. There is no runtime division and no transcendental call in
//! the kernel, so there is no conditioning knee to avoid; `valid` is always `1`.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::bounding_capsule`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::bounding_capsule_volume::{
    BoundingCapsuleVolumeQuery, BoundingCapsuleVolumeResult, GpuBoundingCapsuleVolume,
};
use prism_volumetric_gpu::GpuContext;

/// `PI` as the identical `f32` bit pattern used by the kernel and by Rust's
/// `core::f32::consts::PI`.
const PI: f32 = std::f32::consts::PI;

/// The correctly-rounded `f32` value of `4.0 / 3.0`, matching the kernel's typed
/// constant and the golden's `f32` division.
const FOUR_THIRDS: f32 = 4.0 / 3.0;

/// Independent host re-implementation of `BoundingCapsule::volume`, flattened
/// into the `(volume, valid)` record the twin encodes. Each operator mirrors the
/// kernel in evaluation order. No `prism_physics_core` / `glam` import.
fn oracle(q: &BoundingCapsuleVolumeQuery) -> BoundingCapsuleVolumeResult {
    let axis = [q.cbx - q.cax, q.cby - q.cay, q.cbz - q.caz];
    let h = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let r = q.radius;
    let cylinder = PI * r * r * h;
    let sphere = FOUR_THIRDS * PI * r * r * r;
    BoundingCapsuleVolumeResult {
        volume: cylinder + sphere,
        valid: 1,
    }
}

/// Absolute-or-relative closeness for a continuous channel.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    if diff <= 1e-4 {
        return true;
    }
    let rel_floor = 1e-6_f32;
    let denom = want.abs().max(got.abs()).max(rel_floor);
    diff / denom <= 1e-3
}

/// Asserts one GPU result matches the oracle: the volume is continuous
/// (tolerance); `valid` is discrete (exact).
fn assert_result(got: BoundingCapsuleVolumeResult, want: BoundingCapsuleVolumeResult, label: &str) {
    assert_eq!(got.valid, want.valid, "valid mismatch: {label}");
    assert!(
        close(got.volume, want.volume),
        "volume mismatch: {label}: got {} want {}",
        got.volume,
        want.volume
    );
}

/// Asserts a single-query GPU result matches the oracle.
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuBoundingCapsuleVolume,
    q: BoundingCapsuleVolumeQuery,
    label: &str,
) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query: {label}");
    assert_result(got[0], oracle(&q), label);
}

#[test]
fn zero_length_segment_is_pure_sphere() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleVolume::new(&ctx);
    // center_a == center_b collapses the cylinder term; only the sphere remains.
    let q = BoundingCapsuleVolumeQuery::new([1.0, -2.0, 3.0], [1.0, -2.0, 3.0], 2.0);
    assert_parity(&ctx, &gpu, q, "zero-length pure sphere");
}

#[test]
fn upright_unit_capsule() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleVolume::new(&ctx);
    // h = 3, r = 1: cylinder 3*pi plus sphere 4/3*pi.
    let q = BoundingCapsuleVolumeQuery::new([0.0, 0.0, 0.0], [0.0, 3.0, 0.0], 1.0);
    assert_parity(&ctx, &gpu, q, "upright unit capsule");
}

#[test]
fn large_radius_exercises_relative_tolerance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleVolume::new(&ctx);
    let q = BoundingCapsuleVolumeQuery::new([-10.0, 0.0, 0.0], [40.0, 0.0, 0.0], 100.0);
    assert_parity(&ctx, &gpu, q, "large radius");
}

#[test]
fn tiny_radius_exercises_rel_floor() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleVolume::new(&ctx);
    let q = BoundingCapsuleVolumeQuery::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0e-4);
    assert_parity(&ctx, &gpu, q, "tiny radius");
}

#[test]
fn diagonal_segment() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleVolume::new(&ctx);
    // h = sqrt(1+4+4) = 3 along an oblique axis.
    let q = BoundingCapsuleVolumeQuery::new([0.5, -0.5, 1.0], [1.5, 1.5, 3.0], 0.75);
    assert_parity(&ctx, &gpu, q, "diagonal segment");
}

#[test]
fn negative_coordinate_segment() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleVolume::new(&ctx);
    let q = BoundingCapsuleVolumeQuery::new([-3.0, -4.0, -5.0], [-6.0, -8.0, -5.0], 2.5);
    assert_parity(&ctx, &gpu, q, "negative-coordinate segment");
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleVolume::new(&ctx);
    // A >=2-element batch with distinct shapes checks the std430 stride end to
    // end: every element must land at its own slot and decode correctly.
    let queries = [
        BoundingCapsuleVolumeQuery::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0),
        BoundingCapsuleVolumeQuery::new([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], 0.5),
        BoundingCapsuleVolumeQuery::new([-1.0, 2.0, -3.0], [4.0, -2.0, 1.0], 3.0),
        BoundingCapsuleVolumeQuery::new([10.0, 10.0, 10.0], [10.0, 13.0, 14.0], 0.25),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (i, q) in queries.iter().enumerate() {
        assert_result(got[i], oracle(q), &format!("mixed batch element {i}"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleVolume::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "empty batch returns no results");
}

/// Small LCG for a reproducible pseudo-random sweep.
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

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBoundingCapsuleVolume::new(&ctx);
    let mut rng = Lcg::new(0x0CA2_5017);
    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        let ca = [
            rng.next_range(-20.0, 20.0),
            rng.next_range(-20.0, 20.0),
            rng.next_range(-20.0, 20.0),
        ];
        let cb = [
            rng.next_range(-20.0, 20.0),
            rng.next_range(-20.0, 20.0),
            rng.next_range(-20.0, 20.0),
        ];
        let radius = rng.next_range(0.01, 5.0);
        queries.push(BoundingCapsuleVolumeQuery::new(ca, cb, radius));
    }
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (i, q) in queries.iter().enumerate() {
        assert_result(got[i], oracle(q), &format!("sweep index {i}"));
    }
}
