//! Real-device parity tests for the `GPU` `LBVH` build.
//!
//! Each test builds the hierarchy on the device and asserts bit-for-bit
//! equality with the [`cpu_build_lbvh`] golden twin: the build is a pure
//! integer permutation (Morton sort plus Karras tree) followed by exact
//! componentwise `min`/`max` bounds, identical on host and device, so parity is
//! exact rather than within a tolerance. Every test skips cleanly when no
//! adapter is available (for example inside a sandbox) so the suite never fails
//! for lack of a `GPU`.
//!
//! Provenance: exercises Prism's own Morton, tree, and bounds kernels (Karras,
//! "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees",
//! High Performance Graphics 2012) against their `CPU` twin. No Unreal Engine
//! source or derived code.

use std::time::Instant;

use glam::Vec3;
use prism_physics_gpu::{cpu_build_lbvh, Aabb, GpuContext, GpuLbvh};

/// A small xorshift generator so the tests are deterministic without pulling in
/// an `RNG` dependency.
struct Rng {
    /// Mutable generator state; never zero.
    state: u64,
}

impl Rng {
    /// Seeds the generator, forcing a non-zero state.
    fn new(seed: u64) -> Rng {
        Rng { state: seed | 1 }
    }

    /// Advances the state and returns the next 64-bit value.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Returns an `f32` in `[lo, lo + span)` quantised to a small grid so many
    /// boxes deliberately share Morton codes.
    fn coord(&mut self, lo: f32, span: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + unit * span
    }
}

/// Builds a small box around a random centre.
fn random_box(rng: &mut Rng, span: f32) -> Aabb {
    let c = Vec3::new(
        rng.coord(-span, 2.0 * span),
        rng.coord(-span, 2.0 * span),
        rng.coord(-span, 2.0 * span),
    );
    let h = Vec3::splat(0.25);
    Aabb::new(c - h, c + h)
}

/// Asserts a `GPU` `LBVH` build of `boxes` equals the twin, bit-for-bit.
fn assert_build_matches(builder: &GpuLbvh, ctx: &GpuContext, boxes: &[Aabb]) {
    let want = cpu_build_lbvh(boxes);
    let got = builder.build(ctx, boxes);
    ctx.wait();
    assert_eq!(got.num_leaves, want.num_leaves, "leaf count");
    assert_eq!(got.num_internal, want.num_internal, "internal count");
    assert_eq!(got.root, want.root, "root");
    assert_eq!(got.sorted_indices, want.sorted_indices, "sorted indices");
    assert_eq!(got.sorted_codes, want.sorted_codes, "sorted codes");
    assert_eq!(got.left, want.left, "left children");
    assert_eq!(got.right, want.right, "right children");
    assert_eq!(got.parent, want.parent, "parent links");
    assert_eq!(got.internal_aabb, want.internal_aabb, "internal bounds");
    assert_eq!(got.leaf_aabb, want.leaf_aabb, "leaf bounds");
}

#[test]
fn two_boxes_match_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let boxes = [
        Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
        Aabb::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(6.0, 6.0, 6.0)),
    ];
    assert_build_matches(&builder, &ctx, &boxes);
}

#[test]
fn random_cloud_matches_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let mut rng = Rng::new(0x1234_5678);
    let boxes: Vec<Aabb> = (0..1000).map(|_| random_box(&mut rng, 8.0)).collect();
    assert_build_matches(&builder, &ctx, &boxes);
}

#[test]
fn duplicate_codes_match_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    // Coarse quantisation over a tiny span forces many coincident Morton codes,
    // exercising the index tiebreak in both `delta` implementations.
    let mut rng = Rng::new(0x0bad_c0de);
    let boxes: Vec<Aabb> = (0..500).map(|_| random_box(&mut rng, 0.5)).collect();
    assert_build_matches(&builder, &ctx, &boxes);
}

#[test]
fn degenerate_axis_matches_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    // A perfectly flat scene: every box sits on z = 0, so that axis has zero
    // extent and must quantise to bucket zero on host and device alike.
    let mut rng = Rng::new(0xfeed_face);
    let boxes: Vec<Aabb> = (0..300)
        .map(|_| {
            let c = Vec3::new(rng.coord(-4.0, 8.0), rng.coord(-4.0, 8.0), 0.0);
            Aabb::new(c, c)
        })
        .collect();
    assert_build_matches(&builder, &ctx, &boxes);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "reports build timing for a large scene"
)]
fn large_scene_matches_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let mut rng = Rng::new(0xdead_beef);
    let boxes: Vec<Aabb> = (0..100_000).map(|_| random_box(&mut rng, 64.0)).collect();
    let start = Instant::now();
    let got = builder.build(&ctx, &boxes);
    ctx.wait();
    let elapsed = start.elapsed();
    let want = cpu_build_lbvh(&boxes);
    assert_eq!(got.sorted_indices, want.sorted_indices, "sorted indices");
    assert_eq!(got.sorted_codes, want.sorted_codes, "sorted codes");
    assert_eq!(got.left, want.left, "left children");
    assert_eq!(got.right, want.right, "right children");
    assert_eq!(got.parent, want.parent, "parent links");
    assert_eq!(got.internal_aabb, want.internal_aabb, "internal bounds");
    eprintln!("GPU LBVH build of {} leaves in {elapsed:?}", boxes.len());
}
