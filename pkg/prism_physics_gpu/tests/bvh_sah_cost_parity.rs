//! Real-device parity tests for the device-resident `GPU` `BVH` `SAH`-cost reduction.
//!
//! Each test builds the tree resident on device with
//! [`GpuLbvh::build_resident`], folds its surface areas on device with
//! [`GpuBvhSahCost::sah_cost`], and asserts the result matches the
//! [`lbvh_sah_cost`] golden twin over the host-built [`Lbvh`]. The resident
//! internal-node boxes are bit-for-bit those of the twin (exact `min`/`max`
//! unions over order-encoded bounds), but the surface-area *sum* groups
//! differently on device than the host's sequential sum, so the normalised cost
//! is compared within a tight relative tolerance rather than bit-for-bit — the
//! same treatment the `CFL` reducer gets. The exact edge cases (empty tree,
//! single leaf, and a degenerate point root) are compared exactly because no
//! floating-point sum is involved. Every test skips cleanly when no adapter is
//! available (for example inside a sandbox) so the suite never fails for lack of
//! a `GPU`.
//!
//! Provenance: exercises Prism's own resident-tree `SAH` reducer (Goldsmith and
//! Salmon 1987) over the linear `BVH` of Karras (2012) against its `CPU` twin.
//! No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_build_lbvh, cpu_refit_lbvh, lbvh_sah_cost, Aabb, GpuBvhRefit, GpuBvhSahCost, GpuContext,
    GpuLbvh,
};

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

    /// Returns an `f32` in `[lo, lo + span)`.
    fn coord(&mut self, lo: f32, span: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + unit * span
    }
}

/// A box of half-extent `h` centred on a random point in a cube of side
/// `2 * span`.
fn random_box(rng: &mut Rng, span: f32, h: f32) -> Aabb {
    let c = Vec3::new(
        rng.coord(-span, 2.0 * span),
        rng.coord(-span, 2.0 * span),
        rng.coord(-span, 2.0 * span),
    );
    let half = Vec3::splat(h);
    Aabb::new(c - half, c + half)
}

/// A unit box centred on `(x, y, z)`.
fn box_at(x: f32, y: f32, z: f32) -> Aabb {
    let c = Vec3::new(x, y, z);
    let half = Vec3::splat(0.5);
    Aabb::new(c - half, c + half)
}

/// Asserts the resident `GPU` cost of `boxes` matches the `CPU` twin within a
/// relative tolerance, reporting both values on failure.
#[expect(clippy::print_stderr, reason = "surface both costs on parity failure")]
fn assert_cost_matches(
    builder: &GpuLbvh,
    sah: &GpuBvhSahCost,
    ctx: &GpuContext,
    boxes: &[Aabb],
) {
    let want = lbvh_sah_cost(&cpu_build_lbvh(boxes));
    let resident = builder.build_resident(ctx, boxes);
    let got = sah.sah_cost(ctx, &resident);
    ctx.wait();

    // Relative tolerance: internal-node boxes are bit-identical to the twin, but
    // the surface-area sum groups differently on device, so only rounding of the
    // reduction order can differ.
    let tol = 1e-4_f32 * want.abs().max(1.0);
    if (got - want).abs() > tol {
        eprintln!(
            "sah-cost mismatch over {} boxes: gpu {got} vs cpu {want} (tol {tol})",
            boxes.len()
        );
        panic!("resident SAH cost diverged from the twin");
    }
}

#[test]
fn random_scene_matches_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let sah = GpuBvhSahCost::new(&ctx);
    let mut rng = Rng::new(0x5a1c_c057);
    let boxes: Vec<Aabb> = (0..64).map(|_| random_box(&mut rng, 10.0, 0.5)).collect();
    assert_cost_matches(&builder, &sah, &ctx, &boxes);
}

#[test]
fn odd_sized_scene_matches_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let sah = GpuBvhSahCost::new(&ctx);
    // 129 leaves: not a power of two, so the grid-stride fold handles a ragged
    // tail and the internal count (128) differs from the leaf count.
    let mut rng = Rng::new(0x00d_d517);
    let boxes: Vec<Aabb> = (0..129).map(|_| random_box(&mut rng, 25.0, 1.0)).collect();
    assert_cost_matches(&builder, &sah, &ctx, &boxes);
}

#[test]
fn dense_lattice_matches_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let sah = GpuBvhSahCost::new(&ctx);
    // A 4x4x4 lattice half a unit apart: heavy, structured overlap stresses the
    // internal-node bound unions the cost sums over.
    let mut boxes = Vec::new();
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                boxes.push(box_at(x as f32 * 0.5, y as f32 * 0.5, z as f32 * 0.5));
            }
        }
    }
    assert_cost_matches(&builder, &sah, &ctx, &boxes);
}

#[test]
fn empty_tree_costs_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let sah = GpuBvhSahCost::new(&ctx);
    let resident = builder.build_resident(&ctx, &[]);
    let got = sah.sah_cost(&ctx, &resident);
    ctx.wait();
    assert_eq!(got, 0.0, "an empty tree has no query cost");
    assert_eq!(got, lbvh_sah_cost(&cpu_build_lbvh(&[])), "twin agrees");
}

#[test]
fn single_leaf_costs_the_intersection_weight() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let sah = GpuBvhSahCost::new(&ctx);
    let boxes = [box_at(1.0, 2.0, 3.0)];
    let resident = builder.build_resident(&ctx, &boxes);
    let got = sah.sah_cost(&ctx, &resident);
    ctx.wait();
    let want = lbvh_sah_cost(&cpu_build_lbvh(&boxes));
    assert_eq!(got, want, "a lone leaf is its own root: cost is the twin's");
}

#[test]
fn coincident_boxes_fall_back_to_leaf_count() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let sah = GpuBvhSahCost::new(&ctx);
    // Twenty coincident point boxes: the root has zero surface area, so both the
    // device cost and the twin fall back to the intersection-weighted leaf count
    // exactly (no surface-area sum is involved).
    let p = Vec3::new(3.0, -2.0, 1.0);
    let boxes = vec![Aabb::new(p, p); 20];
    let resident = builder.build_resident(&ctx, &boxes);
    let got = sah.sah_cost(&ctx, &resident);
    ctx.wait();
    let want = lbvh_sah_cost(&cpu_build_lbvh(&boxes));
    assert_eq!(got, want, "point root falls back to the twin's leaf count");
}

#[test]
fn degraded_refit_matches_the_refit_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let refit = GpuBvhRefit::new(&ctx);
    let sah = GpuBvhSahCost::new(&ctx);

    // Build a clustered scene, then move the primitives far apart without
    // rebuilding: the kept topology no longer matches the geometry, so the
    // internal-node bounds balloon and the SAH cost rises. The device cost must
    // track the host cpu_refit_lbvh twin over the same stale topology.
    let mut rng = Rng::new(0xbead_f00d);
    let boxes: Vec<Aabb> = (0..96).map(|_| random_box(&mut rng, 4.0, 0.5)).collect();
    let moved: Vec<Aabb> = boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let shift = Vec3::splat(i as f32 * 3.0);
            Aabb::new(b.min + shift, b.max + shift)
        })
        .collect();

    let resident = builder.build_resident(&ctx, &boxes);
    refit.refit(&ctx, &resident, &moved);
    let got = sah.sah_cost(&ctx, &resident);
    ctx.wait();

    let want = lbvh_sah_cost(&cpu_refit_lbvh(&cpu_build_lbvh(&boxes), &moved));
    let tol = 1e-4_f32 * want.abs().max(1.0);
    assert!(
        (got - want).abs() <= tol,
        "refit SAH cost {got} diverged from the twin {want} (tol {tol})"
    );
}
