//! Real-device tests for the resident `BVH` refit-versus-rebuild driver.
//!
//! Each test drives a [`ResidentBvhDriver`] over a sequence of frames and checks
//! both the policy decision ([`UpdateAction`]) and the resulting `SAH` cost
//! against the [`lbvh_sah_cost`] / [`cpu_refit_lbvh`] golden twins. The driver's
//! refit path keeps the resident topology (so the cost tracks `cpu_refit_lbvh`
//! over the kept tree), and its rebuild path reseeds from a fresh `cpu_build_lbvh`
//! twin; the surface-area sum groups differently on device, so costs are matched
//! within a tight relative tolerance. Every test skips cleanly when no adapter
//! is available so the suite never fails for lack of a `GPU`.
//!
//! Provenance: exercises Prism's own composed builder, refit, `SAH` reducer, and
//! surface-area rebuild policy. No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_build_lbvh, cpu_refit_lbvh, lbvh_sah_cost, Aabb, GpuContext, ResidentBvhDriver,
    UpdateAction,
};

/// A small xorshift generator so the tests are deterministic.
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

/// Shifts every box by `delta`, keeping extents (and so the leaf count) fixed.
fn translate(boxes: &[Aabb], delta: Vec3) -> Vec<Aabb> {
    boxes
        .iter()
        .map(|b| Aabb::new(b.min + delta, b.max + delta))
        .collect()
}

/// Asserts `got` matches `want` within a relative tolerance.
fn close(got: f32, want: f32) -> bool {
    (got - want).abs() <= 1e-4_f32 * want.abs().max(1.0)
}

#[test]
fn small_motion_refits_and_tracks_the_refit_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut rng = Rng::new(0x51ab_1e01);
    let boxes: Vec<Aabb> = (0..80).map(|_| random_box(&mut rng, 20.0, 0.75)).collect();
    let mut driver = ResidentBvhDriver::build(&ctx, &boxes);

    // A tiny rigid drift keeps the topology ideal, so the driver must refit, and
    // its cost must equal the cpu_refit_lbvh twin over the kept tree.
    let moved = translate(&boxes, Vec3::splat(0.05));
    let update = driver.update(&ctx, &moved);
    ctx.wait();

    assert_eq!(update.action, UpdateAction::Refit, "small motion stays a refit");
    let want = lbvh_sah_cost(&cpu_refit_lbvh(&cpu_build_lbvh(&boxes), &moved));
    assert!(
        close(update.cost, want),
        "refit cost {} vs twin {want}",
        update.cost
    );
    assert_eq!(driver.leaf_count(), boxes.len(), "topology leaf count is kept");
}

#[test]
fn cost_growth_forces_a_rebuild_to_the_fresh_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut rng = Rng::new(0x600d_5eed);
    // A tight cluster so the baseline cost is small and easy to inflate.
    let boxes: Vec<Aabb> = (0..64).map(|_| random_box(&mut rng, 2.0, 0.5)).collect();
    let mut driver = ResidentBvhDriver::build_with_policy(&ctx, &boxes, 1.5, 1_000);

    // Fan the primitives far apart: the kept topology's bounds balloon, driving
    // the SAH cost well past 1.5x the baseline, so the driver must rebuild.
    let scattered: Vec<Aabb> = boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let shift = Vec3::splat(i as f32 * 8.0);
            Aabb::new(b.min + shift, b.max + shift)
        })
        .collect();
    let update = driver.update(&ctx, &scattered);
    ctx.wait();

    assert_eq!(
        update.action,
        UpdateAction::Rebuild,
        "a large cost jump forces a rebuild"
    );
    // After a rebuild the resident cost must match a fresh build over the same
    // boxes, and the baseline must be reseeded to it.
    let want = lbvh_sah_cost(&cpu_build_lbvh(&scattered));
    assert!(
        close(update.cost, want),
        "rebuilt cost {} vs fresh twin {want}",
        update.cost
    );
    assert!(
        close(driver.tracker().baseline_cost(), update.cost),
        "baseline reseeded to the rebuilt cost"
    );
    assert_eq!(
        driver.tracker().refits_since_rebuild(),
        0,
        "rebuild clears the staleness counter"
    );
}

#[test]
fn staleness_forces_a_rebuild_after_the_bound() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut rng = Rng::new(0x57a1_e000);
    let boxes: Vec<Aabb> = (0..48).map(|_| random_box(&mut rng, 15.0, 0.5)).collect();
    // A huge cost factor disables the cost trigger, isolating staleness: rebuild
    // only after exactly three refits.
    let mut driver = ResidentBvhDriver::build_with_policy(&ctx, &boxes, 1.0e9, 3);

    // Three identical updates: refit, refit, then the staleness bound fires.
    let actions: Vec<UpdateAction> = (0..3)
        .map(|_| driver.update(&ctx, &boxes).action)
        .collect();
    ctx.wait();

    assert_eq!(
        actions,
        vec![
            UpdateAction::Refit,
            UpdateAction::Refit,
            UpdateAction::Rebuild
        ],
        "the third update hits the staleness bound"
    );
    assert_eq!(
        driver.tracker().refits_since_rebuild(),
        0,
        "the rebuild reset the staleness counter"
    );
}

#[test]
fn leaf_count_change_forces_a_rebuild() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut rng = Rng::new(0x1ea7_c047);
    let boxes: Vec<Aabb> = (0..50).map(|_| random_box(&mut rng, 12.0, 0.5)).collect();
    let mut driver = ResidentBvhDriver::build(&ctx, &boxes);

    // Drop a primitive: no compatible topology exists, so the driver rebuilds
    // and the resident cost matches a fresh build over the smaller set.
    let fewer = boxes[..40].to_vec();
    let update = driver.update(&ctx, &fewer);
    ctx.wait();

    assert_eq!(
        update.action,
        UpdateAction::Rebuild,
        "a leaf-count change cannot be refit"
    );
    assert_eq!(driver.leaf_count(), fewer.len(), "topology adopts the new count");
    let want = lbvh_sah_cost(&cpu_build_lbvh(&fewer));
    assert!(
        close(update.cost, want),
        "rebuilt cost {} vs fresh twin {want}",
        update.cost
    );
}
