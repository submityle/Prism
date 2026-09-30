//! Real-device parity tests for the device-resident `GPU` `BVH` overlap query.
//!
//! Each test builds the tree resident on device with
//! [`GpuLbvh::build_resident`], queries it with
//! [`GpuBvhQuery::query_resident`] (no host round-trip between build and
//! traversal), and asserts the resulting candidate set equals the
//! [`cpu_bvh_pairs`] golden twin over the host-built [`Lbvh`] after sorting.
//! The resident kernel decodes the build's order-encoded internal-node bounds
//! with the exact integer inverse of the encoding and reads leaf boxes as the
//! original-order primitive floats, so its overlap tests are bit-for-bit those
//! of the twin; the atomic append order is irrelevant because each unordered
//! pair is emitted once and canonicalised as `a < b`. Every test skips cleanly
//! when no adapter is available (for example inside a sandbox) so the suite
//! never fails for lack of a `GPU`.
//!
//! Provenance: exercises Prism's own resident-tree traversal kernel (Hapala et
//! al. 2011) over the linear `BVH` of Karras (2012) against its `CPU` twin.
//! No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_build_lbvh, cpu_bvh_pairs, Aabb, CandidatePair, GpuBvhQuery, GpuContext, GpuLbvh,
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

/// Sorts a pair list into the canonical order used for comparison.
fn sorted(mut pairs: Vec<CandidatePair>) -> Vec<CandidatePair> {
    pairs.sort_unstable();
    pairs
}

/// Builds the tree resident on device, queries it directly, and asserts the
/// sorted candidate set equals the `CPU` twin's over the host-built tree.
/// `capacity` must exceed the true pair count so neither side overflows.
///
/// On mismatch the message reports the sizes and the first differing pair only,
/// never the whole set, so a large scene cannot flood the test output.
#[expect(clippy::print_stderr, reason = "surface scene size on parity failure")]
fn assert_resident_matches(
    builder: &GpuLbvh,
    query: &GpuBvhQuery,
    ctx: &GpuContext,
    boxes: &[Aabb],
    capacity: u32,
) {
    let tree = cpu_build_lbvh(boxes);
    let want = sorted(cpu_bvh_pairs(&tree, capacity).expect("cpu query within capacity"));
    let resident = builder.build_resident(ctx, boxes);
    let got = sorted(
        query
            .query_resident(ctx, &resident, capacity)
            .expect("gpu resident query within capacity"),
    );
    ctx.wait();

    if got != want {
        eprintln!(
            "pair-count mismatch: gpu {} vs cpu {} over {} boxes",
            got.len(),
            want.len(),
            boxes.len()
        );
        let first_diff = got
            .iter()
            .zip(want.iter())
            .position(|(g, w)| g != w)
            .unwrap_or(got.len().min(want.len()));
        panic!("first differing pair at sorted index {first_diff}");
    }
}

#[test]
fn two_overlapping_boxes_match_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let query = GpuBvhQuery::new(&ctx);
    let boxes = [box_at(0.0, 0.0, 0.0), box_at(0.5, 0.0, 0.0)];
    assert_resident_matches(&builder, &query, &ctx, &boxes, 64);
}

#[test]
fn disjoint_boxes_emit_no_pairs() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let query = GpuBvhQuery::new(&ctx);
    let boxes: Vec<Aabb> = (0..32).map(|k| box_at(k as f32 * 4.0, 0.0, 0.0)).collect();
    assert_resident_matches(&builder, &query, &ctx, &boxes, 64);
}

#[test]
fn single_and_empty_inputs_yield_no_pairs() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let query = GpuBvhQuery::new(&ctx);
    // Fewer than two leaves has no traversable hierarchy: the resident tree
    // owns no buffers and the query must return an empty set without a dispatch.
    for boxes in [Vec::new(), vec![box_at(1.0, 2.0, 3.0)]] {
        let resident = builder.build_resident(&ctx, &boxes);
        let got = query
            .query_resident(&ctx, &resident, 64)
            .expect("trivial resident query within capacity");
        ctx.wait();
        assert!(got.is_empty(), "trivial input must yield no pairs");
    }
}

#[test]
fn dense_lattice_matches_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let query = GpuBvhQuery::new(&ctx);
    // A 4x4x4 lattice half a unit apart: heavy, structured overlap.
    let mut boxes = Vec::new();
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                boxes.push(box_at(x as f32 * 0.5, y as f32 * 0.5, z as f32 * 0.5));
            }
        }
    }
    assert_resident_matches(&builder, &query, &ctx, &boxes, 1 << 16);
}

#[test]
fn duplicate_positions_match_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let query = GpuBvhQuery::new(&ctx);
    // Coincident boxes stress equal Morton codes and total mutual overlap.
    let boxes = vec![box_at(3.0, -2.0, 1.0); 20];
    assert_resident_matches(&builder, &query, &ctx, &boxes, 4096);
}

#[test]
fn degenerate_axis_matches_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let query = GpuBvhQuery::new(&ctx);
    // All boxes share y and z; only x varies, a flat scene the Morton
    // quantisation collapses onto one axis.
    let mut rng = Rng::new(0x0bad_c0de_dead_beef);
    let boxes: Vec<Aabb> = (0..500)
        .map(|_| {
            let x = rng.coord(-10.0, 20.0);
            let c = Vec3::new(x, 0.0, 0.0);
            let half = Vec3::new(0.4, 0.5, 0.5);
            Aabb::new(c - half, c + half)
        })
        .collect();
    assert_resident_matches(&builder, &query, &ctx, &boxes, 1 << 18);
}

#[test]
fn sparse_random_scene_matches_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let query = GpuBvhQuery::new(&ctx);
    // Small boxes spread wide: sparse overlap, exercises deep pruning.
    let mut rng = Rng::new(0x00c0_ffee_1234_5678);
    let boxes: Vec<Aabb> = (0..1000).map(|_| random_box(&mut rng, 40.0, 0.5)).collect();
    assert_resident_matches(&builder, &query, &ctx, &boxes, 1 << 18);
}

#[test]
fn moderate_overlap_large_scene_matches_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let query = GpuBvhQuery::new(&ctx);
    // A larger scene with a controlled overlap density: boxes slightly bigger
    // than their spacing so the pair count stays well within capacity.
    let mut rng = Rng::new(0xfeed_face_c0de_1010);
    let boxes: Vec<Aabb> = (0..4000).map(|_| random_box(&mut rng, 30.0, 0.8)).collect();
    assert_resident_matches(&builder, &query, &ctx, &boxes, 1 << 20);
}
