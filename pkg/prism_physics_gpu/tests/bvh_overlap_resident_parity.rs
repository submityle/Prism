//! Real-device parity tests for the device-resident `GPU` batched
//! `AABB`-versus-`BVH` overlap gather query.
//!
//! Each test builds the tree resident on device with
//! [`GpuLbvh::build_resident`], runs the batched overlap query directly against
//! those device buffers with [`GpuBvhOverlap::query_resident`] (no host
//! round-trip between build and traversal), and asserts the result equals the
//! [`cpu_bvh_aabb_overlap`] golden twin over the host-built [`Lbvh`] after
//! sorting each query's hit list. The resident kernel decodes the build's
//! order-encoded internal-node bounds with the exact integer inverse of the
//! encoding and reads leaf boxes as the original-order primitive floats, so its
//! overlap tests are bit-for-bit those of the twin; the device's per-query
//! atomic append reorders each query's hits, which the sorted comparison
//! ignores. Every test skips cleanly when no adapter is available (for example
//! inside a sandbox) so the suite never fails for lack of a `GPU`.
//!
//! A resident tree with fewer than two leaves owns no traversable hierarchy, so
//! the trivial-input test asserts empty hit lists directly rather than against
//! the twin (whose single-leaf tree does report hits).
//!
//! Provenance: exercises Prism's own resident-tree stackless traversal kernel
//! (Hapala et al. 2011) over the linear `BVH` of Karras (2012) against its
//! `CPU` twin. No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_build_lbvh, cpu_bvh_aabb_overlap, Aabb, GpuBvhOverlap, GpuContext, GpuLbvh,
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

/// A cube of half-extent `h` centred on `(x, y, z)`.
fn box_at(x: f32, y: f32, z: f32, h: f32) -> Aabb {
    let c = Vec3::new(x, y, z);
    let half = Vec3::splat(h);
    Aabb::new(c - half, c + half)
}

/// Sorts one query's hit list so a query's set can be compared independent of
/// traversal order.
fn sorted(mut v: Vec<u32>) -> Vec<u32> {
    v.sort_unstable();
    v
}

/// Builds the tree resident on device, runs the batch of `queries` against it
/// directly, and asserts the result agrees query-for-query with the twin over
/// the host-built tree after sorting each query's hits. `capacity` must exceed
/// every query's true overlap count so neither side overflows.
///
/// On mismatch the message reports the offending query and the two sizes only,
/// never the whole set, so a large scene cannot flood the test output.
#[expect(clippy::print_stderr, reason = "surface query index on parity failure")]
fn assert_resident_overlap_matches(
    builder: &GpuLbvh,
    overlap: &GpuBvhOverlap,
    ctx: &GpuContext,
    leaves: &[Aabb],
    queries: &[Aabb],
    capacity: u32,
) {
    let tree = cpu_build_lbvh(leaves);
    let want = cpu_bvh_aabb_overlap(&tree, queries, capacity).expect("cpu query within capacity");
    let resident = builder.build_resident(ctx, leaves);
    let got = overlap
        .query_resident(ctx, &resident, queries, capacity)
        .expect("gpu resident query within capacity");
    ctx.wait();

    assert_eq!(
        got.len(),
        want.len(),
        "query count mismatch: gpu {} vs cpu {}",
        got.len(),
        want.len()
    );
    for (q, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        if sorted(g.clone()) != sorted(w.clone()) {
            eprintln!(
                "overlap mismatch at query {q}: gpu {} hits vs cpu {} hits over {} leaves",
                g.len(),
                w.len(),
                leaves.len()
            );
            panic!("query {q} overlap set differs between gpu and cpu");
        }
    }
}

#[test]
fn trivial_inputs_yield_empty_hit_lists() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let overlap = GpuBvhOverlap::new(&ctx);
    // Fewer than two leaves has no traversable hierarchy: the resident tree owns
    // no buffers, so every query returns an empty hit list without a dispatch,
    // even for a query that would overlap the lone leaf.
    let queries = [box_at(1.0, 2.0, 3.0, 0.5), box_at(9.0, 9.0, 9.0, 0.5)];
    for leaves in [Vec::new(), vec![box_at(1.0, 2.0, 3.0, 0.5)]] {
        let resident = builder.build_resident(&ctx, &leaves);
        let got = overlap
            .query_resident(&ctx, &resident, &queries, 8)
            .expect("trivial resident query within capacity");
        ctx.wait();
        assert_eq!(got.len(), queries.len(), "one hit list per query");
        for hits in &got {
            assert!(hits.is_empty(), "trivial input must yield no hits");
        }
    }
}

#[test]
fn empty_query_batch_yields_empty_outer() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let overlap = GpuBvhOverlap::new(&ctx);
    let leaves: Vec<Aabb> = (0..16).map(|k| box_at(k as f32, 0.0, 0.0, 0.5)).collect();
    let resident = builder.build_resident(&ctx, &leaves);
    let got = overlap
        .query_resident(&ctx, &resident, &[], 8)
        .expect("empty-batch resident query within capacity");
    ctx.wait();
    assert!(got.is_empty(), "empty query batch yields an empty outer vector");
}

#[test]
fn grid_with_mixed_size_queries_matches_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let overlap = GpuBvhOverlap::new(&ctx);
    // A 6x6x6 lattice of unit cubes one unit apart.
    let mut leaves = Vec::new();
    for x in 0..6 {
        for y in 0..6 {
            for z in 0..6 {
                leaves.push(box_at(x as f32, y as f32, z as f32, 0.5));
            }
        }
    }
    // Queries of several sizes: a point-ish probe, a mid box spanning a few
    // cells, a wide box covering a large sub-volume, a fully-outside miss, and a
    // corner cell.
    let queries = [
        box_at(2.0, 2.0, 2.0, 0.1),
        box_at(1.5, 1.5, 1.5, 1.2),
        box_at(3.0, 3.0, 3.0, 2.5),
        box_at(-5.0, -5.0, -5.0, 0.5),
        box_at(0.0, 0.0, 0.0, 0.5),
    ];
    assert_resident_overlap_matches(&builder, &overlap, &ctx, &leaves, &queries, 1 << 12);
}

#[test]
fn many_small_queries_over_sparse_scene_match_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let overlap = GpuBvhOverlap::new(&ctx);
    // Small leaves spread wide, probed by a large batch of small queries: deep
    // pruning plus a per-query dispatch over many invocations.
    let mut rng = Rng::new(0x00c0_ffee_90ab_cdef);
    let leaves: Vec<Aabb> = (0..2000).map(|_| random_box(&mut rng, 40.0, 0.5)).collect();
    let queries: Vec<Aabb> = (0..1500).map(|_| random_box(&mut rng, 40.0, 0.8)).collect();
    assert_resident_overlap_matches(&builder, &overlap, &ctx, &leaves, &queries, 1 << 12);
}

#[test]
fn coincident_leaves_match_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let overlap = GpuBvhOverlap::new(&ctx);
    // Coincident leaves stress equal Morton codes; a query over them must gather
    // all of them, exercising the resident single-position bounds decode.
    let leaves = vec![box_at(2.0, -1.0, 3.0, 0.5); 24];
    let queries = [
        box_at(2.0, -1.0, 3.0, 0.5),
        box_at(2.0, -1.0, 3.0, 2.0),
        box_at(10.0, 10.0, 10.0, 0.5),
    ];
    assert_resident_overlap_matches(&builder, &overlap, &ctx, &leaves, &queries, 64);
}

#[test]
fn degenerate_axis_scene_matches_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let overlap = GpuBvhOverlap::new(&ctx);
    // All leaves share y and z; only x varies, a flat scene the Morton
    // quantisation collapses onto one axis.
    let mut rng = Rng::new(0x0bad_c0de_1357_9bdf);
    let leaves: Vec<Aabb> = (0..800)
        .map(|_| {
            let x = rng.coord(-10.0, 20.0);
            let c = Vec3::new(x, 0.0, 0.0);
            let half = Vec3::new(0.4, 0.5, 0.5);
            Aabb::new(c - half, c + half)
        })
        .collect();
    let queries: Vec<Aabb> = (0..200)
        .map(|_| {
            let x = rng.coord(-10.0, 20.0);
            let c = Vec3::new(x, 0.0, 0.0);
            let half = Vec3::new(1.0, 0.6, 0.6);
            Aabb::new(c - half, c + half)
        })
        .collect();
    assert_resident_overlap_matches(&builder, &overlap, &ctx, &leaves, &queries, 1 << 14);
}

#[test]
fn overflow_is_reported_with_the_offending_query() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let overlap = GpuBvhOverlap::new(&ctx);
    // A tight cluster so one wide query overlaps far more leaves than the tiny
    // capacity allows; the resident device query must flag overflow exactly like
    // the twin over the host-built tree.
    let leaves: Vec<Aabb> = (0..32).map(|_| box_at(0.0, 0.0, 0.0, 0.5)).collect();
    let tree = cpu_build_lbvh(&leaves);
    let queries = [box_at(0.0, 0.0, 0.0, 1.0)];
    let capacity = 4;
    let cpu = cpu_bvh_aabb_overlap(&tree, &queries, capacity);
    let resident = builder.build_resident(&ctx, &leaves);
    let gpu = overlap.query_resident(&ctx, &resident, &queries, capacity);
    ctx.wait();
    assert!(cpu.is_err(), "cpu twin should report overflow");
    assert!(gpu.is_err(), "gpu resident query should report overflow");
    assert_eq!(cpu, gpu, "cpu and gpu overflow errors must match");
}
