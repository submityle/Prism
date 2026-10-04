//! Tests for the dynamic `TLAS` update executor.
//!
//! Correctness is pinned by comparing hits against a fresh binned-`SAH`
//! [`Tlas::build`] over the exact same (re-placed) instance table: whatever
//! update the executor runs — reuse, refit, or rebuild — a random-ray
//! closest-hit sweep must agree element-for-element with the golden rebuild on
//! the stable `(t, u, v, primitive, instance_id)` fields, because the
//! ray-instance-triangle tests are identical and only the top-level traversal
//! bounds and order differ (the reordered `instance_index` may legitimately
//! differ between a refit tree and a rebuild and is therefore excluded).

use super::super::acceleration::{
    AccelerationUpdate, AccelerationUpdatePolicy, GeometryChange, RebuildLedger,
};
use super::super::bvh::{Bvh, Triangle};
use super::super::tlas::{Affine3, Instance, Tlas};
use super::super::traversal::Ray;
use super::DynamicTlas;

use alloc::vec::Vec;

/// Small deterministic xorshift generator (self-contained, no dev-deps).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform `f32` in `[-1, 1)`.
    fn signed(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        u * 2.0 - 1.0
    }
}

/// A compact unit-ish triangle soup, offset so each `BLAS` has distinct local
/// geometry. Ids are local to the `BLAS`.
fn blas_soup(rng: &mut Rng, n: u32) -> Bvh {
    let mut tris = Vec::with_capacity(n as usize);
    for primitive in 0..n {
        let base = [rng.signed(), rng.signed(), rng.signed()];
        let jitter = |rng: &mut Rng| {
            [
                base[0] + rng.signed() * 0.4,
                base[1] + rng.signed() * 0.4,
                base[2] + rng.signed() * 0.4,
            ]
        };
        let v0 = jitter(rng);
        let v1 = jitter(rng);
        let v2 = jitter(rng);
        tris.push(Triangle::new(v0, v1, v2, primitive));
    }
    Bvh::build(&tris)
}

/// Builds a `BLAS` pool plus `count` instances scattered on a grid. Returns the
/// pool, the base instances, and the per-instance base offsets (keyed by
/// `instance_id`, which equals the instance's index here).
fn scene(seed: u64, count: u32) -> (Vec<Bvh>, Vec<Instance>, Vec<[f32; 3]>) {
    let mut rng = Rng::new(seed);
    let blases = alloc::vec![blas_soup(&mut rng, 16), blas_soup(&mut rng, 24), blas_soup(&mut rng, 12)];
    let mut instances = Vec::with_capacity(count as usize);
    let mut offsets = Vec::with_capacity(count as usize);
    for id in 0..count {
        let offset = [
            f32::from(u16::try_from(id % 8).unwrap()) * 4.0 - 14.0 + rng.signed(),
            f32::from(u16::try_from((id / 8) % 8).unwrap()) * 4.0 - 14.0 + rng.signed(),
            rng.signed() * 6.0,
        ];
        let blas = (id % 3) as usize;
        let xf = Affine3::from_translation(offset);
        instances.push(Instance::new(xf, blas, id).expect("non-singular placement"));
        offsets.push(offset);
    }
    (blases, instances, offsets)
}

/// Builds the golden instance table by re-placing every base instance with
/// `transforms(instance_id)`, preserving its `BLAS` index, id, and mask.
fn golden_tlas(
    base: &[Instance],
    transforms: &impl Fn(u32) -> Affine3,
    blases: &[Bvh],
) -> Tlas {
    let placed: Vec<Instance> = base
        .iter()
        .map(|inst| {
            Instance::with_mask(
                transforms(inst.instance_id()),
                inst.blas(),
                inst.instance_id(),
                inst.mask(),
            )
            .expect("non-singular golden placement")
        })
        .collect();
    Tlas::build(&placed, blases)
}

/// Assert the executor reports the same closest hits as `golden` over `rays`
/// random rays, comparing only the layout-independent fields.
fn assert_hits_match(dyn_tlas: &DynamicTlas, golden: &Tlas, blases: &[Bvh], rays: u32, seed: u64) {
    let mut rng = Rng::new(seed);
    for _ in 0..rays {
        let origin = [rng.signed() * 20.0, rng.signed() * 20.0, rng.signed() * 20.0];
        let dir = [rng.signed(), rng.signed(), rng.signed()];
        if dir == [0.0, 0.0, 0.0] {
            continue;
        }
        let ray = Ray::new(origin, dir, 0.0, f32::INFINITY);
        let got = dyn_tlas.closest_hit(&ray, blases);
        let want = golden.closest_hit(&ray, blases);
        match (got, want) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert_eq!(a.primitive, b.primitive, "primitive mismatch");
                assert_eq!(a.instance_id, b.instance_id, "instance_id mismatch");
                assert_eq!(a.t.to_bits(), b.t.to_bits(), "t mismatch");
                assert_eq!(a.u.to_bits(), b.u.to_bits(), "u mismatch");
                assert_eq!(a.v.to_bits(), b.v.to_bits(), "v mismatch");
            }
            _ => panic!("hit presence mismatch: {got:?} vs {want:?}"),
        }
    }
}

/// Translate every instance by a uniform `delta` from its base offset, keyed by
/// stable id.
fn uniform_move(offsets: &[[f32; 3]], delta: [f32; 3]) -> impl Fn(u32) -> Affine3 + '_ {
    move |id| {
        let o = offsets[id as usize];
        Affine3::from_translation([o[0] + delta[0], o[1] + delta[1], o[2] + delta[2]])
    }
}

/// Per-instance divergent displacement that scrambles the world-space layout
/// relative to the original top-level split planes, scaled by `k`.
fn divergent_move(offsets: &[[f32; 3]], k: f32) -> impl Fn(u32) -> Affine3 + '_ {
    move |id| {
        let o = offsets[id as usize];
        let dx = f32::from(u16::try_from(id % 7).unwrap()) * k;
        let dy = f32::from(u16::try_from(id % 5).unwrap()) * -k;
        let dz = f32::from(u16::try_from(id % 11).unwrap()) * k;
        Affine3::from_translation([o[0] + dx, o[1] + dy, o[2] + dz])
    }
}

const BYTES_PER_INSTANCE: u32 = 64;

fn policy() -> AccelerationUpdatePolicy {
    AccelerationUpdatePolicy::default()
}

#[test]
fn new_matches_tlas_build() {
    let (blases, instances, _offsets) = scene(0x1234, 40);
    let dyn_tlas = DynamicTlas::new(&instances, &blases, policy(), BYTES_PER_INSTANCE);
    let golden = Tlas::build(&instances, &blases);
    assert_eq!(dyn_tlas.instance_count(), 40);
    assert_hits_match(&dyn_tlas, &golden, &blases, 1200, 0x9999);
}

#[test]
fn no_motion_reuses_structure() {
    let (blases, instances, offsets) = scene(0x5151, 32);
    let mut dyn_tlas = DynamicTlas::new(&instances, &blases, policy(), BYTES_PER_INSTANCE);
    let mut ledger = RebuildLedger::new(1 << 20);
    let change = GeometryChange {
        moved_primitives: 0,
        total_primitives: 32,
        max_vertex_deformation: 0.0,
        topology_changed: false,
        fragmentation: 0.0,
    };
    let xf = uniform_move(&offsets, [0.0, 0.0, 0.0]);
    let receipt = dyn_tlas.deform(change, &xf, &blases, &mut ledger);
    assert_eq!(receipt.executed, AccelerationUpdate::Reuse);
    assert_eq!(receipt.cost_bytes, 0);
    let golden = Tlas::build(&instances, &blases);
    assert_hits_match(&dyn_tlas, &golden, &blases, 1000, 0x2222);
}

#[test]
fn small_motion_refits_and_stays_correct() {
    let (blases, instances, offsets) = scene(0x7f7f, 48);
    let mut dyn_tlas = DynamicTlas::new(&instances, &blases, policy(), BYTES_PER_INSTANCE);
    let mut ledger = RebuildLedger::new(1 << 20);
    let change = GeometryChange {
        moved_primitives: 6,
        total_primitives: 48,
        max_vertex_deformation: 0.02,
        topology_changed: false,
        fragmentation: 0.0,
    };
    let delta = [0.3, -0.2, 0.1];
    let xf = uniform_move(&offsets, delta);
    let receipt = dyn_tlas.deform(change, &xf, &blases, &mut ledger);
    assert_eq!(receipt.executed, AccelerationUpdate::Refit);
    assert!(receipt.admitted);
    assert!(receipt.refit_quality.is_some());
    let golden = golden_tlas(&instances, &xf, &blases);
    assert_hits_match(&dyn_tlas, &golden, &blases, 1500, 0x3131);
}

#[test]
fn large_motion_rebuilds_and_stays_correct() {
    let (blases, instances, offsets) = scene(0xabcd, 48);
    let mut dyn_tlas = DynamicTlas::new(&instances, &blases, policy(), BYTES_PER_INSTANCE);
    let mut ledger = RebuildLedger::new(1 << 20);
    let change = GeometryChange {
        moved_primitives: 40,
        total_primitives: 48,
        max_vertex_deformation: 0.4,
        topology_changed: false,
        fragmentation: 0.0,
    };
    let delta = [9.0, -7.0, 5.0];
    let xf = uniform_move(&offsets, delta);
    let receipt = dyn_tlas.deform(change, &xf, &blases, &mut ledger);
    assert_eq!(receipt.executed, AccelerationUpdate::Rebuild);
    assert!(receipt.admitted);
    assert!(!receipt.rebuild_pending);
    let golden = golden_tlas(&instances, &xf, &blases);
    assert_hits_match(&dyn_tlas, &golden, &blases, 1500, 0x4747);
}

#[test]
fn high_fragmentation_reinstance_compacts() {
    let (blases, instances, _offsets) = scene(0x2020, 40);
    let mut dyn_tlas = DynamicTlas::new(&instances, &blases, policy(), BYTES_PER_INSTANCE);
    let mut ledger = RebuildLedger::new(1 << 20);
    // A reduced instance set with high fragmentation forces BuildAndCompact.
    let (_b2, kept, _o2) = scene(0x2020, 24);
    let change = GeometryChange {
        moved_primitives: 24,
        total_primitives: 24,
        max_vertex_deformation: 0.0,
        topology_changed: true,
        fragmentation: 0.7,
    };
    let receipt = dyn_tlas.reinstance(change, &kept, &blases, &mut ledger);
    assert_eq!(receipt.executed, AccelerationUpdate::BuildAndCompact);
    assert!(receipt.admitted);
    assert_eq!(dyn_tlas.instance_count(), 24);
    let golden = Tlas::build(&kept, &blases);
    assert_hits_match(&dyn_tlas, &golden, &blases, 1200, 0x5a5a);
}

#[test]
fn reinstance_rebuilds_from_new_table() {
    let (blases, instances, _offsets) = scene(0x6161, 32);
    let mut dyn_tlas = DynamicTlas::new(&instances, &blases, policy(), BYTES_PER_INSTANCE);
    let mut ledger = RebuildLedger::new(1 << 20);
    // Append new instances (grow the set) with low fragmentation → plain Rebuild.
    let mut grown = instances.clone();
    for id in 32..48 {
        let xf = Affine3::from_translation([f32::from(u16::try_from(id).unwrap()) * 0.7 - 6.0, 3.0, -4.0]);
        grown.push(Instance::new(xf, (id % 3) as usize, id).expect("non-singular"));
    }
    let change = GeometryChange {
        moved_primitives: 48,
        total_primitives: 48,
        max_vertex_deformation: 0.0,
        topology_changed: true,
        fragmentation: 0.1,
    };
    let receipt = dyn_tlas.reinstance(change, &grown, &blases, &mut ledger);
    assert_eq!(receipt.executed, AccelerationUpdate::Rebuild);
    assert_eq!(dyn_tlas.instance_count(), 48);
    let golden = Tlas::build(&grown, &blases);
    assert_hits_match(&dyn_tlas, &golden, &blases, 1500, 0x6c6c);
}

#[test]
fn exhausted_budget_defers_motion_rebuild_to_refit() {
    let (blases, instances, offsets) = scene(0x0a0a, 48);
    let mut dyn_tlas = DynamicTlas::new(&instances, &blases, policy(), BYTES_PER_INSTANCE);
    // Zero budget: a motion-driven rebuild cannot be admitted.
    let mut ledger = RebuildLedger::new(0);
    let change = GeometryChange {
        moved_primitives: 40,
        total_primitives: 48,
        max_vertex_deformation: 0.4,
        topology_changed: false,
        fragmentation: 0.0,
    };
    let delta = [8.0, -6.0, 4.0];
    let xf = uniform_move(&offsets, delta);
    let receipt = dyn_tlas.deform(change, &xf, &blases, &mut ledger);
    assert_eq!(receipt.requested, AccelerationUpdate::Rebuild);
    assert_eq!(receipt.executed, AccelerationUpdate::Refit);
    assert!(!receipt.admitted);
    assert!(receipt.rebuild_pending);
    assert!(dyn_tlas.rebuild_pending());
    // Fallback refit still yields a correct structure.
    let golden = golden_tlas(&instances, &xf, &blases);
    assert_hits_match(&dyn_tlas, &golden, &blases, 1500, 0x7d7d);
}

#[test]
fn pending_rebuild_fires_next_frame_with_budget() {
    let (blases, instances, offsets) = scene(0x0b0b, 48);
    let mut dyn_tlas = DynamicTlas::new(&instances, &blases, policy(), BYTES_PER_INSTANCE);
    // Frame 1: zero budget defers the motion rebuild to a refit.
    let mut starved = RebuildLedger::new(0);
    let change = GeometryChange {
        moved_primitives: 40,
        total_primitives: 48,
        max_vertex_deformation: 0.4,
        topology_changed: false,
        fragmentation: 0.0,
    };
    let delta1 = [8.0, -6.0, 4.0];
    let xf1 = uniform_move(&offsets, delta1);
    let r1 = dyn_tlas.deform(change, &xf1, &blases, &mut starved);
    assert!(r1.rebuild_pending);

    // Frame 2: ample budget, and even a tiny-motion frame escalates to a rebuild
    // because a quality-driven rebuild is queued.
    let mut ample = RebuildLedger::new(1 << 20);
    let small = GeometryChange {
        moved_primitives: 1,
        total_primitives: 48,
        max_vertex_deformation: 0.001,
        topology_changed: false,
        fragmentation: 0.0,
    };
    let delta2 = [8.05, -6.02, 4.03];
    let xf2 = uniform_move(&offsets, delta2);
    let r2 = dyn_tlas.deform(small, &xf2, &blases, &mut ample);
    assert!(r2.executed.is_rebuild());
    assert!(!r2.rebuild_pending);
    assert!(!dyn_tlas.rebuild_pending());
    let golden = golden_tlas(&instances, &xf2, &blases);
    assert_hits_match(&dyn_tlas, &golden, &blases, 1500, 0x8e8e);
}

#[test]
fn repeated_refits_eventually_escalate_on_quality() {
    let (blases, instances, offsets) = scene(0x0c0c, 64);
    let mut dyn_tlas = DynamicTlas::new(&instances, &blases, policy(), BYTES_PER_INSTANCE);
    let mut ledger = RebuildLedger::new(1 << 20);
    let change = GeometryChange {
        moved_primitives: 8,
        total_primitives: 64,
        max_vertex_deformation: 0.02,
        topology_changed: false,
        fragmentation: 0.0,
    };
    // Each frame scrambles instances farther apart with divergent per-instance
    // motion, so the original top-level split planes drift and refit_quality
    // climbs past the escalation ratio.
    let mut escalated = false;
    let mut last_xf_k = 0.0_f32;
    for frame in 1..=24u32 {
        let k = frame as f32 * 0.6;
        last_xf_k = k;
        let xf = divergent_move(&offsets, k);
        let receipt = dyn_tlas.deform(change, &xf, &blases, &mut ledger);
        // While refitting, correctness holds against a golden rebuild each frame.
        if receipt.executed == AccelerationUpdate::Refit {
            let golden = golden_tlas(&instances, &xf, &blases);
            assert_hits_match(&dyn_tlas, &golden, &blases, 400, 0x9100 + u64::from(frame));
        }
        if receipt.rebuild_pending || receipt.executed.is_rebuild() {
            escalated = true;
            break;
        }
    }
    assert!(
        escalated,
        "accumulated divergent refits never degraded refit_quality past the rebuild ratio",
    );
    // A final frame with budget flushes the queued rebuild to a correct tree.
    let xf = divergent_move(&offsets, last_xf_k);
    let flush = dyn_tlas.deform(change, &xf, &blases, &mut ledger);
    if flush.executed.is_rebuild() {
        let golden = golden_tlas(&instances, &xf, &blases);
        assert_hits_match(&dyn_tlas, &golden, &blases, 800, 0x9fff);
    }
}
