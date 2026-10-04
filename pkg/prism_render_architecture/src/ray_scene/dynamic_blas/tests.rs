//! Tests for the dynamic `BLAS` update executor.
//!
//! Correctness is pinned by comparing hits against a fresh binned-`SAH`
//! [`Bvh::build`] over the exact same (moved) triangle soup: whatever update the
//! executor runs — reuse, refit, or `LBVH` rebuild — a random-ray closest-hit
//! sweep must agree element-for-element with the golden rebuild, because the
//! ray-triangle tests are identical and only traversal bounds/order differ.

use super::super::acceleration::{
    AccelerationUpdate, AccelerationUpdatePolicy, GeometryChange, RebuildLedger,
};
use super::super::bvh::{Bvh, Triangle};
use super::super::traversal::Ray;
use super::DynamicBlas;

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

fn random_triangles(rng: &mut Rng, n: u32) -> Vec<Triangle> {
    let mut tris = Vec::with_capacity(n as usize);
    for primitive in 0..n {
        let base = [rng.signed() * 10.0, rng.signed() * 10.0, rng.signed() * 10.0];
        let jitter = |rng: &mut Rng| {
            [
                base[0] + rng.signed(),
                base[1] + rng.signed(),
                base[2] + rng.signed(),
            ]
        };
        let v0 = jitter(rng);
        let v1 = jitter(rng);
        let v2 = jitter(rng);
        tris.push(Triangle::new(v0, v1, v2, primitive));
    }
    tris
}

/// Translate every triangle in `originals` by `delta`, keyed by stable id.
fn translate(originals: &[Triangle], delta: [f32; 3]) -> impl Fn(u32) -> [[f32; 3]; 3] + '_ {
    move |id| {
        let t = originals[id as usize];
        let shift = |v: [f32; 3]| [v[0] + delta[0], v[1] + delta[1], v[2] + delta[2]];
        [shift(t.v0), shift(t.v1), shift(t.v2)]
    }
}

/// Assert `blas` reports the same closest hits as a golden rebuild of `expected`
/// over `rays` random rays.
fn assert_hits_match(blas: &DynamicBlas, expected: &[Triangle], rays: u32, seed: u64) {
    let golden = Bvh::build(expected);
    let mut rng = Rng::new(seed);
    for _ in 0..rays {
        let origin = [rng.signed() * 15.0, rng.signed() * 15.0, rng.signed() * 15.0];
        let dir = [rng.signed(), rng.signed(), rng.signed()];
        if dir == [0.0, 0.0, 0.0] {
            continue;
        }
        let ray = Ray::new(origin, dir, 0.0, f32::INFINITY);
        let got = blas.bvh().closest_hit(&ray);
        let want = golden.closest_hit(&ray);
        match (got, want) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert_eq!(a.primitive, b.primitive, "primitive mismatch");
                assert_eq!(a.t.to_bits(), b.t.to_bits(), "t mismatch");
            }
            _ => panic!("hit presence mismatch: {got:?} vs {want:?}"),
        }
    }
}

/// Expected positions of `originals` after translation by `delta`.
fn moved(originals: &[Triangle], delta: [f32; 3]) -> Vec<Triangle> {
    let shift = translate(originals, delta);
    originals
        .iter()
        .map(|t| {
            let [v0, v1, v2] = shift(t.primitive);
            Triangle::new(v0, v1, v2, t.primitive)
        })
        .collect()
}

#[test]
fn new_matches_binned_sah_build() {
    let mut rng = Rng::new(1);
    let tris = random_triangles(&mut rng, 400);
    let blas = DynamicBlas::new(&tris, AccelerationUpdatePolicy::default(), 64);
    assert_eq!(blas.primitive_count(), 400);
    assert_hits_match(&blas, &tris, 1500, 99);
}

#[test]
fn no_motion_reuses_structure() {
    let mut rng = Rng::new(2);
    let tris = random_triangles(&mut rng, 200);
    let mut blas = DynamicBlas::new(&tris, AccelerationUpdatePolicy::default(), 64);
    let before = blas.bvh().clone();
    let mut ledger = RebuildLedger::new(1 << 30);
    let change = GeometryChange {
        moved_primitives: 0,
        total_primitives: 200,
        max_vertex_deformation: 0.0,
        topology_changed: false,
        fragmentation: 0.0,
    };
    let report = blas.deform(change, translate(&tris, [0.0, 0.0, 0.0]), &mut ledger);
    assert_eq!(report.executed, AccelerationUpdate::Reuse);
    assert_eq!(report.cost_bytes, 0);
    assert_eq!(blas.bvh(), &before, "reuse must not touch the tree");
    assert_eq!(ledger.spent_bytes(), 0);
}

#[test]
fn small_motion_refits_and_stays_correct() {
    let mut rng = Rng::new(3);
    let tris = random_triangles(&mut rng, 300);
    let mut blas = DynamicBlas::new(&tris, AccelerationUpdatePolicy::default(), 64);
    let mut ledger = RebuildLedger::new(1 << 30);
    let delta = [0.02, -0.01, 0.015];
    let change = GeometryChange {
        moved_primitives: 3,
        total_primitives: 300,
        max_vertex_deformation: 0.01,
        topology_changed: false,
        fragmentation: 0.0,
    };
    let report = blas.deform(change, translate(&tris, delta), &mut ledger);
    assert_eq!(report.executed, AccelerationUpdate::Refit);
    assert!(report.admitted);
    assert!(report.refit_quality.is_some());
    // Refit charges a quarter of a full build; mandatory, so always spent.
    assert!(ledger.spent_bytes() > 0);
    assert_hits_match(&blas, &moved(&tris, delta), 1500, 7);
}

#[test]
fn large_motion_rebuilds_and_stays_correct() {
    let mut rng = Rng::new(4);
    let tris = random_triangles(&mut rng, 300);
    let mut blas = DynamicBlas::new(&tris, AccelerationUpdatePolicy::default(), 64);
    let mut ledger = RebuildLedger::new(1 << 30);
    let delta = [5.0, -4.0, 3.0];
    let change = GeometryChange {
        moved_primitives: 300,
        total_primitives: 300,
        max_vertex_deformation: 0.4,
        topology_changed: false,
        fragmentation: 0.0,
    };
    let report = blas.deform(change, translate(&tris, delta), &mut ledger);
    assert_eq!(report.executed, AccelerationUpdate::Rebuild);
    assert!(report.admitted);
    assert!(!report.rebuild_pending);
    assert_hits_match(&blas, &moved(&tris, delta), 1500, 11);
}

#[test]
fn high_fragmentation_rebuild_compacts() {
    let mut rng = Rng::new(5);
    let tris = random_triangles(&mut rng, 150);
    let mut blas = DynamicBlas::new(&tris, AccelerationUpdatePolicy::default(), 64);
    let mut ledger = RebuildLedger::new(1 << 30);
    let delta = [8.0, 0.0, 0.0];
    let change = GeometryChange {
        moved_primitives: 150,
        total_primitives: 150,
        max_vertex_deformation: 0.6,
        topology_changed: false,
        fragmentation: 0.8,
    };
    let report = blas.deform(change, translate(&tris, delta), &mut ledger);
    assert_eq!(report.executed, AccelerationUpdate::BuildAndCompact);
    assert_hits_match(&blas, &moved(&tris, delta), 1200, 13);
}

#[test]
fn topology_change_rebuilds_from_new_soup() {
    let mut rng = Rng::new(6);
    let tris = random_triangles(&mut rng, 200);
    let mut blas = DynamicBlas::new(&tris, AccelerationUpdatePolicy::default(), 64);
    let mut ledger = RebuildLedger::new(1 << 30);
    // A fresh, differently-sized soup.
    let new_tris = random_triangles(&mut rng, 275);
    let change = GeometryChange {
        moved_primitives: 0,
        total_primitives: 200,
        max_vertex_deformation: 0.0,
        topology_changed: true,
        fragmentation: 0.0,
    };
    let report = blas.retopology(change, &new_tris, &mut ledger);
    assert_eq!(report.executed, AccelerationUpdate::Rebuild);
    assert!(report.admitted);
    assert_eq!(blas.primitive_count(), 275);
    assert_hits_match(&blas, &new_tris, 1500, 17);
}

#[test]
fn exhausted_budget_defers_motion_rebuild_to_refit() {
    let mut rng = Rng::new(7);
    let tris = random_triangles(&mut rng, 300);
    let mut blas = DynamicBlas::new(&tris, AccelerationUpdatePolicy::default(), 64);
    // Budget admits a refit's scratch but not a full rebuild's.
    let prims = 300u64;
    let bpp = 64u64;
    let refit_cost = prims * bpp / 4;
    let mut ledger = RebuildLedger::new(refit_cost); // < full rebuild cost (prims*bpp)
    let delta = [6.0, 6.0, 6.0];
    let change = GeometryChange {
        moved_primitives: 300,
        total_primitives: 300,
        max_vertex_deformation: 0.5,
        topology_changed: false,
        fragmentation: 0.0,
    };
    let report = blas.deform(change, translate(&tris, delta), &mut ledger);
    assert_eq!(report.requested, AccelerationUpdate::Rebuild);
    assert_eq!(report.executed, AccelerationUpdate::Refit, "rebuild deferred");
    assert!(!report.admitted);
    assert!(report.rebuild_pending, "rebuild stays queued");
    assert!(blas.rebuild_pending());
    // Even deferred, the structure is correct via the fallback refit.
    assert_hits_match(&blas, &moved(&tris, delta), 1200, 19);
}

#[test]
fn pending_rebuild_fires_next_frame_with_budget() {
    let mut rng = Rng::new(8);
    let tris = random_triangles(&mut rng, 250);
    let mut blas = DynamicBlas::new(&tris, AccelerationUpdatePolicy::default(), 64);
    let delta = [6.0, 6.0, 6.0];
    let change = GeometryChange {
        moved_primitives: 250,
        total_primitives: 250,
        max_vertex_deformation: 0.5,
        topology_changed: false,
        fragmentation: 0.0,
    };
    // Frame 1: tight budget forces a deferral.
    let mut tight = RebuildLedger::new(250 * 64 / 4);
    let r1 = blas.deform(change, translate(&tris, delta), &mut tight);
    assert_eq!(r1.executed, AccelerationUpdate::Refit);
    assert!(blas.rebuild_pending());
    // Frame 2: ample budget, even a tiny reported motion escalates to rebuild
    // because a quality/deferral-driven rebuild is pending.
    let mut ample = RebuildLedger::new(1 << 30);
    let small = GeometryChange {
        moved_primitives: 1,
        total_primitives: 250,
        max_vertex_deformation: 0.001,
        topology_changed: false,
        fragmentation: 0.0,
    };
    let r2 = blas.deform(small, translate(&tris, delta), &mut ample);
    assert_eq!(r2.executed, AccelerationUpdate::Rebuild, "pending rebuild fires");
    assert!(!blas.rebuild_pending());
    assert_hits_match(&blas, &moved(&tris, delta), 1200, 23);
}

#[test]
fn repeated_refits_eventually_escalate_on_quality() {
    let mut rng = Rng::new(9);
    let tris = random_triangles(&mut rng, 500);
    let mut blas = DynamicBlas::new(&tris, AccelerationUpdatePolicy::default(), 64);
    let mut ledger = RebuildLedger::new(1 << 40);

    // Divergent, per-primitive displacement: translating *every* triangle by the
    // same vector would preserve relative layout and never degrade the tree, so
    // instead each primitive drifts in an id-dependent direction. That scrambles
    // the geometry relative to the original split planes, which is exactly what
    // erodes [`Bvh::refit_quality`] until the policy escalates to a rebuild.
    let disp = |id: u32, k: f32| {
        [
            (id % 7) as f32 - 3.0,
            (id % 5) as f32 - 2.0,
            (id % 11) as f32 - 5.0,
        ]
        .map(|c| c * k)
    };
    let expected = |k: f32| {
        tris.iter()
            .map(|t| {
                let d = disp(t.primitive, k);
                let shift = |v: [f32; 3]| [v[0] + d[0], v[1] + d[1], v[2] + d[2]];
                Triangle::new(shift(t.v0), shift(t.v1), shift(t.v2), t.primitive)
            })
            .collect::<Vec<_>>()
    };

    let mut saw_escalation = false;
    for frame in 1..40u32 {
        let k = frame as f32 * 0.5;
        let change = GeometryChange {
            moved_primitives: 10,
            total_primitives: 500,
            max_vertex_deformation: 0.02,
            topology_changed: false,
            fragmentation: 0.0,
        };
        let pos = |id: u32| {
            let t = tris[id as usize];
            let d = disp(id, k);
            let shift = |v: [f32; 3]| [v[0] + d[0], v[1] + d[1], v[2] + d[2]];
            [shift(t.v0), shift(t.v1), shift(t.v2)]
        };
        let report = blas.deform(change, pos, &mut ledger);
        // Correct every frame regardless of refit vs rebuild.
        assert_hits_match(&blas, &expected(k), 400, 1000 + u64::from(frame));
        if report.rebuild_pending || report.executed.is_rebuild() {
            saw_escalation = true;
            break;
        }
    }
    assert!(
        saw_escalation,
        "accumulated relative drift never escalated past the quality ratio"
    );
}
