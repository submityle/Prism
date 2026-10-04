//! Tests for the dynamic scene orchestrator.
//!
//! Every frame is pinned against a golden rebuild of the *entire* hierarchy from
//! scratch: each `BLAS` is rebuilt from its current triangle soup with the same
//! parallel linear-`BVH` builder the scene uses, and a fresh binned-`SAH`
//! [`Tlas::build`] is placed over the current instance table. Whatever updates
//! the orchestrator ran — `BLAS` refit/rebuild/retopology and top-level
//! reuse/refit/rebuild — a random-ray closest-hit sweep must agree with the
//! golden element-for-element on the stable `(t, u, v, primitive, instance_id)`
//! fields. The reordered `instance_index` may legitimately differ between a
//! refit tree and a rebuild and is therefore excluded.
//!
//! The orchestrator-specific property under test is the cross-level invariant: a
//! changed `BLAS` must refresh the top-level world boxes even when no instance
//! moved. The static-instance tests exercise exactly that path.

use super::super::acceleration::{
    AccelerationUpdate, AccelerationUpdatePolicy, GeometryChange, RebuildLedger,
};
use super::super::bvh::{Bvh, Triangle};
use super::super::lbvh::LinearBvh;
use super::super::tlas::{Affine3, Instance, Tlas};
use super::super::traversal::Ray;
use super::DynamicScene;

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

/// A compact triangle soup centred on `center`; primitive ids equal the index.
fn blas_soup(rng: &mut Rng, n: u32, center: [f32; 3]) -> Vec<Triangle> {
    let mut tris = Vec::with_capacity(n as usize);
    for primitive in 0..n {
        let base = [
            center[0] + rng.signed(),
            center[1] + rng.signed(),
            center[2] + rng.signed(),
        ];
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
    tris
}

/// Builds the triangle soups (one per `BLAS`) and the base instance table on a
/// grid. Returns the soups, instances, and per-instance base offsets keyed by
/// `instance_id` (which equals the instance index here).
fn scene_data(seed: u64, count: u32) -> (Vec<Vec<Triangle>>, Vec<Instance>, Vec<[f32; 3]>) {
    let mut rng = Rng::new(seed);
    let soups = alloc::vec![
        blas_soup(&mut rng, 16, [0.0, 0.0, 0.0]),
        blas_soup(&mut rng, 24, [0.5, -0.3, 0.2]),
        blas_soup(&mut rng, 12, [-0.4, 0.6, -0.1]),
    ];
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
    (soups, instances, offsets)
}

/// Borrowed view of the soups as `&[&[Triangle]]` for [`DynamicScene::new`].
fn soup_refs(soups: &[Vec<Triangle>]) -> Vec<&[Triangle]> {
    soups.iter().map(Vec::as_slice).collect()
}

/// Translate every triangle of a soup uniformly by `delta`, preserving ids.
fn translate_soup(soup: &[Triangle], delta: [f32; 3]) -> Vec<Triangle> {
    let add = |v: [f32; 3]| [v[0] + delta[0], v[1] + delta[1], v[2] + delta[2]];
    soup.iter()
        .map(|t| Triangle::new(add(t.v0), add(t.v1), add(t.v2), t.primitive))
        .collect()
}

/// Positions closure for [`DynamicScene::deform_blas`] reading a rebuilt soup.
fn positions_of(soup: &[Triangle]) -> impl Fn(u32) -> [[f32; 3]; 3] + '_ {
    move |pid| {
        let t = &soup[pid as usize];
        [t.v0, t.v1, t.v2]
    }
}

/// Re-place every base instance with `transforms(instance_id)`.
fn place(base: &[Instance], transforms: &impl Fn(u32) -> Affine3) -> Vec<Instance> {
    base.iter()
        .map(|inst| {
            Instance::with_mask(
                transforms(inst.instance_id()),
                inst.blas(),
                inst.instance_id(),
                inst.mask(),
            )
            .expect("non-singular golden placement")
        })
        .collect()
}

/// Per-instance divergent displacement from each base offset, scaled by `k`.
fn divergent_move(offsets: &[[f32; 3]], k: f32) -> impl Fn(u32) -> Affine3 + '_ {
    move |id| {
        let o = offsets[id as usize];
        let dx = f32::from(u16::try_from(id % 7).unwrap()) * k;
        let dy = f32::from(u16::try_from(id % 5).unwrap()) * -k;
        let dz = f32::from(u16::try_from(id % 11).unwrap()) * k;
        Affine3::from_translation([o[0] + dx, o[1] + dy, o[2] + dz])
    }
}

/// Golden rebuild: the scene's exact pool (same linear-`BVH` builder) plus a
/// fresh top-level build over the current instances.
fn golden(soups: &[Vec<Triangle>], instances: &[Instance]) -> (Vec<Bvh>, Tlas) {
    let pool: Vec<Bvh> = soups.iter().map(|s| LinearBvh::build(s)).collect();
    let tlas = Tlas::build(instances, &pool);
    (pool, tlas)
}

/// Assert the scene reports the same closest hits as the golden rebuild of
/// `soups`/`instances`, comparing only the layout-independent fields. Returns
/// the number of rays that hit, so callers can insist the sweep was meaningful.
fn assert_scene_matches(
    scene: &DynamicScene,
    soups: &[Vec<Triangle>],
    instances: &[Instance],
    rays: u32,
    seed: u64,
) -> u32 {
    let (pool, gold) = golden(soups, instances);
    let mut rng = Rng::new(seed);
    let mut hits = 0u32;
    for _ in 0..rays {
        let origin = [
            rng.signed() * 20.0,
            rng.signed() * 20.0,
            rng.signed() * 20.0,
        ];
        let dir = [rng.signed(), rng.signed(), rng.signed()];
        if dir == [0.0, 0.0, 0.0] {
            continue;
        }
        let ray = Ray::new(origin, dir, 0.0, f32::INFINITY);
        let got = scene.closest_hit(&ray);
        let want = gold.closest_hit(&ray, &pool);
        match (got, want) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert_eq!(a.primitive, b.primitive, "primitive mismatch");
                assert_eq!(a.instance_id, b.instance_id, "instance_id mismatch");
                assert_eq!(a.t.to_bits(), b.t.to_bits(), "t mismatch");
                assert_eq!(a.u.to_bits(), b.u.to_bits(), "u mismatch");
                assert_eq!(a.v.to_bits(), b.v.to_bits(), "v mismatch");
                hits += 1;
            }
            _ => panic!("hit presence mismatch: {got:?} vs {want:?}"),
        }
    }
    hits
}

const BYTES_PER_PRIMITIVE: u32 = 48;
const BYTES_PER_INSTANCE: u32 = 64;

fn policy() -> AccelerationUpdatePolicy {
    AccelerationUpdatePolicy::default()
}

/// A `GeometryChange` the top-level policy resolves to `Reuse` (nothing moved).
fn reuse_change(total: u32) -> GeometryChange {
    GeometryChange {
        moved_primitives: 0,
        total_primitives: total,
        max_vertex_deformation: 0.0,
        topology_changed: false,
        fragmentation: 0.0,
    }
}

/// A `GeometryChange` the policy resolves to `Refit` (small motion).
fn refit_change(moved: u32, total: u32) -> GeometryChange {
    GeometryChange {
        moved_primitives: moved,
        total_primitives: total,
        max_vertex_deformation: 0.02,
        topology_changed: false,
        fragmentation: 0.1,
    }
}

#[test]
fn new_matches_golden() {
    let (soups, instances, _offsets) = scene_data(0x1111, 48);
    let scene = DynamicScene::new(
        &soup_refs(&soups),
        &instances,
        policy(),
        BYTES_PER_PRIMITIVE,
        BYTES_PER_INSTANCE,
    );
    assert_eq!(scene.blas_count(), 3);
    assert_eq!(scene.instance_count(), 48);
    let hits = assert_scene_matches(&scene, &soups, &instances, 1400, 0xa1);
    assert!(hits > 0, "sweep hit nothing");
}

#[test]
fn blas_refit_with_static_instances_refreshes_tlas() {
    let (mut soups, instances, _offsets) = scene_data(0x2222, 40);
    let mut scene = DynamicScene::new(
        &soup_refs(&soups),
        &instances,
        policy(),
        BYTES_PER_PRIMITIVE,
        BYTES_PER_INSTANCE,
    );
    let mut ledger = RebuildLedger::new(1 << 20);

    // Move BLAS 1's geometry; the instances referencing it never move.
    let moved = translate_soup(&soups[1], [0.6, -0.4, 0.3]);
    let count = u32::try_from(soups[1].len()).unwrap();
    let blas_receipt = scene.deform_blas(1, refit_change(4, count), positions_of(&moved), &mut ledger);
    assert_eq!(blas_receipt.executed, AccelerationUpdate::Refit);
    soups[1] = moved;
    assert!(scene.has_dirty_blas(), "BLAS change must mark the pool dirty");

    // Top-level change is a pure Reuse, yet the invariant must upgrade it to a
    // refit so the stale world boxes are re-read from the moved BLAS.
    let tlas_receipt = scene.sync_tlas_bounds_only(reuse_change(40), &mut ledger);
    assert_eq!(
        tlas_receipt.executed,
        AccelerationUpdate::Refit,
        "a dirty BLAS must force at least a TLAS refit"
    );
    assert!(!scene.has_dirty_blas(), "sync must clear the dirty set");

    let hits = assert_scene_matches(&scene, &soups, &instances, 1400, 0xb2);
    assert!(hits > 0, "sweep hit nothing");
}

#[test]
fn instance_motion_only_refits() {
    let (soups, instances, offsets) = scene_data(0x3333, 48);
    let mut scene = DynamicScene::new(
        &soup_refs(&soups),
        &instances,
        policy(),
        BYTES_PER_PRIMITIVE,
        BYTES_PER_INSTANCE,
    );
    let mut ledger = RebuildLedger::new(1 << 20);

    let xf = divergent_move(&offsets, 0.02);
    let receipt = scene.sync_tlas(refit_change(6, 48), &xf, &mut ledger);
    assert_eq!(receipt.executed, AccelerationUpdate::Refit);

    let moved = place(&instances, &xf);
    let hits = assert_scene_matches(&scene, &soups, &moved, 1400, 0xc3);
    assert!(hits > 0, "sweep hit nothing");
}

#[test]
fn large_instance_motion_rebuilds() {
    let (soups, instances, offsets) = scene_data(0x4444, 48);
    let mut scene = DynamicScene::new(
        &soup_refs(&soups),
        &instances,
        policy(),
        BYTES_PER_PRIMITIVE,
        BYTES_PER_INSTANCE,
    );
    let mut ledger = RebuildLedger::new(1 << 20);

    let xf = divergent_move(&offsets, 1.5);
    let change = GeometryChange {
        moved_primitives: 48,
        total_primitives: 48,
        max_vertex_deformation: 0.9,
        topology_changed: false,
        fragmentation: 0.2,
    };
    let receipt = scene.sync_tlas(change, &xf, &mut ledger);
    assert!(receipt.executed.is_rebuild(), "large motion should rebuild");

    let moved = place(&instances, &xf);
    let hits = assert_scene_matches(&scene, &soups, &moved, 1400, 0xd4);
    assert!(hits > 0, "sweep hit nothing");
}

#[test]
fn blas_retopology_refreshes_tlas() {
    let (mut soups, instances, _offsets) = scene_data(0x5555, 36);
    let mut scene = DynamicScene::new(
        &soup_refs(&soups),
        &instances,
        policy(),
        BYTES_PER_PRIMITIVE,
        BYTES_PER_INSTANCE,
    );
    let mut ledger = RebuildLedger::new(1 << 20);

    // Replace BLAS 0 with a fresh, differently sized soup (topology change).
    let mut rng = Rng::new(0x5a5a);
    let fresh = blas_soup(&mut rng, 20, [0.2, 0.1, -0.3]);
    let change = GeometryChange {
        moved_primitives: 0,
        total_primitives: 20,
        max_vertex_deformation: 0.0,
        topology_changed: true,
        fragmentation: 0.6,
    };
    let receipt = scene.retopology_blas(0, change, &fresh, &mut ledger);
    assert!(receipt.executed.is_rebuild());
    soups[0] = fresh;

    let tlas_receipt = scene.sync_tlas_bounds_only(reuse_change(36), &mut ledger);
    assert_eq!(tlas_receipt.executed, AccelerationUpdate::Refit);

    let hits = assert_scene_matches(&scene, &soups, &instances, 1400, 0xe5);
    assert!(hits > 0, "sweep hit nothing");
}

#[test]
fn combined_blas_and_instance_motion() {
    let (mut soups, instances, offsets) = scene_data(0x6666, 48);
    let mut scene = DynamicScene::new(
        &soup_refs(&soups),
        &instances,
        policy(),
        BYTES_PER_PRIMITIVE,
        BYTES_PER_INSTANCE,
    );
    let mut ledger = RebuildLedger::new(1 << 20);

    let moved0 = translate_soup(&soups[0], [0.3, 0.2, -0.1]);
    let c0 = u32::try_from(soups[0].len()).unwrap();
    scene.deform_blas(0, refit_change(4, c0), positions_of(&moved0), &mut ledger);
    soups[0] = moved0;

    let moved2 = translate_soup(&soups[2], [-0.2, 0.4, 0.1]);
    let c2 = u32::try_from(soups[2].len()).unwrap();
    scene.deform_blas(2, refit_change(4, c2), positions_of(&moved2), &mut ledger);
    soups[2] = moved2;

    let xf = divergent_move(&offsets, 0.015);
    scene.sync_tlas(refit_change(8, 48), &xf, &mut ledger);

    let moved = place(&instances, &xf);
    let hits = assert_scene_matches(&scene, &soups, &moved, 1500, 0xf6);
    assert!(hits > 0, "sweep hit nothing");
}

#[test]
fn reinstance_changes_instance_set() {
    let (soups, instances, _offsets) = scene_data(0x7777, 48);
    let mut scene = DynamicScene::new(
        &soup_refs(&soups),
        &instances,
        policy(),
        BYTES_PER_PRIMITIVE,
        BYTES_PER_INSTANCE,
    );
    let mut ledger = RebuildLedger::new(1 << 20);

    // Drop every third instance; the rest keep their ids and placement.
    let kept: Vec<Instance> = instances
        .iter()
        .filter(|inst| inst.instance_id() % 3 != 0)
        .copied()
        .collect();
    let change = GeometryChange {
        moved_primitives: 0,
        total_primitives: u32::try_from(kept.len()).unwrap(),
        max_vertex_deformation: 0.0,
        topology_changed: true,
        fragmentation: 0.7,
    };
    let receipt = scene.reinstance(change, &kept, &mut ledger);
    assert!(receipt.executed.is_rebuild());
    assert_eq!(scene.instance_count(), kept.len());

    let hits = assert_scene_matches(&scene, &soups, &kept, 1400, 0x17);
    assert!(hits > 0, "sweep hit nothing");
}

#[test]
fn budget_defers_tlas_rebuild_but_stays_correct() {
    let (soups, instances, offsets) = scene_data(0x8888, 40);
    let mut scene = DynamicScene::new(
        &soup_refs(&soups),
        &instances,
        policy(),
        BYTES_PER_PRIMITIVE,
        BYTES_PER_INSTANCE,
    );
    // Budget too small for a top-level rebuild, forcing a fallback refit.
    let mut ledger = RebuildLedger::new(16);

    let xf = divergent_move(&offsets, 1.2);
    let change = GeometryChange {
        moved_primitives: 40,
        total_primitives: 40,
        max_vertex_deformation: 0.8,
        topology_changed: false,
        fragmentation: 0.2,
    };
    let receipt = scene.sync_tlas(change, &xf, &mut ledger);
    assert_eq!(receipt.executed, AccelerationUpdate::Refit, "rebuild should defer to refit");
    assert!(!receipt.admitted, "the deferred rebuild must report not admitted");
    assert!(receipt.rebuild_pending, "a rebuild should stay queued");

    // A deferred rebuild still leaves a correct (refit) structure.
    let moved = place(&instances, &xf);
    let hits = assert_scene_matches(&scene, &soups, &moved, 1400, 0x18);
    assert!(hits > 0, "sweep hit nothing");
}

#[test]
fn multi_frame_sequence_stays_consistent() {
    let (mut soups, instances, offsets) = scene_data(0x9999, 48);
    let mut scene = DynamicScene::new(
        &soup_refs(&soups),
        &instances,
        policy(),
        BYTES_PER_PRIMITIVE,
        BYTES_PER_INSTANCE,
    );
    let mut ledger = RebuildLedger::new(1 << 24);

    for frame in 0..5u32 {
        // Deform a rotating BLAS each frame.
        let b = (frame % 3) as usize;
        let delta = [0.05 * f32::from(u16::try_from(frame + 1).unwrap()), 0.0, 0.03];
        let moved_soup = translate_soup(&soups[b], delta);
        let c = u32::try_from(soups[b].len()).unwrap();
        scene.deform_blas(b, refit_change(4, c), positions_of(&moved_soup), &mut ledger);
        soups[b] = moved_soup;

        // Small cumulative instance motion.
        let k = 0.01 * f32::from(u16::try_from(frame + 1).unwrap());
        let xf = divergent_move(&offsets, k);
        scene.sync_tlas(refit_change(10, 48), &xf, &mut ledger);

        let moved = place(&instances, &xf);
        let hits = assert_scene_matches(&scene, &soups, &moved, 900, 0x1000 + u64::from(frame));
        assert!(hits > 0, "frame {frame} hit nothing");
    }
}

#[test]
fn dirty_set_tracks_blas_changes() {
    let (soups, instances, _offsets) = scene_data(0xaaaa, 24);
    let mut scene = DynamicScene::new(
        &soup_refs(&soups),
        &instances,
        policy(),
        BYTES_PER_PRIMITIVE,
        BYTES_PER_INSTANCE,
    );
    let mut ledger = RebuildLedger::new(1 << 20);
    assert!(!scene.has_dirty_blas());

    // A no-op deform (Reuse) must not dirty the pool.
    let noop = reuse_change(u32::try_from(soups[0].len()).unwrap());
    let receipt = scene.deform_blas(0, noop, positions_of(&soups[0]), &mut ledger);
    assert_eq!(receipt.executed, AccelerationUpdate::Reuse);
    assert!(!scene.has_dirty_blas(), "a Reuse frame must not dirty the pool");

    // A refit frame dirties it until the next sync.
    let moved = translate_soup(&soups[0], [0.2, 0.0, 0.0]);
    let c = u32::try_from(soups[0].len()).unwrap();
    scene.deform_blas(0, refit_change(4, c), positions_of(&moved), &mut ledger);
    assert!(scene.has_dirty_blas());
    scene.sync_tlas_bounds_only(reuse_change(24), &mut ledger);
    assert!(!scene.has_dirty_blas());
}
