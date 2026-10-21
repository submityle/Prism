extern crate std;

use super::*;
use alloc::vec;
use core::convert::Infallible;
use prism_world_residency::{SdfQueryConsumer, SdfQueryResult, SparseSdfQuery};

fn config() -> WorldSdfObjectConfig {
    WorldSdfObjectConfig {
        origin_mm: [0; 3],
        voxel_size_mm: 1,
        distance_scale_mm: 7,
        brick_size: 2,
        mip_levels: 1,
    }
}

fn limits() -> WorldSdfObjectLimits {
    WorldSdfObjectLimits {
        max_objects: 256,
        max_bricks: 256,
        max_links: 1024,
        max_dirty_bricks: 256,
        max_edit_bricks: 512,
        max_edit_scratch_bytes: 1024 * 1024,
    }
}

fn budget() -> WorldSdfObjectBakeBudget {
    WorldSdfObjectBakeBudget {
        max_bricks: usize::MAX,
        max_evaluations: usize::MAX,
        max_scratch_bytes: usize::MAX,
    }
}

fn object(value: i16, translation: [i64; 3], max: [i64; 3]) -> WorldSdfObject<i16> {
    WorldSdfObject {
        translation_mm: translation,
        local_min_mm: [0; 3],
        local_max_mm: max,
        source: Arc::new(value),
    }
}

fn cube(value: i16, translation: [i64; 3]) -> WorldSdfObject<i16> {
    object(value, translation, [1; 3])
}

fn store() -> WorldSdfObjectStore<i16> {
    WorldSdfObjectStore::new(config(), limits()).unwrap()
}

fn constants(work: WorldSdfObjectBake<i16>) -> WorldSdfObjectUpdate<i16> {
    work.evaluate(|value, _| Ok::<_, Infallible>(*value))
        .unwrap()
}

fn bake(store: &mut WorldSdfObjectStore<i16>) -> WorldSdfObjectBakeReport {
    let update = constants(store.prepare(budget()).unwrap());
    store.commit(update).unwrap()
}

fn query(grid: &SparseSdfBrickGrid, voxel: [i32; 3], mip: u8) -> SdfQueryResult<i16> {
    SparseSdfQuery::new(grid).query_sample(voxel, mip).unwrap()
}

// Include source identity, index contents and the entire old grid, not merely
// counts: a rejected edit must not silently replace a source or fence valid work.
#[derive(Debug, Eq, PartialEq)]
struct State {
    revision: u64,
    objects: Vec<(u64, [[i64; 3]; 3], i16, *const i16)>,
    coverage: Vec<(BrickCoord, Vec<u64>)>,
    links: usize,
    reserved: usize,
    dirty: Vec<BrickCoord>,
    grid: SparseSdfBrickGrid,
}

fn state(store: &WorldSdfObjectStore<i16>) -> State {
    State {
        revision: store.revision(),
        objects: store
            .objects
            .iter()
            .map(|(&id, o)| {
                (
                    id,
                    [o.translation_mm, o.local_min_mm, o.local_max_mm],
                    *o.source,
                    Arc::as_ptr(&o.source),
                )
            })
            .collect(),
        coverage: store
            .coverage
            .iter()
            .map(|(&c, ids)| (c, ids.clone()))
            .collect(),
        links: store.link_count(),
        reserved: store.reserved_bricks(),
        dirty: store.dirty_bricks().to_vec(),
        grid: store.grid().clone(),
    }
}

fn assert_capacity(store: &WorldSdfObjectStore<i16>) {
    let mut union = store.grid().allocated_bricks();
    union.extend(store.coverage.keys().copied());
    union.sort_unstable();
    union.dedup();
    assert_eq!(store.reserved_bricks(), union.len());
    assert!(union.len() <= store.limits.max_bricks);
    assert!(store.grid().brick_count() <= store.limits.max_bricks);
}

#[test]
fn pointwise_union_move_and_delete_restore_survivors_then_unknown() {
    let mut store = store();
    // Neither contributor dominates the entire brick.
    let compose = |store: &mut WorldSdfObjectStore<i16>| {
        let update = store
            .prepare(budget())
            .unwrap()
            .evaluate(|value, sample| {
                let odd = sample.local_position_mm[0] % 2 != 0;
                Ok::<_, Infallible>(match (*value, odd) {
                    (0, false) => -6,
                    (0, true) => 8,
                    (_, false) => 3,
                    (_, true) => -2,
                })
            })
            .unwrap();
        store.commit(update).unwrap()
    };
    store.upsert(90, object(0, [0; 3], [3, 1, 1])).unwrap();
    store.upsert(5, object(10, [0; 3], [3, 1, 1])).unwrap();
    assert_eq!(compose(&mut store).evaluations, 32);
    for x in 0..4 {
        assert_eq!(
            query(store.ready_grid().unwrap(), [x, 1, 1], 0),
            SdfQueryResult::Value(if x % 2 == 0 { -6 } else { -2 })
        );
    }

    store.upsert(90, object(0, [2, 0, 0], [3, 1, 1])).unwrap();
    compose(&mut store);
    let grid = store.ready_grid().unwrap();
    assert_eq!(
        grid.brick_samples(BrickCoord::new(0, 0, 0, 0))
            .unwrap()
            .unwrap(),
        &[3, -2, 3, -2, 3, -2, 3, -2]
    );
    assert_eq!(query(grid, [2, 0, 0], 0), SdfQueryResult::Value(-6));
    assert_eq!(query(grid, [3, 0, 0], 0), SdfQueryResult::Value(-2));
    assert_eq!(query(grid, [4, 0, 0], 0), SdfQueryResult::Value(-6));
    assert_eq!(query(grid, [5, 0, 0], 0), SdfQueryResult::Value(8));

    assert!(store.remove(90).unwrap());
    let report = compose(&mut store);
    assert_eq!(report.evaluations, 8); // Only the overlapping surviving brick.
    for x in 0..4 {
        assert_eq!(
            query(store.ready_grid().unwrap(), [x, 0, 0], 0),
            SdfQueryResult::Value(if x % 2 == 0 { 3 } else { -2 })
        );
    }
    assert_eq!(query(store.grid(), [4, 0, 0], 0), SdfQueryResult::Unknown);
    assert!(store.remove(5).unwrap());
    assert_eq!(store.covered_bricks(), 0);
    assert_eq!(store.grid().brick_count(), 2); // Removal is not publication.
    assert_eq!(compose(&mut store).evaluations, 0);
    assert_eq!(store.ready_grid().unwrap().brick_count(), 0);
    assert_eq!(query(store.grid(), [0; 3], 0), SdfQueryResult::Unknown);
    assert_eq!(store.link_count(), 0);
    assert_capacity(&store);
}

#[test]
fn negative_bounds_nonzero_origin_and_mips_have_exact_x_fastest_samples() {
    let cfg = WorldSdfObjectConfig {
        origin_mm: [100, -200, 17],
        voxel_size_mm: 3,
        mip_levels: 3,
        ..config()
    };
    let translation = [99, -201, 16];
    let mut store = WorldSdfObjectStore::new(cfg, limits()).unwrap();
    store
        .upsert(
            81,
            WorldSdfObject {
                translation_mm: translation,
                local_min_mm: [-11; 3],
                local_max_mm: [1; 3],
                source: Arc::new(0),
            },
        )
        .unwrap();
    assert_eq!(store.covered_bricks(), 43); // 3^3 + 2^3 + 2^3.
    let mut seen = Vec::new();
    let distance = |p: WorldSdfObjectSample| {
        (p.local_position_mm[0]
            + 2 * p.local_position_mm[1]
            + 4 * p.local_position_mm[2]
            + 100 * i128::from(p.brick.mip)) as i16
    };
    let work = store.prepare(budget()).unwrap();
    assert_eq!(work.evaluation_bound(), 43 * 8);
    let update = work
        .evaluate(|_, p| {
            seen.push(p);
            Ok::<_, Infallible>(distance(p))
        })
        .unwrap();

    // Enumerate physical lattice positions independently of brick-local loops.
    let mut expected = Vec::new();
    for (mip, lo) in [(0_u8, -4_i32), (1, -2), (2, -1)] {
        for z in lo..=0 {
            for y in lo..=0 {
                for x in lo..=0 {
                    let voxel = [x, y, z];
                    let world = core::array::from_fn(|a| {
                        i128::from(cfg.origin_mm[a]) + i128::from(voxel[a]) * (3_i128 << mip)
                    });
                    expected.push(WorldSdfObjectSample {
                        object_id: 81,
                        world_position_mm: world,
                        local_position_mm: core::array::from_fn(|a| {
                            world[a] - i128::from(translation[a])
                        }),
                        brick: BrickCoord::new(
                            x.div_euclid(2),
                            y.div_euclid(2),
                            z.div_euclid(2),
                            mip,
                        ),
                        distance_scale_mm: 7,
                    });
                }
            }
        }
    }
    expected.sort_unstable_by_key(|p| {
        (
            p.brick,
            p.world_position_mm[2],
            p.world_position_mm[1],
            p.world_position_mm[0],
        )
    });
    assert_eq!(seen, expected);
    let report = store.commit(update).unwrap();
    assert_eq!(report.evaluations, 125 + 27 + 8);
    for p in &expected {
        let voxel = core::array::from_fn(|a| {
            ((p.world_position_mm[a] - i128::from(cfg.origin_mm[a])) / (3_i128 << p.brick.mip))
                as i32
        });
        assert_eq!(
            query(store.grid(), voxel, p.brick.mip),
            SdfQueryResult::Value(distance(*p))
        );
    }
    assert_eq!(
        query(store.grid(), [1, 0, 0], 0),
        SdfQueryResult::Value(i16::MAX)
    );
    assert_eq!(query(store.grid(), [-6, 0, 0], 0), SdfQueryResult::Unknown);

    store.remove(81).unwrap();
    let work = store.prepare(budget()).unwrap();
    assert_eq!(work.evaluation_bound(), 0);
    let update = work
        .evaluate::<Infallible>(|_, _| panic!("removals must not sample"))
        .unwrap();
    store.commit(update).unwrap();
    assert_eq!(store.ready_grid().unwrap().brick_count(), 0);
    assert_eq!(store.grid().dirty_count(), 0); // Legacy parent dirtiness is drained.
    assert_capacity(&store);
}

#[test]
fn physical_positions_can_exceed_i64_without_narrowing_even_at_mip_31() {
    let cfg = WorldSdfObjectConfig {
        origin_mm: [i64::MAX - 1, i64::MIN + 1, 23],
        brick_size: 1,
        mip_levels: 32,
        ..config()
    };
    let mut store = WorldSdfObjectStore::new(cfg, limits()).unwrap();
    store
        .upsert(
            u64::MAX,
            WorldSdfObject {
                translation_mm: [i64::MAX, i64::MIN, 20],
                local_min_mm: [3, -3, 3],
                local_max_mm: [3, -3, 3],
                source: Arc::new(-12),
            },
        )
        .unwrap();
    assert_eq!(store.covered_bricks(), 32);
    let mut seen = Vec::new();
    let update = store
        .prepare(budget())
        .unwrap()
        .evaluate(|value, p| {
            assert_eq!(
                p.world_position_mm,
                [i128::from(i64::MAX) + 3, i128::from(i64::MIN) - 3, 23]
            );
            assert_eq!(p.local_position_mm, [3, -3, 3]);
            assert_eq!(p.object_id, u64::MAX);
            assert_eq!(p.distance_scale_mm, 7);
            seen.push(p.brick);
            Ok::<_, Infallible>(*value)
        })
        .unwrap();
    assert_eq!(
        seen,
        vec![
            BrickCoord::new(1, -1, 0, 2),
            BrickCoord::new(2, -2, 0, 1),
            BrickCoord::new(4, -4, 0, 0)
        ]
    );
    let report = store.commit(update).unwrap();
    assert_eq!(report.evaluation_bound, 32);
    assert_eq!(report.evaluations, 3);
    assert_eq!(
        query(store.ready_grid().unwrap(), [0, -1, 0], 31),
        SdfQueryResult::Value(i16::MAX)
    );
    assert_eq!(
        query(store.grid(), [4, -4, 0], 0),
        SdfQueryResult::Value(-12)
    );
}

#[test]
fn contributors_are_sorted_by_identity_and_local_work_retains_no_remote_sources() {
    let mut store = store();
    for id in [42, 7, 19] {
        store.upsert(id, cube(id as i16, [0; 3])).unwrap();
    }
    let remote = cube(-300, [100, 0, 0]);
    store.upsert(900, remote.clone()).unwrap();
    bake(&mut store);
    let remote_samples = store
        .grid()
        .brick_samples(BrickCoord::new(50, 0, 0, 0))
        .unwrap()
        .unwrap()
        .to_vec();
    store.upsert(19, cube(-19, [0; 3])).unwrap();
    store.remove(7).unwrap();
    store.upsert(7, cube(7, [0; 3])).unwrap();
    let refs = Arc::strong_count(&remote.source);
    let work = store.prepare(budget()).unwrap();
    assert_eq!(Arc::strong_count(&remote.source), refs);
    assert_eq!(work.brick_count(), 1);
    assert_eq!(work.evaluation_bound(), 24);
    let mut seen = Vec::new();
    let update = work
        .evaluate(|source, sample| {
            assert_eq!(sample.brick, BrickCoord::new(0, 0, 0, 0));
            assert!(sample.object_id != 900, "unrelated source was evaluated");
            seen.push((sample.object_id, *source));
            Ok::<_, Infallible>(*source)
        })
        .unwrap();
    assert_eq!(
        update.changed_bricks().collect::<Vec<_>>(),
        vec![BrickCoord::new(0, 0, 0, 0)]
    );
    assert_eq!(seen, [(7, 7), (19, -19), (42, 42)].repeat(8));
    assert_eq!(store.commit(update).unwrap().evaluations, 24);
    assert_eq!(
        query(store.ready_grid().unwrap(), [0; 3], 0),
        SdfQueryResult::Value(-19)
    );
    assert_eq!(
        store
            .grid()
            .brick_samples(BrickCoord::new(50, 0, 0, 0))
            .unwrap()
            .unwrap(),
        remote_samples
    );
    assert_eq!(Arc::strong_count(&remote.source), refs);
}

#[test]
fn partial_publication_fences_ready_grid_and_never_exceeds_union_capacity() {
    let mut store = WorldSdfObjectStore::new(
        config(),
        WorldSdfObjectLimits {
            max_bricks: 3,
            ..limits()
        },
    )
    .unwrap();
    store.upsert(1, object(7, [0; 3], [3, 1, 1])).unwrap();
    bake(&mut store);
    store.upsert(1, object(-11, [2, 0, 0], [3, 1, 1])).unwrap();
    assert_eq!(store.covered_bricks(), 2);
    assert_eq!(store.reserved_bricks(), 3);
    let one = WorldSdfObjectBakeBudget {
        max_bricks: 1,
        ..budget()
    };
    for (remaining, coord) in [(3, 2), (2, 1), (1, 0)] {
        assert!(
            matches!(store.ready_grid(), Err(WorldSdfObjectError::PendingBake { bricks }) if bricks == remaining)
        );
        let before = state(&store);
        let update = constants(store.prepare(one).unwrap());
        assert_eq!(state(&store), before, "evaluation must not publish");
        assert_eq!(
            update.changed_bricks().collect::<Vec<_>>(),
            vec![BrickCoord::new(coord, 0, 0, 0)]
        );
        if remaining == 2 {
            // Old bricks remain charged even after the new suffix was inserted.
            let error = store
                .upsert(1, object(-22, [4, 0, 0], [3, 1, 1]))
                .err()
                .unwrap();
            assert_eq!(
                error,
                WorldSdfObjectError::BudgetExceeded {
                    resource: WorldSdfObjectResource::Bricks,
                    required: 4,
                    limit: 3
                }
            );
            assert_eq!(state(&store), before);
        }
        let report = store.commit(update).unwrap();
        assert_eq!(report.changed_bricks, 1);
        assert_eq!(report.pending_bricks, remaining - 1);
        assert_eq!(report.evaluations, if coord == 0 { 0 } else { 8 });
        assert_eq!(report.revision, before.revision + 1);
        assert_capacity(&store);
        if remaining > 1 {
            assert_eq!(store.grid().brick_count(), 3);
            assert_eq!(query(store.grid(), [0; 3], 0), SdfQueryResult::Value(7));
            assert!(store.ready_grid().is_err());
        }
    }
    assert_eq!(store.ready_grid().unwrap().brick_count(), 2);
    assert_eq!(query(store.grid(), [0; 3], 0), SdfQueryResult::Unknown);
    assert_eq!(
        query(store.grid(), [2, 0, 0], 0),
        SdfQueryResult::Value(-11)
    );
    assert_eq!(
        query(store.grid(), [4, 0, 0], 0),
        SdfQueryResult::Value(-11)
    );
    assert_eq!(store.reserved_bricks(), 2);
}

#[test]
fn capacity_one_requires_remove_then_bake_before_adding_elsewhere() {
    let cap = WorldSdfObjectLimits {
        max_bricks: 1,
        ..limits()
    };
    let mut store = WorldSdfObjectStore::new(config(), cap).unwrap();
    store.upsert(1, cube(5, [0; 3])).unwrap();
    bake(&mut store);
    let before = state(&store);
    assert!(matches!(
        store.upsert(1, cube(-5, [2, 0, 0])),
        Err(WorldSdfObjectError::BudgetExceeded {
            resource: WorldSdfObjectResource::Bricks,
            required: 2,
            limit: 1,
        })
    ));
    assert_eq!(state(&store), before);
    store.remove(1).unwrap();
    assert_eq!(store.covered_bricks(), 0);
    assert_eq!(store.reserved_bricks(), 1);
    let before = state(&store);
    assert!(matches!(
        store.upsert(2, cube(-5, [2, 0, 0])),
        Err(WorldSdfObjectError::BudgetExceeded {
            resource: WorldSdfObjectResource::Bricks,
            required: 2,
            limit: 1,
        })
    ));
    assert_eq!(state(&store), before);
    assert_eq!(query(store.grid(), [0; 3], 0), SdfQueryResult::Value(5));
    assert_eq!(bake(&mut store).evaluations, 0);
    assert_eq!(store.reserved_bricks(), 0);
    assert_eq!(
        query(store.ready_grid().unwrap(), [0; 3], 0),
        SdfQueryResult::Unknown
    );
    store.upsert(2, cube(-5, [2, 0, 0])).unwrap();
    bake(&mut store);
    assert_eq!(
        query(store.ready_grid().unwrap(), [2, 0, 0], 0),
        SdfQueryResult::Value(-5)
    );
    assert_capacity(&store);

    // An unmaterialized old coordinate needs no persistent reservation. Its
    // later no-op removal must not decrement the surviving reservation.
    let mut unbaked = WorldSdfObjectStore::new(config(), cap).unwrap();
    unbaked.upsert(1, cube(5, [0; 3])).unwrap();
    unbaked.upsert(1, cube(-5, [2, 0, 0])).unwrap();
    assert_eq!(unbaked.dirty_bricks().len(), 2);
    for _ in 0..2 {
        let update = constants(
            unbaked
                .prepare(WorldSdfObjectBakeBudget {
                    max_bricks: 1,
                    ..budget()
                })
                .unwrap(),
        );
        unbaked.commit(update).unwrap();
        assert_capacity(&unbaked);
    }
    assert_eq!(unbaked.ready_grid().unwrap().brick_count(), 1);
}

#[test]
fn owner_edit_and_publication_fences_distinguish_real_changes_from_noops() {
    let mut a = store();
    let mut b = store();
    let source = cube(9, [0; 3]);
    a.upsert(1, source.clone()).unwrap();
    b.upsert(1, source.clone()).unwrap();
    assert_eq!(a.revision(), b.revision());
    let before = state(&a);
    assert_eq!(
        a.commit(constants(b.prepare(budget()).unwrap())),
        Err(WorldSdfObjectError::StaleWork)
    );
    assert_eq!(state(&a), before);

    let old = constants(a.prepare(budget()).unwrap());
    a.upsert(1, source).unwrap(); // Same bounds and Arc still constitute an edit.
    assert_eq!(a.revision(), before.revision + 1);
    let before = state(&a);
    assert_eq!(a.commit(old), Err(WorldSdfObjectError::StaleWork));
    assert_eq!(state(&a), before);
    let current = constants(a.prepare(budget()).unwrap());
    let duplicate = constants(a.prepare(budget()).unwrap());
    let zero = WorldSdfObjectBakeBudget {
        max_bricks: 0,
        max_evaluations: 0,
        max_scratch_bytes: 0,
    };
    let empty = constants(a.prepare(zero).unwrap());
    let empty_before_publication = constants(a.prepare(zero).unwrap());
    assert!(!a.remove(999).unwrap());
    assert_eq!(state(&a), before);
    let report = a.commit(empty).unwrap();
    assert_eq!(report.changed_bricks, 0);
    assert_eq!(report.pending_bricks, 1);
    assert_eq!(state(&a), before);
    a.commit(current).unwrap();
    let ready = state(&a);
    assert_eq!(a.commit(duplicate), Err(WorldSdfObjectError::StaleWork));
    assert_eq!(
        a.commit(empty_before_publication),
        Err(WorldSdfObjectError::StaleWork)
    );
    assert_eq!(state(&a), ready);
    assert_eq!(bake(&mut a).changed_bricks, 0);
    assert_eq!(state(&a), ready);
}

#[test]
fn evaluator_error_and_panic_after_partial_progress_preserve_the_old_grid() {
    let mut store = store();
    store.upsert(44, object(7, [0; 3], [3, 1, 1])).unwrap();
    bake(&mut store);
    store.upsert(44, object(-9, [0; 3], [3, 1, 1])).unwrap();
    let before = state(&store);
    let mut calls = 0;
    let error = store
        .prepare(budget())
        .unwrap()
        .evaluate(|value, _| {
            calls += 1;
            if calls == 10 {
                Err("cancelled")
            } else {
                Ok(*value)
            }
        })
        .err()
        .unwrap();
    assert_eq!(calls, 10); // The first brick has already been fully evaluated.
    assert_eq!(
        error,
        WorldSdfObjectBakeError {
            sample: WorldSdfObjectSample {
                object_id: 44,
                world_position_mm: [3, 0, 0],
                local_position_mm: [3, 0, 0],
                brick: BrickCoord::new(1, 0, 0, 0),
                distance_scale_mm: 7,
            },
            source: "cancelled",
        }
    );
    assert_eq!(state(&store), before);
    #[cfg(panic = "unwind")]
    {
        let work = store.prepare(budget()).unwrap();
        let mut calls = 0;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = work.evaluate::<Infallible>(|value, _| {
                calls += 1;
                assert!(
                    calls != 10,
                    "intentional evaluator panic after one completed brick"
                );
                Ok(*value)
            });
        }));
        assert!(result.is_err());
        assert_eq!(calls, 10);
        assert_eq!(state(&store), before);
    }
    assert!(matches!(
        store.ready_grid(),
        Err(WorldSdfObjectError::PendingBake { bricks: 2 })
    ));
    assert_eq!(bake(&mut store).evaluations, 16);
    assert_eq!(
        store
            .ready_grid()
            .unwrap()
            .brick_samples(BrickCoord::new(0, 0, 0, 0))
            .unwrap()
            .unwrap(),
        &[-9; 8]
    );
    assert_eq!(query(store.grid(), [3, 1, 1], 0), SdfQueryResult::Value(-9));
}

#[test]
fn each_edit_budget_failure_is_atomic_including_raw_old_plus_new_coverage() {
    use WorldSdfObjectResource as R;
    let first_edit_bytes = 3 * size_of::<BrickCoord>() + size_of::<Coverage>() + size_of::<u64>();
    for resource in [
        R::Objects,
        R::Bricks,
        R::Links,
        R::DirtyBricks,
        R::EditBricks,
        R::ScratchBytes,
    ] {
        let mut lim = limits();
        match resource {
            R::Objects => lim.max_objects = 1,
            R::Bricks => lim.max_bricks = 1,
            R::Links => lim.max_links = 1,
            R::DirtyBricks => lim.max_dirty_bricks = 1,
            R::EditBricks => lim.max_edit_bricks = 1,
            R::ScratchBytes => lim.max_edit_scratch_bytes = first_edit_bytes,
            _ => unreachable!(),
        }
        let mut store = WorldSdfObjectStore::new(config(), lim).unwrap();
        store.upsert(1, cube(11, [0; 3])).unwrap();
        bake(&mut store);
        let before = state(&store);
        let valid_empty = constants(store.prepare(budget()).unwrap());
        let (id, translation) = match resource {
            R::Objects | R::Links => (2, [0; 3]),
            R::Bricks | R::DirtyBricks => (1, [2, 0, 0]),
            _ => (1, [0; 3]),
        };
        let (required, limit) = if resource == R::ScratchBytes {
            (
                first_edit_bytes + 2 * size_of::<BrickCoord>(),
                first_edit_bytes,
            )
        } else {
            (2, 1)
        };
        assert_eq!(
            store.upsert(id, cube(-11, translation)),
            Err(WorldSdfObjectError::BudgetExceeded {
                resource,
                required,
                limit,
            })
        );
        assert_eq!(state(&store), before, "resource={resource:?}");
        assert_eq!(store.commit(valid_empty).unwrap().changed_bricks, 0);
        assert_eq!(state(&store), before);
        assert_capacity(&store);
    }

    // Removing a different identity can fail dirty admission as well. A job
    // prepared before that rejection remains publishable at the same revision.
    let mut store = WorldSdfObjectStore::new(
        config(),
        WorldSdfObjectLimits {
            max_dirty_bricks: 1,
            ..limits()
        },
    )
    .unwrap();
    store.upsert(1, cube(1, [0; 3])).unwrap();
    bake(&mut store);
    store.upsert(2, cube(2, [2, 0, 0])).unwrap();
    bake(&mut store);
    store.upsert(1, cube(-1, [0; 3])).unwrap();
    let update = constants(store.prepare(budget()).unwrap());
    let before = state(&store);
    assert_eq!(
        store.remove(2),
        Err(WorldSdfObjectError::BudgetExceeded {
            resource: R::DirtyBricks,
            required: 2,
            limit: 1,
        })
    );
    assert_eq!(state(&store), before);
    store.commit(update).unwrap();
    assert_eq!(store.object_count(), 2);
    assert_eq!(
        query(store.ready_grid().unwrap(), [2, 0, 0], 0),
        SdfQueryResult::Value(2)
    );
}

#[test]
fn bake_budget_failures_do_not_shrink_work_and_zero_batch_is_a_true_noop() {
    let mut store = store();
    store.upsert(1, object(4, [0; 3], [3, 1, 1])).unwrap();
    let before = state(&store);
    let work = store.prepare(budget()).unwrap();
    assert_eq!(work.brick_count(), 2);
    assert_eq!(work.evaluation_bound(), 16);
    let bytes = work.scratch_bytes();
    assert!(bytes > 0);
    drop(work);
    assert_eq!(
        store
            .prepare(WorldSdfObjectBakeBudget {
                max_evaluations: 15,
                ..budget()
            })
            .err()
            .unwrap(),
        WorldSdfObjectError::BudgetExceeded {
            resource: WorldSdfObjectResource::Evaluations,
            required: 16,
            limit: 15,
        }
    );
    assert_eq!(state(&store), before);
    assert_eq!(
        store
            .prepare(WorldSdfObjectBakeBudget {
                max_scratch_bytes: bytes - 1,
                ..budget()
            })
            .err()
            .unwrap(),
        WorldSdfObjectError::BudgetExceeded {
            resource: WorldSdfObjectResource::ScratchBytes,
            required: bytes,
            limit: bytes - 1,
        }
    );
    assert_eq!(state(&store), before);
    let zero = store
        .prepare(WorldSdfObjectBakeBudget {
            max_bricks: 0,
            max_evaluations: 0,
            max_scratch_bytes: 0,
        })
        .unwrap();
    assert_eq!(
        (
            zero.brick_count(),
            zero.evaluation_bound(),
            zero.scratch_bytes()
        ),
        (0, 0, 0)
    );
    let update = zero
        .evaluate::<Infallible>(|_, _| panic!("zero batch called evaluator"))
        .unwrap();
    let report = store.commit(update).unwrap();
    assert_eq!(
        (
            report.changed_bricks,
            report.evaluations,
            report.evaluation_bound,
            report.scratch_bytes
        ),
        (0, 0, 0, 0)
    );
    assert_eq!(report.pending_bricks, 2);
    assert_eq!(state(&store), before);
    let update = constants(
        store
            .prepare(WorldSdfObjectBakeBudget {
                max_bricks: 2,
                max_evaluations: 16,
                max_scratch_bytes: bytes,
            })
            .unwrap(),
    );
    let report = store.commit(update).unwrap();
    assert_eq!(
        (
            report.changed_bricks,
            report.evaluations,
            report.scratch_bytes
        ),
        (2, 16, bytes)
    );
    assert!(store.ready_grid().is_ok());
}

#[test]
fn invalid_geometry_bounds_and_coordinate_arithmetic_do_not_poison_the_store() {
    let invalid = [
        WorldSdfObjectConfig {
            voxel_size_mm: 0,
            ..config()
        },
        WorldSdfObjectConfig {
            distance_scale_mm: 0,
            ..config()
        },
        WorldSdfObjectConfig {
            brick_size: 0,
            ..config()
        },
        WorldSdfObjectConfig {
            brick_size: 33,
            ..config()
        },
        WorldSdfObjectConfig {
            brick_size: u8::MAX,
            ..config()
        },
        WorldSdfObjectConfig {
            mip_levels: 0,
            ..config()
        },
        WorldSdfObjectConfig {
            mip_levels: 33,
            ..config()
        },
        WorldSdfObjectConfig {
            mip_levels: u8::MAX,
            ..config()
        },
    ];
    for cfg in invalid {
        assert!(matches!(
            WorldSdfObjectStore::<i16>::new(cfg, limits()),
            Err(WorldSdfObjectError::InvalidConfiguration)
        ));
    }
    let zero_limits = WorldSdfObjectLimits {
        max_objects: 0,
        max_bricks: 0,
        max_links: 0,
        max_dirty_bricks: 0,
        max_edit_bricks: 0,
        max_edit_scratch_bytes: 0,
    };
    for brick_size in [1, 32] {
        for mip_levels in [1, 32] {
            let cfg = WorldSdfObjectConfig {
                origin_mm: [i64::MIN, i64::MAX, -17],
                voxel_size_mm: u32::MAX,
                distance_scale_mm: u32::MAX,
                brick_size,
                mip_levels,
            };
            let mut empty = WorldSdfObjectStore::<i16>::new(cfg, zero_limits).unwrap();
            assert_eq!(empty.config(), cfg);
            assert_eq!(empty.ready_grid().unwrap().brick_count(), 0);
            let before = state(&empty);
            assert_eq!(
                empty.upsert(1, cube(1, [0; 3])),
                Err(WorldSdfObjectError::BudgetExceeded {
                    resource: WorldSdfObjectResource::Objects,
                    required: 1,
                    limit: 0,
                })
            );
            assert_eq!(state(&empty), before);
            assert_eq!(bake(&mut empty).changed_bricks, 0);
            assert_eq!(state(&empty), before);
        }
    }

    let mut store = store();
    store.upsert(1, cube(8, [0; 3])).unwrap();
    let update = constants(store.prepare(budget()).unwrap());
    let before = state(&store);
    for axis in 0..3 {
        let mut invalid = cube(-8, [0; 3]);
        invalid.local_min_mm[axis] = 2;
        assert_eq!(
            store.upsert(1, invalid),
            Err(WorldSdfObjectError::InvalidBounds)
        );
        assert_eq!(state(&store), before);
        for extreme in [i64::MIN, i64::MAX] {
            let mut invalid = cube(-8, [0; 3]);
            invalid.translation_mm[axis] = extreme;
            assert_eq!(
                store.upsert(1, invalid),
                Err(WorldSdfObjectError::CoordinateOverflow)
            );
            assert_eq!(state(&store), before);
        }
    }
    assert_eq!(
        store.upsert(1, object(-8, [0; 3], [i64::from(i32::MAX); 3])),
        Err(WorldSdfObjectError::ArithmeticOverflow)
    );
    assert_eq!(state(&store), before);
    store.commit(update).unwrap();
    assert_eq!(
        query(store.ready_grid().unwrap(), [0; 3], 0),
        SdfQueryResult::Value(8)
    );
}

#[test]
fn multi_mip_pure_removal_has_exact_edit_and_bake_scratch_boundaries() {
    let cfg = WorldSdfObjectConfig {
        mip_levels: 4,
        ..config()
    };
    let mut store = WorldSdfObjectStore::new(cfg, limits()).unwrap();
    store.upsert(1, cube(-17, [0; 3])).unwrap();
    bake(&mut store);
    let coords: Vec<_> = (0..cfg.mip_levels)
        .map(|mip| BrickCoord::new(0, 0, 0, mip))
        .collect();
    assert_eq!(store.grid().brick_count(), coords.len());

    // Pure deletion stages old/touched/dirty coordinates and Coverage entries,
    // but neither replacement coordinates nor surviving contributor ids.
    let edit_bytes = coords.len() * (3 * size_of::<BrickCoord>() + size_of::<Coverage>());
    store.limits.max_edit_scratch_bytes = edit_bytes - 1;
    let before = state(&store);
    assert_eq!(
        store.remove(1),
        Err(WorldSdfObjectError::BudgetExceeded {
            resource: WorldSdfObjectResource::ScratchBytes,
            required: edit_bytes,
            limit: edit_bytes - 1,
        })
    );
    assert_eq!(state(&store), before);
    store.limits.max_edit_scratch_bytes = edit_bytes;
    assert!(store.remove(1).unwrap());
    assert_eq!(store.revision(), before.revision + 1);
    assert_eq!(store.dirty_bricks(), coords.as_slice());
    assert_eq!(store.covered_bricks(), 0);
    assert_eq!(store.reserved_bricks(), coords.len());
    assert!(matches!(
        store.ready_grid(),
        Err(WorldSdfObjectError::PendingBake { bricks: 4 })
    ));
    let before = state(&store);

    // Deterministic suffixes select mips 3, then 2..3, then 1..3, then 0..3.
    // Their legacy parent flags cost 0, 1, 3 and 6 coordinates respectively;
    // removal must not charge any source descriptors or replacement samples.
    for (count, parent_count) in [(1, 0), (2, 1), (3, 3), (4, 6)] {
        let bytes = count * size_of::<BakeBrick<i16>>() + parent_count * size_of::<BrickCoord>();
        let exact = WorldSdfObjectBakeBudget {
            max_bricks: count,
            max_evaluations: 0,
            max_scratch_bytes: bytes,
        };
        assert_eq!(
            store
                .prepare(WorldSdfObjectBakeBudget {
                    max_scratch_bytes: bytes - 1,
                    ..exact
                })
                .err()
                .unwrap(),
            WorldSdfObjectError::BudgetExceeded {
                resource: WorldSdfObjectResource::ScratchBytes,
                required: bytes,
                limit: bytes - 1,
            }
        );
        assert_eq!(state(&store), before);
        let work = store.prepare(exact).unwrap();
        assert_eq!(work.brick_count(), count);
        assert_eq!(work.evaluation_bound(), 0);
        assert_eq!(work.scratch_bytes(), bytes);
        let update = work
            .evaluate::<Infallible>(|_, _| panic!("pure removal called evaluator"))
            .unwrap();
        assert_eq!(
            update.changed_bricks().collect::<Vec<_>>(),
            coords[coords.len() - count..]
        );
        assert_eq!(state(&store), before);
        if count == coords.len() {
            let report = store.commit(update).unwrap();
            assert_eq!(report.revision, before.revision + 1);
            assert_eq!(report.changed_bricks, count);
            assert_eq!(report.scratch_bytes, bytes);
            assert_eq!(report.evaluations, 0);
            assert_eq!(report.evaluation_bound, 0);
            assert_eq!(report.pending_bricks, 0);
        }
    }
    assert_eq!(store.ready_grid().unwrap().brick_count(), 0);
    assert_eq!(store.grid().dirty_count(), 0);
    assert_eq!(store.reserved_bricks(), 0);
    for mip in 0..cfg.mip_levels {
        assert_eq!(query(store.grid(), [0; 3], mip), SdfQueryResult::Unknown);
    }
    assert_capacity(&store);
}

#[test]
fn revision_max_rejects_edits_atomically_but_preserves_missing_remove_and_empty_commit() {
    for pending in [false, true] {
        let mut store = store();
        store.upsert(1, cube(5, [0; 3])).unwrap();
        store.upsert(2, cube(11, [16, 0, 0])).unwrap();
        bake(&mut store);
        if pending {
            // The final representable edit succeeds, but does not publish its
            // replacement samples or drop the already materialized old grid.
            store.revision = u64::MAX - 1;
            store.upsert(1, cube(-5, [0; 3])).unwrap();
        } else {
            store.revision = u64::MAX;
        }
        assert_eq!(store.revision(), u64::MAX);
        let before = state(&store);
        let empty = constants(
            store
                .prepare(WorldSdfObjectBakeBudget {
                    max_bricks: 0,
                    max_evaluations: 0,
                    max_scratch_bytes: 0,
                })
                .unwrap(),
        );

        let same_source = store.objects[&1].clone();
        for (id, replacement) in [
            (1, same_source),
            (1, cube(-20, [0; 3])),
            (1, cube(-30, [2, 0, 0])),
            (3, cube(-40, [32, 0, 0])),
        ] {
            assert_eq!(
                store.upsert(id, replacement),
                Err(WorldSdfObjectError::ArithmeticOverflow)
            );
            assert_eq!(state(&store), before);
        }
        for id in [1, 2] {
            assert_eq!(
                store.remove(id),
                Err(WorldSdfObjectError::ArithmeticOverflow)
            );
            assert_eq!(state(&store), before);
        }
        assert!(!store.remove(999).unwrap());
        assert_eq!(state(&store), before);
        let report = store.commit(empty).unwrap();
        assert_eq!(report.revision, u64::MAX);
        assert_eq!(report.changed_bricks, 0);
        assert_eq!(report.evaluations, 0);
        assert_eq!(report.evaluation_bound, 0);
        assert_eq!(report.scratch_bytes, 0);
        assert_eq!(report.pending_bricks, usize::from(pending));
        assert_eq!(state(&store), before);
        assert_eq!(query(store.grid(), [0; 3], 0), SdfQueryResult::Value(5));
        assert_eq!(
            query(store.grid(), [16, 0, 0], 0),
            SdfQueryResult::Value(11)
        );
        if pending {
            assert!(matches!(
                store.ready_grid(),
                Err(WorldSdfObjectError::PendingBake { bricks: 1 })
            ));
        } else {
            assert!(store.ready_grid().is_ok());
        }
        assert_capacity(&store);
    }
}

#[test]
fn revision_max_commit_is_atomic_for_replacement_and_removal_at_the_exact_boundary() {
    for remove in [false, true] {
        for revision in [u64::MAX - 1, u64::MAX] {
            let cfg = WorldSdfObjectConfig {
                mip_levels: 3,
                ..config()
            };
            let mut store = WorldSdfObjectStore::new(cfg, limits()).unwrap();
            store.upsert(1, cube(5, [0; 3])).unwrap();
            store.upsert(2, cube(11, [16, 0, 0])).unwrap();
            bake(&mut store);
            if remove {
                assert!(store.remove(1).unwrap());
            } else {
                store.upsert(1, cube(-5, [0; 3])).unwrap();
            }
            store.revision = revision;
            let before = state(&store);
            let update = constants(store.prepare(budget()).unwrap());
            let duplicate = constants(store.prepare(budget()).unwrap());
            assert_eq!(update.changed_bricks().len(), 3);
            assert_eq!(state(&store), before);

            if revision == u64::MAX {
                assert_eq!(
                    store.commit(update),
                    Err(WorldSdfObjectError::ArithmeticOverflow)
                );
                assert_eq!(state(&store), before);
                // Rejection neither consumes pending coordinates nor fences
                // another job from the same owner and revision as stale.
                assert_eq!(
                    store.commit(duplicate),
                    Err(WorldSdfObjectError::ArithmeticOverflow)
                );
                assert_eq!(state(&store), before);
                assert!(matches!(
                    store.ready_grid(),
                    Err(WorldSdfObjectError::PendingBake { bricks: 3 })
                ));
                assert_eq!(store.reserved_bricks(), 6);
            } else {
                let report = store.commit(update).unwrap();
                assert_eq!(report.revision, u64::MAX);
                assert_eq!(report.changed_bricks, 3);
                assert_eq!(report.pending_bricks, 0);
                assert_eq!(report.evaluations, if remove { 0 } else { 10 });
                assert_eq!(report.evaluation_bound, if remove { 0 } else { 24 });
                let published = state(&store);
                assert_eq!(store.commit(duplicate), Err(WorldSdfObjectError::StaleWork));
                assert_eq!(state(&store), published);
                assert!(store.ready_grid().is_ok());
                assert_eq!(store.reserved_bricks(), if remove { 3 } else { 6 });
            }
            for mip in 0..cfg.mip_levels {
                let expected = if revision == u64::MAX {
                    SdfQueryResult::Value(5)
                } else if remove {
                    SdfQueryResult::Unknown
                } else {
                    SdfQueryResult::Value(-5)
                };
                assert_eq!(query(store.grid(), [0; 3], mip), expected);
                assert_eq!(
                    query(store.grid(), [16_i32 >> mip, 0, 0], mip),
                    SdfQueryResult::Value(11)
                );
            }
            assert_eq!(store.grid().dirty_count(), 0);
            assert_capacity(&store);
        }
    }
}
