//! Independent voxel/parent oracles and failure injection for atomic local edits.

extern crate alloc;

use alloc::{collections::BTreeSet, vec::Vec};
use core::convert::Infallible;
use prism_world_physics_bridge::{
    SdfBridgeError, SdfPhysicsBridgeBudget, SdfQueryResult, SdfRebakeError, SdfRebakeLimits,
    SdfRebakePoint, SdfRebakeRegion, SdfRebakeResource, WorldSdfPhysicsBridge, WorldSdfSnapshot,
    WorldSdfTransform,
};
use prism_world_residency::{BrickCoord, SdfError, SparseSdfBrickGrid};
use std::panic::{catch_unwind, AssertUnwindSafe};

const LIMITS: SdfRebakeLimits = SdfRebakeLimits {
    max_bricks: 4096,
    max_samples: 1_000_000,
    max_scratch_bytes: 4_000_000,
};

fn region(min_voxel: [i32; 3], max_voxel: [i32; 3], mip: u8) -> SdfRebakeRegion {
    SdfRebakeRegion {
        min_voxel,
        max_voxel,
        mip,
    }
}

fn make_bridge(grid: SparseSdfBrickGrid, transform: WorldSdfTransform) -> WorldSdfPhysicsBridge {
    let mut bridge = WorldSdfPhysicsBridge::new(
        SdfPhysicsBridgeBudget::new(4096, 1_000_000, 4_000_000).expect("budget"),
    );
    bridge
        .publish(WorldSdfSnapshot::new(transform, 7, grid))
        .expect("publish");
    bridge
}

fn empty(edge: u8, mips: u8) -> WorldSdfPhysicsBridge {
    make_bridge(
        SparseSdfBrickGrid::new(edge, mips).expect("grid"),
        WorldSdfTransform::new([0; 3], 10, 2).expect("transform"),
    )
}

fn quantized(point: SdfRebakePoint) -> Result<i16, Infallible> {
    let [x, y, z] = point.voxel;
    Ok(i16::try_from((x + 2 * y - 3 * z) * (1_i64 << point.brick.mip)).expect("bounded field"))
}

/// Detached evaluation preserves traversal, costs, and output of the synchronous API.
#[test]
fn detached_work_matches_sync_without_holding_the_resident_borrow() {
    let mut direct = empty(3, 3);
    let mut detached = direct.clone();
    let before = detached.current().expect("current").clone();
    let edit = region([-4, -1, 2], [0, 2, 3], 0);
    let mut direct_points = Vec::new();
    let report = direct
        .rebake_region(edit, 8, LIMITS, |point| {
            direct_points.push(point);
            quantized(point)
        })
        .expect("sync");
    let work = detached.prepare_rebake(edit, 8, LIMITS).expect("prepare");
    assert_eq!(work.report(), report);
    assert_eq!(detached.current(), Some(&before));
    let mut detached_points = Vec::new();
    let update = work
        .evaluate(|point| {
            // This immutable borrow could not compile if work retained &mut bridge.
            assert_eq!(detached.current(), Some(&before));
            detached_points.push(point);
            quantized(point)
        })
        .expect("evaluate");
    assert_eq!(detached_points, direct_points);
    assert_eq!(detached.current(), Some(&before));
    assert_eq!(detached.commit_rebake(update).expect("commit"), report);
    assert_eq!(detached.current(), direct.current());
}

/// Same content/revision does not confer publication identity, including clones.
#[test]
fn foreign_and_cloned_bridges_reject_work_without_resident_changes() {
    let original = empty(2, 2);
    for mut other in [empty(2, 2), original.clone()] {
        let before = other.current().expect("current").clone();
        let update = original
            .prepare_rebake(region([0; 3], [0; 3], 0), 8, LIMITS)
            .expect("prepare")
            .evaluate(quantized)
            .expect("evaluate");
        assert_eq!(
            other.commit_rebake(update),
            Err(SdfRebakeError::StalePublication)
        );
        assert_eq!(other.current(), Some(&before));
    }
}

/// Retained identity tokens prevent eviction/reload ABA even at identical revision.
#[test]
fn eviction_reload_and_newer_publication_invalidate_old_work() {
    for reload in [false, true] {
        let mut bridge = empty(2, 2);
        let update = bridge
            .prepare_rebake(region([0; 3], [0; 3], 0), 8, LIMITS)
            .expect("prepare")
            .evaluate(quantized)
            .expect("evaluate");
        let source = bridge.evict().expect("resident").into_source();
        if reload {
            bridge.publish(source).expect("same revision reload");
        }
        let before = bridge.current().cloned();
        let expected = if reload {
            SdfRebakeError::StalePublication
        } else {
            SdfRebakeError::NoSnapshot
        };
        assert_eq!(bridge.commit_rebake(update), Err(expected));
        assert_eq!(bridge.current(), before.as_ref());
    }
    let mut bridge = empty(2, 2);
    let work = bridge
        .prepare_rebake(region([0; 3], [0; 3], 0), 100, LIMITS)
        .expect("prepare");
    bridge
        .publish(WorldSdfSnapshot::new(
            WorldSdfTransform::new([7; 3], 2, 3).expect("transform"),
            8,
            SparseSdfBrickGrid::new(3, 1).expect("grid"),
        ))
        .expect("new shape publication");
    let before = bridge.current().expect("current").clone();
    assert_eq!(
        bridge.commit_rebake(work.evaluate(quantized).expect("evaluate")),
        Err(SdfRebakeError::StalePublication)
    );
    assert_eq!(bridge.current(), Some(&before));
}

/// Concurrent edits, including disjoint regions, must reprepare after first commit.
#[test]
fn first_commit_invalidates_all_other_works_at_that_base() {
    let mut bridge = empty(2, 2);
    let early = bridge
        .prepare_rebake(region([0; 3], [0; 3], 0), 8, LIMITS)
        .expect("early");
    let late = bridge
        .prepare_rebake(region([8; 3], [8; 3], 0), 9, LIMITS)
        .expect("late");
    bridge
        .commit_rebake(early.evaluate(quantized).expect("evaluate"))
        .expect("first commit");
    let before = bridge.current().expect("current").clone();
    assert_eq!(
        bridge.commit_rebake(late.evaluate(quantized).expect("evaluate")),
        Err(SdfRebakeError::StalePublication)
    );
    assert_eq!(bridge.current(), Some(&before));
}

/// Failed publication and abandoned/failed evaluation do not revoke valid work.
#[test]
fn failed_and_abandoned_work_leave_prepared_publication_valid() {
    let mut bridge = empty(2, 2);
    let edit = region([0; 3], [0; 3], 0);
    let work = bridge.prepare_rebake(edit, 8, LIMITS).expect("work");
    let before = bridge.current().expect("current").clone();
    assert!(bridge.publish(before.source().clone()).is_err());
    drop(
        bridge
            .prepare_rebake(edit, 9, LIMITS)
            .expect("abandoned work"),
    );
    drop(
        bridge
            .prepare_rebake(edit, 9, LIMITS)
            .expect("work")
            .evaluate(quantized)
            .expect("abandoned update"),
    );
    let failed = bridge
        .prepare_rebake(edit, 9, LIMITS)
        .expect("work")
        .evaluate(|_| Err::<i16, _>("source"));
    assert!(matches!(
        failed,
        Err(SdfRebakeError::Source {
            error: "source",
            ..
        })
    ));
    let panicked = bridge.prepare_rebake(edit, 9, LIMITS).expect("work");
    assert!(catch_unwind(AssertUnwindSafe(|| panicked.evaluate(
        |_| -> Result<i16, Infallible> {
            panic!("detached source unwind");
        }
    )))
    .is_err());
    assert_eq!(bridge.current(), Some(&before));
    bridge
        .commit_rebake(work.evaluate(quantized).expect("evaluate"))
        .expect("still valid");
    assert_eq!(bridge.current().expect("current").revision(), 8);
}

// Enumerate edited voxels individually, then derive ancestors independently
// from voxel scale. This intentionally does not use brick rectangle propagation.
fn expected_keys(edit: SdfRebakeRegion, edge: i64, mips: u8) -> BTreeSet<BrickCoord> {
    let mut keys = BTreeSet::new();
    for z in edit.min_voxel[2]..=edit.max_voxel[2] {
        for y in edit.min_voxel[1]..=edit.max_voxel[1] {
            for x in edit.min_voxel[0]..=edit.max_voxel[0] {
                for mip in edit.mip..mips {
                    let scale = edge * (1_i64 << (mip - edit.mip));
                    keys.insert(BrickCoord::new(
                        i32::try_from(i64::from(x).div_euclid(scale)).expect("x"),
                        i32::try_from(i64::from(y).div_euclid(scale)).expect("y"),
                        i32::try_from(i64::from(z).div_euclid(scale)).expect("z"),
                        mip,
                    ));
                }
            }
        }
    }
    keys
}

/// Non-power-of-two bricks, negative boundaries and three mips preserve exact units.
#[test]
fn local_edit_matches_independent_coverage_and_physical_field() {
    let origin = [i64::MAX - 100, i64::MIN + 100, -700];
    let transform = WorldSdfTransform::new(origin, 10, 2).expect("transform");
    let mut bridge = make_bridge(SparseSdfBrickGrid::new(3, 3).expect("grid"), transform);
    let edit = region([-4, -1, 2], [0, 2, 3], 0);
    let keys = expected_keys(edit, 3, 3);
    let mut calls = Vec::new();
    let report = bridge
        .rebake_region(edit, 8, LIMITS, |point| {
            assert_eq!(point.voxel_size_mm, 10_i128 << point.brick.mip);
            assert_eq!(point.distance_scale_mm, 2);
            for (axis, origin_mm) in origin.iter().enumerate() {
                assert_eq!(
                    point.world_position_mm[axis],
                    i128::from(*origin_mm) + i128::from(point.voxel[axis]) * point.voxel_size_mm
                );
            }
            calls.push(point);
            quantized(point)
        })
        .expect("rebake");
    let current = bridge.current().expect("current");
    assert_eq!(report.rebaked_bricks, keys.len());
    assert_eq!(report.inserted_bricks, keys.len());
    assert_eq!(report.sampled_voxels, keys.len() * 27);
    assert_eq!((report.previous_revision, report.revision), (7, 8));
    assert_eq!(
        current.grid().allocated_bricks(),
        keys.into_iter().collect::<Vec<_>>()
    );
    assert_eq!(current.grid().dirty_count(), 0);
    assert_eq!(report.logical_bytes, 48 + report.rebaked_bricks * (16 + 54));
    assert_eq!(current.logical_bytes(), report.logical_bytes);
    for chunk in calls.chunks_exact(27) {
        let key = chunk[0].brick;
        let samples = current
            .grid()
            .brick_samples(key)
            .expect("key")
            .expect("resident");
        for (index, point) in chunk.iter().enumerate() {
            assert_eq!(point.brick, key);
            assert_eq!(
                point.voxel,
                [
                    i64::from(key.x) * 3 + (index % 3) as i64,
                    i64::from(key.y) * 3 + ((index / 3) % 3) as i64,
                    i64::from(key.z) * 3 + (index / 9) as i64
                ]
            );
            assert_eq!(samples[index], quantized(*point).expect("value"));
        }
    }
    assert!(calls.windows(2).all(|pair| {
        let a = pair[0].brick;
        let b = pair[1].brick;
        (a.mip, a.z, a.y, a.x) <= (b.mip, b.z, b.y, b.x)
    }));
}

/// Every small box has the same parent coverage as independently enumerated voxels.
#[test]
fn many_regions_match_voxel_oracle_without_duplicate_samples() {
    for edge in [1, 2, 3, 8] {
        for seed in 0..24 {
            let min = [seed % 7 - 4, (seed * 3) % 9 - 5, (seed * 7) % 11 - 6];
            let max = [min[0] + seed % 3, min[1] + seed % 2, min[2] + seed % 4];
            let edit = region(min, max, (seed % 2) as u8);
            let keys = expected_keys(edit, i64::from(edge), 4);
            let mut bridge = empty(edge, 4);
            let mut points = BTreeSet::new();
            let report = bridge
                .rebake_region(edit, 8, LIMITS, |p| {
                    assert!(points.insert((p.brick, p.voxel)));
                    quantized(p)
                })
                .expect("rebake");
            assert_eq!(report.rebaked_bricks, keys.len());
            assert_eq!(report.sampled_voxels, points.len());
            assert_eq!(
                bridge.current().expect("current").grid().allocated_bricks(),
                keys.into_iter().collect::<Vec<_>>()
            );
        }
    }
}

/// Replacements preserve unaffected allocations and make continuous queries see the new revision.
#[test]
fn local_publication_retains_unaffected_storage_and_updates_queries() {
    let mut grid = SparseSdfBrickGrid::new(2, 2).expect("grid");
    let untouched = BrickCoord::new(1000, 0, 0, 0);
    grid.insert_brick(untouched, vec![123; 8]).expect("brick");
    let mut bridge = make_bridge(
        grid,
        WorldSdfTransform::new([0; 3], 10, 2).expect("transform"),
    );
    let pointer = bridge
        .current()
        .expect("current")
        .grid()
        .brick_samples(untouched)
        .expect("key")
        .expect("resident")
        .as_ptr();
    let report = bridge
        .rebake_region(region([0; 3], [0; 3], 0), 8, LIMITS, quantized)
        .expect("rebake");
    assert_eq!((report.rebaked_bricks, report.inserted_bricks), (2, 2));
    let current = bridge.current().expect("current");
    assert_eq!(
        current
            .grid()
            .brick_samples(untouched)
            .expect("key")
            .expect("resident")
            .as_ptr(),
        pointer
    );
    let SdfQueryResult::Value(sample) = current
        .query_world_distance_gradient_mm([5; 3], 0)
        .expect("query")
    else {
        panic!("resident field")
    };
    assert_eq!(sample.distance_mm, 0.0);
    for (actual, expected) in sample.gradient.into_iter().zip([0.2, 0.4, -0.6]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    let second = bridge
        .rebake_region(region([0; 3], [0; 3], 0), 9, LIMITS, |_| {
            Ok::<_, Infallible>(i16::MIN)
        })
        .expect("replace");
    assert_eq!(second.inserted_bricks, 0);
    assert_eq!(second.logical_bytes, report.logical_bytes);
    assert_eq!(
        bridge
            .current()
            .expect("current")
            .query_world_distance_mm([0; 3], 0),
        Ok(SdfQueryResult::Value(-65536))
    );
}

/// Failure at every callback ordinal, including the final sample, cannot publish a prefix.
#[test]
fn all_source_failure_positions_preserve_old_snapshot_and_allow_retry() {
    let edit = region([-1; 3], [0; 3], 0);
    let total = expected_keys(edit, 2, 2).len() * 8;
    let mut bridge = empty(2, 2);
    bridge
        .rebake_region(edit, 8, LIMITS, quantized)
        .expect("initial");
    let old = bridge.current().expect("current").clone();
    for failure in 0..total {
        let mut count = 0;
        let error = bridge
            .rebake_region(edit, 9, LIMITS, |p| {
                let index = count;
                count += 1;
                if index == failure {
                    Err((p.brick, p.voxel))
                } else {
                    Ok(777)
                }
            })
            .expect_err("injected failure");
        let SdfRebakeError::Source {
            brick,
            voxel,
            error,
        } = error
        else {
            panic!("source error")
        };
        assert_eq!(error, (brick, voxel));
        assert_eq!(count, failure + 1);
        assert_eq!(bridge.current(), Some(&old));
    }
    bridge
        .rebake_region(edit, 9, LIMITS, |_| Ok::<_, Infallible>(i16::MAX))
        .expect("retry");
    assert_eq!(bridge.current().expect("current").revision(), 9);
}

/// Unwinding drops staged samples; publication remains unchanged and reusable.
#[test]
fn evaluator_panic_preserves_publication() {
    let mut bridge = empty(2, 2);
    let old = bridge.current().expect("current").clone();
    let mut calls = 0;
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _result = bridge.rebake_region(region([0; 3], [0; 3], 0), 8, LIMITS, |_| {
            calls += 1;
            assert!(calls < 16, "injected last-sample panic");
            Ok::<_, Infallible>(1)
        });
    }))
    .is_err());
    assert_eq!(calls, 16);
    assert_eq!(bridge.current(), Some(&old));
    bridge
        .rebake_region(region([0; 3], [0; 3], 0), 8, LIMITS, quantized)
        .expect("retry");
}

/// Each operation resource admits its exact boundary and rejects one below before callbacks.
#[test]
fn all_operation_limits_are_independent_and_preflighted() {
    let edit = region([-1; 3], [0; 3], 0);
    let report = empty(2, 2)
        .rebake_region(edit, 8, LIMITS, quantized)
        .expect("baseline");
    let exact = SdfRebakeLimits {
        max_bricks: report.rebaked_bricks,
        max_samples: report.sampled_voxels,
        max_scratch_bytes: report.scratch_bytes,
    };
    empty(2, 2)
        .rebake_region(edit, 8, exact, quantized)
        .expect("exact boundary");
    for resource in [
        SdfRebakeResource::Bricks,
        SdfRebakeResource::Samples,
        SdfRebakeResource::ScratchBytes,
    ] {
        let mut limits = exact;
        let required = match resource {
            SdfRebakeResource::Bricks => {
                limits.max_bricks -= 1;
                exact.max_bricks
            }
            SdfRebakeResource::Samples => {
                limits.max_samples -= 1;
                exact.max_samples
            }
            SdfRebakeResource::ScratchBytes => {
                limits.max_scratch_bytes -= 1;
                exact.max_scratch_bytes
            }
        };
        let mut bridge = empty(2, 2);
        let old = bridge.current().expect("current").clone();
        let error = bridge
            .rebake_region(edit, 8, limits, |_| -> Result<i16, Infallible> {
                panic!("preflight must reject")
            })
            .expect_err("over budget");
        assert_eq!(
            error,
            SdfRebakeError::BudgetExceeded {
                resource,
                required,
                limit: required - 1
            }
        );
        assert_eq!(bridge.current(), Some(&old));
    }
}

/// Admission checks final resident usage, including newly introduced coarser parents.
#[test]
fn every_resident_limit_rejects_before_sampling() {
    let cases = [
        (
            SdfPhysicsBridgeBudget::new(1, 16, 112).expect("budget"),
            SdfBridgeError::BrickCapacityExceeded {
                required: 2,
                capacity: 1,
            },
        ),
        (
            SdfPhysicsBridgeBudget::new(2, 15, 112).expect("budget"),
            SdfBridgeError::SampleCapacityExceeded {
                required: 16,
                capacity: 15,
            },
        ),
        (
            SdfPhysicsBridgeBudget::new(2, 16, 111).expect("budget"),
            SdfBridgeError::UploadBudgetExceeded {
                required: 112,
                limit: 111,
            },
        ),
    ];
    for (budget, expected) in cases {
        let mut bridge = WorldSdfPhysicsBridge::new(budget);
        bridge
            .publish(WorldSdfSnapshot::new(
                WorldSdfTransform::new([0; 3], 1, 1).expect("transform"),
                7,
                SparseSdfBrickGrid::new(2, 2).expect("grid"),
            ))
            .expect("publish");
        let old = bridge.current().expect("current").clone();
        let error = bridge
            .rebake_region(
                region([0; 3], [0; 3], 0),
                8,
                LIMITS,
                |_| -> Result<i16, Infallible> { panic!("resident preflight") },
            )
            .expect_err("budget");
        assert_eq!(error, SdfRebakeError::Bridge(expected));
        assert_eq!(bridge.current(), Some(&old));
    }
}

/// Invalid inputs, stale revisions and unowned dirty state cannot call the source.
#[test]
fn identity_and_schema_errors_are_atomic() {
    let mut bridge = empty(2, 2);
    let good = region([0; 3], [0; 3], 0);
    let old = bridge.current().expect("current").clone();
    for (edit, revision, expected) in [
        (
            good,
            7,
            SdfRebakeError::Bridge(SdfBridgeError::StaleRevision {
                current: 7,
                incoming: 7,
            }),
        ),
        (
            good,
            6,
            SdfRebakeError::Bridge(SdfBridgeError::StaleRevision {
                current: 7,
                incoming: 6,
            }),
        ),
        (
            region([0; 3], [0; 3], 2),
            8,
            SdfRebakeError::Bridge(SdfBridgeError::GridQuery(SdfError::InvalidMip)),
        ),
        (
            region([1, 0, 0], [0; 3], 0),
            8,
            SdfRebakeError::InvalidRegion,
        ),
        (
            region([0, 1, 0], [0; 3], 0),
            8,
            SdfRebakeError::InvalidRegion,
        ),
        (
            region([0, 0, 1], [0; 3], 0),
            8,
            SdfRebakeError::InvalidRegion,
        ),
    ] {
        assert_eq!(
            bridge.rebake_region(edit, revision, LIMITS, |_| -> Result<i16, Infallible> {
                panic!("invalid input")
            }),
            Err(expected)
        );
        assert_eq!(bridge.current(), Some(&old));
    }
    bridge.evict();
    assert_eq!(
        bridge.rebake_region(good, 8, LIMITS, quantized),
        Err(SdfRebakeError::NoSnapshot)
    );
    let mut grid = SparseSdfBrickGrid::new(2, 2).expect("grid");
    grid.mark_dirty_region([0; 3], [0; 3], 0).expect("dirty");
    bridge
        .publish(WorldSdfSnapshot::new(
            WorldSdfTransform::new([0; 3], 1, 1).expect("transform"),
            7,
            grid,
        ))
        .expect("existing bridge admits dirty state");
    let old = bridge.current().expect("current").clone();
    assert_eq!(
        bridge.rebake_region(good, 8, LIMITS, quantized),
        Err(SdfRebakeError::DirtySnapshot)
    );
    assert_eq!(bridge.current(), Some(&old));
}

/// Large regions reject from rectangular arithmetic without enumerating trillions of bricks.
#[test]
fn huge_edit_rejects_before_enumeration_or_sampling() {
    let mut bridge = empty(1, 1);
    let edit = region([i32::MIN; 3], [i32::MAX; 3], 0);
    assert_eq!(
        bridge.rebake_region(edit, 8, LIMITS, quantized),
        Err(SdfRebakeError::Bridge(SdfBridgeError::ArithmeticOverflow))
    );
    let edit = region([i32::MIN, 0, 0], [i32::MAX, 0, 0], 0);
    let error = bridge
        .rebake_region(edit, 8, LIMITS, |_| -> Result<i16, Infallible> {
            panic!("unbounded callback")
        })
        .expect_err("bounded");
    assert!(matches!(
        error,
        SdfRebakeError::BudgetExceeded {
            resource: SdfRebakeResource::Bricks,
            ..
        }
    ));
}

/// Full edge coverage beyond the i32 voxel limit is represented exactly with i64 samples.
#[test]
fn extreme_voxels_and_origins_preserve_integer_sample_positions() {
    let origin = [i64::MAX, i64::MIN, 0];
    let transform = WorldSdfTransform::new(origin, u32::MAX, u32::MAX).expect("transform");
    let mut bridge = make_bridge(SparseSdfBrickGrid::new(3, 1).expect("grid"), transform);
    let mut voxels = Vec::new();
    bridge
        .rebake_region(region([i32::MIN; 3], [i32::MIN; 3], 0), 8, LIMITS, |p| {
            for (axis, origin_mm) in origin.iter().enumerate() {
                assert_eq!(
                    p.world_position_mm[axis],
                    i128::from(*origin_mm) + i128::from(p.voxel[axis]) * i128::from(u32::MAX)
                );
            }
            voxels.push(p.voxel);
            Ok::<_, Infallible>(i16::MIN)
        })
        .expect("extreme valid");
    assert!(voxels.iter().flatten().any(|v| *v < i64::from(i32::MIN)));
    assert_eq!(voxels.len(), 27);
}

/// An origin offset can make the final coordinate valid despite product overflow.
#[test]
fn final_affine_position_accepts_origin_cancellation_at_i128_max() {
    let mut bridge = make_bridge(
        SparseSdfBrickGrid::new(3, 127).expect("grid"),
        WorldSdfTransform::new([-1; 3], 1, 1).expect("transform"),
    );
    let mut calls = 0;
    let report = bridge
        .rebake_region(region([0; 3], [0; 3], 126), 8, LIMITS, |point| {
            let expected = point.voxel.map(|voxel| match voxel {
                0 => -1,
                1 => (1_i128 << 126) - 1,
                2 => i128::MAX,
                _ => unreachable!("one brick"),
            });
            assert_eq!(point.world_position_mm, expected);
            calls += 1;
            Ok::<_, Infallible>(17)
        })
        .expect("final positions fit");
    assert_eq!((calls, report.rebaked_bricks), (27, 1));
    assert_eq!(bridge.current().expect("snapshot").revision(), 8);

    // The negative endpoint itself fits on the signed fast path, including a
    // positive origin. With the public power-of-two mip spacing, an actual
    // negative product overflow is too large for an i64 origin to rescue.
    for origin in [0, 1, i64::MAX] {
        let mut negative = make_bridge(
            SparseSdfBrickGrid::new(1, 127).expect("grid"),
            WorldSdfTransform::new([origin; 3], 1, 1).expect("transform"),
        );
        negative
            .rebake_region(region([-2; 3], [-2; 3], 126), 8, LIMITS, |point| {
                assert_eq!(point.world_position_mm, [i128::MIN + i128::from(origin); 3]);
                Ok::<_, Infallible>(i16::MIN)
            })
            .expect("negative final position fits");
    }
}

/// All coarser mip extrema are checked before even the first valid fine sample.
#[test]
fn mip_and_world_coordinate_overflows_preflight_the_entire_plan() {
    let mut bridge = empty(1, 128);
    assert_eq!(
        bridge.rebake_region(region([0; 3], [0; 3], 0), 8, LIMITS, quantized),
        Err(SdfRebakeError::Bridge(SdfBridgeError::MipScaleOverflow))
    );
    let mut bridge = make_bridge(
        SparseSdfBrickGrid::new(3, 127).expect("grid"),
        WorldSdfTransform::new([0; 3], 1, 1).expect("transform"),
    );
    // At mip 126 the negative brick's first voxel is -3 * 2^126,
    // outside i128, although its last voxel -1 * 2^126 would fit.
    assert_eq!(
        bridge.rebake_region(
            region([-1; 3], [-1; 3], 126),
            8,
            LIMITS,
            |_| -> Result<i16, Infallible> { panic!("mip extent overflow") }
        ),
        Err(SdfRebakeError::Bridge(SdfBridgeError::ArithmeticOverflow))
    );
    let mut positive = make_bridge(
        SparseSdfBrickGrid::new(3, 127).expect("grid"),
        WorldSdfTransform::new([0; 3], 1, 1).expect("transform"),
    );
    // Fine mip positions fit; a coarser parent's full-brick maximum does not.
    assert_eq!(
        positive.rebake_region(
            region([0; 3], [0; 3], 0),
            8,
            LIMITS,
            |_| -> Result<i16, Infallible> { panic!("late parent maximum overflow") }
        ),
        Err(SdfRebakeError::Bridge(SdfBridgeError::ArithmeticOverflow))
    );
}

/// Both supported brick edge extrema work at exact resident capacity, with no revision wrap.
#[test]
fn edge_extrema_exact_resident_capacity_and_last_revision() {
    for edge in [1, 32] {
        let volume = usize::from(edge).pow(3);
        let mut bridge = WorldSdfPhysicsBridge::new(
            SdfPhysicsBridgeBudget::new(2, 2 * volume, 80 + 4 * volume).expect("budget"),
        );
        bridge
            .publish(WorldSdfSnapshot::new(
                WorldSdfTransform::new([0; 3], 1, 1).expect("transform"),
                u64::MAX - 2,
                SparseSdfBrickGrid::new(edge, 2).expect("grid"),
            ))
            .expect("publish");
        let first = bridge
            .rebake_region(region([0; 3], [0; 3], 0), u64::MAX - 1, LIMITS, |_| {
                Ok::<_, Infallible>(i16::MIN)
            })
            .expect("exact capacity insertion");
        assert_eq!(first.inserted_bricks, 2);
        assert_eq!(first.logical_bytes, 80 + 4 * volume);
        let second = bridge
            .rebake_region(region([0; 3], [0; 3], 0), u64::MAX, LIMITS, |_| {
                Ok::<_, Infallible>(i16::MAX)
            })
            .expect("full capacity replacement");
        assert_eq!(second.inserted_bricks, 0);
        let before = bridge.current().expect("current").clone();
        assert_eq!(
            bridge.rebake_region(region([0; 3], [0; 3], 0), 0, LIMITS, quantized),
            Err(SdfRebakeError::Bridge(SdfBridgeError::StaleRevision {
                current: u64::MAX,
                incoming: 0
            }))
        );
        assert_eq!(bridge.current(), Some(&before));
    }
}
