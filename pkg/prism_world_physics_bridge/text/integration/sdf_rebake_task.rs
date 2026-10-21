//! Real worker execution, cancellation checkpoints, and atomic publication.

extern crate alloc;

use core::convert::Infallible;
use prism_tasks::{CancelToken, TaskPool};
use prism_world_physics_bridge::{
    SdfPhysicsBridgeBudget, SdfQueryResult, SdfRebakeError, SdfRebakeLimits, SdfRebakePoint,
    SdfRebakeRegion, SdfRebakeTask, SdfRebakeTaskError, WorldSdfPhysicsBridge, WorldSdfSnapshot,
    WorldSdfTransform,
};
use prism_world_residency::{BrickCoord, SparseSdfBrickGrid};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, Instant},
};

const TIMEOUT: Duration = Duration::from_secs(10);
const EDIT: SdfRebakeRegion = SdfRebakeRegion {
    min_voxel: [0; 3],
    max_voxel: [0; 3],
    mip: 0,
};
const LIMITS: SdfRebakeLimits = SdfRebakeLimits {
    max_bricks: 2,
    max_samples: 16,
    max_scratch_bytes: 4096,
};

fn bridge() -> WorldSdfPhysicsBridge {
    let mut grid = SparseSdfBrickGrid::new(2, 2).expect("grid");
    grid.insert_brick(BrickCoord::new(0, 0, 0, 0), alloc::vec![3; 8])
        .expect("brick");
    let mut bridge =
        WorldSdfPhysicsBridge::new(SdfPhysicsBridgeBudget::new(2, 16, 112).expect("budget"));
    bridge
        .publish(WorldSdfSnapshot::new(
            WorldSdfTransform::new([0; 3], 10, 2).expect("transform"),
            1,
            grid,
        ))
        .expect("publish");
    bridge
}

fn sample(point: SdfRebakePoint) -> Result<i16, Infallible> {
    Ok(i16::try_from(point.voxel.iter().sum::<i64>() + 7).expect("small field"))
}

fn finished<E>(task: &SdfRebakeTask<'_, E>) {
    let deadline = Instant::now() + TIMEOUT;
    while !task.is_finished() {
        assert!(Instant::now() < deadline, "worker did not finish");
        thread::yield_now();
    }
}

/// The owner polls while a real worker is blocked in the source; no help-on-wait.
#[test]
fn real_worker_keeps_old_queries_available_until_explicit_commit() {
    let pool = TaskPool::with_threads(1);
    if pool.worker_count() == 0 {
        // The forced-single matrix tests the inline path in the dedicated case.
        assert!(pool.is_single_threaded());
        return;
    }
    let mut bridge = bridge();
    let old = bridge.current().expect("current").clone();
    let work = bridge.prepare_rebake(EDIT, 2, LIMITS).expect("prepare");
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let worker_pool = pool.clone();
    let mut first = true;
    let mut task = work.spawn(&pool, &CancelToken::new(), move |point| {
        if first {
            first = false;
            started_tx
                .send((thread::current().id(), worker_pool.current_worker_index()))
                .expect("started");
            release_rx
                .recv_timeout(TIMEOUT)
                .expect("owner releases worker");
        }
        sample(point)
    });
    let (worker_thread, worker_index) = started_rx.recv_timeout(TIMEOUT).expect("started");
    assert_ne!(worker_thread, thread::current().id());
    assert!(worker_index.is_some());
    assert!(!task.is_finished());
    assert_eq!(task.try_commit(&mut bridge).expect("pending"), None);
    assert_eq!(task.try_discard().expect("pending discard"), None);
    assert_eq!(bridge.current(), Some(&old));
    assert_eq!(
        bridge
            .current()
            .expect("current")
            .query_world_distance_mm([0; 3], 0)
            .expect("old query"),
        SdfQueryResult::Value(6)
    );
    release_tx.send(()).expect("release");
    finished(&task);
    assert_eq!(bridge.current(), Some(&old));
    let report = task
        .try_commit(&mut bridge)
        .expect("commit")
        .expect("ready");
    assert_eq!((report.previous_revision, report.revision), (1, 2));
    assert_eq!(report.sampled_voxels, 16);
    assert_eq!(
        bridge
            .current()
            .expect("current")
            .query_world_distance_mm([0; 3], 0)
            .expect("new query"),
        SdfQueryResult::Value(14)
    );
    assert!(matches!(
        task.try_commit(&mut bridge),
        Err(SdfRebakeTaskError::AlreadyCollected)
    ));
}

/// Discard frees terminal results once while preserving source/panic diagnostics.
#[test]
fn terminal_discard_never_publishes_and_preserves_failures() {
    let pool = TaskPool::with_threads(0);
    let mut bridge = bridge();
    let before = bridge.current().expect("current").clone();
    let mut success = bridge.prepare_rebake(EDIT, 2, LIMITS).expect("work").spawn(
        &pool,
        &CancelToken::new(),
        sample,
    );
    assert_eq!(success.try_discard().expect("discard"), Some(()));
    assert!(matches!(
        success.try_discard(),
        Err(SdfRebakeTaskError::AlreadyCollected)
    ));
    assert!(matches!(
        success.try_commit(&mut bridge),
        Err(SdfRebakeTaskError::AlreadyCollected)
    ));
    let mut failure = bridge.prepare_rebake(EDIT, 2, LIMITS).expect("work").spawn(
        &pool,
        &CancelToken::new(),
        |_| Err::<i16, _>("source failure"),
    );
    assert!(matches!(
        failure.try_discard(),
        Err(SdfRebakeTaskError::Rebake(SdfRebakeError::Source {
            error: "source failure",
            ..
        }))
    ));
    let mut panicked = bridge.prepare_rebake(EDIT, 2, LIMITS).expect("work").spawn(
        &pool,
        &CancelToken::new(),
        |_| -> Result<i16, Infallible> {
            std::panic::panic_any(37_u32);
        },
    );
    match panicked.try_discard() {
        Err(SdfRebakeTaskError::Panicked(payload)) => {
            assert_eq!(payload.downcast_ref::<u32>(), Some(&37));
        }
        other => panic!("expected original panic payload, got {other:?}"),
    }
    let mut cancelled = bridge.prepare_rebake(EDIT, 2, LIMITS).expect("work").spawn(
        &pool,
        &CancelToken::new(),
        sample,
    );
    cancelled.cancel();
    assert!(matches!(
        cancelled.try_discard(),
        Err(SdfRebakeTaskError::Rebake(SdfRebakeError::Cancelled))
    ));
    assert_eq!(bridge.current(), Some(&before));
}

/// Collecting a completed result cannot join a different job on the same pool.
#[test]
fn completed_result_collection_does_not_wait_for_unrelated_pool_work() {
    let pool = TaskPool::with_threads(1);
    if pool.worker_count() == 0 {
        assert!(pool.is_single_threaded());
        return;
    }
    let mut bridge = bridge();
    let mut task = bridge
        .prepare_rebake(EDIT, 2, LIMITS)
        .expect("prepare")
        .spawn(&pool, &CancelToken::new(), sample);
    finished(&task);
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let unrelated = pool.spawn_catch(move || {
        started_tx.send(()).expect("started");
        release_rx.recv_timeout(TIMEOUT).expect("release");
    });
    started_rx.recv_timeout(TIMEOUT).expect("unrelated running");
    assert!(!unrelated.is_finished());
    assert_eq!(
        task.try_commit(&mut bridge)
            .expect("commit while worker busy")
            .expect("ready")
            .revision,
        2
    );
    // The blocked job cannot complete until collection above has returned.
    release_tx.send(()).expect("release after commit returned");
    unrelated.join_catch().expect("unrelated completed");
}

/// A zero-worker pool executes inline and still requires explicit publication.
#[test]
fn zero_worker_fallback_has_the_same_output_and_explicit_commit() {
    let pool = TaskPool::with_threads(0);
    let mut actual = bridge();
    let old = actual.current().expect("current").clone();
    let mut expected = actual.clone();
    let report = expected
        .rebake_region(EDIT, 2, LIMITS, sample)
        .expect("sync");
    let calls = Arc::new(AtomicUsize::new(0));
    let worker_calls = Arc::clone(&calls);
    let mut task = actual
        .prepare_rebake(EDIT, 2, LIMITS)
        .expect("prepare")
        .spawn(&pool, &CancelToken::new(), move |point| {
            worker_calls.fetch_add(1, Ordering::Relaxed);
            sample(point)
        });
    assert_eq!(calls.load(Ordering::Relaxed), 16);
    assert!(task.is_finished());
    assert_eq!(actual.current(), Some(&old));
    assert_eq!(task.try_commit(&mut actual).expect("commit"), Some(report));
    assert_eq!(actual.current(), expected.current());
}

/// Pre-cancel, intermediate cancel, and cancel in the final sample all fail closed.
#[test]
fn cancellation_checks_cover_every_sampling_boundary() {
    for workers in [0, 1] {
        let pool = TaskPool::with_threads(workers);
        for cancel_at in [0, 1, 16] {
            let mut bridge = bridge();
            let before = bridge.current().expect("current").clone();
            let scope = CancelToken::new();
            if cancel_at == 0 {
                scope.cancel();
            }
            let worker_scope = scope.clone();
            let calls = Arc::new(AtomicUsize::new(0));
            let worker_calls = Arc::clone(&calls);
            let mut task = bridge
                .prepare_rebake(EDIT, 2, LIMITS)
                .expect("prepare")
                .spawn(&pool, &scope, move |point| {
                    if worker_calls.fetch_add(1, Ordering::Relaxed) + 1 == cancel_at {
                        worker_scope.cancel();
                    }
                    sample(point)
                });
            finished(&task);
            assert!(matches!(
                task.try_commit(&mut bridge),
                Err(SdfRebakeTaskError::Rebake(SdfRebakeError::Cancelled))
            ));
            assert_eq!(calls.load(Ordering::Relaxed), cancel_at);
            assert_eq!(bridge.current(), Some(&before));
        }
    }
}

/// Completed output cannot sneak through cancellation; sibling scopes stay live.
#[test]
fn cancellation_after_completion_suppresses_commit_without_cancelling_siblings() {
    let pool = TaskPool::with_threads(1);
    let scope = CancelToken::new();
    let mut bridge = bridge();
    let before = bridge.current().expect("current").clone();
    let mut task = bridge
        .prepare_rebake(EDIT, 2, LIMITS)
        .expect("prepare")
        .spawn(&pool, &scope, sample);
    finished(&task);
    task.cancel();
    assert!(!scope.is_cancelled());
    assert!(matches!(
        task.try_commit(&mut bridge),
        Err(SdfRebakeTaskError::Rebake(SdfRebakeError::Cancelled))
    ));
    assert_eq!(bridge.current(), Some(&before));
    let mut sibling = bridge
        .prepare_rebake(EDIT, 2, LIMITS)
        .expect("prepare")
        .spawn(&pool, &scope, sample);
    finished(&sibling);
    sibling
        .try_commit(&mut bridge)
        .expect("sibling commit")
        .expect("ready");
}

/// Parent cancellation after worker completion also prevents publication.
#[test]
fn parent_scope_can_cancel_a_finished_uncollected_result() {
    let pool = TaskPool::with_threads(0);
    let scope = CancelToken::new();
    let mut bridge = bridge();
    let old = bridge.current().expect("current").clone();
    let mut task = bridge
        .prepare_rebake(EDIT, 2, LIMITS)
        .expect("prepare")
        .spawn(&pool, &scope, sample);
    assert!(task.is_finished());
    scope.cancel();
    assert!(matches!(
        task.try_commit(&mut bridge),
        Err(SdfRebakeTaskError::Rebake(SdfRebakeError::Cancelled))
    ));
    assert_eq!(bridge.current(), Some(&old));
}

/// Captured source errors and panics retain old data and do not poison the pool.
#[test]
fn source_failure_and_panic_are_observable_and_pool_remains_usable() {
    for workers in [0, 1] {
        let pool = TaskPool::with_threads(workers);
        let mut bridge = bridge();
        let old = bridge.current().expect("current").clone();
        let scope = CancelToken::new();
        let mut failed = bridge
            .prepare_rebake(EDIT, 2, LIMITS)
            .expect("prepare")
            .spawn(&pool, &scope, |_| Err::<i16, _>("field unavailable"));
        finished(&failed);
        assert!(matches!(
            failed.try_commit(&mut bridge),
            Err(SdfRebakeTaskError::Rebake(SdfRebakeError::Source {
                error: "field unavailable",
                brick: BrickCoord {
                    x: 0,
                    y: 0,
                    z: 0,
                    mip: 0
                },
                voxel: [0, 0, 0],
            }))
        ));
        assert_eq!(bridge.current(), Some(&old));
        let mut panicked = bridge
            .prepare_rebake(EDIT, 2, LIMITS)
            .expect("prepare")
            .spawn(&pool, &scope, |_| -> Result<i16, Infallible> {
                panic!("field panic")
            });
        finished(&panicked);
        let Err(SdfRebakeTaskError::Panicked(payload)) = panicked.try_commit(&mut bridge) else {
            panic!("expected captured panic")
        };
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"field panic"));
        assert_eq!(bridge.current(), Some(&old));
        let mut good = bridge
            .prepare_rebake(EDIT, 2, LIMITS)
            .expect("prepare")
            .spawn(&pool, &scope, sample);
        finished(&good);
        good.try_commit(&mut bridge)
            .expect("pool survives")
            .expect("ready");
    }
}

/// Replacement during evaluation is rejected at owner publication, not silently merged.
#[test]
fn completed_task_rejects_evicted_reloaded_or_concurrently_edited_base() {
    let pool = TaskPool::with_threads(0);
    for reload in [false, true] {
        let mut bridge = bridge();
        let mut task = bridge
            .prepare_rebake(EDIT, 3, LIMITS)
            .expect("prepare")
            .spawn(&pool, &CancelToken::new(), sample);
        if reload {
            let source = bridge.evict().expect("resident").into_source();
            bridge.publish(source).expect("same revision reload");
        } else {
            bridge
                .rebake_region(EDIT, 2, LIMITS, sample)
                .expect("intervening edit");
        }
        let before = bridge.current().expect("current").clone();
        assert!(matches!(
            task.try_commit(&mut bridge),
            Err(SdfRebakeTaskError::Rebake(SdfRebakeError::StalePublication))
        ));
        assert_eq!(bridge.current(), Some(&before));
    }
}

struct NotifyDrop(mpsc::SyncSender<()>);

impl Drop for NotifyDrop {
    fn drop(&mut self) {
        // Receiver may be gone if another assertion failed; notification has no
        // publication effect and must not produce a second panic during unwind.
        let _ = self.0.send(());
    }
}

/// Dropping a blocked task returns immediately; its next checkpoint stops sampling.
#[test]
fn dropping_task_requests_cancellation_without_joining_or_publishing() {
    let pool = TaskPool::with_threads(1);
    if pool.worker_count() == 0 {
        assert!(pool.is_single_threaded());
        return;
    }
    let bridge = bridge();
    let before = bridge.current().expect("current").clone();
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let (dropped_tx, dropped_rx) = mpsc::sync_channel(1);
    let notify = NotifyDrop(dropped_tx);
    let calls = Arc::new(AtomicUsize::new(0));
    let worker_calls = Arc::clone(&calls);
    let scope = CancelToken::new();
    let task = bridge
        .prepare_rebake(EDIT, 2, LIMITS)
        .expect("prepare")
        .spawn(&pool, &scope, move |point| {
            let _keep_notification_until_callback_drop = &notify;
            worker_calls.fetch_add(1, Ordering::Relaxed);
            started_tx.send(()).expect("started");
            release_rx.recv_timeout(TIMEOUT).expect("release");
            sample(point)
        });
    started_rx.recv_timeout(TIMEOUT).expect("started");
    drop(task);
    assert!(!scope.is_cancelled());
    release_tx.send(()).expect("release after drop returned");
    dropped_rx
        .recv_timeout(TIMEOUT)
        .expect("worker callback dropped");
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(bridge.current(), Some(&before));
}
