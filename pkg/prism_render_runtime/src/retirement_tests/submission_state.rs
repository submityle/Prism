//! Submission acknowledgement is separate from recording and delayed retirement.

use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[test]
fn runtime_marks_only_after_actual_submit_before_delayed_completion() {
    let (runtime, instance) = runtime(2);
    let inspector = instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    let mut graph = frame(&calls);
    let submission = graph.submission_token();
    let recording_token = submission.clone();
    graph.add_compute_pass("observe record", PassFlags::SIDE_EFFECT, move |_| {
        move |_| assert!(!recording_token.upgrade().unwrap().is_submitted())
    });
    let submit_calls = Arc::new(AtomicUsize::new(0));
    let native_calls = submit_calls.clone();
    let native_token = submission.clone();
    lock(&inspector.queue.0).before_submit = Some(Box::new(move || {
        assert!(!native_token.upgrade().unwrap().is_submitted());
        native_calls.fetch_add(1, Ordering::SeqCst);
    }));
    let prepared = graph.prepare().unwrap();
    assert!(!submission.upgrade().unwrap().is_submitted());
    let owner = runtime.execute_prepared_owned(prepared, &[]).unwrap();
    assert_eq!(submit_calls.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), FRAME_RECORDINGS);
    assert!(submission.upgrade().unwrap().is_submitted());
    assert!(!owner.is_complete());
    inspector.device.assert_counts(FRAME_ALLOCATIONS, 0);
    // Dropping the explicit owner leaves the completion callback owning the
    // submission guard. It must not keep prepare blocked until retirement.
    drop(owner);
    assert!(submission.upgrade().unwrap().is_submitted());
    inspector.queue.complete_through(1);
    assert!(submission.upgrade().is_none());
    inspector
        .device
        .assert_counts(FRAME_ALLOCATIONS, FRAME_ALLOCATIONS);
}

#[test]
fn pacing_and_value_rejections_drop_unsubmitted_graph_guard_without_recording() {
    let (runtime, instance) = runtime(1);
    let inspector = instance.device(0);
    let calls = Arc::new(AtomicUsize::new(0));
    let first = runtime.execute_owned(frame(&calls)).unwrap();
    inspector.queue.set_wait_results(false, false);
    for exhausted in [false, true] {
        if exhausted {
            runtime.submitted_value.store(u64::MAX, Ordering::Release);
        }
        let graph = frame(&calls);
        let submission = graph.submission_token();
        let observer = submission.upgrade().unwrap();
        let result = runtime.execute_prepared_owned(graph.prepare().unwrap(), &[]);
        if exhausted {
            assert!(matches!(
                result,
                Err(SubmissionError::CompletionValueExhausted)
            ));
        } else {
            assert!(matches!(
                result,
                Err(SubmissionError::CompletionUnconfirmed)
            ));
        }
        assert!(!observer.is_submitted());
        assert_eq!(submission.strong_count(), 1);
        drop(observer);
        assert!(submission.upgrade().is_none());
    }
    assert_eq!(calls.load(Ordering::SeqCst), FRAME_RECORDINGS);
    inspector.queue.assert_submissions(1);
    inspector.device.assert_counts(FRAME_ALLOCATIONS, 0);
    inspector.queue.complete_through(1);
    drop(first);
    inspector.queue.set_wait_results(true, true);
}

#[test]
fn compile_failure_drops_unsubmitted_runtime_graph_token() {
    let (runtime, instance) = runtime(1);
    let inspector = instance.device(0);
    let mut graph = runtime.new_graph();
    let submission = graph.submission_token();
    graph.add_compute_pass("overflow", PassFlags::SIDE_EFFECT, |b| {
        let overflow = b.create_buffer(
            "overflow",
            BufferDesc::new(u64::MAX, rhi::BufferUsages::STORAGE),
        );
        b.write_storage_buffer(overflow);
        |_| panic!("compile failure cannot record")
    });
    assert!(matches!(
        runtime.execute_owned(graph),
        Err(SubmissionError::Compile(CompileError::MemoryOverflow))
    ));
    assert!(submission.upgrade().is_none());
    inspector.queue.assert_submissions(0);
    inspector.device.assert_counts(0, 0);
}

#[test]
fn foreign_import_owner_rejection_never_marks_or_keeps_unsubmitted_guard() {
    let (producer, _) = runtime(1);
    let (consumer, instance) = runtime(1);
    let calls = Arc::new(AtomicUsize::new(0));
    let foreign = producer.execute_owned(frame(&calls)).unwrap();
    let graph = frame(&calls);
    let submission = graph.submission_token();
    assert!(matches!(
        consumer.execute_prepared_owned(graph.prepare().unwrap(), &[foreign]),
        Err(SubmissionError::ForeignImportOwner { index: 0 })
    ));
    assert!(submission.upgrade().is_none());
    let inspector = instance.device(0);
    inspector.queue.assert_submissions(0);
    inspector.device.assert_counts(0, 0);
}

#[test]
fn panicking_actual_submit_does_not_acknowledge_or_strand_guard() {
    let (runtime, instance) = runtime(1);
    let inspector = instance.device(0);
    // No physical allocations: the normal conservative allocation leak on a
    // submit panic is independent of whether the CPU-only guard is released.
    let graph = runtime.new_graph();
    let submission = graph.submission_token();
    let observer = submission.upgrade().unwrap();
    let native_token = submission.clone();
    lock(&inspector.queue.0).before_submit = Some(Box::new(move || {
        assert!(!native_token.upgrade().unwrap().is_submitted());
        panic!("injected failure before actual queue submit");
    }));
    assert!(catch_unwind(AssertUnwindSafe(|| runtime.execute_owned(graph))).is_err());
    assert!(!observer.is_submitted());
    assert_eq!(submission.strong_count(), 1);
    drop(observer);
    assert!(submission.upgrade().is_none());
    inspector.queue.assert_submissions(0);
    inspector.device.assert_counts(0, 0);
    assert!(lock(&inspector.queue.0).callbacks.is_empty());
}
