use prism_asset_build_core::{
    BakeError, BakeRequest, BakeService, BuildGraph, BuildJob, CancellationToken, DirtyBakeError,
    JobId, KernelError, RuntimeCacheConfig, StableInputKey,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

fn job(id: u128, dependencies: &[u128]) -> BuildJob {
    BuildJob::new(
        JobId(id),
        dependencies.iter().copied().map(JobId).collect::<Vec<_>>(),
        |_: &[&[u8]]| Ok(Vec::new()),
    )
}

fn request(input: &[u8]) -> BakeRequest {
    BakeRequest::new(
        StableInputKey::from_bytes(input),
        input.to_vec(),
        0x4b45524e454c,
        0x524543495045,
        1,
        b"settings-v1".to_vec(),
        0x52554e54494d45,
        1,
        16,
    )
}

fn requests(ids: &[u128]) -> BTreeMap<JobId, BakeRequest> {
    ids.iter()
        .map(|&id| (JobId(id), request(&[id as u8])))
        .collect()
}

#[test]
fn dirty_bake_selection_uses_closure_order_and_rejects_invalid_plans() {
    let mut graph = BuildGraph::new();
    graph.add_job(job(4, &[2])).unwrap();
    graph.add_job(job(3, &[1])).unwrap();
    graph.add_job(job(5, &[])).unwrap();
    graph.add_job(job(2, &[1])).unwrap();
    graph.add_job(job(1, &[])).unwrap();

    let dirty = graph.dirty_set([JobId(1)]).unwrap();
    let service = BakeService::new(
        4,
        Arc::new(|request: &BakeRequest, _: &CancellationToken| Ok(request.input_bytes.clone())),
    )
    .unwrap();
    let submissions = service
        .submit_dirty(&graph, &dirty, &requests(&[1, 2, 3, 4, 5]))
        .unwrap();
    assert_eq!(
        submissions.iter().map(|(job, _)| *job).collect::<Vec<_>>(),
        vec![JobId(1), JobId(2), JobId(3), JobId(4)]
    );
    assert_eq!(
        submissions
            .into_iter()
            .map(|(_, handle)| handle.wait().unwrap().output_bytes)
            .collect::<Vec<_>>(),
        vec![vec![1], vec![2], vec![3], vec![4]]
    );

    let empty = graph.dirty_set([]).unwrap();
    assert!(service
        .submit_dirty(&graph, &empty, &requests(&[1, 2, 3, 4, 5]))
        .unwrap()
        .is_empty());

    let mut foreign_graph = BuildGraph::new();
    foreign_graph.add_job(job(99, &[])).unwrap();
    let foreign_dirty = foreign_graph.dirty_set([JobId(99)]).unwrap();
    assert!(matches!(
        service.submit_dirty(&graph, &foreign_dirty, &BTreeMap::new()),
        Err(DirtyBakeError::Graph(
            prism_asset_build_core::BuildError::UnknownDirtyJob(JobId(99))
        ))
    ));

    let mut missing = requests(&[1, 2, 4]);
    assert!(matches!(
        service.submit_dirty(&graph, &dirty, &missing),
        Err(DirtyBakeError::MissingRequest(JobId(3)))
    ));
    missing.insert(JobId(99), request(b"unknown"));
    assert!(matches!(
        service.submit_dirty(&graph, &dirty, &missing),
        Err(DirtyBakeError::UnknownRequestJob(JobId(99)))
    ));
}

#[test]
fn duplicate_dirty_requests_share_execution_and_cached_results_are_stable() {
    let mut graph = BuildGraph::new();
    graph.add_job(job(20, &[10])).unwrap();
    graph.add_job(job(10, &[])).unwrap();
    let dirty = graph.dirty_set([JobId(10)]).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_by_kernel = Arc::clone(&calls);
    let service = BakeService::new_with_cache(
        2,
        RuntimeCacheConfig::new(4, 1024),
        Arc::new(move |request: &BakeRequest, _: &CancellationToken| {
            calls_by_kernel.fetch_add(1, Ordering::SeqCst);
            Ok(request.input_bytes.clone())
        }),
    )
    .unwrap();
    let duplicate = request(b"same-request");
    let request_map = [(JobId(10), duplicate.clone()), (JobId(20), duplicate)]
        .into_iter()
        .collect();

    let first = service.submit_dirty(&graph, &dirty, &request_map).unwrap();
    assert_eq!(
        first.iter().map(|(job, _)| *job).collect::<Vec<_>>(),
        vec![JobId(10), JobId(20)]
    );
    let first_results = first
        .into_iter()
        .map(|(_, handle)| handle.wait().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(first_results[0], first_results[1]);

    let second_results = service
        .submit_dirty(&graph, &dirty, &request_map)
        .unwrap()
        .into_iter()
        .map(|(_, handle)| handle.wait().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(first_results, second_results);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(service.cache_stats().entries, 1);
    // The first duplicate may arrive after the shared kernel has already
    // published, in which case it is a cache hit rather than a live-job
    // subscription. The two requests in the second submission are always
    // hits; the first submission contributes zero or one more depending on
    // scheduler timing.
    assert!(
        (2..=3).contains(&service.cache_stats().hits),
        "cache hits should account for both second-pass requests"
    );
}

#[test]
fn scheduled_failure_and_cancellation_do_not_pollute_runtime_cache() {
    let mut graph = BuildGraph::new();
    graph.add_job(job(1, &[])).unwrap();
    let dirty = graph.dirty_set([JobId(1)]).unwrap();

    let failed = BakeService::new_with_cache(
        1,
        RuntimeCacheConfig::new(4, 1024),
        Arc::new(|_: &BakeRequest, _: &CancellationToken| {
            Err(KernelError::new("expected failure"))
        }),
    )
    .unwrap();
    let failed_request = request(b"failed");
    let failed_key = failed_request.request_key();
    let failed_map = [(JobId(1), failed_request)].into_iter().collect();
    assert!(matches!(
        failed
            .submit_dirty(&graph, &dirty, &failed_map)
            .unwrap()
            .pop()
            .unwrap()
            .1
            .wait(),
        Err(BakeError::Kernel(error)) if error.message() == "expected failure"
    ));
    assert!(!failed.is_cached(failed_key));

    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let entered_by_kernel = Arc::clone(&entered);
    let release_by_kernel = Arc::clone(&release);
    let cancelled = BakeService::new_with_cache(
        1,
        RuntimeCacheConfig::new(4, 1024),
        Arc::new(move |_: &BakeRequest, token: &CancellationToken| {
            entered_by_kernel.wait();
            release_by_kernel.wait();
            assert!(token.is_cancelled());
            Ok(b"must-not-cache".to_vec())
        }),
    )
    .unwrap();
    let cancelled_request = request(b"cancelled");
    let cancelled_key = cancelled_request.request_key();
    let cancelled_map = [(JobId(1), cancelled_request)].into_iter().collect();
    let handle = cancelled
        .submit_dirty(&graph, &dirty, &cancelled_map)
        .unwrap()
        .pop()
        .unwrap()
        .1;
    entered.wait();
    assert!(handle.cancel());
    release.wait();
    assert!(matches!(handle.wait(), Err(BakeError::Cancelled)));
    assert!(!cancelled.is_cached(cancelled_key));
}
