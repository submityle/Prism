//! Acceptance tests for the public, in-process `BakeService` API.

use prism_asset_build_core::{
    BakeError, BakeKernel, BakeKernelRegistry, BakeRequest, BakeService, CancellationToken,
    ContentHash, KernelDispatchError, KernelError, RuntimeCacheConfig, StableInputKey,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

fn request(input: &[u8], settings: &[u8]) -> BakeRequest {
    BakeRequest::new(
        StableInputKey::from_bytes(input),
        input.to_vec(),
        0x4b45524e454c,
        0x524543495045,
        7,
        settings.to_vec(),
        0x4f5554505554,
        3,
        16,
    )
}

fn service<K>(max_in_flight: usize, kernel: K) -> BakeService
where
    K: BakeKernel,
{
    BakeService::new(max_in_flight, Arc::new(kernel)).unwrap()
}

fn cached_service<K>(
    max_in_flight: usize,
    cache_config: RuntimeCacheConfig,
    kernel: K,
) -> BakeService
where
    K: BakeKernel,
{
    BakeService::new_with_cache(max_in_flight, cache_config, Arc::new(kernel)).unwrap()
}

#[test]
fn max_in_flight_rejects_without_an_unbounded_queue() {
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let first_call = Arc::new(AtomicUsize::new(0));
    let entered_by_kernel = Arc::clone(&entered);
    let release_by_kernel = Arc::clone(&release);
    let first_call_by_kernel = Arc::clone(&first_call);
    let service = service(1, move |_: &BakeRequest, _: &CancellationToken| {
        if first_call_by_kernel.fetch_add(1, Ordering::SeqCst) == 0 {
            entered_by_kernel.wait();
            release_by_kernel.wait();
        }
        Ok(b"first".to_vec())
    });

    let first = service.submit(request(b"one", b"settings")).unwrap();
    entered.wait();
    assert!(matches!(
        service.submit(request(b"two", b"settings")),
        Err(BakeError::QueueFull { max_in_flight: 1 })
    ));
    release.wait();
    first.wait().unwrap();
    assert_eq!(service.in_flight(), 0);
    service
        .submit(request(b"three", b"settings"))
        .unwrap()
        .wait()
        .unwrap();
}

#[test]
fn successful_result_has_reproducible_key_and_descriptor() {
    let service = service(2, |request: &BakeRequest, _: &CancellationToken| {
        let mut output = request.input_bytes.clone();
        output.extend_from_slice(&request.canonical_settings);
        Ok(output)
    });
    let first = service
        .submit(request(b"input", b"canonical"))
        .unwrap()
        .wait()
        .unwrap();
    let second = service
        .submit(request(b"input", b"canonical"))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.key, first.descriptor.key);
    assert_eq!(
        first.descriptor.content_hash,
        ContentHash::digest(&first.output_bytes)
    );
    assert_eq!(first.descriptor.byte_len, first.output_bytes.len() as u64);
}

#[test]
fn cancellation_before_publication_returns_no_result() {
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let entered_by_kernel = Arc::clone(&entered);
    let release_by_kernel = Arc::clone(&release);
    let service = service(1, move |_: &BakeRequest, token: &CancellationToken| {
        entered_by_kernel.wait();
        release_by_kernel.wait();
        assert!(token.is_cancelled());
        Ok(b"must not publish".to_vec())
    });
    let handle = service.submit(request(b"cancel", b"settings")).unwrap();
    entered.wait();
    assert!(handle.cancel());
    release.wait();
    assert!(matches!(handle.wait(), Err(BakeError::Cancelled)));
    assert_eq!(service.in_flight(), 0);
}

#[test]
fn kernel_failure_is_structured_and_gate_recovers() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_by_kernel = Arc::clone(&calls);
    let service = service(1, move |_: &BakeRequest, _: &CancellationToken| {
        calls_by_kernel.fetch_add(1, Ordering::Relaxed);
        Err(KernelError::new("bad input"))
    });
    assert!(matches!(
        service.submit(request(b"bad", b"settings")).unwrap().wait(),
        Err(BakeError::Kernel(error)) if error.message() == "bad input"
    ));
    assert_eq!(service.in_flight(), 0);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn panic_is_caught_and_gate_recovers() {
    let should_panic = Arc::new(AtomicUsize::new(1));
    let should_panic_by_kernel = Arc::clone(&should_panic);
    let service = service(1, move |_: &BakeRequest, _: &CancellationToken| {
        if should_panic_by_kernel.fetch_sub(1, Ordering::SeqCst) == 1 {
            panic!("kernel panic for regression test");
        }
        Ok(b"recovered".to_vec())
    });
    assert!(matches!(
        service
            .submit(request(b"panic", b"settings"))
            .unwrap()
            .wait(),
        Err(BakeError::KernelPanicked)
    ));
    assert_eq!(service.in_flight(), 0);
    assert_eq!(
        service
            .submit(request(b"after-panic", b"settings"))
            .unwrap()
            .wait()
            .unwrap()
            .output_bytes,
        b"recovered"
    );
}

#[test]
fn invalid_request_is_rejected_before_admission() {
    let service = service(1, |_: &BakeRequest, _: &CancellationToken| Ok(Vec::new()));
    let mut invalid = request(b"invalid", b"settings");
    invalid.output_alignment = 3;
    assert!(matches!(
        service.submit(invalid),
        Err(BakeError::InvalidRequest(_))
    ));
    assert_eq!(service.in_flight(), 0);
}

#[test]
fn service_does_not_publish_external_side_effects() {
    let writes = Arc::new(Mutex::new(Vec::<Vec<u8>>::new()));
    let writes_by_kernel = Arc::clone(&writes);
    let service = service(1, move |request: &BakeRequest, _: &CancellationToken| {
        let output = request.input_bytes.clone();
        writes_by_kernel.lock().unwrap().push(output.clone());
        Ok(output)
    });
    let result = service
        .submit(request(b"memory-only", b"settings"))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(result.output_bytes, b"memory-only");
    assert_eq!(
        writes.lock().unwrap().as_slice(),
        &[b"memory-only".to_vec()]
    );
    assert_eq!(service.in_flight(), 0);
}

#[test]
fn concurrent_submissions_are_bounded() {
    let active = Arc::new(AtomicUsize::new(0));
    let maximum = Arc::new(AtomicUsize::new(0));
    let active_by_kernel = Arc::clone(&active);
    let maximum_by_kernel = Arc::clone(&maximum);
    let service = service(2, move |_: &BakeRequest, _: &CancellationToken| {
        let now = active_by_kernel.fetch_add(1, Ordering::SeqCst) + 1;
        maximum_by_kernel.fetch_max(now, Ordering::SeqCst);
        thread::yield_now();
        active_by_kernel.fetch_sub(1, Ordering::SeqCst);
        Ok(Vec::new())
    });
    let mut handles = Vec::new();
    for index in 0..2 {
        handles.push(
            service
                .submit(request(&[index as u8], b"settings"))
                .unwrap(),
        );
    }
    for handle in handles {
        handle.wait().unwrap();
    }
    assert!(maximum.load(Ordering::SeqCst) <= 2);
    assert_eq!(service.in_flight(), 0);
}

#[test]
fn successful_result_is_cached_by_artifact_key() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_by_kernel = Arc::clone(&calls);
    let service = cached_service(
        2,
        RuntimeCacheConfig::new(4, 1024),
        move |request: &BakeRequest, _: &CancellationToken| {
            calls_by_kernel.fetch_add(1, Ordering::SeqCst);
            Ok(request.input_bytes.clone())
        },
    );
    let first = service
        .submit(request(b"cache-hit", b"settings"))
        .unwrap()
        .wait()
        .unwrap();
    let second = service
        .submit(request(b"cache-hit", b"settings"))
        .unwrap()
        .wait()
        .unwrap();

    assert_eq!(first, second);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(service.is_cached(first.key));
    assert_eq!(service.cache_stats().hits, 1);
    assert_eq!(service.cache_stats().misses, 1);
}

#[test]
fn entry_capacity_evicts_largest_key_deterministically() {
    let service = cached_service(
        3,
        RuntimeCacheConfig::new(2, 1024),
        |request: &BakeRequest, _: &CancellationToken| Ok(request.input_bytes.clone()),
    );
    let requests = [
        request(b"entry-a", b"settings"),
        request(b"entry-b", b"settings"),
        request(b"entry-c", b"settings"),
    ];
    let keys: Vec<_> = requests.iter().map(BakeRequest::request_key).collect();
    for item in requests.iter().cloned() {
        service.submit(item).unwrap().wait().unwrap();
    }

    let mut sorted = keys;
    sorted.sort();
    assert!(service.is_cached(sorted[0]));
    assert!(service.is_cached(sorted[1]));
    assert!(!service.is_cached(sorted[2]));
    assert_eq!(service.cache_stats().evictions, 1);
}

#[test]
fn byte_capacity_evicts_largest_key_without_exceeding_budget() {
    let service = cached_service(
        3,
        RuntimeCacheConfig::new(8, 5),
        |request: &BakeRequest, _: &CancellationToken| Ok(vec![request.input_bytes[0]; 3]),
    );
    let requests = [
        request(b"byte-a", b"settings"),
        request(b"byte-b", b"settings"),
        request(b"byte-c", b"settings"),
    ];
    let keys: Vec<_> = requests.iter().map(BakeRequest::request_key).collect();
    for item in requests.iter().cloned() {
        service.submit(item).unwrap().wait().unwrap();
    }

    let mut sorted = keys;
    sorted.sort();
    assert!(service.is_cached(sorted[0]));
    assert!(!service.is_cached(sorted[1]));
    assert!(!service.is_cached(sorted[2]));
    assert_eq!(service.cache_stats().bytes, 3);
    assert_eq!(service.cache_stats().evictions, 2);
}

#[test]
fn eviction_result_is_independent_of_completion_insertion_order() {
    let config = RuntimeCacheConfig::new(2, 1024);
    let first = cached_service(3, config, |request: &BakeRequest, _: &CancellationToken| {
        Ok(request.input_bytes.clone())
    });
    let second = cached_service(3, config, |request: &BakeRequest, _: &CancellationToken| {
        Ok(request.input_bytes.clone())
    });
    let requests = [
        request(b"order-a", b"settings"),
        request(b"order-b", b"settings"),
        request(b"order-c", b"settings"),
    ];
    let keys: Vec<_> = requests.iter().map(BakeRequest::request_key).collect();
    for item in requests.iter().cloned() {
        first.submit(item).unwrap().wait().unwrap();
    }
    for item in requests.iter().rev().cloned() {
        second.submit(item).unwrap().wait().unwrap();
    }
    for key in keys {
        assert_eq!(first.is_cached(key), second.is_cached(key));
    }
}

#[test]
fn failed_and_cancelled_results_do_not_enter_cache() {
    let failed_calls = Arc::new(AtomicUsize::new(0));
    let failed_calls_by_kernel = Arc::clone(&failed_calls);
    let failed = cached_service(
        1,
        RuntimeCacheConfig::new(4, 1024),
        move |_: &BakeRequest, _: &CancellationToken| {
            failed_calls_by_kernel.fetch_add(1, Ordering::SeqCst);
            Err(KernelError::new("expected failure"))
        },
    );
    let failed_request = request(b"failed-cache", b"settings");
    let failed_key = failed_request.request_key();
    assert!(matches!(
        failed.submit(failed_request.clone()).unwrap().wait(),
        Err(BakeError::Kernel(error)) if error.message() == "expected failure"
    ));
    assert!(matches!(
        failed.submit(failed_request).unwrap().wait(),
        Err(BakeError::Kernel(_))
    ));
    assert_eq!(failed_calls.load(Ordering::SeqCst), 2);
    assert!(!failed.is_cached(failed_key));

    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let cancelled_calls = Arc::new(AtomicUsize::new(0));
    let entered_by_kernel = Arc::clone(&entered);
    let release_by_kernel = Arc::clone(&release);
    let cancelled_calls_by_kernel = Arc::clone(&cancelled_calls);
    let cancelled = cached_service(
        1,
        RuntimeCacheConfig::new(4, 1024),
        move |_: &BakeRequest, token: &CancellationToken| {
            if cancelled_calls_by_kernel.fetch_add(1, Ordering::SeqCst) == 0 {
                entered_by_kernel.wait();
                release_by_kernel.wait();
                assert!(token.is_cancelled());
            }
            Ok(b"cancelled-output".to_vec())
        },
    );
    let cancelled_request = request(b"cancelled-cache", b"settings");
    let cancelled_key = cancelled_request.request_key();
    let handle = cancelled.submit(cancelled_request.clone()).unwrap();
    entered.wait();
    assert!(handle.cancel());
    release.wait();
    assert!(matches!(handle.wait(), Err(BakeError::Cancelled)));
    assert!(!cancelled.is_cached(cancelled_key));
    assert!(cancelled.submit(cancelled_request).unwrap().wait().is_ok());
}

#[test]
fn registry_dispatch_rejects_unknown_kernel_without_admission() {
    let registry = Arc::new(BakeKernelRegistry::new());
    registry
        .register(
            0x11,
            Arc::new(|_: &BakeRequest, _: &CancellationToken| Ok(b"known".to_vec())),
        )
        .unwrap();
    let service =
        BakeService::new_with_registry(1, RuntimeCacheConfig::new(2, 32), Arc::clone(&registry))
            .unwrap();
    let mut unknown = request(b"unknown-kernel", b"settings");
    unknown.kernel_id = 0x22;
    assert!(matches!(
        service.submit(unknown),
        Err(BakeError::KernelDispatch(
            KernelDispatchError::UnknownKernel { kernel_id: 0x22 }
        ))
    ));
    assert_eq!(service.in_flight(), 0);
    assert!(!registry.contains(0x22));
}

#[test]
fn concurrent_duplicate_requests_share_one_kernel_execution() {
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let calls = Arc::new(AtomicUsize::new(0));
    let entered_by_kernel = Arc::clone(&entered);
    let release_by_kernel = Arc::clone(&release);
    let calls_by_kernel = Arc::clone(&calls);
    let service = cached_service(
        2,
        RuntimeCacheConfig::new(4, 1024),
        move |request: &BakeRequest, _: &CancellationToken| {
            calls_by_kernel.fetch_add(1, Ordering::SeqCst);
            entered_by_kernel.wait();
            release_by_kernel.wait();
            Ok(request.input_bytes.clone())
        },
    );
    let duplicate = request(b"duplicate", b"settings");
    let first = service.submit(duplicate.clone()).unwrap();
    entered.wait();
    let second = service.submit(duplicate).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    release.wait();
    assert_eq!(first.wait().unwrap().output_bytes, b"duplicate");
    assert_eq!(second.wait().unwrap().output_bytes, b"duplicate");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
