//! Acceptance tests for the public, in-process `BakeService` API.

use prism_asset_build_core::{
    BakeError, BakeKernel, BakeRequest, BakeService, CancellationToken, ContentHash, KernelError,
    StableInputKey,
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
