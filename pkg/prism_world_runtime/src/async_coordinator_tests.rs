use super::*;
use alloc::vec;
use core::task::Waker;
use prism_world_stream::{AabbI64, GlobalPos};

struct DeferredIo {
    request: Option<CellReadRequest>,
    deferred: bool,
    cancelled: bool,
    wrong_header: bool,
}

impl CellIo for DeferredIo {
    type Error = ();
    fn submit(&mut self, request: CellReadRequest) -> Result<RequestId, SubmitError> {
        assert!(self.request.replace(request).is_none());
        Ok(RequestId(1))
    }
    fn cancel(&mut self, _: RequestId) -> bool {
        self.cancelled = self.request.is_some();
        self.cancelled
    }
    fn in_flight(&self) -> usize {
        usize::from(self.request.is_some())
    }
    fn poll_completion(&mut self, cx: &mut Context<'_>) -> Poll<Result<IoCompletion<()>, ()>> {
        if !self.deferred {
            self.deferred = true;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        let Some(request) = self.request.take() else {
            return Poll::Pending;
        };
        let completion = if self.cancelled {
            IoCompletion::Cancelled {
                request_id: RequestId(1),
                request,
            }
        } else {
            let mut metadata = request.manifest();
            if self.wrong_header {
                metadata.content_version += 1;
            }
            let bytes = ManifestHeader::from_manifest(metadata).encode();
            IoCompletion::Completed {
                request_id: RequestId(1),
                request,
                bytes,
            }
        };
        Poll::Ready(Ok(completion))
    }
}

#[derive(Default)]
struct Host(usize);
impl SceneHost for Host {
    type Error = ();
    fn load_cell(&mut self, _: &CellManifest, payload: &[u8]) -> Result<(), ()> {
        assert!(payload.is_empty());
        self.0 += 1;
        Ok(())
    }
    fn evict_cell(&mut self, _: SectorId) -> Result<(), ()> {
        Ok(())
    }
}

fn manifest() -> CellManifest {
    CellManifest {
        sector: SectorId::new(0, 0, 0, 0),
        bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1)).unwrap(),
        scene_key: 42,
        content_version: 1,
        derived_hash: 7,
        io_bytes: CELL_HEADER_SIZE as u64,
    }
}

#[test]
fn pending_preserves_load_and_cancel_never_commits() {
    for cancel in [false, true] {
        let mut coordinator = CellFrameCoordinator::with_io(DeferredIo {
            request: None,
            deferred: false,
            cancelled: false,
            wrong_header: false,
        });
        let mut host = Host::default();
        let m = manifest();
        let report = coordinator.plan_frame(
            &[DemandSource::new(m.sector, 0, 1, 10).unwrap()],
            &[m],
            &[],
            StreamBudget {
                io_bytes: m.io_bytes,
                max_actions: 1,
            },
            &mut host,
        );
        assert_eq!(report.submitted, 1);
        let mut cx = Context::from_waker(Waker::noop());
        assert!(coordinator
            .poll_io_with_context(&mut cx, &mut host)
            .is_pending());
        assert_eq!(coordinator.in_flight(), 1);
        assert_eq!(host.0, 0);
        if cancel {
            assert!(coordinator.cancel_sector(m.sector));
        }
        assert_eq!(
            coordinator.poll_io_with_context(&mut cx, &mut host),
            Poll::Ready(Ok(()))
        );
        assert_eq!(coordinator.in_flight(), 0);
        assert_eq!(host.0, usize::from(!cancel));
        assert_eq!(coordinator.trace().replay(), Ok(1));
    }
}

#[test]
fn stale_routing_and_overflow_fail_before_admission() {
    let mut io = RoutedCellIo::new(DeferredIo {
        request: None,
        deferred: false,
        cancelled: false,
        wrong_header: false,
    });
    let source = CellReadRequest::new(manifest(), 2)
        .with_byte_offset(10)
        .with_expected_checksum(123);
    io.register(source).unwrap();
    assert!(io.register(source.with_byte_offset(u64::MAX)).is_err());
    let resolved = io.resolve(CellReadRequest::new(manifest(), 99)).unwrap();
    assert_eq!(resolved.io_request().offset(), 10);
    assert_eq!(resolved.io_request().expected_checksum(), Some(123));
    assert_eq!(resolved.priority(), 99);
    let mut stale = manifest();
    stale.content_version += 1;
    assert!(matches!(
        io.submit(CellReadRequest::new(stale, 0)),
        Err(SubmitError::Provider { .. })
    ));
    assert_eq!(io.in_flight(), 0);
}

#[test]
fn synchronous_runtime_fails_closed_at_identity_exhaustion() {
    struct Empty;
    impl CellProvider for Empty {
        type Error = ();
        fn read_cell(&self, _: &CellReadRequest, _: CancellationToken) -> Result<Vec<u8>, ()> {
            Ok(vec![])
        }
    }
    let mut io = IoRuntime::new(Empty, 2).unwrap();
    io.next_request_id = Some(u64::MAX);
    assert_eq!(
        io.submit(CellReadRequest::new(manifest(), 0))
            .unwrap()
            .get(),
        u64::MAX
    );
    assert_eq!(
        io.submit(CellReadRequest::new(manifest(), 0)),
        Err(SubmitError::RequestIdExhausted)
    );
    assert_eq!(io.in_flight(), 1);
}
