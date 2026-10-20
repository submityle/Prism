//! Runtime process lifecycle and adapter integration tests.

use prism_editor_remote::{
    EngineRuntimeAdapter, JsonRpcId, JsonRpcMessage, JsonRpcRequest, LoopbackRuntimeAdapter,
    RuntimeBridge, RuntimeError, RuntimeLaunchSpec, RuntimeLimits, RuntimeState, RuntimeTransport,
    TransportLimits,
};
use std::ffi::OsString;
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::Duration;

const MODE: &str = "PRISM_EDITOR_REMOTE_CHILD_MODE";
const ADDRESS: &str = "PRISM_EDITOR_REMOTE_CHILD_ADDRESS";

fn spec(mode: &str, transport: RuntimeTransport) -> RuntimeLaunchSpec {
    let mut result = RuntimeLaunchSpec::new(
        PathBuf::from(env!("CARGO_BIN_EXE_prism_editor_remote_runtime_child")),
        transport,
    );
    result
        .env
        .insert(OsString::from(MODE), OsString::from(mode));
    if let RuntimeTransport::Tcp { address } = transport {
        result
            .env
            .insert(OsString::from(ADDRESS), OsString::from(address.to_string()));
    }
    result
}

fn limits() -> RuntimeLimits {
    RuntimeLimits {
        transport: TransportLimits {
            read_timeout: Some(Duration::from_secs(2)),
            write_timeout: Some(Duration::from_secs(2)),
            connect_timeout: Some(Duration::from_millis(100)),
            ..TransportLimits::default()
        },
        startup_timeout: Duration::from_secs(2),
        shutdown_timeout: Duration::from_secs(2),
        max_pending_requests: 2,
    }
}

fn discovery() -> JsonRpcMessage {
    JsonRpcMessage::Request(JsonRpcRequest {
        id: JsonRpcId::Number(44),
        method: String::from("rpc.discover"),
        params: None,
    })
}

#[test]
fn stdio_runtime_can_start_request_restart_and_stop() {
    let mut bridge = RuntimeBridge::new(
        LoopbackRuntimeAdapter::new(),
        spec("serve", RuntimeTransport::Stdio),
        limits(),
    );
    assert_eq!(bridge.adapter().engine_id(), "loopback");
    assert_eq!(bridge.start().expect("start").state, RuntimeState::Running);
    assert!(bridge
        .request(discovery())
        .expect("request")
        .wait()
        .expect("response")
        .response
        .error
        .is_none());
    let first_generation = bridge.status().generation;
    assert_eq!(bridge.stop().expect("stop").state, RuntimeState::Stopped);
    assert_eq!(
        bridge.restart().expect("restart").state,
        RuntimeState::Running
    );
    assert!(bridge.status().generation > first_generation);
    assert_eq!(
        bridge.stop().expect("second stop").state,
        RuntimeState::Stopped
    );
}

#[test]
fn tcp_runtime_waits_for_child_listener_and_uses_bounded_connection() {
    let probe = TcpListener::bind("127.0.0.1:0").expect("probe bind");
    let address = probe.local_addr().expect("probe address");
    drop(probe);
    let mut bridge = RuntimeBridge::new(
        LoopbackRuntimeAdapter::with_engine_id("mock-engine"),
        spec("tcp", RuntimeTransport::Tcp { address }),
        limits(),
    );
    assert_eq!(bridge.start().expect("start").state, RuntimeState::Running);
    assert!(bridge.request(discovery()).expect("request").wait().is_ok());
    assert_eq!(bridge.stop().expect("stop").state, RuntimeState::Stopped);
}

#[test]
fn request_timeout_terminates_runtime_and_reports_bounded_deadline() {
    let mut bridge = RuntimeBridge::new(
        LoopbackRuntimeAdapter::default(),
        spec("block-after-discover", RuntimeTransport::Stdio),
        limits(),
    );
    bridge.start().expect("start");
    let error = bridge
        .request(discovery())
        .expect("request")
        .wait_timeout(Duration::from_millis(40))
        .expect_err("timeout");
    assert!(matches!(error, RuntimeError::Timeout { .. }));
    assert_eq!(
        bridge.wait(Duration::from_secs(2)).expect("wait").state,
        RuntimeState::Exited
    );
}

#[test]
fn request_cancel_terminates_runtime_and_reports_cancellation() {
    let mut bridge = RuntimeBridge::new(
        LoopbackRuntimeAdapter::default(),
        spec("block-after-discover", RuntimeTransport::Stdio),
        limits(),
    );
    bridge.start().expect("start");
    let request = bridge.request(discovery()).expect("request");
    request.cancel().expect("cancel");
    assert!(matches!(
        request.wait(),
        Err(RuntimeError::Cancelled { .. })
    ));
    assert_eq!(
        bridge.wait(Duration::from_secs(2)).expect("wait").state,
        RuntimeState::Exited
    );
}

#[test]
fn request_queue_is_bounded_before_more_work_is_accepted() {
    let mut bridge = RuntimeBridge::new(
        LoopbackRuntimeAdapter::default(),
        spec("block-after-discover", RuntimeTransport::Stdio),
        limits(),
    );
    bridge.start().expect("start");
    let first = bridge.request(discovery()).expect("first request");
    let _second = bridge.request(discovery()).expect("second request");
    assert!(matches!(
        bridge.request(discovery()),
        Err(RuntimeError::RequestQueueFull)
    ));
    first.cancel().expect("cancel");
    assert!(matches!(first.wait(), Err(RuntimeError::Cancelled { .. })));
    assert_eq!(
        bridge.wait(Duration::from_secs(2)).expect("wait").state,
        RuntimeState::Exited
    );
}

#[test]
fn completed_requests_release_capacity_before_delivering_the_response() {
    let mut request_limits = limits();
    request_limits.max_pending_requests = 1;
    let mut bridge = RuntimeBridge::new(
        LoopbackRuntimeAdapter::default(),
        spec("serve", RuntimeTransport::Stdio),
        request_limits,
    );
    bridge.start().expect("start");
    for _ in 0..16 {
        bridge
            .request(discovery())
            .expect("capacity released")
            .wait()
            .expect("response");
    }
    bridge.stop().expect("stop");
}

#[test]
fn dropping_a_request_handle_does_not_release_unfinished_work() {
    let mut request_limits = limits();
    request_limits.max_pending_requests = 1;
    let mut bridge = RuntimeBridge::new(
        LoopbackRuntimeAdapter::default(),
        spec("block-after-discover", RuntimeTransport::Stdio),
        request_limits,
    );
    bridge.start().expect("start");
    drop(bridge.request(discovery()).expect("first request"));
    assert!(matches!(
        bridge.request(discovery()),
        Err(RuntimeError::RequestQueueFull)
    ));
    bridge.stop().expect("stop");
}

#[test]
fn unexpected_process_exit_exposes_exit_code() {
    let mut bridge = RuntimeBridge::new(
        LoopbackRuntimeAdapter::default(),
        spec("exit-after-discover", RuntimeTransport::Stdio),
        limits(),
    );
    bridge.start().expect("start");
    let status = bridge.wait(Duration::from_secs(2)).expect("wait");
    assert_eq!(status.state, RuntimeState::Exited);
    assert_eq!(status.exit_code, Some(23));
}

#[test]
fn adapter_can_transform_launch_and_reject_startup() {
    struct RejectingAdapter;

    impl EngineRuntimeAdapter for RejectingAdapter {
        fn engine_id(&self) -> &str {
            "rejecting"
        }

        fn prepare_launch(
            &self,
            _requested: &RuntimeLaunchSpec,
        ) -> Result<RuntimeLaunchSpec, RuntimeError> {
            Err(RuntimeError::Adapter(String::from("launch rejected")))
        }

        fn initialize(
            &self,
            _session: &mut dyn prism_editor_remote::RuntimeSession,
            _timeout: Duration,
        ) -> Result<(), RuntimeError> {
            Ok(())
        }
    }

    let mut bridge = RuntimeBridge::new(
        RejectingAdapter,
        RuntimeLaunchSpec::new(PathBuf::from("unused"), RuntimeTransport::Stdio),
        limits(),
    );
    assert!(matches!(bridge.start(), Err(RuntimeError::Adapter(_))));
}
