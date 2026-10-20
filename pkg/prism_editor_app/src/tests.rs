//! Regression coverage for application assembly and host lifecycle.

use super::*;
use alloc::collections::VecDeque;
use alloc::sync::Arc;
use std::sync::Mutex;

use prism_editor_core::{DocumentId, DocumentState};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HostOperation {
    WindowStart,
    GpuStart,
    CreateSurface(WindowId),
    ResizeSurface(SurfaceId, PixelSize),
    SuspendSurface(SurfaceId),
    ResumeSurface(SurfaceId),
    PresentSurface(SurfaceId),
    PresentWindow(WindowId),
    DestroySurface(SurfaceId),
    GpuStop,
    WindowStop,
}

#[derive(Clone)]
struct MockWindowBackend {
    events: Arc<Mutex<VecDeque<WindowEventEnvelope>>>,
    operations: Arc<Mutex<Vec<HostOperation>>>,
}

impl WindowBackend for MockWindowBackend {
    fn start(&mut self, _windows: &[WindowDescriptor]) -> Result<(), WindowBackendError> {
        self.operations
            .lock()
            .expect("operations lock")
            .push(HostOperation::WindowStart);
        Ok(())
    }

    fn poll_events(
        &mut self,
        _limit: usize,
    ) -> Result<Vec<WindowEventEnvelope>, WindowBackendError> {
        let mut events = self.events.lock().expect("events lock");
        Ok(events.drain(..).collect())
    }

    fn present(&mut self, window: WindowId) -> Result<(), WindowBackendError> {
        self.operations
            .lock()
            .expect("operations lock")
            .push(HostOperation::PresentWindow(window));
        Ok(())
    }

    fn stop(&mut self) -> Result<(), WindowBackendError> {
        self.operations
            .lock()
            .expect("operations lock")
            .push(HostOperation::WindowStop);
        Ok(())
    }
}

#[derive(Clone, Default)]
struct MockGpuBackend {
    operations: Arc<Mutex<Vec<HostOperation>>>,
    next_surface: u64,
    fail_create_surface: Option<u64>,
    fail_present: bool,
}

impl GpuSurfaceLifecycle for MockGpuBackend {
    fn start(&mut self) -> Result<(), GpuSurfaceError> {
        self.operations
            .lock()
            .expect("operations lock")
            .push(HostOperation::GpuStart);
        Ok(())
    }

    fn create_surface(
        &mut self,
        descriptor: SurfaceDescriptor,
    ) -> Result<SurfaceId, GpuSurfaceError> {
        self.next_surface = self.next_surface.saturating_add(1);
        self.operations
            .lock()
            .expect("operations lock")
            .push(HostOperation::CreateSurface(descriptor.window));
        if self.fail_create_surface == Some(self.next_surface) {
            self.fail_create_surface = None;
            return Err(GpuSurfaceError::new(
                GpuSurfaceErrorKind::Recoverable,
                "surface creation failed",
            ));
        }
        Ok(SurfaceId::from_raw(self.next_surface))
    }

    fn resize_surface(
        &mut self,
        surface: SurfaceId,
        size: PixelSize,
    ) -> Result<(), GpuSurfaceError> {
        self.operations
            .lock()
            .expect("operations lock")
            .push(HostOperation::ResizeSurface(surface, size));
        Ok(())
    }

    fn destroy_surface(&mut self, surface: SurfaceId) -> Result<(), GpuSurfaceError> {
        self.operations
            .lock()
            .expect("operations lock")
            .push(HostOperation::DestroySurface(surface));
        Ok(())
    }

    fn suspend_surface(&mut self, surface: SurfaceId) -> Result<(), GpuSurfaceError> {
        self.operations
            .lock()
            .expect("operations lock")
            .push(HostOperation::SuspendSurface(surface));
        Ok(())
    }

    fn resume_surface(&mut self, surface: SurfaceId) -> Result<(), GpuSurfaceError> {
        self.operations
            .lock()
            .expect("operations lock")
            .push(HostOperation::ResumeSurface(surface));
        Ok(())
    }

    fn present_surface(&mut self, surface: SurfaceId) -> Result<(), GpuSurfaceError> {
        self.operations
            .lock()
            .expect("operations lock")
            .push(HostOperation::PresentSurface(surface));
        if core::mem::take(&mut self.fail_present) {
            return Err(GpuSurfaceError::new(
                GpuSurfaceErrorKind::DeviceLost,
                "device lost during presentation",
            ));
        }
        Ok(())
    }

    fn stop(&mut self) -> Result<(), GpuSurfaceError> {
        self.operations
            .lock()
            .expect("operations lock")
            .push(HostOperation::GpuStop);
        Ok(())
    }
}

fn windowed_fixture() -> (
    EditorApplication,
    Arc<Mutex<Vec<HostOperation>>>,
    Arc<Mutex<VecDeque<WindowEventEnvelope>>>,
) {
    let operations = Arc::new(Mutex::new(Vec::new()));
    let events = Arc::new(Mutex::new(VecDeque::new()));
    let app = EditorApplication::new(EditorAppConfig {
        mode: EditorRunMode::Windowed,
        ..EditorAppConfig::default()
    })
    .expect("app")
    .with_backends(
        MockWindowBackend {
            events: Arc::clone(&events),
            operations: Arc::clone(&operations),
        },
        MockGpuBackend {
            operations: Arc::clone(&operations),
            ..MockGpuBackend::default()
        },
    )
    .expect("backends");
    (app, operations, events)
}

fn enqueue(events: &Mutex<VecDeque<WindowEventEnvelope>>, window: u64, event: WindowEvent) {
    events
        .lock()
        .expect("events lock")
        .push_back(WindowEventEnvelope {
            window: WindowId::from_raw(window),
            event,
        });
}

#[test]
fn headless_run_is_bounded_and_deterministic() {
    let mut first = EditorApplication::new(EditorAppConfig::default()).expect("app");
    let mut second = EditorApplication::new(EditorAppConfig::default()).expect("app");
    let first_report = first.run_headless(4, |_| {}).expect("run");
    let second_report = second.run_headless(4, |_| {}).expect("run");
    assert_eq!(first_report, second_report);
    assert_eq!(first_report.snapshot.frame, 4);
    assert!(first.run_headless(10_001, |_| {}).is_err());
}

#[test]
fn document_registration_changes_snapshot_without_global_state() {
    let mut app = EditorApplication::new(EditorAppConfig::default()).expect("app");
    let document = DocumentState::new(DocumentId::from_raw(7));
    app.insert_document(document).expect("insert");
    assert_eq!(app.snapshot().document_count, 1);
    assert!(app.remove_document(DocumentId::from_raw(7)).is_ok());
    assert_eq!(app.snapshot().document_count, 0);
}

#[test]
fn frame_hook_can_observe_and_mutate_each_frame() {
    let mut app = EditorApplication::new(EditorAppConfig::default()).expect("app");
    let mut observed = Vec::new();
    app.run_headless(3, |app| observed.push(app.frame()))
        .expect("run");
    assert_eq!(observed, [1, 2, 3]);
}

#[test]
fn workspace_assembles_the_complete_renderer_independent_surface() {
    let app = EditorApplication::new(EditorAppConfig::default()).expect("app");
    let workspace = app.workspace();
    assert_eq!(
        workspace.kind(),
        &prism_ui::ElementKind::Custom("prism.editor.workspace".into())
    );
    assert_eq!(workspace.child_elements().len(), 3);
    assert_eq!(workspace.child_elements()[1].child_elements().len(), 3);
}

#[test]
fn windowed_lifecycle_orders_resources_events_and_shutdown() {
    let operations = Arc::new(Mutex::new(Vec::new()));
    let events = Arc::new(Mutex::new(VecDeque::from([
        WindowEventEnvelope {
            window: WindowId::from_raw(1),
            event: WindowEvent::Resized(PixelSize::new(1600, 900).expect("size")),
        },
        WindowEventEnvelope {
            window: WindowId::from_raw(2),
            event: WindowEvent::Suspended,
        },
        WindowEventEnvelope {
            window: WindowId::from_raw(2),
            event: WindowEvent::Resumed,
        },
    ])));
    let window_backend = MockWindowBackend {
        events,
        operations: Arc::clone(&operations),
    };
    let gpu_backend = MockGpuBackend {
        operations: Arc::clone(&operations),
        ..MockGpuBackend::default()
    };
    let mut app = EditorApplication::new(EditorAppConfig {
        mode: EditorRunMode::Windowed,
        ..EditorAppConfig::default()
    })
    .expect("app")
    .with_backends(window_backend, gpu_backend)
    .expect("backends");
    app.configure_window(WindowDescriptor::new(
        WindowId::from_raw(2),
        "Diagnostics",
        PixelSize::new(800, 600).expect("size"),
    ))
    .expect("configure window");

    app.start().expect("start");
    assert_eq!(app.state(), ApplicationState::Running);
    assert_eq!(
        app.surface_for(WindowId::from_raw(1)),
        Some(SurfaceId::from_raw(1))
    );
    assert_eq!(
        app.surface_for(WindowId::from_raw(2)),
        Some(SurfaceId::from_raw(2))
    );
    app.tick().expect("tick");
    let report = app.stop(ExitReason::Requested).expect("stop");

    assert_eq!(report.frame, 1);
    assert_eq!(app.state(), ApplicationState::Stopped);
    assert_eq!(
        app.exit_status(),
        ExitStatus::Completed(ExitReason::Requested)
    );
    assert_eq!(
        *operations.lock().expect("operations lock"),
        vec![
            HostOperation::WindowStart,
            HostOperation::GpuStart,
            HostOperation::CreateSurface(WindowId::from_raw(1)),
            HostOperation::CreateSurface(WindowId::from_raw(2)),
            HostOperation::ResizeSurface(
                SurfaceId::from_raw(1),
                PixelSize::new(1600, 900).expect("size"),
            ),
            HostOperation::SuspendSurface(SurfaceId::from_raw(2)),
            HostOperation::ResumeSurface(SurfaceId::from_raw(2)),
            HostOperation::PresentSurface(SurfaceId::from_raw(1)),
            HostOperation::PresentWindow(WindowId::from_raw(1)),
            HostOperation::PresentSurface(SurfaceId::from_raw(2)),
            HostOperation::PresentWindow(WindowId::from_raw(2)),
            HostOperation::DestroySurface(SurfaceId::from_raw(2)),
            HostOperation::DestroySurface(SurfaceId::from_raw(1)),
            HostOperation::GpuStop,
            HostOperation::WindowStop,
        ]
    );
}

#[test]
fn windowed_start_without_adapters_fails_explicitly() {
    let mut app = EditorApplication::new(EditorAppConfig {
        mode: EditorRunMode::Windowed,
        ..EditorAppConfig::default()
    })
    .expect("app");

    assert_eq!(app.start(), Err(EditorAppError::MissingWindowBackend));
    assert_eq!(app.state(), ApplicationState::Failed);
    assert_eq!(app.exit_status(), ExitStatus::Failed);
    assert_eq!(
        app.last_error(),
        Some(&EditorAppError::MissingWindowBackend)
    );
}

#[test]
fn window_event_burst_fails_without_unbounded_frame_work() {
    let operations = Arc::new(Mutex::new(Vec::new()));
    let events = Arc::new(Mutex::new(VecDeque::from([
        WindowEventEnvelope {
            window: WindowId::from_raw(1),
            event: WindowEvent::RedrawRequested,
        },
        WindowEventEnvelope {
            window: WindowId::from_raw(1),
            event: WindowEvent::RedrawRequested,
        },
    ])));
    let mut app = EditorApplication::new(EditorAppConfig {
        mode: EditorRunMode::Windowed,
        max_window_events_per_frame: 1,
        ..EditorAppConfig::default()
    })
    .expect("app")
    .with_backends(
        MockWindowBackend {
            events,
            operations: Arc::clone(&operations),
        },
        MockGpuBackend {
            operations: Arc::clone(&operations),
            ..MockGpuBackend::default()
        },
    )
    .expect("backends");
    app.start().expect("start");

    assert_eq!(
        app.tick(),
        Err(EditorAppError::WindowEventLimitExceeded {
            observed: 2,
            limit: 1,
        })
    );
    assert_eq!(app.state(), ApplicationState::Failed);
    assert_eq!(app.exit_status(), ExitStatus::Failed);
    assert_eq!(
        *operations.lock().expect("operations lock"),
        vec![
            HostOperation::WindowStart,
            HostOperation::GpuStart,
            HostOperation::CreateSurface(WindowId::from_raw(1)),
            HostOperation::DestroySurface(SurfaceId::from_raw(1)),
            HostOperation::GpuStop,
            HostOperation::WindowStop,
        ]
    );
}

#[test]
fn one_suspended_window_does_not_pause_other_windows() {
    let (mut app, operations, events) = windowed_fixture();
    app.configure_window(WindowDescriptor::new(
        WindowId::from_raw(2),
        "Second",
        PixelSize::new(640, 480).expect("size"),
    ))
    .expect("window");
    app.start().expect("start");
    operations.lock().expect("operations lock").clear();
    enqueue(&events, 2, WindowEvent::Suspended);
    enqueue(&events, 2, WindowEvent::Suspended);
    assert_eq!(app.tick().expect("tick").frame, 1);
    assert_eq!(app.state(), ApplicationState::Running);
    assert_eq!(
        *operations.lock().expect("operations lock"),
        [
            HostOperation::SuspendSurface(SurfaceId::from_raw(2)),
            HostOperation::PresentSurface(SurfaceId::from_raw(1)),
            HostOperation::PresentWindow(WindowId::from_raw(1))
        ],
    );
}

#[test]
fn zero_size_pauses_and_nonzero_resize_restores_presentation() {
    let (mut app, operations, events) = windowed_fixture();
    app.start().expect("start");
    operations.lock().expect("operations lock").clear();
    enqueue(
        &events,
        1,
        WindowEvent::Resized(PixelSize {
            width: 0,
            height: 0,
        }),
    );
    assert_eq!(app.tick().expect("minimize").frame, 0);
    assert_eq!(app.state(), ApplicationState::Suspended);
    let size = PixelSize::new(640, 480).expect("size");
    enqueue(&events, 1, WindowEvent::Resized(size));
    assert_eq!(app.tick().expect("restore").frame, 1);
    assert_eq!(app.state(), ApplicationState::Running);
    assert_eq!(
        *operations.lock().expect("operations lock"),
        [
            HostOperation::SuspendSurface(SurfaceId::from_raw(1)),
            HostOperation::ResizeSurface(SurfaceId::from_raw(1), size),
            HostOperation::ResumeSurface(SurfaceId::from_raw(1)),
            HostOperation::PresentSurface(SurfaceId::from_raw(1)),
            HostOperation::PresentWindow(WindowId::from_raw(1))
        ],
    );
}

#[test]
fn close_request_stops_without_presenting_and_drop_does_not_stop_twice() {
    let (mut app, operations, events) = windowed_fixture();
    app.start().expect("start");
    operations.lock().expect("operations lock").clear();
    enqueue(&events, 1, WindowEvent::CloseRequested);
    assert_eq!(app.tick().expect("close").frame, 0);
    assert_eq!(
        app.exit_status(),
        ExitStatus::Completed(ExitReason::WindowClosed)
    );
    drop(app);
    assert_eq!(
        *operations.lock().expect("operations lock"),
        [
            HostOperation::DestroySurface(SurfaceId::from_raw(1)),
            HostOperation::GpuStop,
            HostOperation::WindowStop
        ],
    );
}

#[test]
fn dropping_a_running_application_releases_host_resources() {
    let (mut app, operations, _) = windowed_fixture();
    app.start().expect("start");
    operations.lock().expect("operations lock").clear();
    drop(app);
    assert_eq!(
        *operations.lock().expect("operations lock"),
        [
            HostOperation::DestroySurface(SurfaceId::from_raw(1)),
            HostOperation::GpuStop,
            HostOperation::WindowStop
        ],
    );
}

#[test]
fn partial_startup_is_rolled_back_and_can_be_recovered() {
    let operations = Arc::new(Mutex::new(Vec::new()));
    let events = Arc::new(Mutex::new(VecDeque::new()));
    let mut app = EditorApplication::new(EditorAppConfig {
        mode: EditorRunMode::Windowed,
        ..EditorAppConfig::default()
    })
    .expect("app")
    .with_backends(
        MockWindowBackend {
            events,
            operations: Arc::clone(&operations),
        },
        MockGpuBackend {
            operations: Arc::clone(&operations),
            fail_create_surface: Some(2),
            ..MockGpuBackend::default()
        },
    )
    .expect("backends");
    app.configure_window(WindowDescriptor::new(
        WindowId::from_raw(2),
        "Second",
        PixelSize::new(640, 480).expect("size"),
    ))
    .expect("window");
    assert!(matches!(app.start(), Err(EditorAppError::GpuSurface(_))));
    assert_eq!(app.state(), ApplicationState::Failed);
    assert_eq!(app.surface_for(WindowId::from_raw(1)), None);
    assert_eq!(
        *operations.lock().expect("operations lock"),
        [
            HostOperation::WindowStart,
            HostOperation::GpuStart,
            HostOperation::CreateSurface(WindowId::from_raw(1)),
            HostOperation::CreateSurface(WindowId::from_raw(2)),
            HostOperation::DestroySurface(SurfaceId::from_raw(1)),
            HostOperation::GpuStop,
            HostOperation::WindowStop
        ],
    );
    app.recover().expect("recover");
    assert_eq!(app.state(), ApplicationState::Running);
    assert_eq!(app.last_error(), None);
    assert_eq!(
        app.surface_for(WindowId::from_raw(1)),
        Some(SurfaceId::from_raw(3))
    );
    assert_eq!(
        app.surface_for(WindowId::from_raw(2)),
        Some(SurfaceId::from_raw(4))
    );
}

#[test]
fn app_update_propagates_failure_and_exit_to_host() {
    let operations = Arc::new(Mutex::new(Vec::new()));
    let mut app = build_prism_app_with_backends(
        EditorAppConfig {
            mode: EditorRunMode::Windowed,
            ..EditorAppConfig::default()
        },
        MockWindowBackend {
            events: Arc::new(Mutex::new(VecDeque::new())),
            operations: Arc::clone(&operations),
        },
        MockGpuBackend {
            operations,
            fail_present: true,
            ..MockGpuBackend::default()
        },
    )
    .expect("host app");
    app.update();
    assert_eq!(app.should_exit(), Some(prism_app::AppExit::error()));
    let editor = app.world().resource::<EditorApplication>();
    assert_eq!(editor.state(), ApplicationState::Failed);
    assert!(matches!(
        editor.last_error(),
        Some(EditorAppError::GpuSurface(_))
    ));

    let mut app = build_prism_app(EditorAppConfig::default()).expect("headless app");
    app.world_mut()
        .resource_mut::<EditorApplication>()
        .request_exit(ExitReason::Requested)
        .expect("exit request");
    app.update();
    assert_eq!(app.should_exit(), Some(prism_app::AppExit::Success));
    assert_eq!(app.world().resource::<EditorApplication>().frame(), 0);
}

#[test]
fn headless_hook_exit_preserves_reason_and_actual_frame_count() {
    let mut app = EditorApplication::new(EditorAppConfig::default()).expect("app");
    let report = app
        .run_headless(10, |app| {
            if app.frame() == 2 {
                app.request_exit(ExitReason::Requested)
                    .expect("request exit");
            }
        })
        .expect("headless run");
    assert_eq!(report.frames, 2);
    assert_eq!(report.snapshot.frame, 2);
    assert_eq!(
        app.exit_status(),
        ExitStatus::Completed(ExitReason::Requested)
    );
}

#[test]
fn window_budgets_and_descriptors_are_validated_before_start() {
    assert!(matches!(
        EditorApplication::new(EditorAppConfig {
            max_windows: 0,
            ..EditorAppConfig::default()
        }),
        Err(EditorAppError::InvalidWindowLimits)
    ));
    let mut app = EditorApplication::new(EditorAppConfig {
        max_windows: 1,
        ..EditorAppConfig::default()
    })
    .expect("app");
    assert!(app
        .configure_window(WindowDescriptor::new(
            WindowId::from_raw(1),
            "Invalid",
            PixelSize {
                width: 0,
                height: 1
            },
        ))
        .is_err());
    assert_eq!(
        app.configure_window(WindowDescriptor::new(
            WindowId::from_raw(2),
            "Second",
            PixelSize::new(640, 480).expect("size"),
        )),
        Err(EditorAppError::WindowLimitExceeded { limit: 1 })
    );
    assert_eq!(app.windows().count(), 1);
}

#[test]
fn duplicate_document_rejection_retains_original_state() {
    let id = DocumentId::from_raw(7);
    let mut original = DocumentState::new(id);
    original
        .add_layer(prism_editor_core::DocumentLayer::new(
            prism_editor_core::LayerId::from_raw(1),
            "Original",
            prism_editor_core::LayerKind::Base,
        ))
        .expect("layer");
    let mut app = EditorApplication::new(EditorAppConfig::default()).expect("app");
    app.insert_document(original.clone()).expect("insert");
    assert_eq!(
        app.insert_document(DocumentState::new(id)),
        Err(EditorAppError::DuplicateDocument(id))
    );
    assert_eq!(app.remove_document(id).expect("original"), original);
}
