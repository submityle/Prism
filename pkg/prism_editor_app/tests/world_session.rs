//! Real cell-file IO through the native host, without runtime consumers.
//! These tests deliberately do not claim ECS/OFPA attachment or transactions.
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use prism_asset::StableGuid;
use prism_ecs::partition::{cell::CellCoord, data_layer::DataLayerId};
use prism_editor_app::{
    DesktopPanelUpdate, WorldProjectSession, WorldProjectSessionLimits,
    MAX_WORLD_SESSION_WORK_PER_POLL,
};
use prism_editor_ui::{
    WorldPanelAction as A, WorldPanelController as Controller, WorldPanelIntentKind,
    WorldPanelLimits, WorldPanelViewport,
};
use prism_editor_world::{
    project::ProjectEdit, CellResidency, DataLayer, EditorWorldData, EditorWorldLimits,
    IoCompletion,
};

fn cell(x: i32) -> CellCoord {
    CellCoord::new(x, 0, 0)
}

fn controller(count: usize) -> Controller {
    let mut world = EditorWorldData::new(
        StableGuid::from_path("world/native-session-tests"),
        EditorWorldLimits::new(count.max(8), 4096, 64, 4, 64).unwrap(),
    )
    .unwrap();
    world
        .add_layer(DataLayer::new(DataLayerId::new(701), "test layer").unwrap())
        .unwrap();
    for x in 0..count {
        world
            .add_cell(cell(x as i32), [DataLayerId::new(701)])
            .unwrap();
    }
    Controller::new(
        world,
        StableGuid::from_path("user/native-session-tests"),
        WorldPanelViewport {
            size: [128., 64.],
            ..Default::default()
        },
        WorldPanelLimits::default(),
    )
    .unwrap()
}

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "prism-native-world-session-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn dirty(controller: &Controller) -> bool {
    controller.world().model().cell(cell(0)).unwrap().is_dirty()
}

// Establish resident authoring bytes using the real transition API. This is
// test setup, not a fake filesystem completion used to exercise the session.
fn editable(controller: &mut Controller, bytes: &[u8]) {
    controller.apply(A::Select(cell(0))).unwrap();
    controller.apply(A::Load).unwrap();
    let intent = controller.take_intent().unwrap();
    let transition = controller.begin_residency(&intent).unwrap();
    controller
        .complete_load(transition, b"initial".to_vec())
        .unwrap();
    controller.apply(A::AcquireLease).unwrap();
    controller.replace_payload(cell(0), bytes.to_vec()).unwrap();
}

fn session(temp: &Temp, controller: &Controller, now: Instant) -> WorldProjectSession {
    WorldProjectSession::new_detached(
        &temp.0,
        controller,
        WorldProjectSessionLimits::default(),
        now,
    )
    .unwrap()
}

// Only test code waits. Host time is deliberately fixed and independent from
// worker scheduling; a real wall-clock deadline bounds a broken test worker.
fn settle(
    session: &mut WorldProjectSession,
    controller: &mut Controller,
    now: Instant,
) -> Result<DesktopPanelUpdate, String> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let update = session.poll(controller, now)?;
        if !session.is_busy() && controller.pending_intents() == 0 {
            return Ok(update);
        }
        assert!(
            Instant::now() < deadline,
            "cell IO did not reach a terminal result"
        );
        std::thread::yield_now();
    }
}

#[test]
fn actual_file_save_independent_session_load_and_detached_unload() {
    let temp = Temp::new();
    let now = Instant::now();
    let mut writer = controller(1);
    editable(&mut writer, b"saved opaque authoring bytes");
    let mut write = session(&temp, &writer, now);
    writer.apply(A::Save).unwrap();
    let submitted = write.poll(&mut writer, now).unwrap();
    assert!(submitted.changed && submitted.fenced && submitted.wake_at.is_some());
    assert!(write.pending_request().is_some());
    assert!(dirty(&writer), "submission alone must not clear dirty");
    settle(&mut write, &mut writer, now).unwrap();
    assert!(!dirty(&writer));
    assert_eq!(write.work().requests_submitted, 1);
    assert_eq!(write.work().completions_received, 1);
    let encoded = std::fs::read(temp.0.join("cells/0_0_0.prismcell")).unwrap();
    assert!(encoded.len() > b"saved opaque authoring bytes".len());

    let mut reader = controller(1);
    let mut read = session(&temp, &reader, now);
    reader.apply(A::Select(cell(0))).unwrap();
    reader.apply(A::Load).unwrap();
    settle(&mut read, &mut reader, now).unwrap();
    assert_eq!(
        reader.world().payload(cell(0)),
        Some(&b"saved opaque authoring bytes"[..])
    );
    assert!(!dirty(&reader));
    assert_eq!(
        reader.world().model().cell(cell(0)).unwrap().revision(),
        writer.world().model().cell(cell(0)).unwrap().revision()
    );
    reader.apply(A::Unload).unwrap();
    let update = settle(&mut read, &mut reader, now).unwrap();
    assert!(update.changed);
    assert!(
        update.fenced && reader.project_is_dirty(),
        "cell IO does not invent a durable project-membership receipt"
    );
    assert_eq!(reader.world().payload(cell(0)), None);
    assert_eq!(
        reader.world().model().cell(cell(0)).unwrap().residency(),
        CellResidency::Unloaded
    );
    assert_eq!(read.work().unloads_completed, 1);
    assert_eq!(
        read.work().requests_submitted,
        1,
        "unload needs no invented disk IO"
    );
}

#[test]
fn filesystem_save_failure_retains_snapshot_receipt_and_dirty_then_explicit_retry() {
    let temp = Temp::new();
    std::fs::write(temp.0.join("cells"), b"not a directory").unwrap();
    let now = Instant::now();
    let mut c = controller(1);
    editable(&mut c, b"must survive failed save");
    let mut s = session(&temp, &c, now);
    c.apply(A::Save).unwrap();
    assert!(settle(&mut s, &mut c, now).is_err());
    assert!(dirty(&c));
    assert!(!s.is_busy());
    let failure = s.failure().unwrap();
    assert_eq!(failure.intent.kind(), WorldPanelIntentKind::Save(cell(0)));
    assert_eq!(
        failure.save.as_ref().unwrap().payload(),
        b"must survive failed save"
    );
    assert!(matches!(
        failure.completion.as_ref(),
        Some(IoCompletion::Failed { .. })
    ));
    assert!(
        failure.reconciled,
        "real failure was delivered without cleaning"
    );
    let work = s.work();
    assert!(s.poll(&mut c, now).is_err());
    assert_eq!(
        s.work(),
        work,
        "failure fence must not resubmit automatically"
    );

    // Explicit repair of this test's own path and transfer of failure ownership.
    std::fs::remove_file(temp.0.join("cells")).unwrap();
    std::fs::create_dir(temp.0.join("cells")).unwrap();
    let retained = s.take_failure().unwrap();
    assert!(retained.save.is_some());
    c.apply(A::Save).unwrap();
    settle(&mut s, &mut c, now).unwrap();
    assert!(!dirty(&c));
    assert_eq!(s.work().requests_submitted, 2);
}

#[test]
fn late_saved_snapshot_does_not_clean_newer_edits_and_keeps_durable_proof() {
    let temp = Temp::new();
    let now = Instant::now();
    let mut c = controller(1);
    editable(&mut c, b"snapshot A");
    let mut s = session(&temp, &c, now);
    c.apply(A::Save).unwrap();
    s.poll(&mut c, now).unwrap(); // Submitted, never reconciled in this poll.
    c.replace_payload(cell(0), b"newer B".to_vec()).unwrap();
    assert!(settle(&mut s, &mut c, now).is_err());
    assert!(dirty(&c));
    assert_eq!(c.world().payload(cell(0)), Some(&b"newer B"[..]));
    let failure = s.failure().unwrap();
    assert!(matches!(
        failure.completion.as_ref(),
        Some(IoCompletion::Saved { .. })
    ));
    assert_eq!(failure.save.as_ref().unwrap().payload(), b"snapshot A");
    assert!(!failure.reconciled);

    let mut reader = controller(1);
    let mut read = session(&temp, &reader, now);
    reader.apply(A::Select(cell(0))).unwrap();
    reader.apply(A::Load).unwrap();
    settle(&mut read, &mut reader, now).unwrap();
    assert_eq!(reader.world().payload(cell(0)), Some(&b"snapshot A"[..]));

    // The real successful A receipt advanced the controller's persisted base;
    // an explicit new save can publish B without losing the dirty edit.
    let _proof = s.take_failure().unwrap();
    c.apply(A::Save).unwrap();
    settle(&mut s, &mut c, now).unwrap();
    assert!(!dirty(&c));
}

#[test]
fn missing_load_rolls_back_with_original_transition_and_failure_record() {
    let temp = Temp::new();
    let now = Instant::now();
    let mut c = controller(1);
    let mut s = session(&temp, &c, now);
    c.apply(A::Select(cell(0))).unwrap();
    c.apply(A::Load).unwrap();
    s.poll(&mut c, now).unwrap();
    assert!(matches!(
        c.world().model().cell(cell(0)).unwrap().residency(),
        CellResidency::Loading(_)
    ));
    assert!(settle(&mut s, &mut c, now).is_err());
    assert_eq!(
        c.world().model().cell(cell(0)).unwrap().residency(),
        CellResidency::Unloaded
    );
    let failure = s.take_failure().unwrap();
    assert!(failure.transition.is_some() && failure.reconciled);
    assert!(matches!(
        failure.completion,
        Some(IoCompletion::Failed { .. })
    ));
    assert_eq!(c.world().payload(cell(0)), None);
}

#[test]
fn byte_budget_rejects_before_snapshot_submission_without_cleaning() {
    let temp = Temp::new();
    let now = Instant::now();
    let mut c = controller(1);
    editable(&mut c, b"larger than four bytes");
    let limits = WorldProjectSessionLimits {
        max_cell_bytes: 4,
        ..Default::default()
    };
    let mut s = WorldProjectSession::new_detached(&temp.0, &c, limits, now).unwrap();
    c.apply(A::Save).unwrap();
    assert!(s.poll(&mut c, now).unwrap_err().contains("payload budget"));
    assert_eq!(s.work().intents_taken, 1);
    assert_eq!(s.work().requests_submitted, 0);
    assert!(s.failure().unwrap().save.is_none());
    assert!(dirty(&c));
}

#[test]
fn invalid_admission_limits_are_rejected_without_taking_queued_work() {
    let temp = Temp::new();
    let now = Instant::now();
    let mut c = controller(2);
    c.apply(A::Select(cell(0))).unwrap();
    c.apply(A::Load).unwrap();
    let defaults = WorldProjectSessionLimits::default();
    for limits in [
        WorldProjectSessionLimits {
            max_cell_bytes: 0,
            ..defaults
        },
        WorldProjectSessionLimits {
            max_cell_bytes: usize::MAX,
            ..defaults
        },
        WorldProjectSessionLimits {
            max_cells: 0,
            ..defaults
        },
        WorldProjectSessionLimits {
            max_cells: 1,
            ..defaults
        },
        WorldProjectSessionLimits {
            max_cells: 65537,
            ..defaults
        },
        WorldProjectSessionLimits {
            work_per_poll: 0,
            ..defaults
        },
        WorldProjectSessionLimits {
            work_per_poll: MAX_WORLD_SESSION_WORK_PER_POLL + 1,
            ..defaults
        },
        WorldProjectSessionLimits {
            poll_interval: Duration::ZERO,
            ..defaults
        },
        WorldProjectSessionLimits {
            poll_interval: Duration::from_secs(2),
            ..defaults
        },
        WorldProjectSessionLimits {
            operation_timeout: Duration::ZERO,
            ..defaults
        },
        WorldProjectSessionLimits {
            operation_timeout: Duration::from_secs(3601),
            ..defaults
        },
        WorldProjectSessionLimits {
            clock_tick: Duration::ZERO,
            ..defaults
        },
        WorldProjectSessionLimits {
            clock_tick: Duration::from_secs(61),
            ..defaults
        },
    ] {
        assert!(
            WorldProjectSession::new_detached(&temp.0, &c, limits, now).is_err(),
            "{limits:?}"
        );
        assert_eq!(c.pending_intents(), 1);
    }
    assert!(WorldProjectSession::new_detached(temp.0.join("absent"), &c, defaults, now).is_err());
}

#[test]
fn work_budget_and_stale_queued_epoch_are_observable_not_silently_discarded() {
    let temp = Temp::new();
    let now = Instant::now();
    let mut c = controller(1);
    editable(&mut c, b"serial saves");
    let limits = WorldProjectSessionLimits {
        work_per_poll: 1,
        ..Default::default()
    };
    let mut s = WorldProjectSession::new_detached(&temp.0, &c, limits, now).unwrap();
    c.apply(A::Save).unwrap();
    c.apply(A::Save).unwrap();
    s.poll(&mut c, now).unwrap();
    assert_eq!(s.work().intents_taken, 1);
    assert_eq!(s.work().requests_submitted, 1);
    assert_eq!(c.pending_intents(), 1);
    let deadline = Instant::now() + Duration::from_secs(5);
    while s.is_busy() {
        let before = s.work();
        s.poll(&mut c, now).unwrap();
        let after = s.work();
        assert!(
            after.intents_taken - before.intents_taken + after.completions_received
                - before.completions_received
                <= 1
        );
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(
        c.pending_intents(),
        1,
        "terminal delivery consumed this poll's work budget"
    );
    assert!(
        settle(&mut s, &mut c, now).is_err(),
        "old queued epoch must not become a fresh save"
    );
    let failure = s.take_failure().unwrap();
    assert_eq!(failure.intent.kind(), WorldPanelIntentKind::Save(cell(0)));
    assert_eq!(s.work().requests_submitted, 1);
    assert_eq!(s.work().intents_taken, 2);
    assert!(!dirty(&c));
}

#[test]
fn backwards_and_overflow_clock_fail_before_dequeue() {
    let temp = Temp::new();
    let now = Instant::now();
    let mut c = controller(1);
    let limits = WorldProjectSessionLimits {
        clock_tick: Duration::from_nanos(1),
        ..Default::default()
    };
    let mut s = WorldProjectSession::new_detached(&temp.0, &c, limits, now).unwrap();
    c.apply(A::Select(cell(0))).unwrap();
    c.apply(A::Load).unwrap();
    assert!(s
        .poll(&mut c, now - Duration::from_nanos(1))
        .unwrap_err()
        .contains("backwards"));
    if let Some(far_future) = now.checked_add(Duration::from_secs(20_000_000_000)) {
        assert!(s.poll(&mut c, far_future).unwrap_err().contains("overflow"));
    }
    assert_eq!(s.work().intents_taken, 0);
    assert_eq!(c.pending_intents(), 1);
    assert!(!s.is_busy());
}

#[test]
fn advancing_controller_lease_clock_reports_view_change_without_io() {
    let temp = Temp::new();
    let now = Instant::now();
    let mut c = controller(1);
    let mut s = session(&temp, &c, now);
    assert!(!s.poll(&mut c, now).unwrap().changed);
    assert!(
        s.poll(&mut c, now + Duration::from_secs(1))
            .unwrap()
            .changed
    );
    assert!(
        !s.poll(&mut c, now + Duration::from_secs(1))
            .unwrap()
            .changed
    );
    assert_eq!(s.work().requests_submitted, 0);
}

#[test]
fn deadline_is_nonblocking_observation_and_never_fabricates_terminal_success() {
    let temp = Temp::new();
    let now = Instant::now();
    let mut c = controller(1);
    editable(&mut c, b"real late save");
    let limits = WorldProjectSessionLimits {
        operation_timeout: Duration::from_millis(5),
        ..Default::default()
    };
    let mut s = WorldProjectSession::new_detached(&temp.0, &c, limits, now).unwrap();
    c.apply(A::Save).unwrap();
    s.poll(&mut c, now).unwrap();
    assert_eq!(s.work().completions_received, 0);
    assert!(dirty(&c));
    let late = now + limits.operation_timeout;
    let update = s.poll(&mut c, late).unwrap();
    assert_eq!(s.work().deadlines_exceeded, 1);
    if s.is_busy() {
        assert!(s.deadline_exceeded());
        assert!(dirty(&c));
        assert!(update.fenced && update.wake_at.is_some());
    }
    settle(&mut s, &mut c, late).unwrap();
    assert_eq!(s.work().deadlines_exceeded, 1);
    assert_eq!(s.work().completions_received, 1);
    assert!(!dirty(&c));
    assert!(temp.0.join("cells/0_0_0.prismcell").is_file());
}

#[test]
fn actual_save_after_lease_expiry_retains_dirty_and_successful_disk_receipt() {
    let temp = Temp::new();
    let now = Instant::now();
    let mut c = controller(1);
    editable(&mut c, b"persisted but lease expired");
    let mut s = session(&temp, &c, now);
    c.apply(A::Save).unwrap();
    s.poll(&mut c, now).unwrap();
    assert!(settle(&mut s, &mut c, now + Duration::from_secs(60)).is_err());
    assert!(dirty(&c));
    assert!(matches!(
        s.failure().unwrap().completion.as_ref(),
        Some(IoCompletion::Saved { .. })
    ));
    assert!(!s.is_busy());
    assert!(temp.0.join("cells/0_0_0.prismcell").is_file());
    let _receipt = s.take_failure().unwrap();
    c.apply(A::AcquireLease).unwrap();
    c.apply(A::Save).unwrap();
    settle(&mut s, &mut c, now + Duration::from_secs(60)).unwrap();
    assert!(!dirty(&c));
}

#[test]
fn equal_world_identity_is_not_controller_adoption_or_generation_authority() {
    let temp = Temp::new();
    let now = Instant::now();
    let mut original = controller(1);
    let mut foreign = controller(1);
    editable(&mut original, b"original generation only");
    assert_eq!(original.project().identity(), foreign.project().identity());
    assert_eq!(original.project().revision(), foreign.project().revision());
    assert!(!foreign.project().matches_token(&original.project().token()));
    let mut s = session(&temp, &original, now);
    original.apply(A::Save).unwrap();
    s.poll(&mut original, now).unwrap();
    let request = s.pending_request();
    let work = s.work();
    assert!(s
        .poll(&mut foreign, now)
        .unwrap_err()
        .contains("snapshot changed"));
    assert_eq!(s.pending_request(), request);
    assert_eq!(s.work(), work);

    s.rebind_detached(&foreign).unwrap();
    let error = settle(&mut s, &mut foreign, now).unwrap_err();
    assert!(error.contains("stale controller generation"));
    assert_eq!(
        s.pending_request(),
        request,
        "terminal remains owned, not lost on Err"
    );
    assert!(s.is_busy() && s.failure().is_none());
    assert_eq!(s.work().completions_received, 1);
    assert_eq!(foreign.world().payload(cell(0)), None);
    assert!(dirty(&original));
    s.rebind_detached(&original).unwrap();
    settle(&mut s, &mut original, now).unwrap();
    assert!(!dirty(&original));
    assert_eq!(
        s.work().completions_received,
        1,
        "reuse the retained real receipt"
    );
}

#[test]
fn project_change_and_retired_world_require_external_acknowledgement() {
    let temp = Temp::new();
    let now = Instant::now();
    let mut c = controller(1);
    let mut s = session(&temp, &c, now);
    c.apply(A::Select(cell(0))).unwrap();
    c.apply(A::Load).unwrap();
    c.apply(A::LayerVisible {
        id: DataLayerId::new(701),
        visible: false,
    })
    .unwrap();
    assert!(s
        .poll(&mut c, now)
        .unwrap_err()
        .contains("snapshot changed"));
    assert_eq!(c.pending_intents(), 1);
    assert_eq!(s.work().intents_taken, 0);

    let mut structural = controller(1);
    let mut host = session(&temp, &structural, now);
    structural
        .edit_project(&[ProjectEdit::Cell {
            cell: cell(1),
            layers: Some([DataLayerId::new(701)].into()),
        }])
        .unwrap();
    assert!(structural.has_retired_world());
    assert!(host
        .poll(&mut structural, now)
        .unwrap_err()
        .contains("retired_world"));
    assert!(host.rebind_detached(&structural).is_err());
    assert!(
        WorldProjectSession::new_detached(&temp.0, &structural, Default::default(), now).is_err()
    );
    assert!(
        structural.has_retired_world(),
        "IO host cannot assert consumer detachment"
    );
    assert_eq!(host.work().intents_taken, 0);
}

#[test]
fn cross_cell_is_explicitly_rejected_with_transferable_ticket_and_no_io() {
    let temp = Temp::new();
    let now = Instant::now();
    let mut c = controller(2);
    // Cross-cell commands require both cells to be resident and editable.
    // Establish authoring residency through the real controller transition
    // protocol before testing this session's explicit transaction rejection.
    for x in 0..2 {
        c.apply(A::Select(cell(x))).unwrap();
        c.apply(A::Load).unwrap();
        let intent = c.take_intent().unwrap();
        let transition = c.begin_residency(&intent).unwrap();
        c.complete_load(transition, b"resident payload".to_vec())
            .unwrap();
    }
    c.apply(A::Select(cell(0))).unwrap();
    c.apply(A::AcquireLease).unwrap();
    c.apply(A::CrossCellSource).unwrap();
    c.apply(A::Select(cell(1))).unwrap();
    c.apply(A::AcquireLease).unwrap();
    c.apply(A::CrossCell).unwrap();
    let mut s = session(&temp, &c, now);
    assert!(s
        .poll(&mut c, now)
        .unwrap_err()
        .contains("CrossCell rejected"));
    assert_eq!(c.pending_intents(), 0);
    let failure = s.take_failure().unwrap();
    assert_eq!(
        failure.intent.kind(),
        WorldPanelIntentKind::CrossCell {
            source: cell(0),
            destination: cell(1)
        }
    );
    assert!(failure.save.is_none() && failure.transition.is_none() && failure.completion.is_none());
    assert_eq!(s.work().requests_submitted, 0);
    assert!(!s.is_busy());
    assert!(!temp.0.join("cells").exists());
}
