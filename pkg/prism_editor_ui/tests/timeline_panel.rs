//! Semantic authoring, integer precision, actual retained geometry and bounds.
use prism_editor_timeline::*;
use prism_editor_ui::{
    ActionValue, SelectionMode, TimelineKeySelection, TimelinePanelAction as A,
    TimelinePanelController, TimelinePanelError as E, TimelineWindow, UiEvent,
};
use prism_ui::{
    backend::RecordingBackend,
    layout::{AvailableSpace, Size},
    Element, Key, Ui,
};
use prism_ui_render_backend::{rasterize, RetainedScene};

fn fixture() -> TimelineEditor {
    let mut timeline = Timeline::default();
    let mut track = Track::new(TrackId::from_raw(1), "Opacity", 1).unwrap();
    for (id, tick, value) in [(1, 0, 0.0), (2, 10, 1.0), (3, 30, 2.0)] {
        track
            .insert(Keyframe::new(
                KeyId::from_raw(id),
                tick,
                vec![value],
                Interpolation::Linear,
            ))
            .unwrap();
    }
    timeline.add_track(track).unwrap();
    TimelineEditor::new(timeline, TimelineLimits::default()).unwrap()
}
fn target(key: u128) -> TimelineKeySelection {
    TimelineKeySelection {
        track: TrackId::from_raw(1),
        key: KeyId::from_raw(key),
    }
}
fn send(
    c: &mut TimelinePanelController,
    e: &mut TimelineEditor,
    key: &str,
    event: UiEvent,
) -> Result<(), E> {
    let view = c.build(e)?;
    c.handle_event(e, &view, key, &event)
}
fn text(value: &str) -> UiEvent {
    UiEvent::Commit {
        value: ActionValue::text(value).unwrap(),
    }
}
fn select(c: &mut TimelinePanelController, e: &mut TimelineEditor, id: u128, mode: SelectionMode) {
    c.apply(
        e,
        A::Select {
            target: target(id),
            mode,
        },
    )
    .unwrap();
}

#[test]
fn bound_creation_fields_produce_tracks_keys_and_events_with_undo() {
    let mut e = TimelineEditor::new(Timeline::default(), TimelineLimits::default()).unwrap();
    let mut c = TimelinePanelController::default();
    send(
        &mut c,
        &mut e,
        "timeline:track-name",
        text("Camera position"),
    )
    .unwrap();
    send(&mut c, &mut e, "timeline:track-dimension", text("3")).unwrap();
    send(&mut c, &mut e, "timeline:add-track", UiEvent::Activate).unwrap();
    send(&mut c, &mut e, "timeline:seek", text("24")).unwrap();
    send(&mut c, &mut e, "timeline:insert-key:1", UiEvent::Activate).unwrap();
    assert_eq!(
        c.selection(),
        &std::collections::BTreeSet::from([target(1)])
    );
    send(&mut c, &mut e, "timeline:key-values", text("1, 2, 3")).unwrap();
    assert_eq!(
        c.samples(&e).unwrap()[&TrackId::from_raw(1)],
        [1.0, 2.0, 3.0]
    );
    send(&mut c, &mut e, "timeline:event-name", text("camera cut")).unwrap();
    send(&mut c, &mut e, "timeline:add-event", UiEvent::Activate).unwrap();
    assert_eq!(e.timeline().events_at(24)[0].name, "camera cut");
    send(&mut c, &mut e, "timeline:delete-event:1", UiEvent::Activate).unwrap();
    send(&mut c, &mut e, "timeline:undo", UiEvent::Activate).unwrap();
    assert_eq!(e.timeline().event_count(), 1);
    send(&mut c, &mut e, "timeline:delete-track:1", UiEvent::Activate).unwrap();
    assert_eq!(e.timeline().track_count(), 0);
    send(&mut c, &mut e, "timeline:undo", UiEvent::Activate).unwrap();
    assert_eq!(
        e.timeline().track(TrackId::from_raw(1)).unwrap().name(),
        "Camera position"
    );
    assert_eq!(
        e.timeline().sample(24).unwrap()[&TrackId::from_raw(1)],
        [1.0, 2.0, 3.0]
    );
}

#[test]
fn selected_curve_fields_change_actual_samples_and_reversible_history() {
    let mut e = fixture();
    let mut c = TimelinePanelController::default();
    send(
        &mut c,
        &mut e,
        "timeline:key:1:1",
        UiEvent::Select {
            mode: SelectionMode::Replace,
        },
    )
    .unwrap();
    send(
        &mut c,
        &mut e,
        "timeline:interpolation:cubic",
        UiEvent::Activate,
    )
    .unwrap();
    send(&mut c, &mut e, "timeline:outgoing", text("0.2")).unwrap();
    c.apply(&mut e, A::Seek(5)).unwrap();
    assert_eq!(c.samples(&e).unwrap()[&TrackId::from_raw(1)], [0.75]);
    let view = c.build(&e).unwrap();
    assert_eq!(view.curve().len(), 128);
    assert!(view
        .curve()
        .iter()
        .all(|p| p.x.is_finite() && (4.0..=92.0).contains(&p.y)));
    send(&mut c, &mut e, "timeline:key-values", text("2")).unwrap();
    send(&mut c, &mut e, "timeline:undo", UiEvent::Activate).unwrap();
    assert_eq!(c.samples(&e).unwrap()[&TrackId::from_raw(1)], [0.75]);
    send(&mut c, &mut e, "timeline:redo", UiEvent::Activate).unwrap();
    assert_eq!(c.samples(&e).unwrap()[&TrackId::from_raw(1)], [1.75]);
    let token = e.snapshot_token();
    assert!(send(&mut c, &mut e, "timeline:key-values", text("NaN")).is_err());
    assert!(send(&mut c, &mut e, "timeline:outgoing", text("1,2")).is_err());
    assert_eq!(e.snapshot_token(), token);
}

#[test]
fn group_moves_use_vacated_ticks_and_collisions_roll_back_every_key() {
    let mut e = fixture();
    let mut c = TimelinePanelController::default();
    select(&mut c, &mut e, 1, SelectionMode::Replace);
    select(&mut c, &mut e, 2, SelectionMode::Extend);
    c.apply(&mut e, A::MoveSelection(10)).unwrap();
    assert_eq!(
        e.timeline()
            .track(TrackId::from_raw(1))
            .unwrap()
            .keyframes()
            .map(Keyframe::tick)
            .collect::<Vec<_>>(),
        [10, 20, 30]
    );
    let before = e.timeline().clone();
    let token = e.snapshot_token();
    assert_eq!(
        c.apply(&mut e, A::MoveSelection(10)),
        Err(E::Timeline(TimelineError::DuplicateTick(30)))
    );
    assert_eq!(e.timeline(), &before);
    assert_eq!(e.snapshot_token(), token);
    c.apply(&mut e, A::Undo).unwrap();
    assert_eq!(
        e.timeline()
            .track(TrackId::from_raw(1))
            .unwrap()
            .keyframes()
            .map(Keyframe::tick)
            .collect::<Vec<_>>(),
        [0, 10, 30]
    );
    assert_eq!(
        c.apply(&mut e, A::MoveSelection(i64::MIN)),
        Err(E::InvalidValue)
    );
}

#[test]
fn exact_view_identity_rejects_old_controller_and_independent_documents() {
    let mut e = fixture();
    let mut c = TimelinePanelController::default();
    let view = c.build(&e).unwrap();
    let mut other = TimelinePanelController::default();
    assert_eq!(
        other.handle_event(&mut e, &view, "timeline:seek", &text("4")),
        Err(E::StaleView)
    );
    let mut another = fixture();
    assert_eq!(
        c.handle_event(&mut another, &view, "timeline:seek", &text("4")),
        Err(E::StaleView)
    );
    c.apply(&mut e, A::Seek(1)).unwrap();
    assert_eq!(
        c.handle_event(&mut e, &view, "timeline:seek", &text("4")),
        Err(E::StaleView)
    );
    let view = c.build(&e).unwrap();
    e.execute(
        "outside",
        vec![TimelineEdit::RemoveKey {
            track: TrackId::from_raw(1),
            key: KeyId::from_raw(1),
        }],
    )
    .unwrap();
    assert_eq!(
        c.handle_event(&mut e, &view, "timeline:seek", &text("4")),
        Err(E::StaleView)
    );
}

#[test]
fn bound_transport_delivers_real_samples_and_ordered_loop_events_without_authoring_writes() {
    use std::time::Duration;
    let mut e = fixture();
    e.execute(
        "events",
        vec![
            TimelineEdit::AddEvent(TimelineEvent {
                id: 1,
                tick: 0,
                name: "start".into(),
            }),
            TimelineEdit::AddEvent(TimelineEvent {
                id: 2,
                tick: 5,
                name: "middle".into(),
            }),
            TimelineEdit::AddEvent(TimelineEvent {
                id: 3,
                tick: 10,
                name: "end".into(),
            }),
        ],
    )
    .unwrap();
    let token = e.snapshot_token();
    let history = e.undo_len();
    let mut c = TimelinePanelController::default();
    send(&mut c, &mut e, "timeline:frequency", text("10")).unwrap();
    send(&mut c, &mut e, "timeline:playback-end", text("10")).unwrap();
    send(&mut c, &mut e, "timeline:loop", UiEvent::Activate).unwrap();
    send(&mut c, &mut e, "timeline:play", UiEvent::Activate).unwrap();
    let before = c.build(&e).unwrap();
    let frame = c.advance(&e, Duration::from_millis(1200)).unwrap();
    assert_eq!(
        (frame.tick, frame.iteration, frame.status),
        (1, 1, PlaybackStatus::Playing)
    );
    assert!((frame.samples[&TrackId::from_raw(1)][0] - 0.1).abs() < 1e-12);
    assert_eq!(
        frame
            .events
            .iter()
            .map(|e| (e.event.name.as_str(), e.iteration))
            .collect::<Vec<_>>(),
        [("start", 0), ("middle", 0), ("end", 0), ("start", 1)]
    );
    assert_eq!(
        c.handle_event(&mut e, &before, "timeline:pause", &UiEvent::Activate),
        Err(E::StaleView)
    );
    send(&mut c, &mut e, "timeline:pause", UiEvent::Activate).unwrap();
    let paused = c.build(&e).unwrap();
    assert!(c
        .advance(&e, Duration::from_secs(60))
        .unwrap()
        .events
        .is_empty());
    assert_eq!(c.playhead(), 1);
    c.handle_event(&mut e, &paused, "timeline:stop", &UiEvent::Activate)
        .unwrap();
    assert_eq!(c.playhead(), 0);
    assert_eq!(c.playback().status(), PlaybackStatus::Paused);
    assert_eq!(e.snapshot_token(), token);
    assert_eq!(e.undo_len(), history);
}

#[test]
fn transport_failures_preserve_view_time_event_cursor_and_document() {
    use std::time::Duration;
    let mut e = fixture();
    e.execute(
        "event",
        vec![TimelineEdit::AddEvent(TimelineEvent {
            id: 1,
            tick: 5,
            name: "crossed".into(),
        })],
    )
    .unwrap();
    let mut c = TimelinePanelController::default();
    c.apply(
        &mut e,
        A::PlaybackConfig(PlaybackConfig {
            ticks_per_second: 10,
            start_tick: 0,
            end_tick: 10,
            mode: PlaybackMode::Loop,
            max_events: 1,
        }),
    )
    .unwrap();
    send(&mut c, &mut e, "timeline:play", UiEvent::Activate).unwrap();
    let before = c.playback().clone();
    let token = e.snapshot_token();
    let view = c.build(&e).unwrap();
    assert_eq!(
        c.advance(&e, Duration::from_secs(3)),
        Err(E::Playback(PlaybackError::EventBudgetExceeded { limit: 1 }))
    );
    assert_eq!(c.playback(), &before);
    assert_eq!(e.snapshot_token(), token);
    assert!(matches!(
        c.apply(&mut e, A::Seek(11)),
        Err(E::Playback(PlaybackError::SeekOutOfBounds))
    ));
    assert!(send(&mut c, &mut e, "timeline:frequency", text("0")).is_err());
    assert!(send(&mut c, &mut e, "timeline:rate-denominator", text("0")).is_err());
    assert_eq!(c.playback(), &before);
    c.handle_event(&mut e, &view, "timeline:pause", &UiEvent::Activate)
        .unwrap();
    send(&mut c, &mut e, "timeline:play", UiEvent::Activate).unwrap();
    let frame = c.advance(&e, Duration::from_millis(500)).unwrap();
    assert_eq!(frame.events.len(), 1);
    assert_eq!(frame.events[0].event.name, "crossed");
    assert_eq!(frame.tick, 5);
}

#[test]
fn bound_rate_configuration_stop_and_one_shot_finish_have_exact_semantics() {
    use std::time::Duration;
    let mut e = fixture();
    let mut c = TimelinePanelController::default();
    send(&mut c, &mut e, "timeline:frequency", text("10")).unwrap();
    send(&mut c, &mut e, "timeline:playback-end", text("10")).unwrap();
    send(&mut c, &mut e, "timeline:playback-start", text("2")).unwrap();
    assert_eq!(c.playhead(), 2);
    send(&mut c, &mut e, "timeline:rate-numerator", text("3")).unwrap();
    send(&mut c, &mut e, "timeline:rate-denominator", text("2")).unwrap();
    send(&mut c, &mut e, "timeline:play", UiEvent::Activate).unwrap();
    assert_eq!(c.advance(&e, Duration::from_millis(100)).unwrap().tick, 3);
    let fraction = c.playback().fractional_tick();
    assert_ne!(fraction.0, 0);
    send(&mut c, &mut e, "timeline:pause", UiEvent::Activate).unwrap();
    c.advance(&e, Duration::from_secs(10)).unwrap();
    assert_eq!(c.playback().fractional_tick(), fraction);
    send(&mut c, &mut e, "timeline:play", UiEvent::Activate).unwrap();
    let last = c.advance(&e, Duration::from_secs(1)).unwrap();
    assert_eq!((last.tick, last.status), (10, PlaybackStatus::Finished));
    assert_eq!(
        send(&mut c, &mut e, "timeline:play", UiEvent::Activate),
        Err(E::UnboundEvent)
    );
    send(&mut c, &mut e, "timeline:stop", UiEvent::Activate).unwrap();
    assert_eq!(
        (c.playhead(), c.playback().status()),
        (2, PlaybackStatus::Paused)
    );
    assert_eq!(c.playback().fractional_tick().0, 0);
    send(&mut c, &mut e, "timeline:play", UiEvent::Activate).unwrap();
}

fn keys(element: &Element, output: &mut Vec<Option<String>>) {
    output.push(match element.explicit_key() {
        Some(Key::Str(key)) => Some(key.clone()),
        _ => None,
    });
    for child in element.child_elements() {
        keys(child, output);
    }
}
#[test]
fn large_integer_ticks_project_to_distinct_real_layout_and_stable_tree() {
    let start = u64::MAX - 100;
    let mut timeline = Timeline::default();
    let mut track = Track::new(TrackId::from_raw(1), "Large ticks", 1).unwrap();
    for (id, offset) in [(1, 1), (2, 2)] {
        track
            .insert(Keyframe::new(
                KeyId::from_raw(id),
                start + offset,
                vec![offset as f64],
                Interpolation::Linear,
            ))
            .unwrap();
    }
    timeline.add_track(track).unwrap();
    let mut e = TimelineEditor::new(timeline, TimelineLimits::default()).unwrap();
    let mut c = TimelinePanelController::default();
    c.apply(
        &mut e,
        A::Window(TimelineWindow {
            start,
            span: 10,
            width: 640.0,
            ..TimelineWindow::default()
        }),
    )
    .unwrap();
    select(&mut c, &mut e, 1, SelectionMode::Replace);
    let view = c.build(&e).unwrap();
    let mut names = Vec::new();
    keys(view.element(), &mut names);
    let mut ui = Ui::new(RetainedScene::new());
    ui.mount(view.element());
    ui.compute_layout(Size::new(
        AvailableSpace::Definite(1800.0),
        AvailableSpace::MaxContent,
    ));
    let mut rects = Vec::new();
    ui.backend().visit_layout(|_, rect| rects.push(rect));
    assert_eq!(rects.len(), names.len());
    let keyed: std::collections::BTreeMap<_, _> = names
        .into_iter()
        .zip(rects)
        .filter_map(|(k, r)| k.map(|k| (k, r)))
        .collect();
    let lane = keyed["timeline:lane:1"];
    assert!((keyed["timeline:key:1:1"].left() - lane.left() - 64.0).abs() < 0.1);
    assert!((keyed["timeline:key:1:2"].left() - lane.left() - 128.0).abs() < 0.1);
    assert!(!ui.backend().to_draw_list().is_empty());
    let image = rasterize(&ui.backend().to_draw_list(), 1800, 1400);
    let key_rect = keyed["timeline:key:1:1"];
    let selected_pixel = image.pixel(
        (key_rect.left() + 4.0) as u32,
        (key_rect.top() + 6.0) as u32,
    );
    assert!(
        (selected_pixel[0] - 59.0 / 255.0).abs() < 0.01,
        "{selected_pixel:?}"
    );
    assert!(
        (selected_pixel[2] - 246.0 / 255.0).abs() < 0.01,
        "{selected_pixel:?}"
    );
    let curve_rect = keyed["timeline:curve:64"];
    assert!(curve_rect.top() + 2.0 < 1400.0);
    let curve_pixel = image.pixel(
        (curve_rect.left() + 1.0) as u32,
        (curve_rect.top() + 1.0) as u32,
    );
    assert!(
        (curve_pixel[2] - 246.0 / 255.0).abs() < 0.01,
        "{curve_pixel:?}"
    );
    let key_rect = keyed["timeline:key:1:2"];
    let unselected_pixel = image.pixel(
        (key_rect.left() + 4.0) as u32,
        (key_rect.top() + 6.0) as u32,
    );
    assert!((unselected_pixel[2] - selected_pixel[2]).abs() > 0.2);
    let mut recorder = Ui::new(RecordingBackend::new());
    recorder.mount(view.element());
    recorder.backend_mut().clear();
    recorder.update(c.build(&e).unwrap().element());
    assert!(recorder.backend().ops().is_empty());
}

#[test]
fn marker_track_event_and_selection_work_are_bounded() {
    let mut timeline = Timeline::default();
    for id in 1..=40 {
        let mut track = Track::new(TrackId::from_raw(id), "track", 1).unwrap();
        for tick in 0..2000 {
            track
                .insert(Keyframe::new(
                    KeyId::from_raw(u128::from(tick)),
                    tick,
                    vec![tick as f64],
                    Interpolation::Linear,
                ))
                .unwrap();
        }
        timeline.add_track(track).unwrap();
    }
    for id in 0..200 {
        timeline
            .add_event(TimelineEvent {
                id,
                tick: id as u64,
                name: "event".into(),
            })
            .unwrap();
    }
    let mut e = TimelineEditor::new(timeline, TimelineLimits::default()).unwrap();
    let mut c = TimelinePanelController::default();
    c.apply(
        &mut e,
        A::Window(TimelineWindow {
            span: 2000,
            rows: 32,
            ..TimelineWindow::default()
        }),
    )
    .unwrap();
    let view = c.build(&e).unwrap();
    assert_eq!(view.visible_keys(), 1024);
    assert!(view.truncated());
    assert_eq!(
        view.bindings()
            .filter(|(k, _)| k.starts_with("timeline:delete-track:"))
            .count(),
        32
    );
    assert_eq!(
        view.bindings()
            .filter(|(k, _)| k.starts_with("timeline:delete-event:"))
            .count(),
        128
    );
    select(&mut c, &mut e, 0, SelectionMode::Replace);
    assert_eq!(
        c.apply(
            &mut e,
            A::Select {
                target: target(1999),
                mode: SelectionMode::Extend
            }
        ),
        Err(E::InvalidValue)
    );
    assert_eq!(c.selection().len(), 1);
    send(&mut c, &mut e, "timeline:next-tracks", UiEvent::Activate).unwrap();
    assert_eq!(c.window().first_track.raw(), 33);
    send(
        &mut c,
        &mut e,
        "timeline:previous-tracks",
        UiEvent::Activate,
    )
    .unwrap();
    assert_eq!(c.window().first_track.raw(), 1);
}
