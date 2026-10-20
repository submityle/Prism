//! Real CPU capture, graphical geometry/paint, navigation, and viewport bounds.

use prism_editor_profiler::{
    capture::CpuProducer, Frame, FrameId, FrameStore, ProfilerLimits, Span, SpanId,
};
use prism_editor_ui::{
    ProfilerAction as Action, ProfilerController, ProfilerDomain, ProfilerPanelError as Error,
    ProfilerPanelLimits, ProfilerSources, ProfilerTimeWindow, UiEvent,
};
use prism_ui::{
    backend::RecordingBackend,
    layout::{AvailableSpace, Rect, Size},
    Element, Key, Ui,
};
use prism_ui_render_backend::{rasterize, RetainedScene};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Duration,
};

fn span(id: u64, parent: Option<u64>, start: u64, duration: u64, lane: &str) -> Span {
    Span::new(
        SpanId::from_raw(id),
        parent.map(SpanId::from_raw),
        start,
        duration,
        format!("span {id}"),
        lane,
    )
}
fn frame(id: u64, duration: u64, spans: Vec<Span>) -> Frame {
    Frame::try_new(
        FrameId::from_raw(id),
        id,
        0,
        duration,
        spans,
        &ProfilerLimits::default(),
    )
    .unwrap()
}
fn store(frames: impl IntoIterator<Item = Frame>) -> Arc<FrameStore> {
    let mut store = FrameStore::new(ProfilerLimits::default());
    for f in frames {
        store.insert(f).unwrap();
    }
    Arc::new(store)
}
fn publish(c: &mut ProfilerController, frames: impl IntoIterator<Item = Frame>) {
    c.publish(ProfilerSources {
        cpu: Some(store(frames)),
        gpu: None,
    });
}
fn fixture() -> Frame {
    frame(
        7,
        1000,
        vec![
            span(1, None, 0, 1000, "main"),
            span(2, Some(1), 250, 250, "main"),
        ],
    )
}
fn keys(e: &Element, out: &mut Vec<Option<String>>) {
    out.push(match e.explicit_key() {
        Some(Key::Str(s)) => Some(s.clone()),
        _ => None,
    });
    for child in e.child_elements() {
        keys(child, out);
    }
}
fn layout(e: &Element) -> (Ui<RetainedScene>, BTreeMap<String, Rect>) {
    let mut ui = Ui::new(RetainedScene::new());
    ui.mount(e);
    ui.compute_layout(Size::new(
        AvailableSpace::Definite(900.0),
        AvailableSpace::MaxContent,
    ));
    let mut element_keys = Vec::new();
    keys(e, &mut element_keys);
    let mut rects = Vec::new();
    ui.backend().visit_layout(|_, r| rects.push(r));
    assert_eq!(element_keys.len(), rects.len());
    let keyed = element_keys
        .into_iter()
        .zip(rects)
        .filter_map(|(k, r)| k.map(|k| (k, r)))
        .collect();
    (ui, keyed)
}
fn activate(c: &mut ProfilerController, key: &str) {
    let view = c.build();
    c.handle_event(&view, key, &UiEvent::Activate).unwrap();
}
fn near(a: f32, b: f32) {
    assert!((a - b).abs() < 0.1, "{a} != {b}");
}

#[test]
fn real_instant_cpu_capture_reaches_selectable_loom_bars() {
    let limits = ProfilerLimits::default();
    let mut producer = CpuProducer::new(limits, 1).unwrap();
    let f = producer.begin_frame(FrameId::from_raw(22), 22).unwrap();
    let root = producer.begin_span(f, "app render frame", "main").unwrap();
    let child = producer.begin_span(f, "Loom assemble", "main").unwrap();
    std::thread::sleep(Duration::from_millis(2));
    producer.end_span(child).unwrap();
    producer.end_span(root).unwrap();
    let captured = producer.finish_frame(f).unwrap();
    assert!(captured.spans().iter().all(|s| s.duration_ns() > 0));
    eprintln!(
        "REAL CPU frame={}ns spans={:?}",
        captured.duration_ns(),
        captured.spans()
    );
    let mut c = ProfilerController::default();
    publish(&mut c, [captured]);
    let view = c.build();
    assert_eq!(view.spans().len(), 2);
    let key = view.spans()[1].key.clone();
    let selected = view.spans()[1].span;
    let (ui, rects) = layout(view.element());
    assert!(rects[&key].size.width > 0.0);
    assert!(!ui.backend().to_draw_list().is_empty());
    c.handle_event(&view, &key, &UiEvent::Activate).unwrap();
    assert_eq!(c.selected_span(), Some(selected));
    activate(&mut c, "profiler:fit-span");
    assert!(c
        .build()
        .spans()
        .iter()
        .any(|s| s.span == selected && s.width > 0.99));
    let mut recorder = Ui::new(RecordingBackend::new());
    recorder.mount(c.build().element());
    recorder.backend_mut().clear();
    recorder.update(c.build().element());
    assert!(recorder.backend().ops().is_empty());
}

#[test]
fn actual_layout_and_pixels_follow_time_selection_zoom_and_clipping() {
    let mut c = ProfilerController::default();
    publish(&mut c, [fixture()]);
    let view = c.build();
    let child = view.spans().iter().find(|s| s.span.raw() == 2).unwrap();
    let key = child.key.clone();
    let plot_key = format!("profiler:plot:{}", child.track);
    let (ui, rects) = layout(view.element());
    let plot = rects[&plot_key];
    let bar = rects[&key];
    assert!(plot.size.width > 100.0);
    near(bar.left(), plot.left() + plot.size.width * 0.25);
    near(bar.size.width, plot.size.width * 0.25);
    let x = (bar.left() + bar.size.width / 2.0) as u32;
    let y = (bar.top() + 8.0) as u32;
    let pixels = rasterize(&ui.backend().to_draw_list(), 900, 1200);
    let before = pixels.pixel(x, y);
    near(before[0], 0.15);
    near(before[2], 0.90);
    c.handle_event(&view, &key, &UiEvent::Activate).unwrap();
    let selected = c.build();
    let (ui, _) = layout(selected.element());
    let pixel = rasterize(&ui.backend().to_draw_list(), 900, 1200).pixel(x, y);
    assert!(
        pixel[0] > 0.99 && pixel[2] < 0.13,
        "selection must change actual paint: {pixel:?}"
    );
    activate(&mut c, "profiler:zoom-in");
    let zoomed = c.build();
    let child = zoomed.spans().iter().find(|s| s.span.raw() == 2).unwrap();
    near(child.left, 0.0);
    near(child.width, 0.5);
    activate(&mut c, "profiler:pan-later");
    assert!(!c.build().spans().iter().any(|s| s.span.raw() == 2));
    c.apply(Action::Window(ProfilerTimeWindow {
        start_ns: 375,
        duration_ns: 250,
    }))
    .unwrap();
    let clipped = c.build();
    let child = clipped.spans().iter().find(|s| s.span.raw() == 2).unwrap();
    near(child.left, 0.0);
    near(child.width, 0.5);
    activate(&mut c, "profiler:fit");
    assert_eq!(c.time_window().duration_ns, 1000);
}

#[test]
fn bounded_paging_reaches_all_bars_frames_and_tracks() {
    let mut c = ProfilerController::new(ProfilerPanelLimits {
        visible_frames: 2,
        visible_tracks: 2,
        visible_spans: 3,
    })
    .unwrap();
    let spans: Vec<_> = (0..1000)
        .map(|i| span(i, None, (i % 100) * 10, 8, &format!("lane {}", i / 100)))
        .collect();
    let frames = (0..40).map(|id| frame(id, 1000, if id == 39 { spans.clone() } else { vec![] }));
    publish(&mut c, frames);
    c.apply(Action::Follow(false)).unwrap();
    let mut seen = BTreeSet::new();
    loop {
        loop {
            let view = c.build();
            let stats = view.stats();
            assert!(
                stats.visible_frames <= 2 && stats.visible_tracks <= 2 && stats.visible_spans <= 3
            );
            let mut ui = Ui::new(RecordingBackend::new());
            ui.mount(view.element());
            assert!(ui.node_count() < 130);
            seen.extend(view.spans().iter().map(|s| s.span.raw()));
            if view.bindings().any(|(k, _)| k == "profiler:spans-next") {
                activate(&mut c, "profiler:spans-next");
            } else {
                break;
            }
        }
        if c.build()
            .bindings()
            .any(|(k, _)| k == "profiler:tracks-next")
        {
            activate(&mut c, "profiler:tracks-next");
        } else {
            break;
        }
    }
    assert_eq!(seen.len(), 1000);
    let mut seen_frames = BTreeSet::new();
    loop {
        let view = c.build();
        seen_frames.extend(
            view.bindings()
                .filter_map(|(k, _)| k.strip_prefix("profiler:frame:cpu:").map(str::to_owned)),
        );
        if view.bindings().any(|(k, _)| k == "profiler:frames-prev") {
            activate(&mut c, "profiler:frames-prev");
        } else {
            break;
        }
    }
    assert_eq!(seen_frames.len(), 40);
}

#[test]
fn nested_and_overlapping_spans_get_distinct_tracks() {
    let mut c = ProfilerController::default();
    publish(
        &mut c,
        [frame(
            1,
            1000,
            vec![
                span(1, None, 0, 1000, "main"),
                span(2, None, 100, 200, "main"),
                span(3, Some(1), 100, 200, "main"),
            ],
        )],
    );
    let view = c.build();
    assert_eq!(view.stats().total_tracks, 3);
    let tracks: BTreeSet<_> = view.spans().iter().map(|s| s.track).collect();
    assert_eq!(tracks.len(), 3);
    activate(&mut c, "profiler:track:1");
    assert_eq!(c.selected_track(), Some(1));
}

#[test]
fn stale_cross_controller_events_and_invalid_commands_are_rejected() {
    let mut c = ProfilerController::default();
    publish(&mut c, [fixture()]);
    let view = c.build();
    assert_eq!(
        c.handle_event(&view, "invented", &UiEvent::Activate),
        Err(Error::UnboundEvent)
    );
    assert_eq!(
        c.apply(Action::Frame(FrameId::from_raw(999))),
        Err(Error::MissingTarget)
    );
    for window in [
        ProfilerTimeWindow {
            start_ns: 0,
            duration_ns: 0,
        },
        ProfilerTimeWindow {
            start_ns: u64::MAX,
            duration_ns: 2,
        },
        ProfilerTimeWindow {
            start_ns: 900,
            duration_ns: 101,
        },
    ] {
        assert_eq!(c.apply(Action::Window(window)), Err(Error::InvalidWindow));
    }
    let mut other = ProfilerController::default();
    publish(&mut other, [fixture()]);
    assert_eq!(
        other.handle_event(&view, "profiler:zoom-in", &UiEvent::Activate),
        Err(Error::StaleView)
    );
    c.handle_event(&view, "profiler:zoom-in", &UiEvent::Activate)
        .unwrap();
    assert_eq!(
        c.handle_event(&view, "profiler:zoom-in", &UiEvent::Activate),
        Err(Error::StaleView)
    );
    let old = c.build();
    publish(&mut c, [fixture()]);
    assert_eq!(
        c.handle_event(&old, "profiler:fit", &UiEvent::Activate),
        Err(Error::StaleView)
    );
}

#[test]
fn paused_selection_survives_publication_and_domains_never_merge() {
    let mut c = ProfilerController::default();
    let cpu = store([fixture()]);
    let gpu = store([frame(7, 2000, vec![span(2, None, 500, 1000, "queue")])]);
    c.publish(ProfilerSources {
        cpu: Some(cpu.clone()),
        gpu: Some(gpu.clone()),
    });
    activate(&mut c, "profiler:frame:cpu:7");
    activate(&mut c, "profiler:zoom-in");
    let window = c.time_window();
    c.apply(Action::Span(SpanId::from_raw(2))).unwrap();
    c.publish(ProfilerSources {
        cpu: Some(cpu),
        gpu: Some(gpu),
    });
    assert_eq!(c.time_window(), window);
    assert_eq!(c.selected_span(), Some(SpanId::from_raw(2)));
    activate(&mut c, "profiler:gpu");
    assert_eq!(c.selection().unwrap().domain, ProfilerDomain::Gpu);
    assert_eq!(c.selected_span(), None);
    assert_eq!(c.time_window().duration_ns, 2000);
    let view = c.build();
    assert!(view
        .spans()
        .iter()
        .all(|s| s.key.starts_with("profiler:span:gpu:")));
    c.publish(ProfilerSources::default());
    assert!(c.selection().is_none());
    assert!(c.build().spans().is_empty());
}

#[test]
fn empty_zero_duration_and_maximum_integer_time_remain_finite_and_selectable() {
    assert!(ProfilerController::new(ProfilerPanelLimits {
        visible_frames: 0,
        ..Default::default()
    })
    .is_err());
    let mut c = ProfilerController::default();
    assert!(c.build().spans().is_empty());
    for duration in [0, u64::MAX] {
        publish(
            &mut c,
            [frame(
                1,
                duration,
                vec![span(1, None, duration, 0, "instant")],
            )],
        );
        let view = c.build();
        assert_eq!(view.spans().len(), 1);
        let bar = &view.spans()[0];
        assert!(bar.left.is_finite() && bar.width > 0.0 && bar.left + bar.width <= 1.0);
        c.handle_event(&view, &bar.key, &UiEvent::Activate).unwrap();
        c.apply(Action::FitSelection).unwrap();
        c.apply(Action::ZoomOut).unwrap();
        c.apply(Action::PanLater).unwrap();
        assert!(c
            .time_window()
            .start_ns
            .checked_add(c.time_window().duration_ns)
            .is_some());
    }
}

#[test]
fn oversized_frame_reports_capacity_without_partial_timeline() {
    let count = prism_editor_ui::MAX_PROFILER_INDEXED_SPANS + 1;
    let limits = ProfilerLimits {
        max_spans_per_frame: count,
        ..Default::default()
    };
    let spans = (0..count as u64)
        .map(|id| span(id, None, 0, 1, "main"))
        .collect();
    let frame = Frame::try_new(FrameId::from_raw(1), 1, 0, 1, spans, &limits).unwrap();
    let mut store = FrameStore::new(limits);
    store.insert(frame).unwrap();
    let mut c = ProfilerController::default();
    c.publish(ProfilerSources {
        cpu: Some(Arc::new(store)),
        gpu: None,
    });
    let view = c.build();
    assert!(view.spans().is_empty());
    let (_, rects) = layout(view.element());
    assert!(rects.contains_key("profiler:error"));
    assert_eq!(view.stats().visible_frames, 1);
}
