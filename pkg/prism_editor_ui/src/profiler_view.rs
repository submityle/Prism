//! Real keyed Loom frame histograms and time-proportional selectable span bars.

use std::{collections::BTreeMap, sync::Arc};

use prism_editor_profiler::{FrameId, SpanId};
use prism_ui::{
    style::{Keyword, StyleProp, StyleValue},
    Element,
};

use crate::{
    profiler::{
        intersects, short, ProfilerAction as Action, ProfilerController, ProfilerDomain,
        ProfilerPanelError, ProfilerTimeWindow,
    },
    UiEvent, UiTrigger,
};

/// Exact source identity and clipped geometry of a visible real span.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfilerSpanVisual {
    /// Key of the painted, interactive Loom box.
    pub key: String,
    /// Source frame identity in this view's selected domain.
    pub frame: FrameId,
    /// Source span identity.
    pub span: SpanId,
    /// Absolute track index in the bounded index.
    pub track: usize,
    /// Left edge as a fraction of the timeline width, after clipping.
    pub left: f32,
    /// Width as a fraction of the timeline (zero-duration spans use a marker).
    pub width: f32,
}

/// Materialization counts; hidden data is explicitly paged rather than lost.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProfilerViewStats {
    /// Number of source frames in the current domain.
    pub total_frames: usize,
    /// Number of chart bars in this view.
    pub visible_frames: usize,
    /// Number of tracks in the selected frame.
    pub total_tracks: usize,
    /// Number of track rows in this view.
    pub visible_tracks: usize,
    /// Intersecting spans across visible tracks before span pagination.
    pub matching_spans: usize,
    /// Number of painted span bars in this view.
    pub visible_spans: usize,
}

/// Actual renderable subtree and its exact semantic activation targets.
/// Retain until the next controller publication/command, then rebuild.
#[derive(Clone, Debug)]
pub struct BoundProfilerPanel {
    element: Element,
    bindings: BTreeMap<String, Action>,
    spans: Vec<ProfilerSpanVisual>,
    stats: ProfilerViewStats,
    window: ProfilerTimeWindow,
    pub(crate) stamp: Arc<()>,
}
impl BoundProfilerPanel {
    /// Mount/reconcile this actual Loom Element in the host's dock body.
    #[must_use]
    pub fn element(&self) -> &Element {
        &self.element
    }
    /// Exact keys to associate with reconciled layout for native hit-testing.
    pub fn bindings(&self) -> impl Iterator<Item = (&str, UiTrigger)> {
        self.bindings
            .keys()
            .map(|key| (key.as_str(), UiTrigger::Activate))
    }
    /// Visible actual source spans and their clipped normalized geometry.
    #[must_use]
    pub fn spans(&self) -> &[ProfilerSpanVisual] {
        &self.spans
    }
    /// Inspect virtualization counts without walking the tree.
    #[must_use]
    pub const fn stats(&self) -> ProfilerViewStats {
        self.stats
    }
    /// Interval used to lay out this immutable tree.
    #[must_use]
    pub const fn time_window(&self) -> ProfilerTimeWindow {
        self.window
    }
    /// Resolve a semantic activation; controller `handle_event` fences freshness.
    pub fn dispatch(&self, target: &str, event: &UiEvent) -> Result<Action, ProfilerPanelError> {
        if *event != UiEvent::Activate {
            return Err(ProfilerPanelError::UnboundEvent);
        }
        self.bindings
            .get(target)
            .copied()
            .ok_or(ProfilerPanelError::UnboundEvent)
    }
}

pub(crate) fn build(c: &ProfilerController) -> BoundProfilerPanel {
    let mut view = BoundProfilerPanel {
        element: Element::box_(),
        bindings: BTreeMap::new(),
        spans: Vec::new(),
        stats: ProfilerViewStats {
            total_frames: c.frames().len(),
            total_tracks: c.tracks.len(),
            matching_spans: c.visible_span_count(),
            ..Default::default()
        },
        window: c.window,
        stamp: c.stamp.clone(),
    };
    let header = row()
        .key_str("profiler:domains")
        .child(button(
            &mut view,
            "profiler:cpu",
            "CPU",
            Some(Action::Domain(ProfilerDomain::Cpu)),
            c.domain == ProfilerDomain::Cpu,
        ))
        .child(button(
            &mut view,
            "profiler:gpu",
            "GPU",
            Some(Action::Domain(ProfilerDomain::Gpu)),
            c.domain == ProfilerDomain::Gpu,
        ))
        .child(button(
            &mut view,
            "profiler:follow",
            if c.follow { "Live: on" } else { "Live: paused" },
            Some(Action::Follow(!c.follow)),
            c.follow,
        ));
    let mut chart = row()
        .key_str("profiler:frame-chart")
        .style(StyleProp::Height, StyleValue::px(48.0));
    let page = c
        .frames()
        .iter()
        .skip(c.frame_offset)
        .take(c.limits.visible_frames);
    let count = page.len();
    let max_duration = page
        .clone()
        .map(|f| f.duration_ns())
        .max()
        .unwrap_or(1)
        .max(1);
    for frame in page {
        let key = format!("profiler:frame:{}:{}", c.domain.key(), frame.id().raw());
        view.bindings.insert(key.clone(), Action::Frame(frame.id()));
        let fraction = (frame.duration_ns() as f64 / max_duration as f64) as f32;
        let selected = c.selected.is_some_and(|s| s.frame == frame.id());
        chart = chart.child(
            Element::box_()
                .key_str(key)
                .class("prism-profiler__frame")
                .class(if selected {
                    "is-selected"
                } else {
                    "is-unselected"
                })
                .style(StyleProp::Width, StyleValue::percent(100.0 / count as f32))
                .style(StyleProp::Height, StyleValue::percent(100.0))
                .style(StyleProp::FlexShrink, StyleValue::number(0.0))
                .style(
                    StyleProp::BackgroundColor,
                    StyleValue::rgba(0.06, 0.08, 0.12, 1.0),
                )
                .child(
                    Element::box_()
                        .class("prism-profiler__frame-bar")
                        .style(StyleProp::Position, StyleValue::keyword(Keyword::Absolute))
                        .style(StyleProp::Left, StyleValue::percent(5.0))
                        .style(
                            StyleProp::Top,
                            StyleValue::percent(100.0 * (1.0 - fraction.max(0.02))),
                        )
                        .style(StyleProp::Width, StyleValue::percent(90.0))
                        .style(
                            StyleProp::Height,
                            StyleValue::percent(100.0 * fraction.max(0.02)),
                        )
                        .style(StyleProp::BackgroundColor, color(selected, c.domain)),
                ),
        );
        view.stats.visible_frames += 1;
    }
    let navigation = row()
        .child(button(
            &mut view,
            "profiler:frames-prev",
            "Earlier frames",
            (c.frame_offset > 0).then_some(Action::PreviousFrames),
            false,
        ))
        .child(button(
            &mut view,
            "profiler:frames-next",
            "Later frames",
            (c.frame_offset + count < c.frames().len()).then_some(Action::NextFrames),
            false,
        ));
    let mut root = column()
        .key_str("profiler:panel")
        .class("prism-editor-profiler")
        .style(StyleProp::Width, StyleValue::percent(100.0))
        .child(header)
        .child(chart)
        .child(navigation)
        .child(Element::text(format!(
            "{} frames · chart {}–{} · {} clock (uncalibrated)",
            c.frames().len(),
            c.frame_offset,
            c.frame_offset + count,
            c.domain.key()
        )));
    let Some(frame) = c.frame() else {
        view.element = root.child(
            Element::text(format!("No {} capture published", c.domain.key()))
                .key_str("profiler:empty"),
        );
        return view;
    };
    root = root.child(
        Element::text(format!(
            "Frame {} · sequence {} · {} ns",
            frame.id().raw(),
            frame.sequence(),
            frame.duration_ns()
        ))
        .key_str("profiler:frame-detail"),
    );
    if let Some(error) = c.index_error {
        view.element = root.child(
            Element::text(format!(
                "Timeline unavailable: {error}; frame exceeds UI index capacity"
            ))
            .key_str("profiler:error"),
        );
        return view;
    }
    let controls = row()
        .child(button(
            &mut view,
            "profiler:zoom-in",
            "Zoom +",
            (c.window.duration_ns > 1).then_some(Action::ZoomIn),
            false,
        ))
        .child(button(
            &mut view,
            "profiler:zoom-out",
            "Zoom −",
            (c.window.duration_ns < c.extent()).then_some(Action::ZoomOut),
            false,
        ))
        .child(button(
            &mut view,
            "profiler:pan-earlier",
            "Pan ←",
            (c.window.start_ns > 0).then_some(Action::PanEarlier),
            false,
        ))
        .child(button(
            &mut view,
            "profiler:pan-later",
            "Pan →",
            (c.window.start_ns + c.window.duration_ns < c.extent()).then_some(Action::PanLater),
            false,
        ))
        .child(button(
            &mut view,
            "profiler:fit",
            "Fit frame",
            Some(Action::FitFrame),
            false,
        ))
        .child(button(
            &mut view,
            "profiler:fit-span",
            "Fit span",
            c.selected_span.map(|_| Action::FitSelection),
            false,
        ));
    let scale = row()
        .key_str("profiler:time-scale")
        .style(
            StyleProp::JustifyContent,
            StyleValue::keyword(Keyword::SpaceBetween),
        )
        .child(Element::text(format!("{} ns", c.window.start_ns)))
        .child(Element::text(format!(
            "{} ns",
            c.window.start_ns + c.window.duration_ns / 2
        )))
        .child(Element::text(format!(
            "{} ns",
            c.window.start_ns + c.window.duration_ns
        )));
    let mut timeline = column().key_str("profiler:timeline");
    let mut matching = 0usize;
    for (track_index, track) in c
        .tracks
        .iter()
        .enumerate()
        .skip(c.track_offset)
        .take(c.limits.visible_tracks)
    {
        let mut plot = Element::box_()
            .key_str(format!("profiler:plot:{track_index}"))
            .style(StyleProp::Width, StyleValue::percent(100.0))
            .style(StyleProp::Height, StyleValue::px(22.0))
            .style(StyleProp::FlexShrink, StyleValue::number(0.0))
            .style(
                StyleProp::BackgroundColor,
                StyleValue::rgba(0.06, 0.08, 0.12, 1.0),
            );
        for &i in &track.spans {
            let span = &frame.spans()[i];
            if !intersects(span, c.window) {
                continue;
            }
            let visible =
                matching >= c.span_offset && matching - c.span_offset < c.limits.visible_spans;
            matching += 1;
            if !visible {
                continue;
            }
            let start = span.start_ns().max(c.window.start_ns);
            let end = (span.start_ns() + span.duration_ns())
                .min(c.window.start_ns + c.window.duration_ns);
            let mut left =
                ((start - c.window.start_ns) as f64 / c.window.duration_ns as f64) as f32;
            let mut width = ((end - start) as f64 / c.window.duration_ns as f64) as f32;
            // A zero-length instant has a visible marker, never a fake duration.
            if span.duration_ns() == 0 {
                left = left.min(0.998);
                width = 0.002;
            }
            width = width.min(1.0 - left);
            let key = format!(
                "profiler:span:{}:{}:{}",
                c.domain.key(),
                frame.id().raw(),
                span.id().raw()
            );
            view.bindings.insert(key.clone(), Action::Span(span.id()));
            plot = plot.child(
                Element::box_()
                    .key_str(&key)
                    .class("prism-profiler__span")
                    .class(if c.selected_span == Some(span.id()) {
                        "is-selected"
                    } else {
                        "is-unselected"
                    })
                    .style(StyleProp::Position, StyleValue::keyword(Keyword::Absolute))
                    .style(StyleProp::Left, StyleValue::percent(left * 100.0))
                    .style(StyleProp::Top, StyleValue::px(2.0))
                    .style(StyleProp::Width, StyleValue::percent(width * 100.0))
                    .style(StyleProp::Height, StyleValue::px(18.0))
                    .style(
                        StyleProp::BackgroundColor,
                        color(c.selected_span == Some(span.id()), c.domain),
                    ),
            );
            view.spans.push(ProfilerSpanVisual {
                key,
                frame: frame.id(),
                span: span.id(),
                track: track_index,
                left,
                width,
            });
        }
        timeline = timeline.child(
            column()
                .key_str(format!("profiler:track-row:{track_index}"))
                .child(button(
                    &mut view,
                    &format!("profiler:track:{track_index}"),
                    &track.label,
                    Some(Action::Track(track_index)),
                    c.selected_track == Some(track_index),
                ))
                .child(plot),
        );
        view.stats.visible_tracks += 1;
    }
    view.stats.visible_spans = view.spans.len();
    let more_tracks =
        (c.track_offset + view.stats.visible_tracks < c.tracks.len()).then_some(Action::NextTracks);
    let paging = row()
        .child(button(
            &mut view,
            "profiler:tracks-prev",
            "Tracks ↑",
            (c.track_offset > 0).then_some(Action::PreviousTracks),
            false,
        ))
        .child(button(
            &mut view,
            "profiler:tracks-next",
            "Tracks ↓",
            more_tracks,
            false,
        ))
        .child(button(
            &mut view,
            "profiler:spans-prev",
            "Previous spans",
            (c.span_offset > 0).then_some(Action::PreviousSpans),
            false,
        ));
    let more_spans = (c.span_offset + view.spans.len() < matching).then_some(Action::NextSpans);
    let paging = paging.child(button(
        &mut view,
        "profiler:spans-next",
        "More spans",
        more_spans,
        false,
    ));
    root = root
        .child(controls)
        .child(scale)
        .child(timeline)
        .child(paging)
        .child(
            Element::text(format!(
                "Tracks {}–{} / {} · spans {}–{} / {} in viewport",
                c.track_offset,
                c.track_offset + view.stats.visible_tracks,
                c.tracks.len(),
                c.span_offset,
                c.span_offset + view.spans.len(),
                matching
            ))
            .key_str("profiler:counts"),
        );
    if let Some(span) = frame
        .spans()
        .iter()
        .find(|s| Some(s.id()) == c.selected_span)
    {
        root = root.child(
            Element::text(format!(
                "{} · {} · start {} ns · duration {} ns · parent {:?}",
                short(span.name()),
                short(span.lane()),
                span.start_ns(),
                span.duration_ns(),
                span.parent().map(SpanId::raw)
            ))
            .key_str("profiler:span-detail"),
        );
    }
    view.element = root;
    view
}
fn button(
    view: &mut BoundProfilerPanel,
    key: &str,
    text: &str,
    action: Option<Action>,
    selected: bool,
) -> Element {
    let enabled = action.is_some();
    if let Some(action) = action {
        view.bindings.insert(key.into(), action);
    }
    Element::box_()
        .key_str(key)
        .class("prism-profiler__button")
        .class(if selected {
            "is-selected"
        } else {
            "is-unselected"
        })
        .style(
            StyleProp::Opacity,
            StyleValue::number(if enabled { 1.0 } else { 0.4 }),
        )
        .style(
            StyleProp::BackgroundColor,
            if selected {
                StyleValue::rgba(0.20, 0.30, 0.40, 1.0)
            } else {
                StyleValue::rgba(0.10, 0.13, 0.17, 1.0)
            },
        )
        .style(StyleProp::PaddingLeft, StyleValue::px(3.0))
        .style(StyleProp::PaddingRight, StyleValue::px(3.0))
        .child(Element::text(text).style(StyleProp::Color, StyleValue::rgba(0.92, 0.95, 1.0, 1.0)))
}
fn color(selected: bool, domain: ProfilerDomain) -> StyleValue {
    if selected {
        StyleValue::rgba(1.0, 0.70, 0.12, 1.0)
    } else if domain == ProfilerDomain::Cpu {
        StyleValue::rgba(0.15, 0.65, 0.90, 1.0)
    } else {
        StyleValue::rgba(0.50, 0.80, 0.30, 1.0)
    }
}
fn column() -> Element {
    Element::box_()
        .style(StyleProp::Display, StyleValue::keyword(Keyword::Flex))
        .style(
            StyleProp::FlexDirection,
            StyleValue::keyword(Keyword::Column),
        )
        .style(StyleProp::Gap, StyleValue::px(2.0))
        .style(StyleProp::FlexShrink, StyleValue::number(0.0))
}
fn row() -> Element {
    Element::box_()
        .style(StyleProp::Display, StyleValue::keyword(Keyword::Flex))
        .style(StyleProp::FlexDirection, StyleValue::keyword(Keyword::Row))
        .style(StyleProp::FlexShrink, StyleValue::number(0.0))
}
