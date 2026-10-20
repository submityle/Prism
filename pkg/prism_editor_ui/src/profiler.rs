//! Bounded navigation over immutable real CPU/GPU profiler stores.

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap},
    sync::Arc,
};

use prism_editor_profiler::{Frame, FrameId, FrameStore, SpanId};

use crate::{profiler_view::BoundProfilerPanel, UiEvent};

/// Hard upper bound on frame chart nodes.
pub const MAX_PROFILER_VISIBLE_FRAMES: usize = 32;
/// Hard upper bound on track rows per view.
pub const MAX_PROFILER_VISIBLE_TRACKS: usize = 16;
/// Hard upper bound on materialized span bars per view.
pub const MAX_PROFILER_VISIBLE_SPANS: usize = 256;
/// Maximum spans indexed in one selected frame, independent of model limits.
pub const MAX_PROFILER_INDEXED_SPANS: usize = 65_536;

/// Separate uncalibrated clock domains; equal IDs never merge CPU and GPU.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProfilerDomain {
    /// Instant clock domain.
    #[default]
    Cpu,
    /// Hardware query clock domain.
    Gpu,
}
impl ProfilerDomain {
    pub(crate) fn key(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

/// Host-published immutable stores. Publish a new Arc after acquisition changes.
/// Holding these snapshots prevents in-place mutation behind displayed events.
#[derive(Clone, Default)]
pub struct ProfilerSources {
    /// Actual completed CPU frames, optionally absent.
    pub cpu: Option<Arc<FrameStore>>,
    /// Actual completed hardware frames, optionally absent.
    pub gpu: Option<Arc<FrameStore>>,
}
impl ProfilerSources {
    pub(crate) fn store(&self, domain: ProfilerDomain) -> Option<&FrameStore> {
        match domain {
            ProfilerDomain::Cpu => self.cpu.as_deref(),
            ProfilerDomain::Gpu => self.gpu.as_deref(),
        }
    }
}

/// Frame identity qualified by its clock domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProfilerFrameSelection {
    /// Selected clock domain.
    pub domain: ProfilerDomain,
    /// Producer-assigned frame identity.
    pub frame: FrameId,
}

/// Frame-relative viewing interval; commands validate it before use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProfilerTimeWindow {
    /// First visible frame-relative nanosecond.
    pub start_ns: u64,
    /// Visible extent; accepted controller windows are at least one nanosecond.
    pub duration_ns: u64,
}

/// Per-view windows, capped by the public hard limits.
#[derive(Clone, Copy, Debug)]
pub struct ProfilerPanelLimits {
    /// Frames materialized in the navigable chart.
    pub visible_frames: usize,
    /// Tracks materialized in the timeline.
    pub visible_tracks: usize,
    /// Span bars materialized across all visible tracks.
    pub visible_spans: usize,
}
impl Default for ProfilerPanelLimits {
    fn default() -> Self {
        Self {
            visible_frames: 16,
            visible_tracks: 8,
            visible_spans: 128,
        }
    }
}

/// Owned panel commands, also available to native wheel/keyboard adapters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfilerAction {
    /// Switch clock domain and select its newest frame.
    Domain(ProfilerDomain),
    /// Select a frame in the current domain and stop following live capture.
    Frame(FrameId),
    /// Highlight a track in the selected frame.
    Track(usize),
    /// Select an actual span and its track.
    Span(SpanId),
    /// Halve the viewing interval around its center.
    ZoomIn,
    /// Double the viewing interval around its center.
    ZoomOut,
    /// Move earlier by half the viewing interval.
    PanEarlier,
    /// Move later by half the viewing interval.
    PanLater,
    /// Show the whole selected frame.
    FitFrame,
    /// Fit the selected span.
    FitSelection,
    /// Set a validated relative interval (native pointer-centered zoom/pan).
    Window(ProfilerTimeWindow),
    /// Enable/disable following the newest published frame.
    Follow(bool),
    /// Frame chart page forward.
    NextFrames,
    /// Frame chart page backward.
    PreviousFrames,
    /// Track viewport page forward.
    NextTracks,
    /// Track viewport page backward.
    PreviousTracks,
    /// Page intersecting bars inside the visible track/time viewport.
    NextSpans,
    /// Previous page of intersecting bars.
    PreviousSpans,
}

/// Explicit invalid input, stale event, or data-index capacity error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfilerPanelError {
    /// All viewport limits must be nonzero and within their hard bounds.
    InvalidLimits,
    /// Rebuild after publication or an accepted command; cross-panel view.
    StaleView,
    /// Target/event is absent from this exact bound tree.
    UnboundEvent,
    /// Frame, track, or span no longer exists in the current snapshot.
    MissingTarget,
    /// Requested time interval is empty or outside the selected frame.
    InvalidWindow,
    /// Selected frame exceeds the UI indexing budget; no partial data claim.
    IndexCapacity,
}
impl std::fmt::Display for ProfilerPanelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "profiler panel: {self:?}")
    }
}
impl std::error::Error for ProfilerPanelError {}

pub(crate) struct Track {
    pub label: String,
    pub spans: Vec<usize>,
}

/// Owns UI navigation and a bounded index, never a capture device or app loop.
pub struct ProfilerController {
    pub(crate) sources: ProfilerSources,
    pub(crate) domain: ProfilerDomain,
    pub(crate) selected: Option<ProfilerFrameSelection>,
    pub(crate) selected_span: Option<SpanId>,
    pub(crate) selected_track: Option<usize>,
    pub(crate) follow: bool,
    pub(crate) window: ProfilerTimeWindow,
    pub(crate) limits: ProfilerPanelLimits,
    pub(crate) frame_offset: usize,
    pub(crate) track_offset: usize,
    pub(crate) span_offset: usize,
    pub(crate) tracks: Vec<Track>,
    pub(crate) index_error: Option<ProfilerPanelError>,
    pub(crate) stamp: Arc<()>,
}
impl Default for ProfilerController {
    fn default() -> Self {
        Self::new(ProfilerPanelLimits::default()).expect("default viewport limits")
    }
}
impl ProfilerController {
    /// Creates a bounded empty panel. Sources are supplied by `publish`.
    pub fn new(limits: ProfilerPanelLimits) -> Result<Self, ProfilerPanelError> {
        if limits.visible_frames == 0
            || limits.visible_frames > MAX_PROFILER_VISIBLE_FRAMES
            || limits.visible_tracks == 0
            || limits.visible_tracks > MAX_PROFILER_VISIBLE_TRACKS
            || limits.visible_spans == 0
            || limits.visible_spans > MAX_PROFILER_VISIBLE_SPANS
        {
            return Err(ProfilerPanelError::InvalidLimits);
        }
        Ok(Self {
            sources: ProfilerSources::default(),
            domain: ProfilerDomain::Cpu,
            selected: None,
            selected_span: None,
            selected_track: None,
            follow: true,
            window: ProfilerTimeWindow {
                start_ns: 0,
                duration_ns: 1,
            },
            limits,
            frame_offset: 0,
            track_offset: 0,
            span_offset: 0,
            tracks: Vec::new(),
            index_error: None,
            stamp: Arc::new(()),
        })
    }
    /// Publish completed acquisition snapshots; fences every previously bound event.
    /// Model stores remain owned by the host through shared immutable Arcs.
    pub fn publish(&mut self, sources: ProfilerSources) {
        self.sources = sources;
        let old = self.selected;
        let navigation = (self.track_offset, self.span_offset, self.selected_track);
        if self.follow || self.frame().is_none() {
            self.select_latest();
        }
        if old != self.selected {
            self.selected_span = None;
        }
        self.reindex();
        if old != self.selected || self.follow {
            self.fit();
        } else {
            self.clamp_window();
        }
        if old == self.selected && !self.follow {
            self.track_offset =
                page_offset(navigation.0, self.tracks.len(), self.limits.visible_tracks);
            self.selected_track = navigation.2.filter(|&i| i < self.tracks.len());
            self.span_offset = page_offset(
                navigation.1,
                self.visible_span_count(),
                self.limits.visible_spans,
            );
        }
        self.frame_offset = page_offset(
            self.frame_offset,
            self.frames().len(),
            self.limits.visible_frames,
        );
        if self.follow {
            self.reveal_frame();
        }
        self.stamp = Arc::new(());
    }
    /// Current domain-qualified frame identity.
    #[must_use]
    pub const fn selection(&self) -> Option<ProfilerFrameSelection> {
        self.selected
    }
    /// Selected span identity within the selected frame.
    #[must_use]
    pub const fn selected_span(&self) -> Option<SpanId> {
        self.selected_span
    }
    /// Selected track index within this immutable frame snapshot.
    #[must_use]
    pub const fn selected_track(&self) -> Option<usize> {
        self.selected_track
    }
    /// Relative interval used for both graph geometry and hit targets.
    #[must_use]
    pub const fn time_window(&self) -> ProfilerTimeWindow {
        self.window
    }
    /// Assemble the keyed Loom tree without re-indexing or cloning a frame.
    #[must_use]
    pub fn build(&self) -> BoundProfilerPanel {
        crate::profiler_view::build(self)
    }
    /// Resolve an event and apply it only against the exact displayed snapshot.
    pub fn handle_event(
        &mut self,
        view: &BoundProfilerPanel,
        target: &str,
        event: &UiEvent,
    ) -> Result<ProfilerAction, ProfilerPanelError> {
        if !Arc::ptr_eq(&self.stamp, &view.stamp) {
            return Err(ProfilerPanelError::StaleView);
        }
        let action = view.dispatch(target, event)?;
        self.apply(action)?;
        Ok(action)
    }
    /// Apply a host command, validating fallible input before mutation.
    pub fn apply(&mut self, action: ProfilerAction) -> Result<(), ProfilerPanelError> {
        match action {
            ProfilerAction::Frame(id) if !self.frames().iter().any(|f| f.id() == id) => {
                return Err(ProfilerPanelError::MissingTarget)
            }
            ProfilerAction::Track(i) if i >= self.tracks.len() => {
                return Err(ProfilerPanelError::MissingTarget)
            }
            ProfilerAction::Span(id)
                if !self.tracks.iter().any(|t| {
                    t.spans
                        .iter()
                        .any(|&i| self.frame().is_some_and(|f| f.spans()[i].id() == id))
                }) =>
            {
                return Err(ProfilerPanelError::MissingTarget)
            }
            ProfilerAction::FitSelection if self.selected_span.is_none() => {
                return Err(ProfilerPanelError::MissingTarget)
            }
            ProfilerAction::Window(w)
                if w.duration_ns == 0
                    || w.start_ns
                        .checked_add(w.duration_ns)
                        .is_none_or(|end| end > self.extent()) =>
            {
                return Err(ProfilerPanelError::InvalidWindow)
            }
            _ => {}
        }
        match action {
            ProfilerAction::Domain(domain) => {
                self.domain = domain;
                self.selected_span = None;
                self.select_latest();
                self.reindex();
                self.fit();
                self.reveal_frame();
            }
            ProfilerAction::Frame(frame) => {
                self.selected = Some(ProfilerFrameSelection {
                    domain: self.domain,
                    frame,
                });
                self.follow = false;
                self.selected_span = None;
                self.reindex();
                self.fit();
                self.reveal_frame();
            }
            ProfilerAction::Track(i) => self.selected_track = Some(i),
            ProfilerAction::Span(id) => {
                self.selected_span = Some(id);
                self.selected_track = self.tracks.iter().position(|t| {
                    t.spans
                        .iter()
                        .any(|&i| self.frame().expect("indexed frame").spans()[i].id() == id)
                });
            }
            ProfilerAction::ZoomIn => self.zoom(false),
            ProfilerAction::ZoomOut => self.zoom(true),
            ProfilerAction::PanEarlier => {
                self.window.start_ns = self
                    .window
                    .start_ns
                    .saturating_sub((self.window.duration_ns / 2).max(1));
            }
            ProfilerAction::PanLater => {
                self.window.start_ns = self
                    .window
                    .start_ns
                    .saturating_add((self.window.duration_ns / 2).max(1));
                self.clamp_window();
            }
            ProfilerAction::FitFrame => self.fit(),
            ProfilerAction::FitSelection => {
                let span = self
                    .frame()
                    .and_then(|f| {
                        f.spans()
                            .iter()
                            .find(|s| Some(s.id()) == self.selected_span)
                    })
                    .ok_or(ProfilerPanelError::MissingTarget)?;
                self.window = ProfilerTimeWindow {
                    start_ns: span.start_ns(),
                    duration_ns: span.duration_ns().max(1),
                };
                self.clamp_window();
            }
            ProfilerAction::Window(window) => self.window = window,
            ProfilerAction::Follow(value) => {
                self.follow = value;
                if value {
                    let old = self.selected;
                    self.select_latest();
                    if old != self.selected {
                        self.selected_span = None;
                    }
                    self.reindex();
                    self.fit();
                    self.reveal_frame();
                }
            }
            ProfilerAction::NextFrames => {
                self.frame_offset = next_offset(
                    self.frame_offset,
                    self.frames().len(),
                    self.limits.visible_frames,
                );
            }
            ProfilerAction::PreviousFrames => {
                self.frame_offset = self.frame_offset.saturating_sub(self.limits.visible_frames);
            }
            ProfilerAction::NextTracks => {
                self.track_offset = next_offset(
                    self.track_offset,
                    self.tracks.len(),
                    self.limits.visible_tracks,
                );
                self.span_offset = 0;
            }
            ProfilerAction::PreviousTracks => {
                self.track_offset = self.track_offset.saturating_sub(self.limits.visible_tracks);
                self.span_offset = 0;
            }
            ProfilerAction::NextSpans => {
                self.span_offset = next_offset(
                    self.span_offset,
                    self.visible_span_count(),
                    self.limits.visible_spans,
                );
            }
            ProfilerAction::PreviousSpans => {
                self.span_offset = self.span_offset.saturating_sub(self.limits.visible_spans);
            }
        }
        if matches!(
            action,
            ProfilerAction::ZoomIn
                | ProfilerAction::ZoomOut
                | ProfilerAction::PanEarlier
                | ProfilerAction::PanLater
                | ProfilerAction::FitFrame
                | ProfilerAction::FitSelection
                | ProfilerAction::Window(_)
        ) {
            self.span_offset = 0;
        }
        self.stamp = Arc::new(());
        Ok(())
    }
    pub(crate) fn frames(&self) -> &[Frame] {
        self.sources
            .store(self.domain)
            .map_or(&[], FrameStore::frames)
    }
    pub(crate) fn frame(&self) -> Option<&Frame> {
        self.selected.and_then(|s| {
            self.sources
                .store(s.domain)?
                .frames()
                .iter()
                .find(|f| f.id() == s.frame)
        })
    }
    pub(crate) fn extent(&self) -> u64 {
        self.frame().map_or(1, |f| f.duration_ns().max(1))
    }
    pub(crate) fn visible_span_count(&self) -> usize {
        let Some(frame) = self.frame() else {
            return 0;
        };
        self.tracks
            .iter()
            .skip(self.track_offset)
            .take(self.limits.visible_tracks)
            .flat_map(|t| t.spans.iter())
            .filter(|&&i| intersects(&frame.spans()[i], self.window))
            .count()
    }
    fn select_latest(&mut self) {
        self.selected = self.frames().last().map(|f| ProfilerFrameSelection {
            domain: self.domain,
            frame: f.id(),
        });
    }
    fn reveal_frame(&mut self) {
        if let Some(index) = self
            .frames()
            .iter()
            .position(|f| Some(f.id()) == self.selected.map(|s| s.frame))
        {
            self.frame_offset = index / self.limits.visible_frames * self.limits.visible_frames;
        }
    }
    fn fit(&mut self) {
        self.window = ProfilerTimeWindow {
            start_ns: 0,
            duration_ns: self.extent(),
        };
        self.span_offset = 0;
    }
    fn clamp_window(&mut self) {
        self.window.duration_ns = self.window.duration_ns.min(self.extent()).max(1);
        self.window.start_ns = self
            .window
            .start_ns
            .min(self.extent() - self.window.duration_ns);
    }
    fn zoom(&mut self, out: bool) {
        let old = self.window.duration_ns;
        let new = if out {
            old.saturating_mul(2).min(self.extent())
        } else {
            (old / 2).max(1)
        };
        let center = u128::from(self.window.start_ns) + u128::from(old / 2);
        self.window = ProfilerTimeWindow {
            start_ns: center
                .saturating_sub(u128::from(new / 2))
                .min(u128::from(u64::MAX)) as u64,
            duration_ns: new,
        };
        self.clamp_window();
    }
    fn reindex(&mut self) {
        self.tracks.clear();
        self.index_error = None;
        self.track_offset = 0;
        self.span_offset = 0;
        self.selected_track = None;
        let Some(frame) = self.frame() else {
            self.selected_span = None;
            return;
        };
        if frame.spans().len() > MAX_PROFILER_INDEXED_SPANS {
            self.index_error = Some(ProfilerPanelError::IndexCapacity);
            self.selected_span = None;
            return;
        }
        // Memoized parent depth; iterative to avoid a deep model consuming stack.
        let by_id: BTreeMap<_, _> = frame.spans().iter().map(|s| (s.id(), s)).collect();
        let mut depths = BTreeMap::new();
        for span in frame.spans() {
            let mut chain = Vec::new();
            let mut current = Some(span.id());
            while let Some(id) = current {
                if depths.contains_key(&id) {
                    break;
                }
                chain.push(id);
                current = by_id[&id].parent();
            }
            let first_depth = current.map_or(0, |id| depths[&id] + 1);
            for (depth, id) in (first_depth..).zip(chain.into_iter().rev()) {
                depths.insert(id, depth);
            }
        }
        // Pack overlapping siblings into separate rows rather than occluding them.
        let mut groups: BTreeMap<(&str, usize), Vec<usize>> = BTreeMap::new();
        for (i, span) in frame.spans().iter().enumerate() {
            groups
                .entry((span.lane(), depths[&span.id()]))
                .or_default()
                .push(i);
        }
        let mut tracks = Vec::new();
        for ((lane, depth), spans) in groups {
            let mut ends: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
            for i in spans {
                let span = &frame.spans()[i];
                let row = if ends
                    .peek()
                    .is_some_and(|Reverse((end, _))| *end <= span.start_ns())
                {
                    ends.pop().expect("peeked row").0 .1
                } else {
                    let row = tracks.len();
                    tracks.push(Track {
                        label: format!("{} · depth {depth} · row {}", short(lane), ends.len() + 1),
                        spans: Vec::new(),
                    });
                    row
                };
                tracks[row].spans.push(i);
                ends.push(Reverse((span.start_ns() + span.duration_ns(), row)));
            }
        }
        if self
            .selected_span
            .is_some_and(|id| !by_id.contains_key(&id))
        {
            self.selected_span = None;
        }
        self.tracks = tracks;
    }
}
pub(crate) fn intersects(span: &prism_editor_profiler::Span, w: ProfilerTimeWindow) -> bool {
    if span.duration_ns() == 0 {
        span.start_ns() >= w.start_ns && span.start_ns() <= w.start_ns + w.duration_ns
    } else {
        span.start_ns() < w.start_ns + w.duration_ns
            && span.start_ns() + span.duration_ns() > w.start_ns
    }
}
pub(crate) fn short(text: &str) -> String {
    text.chars().take(96).collect()
}
fn page_offset(offset: usize, total: usize, size: usize) -> usize {
    offset.min(total.saturating_sub(1) / size * size)
}
fn next_offset(offset: usize, total: usize, size: usize) -> usize {
    page_offset(offset.saturating_add(size), total, size)
}
