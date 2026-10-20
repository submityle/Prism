//! Transactional timeline interaction, independent of a device event backend.
use crate::{BoundTimelinePanel, SelectionMode, UiEvent};
use prism_editor_timeline::{
    Interpolation, KeyId, Keyframe, Playback, PlaybackConfig, PlaybackError, PlaybackFrame,
    PlaybackMode, PlaybackRate, TimelineEdit, TimelineEditor, TimelineError, TimelineEvent, Track,
    TrackId,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::Arc,
    time::Duration,
};

/// A key identity scoped by its stable track ID.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct TimelineKeySelection {
    /// Owning track.
    pub track: TrackId,
    /// Stable key ID.
    pub key: KeyId,
}
/// Logical-pixel view window. Time remains integer until projection to pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimelineWindow {
    /// First visible integer tick.
    pub start: u64,
    /// Visible tick span (nonzero, inclusive end is saturating).
    pub span: u64,
    /// Drawing width, in logical pixels.
    pub width: f32,
    /// First track ID to query.
    pub first_track: TrackId,
    /// Visible track count, at most 32.
    pub rows: usize,
}
impl Default for TimelineWindow {
    fn default() -> Self {
        Self {
            start: 0,
            span: 240,
            width: 640.0,
            first_track: TrackId::from_raw(0),
            rows: 12,
        }
    }
}
impl TimelineWindow {
    pub(crate) fn validate(self) -> Result<(), TimelinePanelError> {
        if self.span == 0
            || !self.width.is_finite()
            || !(64.0..=16_384.0).contains(&self.width)
            || !(1..=32).contains(&self.rows)
        {
            return Err(TimelinePanelError::InvalidValue);
        }
        Ok(())
    }
    pub(crate) fn end(self) -> u64 {
        self.start.saturating_add(self.span)
    }
    pub(crate) fn x(self, tick: u64) -> f32 {
        (tick.saturating_sub(self.start) as f64 / self.span as f64 * f64::from(self.width)) as f32
    }
}
/// Semantic timeline view and authoring commands.
#[derive(Clone, Debug, PartialEq)]
pub enum TimelinePanelAction {
    /// Set the new track name, without altering the document.
    TrackName(String),
    /// Set new track channel count (1 through the editor limit, at most 64).
    TrackDimension(usize),
    /// Add an empty track from the validated creation fields.
    AddTrack,
    /// Delete a track and its keys in one undoable command.
    RemoveTrack(TrackId),
    /// Insert a key at the playhead, using the current sample or zeros.
    InsertKey(TrackId),
    /// Set the point event creation name.
    EventName(String),
    /// Add a named event at the playhead.
    AddEvent,
    /// Delete a point event in one undoable command.
    RemoveEvent(u128),
    /// Set the visible window after checking bounds.
    Window(TimelineWindow),
    /// Seek the playhead without mutating the authoring document.
    Seek(u64),
    /// Start or resume transport. Finished one-shot playback stays finished.
    Play,
    /// Pause without losing the fractional tick or event cursor.
    Pause,
    /// Pause and return to the configured start, rearming its events.
    Stop,
    /// Atomically install validated playback bounds, frequency and event budget.
    /// Pauses transport and clamps its position into the new interval.
    PlaybackConfig(PlaybackConfig),
    /// Change positive rational speed; changing it resets the fractional tick.
    PlaybackRate(PlaybackRate),
    /// Select a key by stable identity.
    Select {
        /// Key identity.
        target: TimelineKeySelection,
        /// Modifier policy. Extend adds an inclusive tick range on the anchor track.
        mode: SelectionMode,
    },
    /// Clear selected keys.
    ClearSelection,
    /// Move the entire selected set by a checked integer tick delta.
    MoveSelection(i64),
    /// Delete selected keys in one undo command.
    DeleteSelection,
    /// Replace one selected key's time.
    KeyTick(u64),
    /// Replace one selected key's numeric vector.
    KeyValues(Vec<f64>),
    /// Replace one selected key's interpolation.
    KeyInterpolation(Interpolation),
    /// Replace one selected key's incoming/outgoing slopes, value per tick.
    KeyTangents {
        /// Incoming slopes.
        incoming: Vec<f64>,
        /// Outgoing slopes.
        outgoing: Vec<f64>,
    },
    /// Choose a numeric channel for the curve editor.
    Channel(usize),
    /// Undo a complete authoring command.
    Undo,
    /// Redo a complete authoring command.
    Redo,
}
/// A rejected event or authoring transaction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TimelinePanelError {
    /// Invalid numeric, dimension, range or selection size.
    InvalidValue,
    /// Target or trigger is absent in the displayed view.
    UnboundEvent,
    /// View belongs to a different controller state or document snapshot.
    StaleView,
    /// Underlying authoring command failed without partial changes.
    Timeline(TimelineError),
    /// Invalid transport request or atomic advancement failure.
    Playback(PlaybackError),
}
impl fmt::Display for TimelinePanelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidValue => f.write_str("invalid timeline value"),
            Self::UnboundEvent => f.write_str("unbound timeline event"),
            Self::StaleView => f.write_str("timeline view is stale"),
            Self::Timeline(error) => error.fmt(f),
            Self::Playback(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for TimelinePanelError {}
impl From<TimelineError> for TimelinePanelError {
    fn from(error: TimelineError) -> Self {
        Self::Timeline(error)
    }
}
impl From<PlaybackError> for TimelinePanelError {
    fn from(error: PlaybackError) -> Self {
        Self::Playback(error)
    }
}

/// View state and command routing. The editor remains the sole document owner.
pub struct TimelinePanelController {
    pub(crate) track_name: String,
    pub(crate) track_dimension: usize,
    pub(crate) event_name: String,
    pub(crate) window: TimelineWindow,
    pub(crate) playback: Playback,
    pub(crate) selection: BTreeSet<TimelineKeySelection>,
    pub(crate) anchor: Option<TimelineKeySelection>,
    pub(crate) channel: usize,
    pub(crate) identity: Arc<()>,
}
impl Default for TimelinePanelController {
    fn default() -> Self {
        Self {
            track_name: "Track".into(),
            track_dimension: 1,
            event_name: "Event".into(),
            window: TimelineWindow::default(),
            playback: Playback::new(PlaybackConfig {
                ticks_per_second: 60,
                start_tick: 0,
                end_tick: u64::MAX,
                mode: PlaybackMode::Once,
                max_events: 1024,
            })
            .expect("nonzero frequency and ordered closed interval"),
            selection: BTreeSet::new(),
            anchor: None,
            channel: 0,
            identity: Arc::new(()),
        }
    }
}
impl TimelinePanelController {
    /// Current bounded logical view window.
    #[must_use]
    pub const fn window(&self) -> TimelineWindow {
        self.window
    }
    /// Current playhead tick.
    #[must_use]
    pub const fn playhead(&self) -> u64 {
        self.playback.tick()
    }
    /// Current transport state, bounds, speed and fractional tick.
    #[must_use]
    pub const fn playback(&self) -> &Playback {
        &self.playback
    }
    /// Advance from the host's monotonic clock and return actual samples and
    /// crossed events to its runtime consumer. Failure consumes no elapsed time;
    /// retry the same interval in smaller steps or explicitly stop playback.
    /// The authoring document and undo history are never changed.
    pub fn advance<'a>(
        &mut self,
        editor: &'a TimelineEditor,
        elapsed: Duration,
    ) -> Result<PlaybackFrame<'a>, TimelinePanelError> {
        let before = self.playback.clone();
        let frame = self.playback.advance(editor.timeline(), elapsed)?;
        if self.playback != before {
            self.identity = Arc::new(());
        }
        Ok(frame)
    }
    /// Selected key identities in deterministic order.
    #[must_use]
    pub fn selection(&self) -> &BTreeSet<TimelineKeySelection> {
        &self.selection
    }
    /// Samples the actual document for a runtime adapter, without authoring writes.
    pub fn samples(
        &self,
        editor: &TimelineEditor,
    ) -> Result<BTreeMap<TrackId, Vec<f64>>, TimelinePanelError> {
        Ok(editor.timeline().sample(self.playhead())?)
    }
    /// Builds actual keyed Loom tracks and curve points. At most 32 tracks,
    /// 1024 key markers and 128 curve samples are materialized per view.
    pub fn build(
        &mut self,
        editor: &TimelineEditor,
    ) -> Result<BoundTimelinePanel, TimelinePanelError> {
        self.window.validate()?;
        self.selection
            .retain(|target| lookup(editor, *target).is_ok());
        if self
            .anchor
            .is_some_and(|target| lookup(editor, target).is_err())
        {
            self.anchor = None;
        }
        crate::timeline_view::build(self, editor)
    }
    /// Validates exact document/controller freshness before resolving the event.
    pub fn handle_event(
        &mut self,
        editor: &mut TimelineEditor,
        view: &BoundTimelinePanel,
        target: &str,
        event: &UiEvent,
    ) -> Result<(), TimelinePanelError> {
        if !Arc::ptr_eq(&self.identity, &view.identity) || view.token != editor.snapshot_token() {
            return Err(TimelinePanelError::StaleView);
        }
        let action = view.dispatch(target, event)?;
        self.apply(editor, action)
    }
    /// Applies a validated semantic command. Document writes use one reversible
    /// transaction; selection, seek and zoom remain view state.
    pub fn apply(
        &mut self,
        editor: &mut TimelineEditor,
        action: TimelinePanelAction,
    ) -> Result<(), TimelinePanelError> {
        use TimelinePanelAction as A;
        match action {
            A::TrackName(name) => {
                validate_name(&name, editor)?;
                self.track_name = name;
            }
            A::TrackDimension(dimension) => {
                if dimension == 0 || dimension > editor.limits().max_dimension.min(64) {
                    return Err(TimelinePanelError::InvalidValue);
                }
                self.track_dimension = dimension;
            }
            A::AddTrack => {
                let id = editor.timeline().next_track_id()?;
                let track = Track::new(id, self.track_name.clone(), self.track_dimension)?;
                editor.execute("Add track", vec![TimelineEdit::AddTrack(track)])?;
            }
            A::RemoveTrack(id) => {
                editor.execute("Delete track", vec![TimelineEdit::RemoveTrack(id)])?;
                self.selection.retain(|key| key.track != id);
                if self.anchor.is_some_and(|key| key.track == id) {
                    self.anchor = None;
                }
            }
            A::InsertKey(id) => {
                let track = editor
                    .timeline()
                    .track(id)
                    .ok_or(TimelineError::MissingTrack(id))?;
                let key_id = track.next_key_id()?;
                let values = track
                    .sample(self.playhead())?
                    .unwrap_or_else(|| vec![0.0; track.dimension()]);
                let key = Keyframe::new(key_id, self.playhead(), values, Interpolation::Linear);
                editor.execute(
                    "Insert keyframe",
                    vec![TimelineEdit::PutKey { track: id, key }],
                )?;
                let target = TimelineKeySelection {
                    track: id,
                    key: key_id,
                };
                self.selection.clear();
                self.selection.insert(target);
                self.anchor = Some(target);
                self.channel = 0;
            }
            A::EventName(name) => {
                validate_name(&name, editor)?;
                self.event_name = name;
            }
            A::AddEvent => {
                let event = TimelineEvent {
                    id: editor.timeline().next_event_id()?,
                    tick: self.playhead(),
                    name: self.event_name.clone(),
                };
                editor.execute("Add timeline event", vec![TimelineEdit::AddEvent(event)])?;
            }
            A::RemoveEvent(id) => {
                editor.execute("Delete timeline event", vec![TimelineEdit::RemoveEvent(id)])?;
            }
            A::Window(window) => {
                window.validate()?;
                self.window = window;
            }
            A::Seek(tick) => {
                self.playback.seek(editor.timeline(), tick)?;
            }
            A::Play => {
                self.playback.play();
            }
            A::Pause => self.playback.pause(),
            A::Stop => {
                let mut next = self.playback.clone();
                next.seek(editor.timeline(), next.config().start_tick)?;
                next.pause();
                self.playback = next;
            }
            A::PlaybackConfig(config) => {
                let mut next = Playback::new(config)?;
                next.set_rate(self.playback.rate());
                next.seek(
                    editor.timeline(),
                    self.playhead().clamp(config.start_tick, config.end_tick),
                )?;
                self.playback = next;
            }
            A::PlaybackRate(rate) => self.playback.set_rate(rate),
            A::ClearSelection => {
                self.selection.clear();
                self.anchor = None;
            }
            A::Select { target, mode } => {
                let target_tick = lookup(editor, target)?.tick();
                let mut selected = self.selection.clone();
                match mode {
                    SelectionMode::Replace => {
                        selected.clear();
                        selected.insert(target);
                    }
                    SelectionMode::Add => {
                        selected.insert(target);
                    }
                    SelectionMode::Toggle => {
                        if !selected.remove(&target) {
                            selected.insert(target);
                        }
                    }
                    SelectionMode::Extend => {
                        let anchor = self
                            .anchor
                            .filter(|anchor| anchor.track == target.track)
                            .unwrap_or(target);
                        let anchor_tick = lookup(editor, anchor)?.tick();
                        let track = editor
                            .timeline()
                            .track(target.track)
                            .ok_or(TimelineError::MissingTrack(target.track))?;
                        for key in track
                            .keys_in(anchor_tick.min(target_tick)..=anchor_tick.max(target_tick))
                            .take(1025)
                        {
                            selected.insert(TimelineKeySelection {
                                track: target.track,
                                key: key.id(),
                            });
                        }
                    }
                }
                if selected.len() > 1024 {
                    return Err(TimelinePanelError::InvalidValue);
                }
                self.selection = selected;
                if mode != SelectionMode::Extend {
                    self.anchor = Some(target);
                }
                self.channel = 0;
            }
            A::MoveSelection(delta) => {
                if self.selection.is_empty() {
                    return Err(TimelinePanelError::InvalidValue);
                }
                if delta != 0 {
                    let mut removes = Vec::with_capacity(self.selection.len() * 2);
                    let mut puts = Vec::with_capacity(self.selection.len());
                    for target in &self.selection {
                        let old = lookup(editor, *target)?;
                        let tick = if delta < 0 {
                            old.tick().checked_sub(delta.unsigned_abs())
                        } else {
                            old.tick().checked_add(delta as u64)
                        }
                        .ok_or(TimelinePanelError::InvalidValue)?;
                        removes.push(TimelineEdit::RemoveKey {
                            track: target.track,
                            key: target.key,
                        });
                        puts.push(TimelineEdit::PutKey {
                            track: target.track,
                            key: modified(old, Some(tick), None, None, None)?,
                        });
                    }
                    removes.extend(puts);
                    editor.execute("Move selected keyframes", removes)?;
                }
            }
            A::DeleteSelection => {
                let edits = self
                    .selection
                    .iter()
                    .map(|target| TimelineEdit::RemoveKey {
                        track: target.track,
                        key: target.key,
                    })
                    .collect();
                editor.execute("Delete selected keyframes", edits)?;
                self.selection.clear();
                self.anchor = None;
            }
            A::KeyTick(tick) => {
                self.edit_key(editor, |key| modified(key, Some(tick), None, None, None))?;
            }
            A::KeyValues(values) => {
                self.edit_key(editor, |key| modified(key, None, Some(values), None, None))?;
            }
            A::KeyInterpolation(interpolation) => self.edit_key(editor, |key| {
                modified(key, None, None, Some(interpolation), None)
            })?,
            A::KeyTangents { incoming, outgoing } => self.edit_key(editor, |key| {
                modified(key, None, None, None, Some((incoming, outgoing)))
            })?,
            A::Channel(channel) => {
                let target = self.single()?;
                if channel >= lookup(editor, target)?.values().len() {
                    return Err(TimelinePanelError::InvalidValue);
                }
                self.channel = channel;
            }
            A::Undo => {
                editor.undo()?;
            }
            A::Redo => {
                editor.redo()?;
            }
        }
        self.identity = Arc::new(());
        Ok(())
    }
    pub(crate) fn single(&self) -> Result<TimelineKeySelection, TimelinePanelError> {
        if self.selection.len() != 1 {
            return Err(TimelinePanelError::InvalidValue);
        }
        self.selection
            .first()
            .copied()
            .ok_or(TimelinePanelError::InvalidValue)
    }
    fn edit_key(
        &mut self,
        editor: &mut TimelineEditor,
        edit: impl FnOnce(&Keyframe) -> Result<Keyframe, TimelineError>,
    ) -> Result<(), TimelinePanelError> {
        let target = self.single()?;
        let old = lookup(editor, target)?;
        let key = edit(old)?;
        if key != *old {
            editor.execute(
                "Edit keyframe",
                vec![TimelineEdit::PutKey {
                    track: target.track,
                    key,
                }],
            )?;
        }
        Ok(())
    }
}
pub(crate) fn lookup(
    editor: &TimelineEditor,
    target: TimelineKeySelection,
) -> Result<&Keyframe, TimelineError> {
    editor
        .timeline()
        .track(target.track)
        .ok_or(TimelineError::MissingTrack(target.track))?
        .key(target.key)
        .ok_or(TimelineError::MissingKey(target.key))
}
fn modified(
    old: &Keyframe,
    tick: Option<u64>,
    values: Option<Vec<f64>>,
    interpolation: Option<Interpolation>,
    tangents: Option<(Vec<f64>, Vec<f64>)>,
) -> Result<Keyframe, TimelineError> {
    let mut key = Keyframe::new(
        old.id(),
        tick.unwrap_or(old.tick()),
        values.unwrap_or_else(|| old.values().to_vec()),
        interpolation.unwrap_or(old.interpolation()),
    );
    let (incoming, outgoing) =
        tangents.unwrap_or_else(|| (old.incoming().to_vec(), old.outgoing().to_vec()));
    key.set_tangents(incoming, outgoing)?;
    Ok(key)
}
fn validate_name(name: &str, editor: &TimelineEditor) -> Result<(), TimelinePanelError> {
    if name.trim().is_empty()
        || name.len()
            > editor
                .limits()
                .max_text_bytes
                .min(crate::MAX_ACTION_TEXT_BYTES)
    {
        return Err(TimelinePanelError::InvalidValue);
    }
    Ok(())
}
