//! Owned-project subsequence authoring and bounded real evaluation diagnostics.
use crate::{sequence_view::BoundSequencePanel, UiEvent};
use prism_editor_timeline::{
    PlaybackMode, PlaybackRate, SequenceClip, SequenceClipId, SequenceDocumentError, SequenceError,
    SequenceId, SequenceInstance, SequencePlayer, SequenceProjectCommand, SequenceProjectEdit,
    SequenceProjectEditor, SequenceProjectToken, SequenceRange, Timeline, TimelineDocument,
};
use std::{fmt, sync::Arc, time::Duration};

/// Hard bounded view and evaluation policy. No topology is silently truncated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequencePanelLimits {
    /// Rows in each independently paged list, 1 through 32.
    pub rows: usize,
    /// Maximum actual evaluation events and samples, 1 through 128 each.
    pub diagnostics: usize,
    /// Maximum real registry traversal work, 1 through 65536.
    pub visits: usize,
}
impl Default for SequencePanelLimits {
    fn default() -> Self {
        Self {
            rows: 12,
            diagnostics: 128,
            visits: 4096,
        }
    }
}
impl SequencePanelLimits {
    fn check(self) -> Result<(), SequencePanelError> {
        if !(1..=32).contains(&self.rows)
            || !(1..=128).contains(&self.diagnostics)
            || !(1..=65536).contains(&self.visits)
        {
            return Err(SequencePanelError::InvalidValue);
        }
        Ok(())
    }
}
/// Staged fields; only SaveClip/AddClip submits an atomic project command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceClipDraft {
    /// Actual child document identity, never a list index.
    pub source: Option<SequenceId>,
    /// Half-open parent interval start.
    pub parent_start: u64,
    /// Half-open parent interval end.
    pub parent_end: u64,
    /// Half-open source interval start.
    pub source_start: u64,
    /// Half-open source interval end.
    pub source_end: u64,
    /// Positive rational numerator.
    pub numerator: u64,
    /// Positive rational denominator.
    pub denominator: u64,
    /// Number of source repeats, clipped by the parent range.
    pub repetitions: u64,
}
impl Default for SequenceClipDraft {
    fn default() -> Self {
        Self {
            source: None,
            parent_start: 0,
            parent_end: 60,
            source_start: 0,
            source_end: 60,
            numerator: 1,
            denominator: 1,
            repetitions: 1,
        }
    }
}
impl SequenceClipDraft {
    fn from_clip(clip: &SequenceClip) -> Self {
        let (numerator, denominator) = clip.rate.ratio();
        Self {
            source: Some(clip.source),
            parent_start: clip.parent.start(),
            parent_end: clip.parent.end(),
            source_start: clip.source_range.start(),
            source_end: clip.source_range.end(),
            numerator,
            denominator,
            repetitions: clip.repetitions,
        }
    }
    fn clip(self, id: SequenceClipId) -> Result<SequenceClip, SequencePanelError> {
        Ok(SequenceClip {
            id,
            source: self.source.ok_or(SequencePanelError::InvalidValue)?,
            parent: SequenceRange::new(self.parent_start, self.parent_end)?,
            source_range: SequenceRange::new(self.source_start, self.source_end)?,
            rate: PlaybackRate::new(self.numerator, self.denominator)
                .map_err(SequenceError::from)?,
            repetitions: self.repetitions,
        })
    }
}
/// Numeric draft field, represented by a real commit control.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SequenceClipField {
    /// Parent inclusive start.
    ParentStart,
    /// Parent exclusive end.
    ParentEnd,
    /// Source inclusive start.
    SourceStart,
    /// Source exclusive end.
    SourceEnd,
    /// Rational numerator.
    Numerator,
    /// Rational denominator.
    Denominator,
    /// Source repeat count.
    Repetitions,
}
/// Independent stable-order page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SequencePage {
    /// Owned document list.
    Documents,
    /// Clips belonging to the selected document.
    Clips,
    /// Incoming clip references.
    Parents,
    /// Crossed event diagnostics.
    Events,
    /// Actual destination sample diagnostics.
    Samples,
}
/// Explicit evaluation interval; uses the actual root clock and integer kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequencePreview {
    /// Root range inclusive start.
    pub start: u64,
    /// Root range exclusive end.
    pub end: u64,
    /// Seek destination before playing; seeking itself emits no crossed events.
    pub seek: u64,
    /// Actual elapsed nanoseconds to advance.
    pub elapsed_ns: u64,
    /// Root looping (separate from clip repetitions).
    pub looping: bool,
}
impl Default for SequencePreview {
    fn default() -> Self {
        Self {
            start: 0,
            end: 240,
            seek: 0,
            elapsed_ns: 1_000_000_000,
            looping: false,
        }
    }
}
/// Owned, bounded identity/clock information from one real crossed event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SequenceEventDiagnostic {
    /// Actual document identity.
    pub sequence: SequenceId,
    /// Actual event identity.
    pub event: u128,
    /// Bounded display of the real name (at most 120 characters).
    pub name: String,
    /// Source event tick.
    pub tick: u64,
    /// Mapped root tick.
    pub root_tick: u64,
    /// Root loop iteration.
    pub root_iteration: u128,
    /// Actual nested clip/iteration route.
    pub path: Vec<SequenceInstance>,
}
/// Destination sample identity from actual SequencePlayer sampling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SequenceSampleDiagnostic {
    /// Actual document identity.
    pub sequence: SequenceId,
    /// Actual local tick after integer rational mapping.
    pub tick: u64,
    /// Number of actual sampled tracks.
    pub tracks: usize,
    /// Actual nested clip/iteration route.
    pub path: Vec<SequenceInstance>,
}
/// All mutations are explicit controller actions; no Element mutates the model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SequencePanelAction {
    /// Open an owned document.
    Open(SequenceId),
    /// Navigate to the authored root.
    OpenRoot,
    /// Choose the opened document as project root.
    SetRoot,
    /// Create a real empty document with its own stable ID and 60 Hz clock.
    AddDocument,
    /// Remove the opened document; referenced documents reject atomically.
    RemoveDocument,
    /// Change the opened document's actual frequency.
    Frequency(u64),
    /// Locate a clip in its owning document and load its inspector.
    Locate(SequenceClipId),
    /// Navigate through a clip to its actual child document.
    OpenChild(SequenceClipId),
    /// Stage a new clip without modifying the project.
    NewClip,
    /// Stage a child document identity.
    Source(SequenceId),
    /// Stage one numeric field (ranges validated together when applied).
    Field(SequenceClipField, u64),
    /// Insert a fresh globally identified clip from the draft.
    AddClip,
    /// Replace the selected clip from the draft in one inverse command.
    SaveClip,
    /// Remove the selected clip.
    RemoveClip,
    /// Set one bounded list's zero-based page.
    Page(SequencePage, usize),
    /// Set explicit real evaluation parameters.
    Preview(SequencePreview),
    /// Seek/play/advance the real nested evaluator and retain bounded diagnostics.
    Evaluate,
    /// Undo the last whole project command.
    Undo,
    /// Redo the last undone whole project command.
    Redo,
}
/// Invalid input, stale event, or real project/evaluation rejection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SequencePanelError {
    /// Invalid field, identity, page, or policy.
    InvalidValue,
    /// No matching painted control/trigger in this view.
    UnboundEvent,
    /// Controller state or exact project snapshot has changed.
    StaleView,
    /// Atomic authoring/registry validation rejected the request.
    Document(SequenceDocumentError),
    /// Actual sampling/event evaluation failed or exceeded its budget.
    Sequence(SequenceError),
}
impl From<SequenceDocumentError> for SequencePanelError {
    fn from(value: SequenceDocumentError) -> Self {
        Self::Document(value)
    }
}
impl From<SequenceError> for SequencePanelError {
    fn from(value: SequenceError) -> Self {
        Self::Sequence(value)
    }
}
impl fmt::Display for SequencePanelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sequence panel {self:?}")
    }
}
impl std::error::Error for SequencePanelError {}

/// Authoring UI state; the host owns the actual SequenceProjectEditor.
pub struct SequencePanelController {
    pub(crate) limits: SequencePanelLimits,
    pub(crate) selected: Option<SequenceId>,
    pub(crate) clip: Option<SequenceClipId>,
    pub(crate) draft: SequenceClipDraft,
    pub(crate) pages: [usize; 5],
    pub(crate) preview: SequencePreview,
    pub(crate) events: Vec<SequenceEventDiagnostic>,
    pub(crate) samples: Vec<SequenceSampleDiagnostic>,
    pub(crate) evaluated_tick: Option<u64>,
    pub(crate) error: Option<SequencePanelError>,
    pub(crate) identity: Arc<()>,
    observed: Option<SequenceProjectToken>,
}
impl Default for SequencePanelController {
    fn default() -> Self {
        Self::new(SequencePanelLimits::default()).expect("valid bounded defaults")
    }
}
impl SequencePanelController {
    /// Validate a view/evaluation budget before allocating any view.
    pub fn new(limits: SequencePanelLimits) -> Result<Self, SequencePanelError> {
        limits.check()?;
        Ok(Self {
            limits,
            selected: None,
            clip: None,
            draft: SequenceClipDraft::default(),
            pages: [0; 5],
            preview: SequencePreview::default(),
            events: Vec::new(),
            samples: Vec::new(),
            evaluated_tick: None,
            error: None,
            identity: Arc::new(()),
            observed: None,
        })
    }
    /// Current selected document identity.
    pub const fn selected(&self) -> Option<SequenceId> {
        self.selected
    }
    /// Current globally stable clip identity.
    pub const fn selected_clip(&self) -> Option<SequenceClipId> {
        self.clip
    }
    /// Current staged inspector values.
    pub const fn draft(&self) -> SequenceClipDraft {
        self.draft
    }
    /// Current explicit evaluation parameters.
    pub const fn preview(&self) -> SequencePreview {
        self.preview
    }
    /// Last actual evaluation events. Cleared on any project/preview change.
    pub fn events(&self) -> &[SequenceEventDiagnostic] {
        &self.events
    }
    /// Last actual destination samples. Cleared on any project/preview change.
    pub fn samples(&self) -> &[SequenceSampleDiagnostic] {
        &self.samples
    }
    /// Last rejected authoring/evaluation request, rendered in the next view.
    pub fn error(&self) -> Option<&SequencePanelError> {
        self.error.as_ref()
    }
    /// Build a bounded real Loom tree. External project changes clear old diagnostics.
    pub fn build(
        &mut self,
        editor: &SequenceProjectEditor,
    ) -> Result<BoundSequencePanel, SequencePanelError> {
        self.sync(editor);
        crate::sequence_view::build(self, editor)
    }
    /// Enforce controller identity and exact snapshot before routing a semantic event.
    pub fn handle_event(
        &mut self,
        editor: &mut SequenceProjectEditor,
        view: &BoundSequencePanel,
        target: &str,
        event: &UiEvent,
    ) -> Result<(), SequencePanelError> {
        if !Arc::ptr_eq(&self.identity, &view.identity) || view.token != editor.snapshot_token() {
            return Err(SequencePanelError::StaleView);
        }
        match view.dispatch(target, event) {
            Ok(action) => self.apply(editor, action),
            Err(error) => {
                self.error = Some(error.clone());
                self.identity = Arc::new(());
                Err(error)
            }
        }
    }
    /// Apply an explicit action. Document writes go exclusively through inverse commands.
    pub fn apply(
        &mut self,
        editor: &mut SequenceProjectEditor,
        action: SequencePanelAction,
    ) -> Result<(), SequencePanelError> {
        self.sync(editor);
        let result = self.perform(editor, action);
        self.sync(editor);
        self.error = result.as_ref().err().cloned();
        self.identity = Arc::new(());
        result
    }
    fn sync(&mut self, editor: &SequenceProjectEditor) {
        let token = editor.snapshot_token();
        if self.observed.as_ref() != Some(&token) {
            self.clear_diagnostics();
            self.error = None;
            self.identity = Arc::new(());
            self.observed = Some(token);
            if self
                .selected
                .is_none_or(|id| editor.document().document(id).is_none())
            {
                self.selected = editor
                    .document()
                    .root()
                    .or_else(|| editor.document().documents().next().map(|(id, _)| id));
                self.pages = [0; 5];
                self.clip = None;
                self.draft = SequenceClipDraft::default();
            }
            if let Some(id) = self.clip {
                if let Some((parent, clip)) =
                    editor.document().clips().find(|(_, clip)| clip.id == id)
                {
                    if self.selected == Some(parent) {
                        self.draft = SequenceClipDraft::from_clip(clip);
                    } else {
                        self.clip = None;
                        self.draft = SequenceClipDraft::default();
                    }
                } else {
                    self.clip = None;
                    self.draft = SequenceClipDraft::default();
                }
            }
            if self
                .draft
                .source
                .is_some_and(|id| editor.document().document(id).is_none())
            {
                self.draft.source = None;
            }
            for page in [
                SequencePage::Documents,
                SequencePage::Clips,
                SequencePage::Parents,
            ] {
                let index = page.index();
                self.pages[index] = self.pages[index]
                    .min(self.count(editor, page).saturating_sub(1) / self.limits.rows);
            }
        }
    }
    fn clear_diagnostics(&mut self) {
        self.events.clear();
        self.samples.clear();
        self.evaluated_tick = None;
        self.pages[3] = 0;
        self.pages[4] = 0;
    }
    pub(crate) fn count(&self, editor: &SequenceProjectEditor, page: SequencePage) -> usize {
        match page {
            SequencePage::Documents => editor.document().documents().count(),
            SequencePage::Clips => editor
                .document()
                .clips()
                .filter(|(p, _)| Some(*p) == self.selected)
                .count(),
            SequencePage::Parents => editor
                .document()
                .clips()
                .filter(|(_, c)| Some(c.source) == self.selected)
                .count(),
            SequencePage::Events => self.events.len(),
            SequencePage::Samples => self.samples.len(),
        }
    }
    fn open(
        &mut self,
        editor: &SequenceProjectEditor,
        id: SequenceId,
    ) -> Result<(), SequencePanelError> {
        if editor.document().document(id).is_none() {
            return Err(SequencePanelError::InvalidValue);
        }
        self.selected = Some(id);
        self.clip = None;
        self.pages[0] = editor
            .document()
            .documents()
            .position(|(other, _)| other == id)
            .ok_or(SequencePanelError::InvalidValue)?
            / self.limits.rows;
        self.pages[1] = 0;
        self.pages[2] = 0;
        self.draft = SequenceClipDraft::default();
        Ok(())
    }
    fn selected_id(&self) -> Result<SequenceId, SequencePanelError> {
        self.selected.ok_or(SequencePanelError::InvalidValue)
    }
    fn command(
        editor: &mut SequenceProjectEditor,
        label: &str,
        edits: Vec<SequenceProjectEdit>,
    ) -> Result<(), SequencePanelError> {
        editor.execute(SequenceProjectCommand {
            label: label.into(),
            edits,
        })?;
        Ok(())
    }
    fn perform(
        &mut self,
        editor: &mut SequenceProjectEditor,
        action: SequencePanelAction,
    ) -> Result<(), SequencePanelError> {
        use SequencePanelAction as A;
        match action {
            A::Open(id) => self.open(editor, id)?,
            A::OpenRoot => self.open(
                editor,
                editor
                    .document()
                    .root()
                    .ok_or(SequencePanelError::InvalidValue)?,
            )?,
            A::SetRoot => Self::command(
                editor,
                "Set sequence root",
                vec![SequenceProjectEdit::SetRoot(Some(self.selected_id()?))],
            )?,
            A::AddDocument => {
                let id = editor.document().next_sequence_id()?;
                let mut edits = vec![SequenceProjectEdit::PutDocument {
                    id,
                    document: TimelineDocument {
                        ticks_per_second: 60,
                        timeline: Timeline::default(),
                    },
                }];
                if editor.document().root().is_none() {
                    edits.push(SequenceProjectEdit::SetRoot(Some(id)));
                }
                Self::command(editor, "Add sequence", edits)?;
                self.open(editor, id)?;
            }
            A::RemoveDocument => {
                let id = self.selected_id()?;
                let mut edits = Vec::new();
                if editor.document().root() == Some(id) {
                    edits.push(SequenceProjectEdit::SetRoot(None));
                }
                edits.push(SequenceProjectEdit::RemoveDocument(id));
                Self::command(
                    editor,
                    "Remove sequence (references must be removed first)",
                    edits,
                )?;
            }
            A::Frequency(ticks_per_second) => Self::command(
                editor,
                "Set sequence clock",
                vec![SequenceProjectEdit::SetFrequency {
                    id: self.selected_id()?,
                    ticks_per_second,
                }],
            )?,
            A::Locate(id) => {
                let (parent, clip) = editor
                    .document()
                    .clips()
                    .find(|(_, c)| c.id == id)
                    .ok_or(SequencePanelError::InvalidValue)?;
                let draft = SequenceClipDraft::from_clip(clip);
                self.open(editor, parent)?;
                self.clip = Some(id);
                self.draft = draft;
                self.pages[1] = editor
                    .document()
                    .clips()
                    .filter(|(p, _)| *p == parent)
                    .position(|(_, c)| c.id == id)
                    .ok_or(SequencePanelError::InvalidValue)?
                    / self.limits.rows;
            }
            A::OpenChild(id) => {
                let source = editor
                    .document()
                    .clips()
                    .find(|(_, c)| c.id == id)
                    .ok_or(SequencePanelError::InvalidValue)?
                    .1
                    .source;
                self.open(editor, source)?;
            }
            A::NewClip => {
                self.clip = None;
                self.draft = SequenceClipDraft::default();
            }
            A::Source(id) => {
                if editor.document().document(id).is_none() {
                    return Err(SequencePanelError::InvalidValue);
                }
                self.draft.source = Some(id);
            }
            A::Field(field, value) => {
                use SequenceClipField as F;
                *match field {
                    F::ParentStart => &mut self.draft.parent_start,
                    F::ParentEnd => &mut self.draft.parent_end,
                    F::SourceStart => &mut self.draft.source_start,
                    F::SourceEnd => &mut self.draft.source_end,
                    F::Numerator => &mut self.draft.numerator,
                    F::Denominator => &mut self.draft.denominator,
                    F::Repetitions => &mut self.draft.repetitions,
                } = value;
            }
            A::AddClip | A::SaveClip => {
                let id = if matches!(action, A::SaveClip) {
                    self.clip.ok_or(SequencePanelError::InvalidValue)?
                } else {
                    editor.document().next_clip_id()?
                };
                Self::command(
                    editor,
                    "Apply subsequence clip",
                    vec![SequenceProjectEdit::PutClip {
                        parent: self.selected_id()?,
                        clip: self.draft.clip(id)?,
                    }],
                )?;
                self.clip = Some(id);
                self.pages[1] = editor
                    .document()
                    .clips()
                    .filter(|(p, _)| Some(*p) == self.selected)
                    .position(|(_, c)| c.id == id)
                    .ok_or(SequencePanelError::InvalidValue)?
                    / self.limits.rows;
            }
            A::RemoveClip => Self::command(
                editor,
                "Remove subsequence clip",
                vec![SequenceProjectEdit::RemoveClip(
                    self.clip.ok_or(SequencePanelError::InvalidValue)?,
                )],
            )?,
            A::Page(page, value) => {
                if value > self.count(editor, page).saturating_sub(1) / self.limits.rows {
                    return Err(SequencePanelError::InvalidValue);
                }
                self.pages[page.index()] = value;
            }
            A::Preview(preview) => {
                SequenceRange::new(preview.start, preview.end)?;
                if !(preview.start..preview.end).contains(&preview.seek) {
                    return Err(SequencePanelError::InvalidValue);
                }
                self.preview = preview;
                self.clear_diagnostics();
            }
            A::Evaluate => self.evaluate(editor)?,
            A::Undo => {
                editor.undo()?;
            }
            A::Redo => {
                editor.redo()?;
            }
        }
        Ok(())
    }
    fn evaluate(&mut self, editor: &SequenceProjectEditor) -> Result<(), SequencePanelError> {
        self.clear_diagnostics();
        let mut limits = editor.limits().sequence;
        limits.max_events = limits.max_events.min(self.limits.diagnostics);
        limits.max_samples = limits.max_samples.min(self.limits.diagnostics);
        limits.max_visits = limits.max_visits.min(self.limits.visits);
        let registry = editor.document().registry(limits)?;
        let root = editor
            .document()
            .root()
            .ok_or(SequencePanelError::InvalidValue)?;
        let p = self.preview;
        let mut player = SequencePlayer::new(
            &registry,
            root,
            SequenceRange::new(p.start, p.end)?,
            if p.looping {
                PlaybackMode::Loop
            } else {
                PlaybackMode::Once
            },
        )?;
        player.seek(p.seek)?;
        player.play();
        let frame = player.advance(Duration::from_nanos(p.elapsed_ns))?;
        // Kernel max_samples counts inspected tracks. Empty documents still
        // produce instance samples, so cap the owned UI output separately.
        if frame.samples.len() > self.limits.diagnostics {
            return Err(SequenceError::Limit("diagnostic sample rows").into());
        }
        self.events = frame
            .events
            .iter()
            .map(|e| SequenceEventDiagnostic {
                sequence: e.sequence,
                event: e.event.id,
                name: e.event.name.chars().take(120).collect(),
                tick: e.event.tick,
                root_tick: e.root_tick,
                root_iteration: e.root_iteration,
                path: e.path.clone(),
            })
            .collect();
        self.samples = frame
            .samples
            .iter()
            .map(|s| SequenceSampleDiagnostic {
                sequence: s.sequence,
                tick: s.tick,
                tracks: s.tracks.len(),
                path: s.path.clone(),
            })
            .collect();
        self.evaluated_tick = Some(frame.tick);
        Ok(())
    }
}
impl SequencePage {
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Documents => 0,
            Self::Clips => 1,
            Self::Parents => 2,
            Self::Events => 3,
            Self::Samples => 4,
        }
    }
}
