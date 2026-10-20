//! Native asynchronous file routes. Loading never implicitly replaces a model.
use super::*;
use prism_ui::style::Keyword;
use std::{
    task::Poll,
    time::{Duration, Instant},
};

/// Cancellation cannot retract a save that already entered publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DesktopPanelCancelOutcome {
    /// No request is owned.
    Idle,
    /// Pre-publication work fenced; retain and poll its terminal result.
    Cancelled,
    /// Publication started; retain and inspect the real eventual receipt.
    TooLate,
}
/// Complete sequence project codec and admission bounds for the shared file worker.
#[derive(Clone, Copy, Debug)]
pub struct DesktopSequenceStorageLimits {
    /// Actual project/document/topology and history budgets.
    pub document: prism_editor_timeline::SequenceDocumentLimits,
    /// Maximum queued/owned requests admitted by the storage service.
    pub max_requests: usize,
}
impl Default for DesktopSequenceStorageLimits {
    fn default() -> Self {
        Self {
            document: prism_editor_timeline::SequenceDocumentLimits::default(),
            max_requests: 2,
        }
    }
}
pub(super) struct FileFailure {
    message: String,
    reconcile: bool,
    cancelled: bool,
    busy: bool,
}
#[derive(Clone, PartialEq)]
pub(super) struct FileSnapshot<T> {
    model: T,
    rate: Option<u64>,
}
pub(super) enum FileOutput<T, D, R> {
    Loaded(D, R),
    Saved {
        token: T,
        revision: R,
        synced: bool,
        late: bool,
    },
}
pub(super) trait StorageBackend {
    type Token: Clone + PartialEq;
    type Revision: Clone;
    type Document;
    type Request;
    type Limits: Copy;
    fn load(&self, timeout: Duration) -> Result<Self::Request, FileFailure>;
    fn save(
        &self,
        bytes: Vec<u8>,
        token: Self::Token,
        revision: Self::Revision,
        timeout: Duration,
    ) -> Result<Self::Request, FileFailure>;
    fn poll(
        request: &mut Self::Request,
    ) -> Poll<Result<FileOutput<Self::Token, Self::Document, Self::Revision>, FileFailure>>;
    fn cancel(request: &Self::Request) -> DesktopPanelCancelOutcome;
    fn matches_receipt(started: &Self::Token, receipt: &Self::Token) -> bool;
}
macro_rules! backend {
    ($module:ident, $service:ident, $token:ident, $revision:ident, $document:ident, $request:ident, $limits:ty, $result:ident, $error:ident, $durability:ident, $cancel:ident) => {
        impl StorageBackend for $module::$service {
            type Token = FileSnapshot<$module::$token>;
            type Revision = $module::$revision;
            type Document = $module::$document;
            type Request = $module::$request;
            type Limits = $limits;
            fn load(&self, timeout: Duration) -> Result<Self::Request, FileFailure> {
                self.load(timeout).map_err(|e| FileFailure {
                    busy: matches!(e, $module::$error::Busy),
                    reconcile: matches!(e, $module::$error::PublicationUncertain { .. }),
                    cancelled: matches!(e, $module::$error::Cancelled),
                    message: e.to_string(),
                })
            }
            fn save(
                &self,
                bytes: Vec<u8>,
                token: Self::Token,
                revision: Self::Revision,
                timeout: Duration,
            ) -> Result<Self::Request, FileFailure> {
                self.save(bytes, token.model, revision, timeout)
                    .map_err(|e| FileFailure {
                        busy: matches!(e, $module::$error::Busy),
                        reconcile: matches!(e, $module::$error::PublicationUncertain { .. }),
                        cancelled: matches!(e, $module::$error::Cancelled),
                        message: e.to_string(),
                    })
            }
            fn poll(
                request: &mut Self::Request,
            ) -> Poll<Result<FileOutput<Self::Token, Self::Document, Self::Revision>, FileFailure>>
            {
                request.poll().map(|result| {
                    result
                        .map(|r| match r {
                            $module::$result::Loaded(c) => {
                                FileOutput::Loaded(c.document, c.revision)
                            }
                            $module::$result::Saved(r) => FileOutput::Saved {
                                token: FileSnapshot {
                                    model: r.snapshot_token,
                                    rate: None,
                                },
                                revision: r.revision,
                                synced: matches!(r.durability, $module::$durability::Synced),
                                late: r.deadline_exceeded,
                            },
                        })
                        .map_err(|e| FileFailure {
                            busy: matches!(e, $module::$error::Busy),
                            reconcile: matches!(e, $module::$error::PublicationUncertain { .. }),
                            cancelled: matches!(e, $module::$error::Cancelled),
                            message: e.to_string(),
                        })
                })
            }
            fn cancel(request: &Self::Request) -> DesktopPanelCancelOutcome {
                match request.cancel() {
                    $module::$cancel::Cancelled => DesktopPanelCancelOutcome::Cancelled,
                    $module::$cancel::TooLate => DesktopPanelCancelOutcome::TooLate,
                }
            }
            fn matches_receipt(started: &Self::Token, receipt: &Self::Token) -> bool {
                started.model == receipt.model
            }
        }
    };
}
backend!(
    prism_editor_graph,
    GraphStorageService,
    GraphSnapshotToken,
    GraphFileRevision,
    GraphDocument,
    GraphStorageRequest,
    prism_editor_graph::GraphStorageLimits,
    GraphStorageResult,
    GraphStorageError,
    GraphSaveDurability,
    GraphCancelOutcome
);
backend!(
    prism_editor_timeline,
    TimelineStorageService,
    TimelineSnapshotToken,
    TimelineFileRevision,
    TimelineDocument,
    TimelineStorageRequest,
    prism_editor_timeline::TimelineStorageLimits,
    TimelineStorageResult,
    TimelineStorageError,
    TimelineSaveDurability,
    TimelineCancelOutcome
);
backend!(
    prism_editor_timeline,
    SequenceProjectStorageService,
    SequenceProjectToken,
    TimelineFileRevision,
    SequenceProjectDocument,
    SequenceProjectStorageRequest,
    DesktopSequenceStorageLimits,
    SequenceProjectStorageResult,
    TimelineStorageError,
    TimelineSaveDurability,
    TimelineCancelOutcome
);

enum FileOperation<R> {
    Load,
    Save { bytes: Vec<u8>, expected: R },
}
struct FilePlan<S: StorageBackend> {
    operation: FileOperation<S::Revision>,
    token: S::Token,
    deadline: Instant,
    attempts: u8,
    retry_at: Instant,
}
pub(super) struct PanelFile<S: StorageBackend> {
    service: S,
    path: std::path::PathBuf,
    pub(super) limits: S::Limits,
    expected: Option<S::Revision>,
    timeout: Duration,
    active: Option<(S::Request, S::Token)>,
    plan: Option<FilePlan<S>>,
    pub(super) candidate: Option<(S::Document, S::Revision, S::Token)>,
    failed: bool,
    status: String,
    cancelling: bool,
}
impl<S: StorageBackend> PanelFile<S> {
    fn new(
        service: S,
        limits: S::Limits,
        expected: S::Revision,
        timeout: Duration,
        path: std::path::PathBuf,
    ) -> Self {
        Self {
            service,
            path,
            limits,
            expected: Some(expected),
            timeout,
            active: None,
            plan: None,
            candidate: None,
            failed: false,
            status: "File binding ready".into(),
            cancelling: false,
        }
    }
    pub(super) fn fenced(&self) -> bool {
        self.plan.is_some()
            || self.active.is_some()
            || self.candidate.is_some()
            || self.failed
            || self.expected.is_none()
    }
    pub(super) fn next_wake(&self) -> Option<Instant> {
        if self.active.is_some() {
            Some(Instant::now() + Duration::from_millis(10))
        } else {
            self.plan.as_ref().map(|p| p.retry_at)
        }
    }
    fn available(&self) -> Result<(), String> {
        if self.plan.is_some() || self.active.is_some() || self.candidate.is_some() {
            Err("Finish I/O or resolve the loaded candidate first".into())
        } else {
            Ok(())
        }
    }
    fn failure(&mut self, error: FileFailure) -> String {
        if error.reconcile {
            self.expected = None;
        }
        self.failed = !(self.cancelling && error.cancelled);
        self.status = format!("Panel file {}: {}", self.path.display(), error.message);
        self.cancelling = false;
        self.status.clone()
    }
    fn save(&mut self, bytes: Vec<u8>, token: S::Token) -> Result<(), String> {
        self.available()?;
        let expected = self
            .expected
            .clone()
            .ok_or("Publication uncertain: load and reconcile the disk revision first")?;
        self.start(FileOperation::Save { bytes, expected }, token)
    }
    fn load(&mut self, token: S::Token) -> Result<(), String> {
        self.available()?;
        self.start(FileOperation::Load, token)
    }
    fn start(
        &mut self,
        operation: FileOperation<S::Revision>,
        token: S::Token,
    ) -> Result<(), String> {
        let now = Instant::now();
        let deadline = now
            .checked_add(self.timeout)
            .ok_or("Invalid storage deadline")?;
        self.plan = Some(FilePlan {
            operation,
            token,
            deadline,
            attempts: 0,
            retry_at: now,
        });
        self.failed = false;
        self.cancelling = false;
        self.submit_plan()
    }
    fn submit_plan(&mut self) -> Result<(), String> {
        let plan = self.plan.as_mut().expect("retained operation");
        plan.attempts += 1;
        let remaining = plan.deadline.saturating_duration_since(Instant::now());
        // Keep the encoded snapshot only during this bounded operation. Retries
        // never recapture a newer model or change the expected disk revision.
        let (result, status) = match &plan.operation {
            FileOperation::Load => (self.service.load(remaining), "Loading candidate"),
            FileOperation::Save { bytes, expected } => (
                self.service.save(
                    bytes.clone(),
                    plan.token.clone(),
                    expected.clone(),
                    remaining,
                ),
                "Saving",
            ),
        };
        match result {
            Ok(request) => {
                self.active = Some((request, plan.token.clone()));
                self.status = status.into();
                Ok(())
            }
            Err(error) => self.retry_failure(error),
        }
    }
    fn retry_failure(&mut self, error: FileFailure) -> Result<(), String> {
        // Busy is pre-publication. Never retry conflicts, cancellation, unsafe
        // paths, codec errors, or an uncertain publication. No UI thread sleep.
        if error.busy
            && !self.cancelling
            && let Some(plan) = &mut self.plan
        {
            let next = Instant::now() + Duration::from_millis(1 << (plan.attempts - 1));
            if plan.attempts < 8 && next < plan.deadline {
                plan.retry_at = next;
                self.status = format!(
                    "Panel file {}: {}; retry {}/8 pending",
                    self.path.display(),
                    error.message,
                    plan.attempts + 1
                );
                return Ok(());
            }
        }
        self.plan = None;
        Err(self.failure(error))
    }
    pub(super) fn poll(&mut self) -> Result<(bool, Option<S::Token>), String> {
        let Some((request, _)) = &mut self.active else {
            if self
                .plan
                .as_ref()
                .is_some_and(|p| Instant::now() >= p.retry_at)
            {
                self.submit_plan()?;
                return Ok((true, None));
            }
            return Ok((false, None));
        };
        let Poll::Ready(result) = S::poll(request) else {
            return Ok((false, None));
        };
        let (_, started) = self.active.take().expect("polled request");
        match result {
            Ok(FileOutput::Loaded(document, revision)) => {
                self.plan = None;
                self.candidate = Some((document, revision, started));
                self.status = "Loaded candidate: review before adoption".into();
                self.cancelling = false;
                Ok((true, None))
            }
            Ok(FileOutput::Saved {
                token,
                revision,
                synced,
                late,
            }) => {
                self.plan = None;
                if !S::matches_receipt(&started, &token) {
                    self.failed = true;
                    self.expected = None;
                    self.status = "Mismatched save receipt: reconcile disk before retry".into();
                    return Err(self.status.clone());
                }
                self.expected = Some(revision);
                self.failed = !synced;
                self.cancelling = false;
                self.status = if !synced {
                    "Published; directory durability uncertain. Retry save or reconcile"
                } else if late {
                    "Saved durably after deadline"
                } else {
                    "Saved durably"
                }
                .into();
                if synced {
                    Ok((true, Some(started)))
                } else {
                    Err(self.status.clone())
                }
            }
            Err(error) => {
                let cancelled = self.cancelling && error.cancelled;
                if cancelled {
                    self.plan = None;
                    self.failure(error);
                    Ok((true, None))
                } else {
                    self.retry_failure(error)?;
                    Ok((true, None))
                }
            }
        }
    }
    fn cancel(&mut self) -> DesktopPanelCancelOutcome {
        let Some((request, _)) = &self.active else {
            if self.plan.take().is_some() {
                self.cancelling = false;
                self.status = "Cancelled before retry; no pending publication".into();
                return DesktopPanelCancelOutcome::Cancelled;
            }
            return DesktopPanelCancelOutcome::Idle;
        };
        let outcome = S::cancel(request);
        self.cancelling = true;
        self.status = if outcome == DesktopPanelCancelOutcome::TooLate {
            "Cancellation too late: awaiting publication outcome"
        } else {
            "Cancelling: awaiting terminal result"
        }
        .into();
        outcome
    }
    fn discard(&mut self) {
        self.candidate = None;
        self.status = "Loaded candidate discarded; live document retained".into();
    }
    pub(super) fn adopted(&mut self) {
        let (_, revision, _) = self.candidate.take().expect("validated candidate");
        self.expected = Some(revision);
        self.failed = false;
        self.status = "Loaded document adopted".into();
    }
    pub(super) fn decorate(
        &self,
        prefix: &str,
        view: &mut DesktopPanelView,
        editable: bool,
        clean: bool,
    ) {
        let mut tools = Element::box_()
            .style(StyleProp::FlexDirection, StyleValue::keyword(Keyword::Row))
            .style(StyleProp::FlexShrink, StyleValue::number(0.0));
        for (name, label, enabled) in [
            (
                "save",
                "Save panel",
                editable
                    && self.active.is_none()
                    && self.plan.is_none()
                    && self.candidate.is_none()
                    && self.expected.is_some(),
            ),
            (
                "load",
                "Load candidate",
                editable
                    && self.active.is_none()
                    && self.plan.is_none()
                    && self.candidate.is_none(),
            ),
            (
                "cancel",
                "Cancel I/O",
                self.active.is_some() || self.plan.is_some(),
            ),
            (
                "adopt",
                "Adopt candidate",
                editable && clean && self.candidate.is_some(),
            ),
            ("discard", "Discard candidate", self.candidate.is_some()),
        ] {
            let key = format!("{prefix}:file:{name}");
            if enabled {
                view.bindings.push((key.clone(), UiTrigger::Activate));
            }
            tools = tools.child(Element::box_().key_str(key).class("desktop-button").child(
                Element::text(format!(
                    "{label}{}",
                    if enabled { "" } else { " (disabled)" }
                )),
            ));
        }
        view.element = Element::box_()
            .style(
                StyleProp::FlexDirection,
                StyleValue::keyword(Keyword::Column),
            )
            .child(tools)
            .child(Element::text(&self.status))
            .child(view.element.clone());
    }
}

macro_rules! panel_file_api {
    ($panel:ident,$prefix:literal,$module:ident,$limits:ty,$revision:ident,$new_service:expr) => {
        impl $panel {
            /// Bind an explicit absolute host file and exact observed revision.
            /// Existing unsafe state prevents rebinding; construction performs no file I/O.
            pub fn mount_storage(
                &mut self,
                path: impl Into<std::path::PathBuf>,
                limits: $limits,
                expected: $module::$revision,
                timeout: Duration,
            ) -> Result<(), String> {
                self.file_ready()?;
                if self.file.as_ref().is_some_and(PanelFile::fenced) {
                    return Err("Resolve existing file state before rebinding".into());
                }
                let path = path.into();
                let service = ($new_service)(path.clone(), limits).map_err(|e| e.to_string())?;
                self.file = Some(PanelFile::new(service, limits, expected, timeout, path));
                Ok(())
            }
            /// Current real storage status; absent means the host did not bind a file.
            pub fn storage_status(&self) -> Option<&str> {
                self.file.as_ref().map(|f| f.status.as_str())
            }
            /// Queue one full authoring snapshot; no model or history mutation.
            pub fn save_file(&mut self) -> Result<(), String> {
                self.file_ready()?;
                let (bytes, token) = self.capture_file()?;
                self.file
                    .as_mut()
                    .ok_or("No host file binding")?
                    .save(bytes, token)
            }
            /// Queue an isolated load. Native adoption remains disabled while dirty.
            pub fn load_file(&mut self) -> Result<(), String> {
                self.file_ready()?;
                let token = self.current_file_token();
                self.file
                    .as_mut()
                    .ok_or("No host file binding")?
                    .load(token)
            }
            /// Retain the request even when cancellation is too late.
            pub fn cancel_file(&mut self) -> DesktopPanelCancelOutcome {
                self.file
                    .as_mut()
                    .map_or(DesktopPanelCancelOutcome::Idle, PanelFile::cancel)
            }
            /// Discard an isolated candidate without modifying live authoring.
            pub fn discard_loaded(&mut self) {
                if let Some(file) = &mut self.file {
                    file.discard();
                }
            }
            pub(super) fn file_event(
                &mut self,
                target: &str,
                event: &UiEvent,
            ) -> Result<(), String> {
                if !matches!(event, UiEvent::Activate) {
                    return Err("File actions require activation".into());
                }
                match target.strip_prefix(concat!($prefix, ":file:")) {
                    Some("save") => self.save_file(),
                    Some("load") => self.load_file(),
                    Some("cancel") => {
                        self.cancel_file();
                        Ok(())
                    }
                    Some("discard") => {
                        self.discard_loaded();
                        Ok(())
                    }
                    Some("adopt") => {
                        if self.dirty() {
                            return Err(
                                "Unsaved authoring: host must explicitly approve replacement"
                                    .into(),
                            );
                        }
                        let token = self.current_file_token();
                        self.adopt_loaded(&token.model)
                    }
                    _ => Err("Unknown panel file action".into()),
                }
            }
            pub(super) fn poll_file(&mut self) -> Result<bool, String> {
                let Some(file) = &mut self.file else {
                    return Ok(false);
                };
                let (changed, token) = file.poll()?;
                if let Some(token) = token {
                    self.acknowledge_file_saved(&token);
                }
                Ok(changed)
            }
        }
    };
}
panel_file_api!(
    DesktopGraphPanel,
    "graph",
    prism_editor_graph,
    prism_editor_graph::GraphStorageLimits,
    GraphFileRevision,
    |path: std::path::PathBuf, limits| prism_editor_graph::GraphStorageService::new(path, limits)
);
panel_file_api!(
    DesktopTimelinePanel,
    "timeline",
    prism_editor_timeline,
    prism_editor_timeline::TimelineStorageLimits,
    TimelineFileRevision,
    |path: std::path::PathBuf, limits| prism_editor_timeline::TimelineStorageService::new(
        path, limits
    )
);
panel_file_api!(
    DesktopSequencePanel,
    "sequence",
    prism_editor_timeline,
    DesktopSequenceStorageLimits,
    TimelineFileRevision,
    |path: std::path::PathBuf, limits: DesktopSequenceStorageLimits| {
        prism_editor_timeline::SequenceProjectStorageService::new(
            path,
            limits.max_requests,
            limits.document,
        )
    }
);

impl DesktopGraphPanel {
    pub(super) fn file_ready(&self) -> Result<(), String> {
        if self.controller.pointer_active() {
            Err("Release or cancel graph capture first".into())
        } else {
            Ok(())
        }
    }
    fn current_file_token(&self) -> FileSnapshot<prism_editor_graph::GraphSnapshotToken> {
        FileSnapshot {
            model: self.controller.editor().snapshot_token(),
            rate: None,
        }
    }
    fn acknowledge_file_saved(
        &mut self,
        token: &FileSnapshot<prism_editor_graph::GraphSnapshotToken>,
    ) {
        self.acknowledge_saved(&token.model);
    }
    fn capture_file(
        &self,
    ) -> Result<
        (
            Vec<u8>,
            FileSnapshot<prism_editor_graph::GraphSnapshotToken>,
        ),
        String,
    > {
        let limits = self.file.as_ref().ok_or("No host file binding")?.limits;
        let snapshot = self
            .controller
            .editor()
            .snapshot(limits.document, limits.max_bytes)
            .map_err(|e| e.to_string())?;
        Ok((
            snapshot.bytes,
            FileSnapshot {
                model: snapshot.token,
                rate: None,
            },
        ))
    }
    /// Explicit host approval to replace the captured live state, even if dirty.
    /// Both the request's starting token and current token must match this capability.
    pub fn adopt_loaded(
        &mut self,
        expected: &prism_editor_graph::GraphSnapshotToken,
    ) -> Result<(), String> {
        self.file_ready()?;
        let file = self.file.as_ref().ok_or("No host file binding")?;
        let (document, _, started) = file.candidate.as_ref().ok_or("No loaded candidate")?;
        if *expected != self.current_file_token().model || *started != self.current_file_token() {
            return Err("Loaded candidate belongs to an older authoring snapshot".into());
        }
        // Validate and construct independently so failure retains both models.
        let limits = prism_editor_graph::GraphDocumentLimits {
            graph: self.controller.editor().limits(),
            groups: file.limits.document.groups,
        };
        let bytes = document
            .encode(limits, file.limits.max_bytes)
            .map_err(|e| e.to_string())?;
        let document =
            prism_editor_graph::GraphDocument::decode(&bytes, limits, file.limits.max_bytes)
                .map_err(|e| e.to_string())?;
        let editor = prism_editor_graph::GraphEditor::from_document(document, limits.graph)
            .map_err(|e| e.to_string())?;
        let controller = prism_editor_ui::GraphPanelController::new(
            editor,
            self.templates.clone(),
            self.controller.viewport(),
            self.limits,
        )
        .map_err(|e| e.to_string())?;
        self.controller = controller;
        self.view = None;
        self.saved = self.current_file_token().model;
        self.file.as_mut().expect("bound file").adopted();
        Ok(())
    }
}

impl DesktopTimelinePanel {
    pub(super) fn file_ready(&self) -> Result<(), String> {
        if self.failed || self.pending_frame().is_some() || self.controller.drag_preview().is_some()
        {
            Err("Resolve timeline capture/delivery before file operations".into())
        } else {
            Ok(())
        }
    }
    fn current_file_token(&self) -> FileSnapshot<prism_editor_timeline::TimelineSnapshotToken> {
        FileSnapshot {
            model: self.editor.snapshot_token(),
            rate: Some(self.controller.playback().config().ticks_per_second),
        }
    }
    fn acknowledge_file_saved(
        &mut self,
        token: &FileSnapshot<prism_editor_timeline::TimelineSnapshotToken>,
    ) {
        self.acknowledge_saved(&token.model, token.rate.expect("timeline file rate"));
    }
    fn capture_file(
        &self,
    ) -> Result<
        (
            Vec<u8>,
            FileSnapshot<prism_editor_timeline::TimelineSnapshotToken>,
        ),
        String,
    > {
        let limits = self.file.as_ref().ok_or("No host file binding")?.limits;
        let document = prism_editor_timeline::TimelineDocument {
            ticks_per_second: self.controller.playback().config().ticks_per_second,
            timeline: self.editor.timeline().clone(),
        };
        let bytes = document
            .encode(limits.document, limits.max_bytes)
            .map_err(|e| e.to_string())?;
        Ok((bytes, self.current_file_token()))
    }
    /// Explicit replacement approval, fenced to both load-start and current model.
    pub fn adopt_loaded(
        &mut self,
        expected: &prism_editor_timeline::TimelineSnapshotToken,
    ) -> Result<(), String> {
        self.file_ready()?;
        let file = self.file.as_ref().ok_or("No host file binding")?;
        let (document, _, started) = file.candidate.as_ref().ok_or("No loaded candidate")?;
        if *expected != self.current_file_token().model || *started != self.current_file_token() {
            return Err("Loaded candidate belongs to an older authoring snapshot".into());
        }
        let mut editor = prism_editor_timeline::TimelineEditor::new(
            document.timeline.clone(),
            self.editor.limits(),
        )
        .map_err(|e| e.to_string())?;
        let mut controller = prism_editor_ui::TimelinePanelController::default();
        let mut config = self.controller.playback().config();
        config.ticks_per_second = document.ticks_per_second;
        controller
            .apply(
                &mut editor,
                prism_editor_ui::TimelinePanelAction::PlaybackConfig(config),
            )
            .map_err(|e| e.to_string())?;
        self.editor = editor;
        self.controller = controller;
        self.view = None;
        self.saved = self.current_file_token().model;
        self.saved_rate = self.controller.playback().config().ticks_per_second;
        self.anchor = None;
        self.deliver = true;
        self.wake = Some(Instant::now());
        self.file.as_mut().expect("bound file").adopted();
        Ok(())
    }
}

impl DesktopSequencePanel {
    pub(super) fn file_ready(&self) -> Result<(), String> {
        if self.has_clip_draft() || self.controller.error().is_some() {
            Err(
                "Apply/discard the clip draft and resolve sequence errors before file operations"
                    .into(),
            )
        } else {
            Ok(())
        }
    }
    fn current_file_token(&self) -> FileSnapshot<prism_editor_timeline::SequenceProjectToken> {
        FileSnapshot {
            model: self.editor.snapshot_token(),
            rate: None,
        }
    }
    fn acknowledge_file_saved(
        &mut self,
        token: &FileSnapshot<prism_editor_timeline::SequenceProjectToken>,
    ) {
        self.acknowledge_saved(&token.model);
    }
    fn capture_file(
        &self,
    ) -> Result<
        (
            Vec<u8>,
            FileSnapshot<prism_editor_timeline::SequenceProjectToken>,
        ),
        String,
    > {
        let limits = self.file.as_ref().ok_or("No host file binding")?.limits;
        let bytes = self
            .editor
            .document()
            .encode(limits.document)
            .map_err(|e| e.to_string())?;
        Ok((bytes, self.current_file_token()))
    }
    /// Explicit replacement approval for both the load-start and current project.
    /// Drafts/errors reject adoption; constructing the replacement can fail safely.
    pub fn adopt_loaded(
        &mut self,
        expected: &prism_editor_timeline::SequenceProjectToken,
    ) -> Result<(), String> {
        self.file_ready()?;
        let file = self.file.as_ref().ok_or("No host file binding")?;
        let (document, _, started) = file.candidate.as_ref().ok_or("No loaded candidate")?;
        if *expected != self.editor.snapshot_token() || *started != self.current_file_token() {
            return Err("Loaded candidate belongs to an older sequence project".into());
        }
        let editor = prism_editor_timeline::SequenceProjectEditor::new(
            document.clone(),
            self.editor.limits(),
        )
        .map_err(|e| e.to_string())?;
        let controller = prism_editor_ui::SequencePanelController::new(self.limits)
            .map_err(|e| e.to_string())?;
        self.editor = editor;
        self.controller = controller;
        self.view = None;
        self.saved = self.editor.snapshot_token();
        self.file.as_mut().expect("bound file").adopted();
        Ok(())
    }
}

impl EditorDesktopUi {
    /// Route native Save/Open to the active panel's explicit host file binding.
    /// Returns false only when no native panel is active (scene route may proceed).
    pub fn panel_document_command(&mut self, save: bool) -> Result<bool, DesktopUiError> {
        if !self.panels.active() {
            return Ok(false);
        }
        if self.focus.is_some() || self.panel_pointer_active() {
            return self.reject(DesktopUiError::Panel(
                "Resolve the draft/capture before panel I/O".into(),
            ));
        }
        let slot = self
            .panels
            .slot_mut()
            .ok_or(DesktopUiError::InvalidProjection)?;
        let host = slot.host.as_mut() as &mut dyn std::any::Any;
        let result = if let Some(panel) = host.downcast_mut::<DesktopGraphPanel>() {
            if save {
                panel.save_file()
            } else {
                panel.load_file()
            }
        } else if let Some(panel) = host.downcast_mut::<DesktopTimelinePanel>() {
            if save {
                panel.save_file()
            } else {
                panel.load_file()
            }
        } else if let Some(panel) = host.downcast_mut::<DesktopSequencePanel>() {
            if save {
                panel.save_file()
            } else {
                panel.load_file()
            }
        } else {
            Err("This panel requires its host's persistence provider".into())
        };
        self.invalidated = true;
        if let Err(e) = result {
            return self.reject(DesktopUiError::Panel(e));
        }
        self.refresh()?;
        Ok(true)
    }
}
