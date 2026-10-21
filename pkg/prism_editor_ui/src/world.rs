//! Bounded world-panel commands over the real engine-neutral authoring model.
use crate::{world_view::BoundWorldPanel, UiEvent};
use prism_asset::StableGuid;
use prism_ecs::partition::{cell::CellCoord, data_layer::DataLayerId};
use prism_editor_timeline::{
    TimelineCancelOutcome, TimelineFileRevision, TimelineSaveDurability, TimelineStorageError,
};
use prism_editor_world::project::{
    ProjectChange, ProjectEdit, ProjectError, ProjectLimits, WorldProject,
    WorldProjectLoadCandidate, WorldProjectStorageRequest, WorldProjectStorageResult,
    WorldProjectStorageService, WorldProjectToken,
};
use prism_editor_world::{
    CellKey, CellLockToken, CellResidency, EditorWorldData, EditorWorldDataError,
    FileWorldTransactionProvider, IoCompletion, PendingCellSave, PendingWorldTransaction,
    ResidencyDemand, TransitionKind, TransitionToken, WorldTransactionEdit,
    WorldTransactionReceipt, WorldTransactionSnapshot,
};
use std::{
    collections::{BTreeMap, VecDeque},
    fmt,
    sync::Arc,
    task::Poll,
    time::Duration,
};

/// Hard ceiling for viewport slots, including absent cells.
pub const MAX_WORLD_PANEL_CELLS: usize = 1024;
/// Hard ceiling for selected-cell layer controls.
pub const MAX_WORLD_PANEL_LAYERS: usize = 64;
/// Hard ceiling for queued host operations.
pub const MAX_WORLD_PANEL_INTENTS: usize = 32;
/// Hard ceiling for retained controller leases.
pub const MAX_WORLD_PANEL_LEASES: usize = 128;

/// Screen horizontal/vertical axes and the remaining fixed slice axis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldGridPlane {
    /// X horizontal, Y vertical, fixed Z.
    Xy,
    /// X horizontal, Z vertical, fixed Y.
    Xz,
    /// Y horizontal, Z vertical, fixed X.
    Yz,
}
impl WorldGridPlane {
    pub(crate) fn coord(self, u: i32, v: i32, slice: i32) -> CellCoord {
        match self {
            Self::Xy => CellCoord::new(u, v, slice),
            Self::Xz => CellCoord::new(u, slice, v),
            Self::Yz => CellCoord::new(slice, u, v),
        }
    }
}
/// Logical-pixel viewport. All edges use half-open containment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldPanelViewport {
    /// Pixel dimensions in 1..=8192.
    pub size: [f64; 2],
    /// Continuous cell coordinates at the top left.
    pub origin: [f64; 2],
    /// Pixels per cell in 8..=256.
    pub cell_pixels: f64,
    /// Screen-to-world plane.
    pub plane: WorldGridPlane,
    /// Coordinate on the remaining axis.
    pub slice: i32,
}
impl Default for WorldPanelViewport {
    fn default() -> Self {
        Self {
            size: [640., 384.],
            origin: [0., 0.],
            cell_pixels: 64.,
            plane: WorldGridPlane::Xy,
            slice: 0,
        }
    }
}
impl WorldPanelViewport {
    pub(crate) fn bounds(self, budget: usize) -> Result<[i64; 4], WorldPanelError> {
        if self
            .size
            .iter()
            .any(|v| !v.is_finite() || !(1.0..=8192.).contains(v))
            || self.origin.iter().any(|v| !v.is_finite())
            || !self.cell_pixels.is_finite()
            || !(8.0..=256.).contains(&self.cell_pixels)
        {
            return Err(WorldPanelError::InvalidViewport);
        }
        let mut b = [0; 4];
        for axis in 0..2 {
            let lo = self.origin[axis].floor();
            let hi = (self.origin[axis] + self.size[axis] / self.cell_pixels).ceil();
            if lo < f64::from(i32::MIN) || hi > f64::from(i32::MAX) + 1. {
                return Err(WorldPanelError::InvalidViewport);
            }
            b[axis] = lo as i64;
            b[axis + 2] = hi as i64;
        }
        let count = (b[2] - b[0])
            .checked_mul(b[3] - b[1])
            .ok_or(WorldPanelError::Capacity)?;
        if count > budget as i64 {
            return Err(WorldPanelError::Capacity);
        }
        Ok(b)
    }
    /// Rejects nonfinite/outside points and deterministically maps grid slots.
    /// Registration is checked separately by the controller.
    pub fn hit_test(self, p: [f64; 2]) -> Option<CellCoord> {
        self.bounds(MAX_WORLD_PANEL_CELLS).ok()?;
        if p.iter()
            .enumerate()
            .any(|(i, v)| !v.is_finite() || *v < 0. || *v >= self.size[i])
        {
            return None;
        }
        Some(self.plane.coord(
            (self.origin[0] + p[0] / self.cell_pixels).floor() as i32,
            (self.origin[1] + p[1] / self.cell_pixels).floor() as i32,
            self.slice,
        ))
    }
}
/// Bounds validated before controller construction or mutation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldPanelLimits {
    /// Visible slots including empty slots.
    pub visible_cells: usize,
    /// Selected-cell layer prefix; hidden controls are explicitly counted.
    pub visible_layers: usize,
    /// Pending host-operation queue capacity.
    pub intents: usize,
    /// Retained lease capacity.
    pub leases: usize,
    /// Lease lifetime in host monotonic ticks.
    pub lease_ttl: u64,
}
impl Default for WorldPanelLimits {
    fn default() -> Self {
        Self {
            visible_cells: 512,
            visible_layers: 16,
            intents: 16,
            leases: 32,
            lease_ttl: 60,
        }
    }
}
/// Lease information actually established by this controller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldPanelLease {
    /// External ownership is unknown, not asserted to be unlocked.
    Unobserved,
    /// Live local lease.
    Owned {
        /// Exclusive expiry tick.
        expires_at: u64,
    },
    /// Retained local token has expired.
    Expired,
}
/// Commands emitted by keyed Loom controls or native viewport input adapters.
#[derive(Clone, Debug, PartialEq)]
pub enum WorldPanelAction {
    /// Undo the last sparse project authoring command.
    ProjectUndo,
    /// Redo the last undone project command.
    ProjectRedo,
    /// Publish actual membership bytes through the bound storage service.
    ProjectSave,
    /// Read a candidate without changing the live world.
    ProjectLoad,
    /// Request cancellation; the handle remains retained until terminal.
    ProjectCancel,
    /// Adopt the fenced candidate, refusing unsaved membership changes.
    ProjectAdopt,
    /// Explicitly discard unsaved membership changes and adopt the candidate.
    ProjectAdoptDiscard,
    /// Discard the pending candidate without changing the live model.
    ProjectDismiss,
    /// First layer index of the bounded project membership page.
    ProjectLayerPage(usize),
    /// Set a direct layer membership on the selected real cell.
    CellLayer {
        /// Existing project layer.
        id: DataLayerId,
        /// Whether the selected cell belongs directly to this layer.
        member: bool,
    },
    /// Replace the viewport after validation.
    Viewport(WorldPanelViewport),
    /// Zoom around the viewport center.
    Zoom(f64),
    /// Pan by screen-axis cell units.
    Pan([f64; 2]),
    /// Select a registered cell.
    Select(CellCoord),
    /// Acquire/renew the selected cell lease.
    AcquireLease,
    /// Release the selected cell lease.
    ReleaseLease,
    /// Queue selected-cell load.
    Load,
    /// Queue selected-cell clean unload.
    Unload,
    /// Queue selected-cell save.
    Save,
    /// Capture selected cell as cross-cell source.
    CrossCellSource,
    /// Queue cross-cell operation to selected destination.
    CrossCell,
    /// Change actual layer visibility.
    LayerVisible {
        /// World-defined layer.
        id: DataLayerId,
        /// New policy.
        visible: bool,
    },
    /// Change actual layer editability.
    LayerEditable {
        /// World-defined layer.
        id: DataLayerId,
        /// New policy.
        editable: bool,
    },
}
/// Requested host operation; no variant implies durable success.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldPanelIntentKind {
    /// Load this cell while preserving other residents.
    Load(CellCoord),
    /// Clean unload protected by the real world lock registry.
    Unload(CellCoord),
    /// Save actual opaque bytes with revision/lease guards.
    Save(CellCoord),
    /// Host must provide both owners' updated opaque membership payloads.
    CrossCell {
        /// Source captured by the UI.
        source: CellCoord,
        /// Selected destination.
        destination: CellCoord,
    },
}
/// Immutable controller/epoch-fenced operation ticket.
#[derive(Clone, Debug)]
pub struct WorldPanelIntent {
    kind: WorldPanelIntentKind,
    epoch: u64,
    owner: Arc<()>,
}
impl WorldPanelIntent {
    /// Requested operation.
    pub const fn kind(&self) -> WorldPanelIntentKind {
        self.kind
    }
    /// Captured model epoch; stale tickets fail before execution.
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }
}
/// Panel validation or real world operation failure.
#[derive(Debug)]
pub enum WorldPanelError {
    /// Bounded project admission, conflict or sparse-history error.
    Project(Box<ProjectError>),
    /// Actual shared-worker submission or completion error.
    ProjectStorage(String),
    /// Invalid limits, identity or finite geometry.
    InvalidViewport,
    /// Materialization, queue or lease budget exhausted.
    Capacity,
    /// View belongs to an older state or another controller.
    StaleView,
    /// Ticket belongs to an older epoch or another controller.
    StaleIntent,
    /// No matching active semantic binding.
    UnboundEvent,
    /// Selection, lease or residency does not permit the action.
    InvalidOperation,
    /// Host clock moved backwards or expiry overflowed.
    InvalidClock,
    /// Actual domain validation failure.
    World(EditorWorldDataError),
}
impl fmt::Display for WorldPanelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "world panel: {self:?}")
    }
}
impl std::error::Error for WorldPanelError {}
impl From<EditorWorldDataError> for WorldPanelError {
    fn from(e: EditorWorldDataError) -> Self {
        Self::World(e)
    }
}
impl From<ProjectError> for WorldPanelError {
    fn from(e: ProjectError) -> Self {
        Self::Project(Box::new(e))
    }
}

/// Project admission and sparse-history retention, independent of grid limits.
#[derive(Clone, Copy, Debug)]
pub struct WorldProjectPanelLimits {
    /// Actual codec limits, also checked before authoring publication.
    pub project: ProjectLimits,
    /// Maximum combined undo and redo entries; exhaustion rejects a command.
    pub history_entries: usize,
    /// Combined charged sparse inverse bytes. No full snapshots are retained.
    pub history_bytes: usize,
}
impl Default for WorldProjectPanelLimits {
    fn default() -> Self {
        Self {
            project: ProjectLimits::default(),
            history_entries: 128,
            history_bytes: 4 * 1024 * 1024,
        }
    }
}
/// Observable storage lifecycle. Publication uncertainty never implies clean.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorldProjectIoState {
    /// No project file service installed.
    Unbound,
    /// Bound service with no active request.
    Idle,
    /// Actual shared-worker read in flight.
    Loading,
    /// Actual shared-worker publication in flight.
    Saving,
    /// Cancellation requested; poll until the real terminal outcome.
    Cancelling,
    /// A validated candidate awaits an explicit decision.
    Candidate,
    /// Actual published outcome, even if the live project has since changed.
    Saved {
        /// True only if the receipt matched the exact current snapshot.
        current: bool,
        /// Real directory-sync result.
        durability: TimelineSaveDurability,
        /// Worker deadline passed after publication became irrevocable.
        deadline_exceeded: bool,
    },
    /// Worker confirmed cancellation before publication.
    Cancelled,
    /// Actual error text; a publication-uncertain error also blocks saves.
    Failed(String),
}
struct ProjectIo {
    service: WorldProjectStorageService,
    expected: Option<TimelineFileRevision>,
    request: Option<WorldProjectStorageRequest>,
    started: Option<WorldProjectToken>,
    candidate: Option<(WorldProjectLoadCandidate, WorldProjectToken)>,
    state: WorldProjectIoState,
    timeout: Duration,
}
/// Exact controller-generation fence around a real residency transition.
#[derive(Clone, Debug)]
pub struct WorldPanelTransition {
    token: TransitionToken,
    owner: Arc<()>,
}
impl WorldPanelTransition {
    /// Real domain token for constructing the provider load request.
    pub const fn token(&self) -> TransitionToken {
        self.token
    }
}
/// Real save snapshot with a controller-generation receipt fence.
#[derive(Clone, Debug)]
pub struct WorldPanelSave {
    pending: PendingCellSave,
    owner: Arc<()>,
}
impl WorldPanelSave {
    /// Actual provider request.
    pub const fn request(&self) -> &prism_editor_world::CellSaveRequest {
        self.pending.request()
    }
    /// Actual opaque snapshot bytes.
    pub fn payload(&self) -> &[u8] {
        self.pending.payload()
    }
}
/// Actual OFPA transaction with a controller-generation receipt fence.
#[derive(Clone, Debug)]
pub struct WorldPanelTransaction {
    pending: PendingWorldTransaction,
    owner: Arc<()>,
}
impl WorldPanelTransaction {
    /// Actual provider transaction request.
    pub fn request(&self) -> &prism_editor_world::WorldTransactionRequest {
        self.pending.request()
    }
}
struct Lease {
    token: CellLockToken,
    expiry: u64,
}
/// Owns the real model and leases. Async I/O is explicitly handed to the host.
pub struct WorldPanelController {
    world: EditorWorldData,
    project: WorldProject,
    project_limits: WorldProjectPanelLimits,
    undo: Vec<ProjectChange>,
    redo: Vec<ProjectChange>,
    history_bytes: usize,
    clean_project: Option<WorldProjectToken>,
    project_io: Option<ProjectIo>,
    retired_world: Option<EditorWorldData>,
    pub(crate) project_layer_offset: usize,
    owner: StableGuid,
    pub(crate) viewport: WorldPanelViewport,
    pub(crate) limits: WorldPanelLimits,
    pub(crate) selected: Option<CellCoord>,
    pub(crate) source: Option<CellCoord>,
    leases: BTreeMap<CellKey, Lease>,
    intents: VecDeque<WorldPanelIntent>,
    now: u64,
    identity: Arc<()>,
    pub(crate) stamp: Arc<()>,
}
impl WorldPanelController {
    /// Takes ownership and captures the bounded real membership catalog once.
    pub fn new(
        world: EditorWorldData,
        owner: StableGuid,
        viewport: WorldPanelViewport,
        limits: WorldPanelLimits,
    ) -> Result<Self, WorldPanelError> {
        Self::with_project_limits(
            world,
            owner,
            viewport,
            limits,
            WorldProjectPanelLimits::default(),
        )
    }
    /// Captures membership with explicit admission and history budgets.
    pub fn with_project_limits(
        world: EditorWorldData,
        owner: StableGuid,
        viewport: WorldPanelViewport,
        limits: WorldPanelLimits,
        project_limits: WorldProjectPanelLimits,
    ) -> Result<Self, WorldPanelError> {
        if owner == StableGuid::NIL
            || limits.visible_cells == 0
            || limits.visible_cells > MAX_WORLD_PANEL_CELLS
            || limits.visible_layers == 0
            || limits.visible_layers > MAX_WORLD_PANEL_LAYERS
            || limits.intents == 0
            || limits.intents > MAX_WORLD_PANEL_INTENTS
            || limits.leases == 0
            || limits.leases > MAX_WORLD_PANEL_LEASES
            || limits.lease_ttl == 0
        {
            return Err(WorldPanelError::InvalidViewport);
        }
        viewport.bounds(limits.visible_cells)?;
        project_limits.project.validate()?;
        if project_limits.history_entries == 0
            || project_limits.history_entries > 4096
            || project_limits.history_bytes == 0
            || project_limits.history_bytes > 64 * 1024 * 1024
        {
            return Err(WorldPanelError::Capacity);
        }
        let project = WorldProject::capture(&world, project_limits.project)?;
        Ok(Self {
            world,
            project,
            project_limits,
            undo: Vec::new(),
            redo: Vec::new(),
            history_bytes: 0,
            clean_project: None,
            project_io: None,
            retired_world: None,
            project_layer_offset: 0,
            owner,
            viewport,
            limits,
            selected: None,
            source: None,
            leases: BTreeMap::new(),
            intents: VecDeque::new(),
            now: 0,
            identity: Arc::new(()),
            stamp: Arc::new(()),
        })
    }
    /// Actual immutable authoring model.
    pub const fn world(&self) -> &EditorWorldData {
        &self.world
    }
    /// Actual synchronized identity/layer/cell/actor membership catalog.
    pub const fn project(&self) -> &WorldProject {
        &self.project
    }
    /// Whether this exact project snapshot lacks a matching real save/load proof.
    pub fn project_is_dirty(&self) -> bool {
        !self
            .clean_project
            .as_ref()
            .is_some_and(|t| self.project.matches_token(t))
    }
    /// Combined sparse inverse bytes and undo/redo entry counts.
    pub fn project_history(&self) -> (usize, usize, usize) {
        (self.undo.len(), self.redo.len(), self.history_bytes)
    }
    /// Real project worker lifecycle; this accessor starts no IO.
    pub fn project_io_state(&self) -> WorldProjectIoState {
        self.project_io
            .as_ref()
            .map_or(WorldProjectIoState::Unbound, |s| s.state.clone())
    }
    /// Whether a real request is retained, including cancellation too late.
    pub fn project_io_pending(&self) -> bool {
        self.project_io
            .as_ref()
            .is_some_and(|s| s.request.is_some())
    }
    /// Validated independent load candidate, never implicitly adopted.
    pub fn project_candidate(&self) -> Option<&WorldProject> {
        self.project_io
            .as_ref()?
            .candidate
            .as_ref()
            .map(|(c, _)| &c.document)
    }
    /// Whether a conditional save has a known file revision.
    pub fn project_can_save(&self) -> bool {
        self.retired_world.is_none()
            && self.project_io.as_ref().is_some_and(|s| {
                s.request.is_none() && s.candidate.is_none() && s.expected.is_some()
            })
    }
    /// True until the host explicitly takes ownership of the replaced world.
    /// Native runtime consumers must be detached before acknowledging retirement.
    pub fn has_retired_world(&self) -> bool {
        self.retired_world.is_some()
    }
    /// Hands back the old world with all locks, edit scope and HLOD state intact.
    /// Taking it acknowledges that the host has detached old runtime consumers.
    pub fn take_retired_world(&mut self) -> Option<EditorWorldData> {
        let old = self.retired_world.take();
        if old.is_some() {
            self.invalidate();
        }
        old
    }
    /// Install the existing typed storage service with an explicitly observed
    /// file revision. Use None for an unknown file; load/adopt before saving.
    /// Binding alone never marks the captured live membership clean.
    pub fn bind_project_storage(
        &mut self,
        service: WorldProjectStorageService,
        expected: Option<TimelineFileRevision>,
        timeout: Duration,
    ) -> Result<(), WorldPanelError> {
        if timeout.is_zero()
            || self
                .project_io
                .as_ref()
                .is_some_and(|s| s.request.is_some() || s.candidate.is_some())
        {
            return Err(WorldPanelError::InvalidOperation);
        }
        self.project_io = Some(ProjectIo {
            service,
            expected,
            request: None,
            started: None,
            candidate: None,
            state: WorldProjectIoState::Idle,
            timeout,
        });
        self.invalidate();
        Ok(())
    }
    /// Poll the actual worker once. Cancellation retains the request until this
    /// method observes its terminal result. Call on host ticks, even when hidden.
    pub fn poll_project_storage(&mut self) -> Result<bool, WorldPanelError> {
        let Some(s) = self.project_io.as_mut() else {
            return Ok(false);
        };
        let Some(request) = s.request.as_mut() else {
            return Ok(false);
        };
        let Poll::Ready(result) = request.poll() else {
            return Ok(false);
        };
        s.request = None;
        let started = s.started.take().expect("admitted project request fence");
        match result {
            Ok(WorldProjectStorageResult::Loaded(candidate)) => {
                // The decoder's limits are service-owned; recheck this panel's
                // admission policy before retaining a candidate for adoption.
                if candidate.document.identity() != self.project.identity() {
                    s.state = WorldProjectIoState::Failed("world identity mismatch".into());
                } else if let Err(e) = candidate.document.validate(self.project_limits.project) {
                    s.state = WorldProjectIoState::Failed(e.to_string());
                } else {
                    s.candidate = Some((candidate, started));
                    s.state = WorldProjectIoState::Candidate;
                }
            }
            Ok(WorldProjectStorageResult::Saved(receipt)) => {
                let current = self.project.matches_token(&receipt.snapshot_token);
                if current {
                    self.clean_project = Some(receipt.snapshot_token);
                }
                s.expected = Some(receipt.revision);
                s.state = WorldProjectIoState::Saved {
                    current,
                    durability: receipt.durability,
                    deadline_exceeded: receipt.deadline_exceeded,
                };
            }
            Err(TimelineStorageError::Cancelled) => s.state = WorldProjectIoState::Cancelled,
            Err(e) => {
                if matches!(e, TimelineStorageError::PublicationUncertain { .. }) {
                    s.expected = None;
                    self.clean_project = None;
                }
                s.state = WorldProjectIoState::Failed(e.to_string());
            }
        }
        self.invalidate();
        Ok(true)
    }
    /// Request real cancellation. TooLate requires polling the publication
    /// receipt; the controller never invents a cancelled/clean outcome.
    pub fn cancel_project_storage(&mut self) -> Result<TimelineCancelOutcome, WorldPanelError> {
        let s = self
            .project_io
            .as_mut()
            .ok_or(WorldPanelError::InvalidOperation)?;
        let r = s
            .request
            .as_ref()
            .ok_or(WorldPanelError::InvalidOperation)?;
        let outcome = r.cancel();
        s.state = WorldProjectIoState::Cancelling;
        self.invalidate();
        Ok(outcome)
    }
    fn start_project_storage(&mut self, save: bool) -> Result<(), WorldPanelError> {
        if self.retired_world.is_some() {
            return Err(WorldPanelError::InvalidOperation);
        }
        let s = self
            .project_io
            .as_mut()
            .ok_or(WorldPanelError::InvalidOperation)?;
        if s.request.is_some() || s.candidate.is_some() {
            return Err(WorldPanelError::InvalidOperation);
        }
        let r = if save {
            s.service
                .save(
                    &self.project,
                    s.expected
                        .clone()
                        .ok_or(WorldPanelError::InvalidOperation)?,
                    s.timeout,
                )
                .map_err(|e| WorldPanelError::ProjectStorage(e.to_string()))?
        } else {
            s.service
                .load(s.timeout)
                .map_err(|e| WorldPanelError::ProjectStorage(e.to_string()))?
        };
        s.request = Some(r);
        s.started = Some(self.project.token());
        s.state = if save {
            WorldProjectIoState::Saving
        } else {
            WorldProjectIoState::Loading
        };
        Ok(())
    }
    fn replacement_allowed(&self) -> bool {
        self.retired_world.is_none()
            && self.intents.is_empty()
            && !self.leases.values().any(|l| l.expiry > self.now)
            && self
                .world
                .model()
                .cells()
                .all(|r| r.residency() == CellResidency::Unloaded && !r.is_dirty())
            && self.world.actor_records().all(|a| !a.is_dirty())
    }
    fn guard_external_leases(&mut self) -> Result<(), WorldPanelError> {
        // The domain intentionally exposes no lock-registry mutation/accessor.
        // Probe through its real capability API, including preexisting locks
        // owned by this same user. Release every acquired probe immediately.
        // Only the old world's monotonic lock sequence/expired leases change.
        let cells: Vec<_> = self.world.model().cells().map(|c| c.coord()).collect();
        for cell in cells {
            let probe = self.world.acquire_lock(cell, self.owner, self.now, 1)?;
            self.world.release_lock(&probe)?;
        }
        Ok(())
    }
    /// Cheap bounded-view preflight; full model checks happen only on adoption.
    pub fn project_candidate_is_current(&self) -> bool {
        self.project_io
            .as_ref()
            .and_then(|s| s.candidate.as_ref())
            .is_some_and(|(_, t)| self.project.matches_token(t))
    }
    fn adopt_project_candidate(&mut self, discard_membership: bool) -> Result<(), WorldPanelError> {
        if !self.project_candidate_is_current()
            || !self.replacement_allowed()
            || (!discard_membership && self.project_is_dirty())
        {
            return Err(WorldPanelError::InvalidOperation);
        }
        let (candidate, _) = self
            .project_io
            .as_ref()
            .and_then(|s| s.candidate.as_ref())
            .ok_or(WorldPanelError::InvalidOperation)?;
        let new_world = candidate
            .document
            .materialize(self.world.limits(), self.project_limits.project)?;
        self.guard_external_leases()?;
        let s = self.project_io.as_mut().expect("candidate service");
        let (candidate, _) = s.candidate.take().expect("validated candidate");
        self.retired_world = Some(std::mem::replace(&mut self.world, new_world));
        self.project = candidate.document;
        self.clean_project = Some(self.project.token());
        s.expected = Some(candidate.revision);
        s.state = WorldProjectIoState::Idle;
        self.undo.clear();
        self.redo.clear();
        self.history_bytes = 0;
        self.leases.clear();
        self.selected = None;
        self.source = None;
        self.project_layer_offset = 0;
        self.identity = Arc::new(());
        Ok(())
    }
    /// Execute a bounded membership command against the current revision.
    /// Layer policy and existing-cell memberships update the real world in place.
    /// Structural authoring is admitted only on a quiescent world and hands the
    /// old world back through take_retired_world. Actor moves use real OFPA IO.
    pub fn edit_project(&mut self, edits: &[ProjectEdit]) -> Result<(), WorldPanelError> {
        if edits.iter().any(|e| matches!(e, ProjectEdit::Actor { .. })) {
            return Err(WorldPanelError::InvalidOperation);
        }
        let kept = self.history_bytes
            - self
                .redo
                .iter()
                .map(ProjectChange::charged_bytes)
                .sum::<usize>();
        if self.undo.len() >= self.project_limits.history_entries {
            return Err(WorldPanelError::Capacity);
        }
        let mut next = self.project.clone();
        let change = next.apply(
            next.revision(),
            edits,
            self.project_limits.project,
            self.project_limits.history_bytes.saturating_sub(kept),
        )?;
        self.install_project_edit(next)?;
        self.history_bytes = kept + change.charged_bytes();
        self.redo.clear();
        self.undo.push(change);
        self.invalidate();
        Ok(())
    }
    fn project_step_history(&mut self, redo: bool) -> Result<(), WorldPanelError> {
        let change = if redo {
            self.redo.last()
        } else {
            self.undo.last()
        }
        .ok_or(ProjectError::EmptyHistory)?;
        let mut next = self.project.clone();
        if redo {
            change.redo(&mut next, self.project_limits.project)?;
        } else {
            change.undo(&mut next, self.project_limits.project)?;
        }
        self.install_project_edit(next)?;
        if redo {
            self.undo.push(self.redo.pop().expect("checked history"));
        } else {
            self.redo.push(self.undo.pop().expect("checked history"));
        }
        Ok(())
    }
    fn install_project_edit(&mut self, next: WorldProject) -> Result<(), WorldPanelError> {
        if self.retired_world.is_some() {
            return Err(WorldPanelError::InvalidOperation);
        }
        // Only policy and direct membership replacements can preserve opaque
        // runtime state. Preflight topology before the first domain mutation.
        let same_topology = self.project.layers().len() == next.layers().len()
            && self.project.layers().all(|l| {
                next.layer(l.id)
                    .is_some_and(|n| n.name == l.name && n.parent == l.parent)
            })
            && self.project.cells().len() == next.cells().len()
            && self
                .project
                .cells()
                .all(|(c, _)| next.cell_layers(c).is_some())
            && self.project.actors().eq(next.actors());
        if same_topology {
            for l in next.layers() {
                let old = self.project.layer(l.id).expect("preflight layer");
                if l.visible != old.visible {
                    self.world.set_layer_visible(l.id, l.visible)?;
                }
                if l.editable != old.editable {
                    self.world.set_layer_editable(l.id, l.editable)?;
                }
            }
            for (cell, layers) in next.cells() {
                if self.project.cell_layers(cell) != Some(layers) {
                    self.world.set_cell_layers(cell, layers.iter().copied())?;
                }
            }
        } else {
            if !self.replacement_allowed() {
                return Err(WorldPanelError::InvalidOperation);
            }
            let new_world = next.materialize(self.world.limits(), self.project_limits.project)?;
            self.guard_external_leases()?;
            self.retired_world = Some(std::mem::replace(&mut self.world, new_world));
            self.identity = Arc::new(());
            self.leases.clear();
            self.source = None;
            if self.selected.is_some_and(|c| next.cell_layers(c).is_none()) {
                self.selected = None;
            }
        }
        self.project = next;
        Ok(())
    }
    /// Current validated viewport.
    pub const fn viewport(&self) -> WorldPanelViewport {
        self.viewport
    }
    /// Selected registered cell, even when panned outside the viewport.
    pub const fn selected(&self) -> Option<CellCoord> {
        self.selected
    }
    /// Queued host tickets, never a durable-success count.
    pub fn pending_intents(&self) -> usize {
        self.intents.len()
    }
    /// Accurate local observation; external locks remain enforced by the world.
    pub fn lease(&self, cell: CellCoord) -> WorldPanelLease {
        match self.leases.get(&CellKey::new(cell)) {
            None => WorldPanelLease::Unobserved,
            Some(l) if l.expiry <= self.now => WorldPanelLease::Expired,
            Some(l) => WorldPanelLease::Owned {
                expires_at: l.expiry,
            },
        }
    }
    /// Advance monotonic host time and invalidate old lease displays.
    pub fn set_time(&mut self, now: u64) -> Result<(), WorldPanelError> {
        if now < self.now {
            return Err(WorldPanelError::InvalidClock);
        }
        if now != self.now {
            self.invalidate();
            self.now = now;
        }
        Ok(())
    }
    fn invalidate(&mut self) {
        self.stamp = Arc::new(());
    }
    fn selected_cell(&self) -> Result<CellCoord, WorldPanelError> {
        self.selected.ok_or(WorldPanelError::InvalidOperation)
    }
    fn lease_token(&self, cell: CellCoord) -> Result<CellLockToken, WorldPanelError> {
        self.leases
            .get(&CellKey::new(cell))
            .filter(|l| l.expiry > self.now)
            .map(|l| l.token.clone())
            .ok_or(WorldPanelError::InvalidOperation)
    }
    fn enqueue(&mut self, kind: WorldPanelIntentKind) -> Result<(), WorldPanelError> {
        if self.intents.len() >= self.limits.intents {
            return Err(WorldPanelError::Capacity);
        }
        self.intents.push_back(WorldPanelIntent {
            kind,
            epoch: self.world.model().epoch(),
            owner: self.identity.clone(),
        });
        Ok(())
    }
    /// Pop oldest host operation. Dropping it changes no model state.
    pub fn take_intent(&mut self) -> Option<WorldPanelIntent> {
        let i = self.intents.pop_front();
        if i.is_some() {
            self.invalidate();
        }
        i
    }
    fn validate_intent(&self, i: &WorldPanelIntent) -> Result<(), WorldPanelError> {
        self.validate_generation(&i.owner)?;
        if !Arc::ptr_eq(&i.owner, &self.identity) || i.epoch != self.world.model().epoch() {
            return Err(WorldPanelError::StaleIntent);
        }
        Ok(())
    }
    fn validate_generation(&self, owner: &Arc<()>) -> Result<(), WorldPanelError> {
        if self.retired_world.is_some() || !Arc::ptr_eq(owner, &self.identity) {
            return Err(WorldPanelError::StaleIntent);
        }
        Ok(())
    }
    /// Build viewport slots plus a bounded selected-layer prefix only.
    pub fn build(&self) -> BoundWorldPanel {
        crate::world_view::build(self)
    }
    /// Resolve a live painted control and apply its actual model command.
    pub fn handle_event(
        &mut self,
        view: &BoundWorldPanel,
        target: &str,
        event: &UiEvent,
    ) -> Result<(), WorldPanelError> {
        if !Arc::ptr_eq(&view.stamp, &self.stamp) {
            return Err(WorldPanelError::StaleView);
        }
        self.apply(view.dispatch(target, event)?)
    }
    /// Apply a command; viewport errors never publish partial geometry.
    pub fn apply(&mut self, action: WorldPanelAction) -> Result<(), WorldPanelError> {
        use WorldPanelAction as A;
        match action {
            A::ProjectUndo => self.project_step_history(false)?,
            A::ProjectRedo => self.project_step_history(true)?,
            A::ProjectSave => self.start_project_storage(true)?,
            A::ProjectLoad => self.start_project_storage(false)?,
            A::ProjectCancel => {
                self.cancel_project_storage()?;
            }
            A::ProjectAdopt => self.adopt_project_candidate(false)?,
            A::ProjectAdoptDiscard => self.adopt_project_candidate(true)?,
            A::ProjectDismiss => {
                let s = self
                    .project_io
                    .as_mut()
                    .ok_or(WorldPanelError::InvalidOperation)?;
                if s.candidate.take().is_none() {
                    return Err(WorldPanelError::InvalidOperation);
                }
                s.state = WorldProjectIoState::Idle;
            }
            A::ProjectLayerPage(offset) => {
                if offset != 0 && offset >= self.project.layers().len() {
                    return Err(WorldPanelError::InvalidOperation);
                }
                self.project_layer_offset = offset;
            }
            A::CellLayer { id, member } => {
                let cell = self.selected_cell()?;
                let mut layers = self
                    .project
                    .cell_layers(cell)
                    .ok_or(WorldPanelError::InvalidOperation)?
                    .clone();
                if member {
                    layers.insert(id);
                } else {
                    layers.remove(&id);
                }
                self.edit_project(&[ProjectEdit::Cell {
                    cell,
                    layers: Some(layers),
                }])?;
            }
            A::Viewport(v) => {
                v.bounds(self.limits.visible_cells)?;
                self.viewport = v;
            }
            A::Zoom(factor) => {
                if !factor.is_finite() || factor <= 0. {
                    return Err(WorldPanelError::InvalidViewport);
                }
                let mut n = self.viewport;
                n.cell_pixels *= factor;
                for i in 0..2 {
                    n.origin[i] += self.viewport.size[i] / self.viewport.cell_pixels / 2.
                        - n.size[i] / n.cell_pixels / 2.;
                }
                n.bounds(self.limits.visible_cells)?;
                self.viewport = n;
            }
            A::Pan(delta) => {
                let mut n = self.viewport;
                for (i, d) in delta.iter().enumerate() {
                    n.origin[i] += d;
                }
                n.bounds(self.limits.visible_cells)?;
                self.viewport = n;
            }
            A::Select(cell) => {
                if self.world.model().cell(cell).is_none() {
                    return Err(WorldPanelError::InvalidOperation);
                }
                self.selected = Some(cell);
            }
            A::AcquireLease => {
                self.validate_generation(&self.identity)?;
                let cell = self.selected_cell()?;
                let expiry = self
                    .now
                    .checked_add(self.limits.lease_ttl)
                    .ok_or(WorldPanelError::InvalidClock)?;
                let key = CellKey::new(cell);
                if let Some(l) = self.leases.get(&key).filter(|l| l.expiry > self.now) {
                    self.world
                        .renew_lock(&l.token, self.now, self.limits.lease_ttl)?;
                    self.leases.get_mut(&key).expect("retained lease").expiry = expiry;
                } else {
                    if !self.leases.contains_key(&key) && self.leases.len() >= self.limits.leases {
                        return Err(WorldPanelError::Capacity);
                    }
                    let token = self.world.acquire_lock(
                        cell,
                        self.owner,
                        self.now,
                        self.limits.lease_ttl,
                    )?;
                    self.leases.insert(key, Lease { token, expiry });
                }
            }
            A::ReleaseLease => {
                let key = CellKey::new(self.selected_cell()?);
                let l = self
                    .leases
                    .get(&key)
                    .ok_or(WorldPanelError::InvalidOperation)?;
                if l.expiry > self.now {
                    self.world.release_lock(&l.token)?;
                }
                self.leases.remove(&key);
            }
            A::Load | A::Unload | A::Save => {
                self.validate_generation(&self.identity)?;
                let cell = self.selected_cell()?;
                let r = self
                    .world
                    .model()
                    .cell(cell)
                    .ok_or(WorldPanelError::InvalidOperation)?;
                let kind = match action {
                    A::Load if r.residency() == CellResidency::Unloaded => {
                        WorldPanelIntentKind::Load(cell)
                    }
                    A::Unload
                        if r.residency() == CellResidency::Loaded
                            && !r.is_dirty()
                            && !matches!(self.lease(cell), WorldPanelLease::Owned { .. }) =>
                    {
                        WorldPanelIntentKind::Unload(cell)
                    }
                    A::Save if r.residency() == CellResidency::Loaded && r.is_dirty() => {
                        self.lease_token(cell)?;
                        WorldPanelIntentKind::Save(cell)
                    }
                    _ => return Err(WorldPanelError::InvalidOperation),
                };
                self.enqueue(kind)?;
            }
            A::CrossCellSource => self.source = Some(self.selected_cell()?),
            A::CrossCell => {
                let source = self.source.ok_or(WorldPanelError::InvalidOperation)?;
                let destination = self.selected_cell()?;
                if source == destination {
                    return Err(WorldPanelError::InvalidOperation);
                }
                self.lease_token(source)?;
                self.lease_token(destination)?;
                if !self.world.model().is_cell_editable(source)
                    || !self.world.model().is_cell_editable(destination)
                {
                    return Err(WorldPanelError::InvalidOperation);
                }
                self.enqueue(WorldPanelIntentKind::CrossCell {
                    source,
                    destination,
                })?;
            }
            A::LayerVisible { id, visible } => {
                let mut layer = self
                    .project
                    .layer(id)
                    .ok_or(WorldPanelError::InvalidOperation)?
                    .clone();
                layer.visible = visible;
                self.edit_project(&[ProjectEdit::Layer {
                    id,
                    value: Some(layer),
                }])?;
            }
            A::LayerEditable { id, editable } => {
                let mut layer = self
                    .project
                    .layer(id)
                    .ok_or(WorldPanelError::InvalidOperation)?
                    .clone();
                layer.editable = editable;
                self.edit_project(&[ProjectEdit::Layer {
                    id,
                    value: Some(layer),
                }])?;
            }
        }
        self.invalidate();
        Ok(())
    }
    /// Explicit host execution: existing domain planning scans the model.
    /// Preserve other resident/loading cells; never evict them due to one click.
    pub fn begin_residency(
        &mut self,
        ticket: &WorldPanelIntent,
    ) -> Result<WorldPanelTransition, WorldPanelError> {
        self.validate_intent(ticket)?;
        let (cell, load) = match ticket.kind {
            WorldPanelIntentKind::Load(c) => (c, true),
            WorldPanelIntentKind::Unload(c) => (c, false),
            _ => return Err(WorldPanelError::InvalidOperation),
        };
        let mut demand = ResidencyDemand::new();
        for r in self.world.model().cells() {
            if r.coord() != cell && r.residency() != CellResidency::Unloaded {
                demand.require(r.coord());
            }
        }
        if load {
            demand.require(cell);
        }
        let (_, tokens) = self.world.begin_residency(&demand, self.now)?;
        self.invalidate();
        tokens
            .into_iter()
            .find(|t| t.cell() == cell)
            .map(|token| WorldPanelTransition {
                token,
                owner: self.identity.clone(),
            })
            .ok_or(WorldPanelError::InvalidOperation)
    }
    /// Real revision-aware I/O driver load completion, including cancellation.
    pub fn apply_load_completion<E>(
        &mut self,
        transition: &WorldPanelTransition,
        c: IoCompletion<E>,
    ) -> Result<bool, WorldPanelError> {
        self.validate_generation(&transition.owner)?;
        let token = match &c {
            IoCompletion::Loaded { request, .. } => Some(request.token()),
            IoCompletion::Failed {
                operation: prism_editor_world::IoOperation::Load(request),
                ..
            }
            | IoCompletion::Cancelled {
                operation: prism_editor_world::IoOperation::Load(request),
                ..
            } => Some(request.token()),
            _ => None,
        };
        if token != Some(transition.token) {
            return Err(WorldPanelError::StaleIntent);
        }
        self.invalidate();
        Ok(self.world.apply_load_completion(c)?)
    }
    /// Complete a host load of opaque bytes. For durable revisions use the
    /// revision-aware driver completion method instead.
    pub fn complete_load(
        &mut self,
        t: WorldPanelTransition,
        bytes: Vec<u8>,
    ) -> Result<(), WorldPanelError> {
        self.validate_generation(&t.owner)?;
        self.invalidate();
        Ok(self.world.complete_load(t.token, bytes)?)
    }
    /// Complete unload after host consumers are actually released.
    pub fn complete_unload(&mut self, t: WorldPanelTransition) -> Result<(), WorldPanelError> {
        self.validate_generation(&t.owner)?;
        self.invalidate();
        Ok(self.world.complete_unload(t.token)?)
    }
    /// Fail/cancel a transition using the actual token, preserving dirty bytes.
    pub fn fail_residency(&mut self, t: WorldPanelTransition) -> Result<(), WorldPanelError> {
        self.validate_generation(&t.owner)?;
        self.invalidate();
        match t.token.kind() {
            TransitionKind::Load => self.world.fail_load(t.token)?,
            TransitionKind::Unload => self.world.fail_unload(t.token)?,
        }
        Ok(())
    }
    /// Replace real opaque resident bytes under a live controller lease.
    pub fn replace_payload(
        &mut self,
        cell: CellCoord,
        bytes: Vec<u8>,
    ) -> Result<(), WorldPanelError> {
        self.validate_generation(&self.identity)?;
        self.world
            .replace_payload(cell, &self.lease_token(cell)?, self.now, bytes)?;
        self.invalidate();
        Ok(())
    }
    /// Capture exact opaque bytes/revision/lease for host I/O.
    pub fn prepare_save(&self, i: &WorldPanelIntent) -> Result<WorldPanelSave, WorldPanelError> {
        self.validate_intent(i)?;
        let WorldPanelIntentKind::Save(cell) = i.kind else {
            return Err(WorldPanelError::InvalidOperation);
        };
        let pending = self
            .world
            .prepare_save(cell, &self.lease_token(cell)?, self.now)?;
        Ok(WorldPanelSave {
            pending,
            owner: self.identity.clone(),
        })
    }
    /// Reconcile actual driver success/failure; stale bytes, dirty actors and
    /// expired leases retain dirty state through the real world safeguards.
    pub fn apply_save_completion<E>(
        &mut self,
        p: &WorldPanelSave,
        c: IoCompletion<E>,
    ) -> Result<bool, WorldPanelError> {
        self.validate_generation(&p.owner)?;
        self.invalidate();
        Ok(self.world.apply_save_completion(&p.pending, c, self.now)?)
    }
    /// Prepare actual coordinated publication from host-supplied opaque edits.
    /// Both cell payloads are required; edits outside the two owners fail first.
    pub fn prepare_transaction(
        &mut self,
        ticket: &WorldPanelIntent,
        id: StableGuid,
        snapshot: &WorldTransactionSnapshot,
        edits: &[WorldTransactionEdit],
        provider: &FileWorldTransactionProvider,
    ) -> Result<WorldPanelTransaction, WorldPanelError> {
        self.validate_intent(ticket)?;
        let WorldPanelIntentKind::CrossCell {
            source,
            destination,
        } = ticket.kind
        else {
            return Err(WorldPanelError::InvalidOperation);
        };
        let mut src = false;
        let mut dst = false;
        for e in edits {
            let cell = match e {
                WorldTransactionEdit::Cell { cell, .. } => {
                    src |= *cell == source;
                    dst |= *cell == destination;
                    *cell
                }
                WorldTransactionEdit::Actor { actor, cell, .. } => {
                    let old = self
                        .world
                        .actor_record(*actor)
                        .ok_or(WorldPanelError::InvalidOperation)?
                        .identity()
                        .cell();
                    if old != source && old != destination {
                        return Err(WorldPanelError::InvalidOperation);
                    }
                    *cell
                }
                WorldTransactionEdit::DeleteActor(actor) => self
                    .world
                    .actor_record(*actor)
                    .ok_or(WorldPanelError::InvalidOperation)?
                    .identity()
                    .cell(),
            };
            if cell != source && cell != destination {
                return Err(WorldPanelError::InvalidOperation);
            }
        }
        if !src || !dst {
            return Err(WorldPanelError::InvalidOperation);
        }
        let leases = [self.lease_token(source)?, self.lease_token(destination)?];
        let p = self
            .world
            .prepare_world_transaction(id, snapshot, edits, &leases, self.now, provider)?;
        self.invalidate();
        Ok(WorldPanelTransaction {
            pending: p,
            owner: self.identity.clone(),
        })
    }
    /// Reconcile durable transaction receipt through all domain lease/dirty guards.
    pub fn commit_transaction(
        &mut self,
        p: &WorldPanelTransaction,
        r: &WorldTransactionReceipt,
    ) -> Result<(), WorldPanelError> {
        self.validate_generation(&p.owner)?;
        let mut actors: BTreeMap<_, _> = self.project.actors().collect();
        for change in p.request().changes() {
            if let prism_editor_world::WorldFileTarget::Actor(actor) = change.target()
                && change.value().is_none()
            {
                actors.remove(&actor.actor());
            }
        }
        for change in p.request().changes() {
            if let prism_editor_world::WorldFileTarget::Actor(actor) = change.target()
                && change.value().is_some()
            {
                actors.insert(actor.actor(), actor.cell());
            }
        }
        let next = self.project_with_actors(actors)?;
        self.world
            .commit_world_transaction(&p.pending, r, self.now)?;
        self.project = next;
        self.invalidate();
        Ok(())
    }
    /// Explicitly discard local edits and restore a complete durable snapshot.
    /// This host recovery operation scans the model; every registered cell must
    /// be loaded, editable and covered by a live retained lease. Never called
    /// automatically on a stale receipt, since that would lose newer edits.
    pub fn discard_edits_and_restore_transaction_snapshot(
        &mut self,
        snapshot: &WorldTransactionSnapshot,
    ) -> Result<(), WorldPanelError> {
        self.validate_generation(&self.identity)?;
        let actors = snapshot
            .files()
            .iter()
            .filter_map(|f| match f.target() {
                prism_editor_world::WorldFileTarget::Actor(a) => Some((a.actor(), a.cell())),
                _ => None,
            })
            .collect();
        let next = self.project_with_actors(actors)?;
        let leases: Vec<_> = self
            .leases
            .values()
            .filter(|l| l.expiry > self.now)
            .map(|l| l.token.clone())
            .collect();
        self.world
            .discard_edits_and_restore_transaction_snapshot(snapshot, &leases, self.now)?;
        self.project = next;
        self.invalidate();
        Ok(())
    }
    fn project_with_actors(
        &self,
        actors: BTreeMap<StableGuid, CellCoord>,
    ) -> Result<WorldProject, WorldPanelError> {
        let mut edits: Vec<_> = self
            .project
            .actors()
            .filter(|(a, _)| !actors.contains_key(a))
            .map(|(actor, _)| ProjectEdit::Actor { actor, cell: None })
            .collect();
        edits.extend(
            actors
                .into_iter()
                .filter(|(a, c)| self.project.actor_cell(*a) != Some(*c))
                .map(|(actor, cell)| ProjectEdit::Actor {
                    actor,
                    cell: Some(cell),
                }),
        );
        let mut next = self.project.clone();
        if !edits.is_empty() {
            next.apply(
                next.revision(),
                &edits,
                self.project_limits.project,
                self.project_limits.project.max_bytes.saturating_mul(8),
            )?;
        }
        Ok(next)
    }
    /// Return the actual model, including outstanding leases and dirty state.
    pub fn into_world(self) -> EditorWorldData {
        self.world
    }
}
