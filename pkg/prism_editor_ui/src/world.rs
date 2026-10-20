//! Bounded world-panel commands over the real engine-neutral authoring model.
use crate::{world_view::BoundWorldPanel, UiEvent};
use prism_asset::StableGuid;
use prism_ecs::partition::{cell::CellCoord, data_layer::DataLayerId};
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
struct Lease {
    token: CellLockToken,
    expiry: u64,
}
/// Owns the real model and leases. Async I/O is explicitly handed to the host.
pub struct WorldPanelController {
    world: EditorWorldData,
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
    /// Takes ownership without scanning cells or guessing external lock owners.
    pub fn new(
        world: EditorWorldData,
        owner: StableGuid,
        viewport: WorldPanelViewport,
        limits: WorldPanelLimits,
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
        Ok(Self {
            world,
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
        if !Arc::ptr_eq(&i.owner, &self.identity) || i.epoch != self.world.model().epoch() {
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
            A::LayerVisible { id, visible } => self.world.set_layer_visible(id, visible)?,
            A::LayerEditable { id, editable } => self.world.set_layer_editable(id, editable)?,
        }
        self.invalidate();
        Ok(())
    }
    /// Explicit host execution: existing domain planning scans the model.
    /// Preserve other resident/loading cells; never evict them due to one click.
    pub fn begin_residency(
        &mut self,
        ticket: &WorldPanelIntent,
    ) -> Result<TransitionToken, WorldPanelError> {
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
            .ok_or(WorldPanelError::InvalidOperation)
    }
    /// Real revision-aware I/O driver load completion, including cancellation.
    pub fn apply_load_completion<E>(
        &mut self,
        c: IoCompletion<E>,
    ) -> Result<bool, WorldPanelError> {
        self.invalidate();
        Ok(self.world.apply_load_completion(c)?)
    }
    /// Complete a host load of opaque bytes. For durable revisions use the
    /// revision-aware driver completion method instead.
    pub fn complete_load(
        &mut self,
        t: TransitionToken,
        bytes: Vec<u8>,
    ) -> Result<(), WorldPanelError> {
        self.invalidate();
        Ok(self.world.complete_load(t, bytes)?)
    }
    /// Complete unload after host consumers are actually released.
    pub fn complete_unload(&mut self, t: TransitionToken) -> Result<(), WorldPanelError> {
        self.invalidate();
        Ok(self.world.complete_unload(t)?)
    }
    /// Fail/cancel a transition using the actual token, preserving dirty bytes.
    pub fn fail_residency(&mut self, t: TransitionToken) -> Result<(), WorldPanelError> {
        self.invalidate();
        match t.kind() {
            TransitionKind::Load => self.world.fail_load(t)?,
            TransitionKind::Unload => self.world.fail_unload(t)?,
        }
        Ok(())
    }
    /// Replace real opaque resident bytes under a live controller lease.
    pub fn replace_payload(
        &mut self,
        cell: CellCoord,
        bytes: Vec<u8>,
    ) -> Result<(), WorldPanelError> {
        self.world
            .replace_payload(cell, &self.lease_token(cell)?, self.now, bytes)?;
        self.invalidate();
        Ok(())
    }
    /// Capture exact opaque bytes/revision/lease for host I/O.
    pub fn prepare_save(&self, i: &WorldPanelIntent) -> Result<PendingCellSave, WorldPanelError> {
        self.validate_intent(i)?;
        let WorldPanelIntentKind::Save(cell) = i.kind else {
            return Err(WorldPanelError::InvalidOperation);
        };
        Ok(self
            .world
            .prepare_save(cell, &self.lease_token(cell)?, self.now)?)
    }
    /// Reconcile actual driver success/failure; stale bytes, dirty actors and
    /// expired leases retain dirty state through the real world safeguards.
    pub fn apply_save_completion<E>(
        &mut self,
        p: &PendingCellSave,
        c: IoCompletion<E>,
    ) -> Result<bool, WorldPanelError> {
        self.invalidate();
        Ok(self.world.apply_save_completion(p, c, self.now)?)
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
    ) -> Result<PendingWorldTransaction, WorldPanelError> {
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
        Ok(p)
    }
    /// Reconcile durable transaction receipt through all domain lease/dirty guards.
    pub fn commit_transaction(
        &mut self,
        p: &PendingWorldTransaction,
        r: &WorldTransactionReceipt,
    ) -> Result<(), WorldPanelError> {
        self.invalidate();
        Ok(self.world.commit_world_transaction(p, r, self.now)?)
    }
    /// Explicitly discard local edits and restore a complete durable snapshot.
    /// This host recovery operation scans the model; every registered cell must
    /// be loaded, editable and covered by a live retained lease. Never called
    /// automatically on a stale receipt, since that would lose newer edits.
    pub fn discard_edits_and_restore_transaction_snapshot(
        &mut self,
        snapshot: &WorldTransactionSnapshot,
    ) -> Result<(), WorldPanelError> {
        let leases: Vec<_> = self
            .leases
            .values()
            .filter(|l| l.expiry > self.now)
            .map(|l| l.token.clone())
            .collect();
        self.world
            .discard_edits_and_restore_transaction_snapshot(snapshot, &leases, self.now)?;
        self.invalidate();
        Ok(())
    }
    /// Return the actual model, including outstanding leases and dirty state.
    pub fn into_world(self) -> EditorWorldData {
        self.world
    }
}
