//! Bounded authored-override controller over the real Prefab command kernel.
use crate::{prefab_view::BoundPrefabPanel, SelectionMode, UiEvent};
use prism_editor_core::{prefab::*, CommandError, CommandStack};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::Arc,
};

// Explicit review work bound, independent of the library's total owner count.
const REVIEW_STEPS: usize = 8192;

pub(crate) struct ApplyReview {
    pub(crate) source: PrefabOwner,
    pub(crate) target: PrefabId,
    pub(crate) properties: Vec<PrefabProperty>,
    pub(crate) pins: BTreeMap<(PrefabOwner, PrefabProperty), PrefabOverride>,
    pub(crate) blocked: bool,
}

/// Limits on UI materialization, selection, propagation and retained history.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefabPanelLimits {
    /// Override rows per page, 1..=128 (one extra entry is inspected).
    pub rows: usize,
    /// Selected properties across pages, 1..=128.
    pub selection: usize,
    /// Host-contributed propagation candidates, 1..=128.
    pub recipients: usize,
    /// Maximum bytes in one copied component/field/slot address, 1..=16384.
    pub address_bytes: usize,
    /// Retained undo plus redo entries, 1..=1024; full history rejects new edits.
    pub history: usize,
}
impl Default for PrefabPanelLimits {
    fn default() -> Self {
        Self {
            rows: 32,
            selection: 64,
            recipients: 64,
            address_bytes: 4096,
            history: 128,
        }
    }
}
impl PrefabPanelLimits {
    fn validate(self) -> Result<(), PrefabPanelError> {
        for (n, max) in [
            (self.rows, 128),
            (self.selection, 128),
            (self.recipients, 128),
            (self.address_bytes, 16384),
            (self.history, 1024),
        ] {
            if n == 0 || n > max {
                return Err(PrefabPanelError::Capacity);
            }
        }
        Ok(())
    }
    pub(crate) fn address(self, p: &PrefabProperty) -> Result<(), PrefabPanelError> {
        if p.fields.len() > 128 || p.entity.slots.len() > 128 {
            return Err(PrefabPanelError::Capacity);
        }
        let base = p.component.len().checked_add(size_of::<PrefabProperty>());
        let bytes = base.and_then(|base| {
            p.fields.iter().try_fold(base, |n, k| {
                n.checked_add(k.len())
                    .and_then(|n| n.checked_add(8))
                    .and_then(|n| n.checked_add(size_of::<String>()))
            })
        });
        let bytes = bytes.and_then(|n| {
            p.entity
                .slots
                .len()
                .checked_mul(16)
                .and_then(|s| n.checked_add(s))
        });
        if bytes.is_none_or(|n| n > self.address_bytes) {
            return Err(PrefabPanelError::Capacity);
        }
        Ok(())
    }
}
/// Exact model-publication capability. Never compare only numeric revisions.
#[derive(Clone, Debug)]
pub struct PrefabPanelSnapshot {
    identity: Arc<()>,
    revision: u64,
}
impl PrefabPanelSnapshot {
    /// Kernel revision for diagnostics; identity remains part of the fence.
    pub const fn revision(&self) -> u64 {
        self.revision
    }
}
/// Explicit panel operations; every authoring action uses one real command.
#[derive(Clone, Debug)]
pub enum PrefabPanelAction {
    /// Inspect an existing asset/instance and clear selection/page/targets.
    Owner(PrefabOwner),
    /// Select an exact authored property; Extend adds only this property.
    Select(PrefabProperty, SelectionMode),
    /// Clear property selection.
    Clear,
    /// First page.
    First,
    /// Page strictly after a stable property key.
    After(PrefabProperty),
    /// Page strictly before a stable property key.
    Before(PrefabProperty),
    /// Toggle one host-contributed propagation candidate.
    ToggleRecipient(PrefabOwner),
    /// Revert all selected explicit overrides in one atomic command.
    Revert,
    /// Move selected explicit overrides to the configured ancestor atomically.
    ApplyBase,
    /// Review bounded overlapping pins along the configured apply ancestry.
    ReviewApplyBase,
    /// Cancel the current review, without touching authoring or history.
    CancelApplyReview,
    /// Revert reviewed exact intervening pins and apply in one real command.
    /// Broader/narrower pins must be resolved separately and cannot be discarded.
    ConfirmApplyReview,
    /// Pin selected asset values into selected proper descendants atomically.
    Propagate,
    /// Real command history undo.
    Undo,
    /// Real command history redo.
    Redo,
}
/// Explicit rejection; failed operations preserve model, history and view state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PrefabPanelError {
    /// A panel bound, address budget, command budget or history ceiling was hit.
    Capacity,
    /// Missing target, selection, or invalid recipient configuration.
    Target,
    /// Foreign controller, replaced store, old model or old view publication.
    StaleView,
    /// No enabled painted control matches the event.
    UnboundEvent,
    /// Typed kernel read/lookup rejection.
    Kernel(PrefabError),
    /// Atomic Prefab command or history rejected by the kernel.
    Command(CommandError),
}
impl fmt::Display for PrefabPanelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kernel(e) => e.fmt(f),
            Self::Command(e) => e.fmt(f),
            _ => write!(f, "{self:?}"),
        }
    }
}
impl std::error::Error for PrefabPanelError {}
impl From<PrefabError> for PrefabPanelError {
    fn from(e: PrefabError) -> Self {
        Self::Kernel(e)
    }
}
impl From<CommandError> for PrefabPanelError {
    fn from(e: CommandError) -> Self {
        Self::Command(e)
    }
}

/// Owns the actual store/history. No mutable store escape can bypass fences.
/// Builds borrow the authored map, cloning bounded addresses and value Arcs only.
pub struct PrefabPanelController {
    store: PrefabStore,
    history: CommandStack<PrefabStore>,
    pub(crate) owner: PrefabOwner,
    pub(crate) limits: PrefabPanelLimits,
    pub(crate) selected: BTreeSet<PrefabProperty>,
    pub(crate) cursor: Option<(PrefabProperty, bool)>,
    pub(crate) base: Option<PrefabId>,
    pub(crate) candidates: Vec<PrefabOwner>,
    pub(crate) recipients: BTreeSet<PrefabOwner>,
    pub(crate) review: Option<ApplyReview>,
    model: Arc<()>,
    pub(crate) view: Arc<()>,
}
impl PrefabPanelController {
    /// Take ownership of a validated real store, without a clone or composition.
    pub fn new(
        store: PrefabStore,
        owner: PrefabOwner,
        limits: PrefabPanelLimits,
    ) -> Result<Self, PrefabPanelError> {
        limits.validate()?;
        store.layer(owner)?;
        Ok(Self {
            store,
            history: CommandStack::default(),
            owner,
            limits,
            selected: BTreeSet::new(),
            cursor: None,
            base: None,
            candidates: Vec::new(),
            recipients: BTreeSet::new(),
            review: None,
            model: Arc::new(()),
            view: Arc::new(()),
        })
    }
    /// Actual authoring model, including composition/codec APIs, read-only.
    pub const fn store(&self) -> &PrefabStore {
        &self.store
    }
    /// Recover model ownership when closing the panel.
    pub fn into_store(self) -> PrefabStore {
        self.store
    }
    /// Inspected layer identity.
    pub const fn owner(&self) -> PrefabOwner {
        self.owner
    }
    /// Exact stable selected addresses across pages.
    pub fn selected(&self) -> &BTreeSet<PrefabProperty> {
        &self.selected
    }
    /// Actual undo and redo depths.
    pub fn history_depths(&self) -> (usize, usize) {
        (self.history.undo_len(), self.history.redo_len())
    }
    /// Capture a cheap exact model capability, without copying any record.
    pub fn snapshot(&self) -> PrefabPanelSnapshot {
        PrefabPanelSnapshot {
            identity: self.model.clone(),
            revision: self.store.revision(),
        }
    }
    /// True only for this controller's current exact model publication.
    pub fn matches_snapshot(&self, token: &PrefabPanelSnapshot) -> bool {
        Arc::ptr_eq(&self.model, &token.identity) && self.store.revision() == token.revision
    }
    fn check(&self, token: &PrefabPanelSnapshot) -> Result<(), PrefabPanelError> {
        if self.matches_snapshot(token) {
            Ok(())
        } else {
            Err(PrefabPanelError::StaleView)
        }
    }
    /// Configure an explicit ancestor and bounded candidate list; all start checked.
    /// The kernel validates ancestry/conflicts on execution. No descendant scan.
    /// Changes invalidate old views, but do not change the model snapshot.
    pub fn configure_targets(
        &mut self,
        expected: &PrefabPanelSnapshot,
        base: Option<PrefabId>,
        recipients: Vec<PrefabOwner>,
    ) -> Result<(), PrefabPanelError> {
        self.check(expected)?;
        if recipients.len() > self.limits.recipients {
            return Err(PrefabPanelError::Capacity);
        }
        if let Some(id) = base {
            self.store.layer(PrefabOwner::Asset(id))?;
        }
        let mut seen = BTreeSet::new();
        for &owner in &recipients {
            self.store.layer(owner)?;
            if !seen.insert(owner) {
                return Err(PrefabPanelError::Target);
            }
        }
        self.base = base;
        self.candidates = recipients;
        self.recipients = seen;
        self.review = None;
        self.view = Arc::new(());
        Ok(())
    }
    /// Host authoring ingress: one revision-fenced atomic PrefabCommand.
    /// A failed command changes neither model, history, selection nor fences.
    pub fn execute(
        &mut self,
        expected: &PrefabPanelSnapshot,
        edits: Vec<PrefabEdit>,
    ) -> Result<(), PrefabPanelError> {
        self.check(expected)?;
        if edits.is_empty() {
            return Ok(());
        }
        if edits.len() > self.store.limits().max_edits
            || self.history.undo_len() >= self.limits.history
        {
            return Err(PrefabPanelError::Capacity);
        }
        self.history.execute(
            Box::new(PrefabCommand::new(
                "Prefab panel",
                self.store.revision(),
                edits,
            )),
            &mut self.store,
        )?;
        self.published_model();
        Ok(())
    }
    /// Atomically replace a loaded store; same-revision reloads also invalidate
    /// all old capabilities. History/selection/targets reset without cloning.
    pub fn replace_store(
        &mut self,
        expected: &PrefabPanelSnapshot,
        store: PrefabStore,
        owner: PrefabOwner,
    ) -> Result<(), PrefabPanelError> {
        self.check(expected)?;
        store.layer(owner)?;
        self.store = store;
        self.owner = owner;
        self.history = CommandStack::default();
        self.selected.clear();
        self.base = None;
        self.candidates.clear();
        self.recipients.clear();
        self.published_model();
        Ok(())
    }
    fn published_model(&mut self) {
        self.review = None;
        self.selected.retain(|p| {
            self.store
                .layer(self.owner)
                .is_ok_and(|l| l.overrides.contains_key(p))
        });
        self.candidates.retain(|o| self.store.layer(*o).is_ok());
        self.recipients.retain(|o| self.store.layer(*o).is_ok());
        if self.base.is_some_and(|id| self.store.asset(id).is_none()) {
            self.base = None;
        }
        self.cursor = None;
        self.model = Arc::new(());
        self.view = Arc::new(());
    }
    /// Materialize only one authored-override page and bounded controls.
    pub fn build(&self) -> Result<BoundPrefabPanel, PrefabPanelError> {
        crate::prefab_view::build(self)
    }
    /// Reject foreign/stale view and model capabilities before interpreting keys.
    pub fn handle_event(
        &mut self,
        view: &BoundPrefabPanel,
        key: &str,
        event: &UiEvent,
    ) -> Result<(), PrefabPanelError> {
        if !Arc::ptr_eq(&self.view, &view.view) {
            return Err(PrefabPanelError::StaleView);
        }
        self.check(view.snapshot())?;
        self.apply(view.dispatch(key, event)?)
    }
    /// Trusted host action ingress; untrusted UI events must use handle_event.
    pub fn apply(&mut self, action: PrefabPanelAction) -> Result<(), PrefabPanelError> {
        use PrefabPanelAction as A;
        match action {
            A::ReviewApplyBase => {
                let review = self.review_apply()?;
                self.review = Some(review);
                self.view = Arc::new(());
                return Ok(());
            }
            A::CancelApplyReview => {
                self.review = None;
            }
            A::ConfirmApplyReview => {
                let review = self.review.as_ref().ok_or(PrefabPanelError::Target)?;
                if review.blocked {
                    return Err(PrefabPanelError::Target);
                }
                let mut edits = Vec::new();
                for (owner, property) in review.pins.keys() {
                    if *owner != PrefabOwner::Asset(review.target) {
                        edits.push(PrefabEdit::Revert {
                            owner: *owner,
                            property: property.clone(),
                        });
                    }
                }
                for property in &review.properties {
                    edits.push(PrefabEdit::ApplyToBase {
                        source: review.source,
                        target: review.target,
                        property: property.clone(),
                    });
                }
                return self.execute(&self.snapshot(), edits);
            }
            A::Owner(owner) => {
                self.store.layer(owner)?;
                self.owner = owner;
                self.selected.clear();
                self.cursor = None;
                self.base = None;
                self.candidates.clear();
                self.recipients.clear();
            }
            A::Select(p, mode) => {
                self.limits.address(&p)?;
                if !self.store.layer(self.owner)?.overrides.contains_key(&p) {
                    return Err(PrefabPanelError::Target);
                }
                if mode == SelectionMode::Replace {
                    self.selected.clear();
                }
                if mode == SelectionMode::Toggle && self.selected.remove(&p) {
                    // Toggling an existing address off needs no new capacity.
                } else {
                    if !self.selected.contains(&p) && self.selected.len() >= self.limits.selection {
                        return Err(PrefabPanelError::Capacity);
                    }
                    self.selected.insert(p);
                }
            }
            A::Clear => self.selected.clear(),
            A::First => self.cursor = None,
            A::After(p) => {
                self.page(p, false)?;
            }
            A::Before(p) => {
                self.page(p, true)?;
            }
            A::ToggleRecipient(o) => {
                if !self.candidates.contains(&o) {
                    return Err(PrefabPanelError::Target);
                }
                if !self.recipients.remove(&o) {
                    self.recipients.insert(o);
                }
            }
            A::Undo | A::Redo => {
                let changed = if matches!(action, A::Undo) {
                    self.history.undo(&mut self.store)?
                } else {
                    self.history.redo(&mut self.store)?
                };
                if changed {
                    self.published_model();
                }
                return Ok(());
            }
            A::Revert | A::ApplyBase | A::Propagate => {
                if self.selected.is_empty() {
                    return Err(PrefabPanelError::Target);
                }
                let mut edits = Vec::with_capacity(self.selected.len());
                for p in &self.selected {
                    if !self.store.layer(self.owner)?.overrides.contains_key(p) {
                        return Err(PrefabPanelError::Target);
                    }
                    edits.push(match action {
                        A::Revert => PrefabEdit::Revert {
                            owner: self.owner,
                            property: p.clone(),
                        },
                        A::ApplyBase => PrefabEdit::ApplyToBase {
                            source: self.owner,
                            target: self.base.ok_or(PrefabPanelError::Target)?,
                            property: p.clone(),
                        },
                        _ => {
                            let PrefabOwner::Asset(source) = self.owner else {
                                return Err(PrefabPanelError::Target);
                            };
                            if self.recipients.is_empty() {
                                return Err(PrefabPanelError::Target);
                            }
                            let work = self
                                .selected
                                .len()
                                .checked_mul(self.recipients.len() + 1)
                                .ok_or(PrefabPanelError::Capacity)?;
                            if work > self.store.limits().max_edits {
                                return Err(PrefabPanelError::Capacity);
                            }
                            PrefabEdit::Propagate {
                                source,
                                property: p.clone(),
                                recipients: self.recipients.iter().copied().collect(),
                            }
                        }
                    });
                }
                return self.execute(&self.snapshot(), edits);
            }
        }
        self.review = None;
        self.view = Arc::new(());
        Ok(())
    }
    fn review_apply(&self) -> Result<ApplyReview, PrefabPanelError> {
        let target = self.base.ok_or(PrefabPanelError::Target)?;
        if self.selected.is_empty() || self.owner == PrefabOwner::Asset(target) {
            return Err(PrefabPanelError::Target);
        }
        let mut cursor = match self.owner {
            PrefabOwner::Asset(id) => self.store.asset(id).ok_or(PrefabPanelError::Target)?.base,
            PrefabOwner::Instance(id) => Some(
                self.store
                    .instance(id)
                    .ok_or(PrefabPanelError::Target)?
                    .prefab,
            ),
        };
        let mut review = ApplyReview {
            source: self.owner,
            target,
            properties: self.selected.iter().cloned().collect(),
            pins: BTreeMap::new(),
            blocked: false,
        };
        let mut steps = 0;
        let mut depth = 0;
        while let Some(id) = cursor {
            depth += 1;
            if depth > self.store.limits().max_depth {
                return Err(PrefabPanelError::Capacity);
            }
            let owner = PrefabOwner::Asset(id);
            let asset = self.store.asset(id).ok_or(PrefabPanelError::Target)?;
            let overrides = &asset.layer.overrides;
            for p in &self.selected {
                let mut ancestor = p.clone();
                // Exact/descendant keys form a contiguous range. Prefix keys
                // preceding it require at most the bounded semantic depth lookups.
                while !ancestor.fields.is_empty() {
                    ancestor.fields.pop();
                    Self::review_step(&mut steps)?;
                    if let Some(value) = overrides.get(&ancestor) {
                        self.review_pin(&mut review, owner, &ancestor, value, true)?;
                    }
                }
                for (key, value) in overrides.range(p.clone()..) {
                    Self::review_step(&mut steps)?;
                    if key.entity != p.entity
                        || key.component != p.component
                        || !key.fields.starts_with(&p.fields)
                    {
                        break;
                    }
                    self.review_pin(&mut review, owner, key, value, key != p)?;
                }
            }
            if id == target {
                let reverts = review
                    .pins
                    .keys()
                    .filter(|(owner, _)| *owner != PrefabOwner::Asset(target))
                    .count();
                if reverts + review.properties.len() > self.store.limits().max_edits {
                    return Err(PrefabPanelError::Capacity);
                }
                return Ok(review);
            }
            cursor = asset.base;
        }
        Err(PrefabPanelError::Target)
    }
    fn review_step(steps: &mut usize) -> Result<(), PrefabPanelError> {
        *steps += 1;
        if *steps > REVIEW_STEPS {
            return Err(PrefabPanelError::Capacity);
        }
        Ok(())
    }
    fn review_pin(
        &self,
        review: &mut ApplyReview,
        owner: PrefabOwner,
        property: &PrefabProperty,
        value: &PrefabOverride,
        blocked: bool,
    ) -> Result<(), PrefabPanelError> {
        self.limits.address(property)?;
        if !review.pins.contains_key(&(owner, property.clone()))
            && review.pins.len() >= self.limits.selection
        {
            return Err(PrefabPanelError::Capacity);
        }
        review.blocked |= blocked;
        review.pins.insert((owner, property.clone()), value.clone());
        Ok(())
    }
    fn page(&mut self, p: PrefabProperty, before: bool) -> Result<(), PrefabPanelError> {
        self.limits.address(&p)?;
        let overrides = &self.store.layer(self.owner)?.overrides;
        let edge = if before {
            overrides.first_key_value()
        } else {
            overrides.last_key_value()
        };
        if !overrides.contains_key(&p) || edge.is_none_or(|(key, _)| key == &p) {
            return Err(PrefabPanelError::Target);
        }
        self.cursor = Some((p, before));
        Ok(())
    }
}
