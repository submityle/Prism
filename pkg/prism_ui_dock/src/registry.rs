//! Pure-data panel contributions for a dock host.
//!
//! The registry is intentionally independent of UI elements, renderers, and
//! panel factories. It provides the stable metadata and allow-list needed by
//! layout and persistence layers; a higher-level integration crate can attach
//! construction behavior without making this crate renderer-dependent.

use alloc::{string::String, vec::Vec};
use core::fmt;

use crate::PanelId;

/// Whether a panel supports one instance or multiple independent instances.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Instancing {
    /// At most one instance of this panel may be present in a workspace.
    #[default]
    Singleton,
    /// The host may create multiple instances of this panel.
    Multi,
}

/// Stable identity assigned to one materialized panel instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PanelInstanceId(u64);

impl PanelInstanceId {
    /// Creates an identity from a persisted numeric value.
    #[must_use]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    /// Returns the monotonic numeric identity.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for PanelInstanceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_fmt(format_args!("{}", self.0))
    }
}

/// Lifecycle state of a materialized panel instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PanelInstanceState {
    /// The instance can participate in layout and factory resolution.
    Open,
    /// The instance identity is retained but it is not currently visible.
    Closed,
}

/// Renderer-independent panel instance record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelInstance {
    id: PanelInstanceId,
    panel_id: PanelId,
    state: PanelInstanceState,
}

impl PanelInstance {
    /// Returns the stable instance identity.
    #[must_use]
    pub const fn id(&self) -> PanelInstanceId {
        self.id
    }

    /// Returns the descriptor ID that created this instance.
    #[must_use]
    pub fn panel_id(&self) -> &PanelId {
        &self.panel_id
    }

    /// Returns the current lifecycle state.
    #[must_use]
    pub const fn state(&self) -> PanelInstanceState {
        self.state
    }

    /// Returns whether the instance is currently open.
    #[must_use]
    pub const fn is_open(&self) -> bool {
        matches!(self.state, PanelInstanceState::Open)
    }
}

/// A failure while creating or changing a panel instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PanelInstanceRegistryError {
    /// The requested descriptor is not registered.
    UnknownPanel(PanelId),
    /// A singleton already owns an instance, including a closed restorable one.
    SingletonAlreadyExists {
        /// Descriptor ID of the singleton.
        panel_id: PanelId,
        /// Existing stable instance identity.
        instance_id: PanelInstanceId,
    },
    /// The requested instance identity is not known.
    UnknownInstance(PanelInstanceId),
    /// The requested instance is already closed.
    InstanceAlreadyClosed(PanelInstanceId),
    /// The requested instance is already open.
    InstanceAlreadyOpen(PanelInstanceId),
    /// No further stable identities can be allocated.
    InstanceIdExhausted,
    /// Persisted records violate ordering or next-ID invariants.
    InvalidPersistedState(&'static str),
}

impl fmt::Display for PanelInstanceRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownPanel(id) => {
                formatter.write_str("panel is not registered: ")?;
                formatter.write_str(id)
            }
            Self::SingletonAlreadyExists {
                panel_id,
                instance_id,
            } => {
                formatter.write_str("singleton already has instance ")?;
                instance_id.fmt(formatter)?;
                formatter.write_str(" for panel ")?;
                formatter.write_str(panel_id)
            }
            Self::UnknownInstance(id) => {
                formatter.write_str("panel instance is not known: ")?;
                id.fmt(formatter)
            }
            Self::InstanceAlreadyClosed(id) => {
                formatter.write_str("panel instance is already closed: ")?;
                id.fmt(formatter)
            }
            Self::InstanceAlreadyOpen(id) => {
                formatter.write_str("panel instance is already open: ")?;
                id.fmt(formatter)
            }
            Self::InstanceIdExhausted => formatter.write_str("panel instance ID space exhausted"),
            Self::InvalidPersistedState(kind) => {
                write!(formatter, "invalid persisted panel instance state: {kind}")
            }
        }
    }
}

/// Deterministic lifecycle registry for materialized panel instances.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelInstanceRegistry {
    instances: Vec<PanelInstance>,
    next_id: u64,
}

impl PanelInstanceRegistry {
    /// Creates an empty registry whose first identity is one.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            instances: Vec::new(),
            next_id: 1,
        }
    }

    /// Creates an open instance for a registered panel descriptor.
    ///
    /// Singleton descriptors retain their identity after close so that a
    /// caller can restore the same instance instead of creating a second one.
    pub fn create(
        &mut self,
        panels: &PanelRegistry,
        panel_id: &str,
    ) -> Result<PanelInstance, PanelInstanceRegistryError> {
        let descriptor = panels
            .get(panel_id)
            .ok_or_else(|| PanelInstanceRegistryError::UnknownPanel(panel_id.into()))?;
        if descriptor.instancing == Instancing::Singleton
            && let Some(existing) = self
                .instances
                .iter()
                .find(|instance| instance.panel_id == descriptor.id)
        {
            return Err(PanelInstanceRegistryError::SingletonAlreadyExists {
                panel_id: descriptor.id.clone(),
                instance_id: existing.id,
            });
        }
        let id = PanelInstanceId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(PanelInstanceRegistryError::InstanceIdExhausted)?;
        let instance = PanelInstance {
            id,
            panel_id: descriptor.id.clone(),
            state: PanelInstanceState::Open,
        };
        self.instances.push(instance.clone());
        Ok(instance)
    }

    /// Marks an open instance closed while retaining its stable identity.
    pub fn close(
        &mut self,
        instance_id: PanelInstanceId,
    ) -> Result<PanelInstance, PanelInstanceRegistryError> {
        let instance = self
            .get_mut(instance_id)
            .ok_or(PanelInstanceRegistryError::UnknownInstance(instance_id))?;
        if instance.state == PanelInstanceState::Closed {
            return Err(PanelInstanceRegistryError::InstanceAlreadyClosed(
                instance_id,
            ));
        }
        instance.state = PanelInstanceState::Closed;
        Ok(instance.clone())
    }

    /// Reopens a previously closed instance.
    pub fn restore(
        &mut self,
        instance_id: PanelInstanceId,
    ) -> Result<PanelInstance, PanelInstanceRegistryError> {
        let instance = self
            .get_mut(instance_id)
            .ok_or(PanelInstanceRegistryError::UnknownInstance(instance_id))?;
        if instance.state == PanelInstanceState::Open {
            return Err(PanelInstanceRegistryError::InstanceAlreadyOpen(instance_id));
        }
        instance.state = PanelInstanceState::Open;
        Ok(instance.clone())
    }

    /// Looks up an instance by stable identity.
    #[must_use]
    pub fn get(&self, instance_id: PanelInstanceId) -> Option<&PanelInstance> {
        self.instances
            .binary_search_by_key(&instance_id, |instance| instance.id)
            .ok()
            .map(|index| &self.instances[index])
    }

    /// Iterates over all instances in creation order.
    pub fn iter(&self) -> core::slice::Iter<'_, PanelInstance> {
        self.instances.iter()
    }

    /// Iterates over currently open instances in creation order.
    pub fn iter_open(&self) -> impl Iterator<Item = &PanelInstance> + '_ {
        self.instances.iter().filter(|instance| instance.is_open())
    }

    /// Iterates over currently open instances belonging to a descriptor.
    pub fn iter_open_for_panel<'a>(
        &'a self,
        panel_id: &'a str,
    ) -> impl Iterator<Item = &'a PanelInstance> + 'a {
        self.iter_open()
            .filter(move |instance| instance.panel_id == panel_id)
    }

    /// Returns the total number of retained instance records.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.instances.len()
    }

    /// Returns the number of currently open instances.
    #[must_use]
    pub fn open_len(&self) -> usize {
        self.iter_open().count()
    }

    /// Returns whether no instance record has been allocated.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Returns the next identity that will be allocated.
    #[must_use]
    pub const fn next_id(&self) -> u64 {
        self.next_id
    }

    pub(crate) fn from_persisted(
        next_id: u64,
        instances: Vec<PanelInstance>,
    ) -> Result<Self, PanelInstanceRegistryError> {
        if next_id == 0 {
            return Err(PanelInstanceRegistryError::InvalidPersistedState("next ID"));
        }
        let mut previous = None;
        for instance in &instances {
            if instance.id.get() == 0 || previous.is_some_and(|id| id >= instance.id.get()) {
                return Err(PanelInstanceRegistryError::InvalidPersistedState(
                    "instance ordering",
                ));
            }
            if instance.panel_id.trim().is_empty() {
                return Err(PanelInstanceRegistryError::InvalidPersistedState(
                    "panel ID",
                ));
            }
            previous = Some(instance.id.get());
        }
        if instances
            .last()
            .is_some_and(|instance| instance.id.get() >= next_id)
        {
            return Err(PanelInstanceRegistryError::InvalidPersistedState(
                "next ID is not ahead of instances",
            ));
        }
        Ok(Self { instances, next_id })
    }

    pub(crate) fn persisted_instance(
        id: PanelInstanceId,
        panel_id: PanelId,
        state: PanelInstanceState,
    ) -> PanelInstance {
        PanelInstance {
            id,
            panel_id,
            state,
        }
    }

    fn get_mut(&mut self, instance_id: PanelInstanceId) -> Option<&mut PanelInstance> {
        self.instances
            .binary_search_by_key(&instance_id, |instance| instance.id)
            .ok()
            .map(|index| &mut self.instances[index])
    }
}

impl Default for PanelInstanceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// The default logical destination for a panel in a newly created layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DefaultSlot {
    /// The left tool-panel bar or docking area.
    #[default]
    LeftBar,
    /// The right tool-panel bar or docking area.
    RightBar,
    /// The bottom tool-panel bar or docking area.
    Bottom,
    /// The document-oriented central area.
    Document,
    /// A floating surface.
    Floating,
}

/// Renderer-independent metadata contributed by one panel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelDescriptor {
    /// Stable, namespaced identity used by layouts and persistence.
    pub id: PanelId,
    /// Human-readable title shown by a host.
    pub title: String,
    /// Optional renderer-independent icon key.
    pub icon: Option<String>,
    /// Optional accessible or tooltip description.
    pub tooltip: Option<String>,
    /// Whether the panel is singleton or may have multiple instances.
    pub instancing: Instancing,
    /// Default destination when a host creates the initial layout.
    pub default_slot: DefaultSlot,
    /// Whether the panel is visible in the initial layout.
    pub default_visible: bool,
    /// Descriptor schema version owned by the contributing extension.
    pub schema_version: u16,
}

impl PanelDescriptor {
    /// Creates a descriptor with standard tool-panel defaults.
    ///
    /// The identifier is validated when the descriptor is registered. Keeping
    /// construction infallible allows callers to assemble data before they
    /// choose a registry, while [`PanelRegistry::register`] remains the single
    /// enforcement point for the registry invariant.
    pub fn new(id: impl Into<PanelId>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            icon: None,
            tooltip: None,
            instancing: Instancing::default(),
            default_slot: DefaultSlot::default(),
            default_visible: true,
            schema_version: 1,
        }
    }

    /// Sets the renderer-independent icon key.
    #[must_use]
    pub fn with_icon(mut self, icon: impl Into<String>) -> Self {
        self.icon = Some(icon.into());
        self
    }

    /// Sets the accessible or tooltip description.
    #[must_use]
    pub fn with_tooltip(mut self, tooltip: impl Into<String>) -> Self {
        self.tooltip = Some(tooltip.into());
        self
    }

    /// Sets the instance policy.
    #[must_use]
    pub const fn with_instancing(mut self, instancing: Instancing) -> Self {
        self.instancing = instancing;
        self
    }

    /// Sets the initial docking destination.
    #[must_use]
    pub const fn with_default_slot(mut self, slot: DefaultSlot) -> Self {
        self.default_slot = slot;
        self
    }

    /// Sets whether the panel starts visible.
    #[must_use]
    pub const fn with_default_visible(mut self, visible: bool) -> Self {
        self.default_visible = visible;
        self
    }

    /// Sets the extension-owned descriptor schema version.
    #[must_use]
    pub const fn with_schema_version(mut self, version: u16) -> Self {
        self.schema_version = version;
        self
    }
}

/// A validation failure while adding a panel contribution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PanelRegistryError {
    /// The descriptor's identifier is empty or consists only of whitespace.
    EmptyPanelId,
    /// The descriptor's identifier is already registered.
    DuplicatePanelId(PanelId),
}

impl fmt::Display for PanelRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPanelId => formatter.write_str("panel ID must not be empty"),
            Self::DuplicatePanelId(id) => {
                formatter.write_str("panel ID is already registered: ")?;
                formatter.write_str(id)
            }
        }
    }
}

/// An ordered, renderer-independent allow-list of panel descriptors.
///
/// Registration order is part of the deterministic contract and is preserved
/// by [`iter`](Self::iter). Queries are intentionally linear: panel registries
/// are expected to be small, and avoiding a hash table means iteration and
/// behavior do not depend on hashing details.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PanelRegistry {
    descriptors: Vec<PanelDescriptor>,
}

impl PanelRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            descriptors: Vec::new(),
        }
    }

    /// Registers a descriptor at the end of the deterministic order.
    ///
    /// Empty (including whitespace-only) IDs and duplicate IDs are rejected;
    /// a failed registration leaves the registry unchanged.
    pub fn register(&mut self, descriptor: PanelDescriptor) -> Result<(), PanelRegistryError> {
        if descriptor.id.trim().is_empty() {
            return Err(PanelRegistryError::EmptyPanelId);
        }

        if self
            .descriptors
            .iter()
            .any(|registered| registered.id == descriptor.id)
        {
            return Err(PanelRegistryError::DuplicatePanelId(descriptor.id));
        }

        self.descriptors.push(descriptor);
        Ok(())
    }

    /// Returns the descriptor for an ID, if it is registered.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&PanelDescriptor> {
        self.descriptors
            .iter()
            .find(|descriptor| descriptor.id == id)
    }

    /// Resolves a [`PanelId`] for persistence allow-list checks.
    #[must_use]
    pub fn resolve(&self, id: &PanelId) -> Option<&PanelDescriptor> {
        self.get(id)
    }

    /// Returns whether an ID is registered.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.get(id).is_some()
    }

    /// Removes a descriptor by ID and returns it for extension shutdown.
    pub fn unregister(&mut self, id: &str) -> Option<PanelDescriptor> {
        let index = self
            .descriptors
            .iter()
            .position(|descriptor| descriptor.id == id)?;
        Some(self.descriptors.remove(index))
    }

    /// Returns descriptors in their registration order.
    pub fn iter(&self) -> core::slice::Iter<'_, PanelDescriptor> {
        self.descriptors.iter()
    }

    /// Returns the number of registered descriptors.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.descriptors.len()
    }

    /// Returns whether no descriptors are registered.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }
}

impl<'a> IntoIterator for &'a PanelRegistry {
    type Item = &'a PanelDescriptor;
    type IntoIter = core::slice::Iter<'a, PanelDescriptor>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use alloc::{string::String, vec, vec::Vec};

    use super::{
        DefaultSlot, Instancing, PanelDescriptor, PanelInstanceId, PanelInstanceRegistry,
        PanelInstanceRegistryError, PanelInstanceState, PanelRegistry, PanelRegistryError,
    };

    fn descriptor(id: &str) -> PanelDescriptor {
        PanelDescriptor::new(id, id)
    }

    #[test]
    fn descriptor_defaults_are_pure_data() {
        let descriptor = descriptor("editor.inspector");

        assert_eq!(descriptor.instancing, Instancing::Singleton);
        assert_eq!(descriptor.default_slot, DefaultSlot::LeftBar);
        assert!(descriptor.default_visible);
        assert!(descriptor.icon.is_none());
        assert!(descriptor.tooltip.is_none());
        assert_eq!(descriptor.schema_version, 1);
    }

    #[test]
    fn registration_preserves_order_and_resolves_by_id() {
        let mut registry = PanelRegistry::new();
        registry
            .register(descriptor("z.last"))
            .expect("first registration should succeed");
        registry
            .register(descriptor("a.first"))
            .expect("second registration should succeed");

        let ids: Vec<_> = registry.iter().map(|panel| panel.id.as_str()).collect();
        assert_eq!(ids, ["z.last", "a.first"]);
        assert_eq!(
            registry.get("a.first").map(|panel| panel.title.as_str()),
            Some("a.first")
        );
        assert_eq!(
            registry
                .resolve(&String::from("z.last"))
                .map(|panel| panel.id.as_str()),
            Some("z.last")
        );
        assert!(registry.contains("z.last"));
        assert!(!registry.contains("missing"));
    }

    #[test]
    fn empty_ids_are_rejected_without_mutation() {
        let mut registry = PanelRegistry::new();

        assert_eq!(
            registry.register(descriptor("")),
            Err(PanelRegistryError::EmptyPanelId)
        );
        assert_eq!(
            registry.register(descriptor(" \t")),
            Err(PanelRegistryError::EmptyPanelId)
        );
        assert!(registry.is_empty());
    }

    #[test]
    fn duplicate_ids_are_rejected_without_mutation() {
        let mut registry = PanelRegistry::new();
        registry
            .register(descriptor("scene.outliner"))
            .expect("initial registration should succeed");

        assert_eq!(
            registry.register(PanelDescriptor::new("scene.outliner", "Other")),
            Err(PanelRegistryError::DuplicatePanelId(String::from(
                "scene.outliner"
            )))
        );
        assert_eq!(registry.len(), 1);
        assert_eq!(
            registry
                .get("scene.outliner")
                .map(|panel| panel.title.as_str()),
            Some("scene.outliner")
        );
    }

    #[test]
    fn unregister_returns_the_descriptor_and_preserves_remaining_order() {
        let mut registry = PanelRegistry::new();
        registry.register(descriptor("first")).unwrap();
        registry.register(descriptor("second")).unwrap();
        assert_eq!(registry.unregister("first").unwrap().id, "first");
        assert_eq!(
            registry
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            ["second"]
        );
        assert!(registry.unregister("missing").is_none());
    }

    #[test]
    fn instance_registry_enforces_singleton_and_restores_identity() {
        let mut panels = PanelRegistry::new();
        panels
            .register(PanelDescriptor::new("scene", "Scene"))
            .unwrap();
        let mut instances = PanelInstanceRegistry::new();
        let first = instances.create(&panels, "scene").unwrap();
        assert_eq!(first.id().get(), 1);
        assert_eq!(first.state(), PanelInstanceState::Open);
        assert_eq!(
            instances.create(&panels, "scene"),
            Err(PanelInstanceRegistryError::SingletonAlreadyExists {
                panel_id: "scene".into(),
                instance_id: first.id(),
            })
        );
        assert_eq!(
            instances.close(first.id()).unwrap().state(),
            PanelInstanceState::Closed
        );
        assert_eq!(instances.open_len(), 0);
        assert_eq!(instances.restore(first.id()).unwrap().id(), first.id());
        assert_eq!(instances.open_len(), 1);
        assert_eq!(
            instances.restore(first.id()),
            Err(PanelInstanceRegistryError::InstanceAlreadyOpen(first.id()))
        );
    }

    #[test]
    fn multi_instances_are_independent_and_unknown_ids_are_rejected() {
        let mut panels = PanelRegistry::new();
        panels
            .register(PanelDescriptor::new("logs", "Logs").with_instancing(Instancing::Multi))
            .unwrap();
        let mut instances = PanelInstanceRegistry::new();
        let first = instances.create(&panels, "logs").unwrap();
        let second = instances.create(&panels, "logs").unwrap();
        assert_ne!(first.id(), second.id());
        assert_eq!(instances.len(), 2);
        assert_eq!(instances.open_len(), 2);
        assert_eq!(
            instances
                .iter_open_for_panel("logs")
                .map(super::PanelInstance::id)
                .collect::<Vec<_>>(),
            vec![first.id(), second.id()]
        );
        let unknown = PanelInstanceId::from_raw(99);
        assert_eq!(
            instances.close(unknown),
            Err(PanelInstanceRegistryError::UnknownInstance(unknown))
        );
        assert_eq!(
            instances.create(&panels, "missing"),
            Err(PanelInstanceRegistryError::UnknownPanel("missing".into()))
        );
    }
}
