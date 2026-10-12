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
    use alloc::vec::Vec;

    use super::{DefaultSlot, Instancing, PanelDescriptor, PanelRegistry, PanelRegistryError};

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
}
