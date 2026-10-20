//! Editor contribution to dock-registry mapping.
//!
//! The editor kernel's panel descriptor is intentionally small and UI-neutral.
//! The dock descriptor adds only host-facing metadata, so this module maps
//! the stable identity, title, and singleton policy while keeping registration
//! order deterministic.

use std::collections::BTreeMap;

use prism_editor_core::PanelDescriptor as EditorPanelDescriptor;
use prism_ui_dock::{Instancing, PanelDescriptor as DockPanelDescriptor, PanelRegistry};

/// A failure while mapping an editor contribution into the dock registry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContributionError {
    /// The editor contribution has an empty or whitespace-only identifier.
    EmptyPanelId,
    /// A panel with the same stable identifier is already mapped.
    DuplicatePanel(String),
}

impl std::fmt::Display for ContributionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPanelId => formatter.write_str("editor panel ID must not be empty"),
            Self::DuplicatePanel(id) => write!(formatter, "editor panel is already mapped: {id}"),
        }
    }
}

impl std::error::Error for ContributionError {}

/// The source descriptor and its mapped dock descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DockPanelContribution {
    editor: EditorPanelDescriptor,
    dock: DockPanelDescriptor,
}

impl DockPanelContribution {
    /// Returns the editor-owned descriptor.
    #[must_use]
    pub const fn editor(&self) -> &EditorPanelDescriptor {
        &self.editor
    }

    /// Returns the host-facing dock descriptor.
    #[must_use]
    pub const fn dock(&self) -> &DockPanelDescriptor {
        &self.dock
    }
}

/// Deterministic mapping from editor panel contributions to a dock registry.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DockContributionRegistry {
    dock: PanelRegistry,
    contributions: BTreeMap<String, DockPanelContribution>,
}

impl DockContributionRegistry {
    /// Creates an empty mapping.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            dock: PanelRegistry::new(),
            contributions: BTreeMap::new(),
        }
    }

    /// Maps and registers one editor panel contribution.
    pub fn register(
        &mut self,
        descriptor: &EditorPanelDescriptor,
    ) -> Result<&DockPanelContribution, ContributionError> {
        if descriptor.id.trim().is_empty() {
            return Err(ContributionError::EmptyPanelId);
        }
        if self.contributions.contains_key(&descriptor.id) {
            return Err(ContributionError::DuplicatePanel(descriptor.id.clone()));
        }

        let dock = DockPanelDescriptor::new(&descriptor.id, &descriptor.title).with_instancing(
            if descriptor.singleton {
                Instancing::Singleton
            } else {
                Instancing::Multi
            },
        );
        self.dock
            .register(dock.clone())
            .map_err(|_| ContributionError::DuplicatePanel(descriptor.id.clone()))?;
        let id = descriptor.id.clone();
        self.contributions.insert(
            id.clone(),
            DockPanelContribution {
                editor: descriptor.clone(),
                dock,
            },
        );
        Ok(self
            .contributions
            .get(&id)
            .expect("inserted contribution must be present"))
    }

    /// Maps a batch transactionally, preserving the input order in the dock registry.
    pub fn register_all<'a, I>(&mut self, descriptors: I) -> Result<(), ContributionError>
    where
        I: IntoIterator<Item = &'a EditorPanelDescriptor>,
    {
        let mut candidate = self.clone();
        for descriptor in descriptors {
            candidate.register(descriptor)?;
        }
        *self = candidate;
        Ok(())
    }

    /// Removes a mapped contribution and returns its descriptors.
    pub fn unregister(&mut self, id: &str) -> Option<DockPanelContribution> {
        let contribution = self.contributions.remove(id)?;
        let removed = self.dock.unregister(id);
        debug_assert!(removed.is_some());
        Some(contribution)
    }

    /// Returns the underlying renderer-independent dock registry.
    #[must_use]
    pub const fn dock_registry(&self) -> &PanelRegistry {
        &self.dock
    }

    /// Returns one mapped contribution by stable editor ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&DockPanelContribution> {
        self.contributions.get(id)
    }

    /// Returns contributions in stable ID order.
    pub fn iter(&self) -> impl Iterator<Item = &DockPanelContribution> {
        self.contributions.values()
    }

    /// Returns the number of mapped panels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.contributions.len()
    }

    /// Returns whether no panels are mapped.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.contributions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(id: &str, singleton: bool) -> EditorPanelDescriptor {
        EditorPanelDescriptor {
            id: id.into(),
            title: id.into(),
            singleton,
        }
    }

    #[test]
    fn maps_core_identity_and_instancing_policy() {
        let mut registry = DockContributionRegistry::new();
        registry
            .register(&descriptor("scene.outliner", true))
            .unwrap();
        registry.register(&descriptor("logs", false)).unwrap();

        assert_eq!(registry.dock_registry().len(), 2);
        assert_eq!(
            registry.get("scene.outliner").unwrap().dock().instancing,
            Instancing::Singleton
        );
        assert_eq!(
            registry.get("logs").unwrap().dock().instancing,
            Instancing::Multi
        );
    }

    #[test]
    fn batch_mapping_is_transactional_and_deterministic() {
        let mut registry = DockContributionRegistry::new();
        registry.register(&descriptor("existing", true)).unwrap();
        let input = [descriptor("a", true), descriptor("existing", false)];

        assert!(registry.register_all(input.iter()).is_err());
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.get("a"), None);

        registry
            .register_all([descriptor("b", true), descriptor("a", true)].iter())
            .unwrap();
        let ids: Vec<_> = registry
            .iter()
            .map(|entry| entry.dock().id.as_str())
            .collect();
        assert_eq!(ids, ["a", "b", "existing"]);
        assert_eq!(
            registry
                .dock_registry()
                .iter()
                .map(|entry| entry.id.as_str())
                .collect::<Vec<_>>(),
            ["existing", "b", "a"]
        );
    }
}
