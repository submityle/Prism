//! Registry-backed default layout reset for workspace hosts.
//!
//! The workspace model stores snapshots only. This module is the small
//! integration boundary that materializes a [`PanelRegistry`] policy and
//! commits the resulting snapshot atomically through the existing workspace
//! APIs.

use crate::{
    DefaultLayoutError, DefaultLayoutPolicy, DockLayout, PanelRegistry, WorkspaceError,
    WorkspaceManager,
};

/// Failure while materializing or committing a registry-backed default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DefaultLayoutResetError {
    /// The registry policy could not produce a legal layout.
    Layout(DefaultLayoutError),
    /// The requested workspace state was not available.
    Workspace(WorkspaceError),
}

impl From<DefaultLayoutError> for DefaultLayoutResetError {
    fn from(error: DefaultLayoutError) -> Self {
        Self::Layout(error)
    }
}

impl From<WorkspaceError> for DefaultLayoutResetError {
    fn from(error: WorkspaceError) -> Self {
        Self::Workspace(error)
    }
}

impl WorkspaceManager {
    /// Replaces the active perspective with the registry-derived default.
    ///
    /// The layout is materialized before the workspace snapshot is changed;
    /// therefore an invalid policy or an unavailable active perspective never
    /// leaves a partially reset workspace behind.
    pub fn reset_active_perspective_to_default(
        &mut self,
        registry: &PanelRegistry,
        policy: DefaultLayoutPolicy,
    ) -> Result<DockLayout, DefaultLayoutResetError> {
        let layout = registry.default_layout(policy)?;
        self.save_active_layout(&layout)?;
        Ok(layout)
    }

    /// Replaces a named perspective in the active workspace with the
    /// registry-derived default.
    pub fn reset_perspective_to_default(
        &mut self,
        name: &str,
        registry: &PanelRegistry,
        policy: DefaultLayoutPolicy,
    ) -> Result<DockLayout, DefaultLayoutResetError> {
        let layout = registry.default_layout(policy)?;
        self.save_layout(name, &layout)?;
        Ok(layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DefaultSlot, DockNode, PanelDescriptor, Ratio};

    fn registry() -> PanelRegistry {
        let mut registry = PanelRegistry::new();
        registry
            .register(PanelDescriptor::new("outliner", "Outliner"))
            .unwrap();
        registry
            .register(
                PanelDescriptor::new("inspector", "Inspector")
                    .with_default_slot(DefaultSlot::RightBar),
            )
            .unwrap();
        registry
    }

    #[test]
    fn active_reset_commits_the_registry_default_and_returns_an_owned_copy() {
        let mut manager = WorkspaceManager::new();
        manager.add_workspace("coding").unwrap();
        manager
            .add_perspective("default", &DockLayout::default())
            .unwrap();

        let layout = manager
            .reset_active_perspective_to_default(&registry(), DefaultLayoutPolicy::default())
            .unwrap();
        assert!(layout.root.contains_panel("outliner"));
        assert!(layout.root.contains_panel("inspector"));
        assert_eq!(manager.active_layout(), Some(layout.clone()));

        let mut returned = layout;
        returned.root = DockNode::Empty;
        assert_ne!(manager.active_layout(), Some(returned));
    }

    #[test]
    fn named_reset_only_changes_the_requested_perspective() {
        let mut manager = WorkspaceManager::new();
        manager.add_workspace("coding").unwrap();
        manager
            .add_perspective("default", &DockLayout::default())
            .unwrap();
        manager
            .add_perspective(
                "debug",
                &DockLayout {
                    root: DockNode::panel("debug.custom"),
                    ..DockLayout::default()
                },
            )
            .unwrap();

        let expected = registry()
            .default_layout(DefaultLayoutPolicy::default())
            .unwrap();
        manager
            .reset_perspective_to_default("debug", &registry(), DefaultLayoutPolicy::default())
            .unwrap();
        assert_eq!(
            manager
                .workspace("coding")
                .unwrap()
                .perspective("debug")
                .unwrap()
                .layout(),
            expected
        );
        assert_eq!(
            manager
                .workspace("coding")
                .unwrap()
                .perspective("default")
                .unwrap()
                .layout(),
            DockLayout::default()
        );
    }

    #[test]
    fn reset_errors_do_not_mutate_existing_layouts() {
        let mut manager = WorkspaceManager::new();
        manager.add_workspace("coding").unwrap();
        let original = DockLayout {
            root: DockNode::panel("custom"),
            ..DockLayout::default()
        };
        manager.add_perspective("default", &original).unwrap();

        let error = manager
            .reset_active_perspective_to_default(
                &registry(),
                DefaultLayoutPolicy {
                    left_ratio: Ratio::from_millionths(Ratio::SCALE + 1),
                    ..DefaultLayoutPolicy::default()
                },
            )
            .unwrap_err();
        assert!(matches!(error, DefaultLayoutResetError::Layout(_)));
        assert_eq!(manager.active_layout(), Some(original));
    }

    #[test]
    fn reset_requires_an_active_perspective() {
        let mut manager = WorkspaceManager::new();
        manager.add_workspace("coding").unwrap();
        let error = manager
            .reset_active_perspective_to_default(&registry(), DefaultLayoutPolicy::default())
            .unwrap_err();
        assert!(matches!(
            error,
            DefaultLayoutResetError::Workspace(WorkspaceError::NoActivePerspective { .. })
        ));
    }
}
