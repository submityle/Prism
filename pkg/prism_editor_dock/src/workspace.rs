//! Workspace management, snapshot validation, and deterministic recovery.

use prism_editor_core::PanelDescriptor as EditorPanelDescriptor;
use prism_ui_dock::{
    DockLayout, DockNode, PanelInstanceRegistry, WorkspaceError, WorkspaceManager,
};

use crate::{
    ContributionError, DockContributionRegistry, PanelLifecycleAdapter, PanelLifecycleError,
};

/// An owned workspace/layout snapshot crossing the persistence boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DockWorkspaceSnapshot {
    manager: WorkspaceManager,
    instances: PanelInstanceRegistry,
}

impl DockWorkspaceSnapshot {
    /// Creates a snapshot by cloning the supplied pure-data state.
    #[must_use]
    pub fn new(manager: &WorkspaceManager, instances: &PanelInstanceRegistry) -> Self {
        Self {
            manager: manager.clone(),
            instances: instances.clone(),
        }
    }

    /// Returns the workspace manager copy.
    #[must_use]
    pub fn manager(&self) -> WorkspaceManager {
        self.manager.clone()
    }

    /// Returns the panel-instance registry copy.
    #[must_use]
    pub fn instances(&self) -> PanelInstanceRegistry {
        self.instances.clone()
    }
}

/// The source selected during deterministic workspace recovery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RestoreSource {
    /// The primary persisted snapshot.
    Primary,
    /// The last known-good persisted snapshot.
    LastGood,
    /// The caller-provided built-in/default snapshot.
    Default,
}

/// A rejected recovery candidate and its validation reason.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RestoreRejection {
    /// Candidate source that was rejected.
    pub source: RestoreSource,
    /// Stable human-readable reason.
    pub reason: String,
}

/// Recovery result including diagnostics for candidates skipped before success.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RestoreReport {
    /// Candidate that was installed.
    pub source: RestoreSource,
    /// Earlier candidates rejected by validation.
    pub rejected: Vec<RestoreRejection>,
}

/// Failures from contribution, lifecycle, or workspace operations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DockWorkspaceError {
    /// A contribution could not be mapped.
    Contribution(ContributionError),
    /// A panel lifecycle operation failed.
    Lifecycle(PanelLifecycleError),
    /// A workspace operation failed.
    Workspace(WorkspaceError),
    /// A panel cannot be removed while state still references it.
    PanelInUse(String),
    /// A snapshot contains an unknown panel ID.
    UnknownPanel(String),
    /// A singleton panel appears in more than one retained instance.
    DuplicateSingletonInstance(String),
    /// No primary, last-good, or default snapshot passed validation.
    NoValidRestoreCandidate(Vec<RestoreRejection>),
}

impl std::fmt::Display for DockWorkspaceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Contribution(error) => error.fmt(formatter),
            Self::Lifecycle(error) => error.fmt(formatter),
            Self::Workspace(error) => error.fmt(formatter),
            Self::PanelInUse(id) => write!(
                formatter,
                "panel is still referenced by workspace state: {id}"
            ),
            Self::UnknownPanel(id) => write!(formatter, "snapshot references unknown panel: {id}"),
            Self::DuplicateSingletonInstance(id) => {
                write!(
                    formatter,
                    "snapshot contains multiple instances of singleton panel: {id}"
                )
            }
            Self::NoValidRestoreCandidate(rejected) => {
                write!(
                    formatter,
                    "no workspace restore candidate passed validation ({})",
                    rejected.len()
                )
            }
        }
    }
}

impl std::error::Error for DockWorkspaceError {}

impl From<ContributionError> for DockWorkspaceError {
    fn from(error: ContributionError) -> Self {
        Self::Contribution(error)
    }
}

impl From<PanelLifecycleError> for DockWorkspaceError {
    fn from(error: PanelLifecycleError) -> Self {
        Self::Lifecycle(error)
    }
}

impl From<WorkspaceError> for DockWorkspaceError {
    fn from(error: WorkspaceError) -> Self {
        Self::Workspace(error)
    }
}

/// Editor-layer controller for dock contributions, workspaces, and instances.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DockWorkspaceController {
    panels: DockContributionRegistry,
    manager: WorkspaceManager,
    lifecycle: PanelLifecycleAdapter,
}

impl DockWorkspaceController {
    /// Creates an empty controller.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            panels: DockContributionRegistry::new(),
            manager: WorkspaceManager::new(),
            lifecycle: PanelLifecycleAdapter::new(),
        }
    }

    /// Maps one editor contribution into the dock registry.
    pub fn register_panel(
        &mut self,
        descriptor: &EditorPanelDescriptor,
    ) -> Result<(), DockWorkspaceError> {
        self.panels.register(descriptor)?;
        Ok(())
    }

    /// Removes a panel only when no live workspace state references it.
    pub fn unregister_panel(&mut self, panel_id: &str) -> Result<bool, DockWorkspaceError> {
        if self
            .lifecycle
            .registry()
            .iter()
            .any(|instance| instance.panel_id() == panel_id)
            || manager_references_panel(&self.manager, panel_id)
        {
            return Err(DockWorkspaceError::PanelInUse(panel_id.into()));
        }
        Ok(self.panels.unregister(panel_id).is_some())
    }

    /// Returns the mapped panel registry.
    #[must_use]
    pub const fn panels(&self) -> &DockContributionRegistry {
        &self.panels
    }

    /// Opens a panel instance.
    pub fn create_panel_instance(
        &mut self,
        panel_id: &str,
    ) -> Result<crate::PanelLifecycleEvent, DockWorkspaceError> {
        Ok(self.lifecycle.create(&self.panels, panel_id)?)
    }

    /// Closes a retained panel instance.
    pub fn close_panel_instance(
        &mut self,
        instance_id: prism_ui_dock::PanelInstanceId,
    ) -> Result<crate::PanelLifecycleEvent, DockWorkspaceError> {
        Ok(self.lifecycle.close(instance_id)?)
    }

    /// Restores a retained panel instance.
    pub fn restore_panel_instance(
        &mut self,
        instance_id: prism_ui_dock::PanelInstanceId,
    ) -> Result<crate::PanelLifecycleEvent, DockWorkspaceError> {
        Ok(self.lifecycle.restore(instance_id)?)
    }

    /// Adds a named workspace.
    pub fn add_workspace(&mut self, name: impl Into<String>) -> Result<(), DockWorkspaceError> {
        self.manager.add_workspace(name)?;
        Ok(())
    }

    /// Adds a perspective to the active workspace.
    pub fn add_perspective(
        &mut self,
        name: impl Into<String>,
        layout: &DockLayout,
    ) -> Result<(), DockWorkspaceError> {
        self.validate_layout(layout)?;
        self.manager.add_perspective(name, layout)?;
        Ok(())
    }

    /// Switches the active workspace.
    pub fn switch_workspace(&mut self, name: &str) -> Result<(), DockWorkspaceError> {
        self.manager.switch_workspace(name)?;
        Ok(())
    }

    /// Switches the active perspective and returns its owned layout.
    pub fn switch_perspective(&mut self, name: &str) -> Result<DockLayout, DockWorkspaceError> {
        Ok(self.manager.switch_perspective(name)?)
    }

    /// Saves the active perspective's layout after validating panel references.
    pub fn save_active_layout(&mut self, layout: &DockLayout) -> Result<(), DockWorkspaceError> {
        self.validate_layout(layout)?;
        self.manager.save_active_layout(layout)?;
        Ok(())
    }

    /// Returns an owned active layout, if one exists.
    #[must_use]
    pub fn active_layout(&self) -> Option<DockLayout> {
        self.manager.active_layout()
    }

    /// Returns an owned snapshot suitable for deterministic persistence.
    #[must_use]
    pub fn snapshot(&self) -> DockWorkspaceSnapshot {
        DockWorkspaceSnapshot::new(&self.manager, self.lifecycle.registry())
    }

    /// Restores a validated snapshot atomically.
    pub fn restore_snapshot(
        &mut self,
        snapshot: &DockWorkspaceSnapshot,
    ) -> Result<(), DockWorkspaceError> {
        self.validate_snapshot(snapshot)?;
        self.manager = snapshot.manager();
        self.lifecycle.replace_registry(snapshot.instances());
        Ok(())
    }

    /// Recovers from primary, last-good, and finally default snapshots in that order.
    pub fn recover(
        &mut self,
        primary: Option<&DockWorkspaceSnapshot>,
        last_good: Option<&DockWorkspaceSnapshot>,
        default: &DockWorkspaceSnapshot,
    ) -> Result<RestoreReport, DockWorkspaceError> {
        let candidates = [
            (RestoreSource::Primary, primary),
            (RestoreSource::LastGood, last_good),
            (RestoreSource::Default, Some(default)),
        ];
        let mut rejected = Vec::new();
        for (source, candidate) in candidates {
            let Some(candidate) = candidate else {
                continue;
            };
            match self.validate_snapshot(candidate) {
                Ok(()) => {
                    self.restore_snapshot(candidate)?;
                    return Ok(RestoreReport { source, rejected });
                }
                Err(error) => rejected.push(RestoreRejection {
                    source,
                    reason: error.to_string(),
                }),
            }
        }
        Err(DockWorkspaceError::NoValidRestoreCandidate(rejected))
    }

    fn validate_snapshot(
        &self,
        snapshot: &DockWorkspaceSnapshot,
    ) -> Result<(), DockWorkspaceError> {
        for instance in snapshot.instances.iter() {
            let Some(descriptor) = self.panels.get(instance.panel_id()) else {
                return Err(DockWorkspaceError::UnknownPanel(
                    instance.panel_id().clone(),
                ));
            };
            if descriptor.editor().singleton
                && snapshot
                    .instances
                    .iter()
                    .filter(|other| other.panel_id() == instance.panel_id())
                    .count()
                    > 1
            {
                return Err(DockWorkspaceError::DuplicateSingletonInstance(
                    instance.panel_id().clone(),
                ));
            }
        }
        for workspace in snapshot.manager.workspaces() {
            for perspective in workspace.perspectives() {
                self.validate_layout(&perspective.layout())?;
            }
        }
        Ok(())
    }

    fn validate_layout(&self, layout: &DockLayout) -> Result<(), DockWorkspaceError> {
        for id in layout_panel_ids(layout) {
            if !self.panels.dock_registry().contains(&id) {
                return Err(DockWorkspaceError::UnknownPanel(id));
            }
        }
        Ok(())
    }
}

fn manager_references_panel(manager: &WorkspaceManager, panel_id: &str) -> bool {
    manager.workspaces().any(|workspace| {
        workspace
            .perspectives()
            .any(|perspective| layout_panel_ids(&perspective.layout()).any(|id| id == panel_id))
    })
}

fn layout_panel_ids(layout: &DockLayout) -> impl Iterator<Item = String> {
    let mut ids = Vec::new();
    collect_node_panel_ids(&layout.root, &mut ids);
    for window in &layout.floating {
        collect_node_panel_ids(&window.root, &mut ids);
    }
    ids.extend(
        layout
            .autohide
            .entries
            .iter()
            .map(|entry| entry.panel.clone()),
    );
    if let Some(maximized) = &layout.maximized {
        ids.push(maximized.panel.clone());
        collect_node_panel_ids(&maximized.root, &mut ids);
        for window in &maximized.floating {
            collect_node_panel_ids(&window.root, &mut ids);
        }
        ids.extend(
            maximized
                .autohide
                .entries
                .iter()
                .map(|entry| entry.panel.clone()),
        );
    }
    ids.into_iter()
}

fn collect_node_panel_ids(node: &DockNode, output: &mut Vec<String>) {
    match node {
        DockNode::Split { children, .. } => {
            for child in children {
                collect_node_panel_ids(&child.node, output);
            }
        }
        DockNode::TabStack { tabs, .. } => output.extend(tabs.iter().cloned()),
        DockNode::Document { .. } | DockNode::Empty => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_editor_core::PanelDescriptor;

    fn descriptor(id: &str, singleton: bool) -> PanelDescriptor {
        PanelDescriptor {
            id: id.into(),
            title: id.into(),
            singleton,
        }
    }

    fn controller() -> DockWorkspaceController {
        let mut controller = DockWorkspaceController::new();
        controller
            .register_panel(&descriptor("scene", true))
            .unwrap();
        controller
            .register_panel(&descriptor("logs", false))
            .unwrap();
        controller
    }

    #[test]
    fn workspace_and_layout_state_roundtrip_through_snapshot() {
        let mut source = controller();
        source.add_workspace("main").unwrap();
        source
            .add_perspective(
                "editing",
                &DockLayout {
                    root: DockNode::panel("scene"),
                    ..DockLayout::default()
                },
            )
            .unwrap();
        let snapshot = source.snapshot();

        let mut restored = controller();
        restored.restore_snapshot(&snapshot).unwrap();
        assert_eq!(restored.active_layout(), source.active_layout());
    }

    #[test]
    fn recovery_skips_incompatible_primary_and_uses_last_good() {
        let mut controller = controller();
        let invalid = DockWorkspaceSnapshot::new(
            &{
                let mut manager = WorkspaceManager::new();
                manager.add_workspace("bad").unwrap();
                manager
                    .add_perspective(
                        "editing",
                        &DockLayout {
                            root: DockNode::panel("missing"),
                            ..DockLayout::default()
                        },
                    )
                    .unwrap();
                manager
            },
            &PanelInstanceRegistry::new(),
        );
        let mut valid_manager = WorkspaceManager::new();
        valid_manager.add_workspace("good").unwrap();
        valid_manager
            .add_perspective(
                "editing",
                &DockLayout {
                    root: DockNode::panel("scene"),
                    ..DockLayout::default()
                },
            )
            .unwrap();
        let good = DockWorkspaceSnapshot::new(&valid_manager, &PanelInstanceRegistry::new());
        let report = controller
            .recover(Some(&invalid), Some(&good), &good)
            .unwrap();
        assert_eq!(report.source, RestoreSource::LastGood);
        assert_eq!(report.rejected.len(), 1);
    }

    #[test]
    fn unregister_rejects_referenced_panel() {
        let mut controller = controller();
        controller.add_workspace("main").unwrap();
        controller
            .add_perspective(
                "editing",
                &DockLayout {
                    root: DockNode::panel("scene"),
                    ..DockLayout::default()
                },
            )
            .unwrap();
        assert!(matches!(
            controller.unregister_panel("scene"),
            Err(DockWorkspaceError::PanelInUse(_))
        ));
    }
}
