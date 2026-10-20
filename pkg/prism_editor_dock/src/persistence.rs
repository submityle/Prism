//! Byte-only persistence boundary for dock workspace state.
//!
//! This module deliberately does not read or write paths. It keeps workspace
//! layout bytes and panel-instance bytes separate so an outer file store can
//! commit, recover, or migrate them independently.

use prism_ui_dock::{
    decode_instance_registry, decode_workspace, encode_instance_registry, encode_workspace,
    DockLayout, DockNode, PanelRegistry, WorkspaceDecodeLimits, WorkspaceDecodeOptions,
    WorkspaceManager, WorkspacePersistError,
};

use crate::DockWorkspaceSnapshot;

/// Encoded workspace and instance streams kept as separate persistence units.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedDockSnapshot {
    workspace: Vec<u8>,
    instances: Vec<u8>,
}

impl EncodedDockSnapshot {
    /// Returns the workspace/layout stream.
    #[must_use]
    pub fn workspace_bytes(&self) -> &[u8] {
        &self.workspace
    }

    /// Returns the panel-instance stream.
    #[must_use]
    pub fn instance_bytes(&self) -> &[u8] {
        &self.instances
    }

    /// Constructs an encoded pair from owned byte streams.
    #[must_use]
    pub fn from_parts(workspace: Vec<u8>, instances: Vec<u8>) -> Self {
        Self {
            workspace,
            instances,
        }
    }
}

/// Failure at the byte-only persistence boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DockPersistenceError {
    /// The workspace stream failed validation or decoding.
    Workspace(WorkspacePersistError),
    /// The panel-instance stream failed validation or decoding.
    Instances(prism_ui_dock::InstancePersistError),
    /// The workspace stream references a panel absent from the current registry.
    UnknownPanel(String),
}

impl std::fmt::Display for DockPersistenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Workspace(error) => write!(formatter, "workspace persistence error: {error}"),
            Self::Instances(error) => write!(formatter, "instance persistence error: {error}"),
            Self::UnknownPanel(id) => write!(formatter, "workspace references unknown panel: {id}"),
        }
    }
}

impl std::error::Error for DockPersistenceError {}

/// Encodes a snapshot using the dock crate's deterministic versioned codecs.
pub fn encode_snapshot(
    snapshot: &DockWorkspaceSnapshot,
) -> Result<EncodedDockSnapshot, DockPersistenceError> {
    Ok(EncodedDockSnapshot {
        workspace: encode_workspace(&snapshot.manager())
            .map_err(DockPersistenceError::Workspace)?,
        instances: encode_instance_registry(&snapshot.instances())
            .map_err(DockPersistenceError::Instances)?,
    })
}

/// Decodes a byte pair with the current mapped panel registry as its allow-list.
pub fn decode_snapshot(
    encoded: &EncodedDockSnapshot,
    panels: &PanelRegistry,
) -> Result<DockWorkspaceSnapshot, DockPersistenceError> {
    let manager = decode_workspace(
        &encoded.workspace,
        WorkspaceDecodeOptions {
            known_panels: None,
            known_documents: None,
            limits: WorkspaceDecodeLimits::default(),
        },
    )
    .map_err(DockPersistenceError::Workspace)?;
    validate_manager_panels(&manager, panels)?;
    let instances =
        decode_instance_registry(&encoded.instances).map_err(DockPersistenceError::Instances)?;
    Ok(DockWorkspaceSnapshot::new(&manager, &instances))
}

fn validate_manager_panels(
    manager: &WorkspaceManager,
    panels: &PanelRegistry,
) -> Result<(), DockPersistenceError> {
    for workspace in manager.workspaces() {
        for perspective in workspace.perspectives() {
            for id in layout_panel_ids(&perspective.layout()) {
                if !panels.contains(&id) {
                    return Err(DockPersistenceError::UnknownPanel(id));
                }
            }
        }
    }
    Ok(())
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
    use prism_ui_dock::{DockLayout, DockNode, PanelInstanceRegistry, WorkspaceManager};

    #[test]
    fn snapshot_uses_two_deterministic_streams() {
        let mut panels = crate::DockContributionRegistry::new();
        panels
            .register(&PanelDescriptor {
                id: "scene".into(),
                title: "Scene".into(),
                singleton: true,
            })
            .unwrap();
        let mut manager = WorkspaceManager::new();
        manager.add_workspace("main").unwrap();
        manager
            .add_perspective(
                "editing",
                &DockLayout {
                    root: DockNode::panel("scene"),
                    ..DockLayout::default()
                },
            )
            .unwrap();
        let snapshot = DockWorkspaceSnapshot::new(&manager, &PanelInstanceRegistry::new());
        let encoded = encode_snapshot(&snapshot).unwrap();
        assert_eq!(encoded.workspace_bytes().get(0..4), Some(&b"PRWS"[..]));
        assert_eq!(encoded.instance_bytes().get(0..4), Some(&b"PRIN"[..]));
        assert_eq!(
            decode_snapshot(&encoded, panels.dock_registry()).unwrap(),
            snapshot
        );
        assert_eq!(encode_snapshot(&snapshot).unwrap(), encoded);
    }

    #[test]
    fn unknown_panel_is_rejected_by_workspace_codec() {
        let mut manager = WorkspaceManager::new();
        manager.add_workspace("main").unwrap();
        manager
            .add_perspective(
                "editing",
                &DockLayout {
                    root: DockNode::panel("missing"),
                    ..DockLayout::default()
                },
            )
            .unwrap();
        let snapshot = DockWorkspaceSnapshot::new(&manager, &PanelInstanceRegistry::new());
        let encoded = encode_snapshot(&snapshot).unwrap();
        assert!(matches!(
            decode_snapshot(&encoded, &PanelRegistry::new()),
            Err(DockPersistenceError::UnknownPanel(id)) if id == "missing"
        ));
    }
}
