use alloc::vec::Vec;

use crate::{DockLayout, DockNode, NameKind, WorkspaceError, WorkspaceManager};

fn layout(panel: &str) -> DockLayout {
    DockLayout {
        root: DockNode::panel(panel),
        ..DockLayout::default()
    }
}

#[test]
fn workspaces_and_perspectives_preserve_insertion_order() {
    let mut manager = WorkspaceManager::new();
    manager.add_workspace("coding").unwrap();
    manager.add_workspace("debugging").unwrap();
    manager.switch_workspace("coding").unwrap();
    manager
        .add_perspective("base", &layout("scene.outliner"))
        .unwrap();
    manager
        .add_perspective("focused", &layout("inspector"))
        .unwrap();

    let workspace_names: Vec<_> = manager.workspaces().map(crate::Workspace::name).collect();
    let perspective_names: Vec<_> = manager
        .workspace("coding")
        .unwrap()
        .perspectives()
        .map(crate::Perspective::name)
        .collect();

    assert_eq!(workspace_names, ["coding", "debugging"]);
    assert_eq!(perspective_names, ["base", "focused"]);
    assert_eq!(manager.active_workspace_name(), Some("coding"));
    assert_eq!(
        manager
            .active_workspace()
            .unwrap()
            .active_perspective_name(),
        Some("base")
    );
}

#[test]
fn empty_and_duplicate_names_are_structured_and_non_mutating() {
    let mut manager = WorkspaceManager::new();
    assert_eq!(
        manager.add_workspace(" \t"),
        Err(WorkspaceError::EmptyName {
            kind: NameKind::Workspace
        })
    );
    manager.add_workspace("coding").unwrap();
    assert_eq!(
        manager.add_workspace("coding"),
        Err(WorkspaceError::DuplicateWorkspace {
            name: String::from("coding")
        })
    );
    assert_eq!(manager.workspace_count(), 1);

    assert_eq!(
        manager.add_perspective("", &DockLayout::default()),
        Err(WorkspaceError::EmptyName {
            kind: NameKind::Perspective
        })
    );
    manager
        .add_perspective("base", &DockLayout::default())
        .unwrap();
    assert_eq!(
        manager.add_perspective("base", &DockLayout::default()),
        Err(WorkspaceError::DuplicatePerspective {
            workspace: String::from("coding"),
            perspective: String::from("base")
        })
    );
    assert_eq!(manager.active_workspace().unwrap().perspective_count(), 1);
}

#[test]
fn saving_and_switching_are_deep_copy_boundaries() {
    let mut manager = WorkspaceManager::new();
    manager.add_workspace("coding").unwrap();

    let mut original = layout("before");
    manager.add_perspective("base", &original).unwrap();
    original.root = DockNode::panel("after");
    assert_eq!(
        manager.active_layout().unwrap().root,
        DockNode::panel("before")
    );

    let mut switched = manager.switch_perspective("base").unwrap();
    switched.root = DockNode::panel("mutated-copy");
    assert_eq!(
        manager.active_layout().unwrap().root,
        DockNode::panel("before")
    );

    manager.save_active_layout(&switched).unwrap();
    switched.root = DockNode::panel("mutated-again");
    assert_eq!(
        manager.active_layout().unwrap().root,
        DockNode::panel("mutated-copy")
    );
}

#[test]
fn switching_and_saving_report_missing_active_state() {
    let mut manager = WorkspaceManager::new();
    assert_eq!(
        manager.add_perspective("base", &DockLayout::default()),
        Err(WorkspaceError::NoActiveWorkspace)
    );
    assert_eq!(
        manager.switch_perspective("base"),
        Err(WorkspaceError::NoActiveWorkspace)
    );
    manager.add_workspace("empty").unwrap();
    assert_eq!(
        manager.save_active_layout(&DockLayout::default()),
        Err(WorkspaceError::NoActivePerspective {
            workspace: String::from("empty")
        })
    );
}
