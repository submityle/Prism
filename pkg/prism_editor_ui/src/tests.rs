use super::*;
use prism_editor_dock::DockContributionRegistry;
use prism_editor_panels::{Console, LogLevel, Outliner, OutlinerId, OutlinerNode};
use prism_editor_reflect::{InspectorProjection, InspectorProjectionNode};
use prism_editor_shell::{
    CommandEntry, CommandIndex, ContextExpr, MenuContribution, ShellContributions,
    StatusContribution, StatusRegion, ToolbarContribution,
};
use prism_reflect::{InspectorHints, InspectorKind, InspectorNode};
use prism_ui::{Element, ElementKind};

fn shell_data() -> ShellChromeProps<'static> {
    ShellChromeProps {
        title: "Prism Studio",
        workspace: "Level",
        menus: &[],
        toolbar: &[],
        status: &[],
    }
}

#[test]
fn shell_has_stable_chrome_order_and_slots() {
    let props = shell_data();
    let tree = shell_chrome(&props);
    assert_eq!(
        tree.kind(),
        &ElementKind::Custom("prism.editor.shell".into())
    );
    assert_eq!(
        tree.child_elements()
            .iter()
            .map(|element| element.class_names()[0].as_str())
            .collect::<Vec<_>>(),
        [
            "prism-editor-shell__title-bar",
            "prism-editor-shell__menu-bar",
            "prism-editor-shell__toolbar",
            "prism-editor-shell__status-bar",
        ]
    );
    assert_eq!(tree.child_elements()[3].child_elements().len(), 3);
}

#[test]
fn shell_contributions_keep_semantic_keys_and_order() {
    let menu = MenuContribution {
        id: "file.open".into(),
        menu_id: "file".into(),
        parent_id: None,
        label: "Open".into(),
        command_id: Some("file.open".into()),
        group: "file".into(),
        order: 0,
        when: ContextExpr::Any,
    };
    let toolbar = ToolbarContribution {
        id: "save".into(),
        toolbar_id: "main".into(),
        label: "Save".into(),
        command_id: "save".into(),
        group: "file".into(),
        order: 0,
        when: ContextExpr::Any,
    };
    let status = StatusContribution {
        id: "mode".into(),
        region: StatusRegion::Right,
        text: "Ready".into(),
        priority: 0,
        when: ContextExpr::Any,
    };
    let menus = [menu];
    let toolbar_items = [toolbar];
    let status_items = [status];
    let tree = shell_chrome(&ShellChromeProps {
        title: "Prism",
        workspace: "Main",
        menus: &menus,
        toolbar: &toolbar_items,
        status: &status_items,
    });
    assert!(tree.child_elements()[1].child_elements()[0]
        .explicit_key()
        .is_some());
    assert!(tree.child_elements()[2].child_elements()[0]
        .explicit_key()
        .is_some());
    assert_eq!(
        tree.child_elements()[3].child_elements()[2].child_elements()[0].text_content(),
        Some("Ready")
    );
}

#[test]
fn command_palette_filters_and_marks_disabled_selection() {
    let mut index = CommandIndex::default();
    index
        .register(CommandEntry::new("scene.save", "Save Scene").with_category("Scene"))
        .expect("command");
    index
        .register(CommandEntry::new("scene.close", "Close Scene").enabled(false))
        .expect("command");
    let tree = command_palette(&CommandPaletteProps {
        index: &index,
        query: "save",
        limit: 8,
        selected_id: Some("scene.save"),
    });
    let rows = tree.child_elements()[1].child_elements();
    assert_eq!(rows.len(), 1);
    assert!(rows[0].explicit_key().is_some());
    assert!(rows[0]
        .class_names()
        .iter()
        .any(|class| class == "is-selected"));
}

#[test]
fn command_palette_has_explicit_empty_state() {
    let index = CommandIndex::default();
    let tree = command_palette(&CommandPaletteProps {
        index: &index,
        query: "",
        limit: 8,
        selected_id: None,
    });
    let list = &tree.child_elements()[1];
    assert_eq!(list.child_elements().len(), 1);
    assert_eq!(
        list.child_elements()[0].class_names()[0],
        "prism-editor-command-palette__empty"
    );
}

#[test]
fn outliner_applies_filter_and_keeps_bounded_rows() {
    let mut model = Outliner::default();
    model
        .insert(OutlinerNode::new(OutlinerId::from_raw(1), None, "Root"))
        .expect("root");
    for raw in 2..=20 {
        model
            .insert(OutlinerNode::new(
                OutlinerId::from_raw(raw),
                Some(OutlinerId::from_raw(1)),
                format!("Node {raw}"),
            ))
            .expect("child");
    }
    let tree = outliner_panel(&OutlinerViewProps {
        outliner: &model,
        filter: "node 2",
        start: 0,
        count: 2,
        selected: Some(OutlinerId::from_raw(11)),
    });
    let rows = &tree.child_elements()[1].child_elements();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].child_elements()[1].text_content(), Some("Node 2"));
}

#[test]
fn outliner_large_count_is_capped_and_zero_window_is_empty() {
    let mut model = Outliner::default();
    for raw in 0..(MAX_VISIBLE_ROWS as u128 + 20) {
        model
            .insert(OutlinerNode::new(
                OutlinerId::from_raw(raw),
                None,
                raw.to_string(),
            ))
            .expect("node");
    }
    let tree = outliner_panel(&OutlinerViewProps {
        outliner: &model,
        filter: "",
        start: 0,
        count: usize::MAX,
        selected: None,
    });
    assert_eq!(
        tree.child_elements()[1].child_elements().len(),
        MAX_VISIBLE_ROWS
    );
    let empty = outliner_panel(&OutlinerViewProps {
        outliner: &model,
        filter: "",
        start: 0,
        count: 0,
        selected: None,
    });
    assert_eq!(empty.child_elements()[1].child_elements().len(), 1);
}

#[test]
fn console_filters_in_retained_order_and_keys_sequence() {
    let mut model = Console::new(8).expect("console");
    model.push(LogLevel::Info, "editor", "ready");
    model.push(LogLevel::Error, "runtime", "failed");
    model.push(LogLevel::Warn, "editor", "watch");
    let tree = console_panel(&ConsolePanelProps {
        console: &model,
        minimum: LogLevel::Warn,
        channel: Some("editor"),
        query: "wat",
        start: 0,
        count: 10,
    });
    let rows = &tree.child_elements()[1].child_elements();
    assert_eq!(rows.len(), 1);
    assert!(rows[0].explicit_key().is_some());
    assert_eq!(rows[0].child_elements()[2].text_content(), Some("watch"));
}

#[test]
fn inspector_renders_paths_values_and_hints() {
    let field_node = InspectorNode {
        label: "speed".into(),
        type_name: "f32".into(),
        kind: InspectorKind::Value,
        variant: None,
        value: Some("1.5".into()),
        hints: InspectorHints {
            docs: Some("Movement speed".into()),
            category: Some("Motion".into()),
            readonly: true,
            range: Some((0.0, 10.0)),
        },
        children: Vec::new(),
    };
    let node = InspectorProjectionNode {
        path: Some(prism_editor_reflect::FieldPath::parse(".speed").expect("path")),
        node: field_node.clone(),
        children: Vec::new(),
    };
    let projection = InspectorProjection {
        root: InspectorProjectionNode {
            path: Some(prism_editor_reflect::FieldPath::root()),
            node: InspectorNode {
                label: "root".into(),
                type_name: "Config".into(),
                kind: InspectorKind::Struct,
                variant: None,
                value: None,
                hints: InspectorHints::default(),
                children: vec![field_node],
            },
            children: vec![node],
        },
    };
    let tree = inspector_view(&InspectorViewProps {
        projection: &projection,
        show_root: false,
    });
    let field = &tree.child_elements()[1].child_elements()[0];
    assert!(field
        .class_names()
        .iter()
        .any(|class| class == "is-readonly"));
    assert!(field.explicit_key().is_some());
    assert!(field
        .child_elements()
        .iter()
        .any(|child| child.text_content() == Some("1.5")));
}

#[test]
fn inspector_empty_projection_is_not_silent() {
    let projection = InspectorProjection {
        root: InspectorProjectionNode {
            path: None,
            node: InspectorNode {
                label: "root".into(),
                type_name: "Unit".into(),
                kind: InspectorKind::Value,
                variant: None,
                value: None,
                hints: InspectorHints::default(),
                children: Vec::new(),
            },
            children: Vec::new(),
        },
    };
    let tree = inspector_view(&InspectorViewProps {
        projection: &projection,
        show_root: false,
    });
    assert_eq!(tree.child_elements()[1].child_elements().len(), 1);
}

#[test]
fn viewport_host_has_surface_and_footer_or_real_empty_state() {
    let mut registry = prism_editor_viewport::ViewportRegistry::new();
    let viewport = registry.open("Scene").expect("viewport");
    let tree = viewport_host(&ViewportHostProps {
        viewport: Some(&viewport),
        overlays: vec![Element::custom("gizmo").key_str("gizmo")],
    });
    assert_eq!(tree.child_elements().len(), 3);
    assert_eq!(
        tree.child_elements()[1].kind(),
        &ElementKind::Custom("prism.editor.viewport-surface".into())
    );
    let empty = viewport_host(&ViewportHostProps {
        viewport: None,
        overlays: Vec::new(),
    });
    assert_eq!(empty.child_elements().len(), 1);
}

#[test]
fn dock_and_workspace_preserve_fixed_region_order() {
    let mut panels = DockContributionRegistry::new();
    panels
        .register(&prism_editor_core::PanelDescriptor {
            id: "outliner".into(),
            title: "Outliner".into(),
            singleton: true,
        })
        .expect("panel");
    let dock = dock_workspace(&DockWorkspaceProps {
        panels: &panels,
        active_panel: Some("outliner"),
        bodies: vec![Element::box_().class("body")],
    });
    assert_eq!(dock.child_elements().len(), 2);
    assert!(dock.child_elements()[0].child_elements()[0]
        .class_names()
        .iter()
        .any(|class| class == "is-active"));

    let workspace = editor_workspace(&EditorWorkspaceProps {
        chrome: Element::custom("chrome"),
        dock: dock.clone(),
        outliner: Element::custom("outliner"),
        viewport: Element::custom("viewport"),
        inspector: Element::custom("inspector"),
        console: Element::custom("console"),
    });
    assert_eq!(workspace.child_elements().len(), 3);
    assert_eq!(workspace.child_elements()[1].child_elements().len(), 3);
}

#[test]
fn shell_contribution_registry_resolves_before_element_assembly() {
    let mut contributions = ShellContributions::default();
    contributions
        .register_menu(MenuContribution {
            id: "file.open".into(),
            menu_id: "file".into(),
            parent_id: None,
            label: "Open".into(),
            command_id: None,
            group: "main".into(),
            order: 0,
            when: ContextExpr::Any,
        })
        .expect("menu");
    let context = prism_editor_shell::ContextKeySet::new();
    let menu = contributions.menu("file", &context);
    let tree = shell_chrome(&ShellChromeProps {
        menus: &menu,
        ..shell_data()
    });
    assert_eq!(tree.child_elements()[1].child_elements().len(), 1);
}
