use crate::{
    AutoHideEntry, DockLayout, DockNode, FloatingWindow, LayoutError, MaximizeRestore, PanelId,
    Ratio, Rect, Side, SplitChild, TabId, ValidationIssue,
};
use alloc::{boxed::Box, string::String, vec, vec::Vec};

/// A node addressed by a layout action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// Address a tool panel by ID.
    Panel(PanelId),
    /// Address a document by ID.
    Document(String),
}

/// How a node is inserted relative to its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Add a panel to the target's tab stack.
    Tab,
    /// Insert a panel before the tab at the given index, or append at `len`.
    TabAt(usize),
    /// Create a split on the selected edge.
    Side(Side),
}

/// Pure layout intents accepted by [`reduce`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// Insert a new panel beside a target.
    Split {
        /// New panel to insert.
        panel: PanelId,
        /// Existing panel or document to split beside.
        target: Target,
        /// Edge at which the new panel is placed.
        side: Side,
    },
    /// Move a panel into the target's tab stack.
    Tabify {
        /// Existing panel to move.
        panel: PanelId,
        /// Existing panel whose tab stack receives it.
        target: Target,
    },
    /// Move a panel by tabifying or splitting it beside a target.
    Move {
        /// Existing panel to move.
        panel: PanelId,
        /// Destination node.
        target: Target,
        /// Tab placement or edge split placement.
        placement: Placement,
    },
    /// Activates a tab in the target's tab stack.
    ActivateTab {
        /// Existing panel or document in the destination stack.
        target: Target,
        /// Zero-based tab index within that stack.
        index: usize,
    },
    /// Remove a panel from the layout.
    Close {
        /// Panel to remove.
        panel: PanelId,
    },
    /// Replace the ratios at a split addressed by a child-index path.
    Resize {
        /// Child-index path from the main root to the split.
        path: Vec<usize>,
        /// New child shares; values are normalized to sum to one.
        ratios: Vec<Ratio>,
    },
    /// Move a panel into a new floating window.
    Float {
        /// Panel to float.
        panel: PanelId,
        /// Initial logical window rectangle.
        rect: Rect,
    },
    /// Dock a floating window into a target.
    Dock {
        /// Index of the floating window to dock.
        window: usize,
        /// Destination node in the main or another floating tree.
        target: Target,
        /// Tab placement or edge split placement.
        placement: Placement,
    },
    /// Maximize one panel while retaining a restore snapshot.
    Maximize {
        /// Panel to show as the sole main surface.
        panel: PanelId,
    },
    /// Restore the layout saved by [`Action::Maximize`].
    Restore,
    /// Collapse a panel into an edge auto-hide bar.
    AutoHide {
        /// Panel to collapse.
        panel: PanelId,
        /// Edge that hosts it.
        side: Side,
    },
    /// Reveal or hide an auto-hide panel.
    ShowAutoHide {
        /// Panel to reveal, or `None` to hide the currently revealed panel.
        panel: Option<PanelId>,
    },
}

/// Builds an activation action for a tab in the current layout.
///
/// The returned index is the tab's index in the containing panel or document
/// stack.  The search covers the main tree and every floating tree, matching
/// the reducer's own [`Action::ActivateTab`] lookup rules.  This function is
/// pure and does not mutate `layout`, so callers can safely use it as an
/// adapter from input or accessibility events before dispatching the action.
pub fn activate_tab_action(layout: &DockLayout, tab: &TabId) -> Result<Action, LayoutError> {
    layout.validate()?;

    let (target, index) = match tab {
        TabId::Panel(panel) => {
            let index = find_tab_index(&layout.root, tab).or_else(|| {
                layout
                    .floating
                    .iter()
                    .find_map(|window| find_tab_index(&window.root, tab))
            });
            (Target::Panel(panel.clone()), index)
        }
        TabId::Document(document) => {
            let index = find_tab_index(&layout.root, tab).or_else(|| {
                layout
                    .floating
                    .iter()
                    .find_map(|window| find_tab_index(&window.root, tab))
            });
            (Target::Document(document.clone()), index)
        }
    };

    index
        .map(|index| Action::ActivateTab { target, index })
        .ok_or_else(|| LayoutError {
            issue: ValidationIssue::TargetNotFound,
            path: Vec::new(),
        })
}

fn find_tab_index(node: &DockNode, tab: &TabId) -> Option<usize> {
    match (node, tab) {
        (DockNode::TabStack { tabs, .. }, TabId::Panel(panel)) => {
            tabs.iter().position(|candidate| candidate == panel)
        }
        (DockNode::Document { tabs, .. }, TabId::Document(document)) => {
            tabs.iter().position(|candidate| candidate == document)
        }
        (DockNode::Split { children, .. }, _) => children
            .iter()
            .find_map(|child| find_tab_index(&child.node, tab)),
        _ => None,
    }
}

/// Applies an action atomically. On failure the input layout is untouched.
pub fn reduce(layout: &DockLayout, action: Action) -> Result<DockLayout, LayoutError> {
    // Never let a reducer operation silently normalize or partially repair
    // malformed persisted/user-provided state.
    layout.validate()?;
    if layout.maximized.is_some() && !matches!(action, Action::Restore) {
        return Err(err(ValidationIssue::LayoutMaximized));
    }
    let mut next = layout.clone();
    apply(&mut next, action)?;
    normalize(&mut next);
    next.validate()?;
    Ok(next)
}

fn err(issue: ValidationIssue) -> LayoutError {
    LayoutError {
        issue,
        path: Vec::new(),
    }
}

fn apply(layout: &mut DockLayout, action: Action) -> Result<(), LayoutError> {
    match action {
        Action::Split {
            panel,
            target,
            side,
        } => {
            if locate_panel(layout, &panel) {
                return Err(err(ValidationIssue::DuplicatePanel(panel)));
            }
            insert_layout_target(
                layout,
                &target,
                DockNode::panel(panel),
                Placement::Side(side),
            )?;
        }
        Action::Tabify { panel, target } => move_panel(layout, &panel, &target, Placement::Tab)?,
        Action::ActivateTab { target, index } => activate_tab(layout, &target, index)?,
        Action::Move {
            panel,
            target,
            placement,
        } => move_panel(layout, &panel, &target, placement)?,
        Action::Close { panel } => {
            let removed = remove_panel_everywhere(layout, &panel);
            if !removed {
                return Err(err(ValidationIssue::PanelNotFound(panel)));
            }
            layout.autohide.entries.retain(|e| e.panel != panel);
            if layout.autohide.active.as_deref() == Some(panel.as_str()) {
                layout.autohide.active = None;
            }
        }
        Action::Resize { path, ratios } => {
            let node = node_at_mut(&mut layout.root, &path)
                .ok_or_else(|| err(ValidationIssue::InvalidSplitArity))?;
            let DockNode::Split { children, .. } = node else {
                return Err(err(ValidationIssue::InvalidSplitArity));
            };
            if ratios.len() != children.len() {
                return Err(err(ValidationIssue::InvalidRatio));
            }
            let normalized = normalize_ratios(&ratios)?;
            for (child, ratio) in children.iter_mut().zip(normalized) {
                child.ratio = ratio;
            }
        }
        Action::Float { panel, rect } => {
            let node = take_panel_node(layout, &panel)
                .ok_or_else(|| err(ValidationIssue::PanelNotFound(panel.clone())))?;
            let z = layout
                .floating
                .iter()
                .map(|w| w.z)
                .max()
                .unwrap_or(0)
                .saturating_add(1);
            layout.floating.push(FloatingWindow {
                rect,
                root: node,
                z,
            });
        }
        Action::Dock {
            window,
            target,
            placement,
        } => {
            if window >= layout.floating.len() {
                return Err(err(ValidationIssue::ResourceLimit("floating window index")));
            }
            let moving = layout.floating.remove(window).root;
            insert_layout_target(layout, &target, moving, placement)?;
        }
        Action::Maximize { panel } => {
            if layout.maximized.is_some() {
                return Err(err(ValidationIssue::MaximizeSnapshotInvalid));
            }
            let saved = MaximizeRestore {
                panel: panel.clone(),
                root: layout.root.clone(),
                floating: layout.floating.clone(),
                autohide: layout.autohide.clone(),
            };
            let node = take_panel_node(layout, &panel)
                .ok_or_else(|| err(ValidationIssue::PanelNotFound(panel.clone())))?;
            // Keep the selected pane visible as the whole dock surface and retain
            // the entire previous state (including sibling splits) for Restore.
            layout.root = node;
            layout.floating.clear();
            layout.autohide = Default::default();
            layout.maximized = Some(Box::new(saved));
        }
        Action::Restore => {
            let saved = layout
                .maximized
                .take()
                .ok_or_else(|| err(ValidationIssue::MaximizeSnapshotInvalid))?;
            layout.root = saved.root;
            layout.floating = saved.floating;
            layout.autohide = saved.autohide;
        }
        Action::AutoHide { panel, side } => {
            if layout.autohide.entries.iter().any(|e| e.panel == panel) {
                return Err(err(ValidationIssue::DuplicateAutohide(panel)));
            }
            if take_panel_node(layout, &panel).is_none() {
                return Err(err(ValidationIssue::PanelNotFound(panel)));
            }
            layout.autohide.entries.push(AutoHideEntry { side, panel });
        }
        Action::ShowAutoHide { panel } => {
            if let Some(ref p) = panel
                && !layout.autohide.entries.iter().any(|e| &e.panel == p)
            {
                return Err(err(ValidationIssue::InvalidAutohideActive));
            }
            layout.autohide.active = panel;
        }
    }
    Ok(())
}

fn activate_tab(layout: &mut DockLayout, target: &Target, index: usize) -> Result<(), LayoutError> {
    if activate_node(&mut layout.root, target, index) {
        return Ok(());
    }
    for window in &mut layout.floating {
        if activate_node(&mut window.root, target, index) {
            return Ok(());
        }
    }
    Err(err(ValidationIssue::TargetNotFound))
}

fn activate_node(node: &mut DockNode, target: &Target, index: usize) -> bool {
    match node {
        DockNode::Split { children, .. } => children
            .iter_mut()
            .any(|child| activate_node(&mut child.node, target, index)),
        DockNode::TabStack { tabs, active } if matches!(target, Target::Panel(panel) if tabs.contains(panel)) =>
        {
            if index >= tabs.len() {
                return false;
            }
            *active = index;
            true
        }
        DockNode::Document { tabs, active } if matches!(target, Target::Document(document) if tabs.contains(document)) =>
        {
            if index >= tabs.len() {
                return false;
            }
            *active = index;
            true
        }
        _ => false,
    }
}

fn move_panel(
    layout: &mut DockLayout,
    panel: &str,
    target: &Target,
    placement: Placement,
) -> Result<(), LayoutError> {
    if matches!(target, Target::Panel(p) if p == panel) {
        return Ok(());
    }
    // `TabAt` is expressed in the destination stack before the move.  When
    // source and target are in that same stack, removing the source shifts
    // every boundary to its right by one.
    let placement = match (target, placement) {
        (Target::Panel(target), Placement::TabAt(index)) => {
            let Some((source_index, stack_len)) =
                same_tab_stack_source_index(&layout.root, panel, target).or_else(|| {
                    layout
                        .floating
                        .iter()
                        .find_map(|window| same_tab_stack_source_index(&window.root, panel, target))
                })
            else {
                return Err(err(ValidationIssue::TargetNotFound));
            };
            if index > stack_len {
                return Err(err(ValidationIssue::TargetNotFound));
            }
            let adjusted = if source_index < index {
                index.saturating_sub(1)
            } else {
                index
            };
            Placement::TabAt(adjusted)
        }
        (_, placement) => placement,
    };
    let node = take_panel_node(layout, panel)
        .ok_or_else(|| err(ValidationIssue::PanelNotFound(panel.into())))?;
    insert_layout_target(layout, target, node, placement)
}

fn same_tab_stack_source_index(
    node: &DockNode,
    source: &str,
    target: &str,
) -> Option<(usize, usize)> {
    match node {
        DockNode::TabStack { tabs, .. }
            if tabs.iter().any(|tab| tab == source) && tabs.iter().any(|tab| tab == target) =>
        {
            tabs.iter()
                .position(|tab| tab == source)
                .map(|index| (index, tabs.len()))
        }
        DockNode::Split { children, .. } => children
            .iter()
            .find_map(|child| same_tab_stack_source_index(&child.node, source, target)),
        _ => None,
    }
}

fn insert_layout_target(
    layout: &mut DockLayout,
    target: &Target,
    moving: DockNode,
    placement: Placement,
) -> Result<(), LayoutError> {
    if insert_target(&mut layout.root, target, moving.clone(), placement).is_ok() {
        return Ok(());
    }
    for window in &mut layout.floating {
        if insert_target(&mut window.root, target, moving.clone(), placement).is_ok() {
            return Ok(());
        }
    }
    Err(err(ValidationIssue::TargetNotFound))
}

fn insert_target(
    root: &mut DockNode,
    target: &Target,
    moving: DockNode,
    placement: Placement,
) -> Result<(), LayoutError> {
    if contains_target(&moving, target) {
        return Err(err(ValidationIssue::TargetNotFound));
    }
    if insert_rec(root, target, moving, placement) {
        Ok(())
    } else {
        Err(err(ValidationIssue::TargetNotFound))
    }
}

fn contains_target(node: &DockNode, target: &Target) -> bool {
    match (node, target) {
        (DockNode::TabStack { tabs, .. }, Target::Panel(p)) => tabs.contains(p),
        (DockNode::Document { tabs, .. }, Target::Document(d)) => tabs.contains(d),
        (DockNode::Split { children, .. }, _) => {
            children.iter().any(|c| contains_target(&c.node, target))
        }
        _ => false,
    }
}

fn insert_rec(
    root: &mut DockNode,
    target: &Target,
    moving: DockNode,
    placement: Placement,
) -> bool {
    match root {
        DockNode::Split { children, .. } => children
            .iter_mut()
            .any(|c| insert_rec(&mut c.node, target, moving.clone(), placement)),
        DockNode::TabStack { tabs, active } if matches!(target, Target::Panel(p) if tabs.contains(p)) => {
            match placement {
                Placement::Tab => {
                    if let DockNode::TabStack {
                        tabs: incoming,
                        active: selected,
                    } = moving
                    {
                        let offset = tabs.len();
                        tabs.extend(incoming);
                        *active = offset + selected;
                        true
                    } else {
                        false
                    }
                }
                Placement::TabAt(index) => {
                    if let DockNode::TabStack {
                        tabs: incoming,
                        active: _,
                    } = moving
                    {
                        if index > tabs.len() {
                            return false;
                        }
                        let incoming_len = incoming.len();
                        let old_active = *active;
                        tabs.splice(index..index, incoming);
                        if index <= old_active {
                            *active = old_active + incoming_len;
                        } else {
                            *active = old_active;
                        }
                        true
                    } else {
                        false
                    }
                }
                Placement::Side(side) => {
                    let old = core::mem::replace(root, DockNode::Empty);
                    *root = side_split(old, moving, side);
                    true
                }
            }
        }
        DockNode::Document { tabs, .. } if matches!(target, Target::Document(d) if tabs.contains(d)) => {
            match placement {
                Placement::Tab | Placement::TabAt(_) => false,
                Placement::Side(side) => {
                    let old = core::mem::replace(root, DockNode::Empty);
                    *root = side_split(old, moving, side);
                    true
                }
            }
        }
        _ => false,
    }
}

fn side_split(existing: DockNode, incoming: DockNode, side: Side) -> DockNode {
    let mut pair = vec![
        SplitChild {
            ratio: Ratio::HALF,
            node: incoming,
        },
        SplitChild {
            ratio: Ratio::HALF,
            node: existing,
        },
    ];
    if !side.is_before() {
        pair.swap(0, 1);
    }
    DockNode::Split {
        axis: side.axis(),
        children: pair,
    }
}

fn remove_panel_everywhere(layout: &mut DockLayout, panel: &str) -> bool {
    let mut found = take_panel_from_node(&mut layout.root, panel).is_some();
    for w in &mut layout.floating {
        found |= take_panel_from_node(&mut w.root, panel).is_some();
    }
    let before = layout.autohide.entries.len();
    layout.autohide.entries.retain(|e| e.panel != panel);
    found || (before != layout.autohide.entries.len())
}
fn take_panel_node(layout: &mut DockLayout, panel: &str) -> Option<DockNode> {
    if let Some(n) = take_panel_from_node(&mut layout.root, panel) {
        return Some(n);
    }
    for i in 0..layout.floating.len() {
        if let Some(n) = take_panel_from_node(&mut layout.floating[i].root, panel) {
            return Some(n);
        }
    }
    if let Some(i) = layout
        .autohide
        .entries
        .iter()
        .position(|e| e.panel == panel)
    {
        layout.autohide.entries.remove(i);
        if layout.autohide.active.as_deref() == Some(panel) {
            layout.autohide.active = None;
        }
        return Some(DockNode::panel(panel));
    }
    None
}
fn take_panel_from_node(node: &mut DockNode, panel: &str) -> Option<DockNode> {
    match node {
        DockNode::TabStack { tabs, active } => {
            if let Some(i) = tabs.iter().position(|p| p == panel) {
                tabs.remove(i);
                if tabs.is_empty() {
                    *node = DockNode::Empty;
                    return Some(DockNode::panel(panel));
                }
                *active = if i < *active {
                    active.saturating_sub(1)
                } else if *active >= tabs.len() {
                    tabs.len() - 1
                } else {
                    *active
                };
                return Some(DockNode::panel(panel));
            }
            None
        }
        DockNode::Split { children, .. } => {
            for c in children.iter_mut() {
                if let Some(n) = take_panel_from_node(&mut c.node, panel) {
                    return Some(n);
                }
            }
            None
        }
        _ => None,
    }
}

fn normalize(layout: &mut DockLayout) {
    normalize_node(&mut layout.root, true);
    for w in &mut layout.floating {
        normalize_node(&mut w.root, true);
    }
    layout.floating.retain(|w| w.root != DockNode::Empty);
    layout.autohide.entries.dedup_by(|a, b| a.panel == b.panel);
    if let Some(p) = &layout.autohide.active
        && !layout.autohide.entries.iter().any(|e| &e.panel == p)
    {
        layout.autohide.active = None;
    }
}
fn normalize_node(node: &mut DockNode, root: bool) {
    if let DockNode::Split { axis, children } = node {
        let axis = *axis;
        for child in children.iter_mut() {
            normalize_node(&mut child.node, false);
        }
        children.retain(|c| c.node != DockNode::Empty);
        let mut flattened = Vec::with_capacity(children.len());
        for child in core::mem::take(children) {
            match child.node {
                DockNode::Split {
                    axis: nested_axis,
                    children: nested,
                } if nested_axis == axis => {
                    for nested_child in nested {
                        let combined = ((u64::from(child.ratio.millionths())
                            * u64::from(nested_child.ratio.millionths())
                            + u64::from(Ratio::SCALE / 2))
                            / u64::from(Ratio::SCALE))
                        .max(1) as u32;
                        flattened.push(SplitChild {
                            ratio: Ratio::from_millionths(combined),
                            node: nested_child.node,
                        });
                    }
                }
                node => flattened.push(SplitChild {
                    ratio: child.ratio,
                    node,
                }),
            }
        }
        *children = flattened;
        if children.is_empty() {
            *node = DockNode::Empty;
        } else if children.len() == 1 {
            *node = children.remove(0).node;
        } else if children.len() >= 2 {
            let ratios: Vec<_> = children.iter().map(|c| c.ratio).collect();
            if let Ok(normalized) = normalize_ratios(&ratios) {
                for (c, r) in children.iter_mut().zip(normalized) {
                    c.ratio = r;
                }
            }
        } else if !root {
            *node = DockNode::Empty;
        }
    }
    match node {
        DockNode::TabStack { tabs, active } => {
            tabs.retain(|p| !p.is_empty());
            if tabs.is_empty() && !root {
                *node = DockNode::Empty;
            } else if !tabs.is_empty() && *active >= tabs.len() {
                *active = tabs.len() - 1;
            }
        }
        DockNode::Document { tabs, active } => {
            if tabs.is_empty() {
                *node = DockNode::Empty;
            } else if *active >= tabs.len() {
                *active = tabs.len() - 1;
            }
        }
        _ => (),
    }
}
fn normalize_ratios(values: &[Ratio]) -> Result<Vec<Ratio>, LayoutError> {
    if values.len() < 2 || values.iter().any(|r| r.millionths() == 0) {
        return Err(err(ValidationIssue::InvalidRatio));
    }
    let sum: u64 = values.iter().map(|r| r.millionths() as u64).sum();
    if sum == 0 {
        return Err(err(ValidationIssue::InvalidRatio));
    }
    let mut out = Vec::with_capacity(values.len());
    let mut used = 0u32;
    for (i, r) in values.iter().enumerate() {
        let part = if i + 1 == values.len() {
            Ratio::SCALE - used
        } else {
            ((r.millionths() as u64 * Ratio::SCALE as u64) / sum) as u32
        };
        if part == 0 {
            return Err(err(ValidationIssue::InvalidRatio));
        }
        used += part;
        out.push(Ratio::from_millionths(part));
    }
    Ok(out)
}
fn locate_panel(layout: &DockLayout, panel: &str) -> bool {
    layout.root.contains_panel(panel)
        || layout.floating.iter().any(|w| w.root.contains_panel(panel))
        || layout.autohide.entries.iter().any(|e| e.panel == panel)
}
fn node_at_mut<'a>(root: &'a mut DockNode, path: &[usize]) -> Option<&'a mut DockNode> {
    let mut node = root;
    for &i in path {
        match node {
            DockNode::Split { children, .. } => node = &mut children.get_mut(i)?.node,
            _ => return None,
        }
    }
    Some(node)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn same_stack_tab_reorder_adjusts_boundaries_after_removal() {
        let layout = DockLayout {
            root: DockNode::TabStack {
                tabs: vec!["a".into(), "b".into(), "c".into()],
                active: 0,
            },
            ..DockLayout::default()
        };

        let moved = reduce(
            &layout,
            Action::Move {
                panel: "a".into(),
                target: Target::Panel("c".into()),
                placement: Placement::TabAt(3),
            },
        )
        .expect("moving the first tab to the end is valid");
        assert_eq!(
            moved.root,
            DockNode::TabStack {
                tabs: vec!["b".into(), "c".into(), "a".into()],
                active: 0,
            }
        );

        assert!(reduce(
            &moved,
            Action::Move {
                panel: "a".into(),
                target: Target::Panel("c".into()),
                placement: Placement::TabAt(4),
            },
        )
        .is_err());

        let moved = reduce(
            &moved,
            Action::Move {
                panel: "a".into(),
                target: Target::Panel("b".into()),
                placement: Placement::TabAt(0),
            },
        )
        .expect("moving a tab to the beginning is valid");
        assert_eq!(
            moved.root,
            DockNode::TabStack {
                tabs: vec!["a".into(), "b".into(), "c".into()],
                active: 1,
            }
        );
    }

    #[test]
    fn activate_tab_changes_only_the_selected_stack() {
        let layout = DockLayout {
            root: DockNode::TabStack {
                tabs: vec!["a".into(), "b".into()],
                active: 0,
            },
            ..DockLayout::default()
        };
        let next = reduce(
            &layout,
            Action::ActivateTab {
                target: Target::Panel("b".into()),
                index: 1,
            },
        )
        .unwrap();
        assert_eq!(
            next.root,
            DockNode::TabStack {
                tabs: vec!["a".into(), "b".into()],
                active: 1,
            }
        );
    }

    #[test]
    fn activate_tab_action_finds_panel_and_document_indexes_in_current_layout() {
        let layout = DockLayout {
            root: DockNode::Split {
                axis: crate::Axis::Horizontal,
                children: vec![
                    SplitChild {
                        ratio: Ratio::HALF,
                        node: DockNode::TabStack {
                            tabs: vec!["panel.a".into(), "panel.b".into()],
                            active: 0,
                        },
                    },
                    SplitChild {
                        ratio: Ratio::HALF,
                        node: DockNode::Document {
                            tabs: vec!["doc.a".into(), "doc.b".into()],
                            active: 0,
                        },
                    },
                ],
            },
            ..DockLayout::default()
        };

        assert_eq!(
            activate_tab_action(&layout, &TabId::Panel("panel.b".into())).unwrap(),
            Action::ActivateTab {
                target: Target::Panel("panel.b".into()),
                index: 1,
            }
        );
        assert_eq!(
            activate_tab_action(&layout, &TabId::Document("doc.b".into())).unwrap(),
            Action::ActivateTab {
                target: Target::Document("doc.b".into()),
                index: 1,
            }
        );
    }

    #[test]
    fn normalize_flattens_adjacent_splits_with_the_same_axis() {
        let layout = DockLayout {
            root: DockNode::Split {
                axis: crate::Axis::Horizontal,
                children: vec![
                    SplitChild {
                        ratio: Ratio::HALF,
                        node: DockNode::Split {
                            axis: crate::Axis::Horizontal,
                            children: vec![
                                SplitChild {
                                    ratio: Ratio::HALF,
                                    node: DockNode::panel("a"),
                                },
                                SplitChild {
                                    ratio: Ratio::HALF,
                                    node: DockNode::panel("b"),
                                },
                            ],
                        },
                    },
                    SplitChild {
                        ratio: Ratio::HALF,
                        node: DockNode::panel("c"),
                    },
                ],
            },
            ..DockLayout::default()
        };
        let reduced = reduce(
            &layout,
            Action::ActivateTab {
                target: Target::Panel("a".into()),
                index: 0,
            },
        )
        .unwrap();
        let DockNode::Split { axis, children } = reduced.root else {
            panic!("expected a normalized split");
        };
        assert_eq!(axis, crate::Axis::Horizontal);
        assert_eq!(children.len(), 3);
        assert_eq!(
            children.iter().map(|child| child.ratio).collect::<Vec<_>>(),
            vec![
                Ratio::from_millionths(250_000),
                Ratio::from_millionths(250_000),
                Ratio::from_millionths(500_000),
            ]
        );
    }

    #[test]
    fn activate_tab_action_rejects_missing_tab_without_mutating_layout() {
        let layout = DockLayout {
            root: DockNode::panel("panel.a"),
            ..DockLayout::default()
        };

        let error = activate_tab_action(&layout, &TabId::Panel("missing".into())).unwrap_err();
        assert_eq!(error.issue, ValidationIssue::TargetNotFound);
        assert_eq!(layout.root, DockNode::panel("panel.a"));
    }
}
