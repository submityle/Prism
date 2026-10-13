use crate::{DockLayout, DockNode, MaximizeRestore, Ratio};
use alloc::{collections::BTreeSet, string::String, vec::Vec};

/// Resource limits applied while validating untrusted layout data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Maximum split depth.
    pub max_depth: usize,
    /// Maximum number of tree nodes across the main tree and floating trees.
    pub max_nodes: usize,
    /// Maximum number of distinct tool panels.
    pub max_panels: usize,
    /// Maximum number of distinct documents.
    pub max_documents: usize,
    /// Maximum number of floating windows.
    pub max_floating: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_depth: 128,
            max_nodes: 16_384,
            max_panels: 4_096,
            max_documents: 4_096,
            max_floating: 256,
        }
    }
}

/// Structured reasons a layout failed validation or an action was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationIssue {
    /// An empty node appeared below the root.
    EmptyNonRoot,
    /// A tab or document leaf contains no entries.
    EmptyTabs,
    /// A leaf's active index is outside its entry list.
    InvalidActiveIndex,
    /// A split has fewer than two children.
    InvalidSplitArity,
    /// Split ratios are zero, out of range, or do not sum to one.
    InvalidRatio,
    /// A panel is inserted more than once.
    DuplicatePanel(String),
    /// A panel appears in separate layout locations.
    PanelInMultipleLocations(String),
    /// A document is inserted more than once.
    DuplicateDocument(String),
    /// A panel or document identifier is empty.
    EmptyIdentifier,
    /// A panel is repeated in the auto-hide bars.
    DuplicateAutohide(String),
    /// The selected auto-hide panel is not registered in the bars.
    InvalidAutohideActive,
    /// Two floating windows use the same stacking order.
    DuplicateZ(u32),
    /// A floating window has an empty extent.
    InvalidFloatingRect,
    /// A layout action other than restore was requested during maximization.
    LayoutMaximized,
    /// A configured resource limit was exceeded.
    ResourceLimit(&'static str),
    /// The maximize restore snapshot is not itself a valid layout.
    MaximizeSnapshotInvalid,
    /// An action addressed a panel that does not exist.
    PanelNotFound(String),
    /// An action addressed a node that does not exist.
    TargetNotFound,
}

/// Validation failure with the path to the offending node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayoutError {
    /// Specific invariant or resource-limit violation.
    pub issue: ValidationIssue,
    /// Child-index path from the relevant tree root.
    pub path: Vec<usize>,
}

impl DockLayout {
    /// Validates using the conservative default resource limits.
    pub fn validate(&self) -> Result<(), LayoutError> {
        self.validate_with(Limits::default())
    }
    /// Validates against caller-supplied resource limits.
    pub fn validate_with(&self, limits: Limits) -> Result<(), LayoutError> {
        let mut state = State {
            limits,
            nodes: 0,
            panels: BTreeSet::new(),
            docs: BTreeSet::new(),
            path: Vec::new(),
        };
        state.walk(&self.root, 0, true)?;
        if self.floating.len() > limits.max_floating {
            return state.fail(ValidationIssue::ResourceLimit("floating windows"));
        }
        let mut zs = BTreeSet::new();
        for (i, w) in self.floating.iter().enumerate() {
            state.path = alloc::vec![i];
            if w.rect.width == 0 || w.rect.height == 0 {
                return state.fail(ValidationIssue::InvalidFloatingRect);
            }
            if !zs.insert(w.z) {
                return state.fail(ValidationIssue::DuplicateZ(w.z));
            }
            state.walk(&w.root, 0, false)?;
        }
        let mut hidden = BTreeSet::new();
        for e in &self.autohide.entries {
            if e.panel.is_empty() {
                return state.fail(ValidationIssue::EmptyIdentifier);
            }
            if !hidden.insert(e.panel.clone()) {
                return state.fail(ValidationIssue::DuplicateAutohide(e.panel.clone()));
            }
            if !state.panels.insert(e.panel.clone()) {
                return state.fail(ValidationIssue::PanelInMultipleLocations(e.panel.clone()));
            }
            if state.panels.len() > limits.max_panels {
                return state.fail(ValidationIssue::ResourceLimit("panels"));
            }
        }
        if let Some(active) = &self.autohide.active
            && !hidden.contains(active)
        {
            return state.fail(ValidationIssue::InvalidAutohideActive);
        }
        if let Some(snapshot) = &self.maximized {
            let current_is_snapshot_panel = matches!(
                &self.root,
                DockNode::TabStack { tabs, active: 0 }
                    if tabs.len() == 1 && tabs.first() == Some(&snapshot.panel)
            ) && self.floating.is_empty()
                && self.autohide.entries.is_empty()
                && self.autohide.active.is_none();
            if !current_is_snapshot_panel {
                return state.fail(ValidationIssue::MaximizeSnapshotInvalid);
            }
            let mut snap = Self {
                root: snapshot.root.clone(),
                floating: snapshot.floating.clone(),
                autohide: snapshot.autohide.clone(),
                maximized: None,
            };
            if snap.validate_with(limits).is_err()
                || count_panel_surfaces(snapshot, &snapshot.panel) != 1
            {
                return state.fail(ValidationIssue::MaximizeSnapshotInvalid);
            }
            let _ = &mut snap;
        }
        Ok(())
    }
}

fn count_panel_surfaces(layout: &MaximizeRestore, panel: &str) -> usize {
    fn count_node(node: &DockNode, panel: &str) -> usize {
        match node {
            DockNode::TabStack { tabs, .. } => {
                tabs.iter().filter(|id| id.as_str() == panel).count()
            }
            DockNode::Document { .. } | DockNode::Empty => 0,
            DockNode::Split { children, .. } => children
                .iter()
                .map(|child| count_node(&child.node, panel))
                .sum(),
        }
    }
    count_node(&layout.root, panel)
        + layout
            .floating
            .iter()
            .map(|window| count_node(&window.root, panel))
            .sum::<usize>()
        + layout
            .autohide
            .entries
            .iter()
            .filter(|entry| entry.panel == panel)
            .count()
}

struct State {
    limits: Limits,
    nodes: usize,
    panels: BTreeSet<String>,
    docs: BTreeSet<String>,
    path: Vec<usize>,
}
impl State {
    fn fail<T>(&self, issue: ValidationIssue) -> Result<T, LayoutError> {
        Err(LayoutError {
            issue,
            path: self.path.clone(),
        })
    }
    fn walk(&mut self, node: &DockNode, depth: usize, root: bool) -> Result<(), LayoutError> {
        self.nodes += 1;
        if self.nodes > self.limits.max_nodes {
            return self.fail(ValidationIssue::ResourceLimit("nodes"));
        }
        if depth > self.limits.max_depth {
            return self.fail(ValidationIssue::ResourceLimit("depth"));
        }
        match node {
            DockNode::Empty if root => Ok(()),
            DockNode::Empty => self.fail(ValidationIssue::EmptyNonRoot),
            DockNode::TabStack { tabs, active } => {
                if tabs.is_empty() {
                    return self.fail(ValidationIssue::EmptyTabs);
                }
                if *active >= tabs.len() {
                    return self.fail(ValidationIssue::InvalidActiveIndex);
                }
                for p in tabs {
                    if p.is_empty() {
                        return self.fail(ValidationIssue::EmptyIdentifier);
                    }
                    if !self.panels.insert(p.clone()) {
                        return self.fail(ValidationIssue::DuplicatePanel(p.clone()));
                    }
                    if self.panels.len() > self.limits.max_panels {
                        return self.fail(ValidationIssue::ResourceLimit("panels"));
                    }
                }
                Ok(())
            }
            DockNode::Document { tabs, active } => {
                if tabs.is_empty() {
                    return self.fail(ValidationIssue::EmptyTabs);
                }
                if *active >= tabs.len() {
                    return self.fail(ValidationIssue::InvalidActiveIndex);
                }
                for d in tabs {
                    if d.is_empty() {
                        return self.fail(ValidationIssue::EmptyIdentifier);
                    }
                    if !self.docs.insert(d.clone()) {
                        return self.fail(ValidationIssue::DuplicateDocument(d.clone()));
                    }
                    if self.docs.len() > self.limits.max_documents {
                        return self.fail(ValidationIssue::ResourceLimit("documents"));
                    }
                }
                Ok(())
            }
            DockNode::Split { children, .. } => {
                if children.len() < 2 {
                    return self.fail(ValidationIssue::InvalidSplitArity);
                }
                let mut sum = 0u64;
                for (i, c) in children.iter().enumerate() {
                    if c.ratio.millionths() == 0 || c.ratio.millionths() > Ratio::SCALE {
                        return self.fail(ValidationIssue::InvalidRatio);
                    }
                    sum += c.ratio.millionths() as u64;
                    self.path.push(i);
                    self.walk(&c.node, depth + 1, false)?;
                    self.path.pop();
                }
                if sum != Ratio::SCALE as u64 {
                    return self.fail(ValidationIssue::InvalidRatio);
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AutoHideBars, AutoHideEntry, Axis, DockNode, Side, SplitChild};
    use alloc::{boxed::Box, vec};

    #[test]
    fn document_and_autohide_entries_obey_their_limits() {
        let layout = DockLayout {
            root: DockNode::Document {
                tabs: vec!["doc.a".into(), "doc.b".into()],
                active: 0,
            },
            ..DockLayout::default()
        };
        let limits = Limits {
            max_documents: 1,
            ..Limits::default()
        };
        assert!(matches!(
            layout.validate_with(limits).unwrap_err().issue,
            ValidationIssue::ResourceLimit("documents")
        ));

        let layout = DockLayout {
            root: DockNode::Split {
                axis: Axis::Horizontal,
                children: vec![
                    SplitChild {
                        ratio: Ratio::HALF,
                        node: DockNode::panel("visible"),
                    },
                    SplitChild {
                        ratio: Ratio::HALF,
                        node: DockNode::panel("other"),
                    },
                ],
            },
            floating: vec![],
            autohide: AutoHideBars {
                entries: vec![AutoHideEntry {
                    side: Side::Left,
                    panel: "hidden".into(),
                }],
                active: None,
            },
            maximized: None,
        };
        let limits = Limits {
            max_panels: 2,
            ..Limits::default()
        };
        assert!(matches!(
            layout.validate_with(limits).unwrap_err().issue,
            ValidationIssue::ResourceLimit("panels")
        ));
        let valid_limits = Limits {
            max_panels: 3,
            ..Limits::default()
        };
        layout.validate_with(valid_limits).unwrap();
    }

    #[test]
    fn maximized_layout_must_show_the_snapshot_panel_as_the_only_surface() {
        let mut layout = DockLayout {
            root: DockNode::panel("other"),
            ..DockLayout::default()
        };
        layout.maximized = Some(Box::new(crate::MaximizeRestore {
            panel: "max".into(),
            root: DockNode::panel("before"),
            floating: Vec::new(),
            autohide: AutoHideBars::default(),
        }));
        assert_eq!(
            layout.validate().unwrap_err().issue,
            ValidationIssue::MaximizeSnapshotInvalid
        );
    }
}
