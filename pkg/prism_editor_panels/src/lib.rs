//! Data models for editor panels.
//!
//! These models own ordering, filtering, bounded storage, and virtualization
//! decisions. UI crates consume snapshots and do not own editor state.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use prism_editor_core::SelectionItem;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct OutlinerId(u128);

impl OutlinerId {
    #[must_use]
    pub const fn from_raw(raw: u128) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u128 {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutlinerNode {
    id: OutlinerId,
    parent: Option<OutlinerId>,
    label: String,
    selection: Option<SelectionItem>,
    children: Vec<OutlinerId>,
    expanded: bool,
    visible: bool,
}

impl OutlinerNode {
    #[must_use]
    pub fn new(id: OutlinerId, parent: Option<OutlinerId>, label: impl Into<String>) -> Self {
        Self {
            id,
            parent,
            label: label.into(),
            selection: None,
            children: Vec::new(),
            expanded: true,
            visible: true,
        }
    }

    #[must_use]
    pub const fn id(&self) -> OutlinerId {
        self.id
    }

    #[must_use]
    pub const fn parent(&self) -> Option<OutlinerId> {
        self.parent
    }

    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    #[must_use]
    pub fn children(&self) -> &[OutlinerId] {
        &self.children
    }

    #[must_use]
    pub const fn is_expanded(&self) -> bool {
        self.expanded
    }

    pub fn set_expanded(&mut self, expanded: bool) {
        self.expanded = expanded;
    }

    #[must_use]
    pub const fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn set_selection(&mut self, item: SelectionItem) {
        self.selection = Some(item);
    }

    #[must_use]
    pub fn selection(&self) -> Option<&SelectionItem> {
        self.selection.as_ref()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OutlinerError {
    DuplicateNode(OutlinerId),
    MissingNode(OutlinerId),
    ParentCycle(OutlinerId),
}

impl std::fmt::Display for OutlinerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateNode(id) => write!(formatter, "outliner node {} already exists", id.raw()),
            Self::MissingNode(id) => write!(formatter, "outliner node {} does not exist", id.raw()),
            Self::ParentCycle(id) => write!(formatter, "outliner parent cycle at node {}", id.raw()),
        }
    }
}

impl std::error::Error for OutlinerError {}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Outliner {
    nodes: BTreeMap<OutlinerId, OutlinerNode>,
    roots: Vec<OutlinerId>,
}

impl Outliner {
    pub fn insert(&mut self, node: OutlinerNode) -> Result<(), OutlinerError> {
        if self.nodes.contains_key(&node.id) {
            return Err(OutlinerError::DuplicateNode(node.id));
        }
        if let Some(parent) = node.parent {
            if !self.nodes.contains_key(&parent) {
                return Err(OutlinerError::MissingNode(parent));
            }
            if parent == node.id {
                return Err(OutlinerError::ParentCycle(node.id));
            }
        }
        let id = node.id;
        let parent = node.parent;
        self.nodes.insert(id, node);
        if let Some(parent) = parent {
            self.nodes.get_mut(&parent).expect("validated parent").children.push(id);
        } else {
            self.roots.push(id);
        }
        Ok(())
    }

    pub fn remove(&mut self, id: OutlinerId) -> Result<OutlinerNode, OutlinerError> {
        let node = self.nodes.remove(&id).ok_or(OutlinerError::MissingNode(id))?;
        if let Some(parent) = node.parent {
            if let Some(parent_node) = self.nodes.get_mut(&parent) {
                parent_node.children.retain(|child| *child != id);
            }
        } else {
            self.roots.retain(|root| *root != id);
        }
        for child in &node.children {
            if let Some(child_node) = self.nodes.get_mut(child) {
                child_node.parent = None;
                self.roots.push(*child);
            }
        }
        Ok(node)
    }

    #[must_use]
    pub fn node(&self, id: OutlinerId) -> Option<&OutlinerNode> {
        self.nodes.get(&id)
    }

    #[must_use]
    pub fn visible_window(&self, start: usize, count: usize) -> Vec<OutlinerId> {
        let mut flattened = Vec::new();
        for root in &self.roots {
            self.flatten_visible(*root, &mut flattened);
        }
        flattened.into_iter().skip(start).take(count).collect()
    }

    fn flatten_visible(&self, id: OutlinerId, output: &mut Vec<OutlinerId>) {
        let Some(node) = self.nodes.get(&id) else { return };
        if !node.visible { return; }
        output.push(id);
        if node.expanded {
            for child in &node.children { self.flatten_visible(*child, output); }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LogLevel { Trace, Debug, Info, Warn, Error }

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConsoleEntry {
    pub sequence: u64,
    pub level: LogLevel,
    pub channel: String,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Console {
    capacity: usize,
    next_sequence: u64,
    entries: Vec<ConsoleEntry>,
}

impl Console {
    pub fn new(capacity: usize) -> Result<Self, ConsoleError> {
        if capacity == 0 { return Err(ConsoleError::ZeroCapacity); }
        Ok(Self { capacity, next_sequence: 0, entries: Vec::with_capacity(capacity) })
    }

    pub fn push(&mut self, level: LogLevel, channel: impl Into<String>, message: impl Into<String>) {
        let entry = ConsoleEntry { sequence: self.next_sequence, level, channel: channel.into(), message: message.into() };
        self.next_sequence = self.next_sequence.saturating_add(1);
        if self.entries.len() == self.capacity { self.entries.remove(0); }
        self.entries.push(entry);
    }

    #[must_use]
    pub fn entries(&self) -> &[ConsoleEntry] { &self.entries }

    #[must_use]
    pub fn filtered(&self, minimum: LogLevel, channel: Option<&str>) -> Vec<&ConsoleEntry> {
        self.entries.iter().filter(|entry| entry.level >= minimum && channel.is_none_or(|value| value == entry.channel)).collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConsoleError { ZeroCapacity }

impl std::fmt::Display for ConsoleError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { formatter.write_str("console capacity must be greater than zero") }
}

impl std::error::Error for ConsoleError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outliner_window_respects_expansion() {
        let mut outliner = Outliner::default();
        outliner.insert(OutlinerNode::new(OutlinerId::from_raw(1), None, "root")).expect("root");
        outliner.insert(OutlinerNode::new(OutlinerId::from_raw(2), Some(OutlinerId::from_raw(1)), "child")).expect("child");
        assert_eq!(outliner.visible_window(0, 10), vec![OutlinerId::from_raw(1), OutlinerId::from_raw(2)]);
        outliner.node(OutlinerId::from_raw(1)).expect("root");
        outliner.nodes.get_mut(&OutlinerId::from_raw(1)).expect("root").set_expanded(false);
        assert_eq!(outliner.visible_window(0, 10), vec![OutlinerId::from_raw(1)]);
    }

    #[test]
    fn console_is_bounded_and_filters_without_reordering() {
        let mut console = Console::new(2).expect("console");
        console.push(LogLevel::Info, "editor", "one");
        console.push(LogLevel::Error, "runtime", "two");
        console.push(LogLevel::Warn, "editor", "three");
        assert_eq!(console.entries()[0].message, "two");
        assert_eq!(console.filtered(LogLevel::Warn, Some("editor")).len(), 1);
    }
}
