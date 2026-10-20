//! Data models for editor panels.
//!
//! These models own ordering, filtering, bounded storage, and virtualization
//! decisions. UI crates consume snapshots and do not own editor state.

#![forbid(unsafe_code)]

extern crate alloc;

use alloc::collections::BTreeMap;

use prism_editor_core::SelectionItem;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
/// Stable identifier for an entry in an [`Outliner`].
pub struct OutlinerId(u128);

impl OutlinerId {
    /// Creates an identifier from its persisted numeric representation.
    #[must_use]
    pub const fn from_raw(raw: u128) -> Self {
        Self(raw)
    }

    /// Returns the persisted numeric representation of this identifier.
    #[must_use]
    pub const fn raw(self) -> u128 {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// A single node in the editor outliner hierarchy.
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
    /// Creates a visible, expanded node with no associated selection item.
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

    /// Returns this node's stable identifier.
    #[must_use]
    pub const fn id(&self) -> OutlinerId {
        self.id
    }

    /// Returns the identifier of this node's parent, if it has one.
    #[must_use]
    pub const fn parent(&self) -> Option<OutlinerId> {
        self.parent
    }

    /// Returns the label displayed for this node.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns the node identifiers whose parent is this node.
    #[must_use]
    pub fn children(&self) -> &[OutlinerId] {
        &self.children
    }

    /// Returns whether this node's children are included in the visible tree.
    #[must_use]
    pub const fn is_expanded(&self) -> bool {
        self.expanded
    }

    /// Sets whether this node's children are included in the visible tree.
    pub fn set_expanded(&mut self, expanded: bool) {
        self.expanded = expanded;
    }

    /// Returns whether this node is included in visible-window results.
    #[must_use]
    pub const fn is_visible(&self) -> bool {
        self.visible
    }

    /// Sets whether this node is included in visible-window results.
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Associates a selectable editor item with this node.
    pub fn set_selection(&mut self, item: SelectionItem) {
        self.selection = Some(item);
    }

    /// Returns the selectable editor item associated with this node, if any.
    #[must_use]
    pub fn selection(&self) -> Option<&SelectionItem> {
        self.selection.as_ref()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// Errors produced while mutating an [`Outliner`].
pub enum OutlinerError {
    /// The inserted identifier is already present in the outliner.
    DuplicateNode(OutlinerId),
    /// A referenced node does not exist in the outliner.
    MissingNode(OutlinerId),
    /// The requested parent relationship would create a cycle.
    ParentCycle(OutlinerId),
}

impl std::fmt::Display for OutlinerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateNode(id) => {
                write!(formatter, "outliner node {} already exists", id.raw())
            }
            Self::MissingNode(id) => write!(formatter, "outliner node {} does not exist", id.raw()),
            Self::ParentCycle(id) => {
                write!(formatter, "outliner parent cycle at node {}", id.raw())
            }
        }
    }
}

impl std::error::Error for OutlinerError {}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
/// Hierarchical model used by the editor's outliner panel.
pub struct Outliner {
    nodes: BTreeMap<OutlinerId, OutlinerNode>,
    roots: Vec<OutlinerId>,
}

impl Outliner {
    /// Inserts a node while preserving insertion order among siblings and roots.
    ///
    /// The parent must already exist when one is supplied. Duplicate identifiers
    /// and self-parenting relationships are rejected without changing the model.
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
            self.nodes
                .get_mut(&parent)
                .expect("validated parent")
                .children
                .push(id);
        } else {
            self.roots.push(id);
        }
        Ok(())
    }

    /// Removes a node and promotes its children to roots.
    ///
    /// The removed node's children retain their relative order. Removing a
    /// missing identifier returns an error and leaves the model unchanged.
    pub fn remove(&mut self, id: OutlinerId) -> Result<OutlinerNode, OutlinerError> {
        let node = self
            .nodes
            .remove(&id)
            .ok_or(OutlinerError::MissingNode(id))?;
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

    /// Returns a node by identifier.
    #[must_use]
    pub fn node(&self, id: OutlinerId) -> Option<&OutlinerNode> {
        self.nodes.get(&id)
    }

    /// Returns a bounded slice of the depth-first visible tree order.
    ///
    /// `start` is the number of visible nodes to skip and `count` is the maximum
    /// number of identifiers returned. An empty window is returned when `start`
    /// is beyond the visible tree or `count` is zero.
    #[must_use]
    pub fn visible_window(&self, start: usize, count: usize) -> Vec<OutlinerId> {
        let mut flattened = Vec::new();
        for root in &self.roots {
            self.flatten_visible(*root, &mut flattened);
        }
        flattened.into_iter().skip(start).take(count).collect()
    }

    fn flatten_visible(&self, id: OutlinerId, output: &mut Vec<OutlinerId>) {
        let Some(node) = self.nodes.get(&id) else {
            return;
        };
        if !node.visible {
            return;
        }
        output.push(id);
        if node.expanded {
            for child in &node.children {
                self.flatten_visible(*child, output);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
/// Severity used to filter console entries.
pub enum LogLevel {
    /// Detailed tracing information.
    Trace,
    /// Diagnostic information useful during development.
    Debug,
    /// Normal operational information.
    Info,
    /// A condition that may require attention.
    Warn,
    /// A failure or error condition.
    Error,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// A bounded console message with a monotonic sequence number.
pub struct ConsoleEntry {
    /// Sequence number assigned when the entry is inserted.
    pub sequence: u64,
    /// Severity of the message.
    pub level: LogLevel,
    /// Logical source channel for the message.
    pub channel: String,
    /// Human-readable message text.
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// In-memory, bounded console history for an editor panel.
pub struct Console {
    capacity: usize,
    next_sequence: u64,
    entries: Vec<ConsoleEntry>,
}

impl Console {
    /// Creates a console that retains at most `capacity` entries.
    ///
    /// A zero capacity is rejected because the console cannot retain any
    /// observable history in that configuration.
    pub fn new(capacity: usize) -> Result<Self, ConsoleError> {
        if capacity == 0 {
            return Err(ConsoleError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            next_sequence: 0,
            entries: Vec::with_capacity(capacity),
        })
    }

    /// Appends an entry, evicting the oldest entry when the console is full.
    pub fn push(
        &mut self,
        level: LogLevel,
        channel: impl Into<String>,
        message: impl Into<String>,
    ) {
        let entry = ConsoleEntry {
            sequence: self.next_sequence,
            level,
            channel: channel.into(),
            message: message.into(),
        };
        self.next_sequence = self.next_sequence.saturating_add(1);
        if self.entries.len() == self.capacity {
            self.entries.remove(0);
        }
        self.entries.push(entry);
    }

    /// Returns retained entries from oldest to newest.
    #[must_use]
    pub fn entries(&self) -> &[ConsoleEntry] {
        &self.entries
    }

    /// Returns entries at or above `minimum`, optionally restricted to a channel.
    #[must_use]
    pub fn filtered(&self, minimum: LogLevel, channel: Option<&str>) -> Vec<&ConsoleEntry> {
        self.entries
            .iter()
            .filter(|entry| {
                entry.level >= minimum && channel.is_none_or(|value| value == entry.channel)
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Errors produced while constructing a [`Console`].
pub enum ConsoleError {
    /// The requested console capacity was zero.
    ZeroCapacity,
}

impl std::fmt::Display for ConsoleError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("console capacity must be greater than zero")
    }
}

impl std::error::Error for ConsoleError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outliner_window_respects_expansion() {
        let mut outliner = Outliner::default();
        outliner
            .insert(OutlinerNode::new(OutlinerId::from_raw(1), None, "root"))
            .expect("root");
        outliner
            .insert(OutlinerNode::new(
                OutlinerId::from_raw(2),
                Some(OutlinerId::from_raw(1)),
                "child",
            ))
            .expect("child");
        assert_eq!(
            outliner.visible_window(0, 10),
            vec![OutlinerId::from_raw(1), OutlinerId::from_raw(2)]
        );
        outliner.node(OutlinerId::from_raw(1)).expect("root");
        outliner
            .nodes
            .get_mut(&OutlinerId::from_raw(1))
            .expect("root")
            .set_expanded(false);
        assert_eq!(
            outliner.visible_window(0, 10),
            vec![OutlinerId::from_raw(1)]
        );
    }

    #[test]
    fn outliner_rejects_duplicate_and_missing_parent_without_mutation() {
        let root_id = OutlinerId::from_raw(1);
        let child_id = OutlinerId::from_raw(2);
        let missing_parent = OutlinerId::from_raw(99);
        let mut outliner = Outliner::default();
        outliner
            .insert(OutlinerNode::new(root_id, None, "root"))
            .expect("root");

        assert_eq!(
            outliner.insert(OutlinerNode::new(
                child_id,
                Some(missing_parent),
                "invalid child"
            )),
            Err(OutlinerError::MissingNode(missing_parent))
        );
        assert_eq!(
            outliner.insert(OutlinerNode::new(root_id, None, "duplicate")),
            Err(OutlinerError::DuplicateNode(root_id))
        );
        assert_eq!(outliner.visible_window(0, usize::MAX), vec![root_id]);
        assert!(outliner.node(child_id).is_none());
    }

    #[test]
    fn removing_parent_promotes_children_in_order() {
        let root_id = OutlinerId::from_raw(1);
        let first_child_id = OutlinerId::from_raw(2);
        let second_child_id = OutlinerId::from_raw(3);
        let mut outliner = Outliner::default();
        outliner
            .insert(OutlinerNode::new(root_id, None, "root"))
            .expect("root");
        outliner
            .insert(OutlinerNode::new(first_child_id, Some(root_id), "first"))
            .expect("first child");
        outliner
            .insert(OutlinerNode::new(second_child_id, Some(root_id), "second"))
            .expect("second child");

        let removed = outliner.remove(root_id).expect("root removal");

        assert_eq!(removed.id(), root_id);
        assert_eq!(removed.children(), &[first_child_id, second_child_id]);
        assert_eq!(outliner.node(first_child_id).expect("first").parent(), None);
        assert_eq!(
            outliner.node(second_child_id).expect("second").parent(),
            None
        );
        assert_eq!(
            outliner.visible_window(0, usize::MAX),
            vec![first_child_id, second_child_id]
        );
    }

    #[test]
    fn outliner_visibility_and_window_boundaries_are_safe() {
        let root_id = OutlinerId::from_raw(1);
        let child_id = OutlinerId::from_raw(2);
        let mut outliner = Outliner::default();
        outliner
            .insert(OutlinerNode::new(root_id, None, "root"))
            .expect("root");
        outliner
            .insert(OutlinerNode::new(child_id, Some(root_id), "child"))
            .expect("child");
        outliner
            .nodes
            .get_mut(&root_id)
            .expect("root")
            .set_visible(false);

        assert!(outliner.visible_window(0, 0).is_empty());
        assert!(outliner.visible_window(usize::MAX, 1).is_empty());
        assert!(outliner.visible_window(0, usize::MAX).is_empty());
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

    #[test]
    fn console_rejects_zero_capacity_and_saturates_sequence_numbers() {
        assert_eq!(Console::new(0), Err(ConsoleError::ZeroCapacity));

        let mut console = Console::new(2).expect("console");
        console.next_sequence = u64::MAX;
        console.push(LogLevel::Debug, "editor", "maximum");
        console.push(LogLevel::Debug, "editor", "saturated");

        assert_eq!(console.entries()[0].sequence, u64::MAX);
        assert_eq!(console.entries()[1].sequence, u64::MAX);
    }
}
