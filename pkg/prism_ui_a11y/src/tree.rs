//! The accessibility tree: an ordered collection of [`A11yNode`]s.

use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Key;

use crate::label::Label;
use crate::node::A11yNode;

/// An ordered collection of accessibility nodes keyed by element [`Key`].
///
/// Nodes are stored in insertion order (which models document order) and can
/// also be looked up by key in `O(log n)`. The tree is the authority for
/// resolving [`Label::LabelledBy`] references and for composing screen-reader
/// text.
#[derive(Clone, Debug, Default)]
pub struct A11yTree {
    /// Nodes in insertion order.
    nodes: Vec<A11yNode>,
    /// Map from key to the index of its node in `nodes`.
    index: BTreeMap<Key, usize>,
}

impl A11yTree {
    /// Creates an empty tree.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a node, preserving insertion order.
    ///
    /// If a node with the same key already exists it is replaced *in place*
    /// (its position in the order is kept). Returns the previous node when one
    /// was replaced.
    pub fn insert(&mut self, node: A11yNode) -> Option<A11yNode> {
        if let Some(&i) = self.index.get(&node.key) {
            let previous = core::mem::replace(&mut self.nodes[i], node);
            Some(previous)
        } else {
            self.index.insert(node.key.clone(), self.nodes.len());
            self.nodes.push(node);
            None
        }
    }

    /// Returns the node for `key`, if present.
    #[must_use]
    pub fn get(&self, key: &Key) -> Option<&A11yNode> {
        self.index.get(key).map(|&i| &self.nodes[i])
    }

    /// Returns `true` when a node with `key` is present.
    #[must_use]
    pub fn contains(&self, key: &Key) -> bool {
        self.index.contains_key(key)
    }

    /// The number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` when the tree holds no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Iterates over nodes in insertion order.
    pub fn iter(&self) -> core::slice::Iter<'_, A11yNode> {
        self.nodes.iter()
    }

    /// Returns the backing nodes in insertion order.
    #[must_use]
    pub fn nodes(&self) -> &[A11yNode] {
        &self.nodes
    }

    /// Resolves the accessible name of the node identified by `key`.
    ///
    /// [`Label::Text`] yields its literal content; [`Label::LabelledBy`] is
    /// followed to the referenced node's label (transitively, with cycle
    /// protection); [`Label::None`], a missing node, or an unresolvable
    /// reference yield `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_ui::Key;
    /// use prism_ui_a11y::{A11yNode, A11yTree, Label, Role};
    ///
    /// let mut tree = A11yTree::new();
    /// tree.insert(
    ///     A11yNode::builder(Key::Str("name".into()), Role::Textbox)
    ///         .label(Label::labelled_by(Key::Str("lbl".into())))
    ///         .build(),
    /// );
    /// tree.insert(
    ///     A11yNode::builder(Key::Str("lbl".into()), Role::Presentation)
    ///         .label(Label::text("Full name"))
    ///         .build(),
    /// );
    ///
    /// assert_eq!(
    ///     tree.label_text(&Key::Str("name".into())).as_deref(),
    ///     Some("Full name"),
    /// );
    /// ```
    #[must_use]
    pub fn label_text(&self, key: &Key) -> Option<String> {
        let mut visited = BTreeSet::new();
        self.resolve_label(key, &mut visited)
    }

    fn resolve_label(&self, key: &Key, visited: &mut BTreeSet<Key>) -> Option<String> {
        if !visited.insert(key.clone()) {
            return None;
        }
        match &self.get(key)?.label {
            Label::Text(text) => Some(text.clone()),
            Label::LabelledBy(target) => self.resolve_label(target, visited),
            Label::None => None,
        }
    }
}

impl<'a> IntoIterator for &'a A11yTree {
    type Item = &'a A11yNode;
    type IntoIter = core::slice::Iter<'a, A11yNode>;

    fn into_iter(self) -> Self::IntoIter {
        self.nodes.iter()
    }
}

impl FromIterator<A11yNode> for A11yTree {
    fn from_iter<I: IntoIterator<Item = A11yNode>>(iter: I) -> Self {
        let mut tree = A11yTree::new();
        for node in iter {
            tree.insert(node);
        }
        tree
    }
}

#[cfg(test)]
mod tests {
    use super::A11yTree;
    use crate::label::Label;
    use crate::node::A11yNode;
    use crate::role::Role;
    use alloc::vec::Vec;
    use prism_ui::Key;

    fn node(key: i64, role: Role, label: Label) -> A11yNode {
        A11yNode::builder(Key::Int(key), role).label(label).build()
    }

    #[test]
    fn insertion_order_preserved() {
        let mut tree = A11yTree::new();
        tree.insert(node(3, Role::Button, Label::None));
        tree.insert(node(1, Role::Link, Label::None));
        tree.insert(node(2, Role::Checkbox, Label::None));

        let keys: Vec<_> = tree.iter().map(|n| n.key.clone()).collect();
        assert_eq!(keys, [Key::Int(3), Key::Int(1), Key::Int(2)]);
    }

    #[test]
    fn insert_replaces_in_place() {
        let mut tree = A11yTree::new();
        tree.insert(node(1, Role::Button, Label::text("a")));
        tree.insert(node(2, Role::Button, Label::text("b")));
        let prev = tree.insert(node(1, Role::Link, Label::text("c")));
        assert_eq!(prev.unwrap().role, Role::Button);
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.get(&Key::Int(1)).unwrap().role, Role::Link);
        // Order unchanged: key 1 still first.
        assert_eq!(tree.nodes()[0].key, Key::Int(1));
    }

    #[test]
    fn labelled_by_resolution() {
        let mut tree = A11yTree::new();
        tree.insert(node(1, Role::Checkbox, Label::labelled_by(Key::Int(2))));
        tree.insert(node(2, Role::Presentation, Label::text("Accept terms")));
        assert_eq!(
            tree.label_text(&Key::Int(1)).as_deref(),
            Some("Accept terms")
        );
    }

    #[test]
    fn labelled_by_cycle_is_safe() {
        let mut tree = A11yTree::new();
        tree.insert(node(1, Role::Group, Label::labelled_by(Key::Int(2))));
        tree.insert(node(2, Role::Group, Label::labelled_by(Key::Int(1))));
        assert_eq!(tree.label_text(&Key::Int(1)), None);
    }

    #[test]
    fn missing_or_none_label() {
        let mut tree = A11yTree::new();
        tree.insert(node(1, Role::Button, Label::None));
        assert_eq!(tree.label_text(&Key::Int(1)), None);
        assert_eq!(tree.label_text(&Key::Int(99)), None);
    }

    #[test]
    fn from_iter_builds_in_order() {
        let tree: A11yTree = [
            node(10, Role::Button, Label::None),
            node(20, Role::Link, Label::None),
        ]
        .into_iter()
        .collect();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.nodes()[0].key, Key::Int(10));
    }
}
