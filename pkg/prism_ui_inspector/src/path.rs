//! Addressing nodes inside a [`TreeSnapshot`] by positional path.
//!
//! A [`NodePath`] is a sequence of child indices describing how to walk from the
//! snapshot root down to a node. The empty path denotes the root itself, and
//! each subsequent index selects a child at that position. Paths render in a
//! stable `/0/2/1` form (the lone root renders as `/`) and parse back from the
//! same syntax, which makes them convenient to store in golden tests and logs.

use alloc::vec::Vec;
use core::fmt;
use core::str::FromStr;

use prism_ui_devtools::{SnapshotNode, TreeSnapshot};

/// A positional address of a node within a [`TreeSnapshot`].
///
/// The inner vector lists the child index to follow at each level, starting
/// from the root. An empty vector is the root path.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodePath(
    /// The child indices to follow, outermost first.
    pub Vec<usize>,
);

impl NodePath {
    /// Returns the root path (the empty index sequence).
    #[must_use]
    pub fn root() -> Self {
        Self(Vec::new())
    }

    /// Builds a path from an iterator of child indices.
    #[must_use]
    pub fn from_indices<I: IntoIterator<Item = usize>>(indices: I) -> Self {
        Self(indices.into_iter().collect())
    }

    /// Returns the child indices this path follows, outermost first.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.0
    }

    /// Returns the depth of the addressed node; the root has depth `0`.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` when this path addresses the root node.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }

    /// Appends a child index in place, descending one level.
    pub fn push(&mut self, index: usize) {
        self.0.push(index);
    }

    /// Returns a new path extended by `index`, leaving `self` untouched.
    #[must_use]
    pub fn child(&self, index: usize) -> Self {
        let mut next = self.clone();
        next.0.push(index);
        next
    }

    /// Returns the parent path, or `None` for the root.
    #[must_use]
    pub fn parent(&self) -> Option<Self> {
        if self.0.is_empty() {
            return None;
        }
        let mut parent = self.clone();
        parent.0.pop();
        Some(parent)
    }

    /// Resolves this path against `root`, returning the addressed node.
    ///
    /// Returns `None` when any index along the path is out of range.
    #[must_use]
    pub fn resolve_in<'a>(&self, root: &'a SnapshotNode) -> Option<&'a SnapshotNode> {
        let mut node = root;
        for &index in &self.0 {
            node = node.children.get(index)?;
        }
        Some(node)
    }

    /// Parses a path from its `/0/2/1` textual form.
    ///
    /// # Errors
    ///
    /// Returns [`ParsePathError`] when the input lacks the leading `/`, contains
    /// an empty segment, or contains a segment that is not a base-10 index.
    pub fn parse(input: &str) -> Result<Self, ParsePathError> {
        let rest = input
            .strip_prefix('/')
            .ok_or(ParsePathError::MissingLeadingSlash)?;
        if rest.is_empty() {
            return Ok(Self::root());
        }
        let mut indices = Vec::new();
        for segment in rest.split('/') {
            if segment.is_empty() {
                return Err(ParsePathError::EmptySegment);
            }
            let index = segment
                .parse::<usize>()
                .map_err(|_| ParsePathError::InvalidIndex)?;
            indices.push(index);
        }
        Ok(Self(indices))
    }
}

impl fmt::Display for NodePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("/");
        }
        for index in &self.0 {
            write!(f, "/{index}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for NodePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodePath({self})")
    }
}

impl FromStr for NodePath {
    type Err = ParsePathError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// The reason a [`NodePath`] string failed to parse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParsePathError {
    /// The input did not begin with the required leading `/`.
    MissingLeadingSlash,
    /// A path segment was empty, as produced by a doubled or trailing `/`.
    EmptySegment,
    /// A path segment was not a valid base-10 index.
    InvalidIndex,
}

impl fmt::Display for ParsePathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::MissingLeadingSlash => "path must begin with '/'",
            Self::EmptySegment => "path contains an empty segment",
            Self::InvalidIndex => "path segment is not a valid index",
        };
        f.write_str(message)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParsePathError {}

/// Resolves `path` against the snapshot `tree`.
///
/// Returns `None` when any index along the path is out of range.
#[must_use]
pub fn resolve<'a>(tree: &'a TreeSnapshot, path: &NodePath) -> Option<&'a SnapshotNode> {
    path.resolve_in(&tree.root)
}

/// Resolves `path` against an arbitrary `root` subtree.
///
/// Returns `None` when any index along the path is out of range.
#[must_use]
pub fn resolve_in_node<'a>(root: &'a SnapshotNode, path: &NodePath) -> Option<&'a SnapshotNode> {
    path.resolve_in(root)
}

/// Enumerates every node path in `tree`, in depth-first preorder.
///
/// The first entry is always the root path, followed by each subtree in sibling
/// order, so the sequence is deterministic for a given tree.
#[must_use]
pub fn paths_of(tree: &TreeSnapshot) -> Vec<NodePath> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    collect_paths(&tree.root, &mut current, &mut out);
    out
}

/// Depth-first helper that records the path to `node` and recurses.
fn collect_paths(node: &SnapshotNode, current: &mut Vec<usize>, out: &mut Vec<NodePath>) {
    out.push(NodePath(current.clone()));
    for (index, child) in node.children.iter().enumerate() {
        current.push(index);
        collect_paths(child, current, out);
        current.pop();
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    extern crate alloc;

    use alloc::string::ToString;

    use prism_ui::Element;
    use prism_ui_devtools::snapshot;

    use super::{paths_of, resolve, resolve_in_node, NodePath, ParsePathError};

    fn sample() -> prism_ui_devtools::TreeSnapshot {
        let view = Element::box_()
            .child(
                Element::box_()
                    .child(Element::text("a"))
                    .child(Element::text("b")),
            )
            .child(Element::text("c"));
        snapshot(&view)
    }

    #[test]
    fn root_path_is_empty() {
        let path = NodePath::root();
        assert!(path.is_root());
        assert_eq!(path.depth(), 0);
        assert_eq!(path.indices(), &[] as &[usize]);
    }

    #[test]
    fn display_root_is_slash() {
        assert_eq!(NodePath::root().to_string(), "/");
    }

    #[test]
    fn display_nested() {
        assert_eq!(NodePath::from_indices([0, 2, 1]).to_string(), "/0/2/1");
    }

    #[test]
    fn debug_uses_display_form() {
        assert_eq!(
            alloc::format!("{:?}", NodePath::from_indices([0, 1])),
            "NodePath(/0/1)"
        );
    }

    #[test]
    fn parse_root() {
        assert_eq!("/".parse::<NodePath>().unwrap(), NodePath::root());
    }

    #[test]
    fn parse_nested() {
        assert_eq!(
            "/0/2/1".parse::<NodePath>().unwrap(),
            NodePath::from_indices([0, 2, 1])
        );
    }

    #[test]
    fn parse_requires_leading_slash() {
        assert_eq!(
            NodePath::parse("0/1"),
            Err(ParsePathError::MissingLeadingSlash)
        );
    }

    #[test]
    fn parse_rejects_empty_segment() {
        assert_eq!(NodePath::parse("/0//1"), Err(ParsePathError::EmptySegment));
    }

    #[test]
    fn parse_rejects_non_numeric() {
        assert_eq!(NodePath::parse("/0/x"), Err(ParsePathError::InvalidIndex));
    }

    #[test]
    fn parse_display_round_trip() {
        let path = NodePath::from_indices([3, 0, 7]);
        let text = path.to_string();
        assert_eq!(text.parse::<NodePath>().unwrap(), path);
    }

    #[test]
    fn child_and_parent() {
        let path = NodePath::root().child(0).child(1);
        assert_eq!(path, NodePath::from_indices([0, 1]));
        assert_eq!(path.parent(), Some(NodePath::from_indices([0])));
        assert_eq!(NodePath::root().parent(), None);
    }

    #[test]
    fn resolve_root_and_children() {
        let tree = sample();
        let root = resolve(&tree, &NodePath::root()).unwrap();
        assert_eq!(root.kind, "Box");

        let first = resolve(&tree, &NodePath::from_indices([0])).unwrap();
        assert_eq!(first.children.len(), 2);

        let leaf = resolve(&tree, &NodePath::from_indices([0, 1])).unwrap();
        assert_eq!(leaf.text.as_deref(), Some("b"));
    }

    #[test]
    fn resolve_out_of_range() {
        let tree = sample();
        assert!(resolve(&tree, &NodePath::from_indices([5])).is_none());
        assert!(resolve(&tree, &NodePath::from_indices([0, 9])).is_none());
    }

    #[test]
    fn resolve_in_node_subtree() {
        let tree = sample();
        let subtree = &tree.root.children[0];
        let node = resolve_in_node(subtree, &NodePath::from_indices([1])).unwrap();
        assert_eq!(node.text.as_deref(), Some("b"));
    }

    #[test]
    fn paths_cover_every_node_in_preorder() {
        let tree = sample();
        let paths = paths_of(&tree);
        assert_eq!(paths.len(), tree.node_count());
        assert_eq!(paths[0], NodePath::root());
        let rendered: Vec<_> = paths.iter().map(ToString::to_string).collect();
        assert_eq!(rendered, ["/", "/0", "/0/0", "/0/1", "/1"]);
        for path in &paths {
            assert!(resolve(&tree, path).is_some());
        }
    }
}
