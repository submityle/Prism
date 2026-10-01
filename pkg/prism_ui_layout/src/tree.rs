//! Arena-backed storage for a tree of styled boxes and their computed
//! layouts.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::flex;
use crate::geometry::{AvailableSpace, Size};
use crate::measure::Measure;
use crate::result::Layout;
use crate::style::LayoutStyle;

/// A handle to a box stored inside a [`LayoutTree`].
///
/// Identifiers are stable for the lifetime of the tree and are only valid for
/// the tree that produced them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    /// Returns the underlying arena index.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Internal per-node storage.
struct NodeData {
    style: LayoutStyle,
    children: Vec<NodeId>,
    measure: Option<usize>,
    layout: Layout,
}

/// A tree of boxes to be laid out.
///
/// Boxes are created with [`LayoutTree::new_leaf`],
/// [`LayoutTree::new_leaf_with_measure`], and [`LayoutTree::new_node`], then
/// laid out with [`LayoutTree::compute_layout`]. Computed results are read
/// back with [`LayoutTree::layout`].
#[derive(Default)]
pub struct LayoutTree {
    nodes: Vec<NodeData>,
    measures: Vec<Box<dyn Measure>>,
}

impl LayoutTree {
    /// Creates an empty tree.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            measures: Vec::new(),
        }
    }

    /// Creates a childless box with the given style and returns its handle.
    pub fn new_leaf(&mut self, style: LayoutStyle) -> NodeId {
        self.push(style, None)
    }

    /// Creates a childless box whose content size is provided by a
    /// [`Measure`] implementation.
    pub fn new_leaf_with_measure(
        &mut self,
        style: LayoutStyle,
        measure: impl Measure + 'static,
    ) -> NodeId {
        let measure_id = self.measures.len();
        self.measures.push(Box::new(measure));
        self.push(style, Some(measure_id))
    }

    /// Creates a box with the given style and children, returning its handle.
    pub fn new_node(&mut self, style: LayoutStyle, children: &[NodeId]) -> NodeId {
        let id = self.push(style, None);
        self.nodes[id.0].children.extend_from_slice(children);
        id
    }

    /// Appends `child` to `parent`'s list of children.
    pub fn add_child(&mut self, parent: NodeId, child: NodeId) {
        self.nodes[parent.0].children.push(child);
    }

    /// Replaces the style of `node`.
    pub fn set_style(&mut self, node: NodeId, style: LayoutStyle) {
        self.nodes[node.0].style = style;
    }

    /// Returns the number of direct children of `node`.
    pub fn child_count(&self, node: NodeId) -> usize {
        self.nodes[node.0].children.len()
    }

    /// Returns the computed layout of `node`.
    ///
    /// The returned value is only meaningful after
    /// [`LayoutTree::compute_layout`] has been called for a tree containing
    /// `node`.
    pub fn layout(&self, node: NodeId) -> &Layout {
        &self.nodes[node.0].layout
    }

    /// Computes layout for the subtree rooted at `root`, given the space
    /// available to it.
    pub fn compute_layout(&mut self, root: NodeId, available_space: Size<AvailableSpace>) {
        flex::compute_root(self, root, available_space);
    }

    // --- Internal accessors used by the layout algorithm. -----------------

    fn push(&mut self, style: LayoutStyle, measure: Option<usize>) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(NodeData {
            style,
            children: Vec::new(),
            measure,
            layout: Layout::ZERO,
        });
        id
    }

    pub(crate) fn style(&self, node: NodeId) -> &LayoutStyle {
        &self.nodes[node.0].style
    }

    pub(crate) fn children(&self, node: NodeId) -> Vec<NodeId> {
        self.nodes[node.0].children.clone()
    }

    pub(crate) fn measure_of(
        &self,
        node: NodeId,
        known: Size<Option<f32>>,
        available: Size<AvailableSpace>,
    ) -> Option<Size<f32>> {
        self.nodes[node.0]
            .measure
            .map(|id| self.measures[id].measure(known, available))
    }

    pub(crate) fn set_layout(&mut self, node: NodeId, layout: Layout) {
        self.nodes[node.0].layout = layout;
    }
}
