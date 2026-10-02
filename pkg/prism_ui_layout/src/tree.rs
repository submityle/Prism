//! Arena-backed storage for a tree of styled boxes and their computed
//! layouts.
//!
//! In addition to the full-solve entry point [`LayoutTree::compute_layout`],
//! the tree tracks parent links, two-level dirty state (see [`crate::dirty`]),
//! a per-leaf measurement cache, and lightweight instrumentation. Those extra
//! facilities power the incremental layout path in [`crate::incremental`]
//! without changing the behavior of the full solver.

use core::cell::{Cell, RefCell};

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::dirty::DirtyFlags;
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

/// A measurement-cache key derived from a leaf's input constraints.
///
/// Floating-point inputs are compared by their exact bit pattern, which makes
/// the key deterministic and hashable while still distinguishing every input
/// the solver can actually produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CacheKey {
    known_w: (u8, u32),
    known_h: (u8, u32),
    avail_w: (u8, u32),
    avail_h: (u8, u32),
}

impl CacheKey {
    fn new(known: Size<Option<f32>>, available: Size<AvailableSpace>) -> Self {
        Self {
            known_w: encode_option(known.width),
            known_h: encode_option(known.height),
            avail_w: encode_available(available.width),
            avail_h: encode_available(available.height),
        }
    }
}

fn encode_option(value: Option<f32>) -> (u8, u32) {
    match value {
        None => (0, 0),
        Some(v) => (1, v.to_bits()),
    }
}

fn encode_available(value: AvailableSpace) -> (u8, u32) {
    match value {
        AvailableSpace::Definite(v) => (0, v.to_bits()),
        AvailableSpace::MinContent => (1, 0),
        AvailableSpace::MaxContent => (2, 0),
    }
}

/// Maximum number of distinct measurement results cached per leaf.
///
/// The flex solver probes a leaf with a bounded set of constraints per pass
/// (definite, min-content, max-content); a small cache absorbs that set while
/// keeping memory bounded as viewports change across frames.
const MEASURE_CACHE_CAPACITY: usize = 8;

/// A cached measurement result together with the key it was produced for.
#[derive(Clone, Copy, Debug)]
struct CacheEntry {
    key: CacheKey,
    size: Size<f32>,
}

/// Internal per-node storage.
struct NodeData {
    style: LayoutStyle,
    children: Vec<NodeId>,
    parent: Option<NodeId>,
    measure: Option<usize>,
    layout: Layout,
    dirty: DirtyFlags,
    measure_cache: RefCell<Vec<CacheEntry>>,
    measure_calls: Cell<u32>,
    last_pass: Cell<u64>,
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
    measure_count: Cell<u32>,
    current_pass: u64,
}

impl LayoutTree {
    /// Creates an empty tree.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            measures: Vec::new(),
            measure_count: Cell::new(0),
            current_pass: 0,
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
        for &child in children {
            self.nodes[child.0].parent = Some(id);
        }
        id
    }

    /// Appends `child` to `parent`'s list of children.
    pub fn add_child(&mut self, parent: NodeId, child: NodeId) {
        self.nodes[parent.0].children.push(child);
        self.nodes[child.0].parent = Some(parent);
    }

    /// Replaces the style of `node`.
    ///
    /// Changing style can change geometry, so this also marks `node` (and its
    /// relayout chain) as needing layout for the incremental path.
    pub fn set_style(&mut self, node: NodeId, style: LayoutStyle) {
        self.nodes[node.0].style = style;
        self.mark_needs_layout(node);
    }

    /// Returns the number of direct children of `node`.
    pub fn child_count(&self, node: NodeId) -> usize {
        self.nodes[node.0].children.len()
    }

    /// Returns the parent of `node`, or `None` when `node` is a root.
    pub fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.nodes[node.0].parent
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
    ///
    /// This always performs a full solve and clears the geometry dirty bit of
    /// every node in the tree.
    pub fn compute_layout(&mut self, root: NodeId, available_space: Size<AvailableSpace>) {
        flex::compute_root(self, root, available_space);
        self.clear_all_needs_layout();
    }

    // --- Incremental support (see `crate::incremental`). ------------------

    /// Returns the current dirty flags of `node`.
    pub fn dirty_flags(&self, node: NodeId) -> DirtyFlags {
        self.nodes[node.0].dirty
    }

    /// Marks `node` as needing a geometry recomputation and propagates the
    /// request up the parent chain, stopping at the nearest relayout
    /// boundary (inclusive). See [`crate::dirty::is_relayout_boundary`].
    pub fn mark_needs_layout(&mut self, node: NodeId) {
        let mut cursor = Some(node);
        while let Some(current) = cursor {
            let data = &mut self.nodes[current.0];
            data.dirty.mark_needs_layout();
            data.measure_cache.borrow_mut().clear();
            if crate::dirty::is_relayout_boundary(&data.style) {
                break;
            }
            cursor = data.parent;
        }
    }

    /// Marks `node` as needing a repaint without invalidating its geometry.
    pub fn mark_needs_paint(&mut self, node: NodeId) {
        self.nodes[node.0].dirty.mark_needs_paint();
    }

    /// Clears the paint bit of `node` after a repaint.
    pub fn clear_needs_paint(&mut self, node: NodeId) {
        self.nodes[node.0].dirty.clear_needs_paint();
    }

    /// Returns the number of leaf measurements that actually executed (cache
    /// misses) over the lifetime of the tree.
    pub fn measure_count(&self) -> u32 {
        self.measure_count.get()
    }

    /// Returns how many times `node`'s [`Measure`] has actually run (cache
    /// misses only).
    pub fn measure_calls(&self, node: NodeId) -> u32 {
        self.nodes[node.0].measure_calls.get()
    }

    /// Returns `true` when `node` had its layout rewritten during the most
    /// recent incremental pass.
    pub fn was_relayouted(&self, node: NodeId) -> bool {
        self.current_pass != 0 && self.nodes[node.0].last_pass.get() == self.current_pass
    }

    /// Returns the root-relative list of children of `node`.
    pub(crate) fn children(&self, node: NodeId) -> Vec<NodeId> {
        self.nodes[node.0].children.clone()
    }

    pub(crate) fn style(&self, node: NodeId) -> &LayoutStyle {
        &self.nodes[node.0].style
    }

    pub(crate) fn measure_of(
        &self,
        node: NodeId,
        known: Size<Option<f32>>,
        available: Size<AvailableSpace>,
    ) -> Option<Size<f32>> {
        let data = &self.nodes[node.0];
        let measure_id = data.measure?;
        let key = CacheKey::new(known, available);
        if let Some(size) = data.measure_cache.borrow().iter().find_map(|entry| {
            if entry.key == key {
                Some(entry.size)
            } else {
                None
            }
        }) {
            return Some(size);
        }
        let size = self.measures[measure_id].measure(known, available);
        data.measure_calls.set(data.measure_calls.get() + 1);
        self.measure_count.set(self.measure_count.get() + 1);
        let mut cache = data.measure_cache.borrow_mut();
        if cache.len() >= MEASURE_CACHE_CAPACITY {
            cache.remove(0);
        }
        cache.push(CacheEntry { key, size });
        Some(size)
    }

    pub(crate) fn set_layout(&mut self, node: NodeId, layout: Layout) {
        let data = &mut self.nodes[node.0];
        data.layout = layout;
        data.last_pass.set(self.current_pass);
    }

    /// Begins a new incremental pass, returning the pass id used to stamp
    /// touched nodes.
    pub(crate) fn begin_pass(&mut self) -> u64 {
        self.current_pass += 1;
        self.current_pass
    }

    /// Clears the geometry dirty bit of every node.
    pub(crate) fn clear_all_needs_layout(&mut self) {
        for data in &mut self.nodes {
            data.dirty.clear_needs_layout();
        }
    }

    /// Returns the handles of every node, in creation order.
    pub(crate) fn node_ids(&self) -> Vec<NodeId> {
        (0..self.nodes.len()).map(NodeId).collect()
    }

    /// Saves then restores the location/order of a subtree root around an
    /// isolated re-solve, which otherwise zeroes them.
    pub(crate) fn resolve_subtree(&mut self, root: NodeId, available: Size<AvailableSpace>) {
        let saved = self.nodes[root.0].layout;
        flex::compute_root(self, root, available);
        let data = &mut self.nodes[root.0];
        data.layout.location = saved.location;
        data.layout.order = saved.order;
    }

    fn push(&mut self, style: LayoutStyle, measure: Option<usize>) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(NodeData {
            style,
            children: Vec::new(),
            parent: None,
            measure,
            layout: Layout::ZERO,
            dirty: DirtyFlags::DIRTY,
            measure_cache: RefCell::new(Vec::new()),
            measure_calls: Cell::new(0),
            last_pass: Cell::new(0),
        });
        id
    }
}
