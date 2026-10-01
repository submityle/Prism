//! The view runtime: the heart of Loom.
//!
//! [`Ui`] owns the retained state that lets an [`Element`] tree be *reconciled*
//! rather than rebuilt. Each frame the application hands the runtime a freshly
//! built, cheap [`Element`] describing the desired UI; the runtime diffs it
//! against its retained [`Tree`] and emits only the [`BackendOp`]s needed to
//! close the gap. This is Loom's central contract: **cost is proportional to
//! what changed, not to the size of the scene.**
//!
//! The runtime also owns the three cross-cutting services a UI needs:
//!
//! * a [`Runtime`] reactive graph, so application state can drive updates;
//! * a style [`StyleSheet`] + [`Theme`] cascade, resolved per node;
//! * a flexbox [`LayoutTree`], solved on demand.
//!
//! # Reconciliation
//!
//! Children are matched by [`Key`]. Explicitly keyed children survive moves and
//! insertions; keyless children fall back to a stable positional key. The keyed
//! diff reuses nodes with the minimum number of moves (see
//! [`prism_ui_tree`]), so list reorders never destroy and rebuild item state.

use alloc::borrow::ToOwned;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use prism_ui_layout::{AvailableSpace, Layout, LayoutTree, Measure, NodeId as LayoutId, Size};
use prism_ui_style::{
    resolve, ComputedStyle, MatchContext, StyleProp, StyleSheet, StyleValue, Theme,
};
use prism_ui_tree::reactive::Runtime;
use prism_ui_tree::{diff_keyed, DiffOp, NodeId, Tree};

use crate::backend::{Backend, BackendId, BackendOp};
use crate::element::{Element, ElementKind, Key};
use crate::paint::PaintStyle;
use crate::style_map::build_styles;

/// The retained per-node state the runtime keeps between frames.
///
/// Everything here is the *last applied* value, so a new frame can diff against
/// it field by field and emit only the ops that actually changed.
#[derive(Clone, Debug)]
struct Realized {
    backend_id: BackendId,
    kind: ElementKind,
    classes: Vec<String>,
    inline: Vec<(StyleProp, StyleValue)>,
    text: Option<String>,
    layout_style: prism_ui_layout::LayoutStyle,
    paint: PaintStyle,
    last_layout: Option<Layout>,
}

/// Measures a text run from its length and font size.
///
/// This is a deliberately simple monospace-ish model: it avoids any floating
/// point transcendental functions (banned by the workspace lints) and gives
/// layout a non-degenerate content size to work with. A real backend can
/// provide a precise [`Measure`] later.
struct TextMeasure {
    chars: f32,
    font_size: f32,
}

impl Measure for TextMeasure {
    fn measure(&self, known: Size<Option<f32>>, _available: Size<AvailableSpace>) -> Size<f32> {
        let width = known.width.unwrap_or(self.chars * self.font_size * 0.5);
        let height = known.height.unwrap_or(self.font_size * 1.2);
        Size::new(width, height)
    }
}

/// The declarative UI runtime.
///
/// Construct it with a [`Backend`], optionally supply a [`Theme`] and
/// [`StyleSheet`], then drive it with [`Ui::mount`] and [`Ui::update`], solving
/// geometry with [`Ui::compute_layout`].
pub struct Ui<B: Backend> {
    runtime: Runtime,
    theme: Theme,
    sheet: StyleSheet,
    viewport_width: f32,
    tree: Tree<Key, Realized>,
    root: Option<NodeId>,
    backend: B,
    next_id: u64,
}

impl<B: Backend> Ui<B> {
    /// Creates a runtime over `backend` with the default palette and an empty
    /// style sheet.
    #[must_use]
    pub fn new(backend: B) -> Self {
        Self {
            runtime: Runtime::new(),
            theme: Theme::with_default_palette(),
            sheet: StyleSheet::new(),
            viewport_width: 0.0,
            tree: Tree::new(),
            root: None,
            backend,
            next_id: 0,
        }
    }

    /// Replaces the theme used to resolve design tokens and breakpoints.
    #[must_use]
    pub fn with_theme(mut self, theme: Theme) -> Self {
        self.theme = theme;
        self
    }

    /// Replaces the style sheet classes are resolved against.
    #[must_use]
    pub fn with_stylesheet(mut self, sheet: StyleSheet) -> Self {
        self.sheet = sheet;
        self
    }

    /// Sets the viewport width used for responsive breakpoint matching.
    pub fn set_viewport_width(&mut self, width: f32) {
        self.viewport_width = width;
    }

    /// Borrows the reactive runtime, so callers can create signals/effects that
    /// drive their view functions.
    #[must_use]
    pub fn reactive(&self) -> &Runtime {
        &self.runtime
    }

    /// Borrows the backend (e.g. to read recorded ops in tests).
    #[must_use]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Mutably borrows the backend.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Number of live nodes currently retained.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.tree.len()
    }

    /// Mounts `element` as the root, replacing any previous tree.
    pub fn mount(&mut self, element: &Element) {
        if let Some(root) = self.root.take() {
            self.emit_remove_subtree(root);
            self.tree.remove_subtree(root);
        }
        let root = self.build_node(element, None, 0);
        self.root = Some(root);
    }

    /// Reconciles the retained tree against `element`, emitting the minimal set
    /// of backend ops. Mounts fresh if nothing is mounted yet.
    pub fn update(&mut self, element: &Element) {
        let Some(root) = self.root else {
            self.mount(element);
            return;
        };
        // A root whose kind changed is replaced wholesale.
        let same_kind = self.tree.get(root).is_some_and(|r| r.kind == element.kind);
        if same_kind {
            self.reconcile_node_props(root, element);
            self.reconcile_children(root, element.child_elements());
        } else {
            self.mount(element);
        }
    }

    /// Solves layout for the whole tree given the space available to the root,
    /// then emits a [`BackendOp::SetLayout`] for every node whose geometry
    /// changed since the last solve.
    pub fn compute_layout(&mut self, available: Size<AvailableSpace>) {
        let Some(root) = self.root else {
            return;
        };

        // Build a fresh layout tree mirroring the retained tree. Layout is
        // inherently a whole-tree solve, but we still only *emit* ops for nodes
        // whose result changed.
        let mut layout = LayoutTree::new();
        let mut pairs: Vec<(NodeId, LayoutId)> = Vec::new();
        let layout_root = self.build_layout(&mut layout, root, &mut pairs);
        layout.compute_layout(layout_root, available);

        for (node, lid) in pairs {
            let result = *layout.layout(lid);
            let (backend_id, changed) = {
                let r = self
                    .tree
                    .get_mut(node)
                    .expect("retained node present during layout");
                let changed = r.last_layout != Some(result);
                if changed {
                    r.last_layout = Some(result);
                }
                (r.backend_id, changed)
            };
            if changed {
                self.backend.apply(BackendOp::SetLayout {
                    id: backend_id,
                    location: result.location,
                    size: result.size,
                });
            }
        }
    }

    // --- Internal: construction ------------------------------------------

    fn alloc_id(&mut self) -> BackendId {
        let id = BackendId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Materialises `element` and its subtree, emitting create/text/paint ops.
    /// The returned node is detached; the caller attaches it to its parent.
    fn build_node(
        &mut self,
        element: &Element,
        parent_backend: Option<BackendId>,
        index: usize,
    ) -> NodeId {
        let backend_id = self.alloc_id();
        let (layout_style, paint) = self.compute_styles(element);
        let key = element.resolved_key(index);
        let realized = Realized {
            backend_id,
            kind: element.kind().clone(),
            classes: element.class_names().to_vec(),
            inline: element.inline_pairs().to_vec(),
            text: element.text_content().map(ToOwned::to_owned),
            layout_style,
            paint,
            last_layout: None,
        };
        let node = self.tree.create(Some(key), realized);

        self.backend.apply(BackendOp::Create {
            id: backend_id,
            kind: element.kind().clone(),
            parent: parent_backend,
            index,
        });
        if let Some(text) = element.text_content() {
            self.backend.apply(BackendOp::SetText {
                id: backend_id,
                text: text.to_owned(),
            });
        }
        self.backend.apply(BackendOp::SetPaint {
            id: backend_id,
            paint,
        });

        for (i, child) in element.child_elements().iter().enumerate() {
            let child_node = self.build_node(child, Some(backend_id), i);
            self.tree.append_child(node, child_node);
        }
        node
    }

    // --- Internal: reconciliation ----------------------------------------

    fn reconcile_node_props(&mut self, node: NodeId, element: &Element) {
        let (layout_style, paint) = self.compute_styles(element);

        let mut emit_text: Option<String> = None;
        let mut emit_paint = false;
        let backend_id;
        {
            let r = self
                .tree
                .get_mut(node)
                .expect("reconciled node must be live");
            backend_id = r.backend_id;

            let new_text = element.text_content().map(ToOwned::to_owned);
            if r.text != new_text {
                r.text = new_text.clone();
                if let Some(text) = new_text {
                    emit_text = Some(text);
                }
            }
            if r.paint != paint {
                r.paint = paint;
                emit_paint = true;
            }
            r.layout_style = layout_style;
            r.classes = element.class_names().to_vec();
            r.inline = element.inline_pairs().to_vec();
        }

        if let Some(text) = emit_text {
            self.backend.apply(BackendOp::SetText {
                id: backend_id,
                text,
            });
        }
        if emit_paint {
            self.backend.apply(BackendOp::SetPaint {
                id: backend_id,
                paint,
            });
        }
    }

    fn reconcile_children(&mut self, parent: NodeId, new_elements: &[Element]) {
        let parent_backend = self
            .tree
            .get(parent)
            .expect("parent must be live")
            .backend_id;

        let old_children: Vec<NodeId> = self.tree.children(parent).to_vec();
        let old_keys: Vec<Key> = old_children
            .iter()
            .map(|&c| {
                self.tree
                    .node(c)
                    .and_then(|n| n.key().cloned())
                    .unwrap_or(Key::Index(0))
            })
            .collect();
        let old_backend_order: Vec<BackendId> = old_children
            .iter()
            .map(|&c| self.tree.get(c).expect("live child").backend_id)
            .collect();

        let new_keys: Vec<Key> = new_elements
            .iter()
            .enumerate()
            .map(|(i, e)| e.resolved_key(i))
            .collect();

        let diff = diff_keyed(&old_keys, &new_keys);

        // Resolve each new slot to a retained node, reusing where the key *and*
        // kind match, otherwise replacing.
        let mut new_child_ids: Vec<(NodeId, bool)> = Vec::with_capacity(new_elements.len());
        let mut to_remove: Vec<NodeId> = Vec::new();
        for (i, op) in diff.ops.iter().enumerate() {
            match *op {
                DiffOp::Keep { old_index } | DiffOp::Move { old_index } => {
                    let child = old_children[old_index];
                    let same_kind = self
                        .tree
                        .get(child)
                        .is_some_and(|r| r.kind == new_elements[i].kind);
                    if same_kind {
                        new_child_ids.push((child, true));
                    } else {
                        to_remove.push(child);
                        let fresh = self.build_node(&new_elements[i], Some(parent_backend), i);
                        new_child_ids.push((fresh, false));
                    }
                }
                DiffOp::Create { new_index } => {
                    let fresh = self.build_node(&new_elements[new_index], Some(parent_backend), i);
                    new_child_ids.push((fresh, false));
                }
            }
        }

        for &old_index in &diff.removals {
            to_remove.push(old_children[old_index]);
        }
        for child in to_remove {
            self.emit_remove_subtree(child);
            self.tree.remove_subtree(child);
        }

        // Re-attach children in their new order. `append_child` detaches first,
        // so iterating the new order rebuilds the child list exactly.
        for &(id, _) in &new_child_ids {
            self.tree.append_child(parent, id);
        }

        let new_backend_order: Vec<BackendId> = new_child_ids
            .iter()
            .map(|&(id, _)| self.tree.get(id).expect("live child").backend_id)
            .collect();
        if new_backend_order != old_backend_order {
            self.backend.apply(BackendOp::Reorder {
                parent: parent_backend,
                order: new_backend_order,
            });
        }

        // Recurse into reused children.
        for (i, &(id, reused)) in new_child_ids.iter().enumerate() {
            if reused {
                self.reconcile_node_props(id, &new_elements[i]);
                self.reconcile_children(id, new_elements[i].child_elements());
            }
        }
    }

    fn emit_remove_subtree(&mut self, root: NodeId) {
        // Collect backend ids in the subtree, then emit removes children-first
        // so a backend can tear down leaves before their parents.
        let mut ids: Vec<BackendId> = Vec::new();
        self.tree.walk(root, |node, _depth| {
            if let Some(r) = self.tree.get(node) {
                ids.push(r.backend_id);
            }
        });
        for id in ids.into_iter().rev() {
            self.backend.apply(BackendOp::Remove { id });
        }
    }

    // --- Internal: style + layout ----------------------------------------

    fn compute_styles(&self, element: &Element) -> (prism_ui_layout::LayoutStyle, PaintStyle) {
        let classes = element.class_names();
        let names: Vec<&str> = classes.iter().map(String::as_str).collect();

        let ctx = MatchContext::new(self.viewport_width);
        let computed = resolve(&self.sheet, &self.theme.tokens, &names, &ctx)
            .unwrap_or_else(|_| ComputedStyle::new());

        let mut map: BTreeMap<StyleProp, StyleValue> = BTreeMap::new();
        for (prop, value) in computed.iter() {
            map.insert(*prop, value.clone());
        }
        for (prop, value) in element.inline_pairs() {
            let resolved = self
                .theme
                .tokens
                .resolve_value(value)
                .unwrap_or_else(|_| value.clone());
            map.insert(*prop, resolved);
        }

        build_styles(&map)
    }

    fn build_layout(
        &self,
        layout: &mut LayoutTree,
        node: NodeId,
        pairs: &mut Vec<(NodeId, LayoutId)>,
    ) -> LayoutId {
        let r = self.tree.get(node).expect("live node for layout");
        let style = r.layout_style.clone();
        let lid = if matches!(r.kind, ElementKind::Text) {
            let text = r.text.as_deref().unwrap_or("");
            let measure = TextMeasure {
                chars: text.chars().count() as f32,
                font_size: r.paint.font_size,
            };
            layout.new_leaf_with_measure(style, measure)
        } else {
            layout.new_leaf(style)
        };
        pairs.push((node, lid));

        let children: Vec<NodeId> = self.tree.children(node).to_vec();
        for child in children {
            let child_lid = self.build_layout(layout, child, pairs);
            layout.add_child(lid, child_lid);
        }
        lid
    }
}
