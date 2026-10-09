//! A retained scene that absorbs [`BackendOp`]s and lowers them to a
//! [`DrawList`].
//!
//! [`RetainedScene`] is the stateful bridge between the `prism_ui` runtime and
//! a renderer. The runtime emits a *minimal* op stream (create/remove/relayout
//! /repaint/reorder); the scene keeps just enough retained state to turn the
//! latest tree into an absolute, paint-ordered draw list on demand. This keeps
//! the engine's "cost ∝ change" contract: ops are O(1) to apply, and only a
//! `to_draw_list` call walks the tree.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::{Backend, BackendId, BackendOp, ElementKind, PaintStyle};
use prism_ui_layout::{Point, Rect, Size};

use crate::draw::{DrawCommand, DrawList, GlyphCmd, RectCmd, ShadowCmd};

/// One materialised node in the retained scene.
#[derive(Clone, Debug, PartialEq)]
struct Node {
    kind: ElementKind,
    parent: Option<BackendId>,
    children: Vec<BackendId>,
    location: Point<f32>,
    size: Size<f32>,
    text: String,
    paint: PaintStyle,
}

impl Node {
    fn new(kind: ElementKind, parent: Option<BackendId>) -> Self {
        Self {
            kind,
            parent,
            children: Vec::new(),
            location: Point::ZERO,
            size: Size::ZERO,
            text: String::new(),
            paint: PaintStyle::default(),
        }
    }
}

/// Retained render state driven by a stream of [`BackendOp`]s.
///
/// Nodes are stored in a [`BTreeMap`] keyed by id so traversal order is
/// deterministic and independent of allocation. Child order is tracked
/// explicitly and honoured by [`BackendOp::Reorder`].
#[derive(Clone, Debug, Default)]
pub struct RetainedScene {
    nodes: BTreeMap<u64, Node>,
    root: Option<BackendId>,
}

impl RetainedScene {
    /// Creates an empty scene.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            root: None,
        }
    }

    /// The root node id, if one has been created.
    #[must_use]
    pub fn root(&self) -> Option<BackendId> {
        self.root
    }

    /// Number of live nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the scene has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Lowers the current tree into an absolute, paint-ordered [`DrawList`].
    ///
    /// Traversal is pre-order (parent painted before children) following each
    /// node's explicit child order, which is exactly back-to-front paint order.
    /// Absolute positions are accumulated from the relative per-node layout, and
    /// each node's `opacity` is folded into every command it emits.
    #[must_use]
    pub fn to_draw_list(&self) -> DrawList {
        let mut list = DrawList::new();
        if let Some(root) = self.root {
            self.emit(root, Point::ZERO, 1.0, &mut list);
        }
        list
    }

    fn emit(&self, id: BackendId, origin: Point<f32>, inherited_opacity: f32, list: &mut DrawList) {
        let Some(node) = self.nodes.get(&id.0) else {
            return;
        };
        let abs = Point::new(origin.x + node.location.x, origin.y + node.location.y);
        let opacity = clamp01(inherited_opacity * node.paint.opacity);
        let rect = Rect::new(abs, node.size);

        match &node.kind {
            ElementKind::Text => {
                if let Some(color) = node.paint.color
                    && !node.text.is_empty()
                {
                    // Intern the run's characters so the rasteriser can draw the
                    // real glyphs; the command keeps only a compact handle.
                    let text = list.intern(&node.text);
                    list.push(DrawCommand::Glyph(GlyphCmd {
                        rect,
                        color,
                        size: node.paint.font_size,
                        opacity,
                        text,
                    }));
                }
            }
            ElementKind::Box | ElementKind::Custom(_) => {
                let radius = node.paint.border_radius;
                let has_border =
                    node.paint.border_width > 0.0 && node.paint.border_color.is_some();

                // Outer drop shadow first, so it sits behind the box. Inset
                // shadows are a hint the reference backend does not render.
                if let Some(sh) = node.paint.shadow
                    && !sh.inset
                {
                    list.push(DrawCommand::Shadow(ShadowCmd {
                        rect,
                        radius,
                        blur: sh.blur,
                        offset: Point::new(sh.offset_x, sh.offset_y),
                        color: sh.color,
                        opacity,
                    }));
                }

                if let Some(glass) = node.paint.glass {
                    // Approximate frosted glass: an optional opaque base keeps
                    // the content behind from fully bleeding through, then a
                    // translucent tint reads as frost, then an optional bright
                    // rim as the lit glass edge. `glass.blur` is a GPU hint and
                    // is ignored by the reference rasteriser.
                    if let Some(base) = node.paint.background_color {
                        list.push(DrawCommand::Rect(RectCmd {
                            rect,
                            fill: Some(base),
                            radius,
                            border_width: 0.0,
                            border_color: None,
                            opacity,
                        }));
                    }
                    list.push(DrawCommand::Rect(RectCmd {
                        rect,
                        fill: Some(glass.tint),
                        radius,
                        border_width: 0.0,
                        border_color: None,
                        opacity,
                    }));
                    if let Some(highlight) = glass.highlight {
                        list.push(DrawCommand::Rect(RectCmd {
                            rect,
                            fill: None,
                            radius,
                            border_width: node.paint.border_width.max(1.0),
                            border_color: Some(highlight),
                            opacity,
                        }));
                    } else if has_border {
                        list.push(DrawCommand::Rect(RectCmd {
                            rect,
                            fill: None,
                            radius,
                            border_width: node.paint.border_width,
                            border_color: node.paint.border_color,
                            opacity,
                        }));
                    }
                } else {
                    let has_fill = node.paint.background_color.is_some();
                    if has_fill || has_border {
                        list.push(DrawCommand::Rect(RectCmd {
                            rect,
                            fill: node.paint.background_color,
                            radius,
                            border_width: if has_border {
                                node.paint.border_width
                            } else {
                                0.0
                            },
                            border_color: node.paint.border_color,
                            opacity,
                        }));
                    }
                }
            }
        }

        // Content coordinates are relative to this node's own box.
        for &child in &node.children {
            self.emit(child, abs, opacity, list);
        }
    }

    fn detach_from_parent(&mut self, id: BackendId) {
        let parent = self.nodes.get(&id.0).and_then(|n| n.parent);
        if let Some(p) = parent
            && let Some(pn) = self.nodes.get_mut(&p.0)
        {
            pn.children.retain(|c| *c != id);
        }
    }

    fn remove_recursive(&mut self, id: BackendId) {
        let children = self
            .nodes
            .get(&id.0)
            .map(|n| n.children.clone())
            .unwrap_or_default();
        for child in children {
            self.remove_recursive(child);
        }
        self.nodes.remove(&id.0);
        if self.root == Some(id) {
            self.root = None;
        }
    }
}

impl Backend for RetainedScene {
    fn apply(&mut self, op: BackendOp) {
        match op {
            BackendOp::Create {
                id,
                kind,
                parent,
                index,
            } => {
                self.nodes.insert(id.0, Node::new(kind, parent));
                match parent {
                    None => self.root = Some(id),
                    Some(p) => {
                        if let Some(pn) = self.nodes.get_mut(&p.0) {
                            let at = index.min(pn.children.len());
                            pn.children.insert(at, id);
                        }
                    }
                }
            }
            BackendOp::SetText { id, text } => {
                if let Some(n) = self.nodes.get_mut(&id.0) {
                    n.text = text;
                }
            }
            BackendOp::SetLayout { id, location, size } => {
                if let Some(n) = self.nodes.get_mut(&id.0) {
                    n.location = location;
                    n.size = size;
                }
            }
            BackendOp::SetPaint { id, paint } => {
                if let Some(n) = self.nodes.get_mut(&id.0) {
                    n.paint = paint;
                }
            }
            BackendOp::Remove { id } => {
                self.detach_from_parent(id);
                self.remove_recursive(id);
            }
            BackendOp::Reorder { parent, order } => {
                if let Some(pn) = self.nodes.get_mut(&parent.0) {
                    pn.children = order;
                }
            }
        }
    }
}

#[inline]
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::Color;

    fn paint_bg(c: Color) -> PaintStyle {
        PaintStyle {
            background_color: Some(c),
            ..PaintStyle::default()
        }
    }

    fn create(
        scene: &mut RetainedScene,
        id: u64,
        kind: ElementKind,
        parent: Option<u64>,
        index: usize,
    ) {
        scene.apply(BackendOp::Create {
            id: BackendId(id),
            kind,
            parent: parent.map(BackendId),
            index,
        });
    }

    #[test]
    fn builds_tree_and_root() {
        let mut s = RetainedScene::new();
        assert!(s.is_empty());
        create(&mut s, 0, ElementKind::Box, None, 0);
        create(&mut s, 1, ElementKind::Box, Some(0), 0);
        assert_eq!(s.root(), Some(BackendId(0)));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn absolute_positions_accumulate() {
        let mut s = RetainedScene::new();
        create(&mut s, 0, ElementKind::Box, None, 0);
        create(&mut s, 1, ElementKind::Box, Some(0), 0);
        s.apply(BackendOp::SetLayout {
            id: BackendId(0),
            location: Point::new(10.0, 10.0),
            size: Size::new(100.0, 100.0),
        });
        s.apply(BackendOp::SetLayout {
            id: BackendId(1),
            location: Point::new(5.0, 5.0),
            size: Size::new(20.0, 20.0),
        });
        s.apply(BackendOp::SetPaint {
            id: BackendId(0),
            paint: paint_bg(Color::rgba(1.0, 0.0, 0.0, 1.0)),
        });
        s.apply(BackendOp::SetPaint {
            id: BackendId(1),
            paint: paint_bg(Color::rgba(0.0, 1.0, 0.0, 1.0)),
        });
        let dl = s.to_draw_list();
        assert_eq!(dl.primitive_count(), 2);
        if let DrawCommand::Rect(child) = dl.commands()[1] {
            assert!((child.rect.left() - 15.0).abs() < 1e-6);
            assert!((child.rect.top() - 15.0).abs() < 1e-6);
        } else {
            panic!("expected child rect second");
        }
    }

    #[test]
    fn opacity_folds_through_ancestors() {
        let mut s = RetainedScene::new();
        create(&mut s, 0, ElementKind::Box, None, 0);
        create(&mut s, 1, ElementKind::Box, Some(0), 0);
        let mut half = paint_bg(Color::rgba(1.0, 1.0, 1.0, 1.0));
        half.opacity = 0.5;
        s.apply(BackendOp::SetPaint {
            id: BackendId(0),
            paint: half,
        });
        s.apply(BackendOp::SetPaint {
            id: BackendId(1),
            paint: half,
        });
        s.apply(BackendOp::SetLayout {
            id: BackendId(0),
            location: Point::ZERO,
            size: Size::new(10.0, 10.0),
        });
        s.apply(BackendOp::SetLayout {
            id: BackendId(1),
            location: Point::ZERO,
            size: Size::new(10.0, 10.0),
        });
        let dl = s.to_draw_list();
        if let DrawCommand::Rect(child) = dl.commands()[1] {
            assert!((child.opacity - 0.25).abs() < 1e-6, "0.5 * 0.5 folded");
        } else {
            panic!("expected child rect");
        }
    }

    #[test]
    fn remove_is_recursive_and_detaches() {
        let mut s = RetainedScene::new();
        create(&mut s, 0, ElementKind::Box, None, 0);
        create(&mut s, 1, ElementKind::Box, Some(0), 0);
        create(&mut s, 2, ElementKind::Box, Some(1), 0);
        s.apply(BackendOp::Remove { id: BackendId(1) });
        assert_eq!(s.len(), 1);
        assert_eq!(s.root(), Some(BackendId(0)));
    }

    #[test]
    fn reorder_changes_paint_order() {
        let mut s = RetainedScene::new();
        create(&mut s, 0, ElementKind::Box, None, 0);
        create(&mut s, 1, ElementKind::Box, Some(0), 0);
        create(&mut s, 2, ElementKind::Box, Some(0), 1);
        for id in [0, 1, 2] {
            s.apply(BackendOp::SetPaint {
                id: BackendId(id),
                paint: paint_bg(Color::rgba(1.0, 1.0, 1.0, 1.0)),
            });
            s.apply(BackendOp::SetLayout {
                id: BackendId(id),
                location: Point::ZERO,
                size: Size::new(10.0, 10.0),
            });
        }
        s.apply(BackendOp::Reorder {
            parent: BackendId(0),
            order: alloc::vec![BackendId(2), BackendId(1)],
        });
        let dl = s.to_draw_list();
        // root, then child 2, then child 1.
        assert_eq!(dl.primitive_count(), 3);
    }

    #[test]
    fn text_requires_color_and_content() {
        let mut s = RetainedScene::new();
        create(&mut s, 0, ElementKind::Text, None, 0);
        s.apply(BackendOp::SetLayout {
            id: BackendId(0),
            location: Point::ZERO,
            size: Size::new(50.0, 16.0),
        });
        // No colour, no text -> nothing emitted.
        assert_eq!(s.to_draw_list().primitive_count(), 0);
        s.apply(BackendOp::SetText {
            id: BackendId(0),
            text: String::from("hi"),
        });
        let p = PaintStyle {
            color: Some(Color::rgba(0.0, 0.0, 0.0, 1.0)),
            ..PaintStyle::default()
        };
        s.apply(BackendOp::SetPaint {
            id: BackendId(0),
            paint: p,
        });
        assert_eq!(s.to_draw_list().primitive_count(), 1);
    }

    #[test]
    fn empty_box_emits_nothing() {
        let mut s = RetainedScene::new();
        create(&mut s, 0, ElementKind::Box, None, 0);
        s.apply(BackendOp::SetLayout {
            id: BackendId(0),
            location: Point::ZERO,
            size: Size::new(10.0, 10.0),
        });
        assert_eq!(s.to_draw_list().primitive_count(), 0);
    }

    #[test]
    fn shadow_is_emitted_behind_fill() {
        use prism_ui::Shadow;
        let mut s = RetainedScene::new();
        create(&mut s, 0, ElementKind::Box, None, 0);
        s.apply(BackendOp::SetLayout {
            id: BackendId(0),
            location: Point::ZERO,
            size: Size::new(40.0, 20.0),
        });
        let paint = PaintStyle {
            background_color: Some(Color::rgba(1.0, 1.0, 1.0, 1.0)),
            shadow: Some(Shadow::drop(0.0, 8.0, 24.0, Color::rgba(0.0, 0.0, 0.0, 0.3))),
            ..PaintStyle::default()
        };
        s.apply(BackendOp::SetPaint {
            id: BackendId(0),
            paint,
        });
        let dl = s.to_draw_list();
        assert_eq!(dl.primitive_count(), 2);
        // Shadow must come first (painted behind the fill).
        assert!(matches!(dl.commands()[0], DrawCommand::Shadow(_)));
        assert!(matches!(dl.commands()[1], DrawCommand::Rect(_)));
    }

    #[test]
    fn inset_shadow_is_not_rendered_by_reference_backend() {
        use prism_ui::Shadow;
        let mut s = RetainedScene::new();
        create(&mut s, 0, ElementKind::Box, None, 0);
        s.apply(BackendOp::SetLayout {
            id: BackendId(0),
            location: Point::ZERO,
            size: Size::new(40.0, 20.0),
        });
        let paint = PaintStyle {
            background_color: Some(Color::rgba(1.0, 1.0, 1.0, 1.0)),
            shadow: Some(Shadow::drop(0.0, 4.0, 8.0, Color::rgba(0.0, 0.0, 0.0, 0.3)).inset()),
            ..PaintStyle::default()
        };
        s.apply(BackendOp::SetPaint {
            id: BackendId(0),
            paint,
        });
        let dl = s.to_draw_list();
        // Only the fill rect, no shadow command.
        assert_eq!(dl.primitive_count(), 1);
        assert!(matches!(dl.commands()[0], DrawCommand::Rect(_)));
    }

    #[test]
    fn glass_lowers_to_shadow_tint_and_highlight() {
        use prism_ui::{Glass, Shadow};
        let mut s = RetainedScene::new();
        create(&mut s, 0, ElementKind::Box, None, 0);
        s.apply(BackendOp::SetLayout {
            id: BackendId(0),
            location: Point::ZERO,
            size: Size::new(60.0, 30.0),
        });
        let tint = Color::rgba8(255, 255, 255, 160);
        let highlight = Color::rgba8(255, 255, 255, 115);
        let paint = PaintStyle::glass_surface(
            18.0,
            20.0,
            tint,
            highlight,
            Shadow::drop(0.0, 8.0, 24.0, Color::rgba8(0, 0, 0, 38)),
        );
        let _ = Glass::new(20.0, tint); // smoke
        s.apply(BackendOp::SetPaint {
            id: BackendId(0),
            paint,
        });
        let dl = s.to_draw_list();
        // shadow + translucent tint rect + highlight rim rect = 3 primitives.
        assert_eq!(dl.primitive_count(), 3);
        assert!(matches!(dl.commands()[0], DrawCommand::Shadow(_)));

        // The tint fill must be translucent (0 < a < 1) after lowering.
        match dl.commands()[1] {
            DrawCommand::Rect(r) => {
                let fill = r.fill.expect("glass tint fill present");
                assert!(fill.a > 0.0 && fill.a < 1.0, "glass tint is translucent");
                assert_eq!(r.border_width, 0.0, "tint layer has no border");
            }
            _ => panic!("expected tint rect second"),
        }
        // The highlight rim is a border-only rect.
        match dl.commands()[2] {
            DrawCommand::Rect(r) => {
                assert!(r.fill.is_none(), "highlight rim has no fill");
                assert!(r.border_width >= 1.0, "highlight rim is stroked");
                assert!(r.border_color.is_some());
            }
            _ => panic!("expected highlight rim third"),
        }
    }

    #[test]
    fn glass_over_opaque_base_adds_base_layer() {
        use prism_ui::{Glass, Shadow};
        let mut s = RetainedScene::new();
        create(&mut s, 0, ElementKind::Box, None, 0);
        s.apply(BackendOp::SetLayout {
            id: BackendId(0),
            location: Point::ZERO,
            size: Size::new(60.0, 30.0),
        });
        let paint = PaintStyle {
            background_color: Some(Color::rgba8(10, 10, 10, 255)),
            shadow: Some(Shadow::drop(0.0, 8.0, 24.0, Color::rgba8(0, 0, 0, 60))),
            glass: Some(
                Glass::new(20.0, Color::rgba8(255, 255, 255, 140))
                    .with_highlight(Color::rgba8(255, 255, 255, 90)),
            ),
            ..PaintStyle::default()
        };
        s.apply(BackendOp::SetPaint {
            id: BackendId(0),
            paint,
        });
        let dl = s.to_draw_list();
        // shadow + opaque base + translucent tint + highlight rim = 4.
        assert_eq!(dl.primitive_count(), 4);
        assert!(matches!(dl.commands()[0], DrawCommand::Shadow(_)));
        match dl.commands()[1] {
            DrawCommand::Rect(r) => {
                assert_eq!(r.fill.unwrap().a, 1.0, "opaque base first");
            }
            _ => panic!("expected opaque base rect"),
        }
    }
}
