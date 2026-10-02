//! Compositing-layer analysis for the draw stream.
//!
//! Animated or opacity-grouped subtrees are promoted to their own compositing
//! layer so a frame can be *recomposited* without re-recording the primitives
//! inside it. [`LayerTree`] parses the balanced `PushLayer`/`PopLayer` markers
//! in a [`DrawList`] into a nested structure, exposing the effective opacity of
//! every layer (the product of its ancestors) and the primitives it owns.

use alloc::vec::Vec;

use crate::draw::{DrawCommand, DrawList};

/// A node in the compositing tree. The implicit root layer has `opacity == 1`
/// and `bounds == None` (unbounded).
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    /// Group opacity declared for this layer.
    pub opacity: f32,
    /// Leaf primitives recorded directly in this layer, in paint order.
    pub primitives: Vec<DrawCommand>,
    /// Nested child layers, in paint order.
    pub children: Vec<Layer>,
}

impl Layer {
    fn new(opacity: f32) -> Self {
        Self {
            opacity,
            primitives: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Recursively counts leaf primitives in this layer and its descendants.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.primitives.len()
            + self
                .children
                .iter()
                .map(Layer::primitive_count)
                .sum::<usize>()
    }
}

/// The parsed compositing hierarchy of a draw list.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerTree {
    root: Layer,
    balanced: bool,
}

impl LayerTree {
    /// Parses a draw list's layer markers into a tree.
    ///
    /// Unbalanced markers (a `PopLayer` with no matching push, or an unclosed
    /// push) are tolerated: parsing never panics and [`LayerTree::is_balanced`]
    /// reports whether the markers were well-formed.
    #[must_use]
    pub fn parse(list: &DrawList) -> Self {
        let mut stack: Vec<Layer> = alloc::vec![Layer::new(1.0)];
        let mut balanced = true;
        for cmd in list.commands() {
            match *cmd {
                DrawCommand::PushLayer(l) => stack.push(Layer::new(l.opacity)),
                DrawCommand::PopLayer => {
                    if stack.len() > 1 {
                        let done = stack.pop().expect("len > 1 checked");
                        stack
                            .last_mut()
                            .expect("root always present")
                            .children
                            .push(done);
                    } else {
                        balanced = false;
                    }
                }
                other => stack
                    .last_mut()
                    .expect("root always present")
                    .primitives
                    .push(other),
            }
        }
        // Any layers left open above the root were never closed.
        while stack.len() > 1 {
            balanced = false;
            let done = stack.pop().expect("len > 1 checked");
            stack
                .last_mut()
                .expect("root always present")
                .children
                .push(done);
        }
        let root = stack.pop().expect("root always present");
        Self { root, balanced }
    }

    /// The implicit root layer.
    #[must_use]
    pub fn root(&self) -> &Layer {
        &self.root
    }

    /// Whether every `PushLayer` had a matching `PopLayer`.
    #[must_use]
    pub fn is_balanced(&self) -> bool {
        self.balanced
    }

    /// Number of explicit (non-root) layers.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        fn count(l: &Layer) -> usize {
            l.children.len() + l.children.iter().map(count).sum::<usize>()
        }
        count(&self.root)
    }

    /// Flattens the tree back into a paint-ordered draw list, folding each
    /// layer's group opacity into the primitives it contains.
    ///
    /// This is the CPU reference for "composite the layers": a renderer that
    /// draws every layer to its own target and blends by group opacity must
    /// produce the same result as drawing these opacity-folded primitives
    /// directly.
    #[must_use]
    pub fn flatten(&self) -> DrawList {
        let mut out = DrawList::new();
        flatten_layer(&self.root, 1.0, &mut out);
        out
    }
}

fn flatten_layer(layer: &Layer, inherited: f32, out: &mut DrawList) {
    let eff = clamp01(inherited * layer.opacity);
    for prim in &layer.primitives {
        out.push(with_opacity(*prim, eff));
    }
    for child in &layer.children {
        flatten_layer(child, eff, out);
    }
}

fn with_opacity(cmd: DrawCommand, factor: f32) -> DrawCommand {
    match cmd {
        DrawCommand::Rect(mut r) => {
            r.opacity = clamp01(r.opacity * factor);
            DrawCommand::Rect(r)
        }
        DrawCommand::Shadow(mut s) => {
            s.opacity = clamp01(s.opacity * factor);
            DrawCommand::Shadow(s)
        }
        DrawCommand::Glyph(mut g) => {
            g.opacity = clamp01(g.opacity * factor);
            DrawCommand::Glyph(g)
        }
        other => other,
    }
}

#[inline]
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::{LayerCmd, RectCmd};
    use prism_ui_layout::{Point, Rect, Size};
    use prism_ui_style::Color;

    fn rect_cmd() -> RectCmd {
        RectCmd {
            rect: Rect::new(Point::ZERO, Size::new(10.0, 10.0)),
            fill: Some(Color::rgba(1.0, 1.0, 1.0, 1.0)),
            radius: 0.0,
            border_width: 0.0,
            border_color: None,
            opacity: 1.0,
        }
    }

    fn layer(opacity: f32) -> LayerCmd {
        LayerCmd {
            bounds: Rect::new(Point::ZERO, Size::new(100.0, 100.0)),
            opacity,
        }
    }

    #[test]
    fn parses_nested_layers() {
        let mut dl = DrawList::new();
        dl.push_rect(rect_cmd());
        dl.push(DrawCommand::PushLayer(layer(0.5)));
        dl.push_rect(rect_cmd());
        dl.push(DrawCommand::PushLayer(layer(0.5)));
        dl.push_rect(rect_cmd());
        dl.push(DrawCommand::PopLayer);
        dl.push(DrawCommand::PopLayer);
        let tree = LayerTree::parse(&dl);
        assert!(tree.is_balanced());
        assert_eq!(tree.layer_count(), 2);
        assert_eq!(tree.root().primitive_count(), 3);
    }

    #[test]
    fn flatten_folds_opacity_multiplicatively() {
        let mut dl = DrawList::new();
        dl.push(DrawCommand::PushLayer(layer(0.5)));
        dl.push(DrawCommand::PushLayer(layer(0.5)));
        dl.push_rect(rect_cmd());
        dl.push(DrawCommand::PopLayer);
        dl.push(DrawCommand::PopLayer);
        let flat = LayerTree::parse(&dl).flatten();
        assert_eq!(flat.primitive_count(), 1);
        if let DrawCommand::Rect(r) = flat.commands()[0] {
            assert!((r.opacity - 0.25).abs() < 1e-6);
        } else {
            panic!("expected rect");
        }
    }

    #[test]
    fn unbalanced_pop_is_detected_not_panicked() {
        let mut dl = DrawList::new();
        dl.push(DrawCommand::PopLayer);
        let tree = LayerTree::parse(&dl);
        assert!(!tree.is_balanced());
    }

    #[test]
    fn unclosed_push_is_detected() {
        let mut dl = DrawList::new();
        dl.push(DrawCommand::PushLayer(layer(0.5)));
        dl.push_rect(rect_cmd());
        let tree = LayerTree::parse(&dl);
        assert!(!tree.is_balanced());
        // The primitive is still accounted for.
        assert_eq!(tree.root().primitive_count(), 1);
    }
}
