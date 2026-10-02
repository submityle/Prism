//! Draw-command batching and instancing.
//!
//! A GPU cannot afford one draw call per rectangle. [`batch`] coalesces a
//! paint-ordered [`DrawList`] into the smallest number of [`Batch`]es that
//! still preserves visual order: consecutive commands of the same *kind* are
//! merged into one instanced batch, and any layer boundary forces a flush so
//! compositing semantics are never reordered across a group.

use alloc::vec::Vec;

use prism_ui_layout::{Point, Rect};
use prism_ui_style::Color;

use crate::draw::{DrawCommand, DrawList};

/// Per-rectangle instance data uploaded to the GPU.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RectInstance {
    /// Device-space bounds.
    pub rect: Rect,
    /// Fill colour (`a == 0` means no fill).
    pub fill: Color,
    /// Corner radius.
    pub radius: f32,
    /// Border thickness.
    pub border_width: f32,
    /// Border colour.
    pub border_color: Color,
    /// Effective opacity.
    pub opacity: f32,
}

/// Per-shadow instance data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowInstance {
    /// Caster bounds.
    pub rect: Rect,
    /// Caster corner radius.
    pub radius: f32,
    /// Blur radius.
    pub blur: f32,
    /// Shadow offset.
    pub offset: Point<f32>,
    /// Shadow colour.
    pub color: Color,
    /// Effective opacity.
    pub opacity: f32,
}

/// Per-glyph-run instance data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlyphInstance {
    /// Run bounds.
    pub rect: Rect,
    /// Text colour.
    pub color: Color,
    /// Em size.
    pub size: f32,
    /// Effective opacity.
    pub opacity: f32,
}

/// A group of instances a renderer can draw in a single instanced call.
#[derive(Clone, Debug, PartialEq)]
pub enum Batch {
    /// A run of rounded rectangles.
    Rects(Vec<RectInstance>),
    /// A run of soft shadows (drawn before the rects they sit under).
    Shadows(Vec<ShadowInstance>),
    /// A run of glyph quads.
    Glyphs(Vec<GlyphInstance>),
    /// Begin a compositing layer.
    PushLayer {
        /// Device-space bounds of the layer.
        bounds: Rect,
        /// Group opacity applied to the layer's contents.
        opacity: f32,
    },
    /// End a compositing layer.
    PopLayer,
}

const TRANSPARENT: Color = Color::rgba(0.0, 0.0, 0.0, 0.0);

/// Coalesces a draw list into instanced batches, preserving paint order.
#[must_use]
pub fn batch(list: &DrawList) -> Vec<Batch> {
    let mut out: Vec<Batch> = Vec::new();
    for cmd in list.commands() {
        match *cmd {
            DrawCommand::Rect(r) => {
                let inst = RectInstance {
                    rect: r.rect,
                    fill: r.fill.unwrap_or(TRANSPARENT),
                    radius: r.radius,
                    border_width: r.border_width,
                    border_color: r.border_color.unwrap_or(TRANSPARENT),
                    opacity: r.opacity,
                };
                match out.last_mut() {
                    Some(Batch::Rects(v)) => v.push(inst),
                    _ => out.push(Batch::Rects(alloc::vec![inst])),
                }
            }
            DrawCommand::Shadow(s) => {
                let inst = ShadowInstance {
                    rect: s.rect,
                    radius: s.radius,
                    blur: s.blur,
                    offset: s.offset,
                    color: s.color,
                    opacity: s.opacity,
                };
                match out.last_mut() {
                    Some(Batch::Shadows(v)) => v.push(inst),
                    _ => out.push(Batch::Shadows(alloc::vec![inst])),
                }
            }
            DrawCommand::Glyph(g) => {
                let inst = GlyphInstance {
                    rect: g.rect,
                    color: g.color,
                    size: g.size,
                    opacity: g.opacity,
                };
                match out.last_mut() {
                    Some(Batch::Glyphs(v)) => v.push(inst),
                    _ => out.push(Batch::Glyphs(alloc::vec![inst])),
                }
            }
            DrawCommand::PushLayer(l) => out.push(Batch::PushLayer {
                bounds: l.bounds,
                opacity: l.opacity,
            }),
            DrawCommand::PopLayer => out.push(Batch::PopLayer),
        }
    }
    out
}

/// Total instance count across every batch, i.e. the number of primitives that
/// survive batching (useful for asserting a batching ratio in tests).
#[must_use]
pub fn instance_count(batches: &[Batch]) -> usize {
    batches
        .iter()
        .map(|b| match b {
            Batch::Rects(v) => v.len(),
            Batch::Shadows(v) => v.len(),
            Batch::Glyphs(v) => v.len(),
            Batch::PushLayer { .. } | Batch::PopLayer => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::{GlyphCmd, LayerCmd, RectCmd};
    use prism_ui_layout::Size;

    fn rect(x: f32, y: f32) -> Rect {
        Rect::new(Point::new(x, y), Size::new(10.0, 10.0))
    }

    fn white() -> Color {
        Color::rgba(1.0, 1.0, 1.0, 1.0)
    }

    fn rect_cmd(x: f32) -> RectCmd {
        RectCmd {
            rect: rect(x, 0.0),
            fill: Some(white()),
            radius: 0.0,
            border_width: 0.0,
            border_color: None,
            opacity: 1.0,
        }
    }

    #[test]
    fn consecutive_rects_merge_into_one_batch() {
        let mut dl = DrawList::new();
        for i in 0..5 {
            dl.push_rect(rect_cmd(i as f32 * 10.0));
        }
        let b = batch(&dl);
        assert_eq!(b.len(), 1);
        assert_eq!(instance_count(&b), 5);
        assert!(matches!(b[0], Batch::Rects(ref v) if v.len() == 5));
    }

    #[test]
    fn layer_boundary_breaks_batch() {
        let mut dl = DrawList::new();
        dl.push_rect(rect_cmd(0.0));
        dl.push(DrawCommand::PushLayer(LayerCmd {
            bounds: rect(0.0, 0.0),
            opacity: 0.5,
        }));
        dl.push_rect(rect_cmd(20.0));
        dl.push(DrawCommand::PopLayer);
        let b = batch(&dl);
        // Rects, PushLayer, Rects, PopLayer.
        assert_eq!(b.len(), 4);
        assert_eq!(instance_count(&b), 2);
    }

    #[test]
    fn kind_change_breaks_batch() {
        let mut dl = DrawList::new();
        dl.push_rect(rect_cmd(0.0));
        dl.push(DrawCommand::Glyph(GlyphCmd {
            rect: rect(0.0, 0.0),
            color: white(),
            size: 16.0,
            opacity: 1.0,
        }));
        dl.push_rect(rect_cmd(20.0));
        let b = batch(&dl);
        assert_eq!(b.len(), 3);
        assert!(matches!(b[0], Batch::Rects(_)));
        assert!(matches!(b[1], Batch::Glyphs(_)));
        assert!(matches!(b[2], Batch::Rects(_)));
    }

    #[test]
    fn empty_list_no_batches() {
        assert!(batch(&DrawList::new()).is_empty());
    }
}
