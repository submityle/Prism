//! The retained draw stream: an ordered list of resolved [`DrawCommand`]s.
//!
//! A [`DrawList`] is the hand-off point between layout/paint and a concrete
//! renderer. It is deliberately flat and absolute: every command carries final
//! device-space geometry, pre-multiplied effective opacity and the layer depth
//! it belongs to. Both the headless rasteriser and the GPU backend consume the
//! exact same list, which is what makes the two verifiable against each other.

use alloc::vec::Vec;

use prism_ui_layout::{Point, Rect, Size};
use prism_ui_style::Color;

/// A single resolved drawing instruction in absolute device coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DrawCommand {
    /// A filled (and optionally stroked) rounded rectangle.
    Rect(RectCmd),
    /// A soft drop shadow cast by a rounded rectangle.
    Shadow(ShadowCmd),
    /// A positioned glyph run. Text shaping lives in `prism_ui_text`; this
    /// command only carries the resolved box, colour and em size so a renderer
    /// can place an atlas quad.
    Glyph(GlyphCmd),
    /// Begin an isolated compositing layer. Everything until the matching
    /// [`DrawCommand::PopLayer`] is composited as a group with `opacity`.
    PushLayer(LayerCmd),
    /// End the most recently pushed compositing layer.
    PopLayer,
}

/// A filled, optionally bordered, rounded rectangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RectCmd {
    /// Device-space bounds.
    pub rect: Rect,
    /// Fill colour, or `None` for a border-only box.
    pub fill: Option<Color>,
    /// Corner radius in device pixels.
    pub radius: f32,
    /// Border thickness in device pixels (0 disables the stroke).
    pub border_width: f32,
    /// Border colour; ignored when `border_width == 0`.
    pub border_color: Option<Color>,
    /// Effective opacity in `0.0..=1.0`, folded in from ancestor layers.
    pub opacity: f32,
}

/// A soft drop shadow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowCmd {
    /// Bounds of the box casting the shadow (before offset).
    pub rect: Rect,
    /// Corner radius of the caster.
    pub radius: f32,
    /// Blur radius in device pixels.
    pub blur: f32,
    /// Shadow offset.
    pub offset: Point<f32>,
    /// Shadow colour (alpha is modulated by the falloff).
    pub color: Color,
    /// Effective opacity in `0.0..=1.0`.
    pub opacity: f32,
}

/// A positioned glyph box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlyphCmd {
    /// Device-space bounds of the run.
    pub rect: Rect,
    /// Text colour.
    pub color: Color,
    /// Em size in device pixels.
    pub size: f32,
    /// Effective opacity in `0.0..=1.0`.
    pub opacity: f32,
}

/// Parameters for a compositing layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerCmd {
    /// Clip/compositing bounds of the layer.
    pub bounds: Rect,
    /// Group opacity applied when the layer is composited back.
    pub opacity: f32,
}

/// An ordered, absolute list of [`DrawCommand`]s plus the viewport they were
/// culled against.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DrawList {
    commands: Vec<DrawCommand>,
}

impl DrawList {
    /// Creates an empty draw list.
    #[must_use]
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    /// Appends a command.
    pub fn push(&mut self, cmd: DrawCommand) {
        self.commands.push(cmd);
    }

    /// Appends a rectangle convenience command.
    pub fn push_rect(&mut self, cmd: RectCmd) {
        self.commands.push(DrawCommand::Rect(cmd));
    }

    /// Appends a shadow convenience command.
    pub fn push_shadow(&mut self, cmd: ShadowCmd) {
        self.commands.push(DrawCommand::Shadow(cmd));
    }

    /// The commands in paint order.
    #[must_use]
    pub fn commands(&self) -> &[DrawCommand] {
        &self.commands
    }

    /// Number of commands.
    #[must_use]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Whether the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Removes commands fully outside `viewport`, in place.
    ///
    /// Layer push/pop markers are always kept so the compositing stack stays
    /// balanced; only the leaf primitives (rects, shadows, glyphs) are culled.
    pub fn cull(&mut self, viewport: Rect) {
        self.commands.retain(|cmd| match cmd {
            DrawCommand::PushLayer(_) | DrawCommand::PopLayer => true,
            DrawCommand::Rect(r) => intersects(r.rect, viewport),
            DrawCommand::Shadow(s) => intersects(offset_rect(s.rect, s.offset, s.blur), viewport),
            DrawCommand::Glyph(g) => intersects(g.rect, viewport),
        });
    }

    /// Returns the number of leaf primitives (ignoring layer markers).
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.commands
            .iter()
            .filter(|c| !matches!(c, DrawCommand::PushLayer(_) | DrawCommand::PopLayer))
            .count()
    }
}

/// Whether two rectangles overlap (touching edges count as overlapping).
#[must_use]
pub fn intersects(a: Rect, b: Rect) -> bool {
    a.left() <= b.right() && a.right() >= b.left() && a.top() <= b.bottom() && a.bottom() >= b.top()
}

/// Expands `rect` by `blur` on every side and shifts it by `offset`, used to
/// bound a shadow's influence for culling.
#[must_use]
fn offset_rect(rect: Rect, offset: Point<f32>, blur: f32) -> Rect {
    Rect::new(
        Point::new(
            rect.location.x + offset.x - blur,
            rect.location.y + offset.y - blur,
        ),
        Size::new(rect.size.width + 2.0 * blur, rect.size.height + 2.0 * blur),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect::new(Point::new(x, y), Size::new(w, h))
    }

    fn solid(c: f32) -> Color {
        Color::rgba(c, c, c, 1.0)
    }

    #[test]
    fn push_and_count() {
        let mut dl = DrawList::new();
        assert!(dl.is_empty());
        dl.push_rect(RectCmd {
            rect: rect(0.0, 0.0, 10.0, 10.0),
            fill: Some(solid(1.0)),
            radius: 0.0,
            border_width: 0.0,
            border_color: None,
            opacity: 1.0,
        });
        assert_eq!(dl.len(), 1);
        assert_eq!(dl.primitive_count(), 1);
    }

    #[test]
    fn intersects_basic() {
        assert!(intersects(
            rect(0.0, 0.0, 10.0, 10.0),
            rect(5.0, 5.0, 10.0, 10.0)
        ));
        assert!(!intersects(
            rect(0.0, 0.0, 10.0, 10.0),
            rect(20.0, 20.0, 5.0, 5.0)
        ));
        // Touching edges count.
        assert!(intersects(
            rect(0.0, 0.0, 10.0, 10.0),
            rect(10.0, 0.0, 5.0, 5.0)
        ));
    }

    #[test]
    fn cull_drops_offscreen_but_keeps_layers() {
        let mut dl = DrawList::new();
        dl.push(DrawCommand::PushLayer(LayerCmd {
            bounds: rect(0.0, 0.0, 100.0, 100.0),
            opacity: 1.0,
        }));
        dl.push_rect(RectCmd {
            rect: rect(10.0, 10.0, 10.0, 10.0),
            fill: Some(solid(1.0)),
            radius: 0.0,
            border_width: 0.0,
            border_color: None,
            opacity: 1.0,
        });
        dl.push_rect(RectCmd {
            rect: rect(500.0, 500.0, 10.0, 10.0),
            fill: Some(solid(1.0)),
            radius: 0.0,
            border_width: 0.0,
            border_color: None,
            opacity: 1.0,
        });
        dl.push(DrawCommand::PopLayer);
        dl.cull(rect(0.0, 0.0, 100.0, 100.0));
        assert_eq!(dl.primitive_count(), 1);
        // Both layer markers survive.
        assert_eq!(dl.len(), 3);
    }

    #[test]
    fn shadow_culling_accounts_for_offset_and_blur() {
        let mut dl = DrawList::new();
        dl.push_shadow(ShadowCmd {
            rect: rect(90.0, 90.0, 10.0, 10.0),
            radius: 0.0,
            blur: 20.0,
            offset: Point::new(5.0, 5.0),
            color: solid(0.0),
            opacity: 1.0,
        });
        // Caster is near the corner; blur spill keeps it visible in a 100x100 viewport.
        dl.cull(rect(0.0, 0.0, 100.0, 100.0));
        assert_eq!(dl.primitive_count(), 1);
    }
}
