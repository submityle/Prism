//! Headless CPU rasteriser — the reference backend.
//!
//! This is the golden twin the optional GPU backend is validated against. It
//! evaluates the exact same [`crate::sdf`] distance functions per pixel and
//! composites with straight alpha-over, producing a deterministic linear-space
//! `RGBA` framebuffer. Because it shares the SDF math with the shader, a parity
//! test can assert the two agree to a tight tolerance.

use alloc::vec;
use alloc::vec::Vec;

use prism_ui_layout::Rect;
use prism_ui_style::Color;

use crate::draw::{DrawCommand, DrawList, GlyphCmd, RectCmd, ShadowCmd};
use crate::sdf;

/// A linear-space `RGBA` framebuffer with straight (non-premultiplied) alpha.
#[derive(Clone, Debug, PartialEq)]
pub struct Framebuffer {
    width: u32,
    height: u32,
    pixels: Vec<[f32; 4]>,
}

impl Framebuffer {
    /// Creates a transparent framebuffer of the given size.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![[0.0; 4]; (width as usize) * (height as usize)],
        }
    }

    /// Framebuffer width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Framebuffer height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Reads a pixel; out-of-bounds reads return transparent black.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> [f32; 4] {
        if x >= self.width || y >= self.height {
            return [0.0; 4];
        }
        self.pixels[(y as usize) * (self.width as usize) + (x as usize)]
    }

    fn blend(&mut self, x: u32, y: u32, src: [f32; 4]) {
        if x >= self.width || y >= self.height || src[3] <= 0.0 {
            return;
        }
        let idx = (y as usize) * (self.width as usize) + (x as usize);
        let dst = self.pixels[idx];
        let sa = clamp01(src[3]);
        let out_a = sa + dst[3] * (1.0 - sa);
        let mut out = [0.0f32; 4];
        out[3] = out_a;
        if out_a > 0.0 {
            for c in 0..3 {
                out[c] = (src[c] * sa + dst[c] * dst[3] * (1.0 - sa)) / out_a;
            }
        }
        self.pixels[idx] = out;
    }

    /// Maximum absolute per-channel difference against another framebuffer of
    /// the same size. Used by the GPU parity test. Returns `f32::INFINITY` when
    /// sizes differ.
    #[must_use]
    pub fn max_diff(&self, other: &Framebuffer) -> f32 {
        if self.width != other.width || self.height != other.height {
            return f32::INFINITY;
        }
        let mut worst = 0.0f32;
        for (a, b) in self.pixels.iter().zip(other.pixels.iter()) {
            for c in 0..4 {
                let d = (a[c] - b[c]).abs();
                if d > worst {
                    worst = d;
                }
            }
        }
        worst
    }
}

/// Rasterises a draw list into a fresh [`Framebuffer`].
///
/// Layer markers are flattened by opacity before rasterising (see
/// [`crate::layer::LayerTree::flatten`]), so nested group opacity is honoured.
#[must_use]
pub fn rasterize(list: &DrawList, width: u32, height: u32) -> Framebuffer {
    let flat = crate::layer::LayerTree::parse(list).flatten();
    let mut fb = Framebuffer::new(width, height);
    for cmd in flat.commands() {
        match *cmd {
            DrawCommand::Shadow(s) => raster_shadow(&mut fb, s),
            DrawCommand::Rect(r) => raster_rect(&mut fb, r),
            DrawCommand::Glyph(g) => raster_glyph(&mut fb, g),
            DrawCommand::PushLayer(_) | DrawCommand::PopLayer => {}
        }
    }
    fb
}

/// Pixel span `[lo, hi)` covering `rect` expanded by `pad`, clamped to `[0, limit)`.
fn span(start: f32, extent: f32, pad: f32, limit: u32) -> (u32, u32) {
    let lo = (start - pad).floor().max(0.0) as u32;
    let hi = ((start + extent + pad).ceil().max(0.0) as u32).min(limit);
    (lo, hi.max(lo))
}

fn raster_rect(fb: &mut Framebuffer, cmd: RectCmd) {
    let hw = cmd.rect.size.width * 0.5;
    let hh = cmd.rect.size.height * 0.5;
    let cx = cmd.rect.left() + hw;
    let cy = cmd.rect.top() + hh;
    let (x0, x1) = span(cmd.rect.left(), cmd.rect.size.width, 1.0, fb.width);
    let (y0, y1) = span(cmd.rect.top(), cmd.rect.size.height, 1.0, fb.height);
    for py in y0..y1 {
        for px in x0..x1 {
            let sx = px as f32 + 0.5 - cx;
            let sy = py as f32 + 0.5 - cy;
            let dist = sdf::sd_rounded_box(sx, sy, hw, hh, cmd.radius);
            // Fill.
            if let Some(fill) = cmd.fill {
                let cov = sdf::coverage(dist) * cmd.opacity * fill.a;
                fb.blend(px, py, [fill.r, fill.g, fill.b, cov]);
            }
            // Border ring on top of the fill.
            if cmd.border_width > 0.0
                && let Some(bc) = cmd.border_color
            {
                let cov = sdf::border_coverage(dist, cmd.border_width) * cmd.opacity * bc.a;
                fb.blend(px, py, [bc.r, bc.g, bc.b, cov]);
            }
        }
    }
}

fn raster_shadow(fb: &mut Framebuffer, cmd: ShadowCmd) {
    let hw = cmd.rect.size.width * 0.5;
    let hh = cmd.rect.size.height * 0.5;
    let cx = cmd.rect.left() + hw + cmd.offset.x;
    let cy = cmd.rect.top() + hh + cmd.offset.y;
    let pad = cmd.blur + 1.0;
    let (x0, x1) = span(
        cmd.rect.left() + cmd.offset.x,
        cmd.rect.size.width,
        pad,
        fb.width,
    );
    let (y0, y1) = span(
        cmd.rect.top() + cmd.offset.y,
        cmd.rect.size.height,
        pad,
        fb.height,
    );
    for py in y0..y1 {
        for px in x0..x1 {
            let sx = px as f32 + 0.5 - cx;
            let sy = py as f32 + 0.5 - cy;
            let dist = sdf::sd_rounded_box(sx, sy, hw, hh, cmd.radius);
            let a = sdf::shadow_alpha(dist, cmd.blur) * cmd.opacity * cmd.color.a;
            fb.blend(px, py, [cmd.color.r, cmd.color.g, cmd.color.b, a]);
        }
    }
}

/// Glyph runs are rendered as a solid coverage block in the reference backend:
/// real glyph atlases live in `prism_ui_text`, but the reference only needs to
/// prove the run's box, colour and opacity reach the target. This is an honest
/// placeholder for shaping, not a stub for the compositing path it exercises.
fn raster_glyph(fb: &mut Framebuffer, cmd: GlyphCmd) {
    let fill = Color::rgba(cmd.color.r, cmd.color.g, cmd.color.b, cmd.color.a);
    raster_rect(
        fb,
        RectCmd {
            rect: cmd.rect,
            fill: Some(fill),
            radius: 0.0,
            border_width: 0.0,
            border_color: None,
            opacity: cmd.opacity,
        },
    );
}

/// Bounding box union helper exposed for tooling/tests.
#[must_use]
pub fn union(a: Rect, b: Rect) -> Rect {
    use prism_ui_layout::{Point, Size};
    let left = a.left().min(b.left());
    let top = a.top().min(b.top());
    let right = a.right().max(b.right());
    let bottom = a.bottom().max(b.bottom());
    Rect::new(Point::new(left, top), Size::new(right - left, bottom - top))
}

#[inline]
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_layout::{Point, Size};

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect::new(Point::new(x, y), Size::new(w, h))
    }

    fn red() -> Color {
        Color::rgba(1.0, 0.0, 0.0, 1.0)
    }

    #[test]
    fn solid_rect_fills_interior() {
        let mut dl = DrawList::new();
        dl.push_rect(RectCmd {
            rect: rect(2.0, 2.0, 16.0, 16.0),
            fill: Some(red()),
            radius: 0.0,
            border_width: 0.0,
            border_color: None,
            opacity: 1.0,
        });
        let fb = rasterize(&dl, 20, 20);
        let p = fb.pixel(10, 10);
        assert!(
            (p[0] - 1.0).abs() < 1e-3 && p[3] > 0.99,
            "interior opaque red"
        );
        assert_eq!(fb.pixel(0, 0)[3], 0.0, "corner untouched");
    }

    #[test]
    fn opacity_scales_alpha() {
        let mut dl = DrawList::new();
        dl.push_rect(RectCmd {
            rect: rect(0.0, 0.0, 20.0, 20.0),
            fill: Some(red()),
            radius: 0.0,
            border_width: 0.0,
            border_color: None,
            opacity: 0.5,
        });
        let fb = rasterize(&dl, 20, 20);
        assert!((fb.pixel(10, 10)[3] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn rounded_corner_is_antialiased_or_empty() {
        let mut dl = DrawList::new();
        dl.push_rect(RectCmd {
            rect: rect(0.0, 0.0, 20.0, 20.0),
            fill: Some(red()),
            radius: 8.0,
            border_width: 0.0,
            border_color: None,
            opacity: 1.0,
        });
        let fb = rasterize(&dl, 20, 20);
        // Extreme corner is outside the rounded shape.
        assert!(fb.pixel(0, 0)[3] < 0.5);
        // Centre is solid.
        assert!(fb.pixel(10, 10)[3] > 0.99);
    }

    #[test]
    fn border_draws_ring() {
        let mut dl = DrawList::new();
        dl.push_rect(RectCmd {
            rect: rect(2.0, 2.0, 16.0, 16.0),
            fill: None,
            radius: 0.0,
            border_width: 2.0,
            border_color: Some(Color::rgba(0.0, 0.0, 1.0, 1.0)),
            opacity: 1.0,
        });
        let fb = rasterize(&dl, 20, 20);
        // Near the edge -> blue ring.
        assert!(fb.pixel(2, 10)[2] > 0.3, "edge is blue");
        // Centre -> no fill.
        assert!(fb.pixel(10, 10)[3] < 0.2, "interior empty");
    }

    #[test]
    fn shadow_fades_outward() {
        let mut dl = DrawList::new();
        dl.push_shadow(ShadowCmd {
            rect: rect(20.0, 20.0, 20.0, 20.0),
            radius: 0.0,
            blur: 8.0,
            offset: Point::ZERO,
            color: Color::rgba(0.0, 0.0, 0.0, 1.0),
            opacity: 1.0,
        });
        let fb = rasterize(&dl, 60, 60);
        let inside = fb.pixel(30, 30)[3];
        let near = fb.pixel(42, 30)[3];
        let far = fb.pixel(50, 30)[3];
        assert!(inside > near && near > far, "shadow fades with distance");
    }

    #[test]
    fn identical_lists_have_zero_diff() {
        let mut dl = DrawList::new();
        dl.push_rect(RectCmd {
            rect: rect(1.0, 1.0, 10.0, 10.0),
            fill: Some(red()),
            radius: 2.0,
            border_width: 1.0,
            border_color: Some(Color::rgba(0.0, 1.0, 0.0, 1.0)),
            opacity: 1.0,
        });
        let a = rasterize(&dl, 16, 16);
        let b = rasterize(&dl, 16, 16);
        assert_eq!(a.max_diff(&b), 0.0, "rasteriser is deterministic");
    }

    #[test]
    fn union_covers_both() {
        let u = union(rect(0.0, 0.0, 10.0, 10.0), rect(20.0, 5.0, 10.0, 10.0));
        assert_eq!(u.left(), 0.0);
        assert_eq!(u.right(), 30.0);
        assert_eq!(u.bottom(), 15.0);
    }
}
