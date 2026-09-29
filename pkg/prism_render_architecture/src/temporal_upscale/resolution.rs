//! Render-resolution derivation and render <-> display pixel mapping.
//!
//! A temporal upsampler renders the scene at a fraction of the display
//! resolution and reconstructs the full-resolution image from the jittered
//! history. Two coordinate spaces are therefore in play every frame: the
//! **render** space the `GPU` rasterizes into, and the **display** space the
//! reconstructor outputs. Motion vectors, jitter offsets, and history sampling
//! all need an exact, deterministic mapping between the two.
//!
//! This module owns that mapping. It derives integer render dimensions from a
//! validated render scale (rounding to the nearest pixel and never collapsing
//! to zero), and provides the forward/inverse pixel transforms plus the
//! conversion from a sub-pixel jitter offset to the clip-space
//! (`NDC`) shift the projection matrix must apply.
//!
//! The `GPU`-side viewport/scissor programming and the reconstruction kernel
//! itself are out of scope and pending the `GPU` backend; this module fixes the
//! pixel arithmetic those stages must reproduce.

use super::validate_render_scale;

/// A paired render/display resolution for one reconstructed view.
///
/// All four dimensions are guaranteed `>= 1`: a zero-area target has no defined
/// pixel mapping, so construction clamps up instead of producing a degenerate
/// viewport.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct UpscaleResolution {
    display_width: u32,
    display_height: u32,
    render_width: u32,
    render_height: u32,
}

impl UpscaleResolution {
    /// Derives render dimensions from a display size and a render scale.
    ///
    /// The render scale is first run through [`validate_render_scale`] (so a
    /// `NaN` becomes the native `1.0` and out-of-range values are clamped into
    /// `(0, 1]`). Each display dimension is then multiplied by the scale and
    /// **rounded to the nearest pixel**, with a floor of `1` so the render
    /// target never collapses. Display dimensions of `0` are themselves clamped
    /// to `1`.
    #[must_use]
    pub fn from_display(display_width: u32, display_height: u32, render_scale: f32) -> Self {
        let scale = validate_render_scale(render_scale);
        let display_width = display_width.max(1);
        let display_height = display_height.max(1);
        let render_width = scale_dimension(display_width, scale);
        let render_height = scale_dimension(display_height, scale);
        Self {
            display_width,
            display_height,
            render_width,
            render_height,
        }
    }

    /// Display (output) width in pixels.
    #[must_use]
    pub const fn display_width(self) -> u32 {
        self.display_width
    }

    /// Display (output) height in pixels.
    #[must_use]
    pub const fn display_height(self) -> u32 {
        self.display_height
    }

    /// Render (input) width in pixels.
    #[must_use]
    pub const fn render_width(self) -> u32 {
        self.render_width
    }

    /// Render (input) height in pixels.
    #[must_use]
    pub const fn render_height(self) -> u32 {
        self.render_height
    }

    /// The effective horizontal scale actually realized after rounding.
    ///
    /// Rounding to integer pixels means the realized scale can differ slightly
    /// from the requested one; the reconstructor uses this exact ratio, not the
    /// requested scale, when relating the two spaces.
    #[must_use]
    pub fn effective_scale_x(self) -> f32 {
        self.render_width as f32 / self.display_width as f32
    }

    /// The effective vertical scale actually realized after rounding.
    #[must_use]
    pub fn effective_scale_y(self) -> f32 {
        self.render_height as f32 / self.display_height as f32
    }

    /// Maps a render-space pixel coordinate to display space.
    ///
    /// Uses the pixel-center convention (`+0.5`) so the transform is exact at
    /// pixel centers and symmetric with [`Self::display_to_render`].
    #[must_use]
    pub fn render_to_display(self, render_x: f32, render_y: f32) -> [f32; 2] {
        let inv_x = self.display_width as f32 / self.render_width as f32;
        let inv_y = self.display_height as f32 / self.render_height as f32;
        [
            (render_x + 0.5) * inv_x - 0.5,
            (render_y + 0.5) * inv_y - 0.5,
        ]
    }

    /// Maps a display-space pixel coordinate to render space.
    ///
    /// Inverse of [`Self::render_to_display`], again using pixel centers.
    #[must_use]
    pub fn display_to_render(self, display_x: f32, display_y: f32) -> [f32; 2] {
        [
            (display_x + 0.5) * self.effective_scale_x() - 0.5,
            (display_y + 0.5) * self.effective_scale_y() - 0.5,
        ]
    }

    /// Converts a sub-pixel jitter offset (in render pixels) to a clip-space
    /// (`NDC`) translation for the projection matrix.
    ///
    /// Clip space spans `[-1, 1]` across the render target, i.e. two `NDC`
    /// units cover `render_width` pixels, so one pixel is `2 / render_width`
    /// `NDC` units. The vertical axis is negated because pixel `+y` points down
    /// while `NDC` `+y` points up.
    #[must_use]
    pub fn jitter_to_clip(self, jitter_x: f32, jitter_y: f32) -> [f32; 2] {
        let ndc_per_pixel_x = 2.0 / self.render_width as f32;
        let ndc_per_pixel_y = 2.0 / self.render_height as f32;
        [jitter_x * ndc_per_pixel_x, -jitter_y * ndc_per_pixel_y]
    }

    /// Total render-space pixel count (useful for budget/cost estimates).
    #[must_use]
    pub const fn render_pixel_count(self) -> u64 {
        self.render_width as u64 * self.render_height as u64
    }
}

/// Multiplies a dimension by a scale, rounding to the nearest pixel with a
/// floor of `1`.
///
/// `round` (allowed by the determinism policy) gives the symmetric
/// nearest-integer result; `max(1)` guarantees a non-degenerate target even at
/// the smallest scales.
#[must_use]
fn scale_dimension(dimension: u32, scale: f32) -> u32 {
    let scaled = (dimension as f32 * scale).round();
    // `scaled` is finite and non-negative here (scale is validated to (0, 1]),
    // so the cast is well-defined; clamp up to a single pixel minimum.
    let pixels = scaled as u32;
    pixels.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-4
    }

    #[test]
    fn half_scale_halves_dimensions() {
        let r = UpscaleResolution::from_display(1920, 1080, 0.5);
        assert_eq!(r.render_width(), 960);
        assert_eq!(r.render_height(), 540);
        assert_eq!(r.display_width(), 1920);
        assert_eq!(r.display_height(), 1080);
    }

    #[test]
    fn native_scale_matches_display() {
        let r = UpscaleResolution::from_display(1280, 720, 1.0);
        assert_eq!(r.render_width(), 1280);
        assert_eq!(r.render_height(), 720);
    }

    #[test]
    fn nan_scale_falls_back_to_native() {
        let r = UpscaleResolution::from_display(800, 600, f32::NAN);
        assert_eq!(r.render_width(), 800);
        assert_eq!(r.render_height(), 600);
    }

    #[test]
    fn dimensions_never_collapse_to_zero() {
        // Tiny display + smallest allowed scale still yields >= 1 pixel.
        let r = UpscaleResolution::from_display(1, 1, 0.000_01);
        assert!(r.render_width() >= 1);
        assert!(r.render_height() >= 1);
        // Zero display dimensions clamp up as well.
        let z = UpscaleResolution::from_display(0, 0, 0.5);
        assert!(z.display_width() >= 1 && z.render_width() >= 1);
    }

    #[test]
    fn pixel_mapping_is_reversible() {
        let r = UpscaleResolution::from_display(1920, 1080, 0.5);
        let [dx, dy] = r.render_to_display(100.0, 200.0);
        let [rx, ry] = r.display_to_render(dx, dy);
        assert!(approx(rx, 100.0), "rx {rx}");
        assert!(approx(ry, 200.0), "ry {ry}");
    }

    #[test]
    fn jitter_to_clip_scales_and_flips_y() {
        let r = UpscaleResolution::from_display(1000, 500, 0.5);
        // render is 500 x 250.
        let [cx, cy] = r.jitter_to_clip(0.5, 0.5);
        assert!(approx(cx, 0.5 * 2.0 / 500.0));
        assert!(approx(cy, -0.5 * 2.0 / 250.0));
    }

    #[test]
    fn effective_scale_reports_realized_ratio() {
        let r = UpscaleResolution::from_display(1920, 1080, 0.5);
        assert!(approx(r.effective_scale_x(), 0.5));
        assert!(approx(r.effective_scale_y(), 0.5));
        assert_eq!(r.render_pixel_count(), 960 * 540);
    }
}
