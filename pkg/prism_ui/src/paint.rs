//! Non-layout visual properties resolved for a node.
//!
//! Layout decides *where* a box is; [`PaintStyle`] decides how it *looks*. The
//! runtime diffs this struct between frames so the backend is only told about
//! visual properties that actually changed.
//!
//! Beyond the flat fill/border/text properties, [`PaintStyle`] can carry a
//! [`Shadow`] (soft drop shadow for depth) and a [`Glass`] descriptor (an
//! approximate "frosted glass"/material surface). The glass effect is a
//! *cheap approximation*: instead of a true backdrop blur — which forces an
//! off-screen resolve and a multi-pass blur per surface — the lowering layers a
//! translucent tint plus a bright rim highlight over the content behind it. A
//! GPU backend may honour [`Glass::blur`] with a real (ideally down-sampled)
//! backdrop blur; the headless reference rasteriser treats it as a hint and
//! renders the translucent approximation, which already reads convincingly as
//! glass at a fraction of the cost.

use prism_ui_style::Color;

/// A soft drop shadow cast by a node's box, in logical pixels.
///
/// `offset_x`/`offset_y` shift the shadow, `blur` controls the softness of the
/// falloff and `spread` inflates (or, when negative, deflates) the caster
/// before blurring. An `inset` shadow is an *inner* shadow; the headless
/// reference backend currently renders only outer shadows and treats inset as a
/// hint, but the field is preserved so a richer backend can honour it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shadow {
    /// Horizontal offset in logical pixels (positive = right).
    pub offset_x: f32,
    /// Vertical offset in logical pixels (positive = down).
    pub offset_y: f32,
    /// Blur radius in logical pixels (`0` = hard edge).
    pub blur: f32,
    /// Caster inflation in logical pixels before blurring.
    pub spread: f32,
    /// Shadow colour; its alpha is modulated by the blur falloff.
    pub color: Color,
    /// Whether this is an inner (`true`) rather than drop (`false`) shadow.
    pub inset: bool,
}

impl Shadow {
    /// A soft outer drop shadow with no spread.
    #[must_use]
    pub const fn drop(offset_x: f32, offset_y: f32, blur: f32, color: Color) -> Self {
        Self {
            offset_x,
            offset_y,
            blur,
            spread: 0.0,
            color,
            inset: false,
        }
    }

    /// Returns a copy with `spread` set, for chaining.
    #[must_use]
    pub const fn with_spread(mut self, spread: f32) -> Self {
        self.spread = spread;
        self
    }

    /// Returns a copy marked as an inner (inset) shadow, for chaining.
    #[must_use]
    pub const fn inset(mut self) -> Self {
        self.inset = true;
        self
    }
}

/// An approximate frosted-glass / material surface.
///
/// Rather than a true (expensive) backdrop blur, the lowering composites a
/// translucent `tint` over whatever is already painted behind the node and
/// draws an optional bright `highlight` rim so the surface reads as glass. The
/// `blur` radius is a device hint: a GPU backend may apply a real, down-sampled
/// backdrop blur, while the reference rasteriser ignores it and relies on the
/// translucent tint alone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glass {
    /// Backdrop-blur radius hint in logical pixels (backend may approximate).
    pub blur: f32,
    /// Translucent tint composited over the content behind the surface.
    pub tint: Color,
    /// Optional bright rim highlight (the lit glass edge).
    pub highlight: Option<Color>,
}

impl Glass {
    /// A glass surface with the given blur hint and translucent `tint`.
    #[must_use]
    pub const fn new(blur: f32, tint: Color) -> Self {
        Self {
            blur,
            tint,
            highlight: None,
        }
    }

    /// Returns a copy with a rim highlight colour, for chaining.
    #[must_use]
    pub const fn with_highlight(mut self, highlight: Color) -> Self {
        self.highlight = Some(highlight);
        self
    }
}

/// The resolved visual appearance of a node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaintStyle {
    /// Background fill color.
    pub background_color: Option<Color>,
    /// Foreground / text color.
    pub color: Option<Color>,
    /// Border color.
    pub border_color: Option<Color>,
    /// Border corner radius, in logical pixels.
    pub border_radius: f32,
    /// Border thickness, in logical pixels.
    pub border_width: f32,
    /// Opacity in `0.0..=1.0`.
    pub opacity: f32,
    /// Font size, in logical pixels (text nodes).
    pub font_size: f32,
    /// Font weight (e.g. `400.0` normal, `700.0` bold).
    pub font_weight: f32,
    /// Optional soft drop shadow for depth.
    pub shadow: Option<Shadow>,
    /// Optional approximate frosted-glass surface.
    pub glass: Option<Glass>,
}

impl Default for PaintStyle {
    fn default() -> Self {
        Self {
            background_color: None,
            color: None,
            border_color: None,
            border_radius: 0.0,
            border_width: 0.0,
            opacity: 1.0,
            font_size: 16.0,
            font_weight: 400.0,
            shadow: None,
            glass: None,
        }
    }
}

impl PaintStyle {
    /// Returns a copy with `shadow` set, for chaining.
    #[must_use]
    pub const fn with_shadow(mut self, shadow: Shadow) -> Self {
        self.shadow = Some(shadow);
        self
    }

    /// Returns a copy with `glass` set, for chaining.
    #[must_use]
    pub const fn with_glass(mut self, glass: Glass) -> Self {
        self.glass = Some(glass);
        self
    }

    /// Builds a frosted-glass surface paint: a rounded box with a translucent
    /// `tint`, a rim `highlight`, a soft drop `shadow` and the given corner
    /// `radius`. This is the ergonomic bridge from theme tokens (tint,
    /// highlight, shadow colour) to a ready-to-render [`PaintStyle`].
    #[must_use]
    pub fn glass_surface(
        radius: f32,
        blur: f32,
        tint: Color,
        highlight: Color,
        shadow: Shadow,
    ) -> Self {
        Self {
            border_radius: radius,
            shadow: Some(shadow),
            glass: Some(Glass::new(blur, tint).with_highlight(highlight)),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_no_effects() {
        let p = PaintStyle::default();
        assert!(p.shadow.is_none());
        assert!(p.glass.is_none());
        assert_eq!(p.opacity, 1.0);
    }

    #[test]
    fn shadow_builders_chain() {
        let s = Shadow::drop(0.0, 4.0, 12.0, Color::rgba8(0, 0, 0, 60))
            .with_spread(2.0)
            .inset();
        assert_eq!(s.offset_y, 4.0);
        assert_eq!(s.blur, 12.0);
        assert_eq!(s.spread, 2.0);
        assert!(s.inset);
    }

    #[test]
    fn glass_builders_chain() {
        let g = Glass::new(20.0, Color::rgba8(255, 255, 255, 160))
            .with_highlight(Color::rgba8(255, 255, 255, 115));
        assert_eq!(g.blur, 20.0);
        assert!(g.highlight.is_some());
    }

    #[test]
    fn glass_surface_sets_all_layers() {
        let p = PaintStyle::glass_surface(
            16.0,
            20.0,
            Color::rgba8(255, 255, 255, 160),
            Color::rgba8(255, 255, 255, 115),
            Shadow::drop(0.0, 8.0, 24.0, Color::rgba8(0, 0, 0, 40)),
        );
        assert_eq!(p.border_radius, 16.0);
        assert!(p.glass.is_some());
        assert!(p.shadow.is_some());
        assert_eq!(p.glass.unwrap().blur, 20.0);
    }

    #[test]
    fn paint_style_stays_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<PaintStyle>();
        assert_copy::<Shadow>();
        assert_copy::<Glass>();
    }
}
