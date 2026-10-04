//! Color operations for [`Color`]: compositing, interpolation and `WCAG`
//! luminance/contrast.
//!
//! [`Color`] stores its channels as *linear* light in the range `0.0..=1.0`
//! (see its type documentation). Blending and interpolation are only physically
//! meaningful in linear space, so every operation here assumes linear input and
//! produces linear output. Callers that start from `sRGB`-encoded text colors
//! should convert through the color-science types in `prism_math` (for example
//! `Srgba`/`LinearRgba`) before building a style [`Color`]; this crate stays
//! deliberately free of transcendental gamma math so it remains `no_std` and
//! dependency-light.
//!
//! The pieces are:
//!
//! * **Alpha compositing** — [`Color::over`] implements straight-alpha
//!   source-over (Porter-Duff `A over B`), the operator that paints a
//!   translucent color on top of a background.
//! * **Interpolation** — [`Color::lerp`] blends two colors channel-wise, the
//!   building block for gradients and state transitions.
//! * **Alpha helpers** — [`Color::with_alpha`], [`Color::scale_alpha`],
//!   [`Color::is_opaque`] and [`Color::premultiplied`].
//! * **`WCAG` metrics** — [`Color::relative_luminance`] and
//!   [`Color::contrast_ratio`], used for accessibility contrast checks.

use crate::value::Color;

impl Color {
    /// Fully transparent black (all channels `0.0`).
    pub const TRANSPARENT: Color = Color::rgba(0.0, 0.0, 0.0, 0.0);
    /// Opaque black.
    pub const BLACK: Color = Color::rgba(0.0, 0.0, 0.0, 1.0);
    /// Opaque white.
    pub const WHITE: Color = Color::rgba(1.0, 1.0, 1.0, 1.0);

    /// Returns a copy of this color with its alpha replaced by `alpha`
    /// (clamped to `0.0..=1.0`).
    #[must_use]
    pub fn with_alpha(self, alpha: f32) -> Self {
        Color::rgba(self.r, self.g, self.b, alpha.clamp(0.0, 1.0))
    }

    /// Returns a copy of this color with its alpha multiplied by `factor`
    /// (the result is clamped to `0.0..=1.0`).
    ///
    /// Fading an element to 50% opacity is `color.scale_alpha(0.5)`.
    #[must_use]
    pub fn scale_alpha(self, factor: f32) -> Self {
        Color::rgba(self.r, self.g, self.b, (self.a * factor).clamp(0.0, 1.0))
    }

    /// Returns `true` if this color is fully opaque (`alpha >= 1.0`).
    #[must_use]
    pub fn is_opaque(self) -> bool {
        self.a >= 1.0
    }

    /// Returns the premultiplied-alpha form `(r*a, g*a, b*a, a)`.
    ///
    /// Premultiplied color is the representation most GPU blenders expect; it is
    /// also the intermediate used by [`Color::over`].
    #[must_use]
    pub fn premultiplied(self) -> (f32, f32, f32, f32) {
        (self.r * self.a, self.g * self.a, self.b * self.a, self.a)
    }

    /// Composites this color (the source) over `background` using straight-alpha
    /// source-over, the Porter-Duff `A over B` operator.
    ///
    /// A fully opaque source replaces the background; a fully transparent source
    /// leaves the background unchanged; intermediate alphas blend linearly in
    /// premultiplied space and the result is un-premultiplied back to straight
    /// alpha.
    ///
    /// # Example
    ///
    /// ```
    /// use prism_ui_style::Color;
    ///
    /// let red = Color::rgba(1.0, 0.0, 0.0, 1.0);
    /// let blue = Color::rgba(0.0, 0.0, 1.0, 1.0);
    /// // An opaque source wins outright.
    /// assert_eq!(red.over(blue), red);
    /// // A half-opaque red over opaque blue is an even mix (alpha 1.0).
    /// let mix = red.with_alpha(0.5).over(blue);
    /// assert!((mix.r - 0.5).abs() < 1e-6);
    /// assert!((mix.b - 0.5).abs() < 1e-6);
    /// assert!((mix.a - 1.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn over(self, background: Self) -> Self {
        let sa = self.a.clamp(0.0, 1.0);
        let ba = background.a.clamp(0.0, 1.0);
        let out_a = sa + ba * (1.0 - sa);
        if out_a <= 0.0 {
            return Color::TRANSPARENT;
        }
        // Blend in premultiplied space, then un-premultiply by the output alpha.
        let blend = |s: f32, b: f32| (s * sa + b * ba * (1.0 - sa)) / out_a;
        Color::rgba(
            blend(self.r, background.r),
            blend(self.g, background.g),
            blend(self.b, background.b),
            out_a,
        )
    }

    /// Linearly interpolates every channel toward `other` by `t`.
    ///
    /// `t == 0.0` returns `self`, `t == 1.0` returns `other`, and `0.5` returns
    /// the channel-wise midpoint. `t` is clamped to `0.0..=1.0` so overshoot
    /// cannot push a channel out of gamut.
    #[must_use]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let mix = |a: f32, b: f32| a + (b - a) * t;
        Color::rgba(
            mix(self.r, other.r),
            mix(self.g, other.g),
            mix(self.b, other.b),
            mix(self.a, other.a),
        )
    }

    /// Returns the `WCAG` relative luminance of this color's RGB channels.
    ///
    /// Because the channels are already linear light, this is the weighted sum
    /// `0.2126 R + 0.7152 G + 0.0722 B` directly, without an `sRGB` transfer
    /// step. Opaque white yields `1.0` and opaque black yields `0.0`. Alpha is
    /// ignored.
    #[must_use]
    pub fn relative_luminance(self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    /// Returns the `WCAG` contrast ratio between this color and `other`.
    ///
    /// The ratio is `(L_hi + 0.05) / (L_lo + 0.05)` where `L_hi`/`L_lo` are the
    /// larger/smaller [`relative_luminance`](Color::relative_luminance). It is
    /// symmetric and ranges from `1.0` (identical luminance) to `21.0` (black
    /// against white). `WCAG` AA body text wants at least `4.5`.
    #[must_use]
    pub fn contrast_ratio(self, other: Self) -> f32 {
        let a = self.relative_luminance();
        let b = other.relative_luminance();
        let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
        (hi + 0.05) / (lo + 0.05)
    }
}

#[cfg(test)]
mod tests {
    use crate::value::Color;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn with_and_scale_alpha_clamp() {
        let c = Color::rgba(0.2, 0.4, 0.6, 0.8);
        assert!(close(c.with_alpha(0.3).a, 0.3));
        assert!(close(c.with_alpha(2.0).a, 1.0));
        assert!(close(c.with_alpha(-1.0).a, 0.0));
        assert!(close(c.scale_alpha(0.5).a, 0.4));
        assert!(close(c.scale_alpha(10.0).a, 1.0));
        // RGB is preserved.
        assert!(close(c.with_alpha(0.1).r, 0.2));
    }

    #[test]
    fn is_opaque_and_premultiplied() {
        assert!(Color::WHITE.is_opaque());
        assert!(!Color::WHITE.with_alpha(0.5).is_opaque());
        let (r, g, b, a) = Color::rgba(1.0, 0.5, 0.25, 0.5).premultiplied();
        assert!(close(r, 0.5) && close(g, 0.25) && close(b, 0.125) && close(a, 0.5));
    }

    #[test]
    fn over_opaque_source_wins() {
        let red = Color::rgba(1.0, 0.0, 0.0, 1.0);
        let blue = Color::rgba(0.0, 0.0, 1.0, 1.0);
        assert_eq!(red.over(blue), red);
    }

    #[test]
    fn over_transparent_source_is_background() {
        let blue = Color::rgba(0.0, 0.0, 1.0, 1.0);
        let clear = Color::rgba(1.0, 1.0, 1.0, 0.0);
        let out = clear.over(blue);
        assert!(close(out.r, 0.0) && close(out.b, 1.0) && close(out.a, 1.0));
    }

    #[test]
    fn over_fully_transparent_pair_is_transparent() {
        let out = Color::TRANSPARENT.over(Color::TRANSPARENT);
        assert_eq!(out, Color::TRANSPARENT);
    }

    #[test]
    fn over_half_alpha_blend_recomputes() {
        let red = Color::rgba(1.0, 0.0, 0.0, 1.0);
        let blue = Color::rgba(0.0, 0.0, 1.0, 1.0);
        let out = red.with_alpha(0.5).over(blue);
        // out_a = 0.5 + 1.0*0.5 = 1.0; r = (1*0.5)/1 = 0.5; b = (1*0.5)/1 = 0.5.
        assert!(close(out.r, 0.5) && close(out.b, 0.5) && close(out.a, 1.0));
    }

    #[test]
    fn lerp_endpoints_and_midpoint() {
        let a = Color::rgba(0.0, 0.0, 0.0, 0.0);
        let b = Color::rgba(1.0, 0.5, 0.25, 1.0);
        assert_eq!(a.lerp(b, 0.0), a);
        assert_eq!(a.lerp(b, 1.0), b);
        let mid = a.lerp(b, 0.5);
        assert!(close(mid.r, 0.5) && close(mid.g, 0.25) && close(mid.b, 0.125) && close(mid.a, 0.5));
        // t is clamped.
        assert_eq!(a.lerp(b, 2.0), b);
        assert_eq!(a.lerp(b, -1.0), a);
    }

    #[test]
    fn luminance_known_values() {
        assert!(close(Color::WHITE.relative_luminance(), 1.0));
        assert!(close(Color::BLACK.relative_luminance(), 0.0));
        // Pure green dominates per the WCAG weights.
        let green = Color::rgba(0.0, 1.0, 0.0, 1.0);
        assert!(close(green.relative_luminance(), 0.7152));
    }

    #[test]
    fn contrast_black_white_is_21_and_symmetric() {
        // 1.05/0.05 is 21.0 mathematically; allow for f32 rounding of 0.05.
        let ratio = Color::BLACK.contrast_ratio(Color::WHITE);
        assert!((ratio - 21.0).abs() < 1e-3);
        assert!((Color::WHITE.contrast_ratio(Color::BLACK) - 21.0).abs() < 1e-3);
        // A color against itself is 1:1.
        assert!(close(Color::WHITE.contrast_ratio(Color::WHITE), 1.0));
    }

    struct Rng(u32);

    impl Rng {
        fn next_f32(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            (x >> 8) as f32 / (1u32 << 24) as f32
        }
    }

    #[test]
    fn over_opaque_source_is_always_source() {
        let mut rng = Rng(0x1234_5678);
        for _ in 0..4000 {
            let src = Color::rgba(rng.next_f32(), rng.next_f32(), rng.next_f32(), 1.0);
            let bg = Color::rgba(rng.next_f32(), rng.next_f32(), rng.next_f32(), rng.next_f32());
            let out = src.over(bg);
            assert!(close(out.r, src.r) && close(out.g, src.g) && close(out.b, src.b));
            assert!(close(out.a, 1.0));
        }
    }

    #[test]
    fn contrast_matches_independent_formula_and_bounds() {
        let mut rng = Rng(0x0bad_f00d);
        let max_ratio = 21.0 + 1e-4;
        for _ in 0..4000 {
            let a = Color::rgba(rng.next_f32(), rng.next_f32(), rng.next_f32(), 1.0);
            let b = Color::rgba(rng.next_f32(), rng.next_f32(), rng.next_f32(), 1.0);
            let la = 0.2126 * a.r + 0.7152 * a.g + 0.0722 * a.b;
            let lb = 0.2126 * b.r + 0.7152 * b.g + 0.0722 * b.b;
            let (hi, lo) = if la >= lb { (la, lb) } else { (lb, la) };
            let expected = (hi + 0.05) / (lo + 0.05);
            assert!(close(a.contrast_ratio(b), expected));
            // Always within the documented bounds.
            assert!((1.0..=max_ratio).contains(&a.contrast_ratio(b)));
        }
    }
}
