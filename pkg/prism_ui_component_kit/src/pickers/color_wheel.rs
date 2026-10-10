//! [`ColorWheel`] — a continuous HSV color ring.
//!
//! Unlike [`super::color_picker`]'s discrete swatch grid, the wheel is a
//! continuous hue/saturation surface. It renders a `pk-color-wheel` box holding
//! a `pk-color-wheel__ring` and a draggable `pk-color-wheel__thumb`; an optional
//! `pk-color-wheel__alpha` track appears when alpha editing is enabled. The ring
//! approximates a conic hue sweep with a necklace of absolutely-positioned
//! `pk-color-wheel__hue` dots (the renderer has no real conic gradient).
//! Concrete thumb placement on the ring is a backend concern — the control only
//! emits the classed boxes and paints the thumb with the current value as
//! *data*.

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;
use prism_ui_style::StyleValue;

use crate::preset::StyleSheet;

/// Props for [`ColorWheel`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ColorWheelProps {
    /// The current color, used to tint the thumb. `None` leaves the thumb
    /// neutral.
    pub value: Option<StyleValue>,
    /// Whether to render the trailing alpha track.
    pub show_alpha: bool,
}

impl ColorWheelProps {
    /// Creates default wheel props (no value, no alpha track).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the current color value.
    #[must_use]
    pub fn value(mut self, value: StyleValue) -> Self {
        self.value = Some(value);
        self
    }

    /// Enables or disables the alpha track.
    #[must_use]
    pub fn show_alpha(mut self, show_alpha: bool) -> Self {
        self.show_alpha = show_alpha;
        self
    }
}

/// The color-wheel control. Zero-sized; config lives in [`ColorWheelProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ColorWheel;

impl ColorWheel {
    /// The accessibility role a continuous color surface exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for ColorWheel {
    type Props = ColorWheelProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::{StyleProp, StyleValue};

        // Hue necklace: `HUE_DOTS` dots evenly spaced around the ring, each
        // painted with its angle's fully-saturated hue, approximate a conic
        // sweep. The ring is 200px, so its centre sits at (100, 100); dots are
        // anchored in px and recentred by the class's negative margins.
        const HUE_DOTS: usize = 36;
        const CENTER: f32 = 100.0;
        const RADIUS: f32 = 90.0;
        let mut ring = Element::box_().class("pk-color-wheel__ring");
        for i in 0..HUE_DOTS {
            let turns = i as f32 / HUE_DOTS as f32;
            let (cos, sin) = cos_sin_turns(turns);
            let left = CENTER + RADIUS * cos;
            // Top-origin space: a positive sine points up, so subtract it.
            let top = CENTER - RADIUS * sin;
            let (r, g, b) = hue_to_rgb(turns);
            let dot = Element::box_()
                .class("pk-color-wheel__hue")
                .style(StyleProp::Left, StyleValue::px(left))
                .style(StyleProp::Top, StyleValue::px(top))
                .style(StyleProp::BackgroundColor, StyleValue::rgba8(r, g, b, 255));
            ring = ring.child(dot);
        }
        let mut thumb = Element::box_().class("pk-color-wheel__thumb");
        if let Some(value) = &props.value {
            thumb = thumb.style(StyleProp::BackgroundColor, value.clone());
        }
        let mut el = Element::box_()
            .class("pk-color-wheel")
            .child(ring)
            .child(thumb);
        if props.show_alpha {
            el = el.child(Element::box_().class("pk-color-wheel__alpha"));
        }
        el
    }
}

/// Registers the `pk-color-wheel` class family: the frame, the hue ring, the
/// thumb knob and the optional alpha track.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Frame: a square column stacking the ring over the alpha track.
    sheet.insert(
        Class::new("pk-color-wheel")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::Width, StyleValue::px(200.0)),
    );

    // Ring: the circular hue/saturation surface.
    sheet.insert(
        Class::new("pk-color-wheel__ring")
            .with(StyleProp::Width, StyleValue::px(200.0))
            .with(StyleProp::Height, StyleValue::px(200.0))
            .with(StyleProp::MinWidth, StyleValue::px(200.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );

    // Hue dot: a small round swatch absolutely positioned on the ring. The
    // `left`/`top` anchors and the hue fill are set inline per dot; the
    // symmetric -7px margins (half the 14px dot) recentre the anchor on the dot
    // instead of its top-left corner.
    sheet.insert(
        Class::new("pk-color-wheel__hue")
            .with(StyleProp::Position, StyleValue::keyword(Keyword::Absolute))
            .with(StyleProp::Width, StyleValue::px(14.0))
            .with(StyleProp::Height, StyleValue::px(14.0))
            .with(StyleProp::MarginLeft, StyleValue::px(-7.0))
            .with(StyleProp::MarginTop, StyleValue::px(-7.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule")),
    );

    // Thumb: a small white knob with a soft shadow (fill overridden inline).
    sheet.insert(
        Class::new("pk-color-wheel__thumb")
            .with(StyleProp::Width, StyleValue::px(16.0))
            .with(StyleProp::Height, StyleValue::px(16.0))
            .with(StyleProp::MinWidth, StyleValue::px(16.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BorderWidth, StyleValue::px(2.0))
            .with(StyleProp::BorderColor, StyleValue::rgba8(255, 255, 255, 255))
            .with(StyleProp::BackgroundColor, StyleValue::rgba8(255, 255, 255, 255))
            .with_shadow(0.0, 1.0, 3.0, tok("glass.shadow")),
    );

    // Alpha: a thin capsule track for opacity.
    sheet.insert(
        Class::new("pk-color-wheel__alpha")
            .with(StyleProp::Width, StyleValue::percent(100.0))
            .with(StyleProp::Height, StyleValue::px(12.0))
            .with(StyleProp::BorderRadius, tok("radius.capsule"))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary")),
    );
}

/// Returns `(cos, sin)` of an angle expressed in turns (1 turn = 2π rad).
///
/// This is a `no_std` substitute for `f32::sin_cos`: it uses the Bhaskara I
/// sine approximation (max absolute error ≈ 1.6e-3) over a half-turn and
/// mirrors it for the other half, so only multiplies and divides are needed.
fn cos_sin_turns(turns: f32) -> (f32, f32) {
    (sin_turns(turns + 0.25), sin_turns(turns))
}

/// Bhaskara I sine approximation, with `turns` measured in turns (1 turn = 2π rad).
fn sin_turns(turns: f32) -> f32 {
    use core::f32::consts::PI;

    let t = turns.rem_euclid(1.0);
    // Fold onto the positive lobe: the first half-turn maps straight through,
    // the second half-turn reuses it with a flipped sign.
    let (sign, x) = if t < 0.5 {
        (1.0, t * 2.0 * PI)
    } else {
        (-1.0, (t - 0.5) * 2.0 * PI)
    };
    let g = x * (PI - x);
    sign * (16.0 * g) / (5.0 * PI * PI - 4.0 * g)
}

/// Converts a hue (in turns) at full saturation and value to an 8-bit RGB
/// triple. This is the `S = V = 1` slice of HSV, which is all the hue ring
/// needs.
fn hue_to_rgb(turns: f32) -> (u8, u8, u8) {
    let h = turns.rem_euclid(1.0) * 6.0;
    let sector = h as usize;
    let f = h - sector as f32;
    let q = 1.0 - f;
    let (r, g, b) = match sector {
        0 => (1.0, f, 0.0),
        1 => (q, 1.0, 0.0),
        2 => (0.0, 1.0, f),
        3 => (0.0, q, 1.0),
        4 => (f, 0.0, 1.0),
        _ => (1.0, 0.0, q),
    };
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::{StyleProp, StyleValue};

    fn render(props: ColorWheelProps) -> Element {
        ColorWheel.render(&props)
    }

    #[test]
    fn ring_precedes_thumb() {
        let el = render(ColorWheelProps::new());
        assert_eq!(el.class_names(), ["pk-color-wheel"]);
        let kids = el.child_elements();
        assert_eq!(kids.len(), 2);
        assert!(kids[0].class_names().iter().any(|c| c == "pk-color-wheel__ring"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-color-wheel__thumb"));
    }

    #[test]
    fn thumb_is_tinted_by_the_value() {
        let el = render(ColorWheelProps::new().value(StyleValue::token("color.green")));
        let thumb = &el.child_elements()[1];
        let bg = thumb
            .inline_pairs()
            .iter()
            .find(|(p, _)| *p == StyleProp::BackgroundColor)
            .map(|(_, v)| v.clone());
        assert_eq!(bg, Some(StyleValue::token("color.green")));
    }

    #[test]
    fn alpha_track_is_opt_in() {
        let without = render(ColorWheelProps::new());
        assert_eq!(without.child_elements().len(), 2);
        let with = render(ColorWheelProps::new().show_alpha(true));
        assert!(with
            .child_elements()
            .iter()
            .any(|c| c.class_names().iter().any(|n| n == "pk-color-wheel__alpha")));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(ColorWheel::role(), Role::Group);
    }

    #[test]
    fn ring_carries_a_hue_necklace() {
        let el = render(ColorWheelProps::new());
        let ring = &el.child_elements()[0];
        let dots = ring.child_elements();
        assert_eq!(dots.len(), 36);
        for dot in dots {
            assert!(dot.class_names().iter().any(|c| c == "pk-color-wheel__hue"));
            let pairs = dot.inline_pairs();
            assert!(pairs.iter().any(|(p, _)| *p == StyleProp::BackgroundColor));
            assert!(pairs.iter().any(|(p, _)| *p == StyleProp::Left));
            assert!(pairs.iter().any(|(p, _)| *p == StyleProp::Top));
        }
    }

    #[test]
    fn trig_matches_cardinal_turns() {
        for (turns, want_cos, want_sin) in [
            (0.0_f32, 1.0_f32, 0.0_f32),
            (0.25, 0.0, 1.0),
            (0.5, -1.0, 0.0),
            (0.75, 0.0, -1.0),
        ] {
            let (cos, sin) = cos_sin_turns(turns);
            assert!((cos - want_cos).abs() < 0.02, "cos({turns}) = {cos}");
            assert!((sin - want_sin).abs() < 0.02, "sin({turns}) = {sin}");
        }
    }

    #[test]
    fn hue_primaries_are_correct() {
        assert_eq!(hue_to_rgb(0.0), (255, 0, 0));
        assert_eq!(hue_to_rgb(1.0 / 3.0), (0, 255, 0));
        assert_eq!(hue_to_rgb(2.0 / 3.0), (0, 0, 255));
    }
}
