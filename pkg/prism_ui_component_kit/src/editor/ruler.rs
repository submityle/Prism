//! [`Ruler`] / [`Guides`] — a ruler strip with tick marks and an optional guide.
//!
//! A ruler renders a `pk-ruler` strip of evenly spaced `pk-ruler__tick` boxes
//! with an optional `pk-ruler__guide` line marking a position. A
//! [`RulerOrientation`] selects a `--horizontal` / `--vertical` modifier so the
//! strip lays out along the correct axis; the guide offset is a data-derived
//! inline margin (percentage), never a color/shadow literal. Surface and tick
//! color resolve from theme tokens via [`crate::preset`].

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::preset::StyleSheet;

/// Clamps `value` into the inclusive range `0.0..=1.0`.
///
/// Hand-rolled replacement for the std-only `f32::clamp`; `NaN` collapses to
/// `0.0` because neither comparison holds for it.
fn clamp01(value: f32) -> f32 {
    if value > 1.0 {
        1.0
    } else if value > 0.0 {
        value
    } else {
        0.0
    }
}

/// The axis a [`Ruler`] runs along.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum RulerOrientation {
    /// A horizontal strip, ticks running left-to-right.
    #[default]
    Horizontal,
    /// A vertical strip, ticks running top-to-bottom.
    Vertical,
}

impl RulerOrientation {
    /// The modifier suffix used in class names (e.g. `pk-ruler--horizontal`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            RulerOrientation::Horizontal => "horizontal",
            RulerOrientation::Vertical => "vertical",
        }
    }
}

/// Props for [`Ruler`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct RulerProps {
    /// The ruler's orientation.
    pub orientation: RulerOrientation,
    /// The number of evenly spaced tick marks.
    pub ticks: usize,
    /// Optional guide-line position in `0.0..=1.0` along the strip.
    pub guide: Option<f32>,
}

impl RulerProps {
    /// Creates props for a horizontal ruler with `ticks` tick marks.
    #[must_use]
    pub fn new(ticks: usize) -> Self {
        Self {
            ticks,
            ..Self::default()
        }
    }

    /// Sets the orientation.
    #[must_use]
    pub fn orientation(mut self, orientation: RulerOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Sets the number of tick marks.
    #[must_use]
    pub fn ticks(mut self, ticks: usize) -> Self {
        self.ticks = ticks;
        self
    }

    /// Sets the guide-line position (clamped to `0.0..=1.0`).
    #[must_use]
    pub fn guide(mut self, position: f32) -> Self {
        self.guide = Some(clamp01(position));
        self
    }
}

/// The ruler control. Zero-sized; all configuration lives in [`RulerProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Ruler;

/// Alias matching the `Guides` naming used by the design doc (section 12).
pub type Guides = Ruler;

impl Ruler {
    /// The accessibility role a ruler exposes (purely decorative chrome).
    #[must_use]
    pub const fn role() -> Role {
        Role::Presentation
    }
}

impl Component for Ruler {
    type Props = RulerProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::{StyleProp, StyleValue};

        let mut el = Element::box_();
        for name in classes("pk-ruler", &[props.orientation.suffix()]) {
            el = el.class(name);
        }

        for _ in 0..props.ticks {
            el = el.child(Element::box_().class("pk-ruler__tick"));
        }

        if let Some(position) = props.guide {
            let offset = StyleValue::percent(clamp01(position) * 100.0);
            let prop = match props.orientation {
                RulerOrientation::Horizontal => StyleProp::MarginLeft,
                RulerOrientation::Vertical => StyleProp::MarginTop,
            };
            el = el.child(Element::box_().class("pk-ruler__guide").style(prop, offset));
        }
        el
    }
}

/// Registers the `pk-ruler` family: strip, orientation modifiers, tick, guide.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Base: a thin surface strip of evenly distributed ticks.
    sheet.insert(
        Class::new("pk-ruler")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::SpaceBetween))
            .with(StyleProp::BackgroundColor, tok("color.surface.secondary")),
    );

    sheet.insert(
        Class::new("pk-ruler--horizontal")
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::Height, StyleValue::px(20.0))
            .with_padding_x(tok("space.xxs")),
    );
    sheet.insert(
        Class::new("pk-ruler--vertical")
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Width, StyleValue::px(20.0))
            .with_padding_y(tok("space.xxs")),
    );

    // Tick: a hairline mark that never shrinks.
    sheet.insert(
        Class::new("pk-ruler__tick")
            .with(StyleProp::Width, StyleValue::px(1.0))
            .with(StyleProp::Height, StyleValue::px(8.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::BackgroundColor, tok("color.separator")),
    );

    // Guide: a thin accent line offset along the strip.
    sheet.insert(
        Class::new("pk-ruler__guide")
            .with(StyleProp::Width, StyleValue::px(1.0))
            .with(StyleProp::Height, StyleValue::px(16.0))
            .with(StyleProp::BackgroundColor, tok("color.tint")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::{StyleProp, StyleValue};

    fn render(props: RulerProps) -> Element {
        Ruler.render(&props)
    }

    #[test]
    fn attaches_base_and_orientation_classes_in_order() {
        let el = render(RulerProps::new(0).orientation(RulerOrientation::Vertical));
        assert_eq!(el.class_names(), ["pk-ruler", "pk-ruler--vertical"]);
    }

    #[test]
    fn renders_one_tick_box_per_tick() {
        let el = render(RulerProps::new(4));
        let ticks = el.child_elements();
        assert_eq!(ticks.len(), 4);
        for tick in ticks {
            assert_eq!(tick.class_names(), ["pk-ruler__tick"]);
        }
    }

    #[test]
    fn guide_appends_a_positioned_line() {
        let el = render(RulerProps::new(2).guide(0.5));
        let guide = el.child_elements().last().unwrap();
        assert_eq!(guide.class_names(), ["pk-ruler__guide"]);
        assert_eq!(
            guide.inline_pairs(),
            [(StyleProp::MarginLeft, StyleValue::percent(50.0))]
        );
    }

    #[test]
    fn vertical_guide_uses_margin_top() {
        let el = render(
            RulerProps::new(1)
                .orientation(RulerOrientation::Vertical)
                .guide(0.25),
        );
        let guide = el.child_elements().last().unwrap();
        assert_eq!(
            guide.inline_pairs(),
            [(StyleProp::MarginTop, StyleValue::percent(25.0))]
        );
    }

    #[test]
    fn role_is_presentation() {
        assert_eq!(Ruler::role(), Role::Presentation);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in [
            "pk-ruler",
            "pk-ruler--horizontal",
            "pk-ruler--vertical",
            "pk-ruler__tick",
            "pk-ruler__guide",
        ] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
