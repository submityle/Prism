//! Lowering from resolved style values to concrete layout and paint structs.
//!
//! The style crate produces a flat, token-resolved property map; this module
//! translates that generic map into the strongly-typed [`LayoutStyle`] the
//! flexbox solver consumes and the [`PaintStyle`] the backend consumes.

use alloc::collections::BTreeMap;

use prism_ui_layout::{
    AlignItems, Dimension, Display, Edges, FlexDirection, JustifyContent, LayoutStyle, Position,
    Size,
};
use prism_ui_style::{Color, Keyword, Length, StyleProp, StyleValue};

use crate::paint::{Glass, PaintStyle, Shadow};

fn length_to_dimension(length: Length) -> Dimension {
    match length {
        Length::Px(px) => Dimension::Points(px),
        Length::Percent(pct) => Dimension::Percent(pct / 100.0),
        Length::Auto => Dimension::Auto,
    }
}

fn as_dimension(value: &StyleValue) -> Option<Dimension> {
    match value {
        StyleValue::Length(length) => Some(length_to_dimension(*length)),
        StyleValue::Number(n) => Some(Dimension::Points(*n)),
        _ => None,
    }
}

fn as_f32(value: &StyleValue) -> Option<f32> {
    match value {
        StyleValue::Number(n) => Some(*n),
        StyleValue::Length(Length::Px(px)) => Some(*px),
        _ => None,
    }
}

fn as_color(value: &StyleValue) -> Option<Color> {
    match value {
        StyleValue::Color(c) => Some(*c),
        _ => None,
    }
}

/// Builds a [`LayoutStyle`] and [`PaintStyle`] from a fully-resolved property
/// map (no [`StyleValue::TokenRef`] entries should remain).
#[must_use]
pub fn build_styles(props: &BTreeMap<StyleProp, StyleValue>) -> (LayoutStyle, PaintStyle) {
    let mut layout = LayoutStyle::default();
    let mut paint = PaintStyle::default();

    // Shadow / glass are composite paint effects. The DSL exposes them as
    // physical longhands (last-writer-wins per field, like every other prop);
    // we accumulate the pieces here and assemble them once after the loop.
    let mut shadow_seen = false;
    let mut shadow_offset_x = 0.0_f32;
    let mut shadow_offset_y = 0.0_f32;
    let mut shadow_blur = 0.0_f32;
    let mut shadow_spread = 0.0_f32;
    let mut shadow_color: Option<Color> = None;
    let mut shadow_inset = false;
    let mut glass_blur = 0.0_f32;
    let mut glass_tint: Option<Color> = None;
    let mut glass_highlight: Option<Color> = None;

    for (prop, value) in props {
        match prop {
            StyleProp::Width => {
                if let Some(d) = as_dimension(value) {
                    layout.size.width = d;
                }
            }
            StyleProp::Height => {
                if let Some(d) = as_dimension(value) {
                    layout.size.height = d;
                }
            }
            StyleProp::MinWidth => {
                if let Some(d) = as_dimension(value) {
                    layout.min_size.width = d;
                }
            }
            StyleProp::MinHeight => {
                if let Some(d) = as_dimension(value) {
                    layout.min_size.height = d;
                }
            }
            StyleProp::MaxWidth => {
                if let Some(d) = as_dimension(value) {
                    layout.max_size.width = d;
                }
            }
            StyleProp::MaxHeight => {
                if let Some(d) = as_dimension(value) {
                    layout.max_size.height = d;
                }
            }
            StyleProp::PaddingTop => set_edge(&mut layout.padding, Edge::Top, value),
            StyleProp::PaddingRight => set_edge(&mut layout.padding, Edge::Right, value),
            StyleProp::PaddingBottom => set_edge(&mut layout.padding, Edge::Bottom, value),
            StyleProp::PaddingLeft => set_edge(&mut layout.padding, Edge::Left, value),
            StyleProp::MarginTop => set_edge(&mut layout.margin, Edge::Top, value),
            StyleProp::MarginRight => set_edge(&mut layout.margin, Edge::Right, value),
            StyleProp::MarginBottom => set_edge(&mut layout.margin, Edge::Bottom, value),
            StyleProp::MarginLeft => set_edge(&mut layout.margin, Edge::Left, value),
            StyleProp::FlexGrow => {
                if let Some(n) = as_f32(value) {
                    layout.flex_grow = n;
                }
            }
            StyleProp::FlexShrink => {
                if let Some(n) = as_f32(value) {
                    layout.flex_shrink = n;
                }
            }
            StyleProp::FlexBasis => {
                if let Some(d) = as_dimension(value) {
                    layout.flex_basis = d;
                }
            }
            StyleProp::Gap => {
                if let Some(n) = as_f32(value) {
                    layout.gap = Size::new(n, n);
                }
            }
            StyleProp::RowGap => {
                if let Some(n) = as_f32(value) {
                    layout.gap.height = n;
                }
            }
            StyleProp::ColumnGap => {
                if let Some(n) = as_f32(value) {
                    layout.gap.width = n;
                }
            }
            StyleProp::Display => {
                if let StyleValue::Keyword(Keyword::None) = value {
                    layout.display = Display::None;
                } else {
                    layout.display = Display::Flex;
                }
            }
            StyleProp::FlexDirection => {
                if let StyleValue::Keyword(kw) = value {
                    layout.flex_direction = match kw {
                        Keyword::Column => FlexDirection::Column,
                        Keyword::RowReverse => FlexDirection::RowReverse,
                        Keyword::ColumnReverse => FlexDirection::ColumnReverse,
                        _ => FlexDirection::Row,
                    };
                }
            }
            StyleProp::JustifyContent => {
                if let StyleValue::Keyword(kw) = value {
                    layout.justify_content = match kw {
                        Keyword::End => JustifyContent::End,
                        Keyword::Center => JustifyContent::Center,
                        Keyword::SpaceBetween => JustifyContent::SpaceBetween,
                        Keyword::SpaceAround => JustifyContent::SpaceAround,
                        Keyword::SpaceEvenly => JustifyContent::SpaceEvenly,
                        _ => JustifyContent::Start,
                    };
                }
            }
            StyleProp::AlignItems => {
                if let StyleValue::Keyword(kw) = value {
                    layout.align_items = match kw {
                        Keyword::Start => AlignItems::Start,
                        Keyword::End => AlignItems::End,
                        Keyword::Center => AlignItems::Center,
                        _ => AlignItems::Stretch,
                    };
                }
            }
            StyleProp::Position => {
                if let StyleValue::Keyword(Keyword::Absolute) = value {
                    layout.position = Position::Absolute;
                } else {
                    layout.position = Position::Relative;
                }
            }
            StyleProp::Top => set_edge(&mut layout.inset, Edge::Top, value),
            StyleProp::Right => set_edge(&mut layout.inset, Edge::Right, value),
            StyleProp::Bottom => set_edge(&mut layout.inset, Edge::Bottom, value),
            StyleProp::Left => set_edge(&mut layout.inset, Edge::Left, value),
            StyleProp::Color => {
                if let StyleValue::Color(c) = value {
                    paint.color = Some(*c);
                }
            }
            StyleProp::BackgroundColor => {
                if let StyleValue::Color(c) = value {
                    paint.background_color = Some(*c);
                }
            }
            StyleProp::BorderColor => {
                if let StyleValue::Color(c) = value {
                    paint.border_color = Some(*c);
                }
            }
            StyleProp::BorderRadius => {
                if let Some(n) = as_f32(value) {
                    paint.border_radius = n;
                }
            }
            StyleProp::BorderWidth => {
                if let Some(n) = as_f32(value) {
                    paint.border_width = n;
                    layout.border = Edges::splat(Dimension::Points(n));
                }
            }
            StyleProp::Opacity => {
                if let Some(n) = as_f32(value) {
                    paint.opacity = n;
                }
            }
            StyleProp::FontSize => {
                if let Some(n) = as_f32(value) {
                    paint.font_size = n;
                }
            }
            StyleProp::FontWeight => {
                if let Some(n) = as_f32(value) {
                    paint.font_weight = n;
                }
            }
            StyleProp::ShadowOffsetX => {
                if let Some(n) = as_f32(value) {
                    shadow_offset_x = n;
                    shadow_seen = true;
                }
            }
            StyleProp::ShadowOffsetY => {
                if let Some(n) = as_f32(value) {
                    shadow_offset_y = n;
                    shadow_seen = true;
                }
            }
            StyleProp::ShadowBlur => {
                if let Some(n) = as_f32(value) {
                    shadow_blur = n;
                    shadow_seen = true;
                }
            }
            StyleProp::ShadowSpread => {
                if let Some(n) = as_f32(value) {
                    shadow_spread = n;
                    shadow_seen = true;
                }
            }
            StyleProp::ShadowColor => {
                if let Some(c) = as_color(value) {
                    shadow_color = Some(c);
                    shadow_seen = true;
                }
            }
            StyleProp::ShadowInset => {
                match value {
                    StyleValue::Keyword(Keyword::Inset) => {
                        shadow_inset = true;
                        shadow_seen = true;
                    }
                    StyleValue::Keyword(Keyword::None) => shadow_inset = false,
                    StyleValue::Number(n) => {
                        shadow_inset = *n != 0.0;
                        shadow_seen = true;
                    }
                    _ => {}
                }
            }
            StyleProp::GlassBlur => {
                if let Some(n) = as_f32(value) {
                    glass_blur = n;
                }
            }
            StyleProp::GlassTint => {
                if let Some(c) = as_color(value) {
                    glass_tint = Some(c);
                }
            }
            StyleProp::GlassHighlight => {
                if let Some(c) = as_color(value) {
                    glass_highlight = Some(c);
                }
            }
            // `StyleProp` is `#[non_exhaustive]`; ignore properties this
            // lowering does not yet map to a layout or paint field.
            _ => {}
        }
    }

    // Assemble composite effects from the collected longhands. A shadow with
    // no explicit color falls back to a soft neutral drop so a bare
    // `shadow-offset`/`shadow-blur` still renders something sensible.
    if shadow_seen {
        paint.shadow = Some(Shadow {
            offset_x: shadow_offset_x,
            offset_y: shadow_offset_y,
            blur: shadow_blur,
            spread: shadow_spread,
            color: shadow_color.unwrap_or(Color::rgba(0.0, 0.0, 0.0, 0.33)),
            inset: shadow_inset,
        });
    }
    // The tint is the essential ingredient: a box becomes glass only once a
    // translucent tint is declared. Blur and highlight are optional refinements.
    if let Some(tint) = glass_tint {
        let mut glass = Glass::new(glass_blur, tint);
        if let Some(highlight) = glass_highlight {
            glass = glass.with_highlight(highlight);
        }
        paint.glass = Some(glass);
    }

    (layout, paint)
}

enum Edge {
    Top,
    Right,
    Bottom,
    Left,
}

fn set_edge(edges: &mut Edges<Dimension>, edge: Edge, value: &StyleValue) {
    if let Some(d) = as_dimension(value) {
        match edge {
            Edge::Top => edges.top = d,
            Edge::Right => edges.right = d,
            Edge::Bottom => edges.bottom = d,
            Edge::Left => edges.left = d,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::Color;

    fn props(entries: &[(StyleProp, StyleValue)]) -> BTreeMap<StyleProp, StyleValue> {
        entries.iter().cloned().collect()
    }

    #[test]
    fn shadow_longhands_assemble_into_paint() {
        let (_, paint) = build_styles(&props(&[
            (StyleProp::ShadowOffsetX, StyleValue::px(0.0)),
            (StyleProp::ShadowOffsetY, StyleValue::px(8.0)),
            (StyleProp::ShadowBlur, StyleValue::px(24.0)),
            (StyleProp::ShadowSpread, StyleValue::px(2.0)),
            (StyleProp::ShadowColor, StyleValue::rgba8(0, 0, 0, 38)),
        ]));
        let sh = paint.shadow.expect("shadow assembled");
        assert_eq!(sh.offset_y, 8.0);
        assert_eq!(sh.blur, 24.0);
        assert_eq!(sh.spread, 2.0);
        assert_eq!(sh.color, Color::rgba8(0, 0, 0, 38));
        assert!(!sh.inset);
    }

    #[test]
    fn shadow_without_color_uses_neutral_fallback() {
        let (_, paint) = build_styles(&props(&[(StyleProp::ShadowBlur, StyleValue::px(10.0))]));
        let sh = paint.shadow.expect("shadow present from a single longhand");
        assert_eq!(sh.blur, 10.0);
        assert_eq!(sh.color, Color::rgba(0.0, 0.0, 0.0, 0.33));
    }

    #[test]
    fn shadow_inset_keyword_flags_inner_shadow() {
        let (_, paint) = build_styles(&props(&[
            (StyleProp::ShadowBlur, StyleValue::px(6.0)),
            (StyleProp::ShadowInset, StyleValue::keyword(Keyword::Inset)),
        ]));
        assert!(paint.shadow.expect("shadow present").inset);
    }

    #[test]
    fn no_shadow_longhands_leaves_shadow_none() {
        let (_, paint) = build_styles(&props(&[(StyleProp::Opacity, StyleValue::number(1.0))]));
        assert!(paint.shadow.is_none());
    }

    #[test]
    fn glass_tint_turns_box_into_glass() {
        let (_, paint) = build_styles(&props(&[
            (StyleProp::GlassBlur, StyleValue::px(20.0)),
            (StyleProp::GlassTint, StyleValue::rgba8(255, 255, 255, 160)),
            (StyleProp::GlassHighlight, StyleValue::rgba8(255, 255, 255, 115)),
        ]));
        let glass = paint.glass.expect("glass assembled");
        assert_eq!(glass.blur, 20.0);
        assert_eq!(glass.tint, Color::rgba8(255, 255, 255, 160));
        assert_eq!(glass.highlight, Some(Color::rgba8(255, 255, 255, 115)));
    }

    #[test]
    fn glass_without_tint_is_not_glass() {
        // Blur/highlight alone are not enough; the tint is the trigger.
        let (_, paint) = build_styles(&props(&[
            (StyleProp::GlassBlur, StyleValue::px(20.0)),
            (StyleProp::GlassHighlight, StyleValue::rgba8(255, 255, 255, 115)),
        ]));
        assert!(paint.glass.is_none());
    }
}
