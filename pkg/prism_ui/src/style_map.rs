//! Lowering from resolved style values to concrete layout and paint structs.
//!
//! The style crate produces a flat, token-resolved property map; this module
//! translates that generic map into the strongly-typed [`LayoutStyle`] the
//! flexbox solver consumes and the [`PaintStyle`] the backend consumes.

use alloc::collections::BTreeMap;

use prism_ui_layout::{
    AlignItems, Dimension, Display, Edges, FlexDirection, JustifyContent, LayoutStyle, Size,
};
use prism_ui_style::{Keyword, Length, StyleProp, StyleValue};

use crate::paint::PaintStyle;

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

/// Builds a [`LayoutStyle`] and [`PaintStyle`] from a fully-resolved property
/// map (no [`StyleValue::TokenRef`] entries should remain).
#[must_use]
pub fn build_styles(props: &BTreeMap<StyleProp, StyleValue>) -> (LayoutStyle, PaintStyle) {
    let mut layout = LayoutStyle::default();
    let mut paint = PaintStyle::default();

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
            // `StyleProp` is `#[non_exhaustive]`; ignore properties this
            // lowering does not yet map to a layout or paint field.
            _ => {}
        }
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
