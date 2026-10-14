//! Element representation of renderer-independent dock drop-preview geometry.
//!
//! The preview is deliberately represented by semantic classes and token
//! references instead of renderer-specific colors. This keeps the overlay
//! cheap to rebuild while allowing a host theme to change the fill, border
//! and contrast roles without changing dock code.

use prism_ui::style::{Keyword, StyleProp, StyleValue};
use prism_ui::Element;

#[cfg(feature = "input")]
use prism_ui_input::{HitNode, NodeId, PointerEvents};

use crate::{DropPreview, DropZone, Side};

/// Base class attached to every dock drop preview.
pub const DROP_PREVIEW_CLASS: &str = "pk-dock-drop-preview";
/// Marker class for the required input-tree `pointer-events: none` behavior.
///
/// `Element` does not own the input hit-test tree. The overlay host must map
/// this marker to [`prism_ui_input::PointerEvents::None`] (or omit the node
/// from that tree) when it builds input nodes.
pub const DROP_PREVIEW_POINTER_EVENTS_NONE_CLASS: &str =
    "pk-dock-drop-preview--pointer-events-none";
/// Marker class for the theme-controlled contrast role.
pub const DROP_PREVIEW_CONTRAST_CLASS: &str = "pk-dock-drop-preview--contrast";

/// Semantic token used for the preview fill.
pub const DROP_PREVIEW_BACKGROUND_TOKEN: &str = "color.primary";
/// Semantic token used for the preview border.
pub const DROP_PREVIEW_BORDER_TOKEN: &str = "color.text";
/// Semantic token used for optional preview content and contrast-aware
/// backend overlays.
pub const DROP_PREVIEW_CONTRAST_TOKEN: &str = "color.text";
/// Semantic token used for the preview corner radius.
pub const DROP_PREVIEW_RADIUS_TOKEN: &str = "radius.sm";

/// The semantic operation represented by a drop preview.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropPreviewKind {
    /// Drop into the target's existing tab stack.
    Center,
    /// Create a new split beside the target.
    Split,
    /// Insert a tab at a specific tab boundary.
    TabInsertion,
}

impl DropPreviewKind {
    /// Class name for this semantic kind.
    #[must_use]
    pub const fn class_name(self) -> &'static str {
        match self {
            Self::Center => "pk-dock-drop-preview--center",
            Self::Split => "pk-dock-drop-preview--split",
            Self::TabInsertion => "pk-dock-drop-preview--tab-insertion",
        }
    }

    /// Classifies a reducer drop zone without inspecting its geometry.
    #[must_use]
    pub const fn from_zone(zone: &DropZone) -> Self {
        match zone {
            DropZone::CenterTab { .. } => Self::Center,
            DropZone::Split { .. } => Self::Split,
            DropZone::TabInsert { .. } => Self::TabInsertion,
        }
    }
}

/// Theme-facing paint roles for a drop preview.
///
/// Fields intentionally use [`StyleValue`] rather than concrete colors. A
/// caller can pass token references for light/dark/high-contrast themes or
/// literals for a local overlay theme. The defaults use tokens present in the
/// standard Prism UI palettes.
#[derive(Clone, Debug, PartialEq)]
pub struct DropPreviewVisualStyle {
    /// Fill color of the highlighted region.
    pub background: StyleValue,
    /// Border color of the highlighted region.
    pub border: StyleValue,
    /// Foreground/contrast role reserved for optional labels or backend
    /// marker primitives associated with the preview.
    pub contrast: StyleValue,
    /// Border width in logical pixels, or a theme length/number.
    pub border_width: StyleValue,
    /// Corner radius in logical pixels, or a theme length/number.
    pub border_radius: StyleValue,
    /// Overall preview opacity in the inclusive `0..=1` range.
    pub opacity: StyleValue,
}

impl Default for DropPreviewVisualStyle {
    fn default() -> Self {
        Self {
            background: StyleValue::token(DROP_PREVIEW_BACKGROUND_TOKEN),
            border: StyleValue::token(DROP_PREVIEW_BORDER_TOKEN),
            contrast: StyleValue::token(DROP_PREVIEW_CONTRAST_TOKEN),
            border_width: StyleValue::px(2.0),
            border_radius: StyleValue::token(DROP_PREVIEW_RADIUS_TOKEN),
            opacity: StyleValue::number(0.22),
        }
    }
}

/// Builds an absolutely positioned, pointer-transparent preview Element.
///
/// The returned element is suitable as the content of a higher-level portal
/// overlay. It has no event handlers and does not mutate the dock model. The
/// attached [`DROP_PREVIEW_POINTER_EVENTS_NONE_CLASS`] is the explicit handoff
/// to the host's input tree, which must make the preview hit-test transparent.
#[must_use]
pub fn drop_preview_element(preview: &DropPreview) -> Element {
    drop_preview_element_with_style(preview, &DropPreviewVisualStyle::default())
}

/// Builds a drop preview with caller-supplied theme-facing paint roles.
///
/// Geometry and semantic classes remain identical to
/// [`drop_preview_element`]. Only the token/literal values for background,
/// border, contrast, radius and opacity are replaced.
#[must_use]
pub fn drop_preview_element_with_style(
    preview: &DropPreview,
    visual_style: &DropPreviewVisualStyle,
) -> Element {
    let rect = preview.highlight_rect;
    let kind = DropPreviewKind::from_zone(&preview.zone);
    let mut element = Element::box_()
        .class(DROP_PREVIEW_CLASS)
        .class(kind.class_name())
        .class(DROP_PREVIEW_POINTER_EVENTS_NONE_CLASS)
        .class(DROP_PREVIEW_CONTRAST_CLASS)
        .key_str(format!(
            "dock-preview:{}:{}:{}:{}",
            rect.x, rect.y, rect.width, rect.height
        ))
        .style(StyleProp::Position, StyleValue::keyword(Keyword::Absolute))
        .style(StyleProp::Left, StyleValue::px(rect.x as f32))
        .style(StyleProp::Top, StyleValue::px(rect.y as f32))
        .style(StyleProp::Width, StyleValue::px(rect.width as f32))
        .style(StyleProp::Height, StyleValue::px(rect.height as f32))
        .style(StyleProp::BackgroundColor, visual_style.background.clone())
        .style(StyleProp::BorderColor, visual_style.border.clone())
        .style(StyleProp::Color, visual_style.contrast.clone())
        .style(StyleProp::BorderWidth, visual_style.border_width.clone())
        .style(StyleProp::BorderRadius, visual_style.border_radius.clone())
        .style(StyleProp::Opacity, visual_style.opacity.clone());

    if let DropZone::Split { side, .. } = preview.zone {
        element = element.class(split_side_class(side));
    }

    element
}

/// Builds the parallel hit-test node for a preview overlay.
///
/// The retained `Element` tree does not own input geometry. Hosts that build
/// a `prism_ui_input` tree can use this helper to preserve the preview's
/// geometry while making it transparent to pointer hits.
#[cfg(feature = "input")]
#[must_use]
pub fn drop_preview_hit_node(preview: &DropPreview, node_id: NodeId) -> HitNode {
    let rect = preview.highlight_rect;
    HitNode::new(
        node_id,
        prism_ui_input::geometry::Rect::new(
            prism_ui_input::geometry::Point::new(rect.x as f32, rect.y as f32),
            prism_ui_input::geometry::Size::new(rect.width as f32, rect.height as f32),
        ),
    )
    .with_pointer_events(PointerEvents::None)
}

const fn split_side_class(side: Side) -> &'static str {
    match side {
        Side::Left => "pk-dock-drop-preview--split-left",
        Side::Right => "pk-dock-drop-preview--split-right",
        Side::Top => "pk-dock-drop-preview--split-top",
        Side::Bottom => "pk-dock-drop-preview--split-bottom",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DropZone, PanelId, Rect, SurfaceId, Target};

    fn preview(zone: DropZone) -> DropPreview {
        DropPreview {
            zone,
            surface: SurfaceId::Main,
            target_rect: Rect {
                x: 0,
                y: 0,
                width: 320,
                height: 200,
            },
            highlight_rect: Rect {
                x: 12,
                y: 24,
                width: 96,
                height: 80,
            },
        }
    }

    fn has_class(element: &Element, class: &str) -> bool {
        element.class_names().iter().any(|name| name == class)
    }

    fn style_value(element: &Element, prop: StyleProp) -> Option<&StyleValue> {
        element
            .inline_pairs()
            .iter()
            .find_map(|(candidate, value)| (*candidate == prop).then_some(value))
    }

    #[test]
    fn preview_element_is_absolute_pointer_transparent_and_geometry_stable() {
        let element = drop_preview_element(&preview(DropZone::CenterTab {
            target: Target::Panel(PanelId::from("scene")),
        }));
        assert!(has_class(&element, DROP_PREVIEW_CLASS));
        assert!(has_class(&element, DropPreviewKind::Center.class_name()));
        assert!(has_class(&element, DROP_PREVIEW_POINTER_EVENTS_NONE_CLASS));
        assert!(has_class(&element, DROP_PREVIEW_CONTRAST_CLASS));
        assert_eq!(
            element.inline_pairs()[0],
            (StyleProp::Position, StyleValue::keyword(Keyword::Absolute))
        );
        assert_eq!(
            element.inline_pairs()[1],
            (StyleProp::Left, StyleValue::px(12.0))
        );
        assert_eq!(
            element.inline_pairs()[4],
            (StyleProp::Height, StyleValue::px(80.0))
        );
        assert_eq!(
            style_value(&element, StyleProp::BackgroundColor),
            Some(&StyleValue::token(DROP_PREVIEW_BACKGROUND_TOKEN))
        );
        assert_eq!(
            style_value(&element, StyleProp::BorderColor),
            Some(&StyleValue::token(DROP_PREVIEW_BORDER_TOKEN))
        );
        assert_eq!(
            style_value(&element, StyleProp::Color),
            Some(&StyleValue::token(DROP_PREVIEW_CONTRAST_TOKEN))
        );
    }

    #[test]
    fn preview_kind_classes_are_distinct_and_split_has_direction() {
        let center = drop_preview_element(&preview(DropZone::CenterTab {
            target: Target::Panel(PanelId::from("scene")),
        }));
        let split = drop_preview_element(&preview(DropZone::Split {
            target: Target::Panel(PanelId::from("scene")),
            side: Side::Right,
        }));
        let insertion = drop_preview_element(&preview(DropZone::TabInsert {
            target: PanelId::from("scene"),
            index: 1,
            tab_count: 2,
        }));

        assert!(has_class(&center, "pk-dock-drop-preview--center"));
        assert!(!has_class(&center, "pk-dock-drop-preview--split"));
        assert!(has_class(&split, "pk-dock-drop-preview--split"));
        assert!(has_class(&split, "pk-dock-drop-preview--split-right"));
        assert!(has_class(&insertion, "pk-dock-drop-preview--tab-insertion"));
        assert!(!has_class(&insertion, "pk-dock-drop-preview--center"));
    }

    #[test]
    fn visual_style_accepts_theme_tokens_without_changing_contract() {
        let visual_style = DropPreviewVisualStyle {
            background: StyleValue::token("dock.preview.background"),
            border: StyleValue::token("dock.preview.border"),
            contrast: StyleValue::token("dock.preview.contrast"),
            border_width: StyleValue::token("dock.preview.border-width"),
            border_radius: StyleValue::token("dock.preview.radius"),
            opacity: StyleValue::token("dock.preview.opacity"),
        };
        let element = drop_preview_element_with_style(
            &preview(DropZone::TabInsert {
                target: PanelId::from("scene"),
                index: 0,
                tab_count: 2,
            }),
            &visual_style,
        );

        assert!(has_class(&element, DROP_PREVIEW_POINTER_EVENTS_NONE_CLASS));
        assert_eq!(
            style_value(&element, StyleProp::BackgroundColor),
            Some(&StyleValue::token("dock.preview.background"))
        );
        assert_eq!(
            style_value(&element, StyleProp::BorderWidth),
            Some(&StyleValue::token("dock.preview.border-width"))
        );
        assert_eq!(
            style_value(&element, StyleProp::Opacity),
            Some(&StyleValue::token("dock.preview.opacity"))
        );
    }

    #[test]
    fn default_visual_tokens_resolve_in_standard_ui_theme() {
        let theme = prism_ui::style::Theme::with_default_palette();
        let visual_style = DropPreviewVisualStyle::default();

        for value in [
            &visual_style.background,
            &visual_style.border,
            &visual_style.contrast,
            &visual_style.border_radius,
        ] {
            assert!(
                theme.tokens.resolve_value(value).is_ok(),
                "default preview token must be available in the standard theme"
            );
        }
    }
}
