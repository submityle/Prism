//! Element representation of renderer-independent dock drop-preview geometry.

use prism_ui::style::{Keyword, StyleProp, StyleValue};
use prism_ui::Element;

use crate::DropPreview;

/// Builds an absolutely positioned, pointer-transparent preview Element.
///
/// The returned element is suitable as the content of a higher-level portal
/// overlay. It has no event handlers and does not mutate the dock model.
#[must_use]
pub fn drop_preview_element(preview: &DropPreview) -> Element {
    let rect = preview.highlight_rect;
    Element::box_()
        .class("pk-dock-drop-preview")
        .key_str(format!(
            "dock-preview:{}:{}:{}:{}",
            rect.x, rect.y, rect.width, rect.height
        ))
        .style(StyleProp::Position, StyleValue::keyword(Keyword::Absolute))
        .style(StyleProp::Left, StyleValue::px(rect.x as f32))
        .style(StyleProp::Top, StyleValue::px(rect.y as f32))
        .style(StyleProp::Width, StyleValue::px(rect.width as f32))
        .style(StyleProp::Height, StyleValue::px(rect.height as f32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DropZone, PanelId, SurfaceId, Target};

    #[test]
    fn preview_element_is_absolute_and_geometry_stable() {
        let preview = DropPreview {
            zone: DropZone::CenterTab {
                target: Target::Panel(PanelId::from("scene")),
            },
            surface: SurfaceId::Main,
            target_rect: crate::Rect {
                x: 0,
                y: 0,
                width: 320,
                height: 200,
            },
            highlight_rect: crate::Rect {
                x: 12,
                y: 24,
                width: 96,
                height: 80,
            },
        };
        let element = drop_preview_element(&preview);
        assert_eq!(element.class_names(), ["pk-dock-drop-preview"]);
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
    }
}
