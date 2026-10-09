//! [`Timeline`] — a vertical sequence of dated/ordered events.
//!
//! A timeline renders a `pk-timeline` container whose children are
//! `pk-timeline__item` rows. Each item pairs a tone-colored `pk-timeline__dot`
//! marker with a `pk-timeline__content` slot. Color and spacing resolve from
//! theme tokens via [`crate::preset`]; the control attaches only kit classes.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::{classes, Tone};
use crate::preset::StyleSheet;

/// A single timeline entry: a content element plus the tone of its dot marker.
#[derive(Clone, Debug, PartialEq)]
pub struct TimelineItem {
    /// The event content rendered beside the dot.
    pub content: Element,
    /// Semantic tone selecting the dot's color.
    pub tone: Tone,
}

impl TimelineItem {
    /// Creates an item with the default (accent) dot tone.
    #[must_use]
    pub fn new(content: Element) -> Self {
        Self {
            content,
            tone: Tone::default(),
        }
    }

    /// Sets the dot tone.
    #[must_use]
    pub fn tone(mut self, tone: Tone) -> Self {
        self.tone = tone;
        self
    }
}

/// Props for [`Timeline`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TimelineProps {
    /// The ordered events, rendered top-to-bottom.
    pub items: Vec<TimelineItem>,
}

impl TimelineProps {
    /// Creates empty timeline props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an event item.
    #[must_use]
    pub fn item(mut self, item: TimelineItem) -> Self {
        self.items.push(item);
        self
    }

    /// Replaces the items with `items`.
    #[must_use]
    pub fn items<I: IntoIterator<Item = TimelineItem>>(mut self, items: I) -> Self {
        self.items = items.into_iter().collect();
        self
    }
}

/// The timeline control. Zero-sized; config lives in [`TimelineProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Timeline;

impl Component for Timeline {
    type Props = TimelineProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-timeline");
        for item in &props.items {
            let mut dot = Element::box_();
            for name in classes("pk-timeline__dot", &[item.tone.suffix()]) {
                dot = dot.class(name);
            }
            let content = item.content.clone().class("pk-timeline__content");
            let row = Element::box_()
                .class("pk-timeline__item")
                .child(dot)
                .child(content);
            el = el.child(row);
        }
        el
    }
}

/// Builds the `--tone` modifier name for a block, e.g. `pk-timeline__dot--success`.
fn tone_class(block: &str, tone: Tone) -> String {
    let mut name = String::with_capacity(block.len() + 2 + tone.suffix().len());
    name.push_str(block);
    name.push_str("--");
    name.push_str(tone.suffix());
    name
}

/// Every tone, for exhaustive style registration.
const TONES: [Tone; 5] = [
    Tone::Accent,
    Tone::Neutral,
    Tone::Success,
    Tone::Warning,
    Tone::Danger,
];

/// Registers the `pk-timeline` class family: container, item, dot tones, content.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Container: a vertical stack of events.
    sheet.insert(
        Class::new("pk-timeline")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.md")),
    );

    // Item: a row pairing the dot with its content.
    sheet.insert(
        Class::new("pk-timeline__item")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Start))
            .with(StyleProp::Gap, tok("space.sm")),
    );

    // Dot: a small fixed circle that never shrinks.
    sheet.insert(
        Class::new("pk-timeline__dot")
            .with(StyleProp::Width, StyleValue::px(10.0))
            .with(StyleProp::Height, StyleValue::px(10.0))
            .with(StyleProp::MinWidth, StyleValue::px(10.0))
            .with(StyleProp::FlexShrink, StyleValue::number(0.0))
            .with(StyleProp::MarginTop, tok("space.xxs"))
            .with(StyleProp::BorderRadius, tok("radius.capsule")),
    );
    for tone in TONES {
        sheet.insert(
            Class::new(tone_class("pk-timeline__dot", tone))
                .with(StyleProp::BackgroundColor, tok(tone.color_token())),
        );
    }

    // Content: fills the rest of the row.
    sheet.insert(
        Class::new("pk-timeline__content")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: TimelineProps) -> Element {
        Timeline.render(&props)
    }

    #[test]
    fn empty_timeline_has_no_items() {
        let el = render(TimelineProps::new());
        assert_eq!(el.class_names(), ["pk-timeline"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn each_item_has_dot_then_content() {
        let el = render(
            TimelineProps::new()
                .item(TimelineItem::new(Element::text("Created")))
                .item(TimelineItem::new(Element::text("Shipped")).tone(Tone::Success)),
        );
        let items = el.child_elements();
        assert_eq!(items.len(), 2);
        for item in items {
            assert_eq!(item.class_names(), ["pk-timeline__item"]);
            assert_eq!(item.child_elements().len(), 2);
            assert!(item.child_elements()[0]
                .class_names()
                .iter()
                .any(|c| c.starts_with("pk-timeline__dot")));
            assert!(item.child_elements()[1]
                .class_names()
                .iter()
                .any(|c| c == "pk-timeline__content"));
        }
    }

    #[test]
    fn dot_carries_item_tone() {
        let el = render(TimelineProps::new().item(
            TimelineItem::new(Element::text("Failed")).tone(Tone::Danger),
        ));
        let dot = &el.child_elements()[0].child_elements()[0];
        assert_eq!(
            dot.class_names(),
            ["pk-timeline__dot", "pk-timeline__dot--danger"]
        );
    }
}
