//! [`Result`] — a full-page status view (success, empty, error, 404).
//!
//! A result is a centered page state: an optional `icon`, a `title`, a
//! `description`, a semantic [`Tone`] and a row of `actions`. Unlike the
//! overlay-backed feedback controls it is **not** a popover — it fills a region
//! in place. It attaches only kit classes; [`crate::preset`] resolves the tone
//! accent and typography against the active theme.
//!
//! Note: this type is named `Result` and is re-exported from the crate as
//! `ResultView` to avoid colliding with the [`core::result::Result`] in the
//! prelude. The [`StatusPage`] alias names the same control.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::{classes, Tone};
use crate::preset::StyleSheet;

/// Props for [`Result`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ResultProps {
    /// The headline status title.
    pub title: String,
    /// Supporting descriptive text.
    pub description: String,
    /// The semantic tone selecting the icon/title accent.
    pub tone: Tone,
    /// Optional status glyph rendered above the title.
    pub icon: Option<Element>,
    /// Trailing call-to-action controls.
    pub actions: Vec<Element>,
}

impl ResultProps {
    /// Creates result props with `title` and the default (accent) tone.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Self::default()
        }
    }

    /// Sets the supporting description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Sets the semantic tone.
    #[must_use]
    pub fn tone(mut self, tone: Tone) -> Self {
        self.tone = tone;
        self
    }

    /// Sets the status glyph.
    #[must_use]
    pub fn icon(mut self, icon: Element) -> Self {
        self.icon = Some(icon);
        self
    }

    /// Appends a call-to-action control.
    #[must_use]
    pub fn action(mut self, action: Element) -> Self {
        self.actions.push(action);
        self
    }

    /// Replaces the action controls with `actions`.
    #[must_use]
    pub fn actions<I: IntoIterator<Item = Element>>(mut self, actions: I) -> Self {
        self.actions = actions.into_iter().collect();
        self
    }
}

/// The result control. Zero-sized; configuration lives in [`ResultProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Result;

/// Alias for [`Result`] under its common "status page" name.
pub type StatusPage = Result;

impl Component for Result {
    type Props = ResultProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-result", &[props.tone.suffix()]) {
            el = el.class(name);
        }

        if let Some(icon) = props.icon.clone() {
            el = el.child(icon.class("pk-result__icon"));
        }
        el = el.child(Element::text(props.title.clone()).class("pk-result__title"));
        el = el.child(
            Element::text(props.description.clone()).class("pk-result__description"),
        );
        if !props.actions.is_empty() {
            el = el.child(
                Element::box_()
                    .class("pk-result__actions")
                    .children(props.actions.iter().cloned()),
            );
        }
        el
    }
}

/// Builds the `--tone` modifier name for a block, e.g. `pk-result--success`.
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

/// Registers the `pk-result` class family: the centered page, per-tone icon
/// accent, the icon/title/description slots and the actions row.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    // Page: a centered vertical stack filling its region.
    sheet.insert(
        Class::new("pk-result")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.md"))
            .with_padding_x(tok("space.xl"))
            .with_padding_y(tok("space.xxl")),
    );

    // Tones paint the icon/title accent.
    for tone in TONES {
        sheet.insert(
            Class::new(tone_class("pk-result", tone))
                .with(StyleProp::Color, tok(tone.color_token())),
        );
    }

    // Icon: a large glyph inheriting the tone accent.
    sheet.insert(
        Class::new("pk-result__icon").with(StyleProp::FontSize, tok("font.size.large-title")),
    );

    // Title: the headline status line.
    sheet.insert(
        Class::new("pk-result__title")
            .with(StyleProp::FontSize, tok("font.size.title2"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold"))
            .with(StyleProp::Color, tok("color.label")),
    );

    // Description: quieter supporting prose.
    sheet.insert(
        Class::new("pk-result__description")
            .with(StyleProp::FontSize, tok("font.size.body"))
            .with(StyleProp::Color, tok("color.label.secondary")),
    );

    // Actions: a centered row of calls-to-action.
    sheet.insert(
        Class::new("pk-result__actions")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::JustifyContent, kw(Keyword::Center))
            .with(StyleProp::Gap, tok("space.sm")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: ResultProps) -> Element {
        Result.render(&props)
    }

    #[test]
    fn attaches_base_and_tone_classes() {
        let el = render(ResultProps::new("All done").tone(Tone::Success));
        assert_eq!(el.class_names(), ["pk-result", "pk-result--success"]);
    }

    #[test]
    fn title_and_description_render() {
        let el = render(ResultProps::new("Not found").description("No such page."));
        let kids = el.child_elements();
        assert!(kids[0].class_names().iter().any(|c| c == "pk-result__title"));
        assert_eq!(kids[0].text_content(), Some("Not found"));
        assert!(kids[1].class_names().iter().any(|c| c == "pk-result__description"));
    }

    #[test]
    fn icon_precedes_title_and_actions_trail() {
        let el = render(
            ResultProps::new("Error")
                .icon(Element::box_().class("glyph"))
                .action(Element::box_().class("retry")),
        );
        let kids = el.child_elements();
        assert!(kids[0].class_names().iter().any(|c| c == "pk-result__icon"));
        assert!(kids.last().unwrap().class_names().iter().any(|c| c == "pk-result__actions"));
    }

    #[test]
    fn actions_row_omitted_when_empty() {
        let el = render(ResultProps::new("Empty"));
        assert!(!el
            .child_elements()
            .iter()
            .any(|c| c.class_names().iter().any(|n| n == "pk-result__actions")));
    }

    #[test]
    fn status_page_alias_is_the_same_control() {
        let el = StatusPage::default().render(&ResultProps::new("Welcome"));
        assert!(el.class_names().iter().any(|c| c == "pk-result"));
    }
}
