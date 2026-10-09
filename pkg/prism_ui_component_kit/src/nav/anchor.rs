//! [`Anchor`] — an in-page section navigator (a.k.a. [`ScrollSpy`]).
//!
//! An anchor renders a `pk-anchor` list of `pk-anchor__link` entries, each a
//! label pointing at an on-page `href`. The entry matching the currently
//! visible section gains the block-level `pk-anchor__link--active` modifier
//! (not a shared `is-*` state class). The container exposes the
//! [`Role::List`](prism_ui_a11y::Role::List) accessibility role and each entry
//! the [`Role::Link`](prism_ui_a11y::Role::Link) role. Only kit class names are
//! attached; which entry is active is decided by the scroll layer upstream.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::preset::StyleSheet;

/// A single anchor entry: a label pointing at an on-page target.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct AnchorLink {
    /// The visible entry label.
    pub label: String,
    /// The on-page target (e.g. `#section-id`).
    pub href: String,
    /// Whether this entry targets the currently visible section.
    pub active: bool,
}

impl AnchorLink {
    /// Creates an inactive entry from a label and target.
    #[must_use]
    pub fn new(label: impl Into<String>, href: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            href: href.into(),
            active: false,
        }
    }

    /// Marks the entry active.
    #[must_use]
    pub fn active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }
}

/// Props for [`Anchor`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct AnchorProps {
    /// The entries, rendered top-to-bottom.
    pub links: Vec<AnchorLink>,
}

impl AnchorProps {
    /// Creates empty anchor props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an entry.
    #[must_use]
    pub fn link(mut self, link: AnchorLink) -> Self {
        self.links.push(link);
        self
    }

    /// Replaces the entries with `links`.
    #[must_use]
    pub fn links<I: IntoIterator<Item = AnchorLink>>(mut self, links: I) -> Self {
        self.links = links.into_iter().collect();
        self
    }
}

/// The anchor control. Zero-sized; config lives in [`AnchorProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Anchor;

/// A scroll-spy navigator — the same control as [`Anchor`], named for the
/// scroll-synchronized use case.
pub type ScrollSpy = Anchor;

impl Anchor {
    /// The accessibility role the anchor container approximates.
    #[must_use]
    pub const fn role() -> Role {
        Role::List
    }

    /// The accessibility role each entry exposes.
    #[must_use]
    pub const fn link_role() -> Role {
        Role::Link
    }
}

impl Component for Anchor {
    type Props = AnchorProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_().class("pk-anchor");
        for link in &props.links {
            let mods: &[&str] = if link.active { &["active"] } else { &[] };
            let mut entry = Element::text(link.label.clone());
            for name in classes("pk-anchor__link", mods) {
                entry = entry.class(name);
            }
            el = el.child(entry);
        }
        el
    }
}

/// Registers the `pk-anchor` class family: the list column, the entry link and
/// its active modifier.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, InteractionState, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // List: a tight vertical column of entries.
    sheet.insert(
        Class::new("pk-anchor")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.xxs"))
            .with_padding_y(tok("space.xs")),
    );

    // Link: a muted footnote entry with a rounded hover target.
    sheet.insert(
        Class::new("pk-anchor__link")
            .with_padding_x(tok("space.xs"))
            .with_padding_y(tok("space.xxs"))
            .with(StyleProp::BorderRadius, tok("radius.sm"))
            .with(StyleProp::FontSize, tok("font.size.footnote"))
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with_state(InteractionState::Hover, StyleProp::Opacity, StyleValue::number(0.8)),
    );

    // Active: accent the entry for the visible section.
    sheet.insert(
        Class::new("pk-anchor__link--active")
            .with(StyleProp::Color, tok("color.tint"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: AnchorProps) -> Element {
        Anchor.render(&props)
    }

    #[test]
    fn empty_anchor_has_only_base_class() {
        let el = render(AnchorProps::new());
        assert_eq!(el.class_names(), ["pk-anchor"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn active_link_gets_block_level_modifier() {
        let el = render(
            AnchorProps::new()
                .link(AnchorLink::new("Intro", "#intro"))
                .link(AnchorLink::new("Usage", "#usage").active(true)),
        );
        let links = el.child_elements();
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].class_names(), ["pk-anchor__link"]);
        assert_eq!(
            links[1].class_names(),
            ["pk-anchor__link", "pk-anchor__link--active"]
        );
        assert_eq!(links[1].text_content(), Some("Usage"));
    }

    #[test]
    fn roles_are_list_and_link() {
        assert_eq!(Anchor::role(), Role::List);
        assert_eq!(Anchor::link_role(), Role::Link);
    }

    #[test]
    fn scroll_spy_is_anchor_alias() {
        let el = ScrollSpy::default().render(&AnchorProps::new());
        assert_eq!(el.class_names(), ["pk-anchor"]);
    }
}
