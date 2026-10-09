//! [`Affix`] — a sticky container that pins its child while scrolling.
//!
//! An affix wraps an optional `child` in a `pk-affix` box. When `pinned` it
//! gains the block-level `pk-affix--pinned` modifier (an elevated chrome
//! surface). The `offset` is approximated with an inline top margin in logical
//! pixels, since the style layer exposes no `position`/`top` property.
//!
//! True scroll pinning — toggling `pinned` and resolving the sticky offset
//! against the scroll position — is the responsibility of the layer that drives
//! this control; the affix only records the intended offset and pinned state.

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::preset::StyleSheet;

/// Props for [`Affix`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct AffixProps {
    /// The content the affix pins.
    pub child: Option<Element>,
    /// Whether the affix is currently pinned.
    pub pinned: bool,
    /// The sticky offset, in logical pixels, applied as a top margin.
    pub offset: f32,
}

impl AffixProps {
    /// Creates empty, unpinned affix props with a zero offset.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the pinned child.
    #[must_use]
    pub fn child(mut self, element: Element) -> Self {
        self.child = Some(element);
        self
    }

    /// Sets the pinned state.
    #[must_use]
    pub fn pinned(mut self, pinned: bool) -> Self {
        self.pinned = pinned;
        self
    }

    /// Sets the sticky offset in logical pixels.
    #[must_use]
    pub fn offset(mut self, offset: f32) -> Self {
        self.offset = offset;
        self
    }
}

/// The affix control. Zero-sized; config lives in [`AffixProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Affix;

impl Component for Affix {
    type Props = AffixProps;

    fn render(&self, props: &Self::Props) -> Element {
        use prism_ui_style::{Length, StyleProp, StyleValue};

        let mods: &[&str] = if props.pinned { &["pinned"] } else { &[] };
        let mut el = Element::box_();
        for name in classes("pk-affix", mods) {
            el = el.class(name);
        }

        // Approximate the sticky offset with an inline top margin; real
        // pinning is driven by the upstream scroll layer.
        el = el.style(
            StyleProp::MarginTop,
            StyleValue::Length(Length::Px(props.offset)),
        );

        if let Some(child) = props.child.clone() {
            el = el.child(child);
        }
        el
    }
}

/// Registers the `pk-affix` class family: the wrapper and its pinned
/// (elevated) modifier.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    // Wrapper: a plain block that only reserves the child's box.
    sheet.insert(Class::new("pk-affix").with(StyleProp::Display, kw(Keyword::Block)));

    // Pinned: lift the content onto an opaque surface with a soft shadow.
    sheet.insert(
        Class::new("pk-affix--pinned")
            .with(StyleProp::BackgroundColor, tok("color.surface"))
            .with_shadow(0.0, 2.0, 12.0, tok("glass.shadow")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::{Length, StyleProp, StyleValue};

    fn render(props: AffixProps) -> Element {
        Affix.render(&props)
    }

    fn inline(el: &Element, prop: StyleProp) -> Option<StyleValue> {
        el.inline_pairs()
            .iter()
            .find(|(p, _)| *p == prop)
            .map(|(_, v)| v.clone())
    }

    #[test]
    fn unpinned_has_only_base_class() {
        let el = render(AffixProps::new());
        assert_eq!(el.class_names(), ["pk-affix"]);
        assert!(el.child_elements().is_empty());
    }

    #[test]
    fn pinned_adds_modifier_class() {
        let el = render(AffixProps::new().pinned(true));
        assert_eq!(el.class_names(), ["pk-affix", "pk-affix--pinned"]);
    }

    #[test]
    fn offset_applies_as_inline_top_margin() {
        let el = render(AffixProps::new().offset(44.0));
        assert_eq!(
            inline(&el, StyleProp::MarginTop),
            Some(StyleValue::Length(Length::Px(44.0)))
        );
    }

    #[test]
    fn wraps_child() {
        let el = render(AffixProps::new().child(Element::box_().class("inner")));
        let kids = el.child_elements();
        assert_eq!(kids.len(), 1);
        assert!(kids[0].class_names().iter().any(|c| c == "inner"));
    }
}
