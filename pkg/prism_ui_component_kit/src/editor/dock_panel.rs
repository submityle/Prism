//! [`DockPanel`] — a dockable panel shell with a title bar and body.
//!
//! A dock panel renders a `pk-dock-panel` glass surface with a
//! `pk-dock-panel__bar` title strip above a `pk-dock-panel__body` content
//! region. A [`DockSide`] selects a `--left` / `--right` / `--top` / `--bottom`
//! / `--floating` modifier so a docking host can place the panel. Only kit
//! class names are attached; surface and spacing resolve from theme tokens via
//! [`crate::preset`].

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::preset::StyleSheet;

/// The edge a [`DockPanel`] is docked to (or floating).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum DockSide {
    /// Docked to the leading (left) edge.
    #[default]
    Left,
    /// Docked to the trailing (right) edge.
    Right,
    /// Docked to the top edge.
    Top,
    /// Docked to the bottom edge.
    Bottom,
    /// Floating (not docked to any edge).
    Floating,
}

impl DockSide {
    /// The modifier suffix used in class names (e.g. `pk-dock-panel--left`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            DockSide::Left => "left",
            DockSide::Right => "right",
            DockSide::Top => "top",
            DockSide::Bottom => "bottom",
            DockSide::Floating => "floating",
        }
    }
}

/// Props for [`DockPanel`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DockPanelProps {
    /// The title shown in the panel's bar.
    pub title: String,
    /// The edge the panel is docked to (or floating).
    pub side: DockSide,
    /// The panel body content.
    pub children: Vec<Element>,
}

impl DockPanelProps {
    /// Creates props for a titled, left-docked panel with no body.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Self::default()
        }
    }

    /// Sets the docking side.
    #[must_use]
    pub fn side(mut self, side: DockSide) -> Self {
        self.side = side;
        self
    }

    /// Appends a single body child.
    #[must_use]
    pub fn child(mut self, child: Element) -> Self {
        self.children.push(child);
        self
    }

    /// Replaces the body children.
    #[must_use]
    pub fn children<I: IntoIterator<Item = Element>>(mut self, children: I) -> Self {
        self.children = children.into_iter().collect();
        self
    }
}

/// The dock-panel control. Zero-sized; config lives in [`DockPanelProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct DockPanel;

impl DockPanel {
    /// The accessibility role a dock panel exposes.
    #[must_use]
    pub const fn role() -> Role {
        Role::Group
    }
}

impl Component for DockPanel {
    type Props = DockPanelProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-dock-panel", &[props.side.suffix()]) {
            el = el.class(name);
        }

        let bar = Element::box_()
            .class("pk-dock-panel__bar")
            .child(Element::text(props.title.clone()).class("pk-dock-panel__title"));

        let body = Element::box_()
            .class("pk-dock-panel__body")
            .children(props.children.iter().cloned());

        el.child(bar).child(body)
    }
}

/// Every side, for exhaustive style registration.
const SIDES: [DockSide; 5] = [
    DockSide::Left,
    DockSide::Right,
    DockSide::Top,
    DockSide::Bottom,
    DockSide::Floating,
];

/// Registers the `pk-dock-panel` family: shell, side modifiers, bar, body.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Shell: a frosted glass column with a rounded edge.
    sheet.insert(
        Class::new("pk-dock-panel")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::BorderRadius, tok("radius.lg"))
            .with(StyleProp::BorderWidth, StyleValue::px(1.0))
            .with(StyleProp::BorderColor, tok("color.separator"))
            .with_glass(0.0, tok("glass.tint"), Some(tok("glass.highlight"))),
    );

    // Side modifiers tweak the docked edge's affordance; floating gains a lift.
    for side in SIDES {
        let mut name = String::from("pk-dock-panel--");
        name.push_str(side.suffix());
        let class = match side {
            DockSide::Floating => Class::new(name).with_shadow(0.0, 12.0, 32.0, tok("glass.shadow")),
            _ => Class::new(name).with(StyleProp::MinWidth, StyleValue::px(180.0)),
        };
        sheet.insert(class);
    }

    // Bar: a compact header strip with headline type.
    sheet.insert(
        Class::new("pk-dock-panel__bar")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::AlignItems, kw(Keyword::Center))
            .with(StyleProp::JustifyContent, kw(Keyword::SpaceBetween))
            .with(StyleProp::BackgroundColor, tok("color.surface.secondary"))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.xs")),
    );

    sheet.insert(
        Class::new("pk-dock-panel__title")
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.headline"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );

    // Body: fills the remaining space, padded.
    sheet.insert(
        Class::new("pk-dock-panel__body")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm"))
            .with_padding_x(tok("space.sm"))
            .with_padding_y(tok("space.sm")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: DockPanelProps) -> Element {
        DockPanel.render(&props)
    }

    #[test]
    fn attaches_base_and_side_classes_in_order() {
        let el = render(DockPanelProps::new("Scene").side(DockSide::Right));
        assert_eq!(el.class_names(), ["pk-dock-panel", "pk-dock-panel--right"]);
    }

    #[test]
    fn renders_bar_then_body() {
        let el = render(DockPanelProps::new("Scene").child(Element::box_().class("content")));
        let children = el.child_elements();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].class_names(), ["pk-dock-panel__bar"]);
        assert_eq!(children[1].class_names(), ["pk-dock-panel__body"]);
        assert_eq!(children[1].child_elements().len(), 1);
    }

    #[test]
    fn bar_carries_title_text() {
        let el = render(DockPanelProps::new("Inspector"));
        let title = &el.child_elements()[0].child_elements()[0];
        assert_eq!(title.class_names(), ["pk-dock-panel__title"]);
        assert_eq!(title.text_content(), Some("Inspector"));
    }

    #[test]
    fn role_is_group() {
        assert_eq!(DockPanel::role(), Role::Group);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in [
            "pk-dock-panel",
            "pk-dock-panel--left",
            "pk-dock-panel--right",
            "pk-dock-panel--top",
            "pk-dock-panel--bottom",
            "pk-dock-panel--floating",
            "pk-dock-panel__bar",
            "pk-dock-panel__title",
            "pk-dock-panel__body",
        ] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
