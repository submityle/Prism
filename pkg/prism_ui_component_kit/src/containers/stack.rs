//! [`Stack`] — a one-dimensional flex container.
//!
//! A stack lays its children out in a single row or column with a token-backed
//! gap and a cross-axis alignment. It attaches the `pk-stack` class family
//! (`--row`/`--col`, `--gap-*`, `--align-*`) and nothing else; spacing and
//! alignment resolve from theme tokens via [`crate::preset`].

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::Component;

use crate::kit::ControlSize;
use crate::preset::StyleSheet;

/// The main axis along which a [`Stack`] lays out its children.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum StackDirection {
    /// Children flow top-to-bottom (a column).
    #[default]
    Vertical,
    /// Children flow left-to-right (a row).
    Horizontal,
}

impl StackDirection {
    /// The modifier class for this direction.
    #[must_use]
    const fn class(self) -> &'static str {
        match self {
            StackDirection::Vertical => "pk-stack--col",
            StackDirection::Horizontal => "pk-stack--row",
        }
    }
}

/// The cross-axis alignment of a [`Stack`]'s children.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum StackAlign {
    /// Pack children at the cross-axis start.
    Start,
    /// Center children on the cross axis.
    Center,
    /// Pack children at the cross-axis end.
    End,
    /// Stretch children to fill the cross axis.
    #[default]
    Stretch,
    /// Align children on their text baseline.
    Baseline,
}

impl StackAlign {
    /// The modifier class for this alignment.
    #[must_use]
    const fn class(self) -> &'static str {
        match self {
            StackAlign::Start => "pk-stack--align-start",
            StackAlign::Center => "pk-stack--align-center",
            StackAlign::End => "pk-stack--align-end",
            StackAlign::Stretch => "pk-stack--align-stretch",
            StackAlign::Baseline => "pk-stack--align-baseline",
        }
    }
}

/// The modifier class for a gap density step.
#[must_use]
const fn gap_class(size: ControlSize) -> &'static str {
    match size {
        ControlSize::Small => "pk-stack--gap-sm",
        ControlSize::Medium => "pk-stack--gap-md",
        ControlSize::Large => "pk-stack--gap-lg",
    }
}

/// Props for [`Stack`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct StackProps {
    /// The main-axis direction.
    pub direction: StackDirection,
    /// The gap density between children.
    pub gap: ControlSize,
    /// The cross-axis alignment.
    pub align: StackAlign,
    /// The children laid out along the main axis.
    pub children: Vec<Element>,
}

impl StackProps {
    /// Creates empty stack props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the main-axis direction.
    #[must_use]
    pub fn direction(mut self, direction: StackDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Sets the gap density between children.
    #[must_use]
    pub fn gap(mut self, gap: ControlSize) -> Self {
        self.gap = gap;
        self
    }

    /// Sets the cross-axis alignment.
    #[must_use]
    pub fn align(mut self, align: StackAlign) -> Self {
        self.align = align;
        self
    }

    /// Appends a child.
    #[must_use]
    pub fn child(mut self, element: Element) -> Self {
        self.children.push(element);
        self
    }

    /// Replaces the children with `children`.
    #[must_use]
    pub fn children<I: IntoIterator<Item = Element>>(mut self, children: I) -> Self {
        self.children = children.into_iter().collect();
        self
    }
}

/// The stack control. Zero-sized; all configuration lives in [`StackProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Stack;

impl Component for Stack {
    type Props = StackProps;

    fn render(&self, props: &Self::Props) -> Element {
        Element::box_()
            .class("pk-stack")
            .class(props.direction.class())
            .class(gap_class(props.gap))
            .class(props.align.class())
            .children(props.children.iter().cloned())
    }
}

/// Registers the `pk-stack` class family: base, directions, gaps, alignments.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp};

    use crate::preset::{kw, tok};

    // Base: a flex container; direction/gap/align come from modifiers.
    sheet.insert(Class::new("pk-stack").with(StyleProp::Display, kw(Keyword::Flex)));

    sheet.insert(
        Class::new("pk-stack--row").with(StyleProp::FlexDirection, kw(Keyword::Row)),
    );
    sheet.insert(
        Class::new("pk-stack--col").with(StyleProp::FlexDirection, kw(Keyword::Column)),
    );

    sheet.insert(Class::new("pk-stack--gap-sm").with(StyleProp::Gap, tok("space.xs")));
    sheet.insert(Class::new("pk-stack--gap-md").with(StyleProp::Gap, tok("space.sm")));
    sheet.insert(Class::new("pk-stack--gap-lg").with(StyleProp::Gap, tok("space.lg")));

    sheet.insert(
        Class::new("pk-stack--align-start").with(StyleProp::AlignItems, kw(Keyword::Start)),
    );
    sheet.insert(
        Class::new("pk-stack--align-center").with(StyleProp::AlignItems, kw(Keyword::Center)),
    );
    sheet.insert(
        Class::new("pk-stack--align-end").with(StyleProp::AlignItems, kw(Keyword::End)),
    );
    sheet.insert(
        Class::new("pk-stack--align-stretch").with(StyleProp::AlignItems, kw(Keyword::Stretch)),
    );
    sheet.insert(
        Class::new("pk-stack--align-baseline")
            .with(StyleProp::AlignItems, kw(Keyword::Baseline)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: StackProps) -> Element {
        Stack.render(&props)
    }

    #[test]
    fn attaches_base_direction_gap_and_align_in_order() {
        let el = render(
            StackProps::new()
                .direction(StackDirection::Horizontal)
                .gap(ControlSize::Large)
                .align(StackAlign::Center),
        );
        assert_eq!(
            el.class_names(),
            [
                "pk-stack",
                "pk-stack--row",
                "pk-stack--gap-lg",
                "pk-stack--align-center"
            ]
        );
    }

    #[test]
    fn defaults_to_column_medium_stretch() {
        let el = render(StackProps::new());
        assert_eq!(
            el.class_names(),
            [
                "pk-stack",
                "pk-stack--col",
                "pk-stack--gap-md",
                "pk-stack--align-stretch"
            ]
        );
    }

    #[test]
    fn children_are_direct_children() {
        let el = render(StackProps::new().children([Element::box_(), Element::box_(), Element::box_()]));
        assert_eq!(el.child_elements().len(), 3);
    }
}
