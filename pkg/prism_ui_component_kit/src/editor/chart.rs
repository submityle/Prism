//! [`Chart`] — a generic chart frame with plot and legend slots.
//!
//! A chart renders a `pk-chart` frame carrying an optional title, a
//! `pk-chart__plot` region (where a backend or the `visualize` capability draws
//! the series) and a `pk-chart__legend` strip. A [`ChartKind`] selects a
//! `--line` / `--bar` / `--area` / `--scatter` modifier. Only kit class names
//! are attached; surface and type resolve from theme tokens via
//! [`crate::preset`].

use alloc::string::String;

use prism_ui::Element;
use prism_ui_a11y::Role;
use prism_ui_component::Component;

use crate::kit::classes;
use crate::preset::StyleSheet;

/// The kind of chart a [`Chart`] frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ChartKind {
    /// A line chart (default).
    #[default]
    Line,
    /// A bar chart.
    Bar,
    /// A filled area chart.
    Area,
    /// A scatter plot.
    Scatter,
}

impl ChartKind {
    /// The modifier suffix used in class names (e.g. `pk-chart--bar`).
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            ChartKind::Line => "line",
            ChartKind::Bar => "bar",
            ChartKind::Area => "area",
            ChartKind::Scatter => "scatter",
        }
    }
}

/// Props for [`Chart`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ChartProps {
    /// Optional chart title, rendered above the plot.
    pub title: Option<String>,
    /// The chart kind selecting the `--kind` modifier.
    pub kind: ChartKind,
    /// Optional content drawn inside the plot region.
    pub plot: Option<Element>,
    /// Optional content drawn inside the legend strip.
    pub legend: Option<Element>,
}

impl ChartProps {
    /// Creates default (untitled, line) chart props.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the chart title.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Sets the chart kind.
    #[must_use]
    pub fn kind(mut self, kind: ChartKind) -> Self {
        self.kind = kind;
        self
    }

    /// Sets the plot-region content.
    #[must_use]
    pub fn plot(mut self, plot: Element) -> Self {
        self.plot = Some(plot);
        self
    }

    /// Sets the legend content.
    #[must_use]
    pub fn legend(mut self, legend: Element) -> Self {
        self.legend = Some(legend);
        self
    }
}

/// The chart control. Zero-sized; all configuration lives in [`ChartProps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Chart;

impl Chart {
    /// The accessibility role a chart exposes (a read-only image).
    #[must_use]
    pub const fn role() -> Role {
        Role::Image
    }
}

impl Component for Chart {
    type Props = ChartProps;

    fn render(&self, props: &Self::Props) -> Element {
        let mut el = Element::box_();
        for name in classes("pk-chart", &[props.kind.suffix()]) {
            el = el.class(name);
        }

        if let Some(title) = props.title.clone() {
            el = el.child(Element::text(title).class("pk-chart__title"));
        }

        let mut plot = Element::box_().class("pk-chart__plot");
        if let Some(content) = props.plot.clone() {
            plot = plot.child(content);
        }
        el = el.child(plot);

        let mut legend = Element::box_().class("pk-chart__legend");
        if let Some(content) = props.legend.clone() {
            legend = legend.child(content);
        }
        el.child(legend)
    }
}

/// Registers the `pk-chart` family: frame, kind modifiers, title, plot, legend.
pub(crate) fn register_styles(sheet: &mut StyleSheet) {
    use prism_ui_style::{Class, Keyword, StyleProp, StyleValue};

    use crate::preset::{kw, tok};

    // Frame: a padded surface column.
    sheet.insert(
        Class::new("pk-chart")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Column))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::BackgroundColor, tok("color.surface"))
            .with(StyleProp::BorderRadius, tok("radius.lg"))
            .with_padding_x(tok("space.md"))
            .with_padding_y(tok("space.md")),
    );

    // Kind modifiers tint the plot's accent via a border color hint.
    for (kind, color) in [
        (ChartKind::Line, "color.blue"),
        (ChartKind::Bar, "color.indigo"),
        (ChartKind::Area, "color.teal"),
        (ChartKind::Scatter, "color.purple"),
    ] {
        let mut name = String::from("pk-chart--");
        name.push_str(kind.suffix());
        sheet.insert(Class::new(name).with(StyleProp::BorderColor, tok(color)));
    }

    sheet.insert(
        Class::new("pk-chart__title")
            .with(StyleProp::Color, tok("color.label"))
            .with(StyleProp::FontSize, tok("font.size.headline"))
            .with(StyleProp::FontWeight, tok("font.weight.semibold")),
    );

    // Plot: the growable drawing region.
    sheet.insert(
        Class::new("pk-chart__plot")
            .with(StyleProp::FlexGrow, StyleValue::number(1.0))
            .with(StyleProp::MinHeight, StyleValue::px(120.0))
            .with(StyleProp::BackgroundColor, tok("color.fill.secondary"))
            .with(StyleProp::BorderRadius, tok("radius.md")),
    );

    // Legend: a wrapping row of series keys.
    sheet.insert(
        Class::new("pk-chart__legend")
            .with(StyleProp::Display, kw(Keyword::Flex))
            .with(StyleProp::FlexDirection, kw(Keyword::Row))
            .with(StyleProp::Gap, tok("space.sm"))
            .with(StyleProp::Color, tok("color.label.secondary"))
            .with(StyleProp::FontSize, tok("font.size.footnote")),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(props: ChartProps) -> Element {
        Chart.render(&props)
    }

    #[test]
    fn attaches_base_and_kind_classes_in_order() {
        let el = render(ChartProps::new().kind(ChartKind::Bar));
        assert_eq!(el.class_names(), ["pk-chart", "pk-chart--bar"]);
    }

    #[test]
    fn default_kind_is_line() {
        assert_eq!(ChartKind::default(), ChartKind::Line);
    }

    #[test]
    fn renders_plot_and_legend_without_title_by_default() {
        let el = render(ChartProps::new());
        let children = el.child_elements();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].class_names(), ["pk-chart__plot"]);
        assert_eq!(children[1].class_names(), ["pk-chart__legend"]);
    }

    #[test]
    fn title_renders_before_plot_when_set() {
        let el = render(ChartProps::new().title("Frame time"));
        let children = el.child_elements();
        assert_eq!(children.len(), 3);
        assert_eq!(children[0].class_names(), ["pk-chart__title"]);
        assert_eq!(children[0].text_content(), Some("Frame time"));
    }

    #[test]
    fn role_is_image() {
        assert_eq!(Chart::role(), Role::Image);
    }

    #[test]
    fn register_adds_classes() {
        let mut sheet = StyleSheet::new();
        register_styles(&mut sheet);
        for name in [
            "pk-chart",
            "pk-chart--line",
            "pk-chart--bar",
            "pk-chart--area",
            "pk-chart--scatter",
            "pk-chart__title",
            "pk-chart__plot",
            "pk-chart__legend",
        ] {
            assert!(sheet.get(name).is_some(), "missing {name}");
        }
    }
}
