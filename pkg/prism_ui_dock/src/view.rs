//! Element assembly for the pure-data dock model.
//!
//! The core crate remains renderer-independent. This feature-gated adapter
//! resolves registered factories into `prism_ui::Element` trees and uses the
//! existing `DockPanel` shell for panel surfaces; GPU painting remains the
//! responsibility of the host UI renderer.

use alloc::{boxed::Box, string::String, vec::Vec};

use prism_ui::style::{Keyword, StyleProp, StyleValue};
use prism_ui::Element;
use prism_ui_component::Component;
use prism_ui_component_kit::editor::{DockPanel, DockPanelProps, DockSide};

use crate::{Axis, DockLayout, DockNode, DocumentId, LayoutError, PanelId, PanelRegistry, Side};

/// A lazily evaluated panel or document content factory.
pub type ViewFactory = Box<dyn Fn() -> Element + Send + Sync + 'static>;

/// Structured failure while converting dock data into an Element tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DockViewError {
    /// The layout failed validation before any factory was invoked.
    InvalidLayout(LayoutError),
    /// A panel in the layout is not present in the core panel registry.
    UnknownPanel(PanelId),
    /// A registered panel has no Element factory.
    MissingPanelFactory(PanelId),
    /// A document in the layout has no Element factory.
    MissingDocumentFactory(DocumentId),
}

/// Renderer-facing factories for registered panel IDs.
pub struct PanelViewRegistry {
    entries: Vec<(PanelId, ViewFactory)>,
}

impl Default for PanelViewRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PanelViewRegistry {
    /// Creates an empty panel factory registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Registers a factory, rejecting a duplicate panel ID.
    pub fn register(
        &mut self,
        id: impl Into<PanelId>,
        factory: impl Fn() -> Element + Send + Sync + 'static,
    ) -> Result<(), PanelId> {
        let id = id.into();
        if self.entries.iter().any(|(registered, _)| registered == &id) {
            return Err(id);
        }
        self.entries.push((id, Box::new(factory)));
        Ok(())
    }

    fn get(&self, id: &str) -> Option<&ViewFactory> {
        self.entries
            .iter()
            .find(|(registered, _)| registered == id)
            .map(|(_, factory)| factory)
    }
}

/// Renderer-facing factories for document IDs.
pub struct DocumentViewRegistry {
    entries: Vec<(DocumentId, ViewFactory)>,
}

impl Default for DocumentViewRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentViewRegistry {
    /// Creates an empty document factory registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Registers a factory, rejecting a duplicate document ID.
    pub fn register(
        &mut self,
        id: impl Into<DocumentId>,
        factory: impl Fn() -> Element + Send + Sync + 'static,
    ) -> Result<(), DocumentId> {
        let id = id.into();
        if self.entries.iter().any(|(registered, _)| registered == &id) {
            return Err(id);
        }
        self.entries.push((id, Box::new(factory)));
        Ok(())
    }

    fn get(&self, id: &str) -> Option<&ViewFactory> {
        self.entries
            .iter()
            .find(|(registered, _)| registered == id)
            .map(|(_, factory)| factory)
    }
}

/// Converts a validated [`DockLayout`] into an Element tree.
pub struct DockView<'a> {
    layout: &'a DockLayout,
    panels: &'a PanelRegistry,
    panel_views: &'a PanelViewRegistry,
    document_views: &'a DocumentViewRegistry,
}

impl<'a> DockView<'a> {
    /// Creates an Element assembler for a layout and its registered views.
    #[must_use]
    pub const fn new(
        layout: &'a DockLayout,
        panels: &'a PanelRegistry,
        panel_views: &'a PanelViewRegistry,
        document_views: &'a DocumentViewRegistry,
    ) -> Self {
        Self {
            layout,
            panels,
            panel_views,
            document_views,
        }
    }

    /// Builds the complete main, floating, and auto-hide surfaces.
    pub fn render(&self) -> Result<Element, DockViewError> {
        self.layout
            .validate()
            .map_err(DockViewError::InvalidLayout)?;

        let mut root = Element::box_().class("pk-dock-root");
        root = root.child(self.render_node(&self.layout.root, DockSide::Left)?);
        root = root.child(self.render_floating()?);
        root = root.child(self.render_autohide()?);
        Ok(root)
    }

    fn render_node(&self, node: &DockNode, side: DockSide) -> Result<Element, DockViewError> {
        match node {
            DockNode::Empty => Ok(Element::box_().class("pk-dock-empty")),
            DockNode::TabStack { tabs, active } => {
                let active_id = &tabs[*active];
                let descriptor = self
                    .panels
                    .resolve(active_id)
                    .ok_or_else(|| DockViewError::UnknownPanel(active_id.clone()))?;
                let factory = self
                    .panel_views
                    .get(active_id)
                    .ok_or_else(|| DockViewError::MissingPanelFactory(active_id.clone()))?;
                let tabs = tabs.iter().enumerate().map(|(index, id)| {
                    let title = self
                        .panels
                        .resolve(id)
                        .map(|panel| panel.title.clone())
                        .unwrap_or_else(|| id.clone());
                    let class = if index == *active {
                        "pk-dock-tab pk-dock-tab--active"
                    } else {
                        "pk-dock-tab"
                    };
                    Element::text(title).class(class).key_str(id.clone())
                });
                let tab_bar = Element::box_().class("pk-dock-tab-bar").children(tabs);
                let content = factory().key_str(format_key("panel", active_id));
                Ok(DockPanel.render(
                    &DockPanelProps::new(descriptor.title.clone())
                        .side(side)
                        .children([tab_bar, content]),
                ))
            }
            DockNode::Document { tabs, active } => {
                let id = &tabs[*active];
                let factory = self
                    .document_views
                    .get(id)
                    .ok_or_else(|| DockViewError::MissingDocumentFactory(id.clone()))?;
                Ok(Element::box_()
                    .class("pk-dock-document")
                    .key_str(format_key("document", id))
                    .child(factory()))
            }
            DockNode::Split { axis, children } => {
                let axis_class = match axis {
                    Axis::Horizontal => "pk-dock-split--horizontal",
                    Axis::Vertical => "pk-dock-split--vertical",
                };
                let child_count = children.len();
                let children = children
                    .iter()
                    .enumerate()
                    .map(|(index, child)| {
                        let child_side = split_side(*axis, index, child_count);
                        self.render_node(&child.node, child_side).map(|node| {
                            node.style(
                                StyleProp::FlexGrow,
                                StyleValue::number(child.ratio.millionths() as f32),
                            )
                            .style(StyleProp::FlexShrink, StyleValue::number(1.0))
                            .style(StyleProp::FlexBasis, StyleValue::px(0.0))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Element::box_()
                    .class("pk-dock-split")
                    .class(axis_class)
                    .style(StyleProp::Display, StyleValue::keyword(Keyword::Flex))
                    .style(
                        StyleProp::FlexDirection,
                        StyleValue::keyword(match axis {
                            Axis::Horizontal => Keyword::Row,
                            Axis::Vertical => Keyword::Column,
                        }),
                    )
                    .children(children))
            }
        }
    }

    fn render_floating(&self) -> Result<Element, DockViewError> {
        let windows = self
            .layout
            .floating
            .iter()
            .map(|window| {
                self.render_node(&window.root, DockSide::Floating)
                    .map(|node| {
                        Element::box_()
                            .class("pk-dock-floating-window")
                            .key_int(i64::from(window.z))
                            .style(StyleProp::Position, StyleValue::keyword(Keyword::Absolute))
                            .style(StyleProp::Left, StyleValue::px(window.rect.x as f32))
                            .style(StyleProp::Top, StyleValue::px(window.rect.y as f32))
                            .style(StyleProp::Width, StyleValue::px(window.rect.width as f32))
                            .style(StyleProp::Height, StyleValue::px(window.rect.height as f32))
                            .child(node)
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Element::box_()
            .class("pk-dock-floating-layer")
            .children(windows))
    }

    fn render_autohide(&self) -> Result<Element, DockViewError> {
        let entries = self
            .layout
            .autohide
            .entries
            .iter()
            .map(|entry| {
                let descriptor = self
                    .panels
                    .resolve(&entry.panel)
                    .ok_or_else(|| DockViewError::UnknownPanel(entry.panel.clone()))?;
                let side = match entry.side {
                    Side::Left => "left",
                    Side::Right => "right",
                    Side::Top => "top",
                    Side::Bottom => "bottom",
                };
                Ok(Element::text(descriptor.title.clone())
                    .class("pk-dock-autohide-tab")
                    .key_str(entry.panel.clone())
                    .class(format_key("pk-dock-autohide-tab--", side)))
            })
            .collect::<Result<Vec<_>, DockViewError>>()?;
        Ok(Element::box_().class("pk-dock-autohide").children(entries))
    }
}

fn split_side(axis: Axis, index: usize, count: usize) -> DockSide {
    match (axis, index, count) {
        (Axis::Horizontal, 0, _) => DockSide::Left,
        (Axis::Horizontal, index, count) if index + 1 == count => DockSide::Right,
        (Axis::Vertical, 0, _) => DockSide::Top,
        (Axis::Vertical, index, count) if index + 1 == count => DockSide::Bottom,
        _ => DockSide::Left,
    }
}

fn format_key(prefix: &str, id: &str) -> String {
    let mut key = String::from(prefix);
    key.push(':');
    key.push_str(id);
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DefaultSlot, PanelDescriptor};

    fn registries() -> (PanelRegistry, PanelViewRegistry, DocumentViewRegistry) {
        let mut panels = PanelRegistry::new();
        panels
            .register(PanelDescriptor::new("outliner", "Outliner"))
            .unwrap();
        panels
            .register(
                PanelDescriptor::new("inspector", "Inspector")
                    .with_default_slot(DefaultSlot::RightBar),
            )
            .unwrap();

        let mut views = PanelViewRegistry::new();
        views
            .register("outliner", || Element::box_().class("outliner-content"))
            .unwrap();
        views
            .register("inspector", || Element::box_().class("inspector-content"))
            .unwrap();
        (panels, views, DocumentViewRegistry::new())
    }

    #[test]
    fn assembles_registered_panel_and_split_surfaces() {
        let (panels, views, documents) = registries();
        let layout = DockLayout {
            root: DockNode::Split {
                axis: Axis::Horizontal,
                children: vec![
                    crate::SplitChild {
                        ratio: crate::Ratio::HALF,
                        node: DockNode::panel("outliner"),
                    },
                    crate::SplitChild {
                        ratio: crate::Ratio::HALF,
                        node: DockNode::panel("inspector"),
                    },
                ],
            },
            ..DockLayout::default()
        };
        let element = DockView::new(&layout, &panels, &views, &documents)
            .render()
            .unwrap();
        assert_eq!(element.class_names(), ["pk-dock-root"]);
        assert_eq!(element.child_elements().len(), 3);
        assert_eq!(
            element.child_elements()[0].class_names(),
            ["pk-dock-split", "pk-dock-split--horizontal"]
        );
        assert_eq!(
            element.child_elements()[0].inline_pairs()[0],
            (StyleProp::Display, StyleValue::keyword(Keyword::Flex))
        );
        assert_eq!(
            element.child_elements()[0].child_elements()[0].inline_pairs()[0],
            (StyleProp::FlexGrow, StyleValue::number(500_000.0))
        );
    }

    #[test]
    fn missing_factory_is_reported_before_partial_render() {
        let (panels, _views, documents) = registries();
        let layout = DockLayout {
            root: DockNode::panel("outliner"),
            ..DockLayout::default()
        };
        let error = DockView::new(&layout, &panels, &PanelViewRegistry::new(), &documents)
            .render()
            .unwrap_err();
        assert_eq!(
            error,
            DockViewError::MissingPanelFactory(String::from("outliner"))
        );
    }

    #[test]
    fn duplicate_view_registration_is_rejected_without_replacement() {
        let mut views = PanelViewRegistry::new();
        views.register("panel", Element::box_).unwrap();
        assert_eq!(
            views.register("panel", || Element::text("other")),
            Err(String::from("panel"))
        );
        let element = (views.get("panel").unwrap())();
        assert_eq!(element.kind(), &prism_ui::ElementKind::Box);
    }

    #[test]
    fn floating_rect_is_translated_to_absolute_geometry() {
        let (panels, views, documents) = registries();
        let layout = DockLayout {
            root: DockNode::Empty,
            floating: vec![crate::FloatingWindow {
                rect: crate::Rect {
                    x: 12,
                    y: 24,
                    width: 320,
                    height: 180,
                },
                root: DockNode::panel("outliner"),
                z: 7,
            }],
            ..DockLayout::default()
        };
        let element = DockView::new(&layout, &panels, &views, &documents)
            .render()
            .unwrap();
        let floating = &element.child_elements()[1].child_elements()[0];
        assert_eq!(
            floating.inline_pairs(),
            &[
                (StyleProp::Position, StyleValue::keyword(Keyword::Absolute)),
                (StyleProp::Left, StyleValue::px(12.0)),
                (StyleProp::Top, StyleValue::px(24.0)),
                (StyleProp::Width, StyleValue::px(320.0)),
                (StyleProp::Height, StyleValue::px(180.0)),
            ]
        );
    }
}
