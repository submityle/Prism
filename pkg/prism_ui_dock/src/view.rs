//! Element assembly for the pure-data dock model.
//!
//! The core crate remains renderer-independent. This feature-gated adapter
//! resolves registered factories into `prism_ui::Element` trees and uses the
//! existing `DockPanel` shell for panel surfaces; GPU painting remains the
//! responsibility of the host UI renderer.

use alloc::{boxed::Box, string::String, vec::Vec};
use core::ops::Range;

use prism_ui::style::{Keyword, StyleProp, StyleValue};
use prism_ui::Element;
use prism_ui_component::Component;
use prism_ui_component_kit::editor::{DockPanel, DockPanelProps, DockSide};

use crate::{
    visible_tab_range, Axis, DockLayout, DockNode, DocumentId, DocumentLayout, LayoutError,
    PanelId, PanelLayout, PanelRegistry, PanelVisibility, RenderPlan, Side, SurfaceId, TabId,
};

/// A lazily evaluated panel or document content factory.
pub type ViewFactory = Box<dyn Fn() -> Element + Send + Sync + 'static>;

/// A factory that can report a structured construction failure.
pub type FallibleViewFactory =
    Box<dyn Fn() -> Result<Element, ViewFactoryError> + Send + Sync + 'static>;

/// Read-only metadata supplied to a context-aware view factory.
///
/// The fields are intentionally private and exposed through accessors so
/// adding metadata later does not make downstream factory code exhaustive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ViewFactoryContext<'a> {
    panel_id: Option<&'a str>,
    document_id: Option<&'a str>,
    generation: u64,
    layout_revision: u64,
}

impl<'a> ViewFactoryContext<'a> {
    fn panel(panel_id: &'a str, generation: u64, layout_revision: u64) -> Self {
        Self {
            panel_id: Some(panel_id),
            document_id: None,
            generation,
            layout_revision,
        }
    }

    fn document(document_id: &'a str, generation: u64, layout_revision: u64) -> Self {
        Self {
            panel_id: None,
            document_id: Some(document_id),
            generation,
            layout_revision,
        }
    }

    /// Returns the panel ID for a panel factory, if this is a panel request.
    #[must_use]
    pub fn panel_id(&self) -> Option<&'a str> {
        self.panel_id
    }

    /// Returns the document ID for a document factory, if this is a document request.
    #[must_use]
    pub fn document_id(&self) -> Option<&'a str> {
        self.document_id
    }

    /// Returns the host generation associated with the render request.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the layout revision associated with the render request.
    #[must_use]
    pub const fn layout_revision(&self) -> u64 {
        self.layout_revision
    }
}

/// A context-aware factory that can report a structured construction failure.
pub type ContextualViewFactory = Box<
    dyn for<'a> Fn(&ViewFactoryContext<'a>) -> Result<Element, ViewFactoryError>
        + Send
        + Sync
        + 'static,
>;

/// Error returned by a panel or document factory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ViewFactoryError {
    /// Stable machine-readable error code.
    pub code: String,
    /// Human-readable diagnostic without layout contents.
    pub message: String,
}

/// Rendering policy for materializing tab headers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DockViewOptions {
    /// Horizontal tab-strip scroll offset in logical pixels.
    pub tab_scroll_px: u32,
    /// Materialized tab-strip viewport width. `None` materializes all tabs.
    pub tab_viewport_px: Option<u32>,
    /// Estimated tab width used by the deterministic window.
    pub tab_width_px: u32,
    /// Number of tab slots retained around the visible range.
    pub tab_overscan: usize,
}

impl Default for DockViewOptions {
    fn default() -> Self {
        Self {
            tab_scroll_px: 0,
            tab_viewport_px: None,
            tab_width_px: 96,
            tab_overscan: 2,
        }
    }
}

impl DockViewOptions {
    /// Returns the tab indexes that should be materialized for one tab strip.
    ///
    /// A missing viewport keeps the legacy all-tabs behavior. When windowing
    /// is enabled, the active tab is always retained, even when it is outside
    /// the scrolled viewport.
    #[must_use]
    pub fn visible_tab_range(&self, total: usize, active: usize) -> Range<usize> {
        match self.tab_viewport_px {
            None => 0..total,
            Some(viewport) => visible_tab_range(
                total,
                self.tab_scroll_px,
                viewport.max(1),
                self.tab_width_px.max(1),
                self.tab_overscan,
                active.min(total.saturating_sub(1)),
            ),
        }
    }
}

impl ViewFactoryError {
    /// Creates a structured factory error.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

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
    /// A factory rejected construction of its content.
    Factory {
        /// Stable panel or document ID.
        id: String,
        /// Structured factory diagnostic.
        error: ViewFactoryError,
    },
    /// A result was produced for an obsolete host generation.
    StaleGeneration {
        /// Generation captured by the render request.
        requested: u64,
        /// Current host generation.
        current: u64,
    },
    /// The supplied render plan does not describe the validated layout.
    PlanLayoutMismatch {
        /// Stable diagnostic describing the mismatched part.
        part: &'static str,
    },
}

/// Renderer-facing factories for registered panel IDs.
pub struct PanelViewRegistry {
    entries: Vec<(PanelId, ViewFactory)>,
    fallible_entries: Vec<(PanelId, FallibleViewFactory)>,
    contextual_entries: Vec<(PanelId, ContextualViewFactory)>,
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
            fallible_entries: Vec::new(),
            contextual_entries: Vec::new(),
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
        if self
            .fallible_entries
            .iter()
            .any(|(registered, _)| registered == &id)
            || self
                .contextual_entries
                .iter()
                .any(|(registered, _)| registered == &id)
        {
            return Err(id);
        }
        self.entries.push((id, Box::new(factory)));
        Ok(())
    }

    /// Registers a fallible factory, rejecting duplicate IDs across both APIs.
    pub fn register_result(
        &mut self,
        id: impl Into<PanelId>,
        factory: impl Fn() -> Result<Element, ViewFactoryError> + Send + Sync + 'static,
    ) -> Result<(), PanelId> {
        let id = id.into();
        if self.entries.iter().any(|(registered, _)| registered == &id)
            || self
                .fallible_entries
                .iter()
                .any(|(registered, _)| registered == &id)
            || self
                .contextual_entries
                .iter()
                .any(|(registered, _)| registered == &id)
        {
            return Err(id);
        }
        self.fallible_entries.push((id, Box::new(factory)));
        Ok(())
    }

    /// Registers a context-aware fallible factory.
    ///
    /// The factory receives immutable panel identity and host/layout version
    /// metadata at render time. The existing register API remains the
    /// zero-context infallible fast path.
    pub fn register_with_context(
        &mut self,
        id: impl Into<PanelId>,
        factory: impl for<'a> Fn(&ViewFactoryContext<'a>) -> Result<Element, ViewFactoryError>
            + Send
            + Sync
            + 'static,
    ) -> Result<(), PanelId> {
        let id = id.into();
        if self.entries.iter().any(|(registered, _)| registered == &id)
            || self
                .fallible_entries
                .iter()
                .any(|(registered, _)| registered == &id)
            || self
                .contextual_entries
                .iter()
                .any(|(registered, _)| registered == &id)
        {
            return Err(id);
        }
        self.contextual_entries.push((id, Box::new(factory)));
        Ok(())
    }

    /// Explicit alias for the context-aware registration API.
    pub fn register_result_with_context(
        &mut self,
        id: impl Into<PanelId>,
        factory: impl for<'a> Fn(&ViewFactoryContext<'a>) -> Result<Element, ViewFactoryError>
            + Send
            + Sync
            + 'static,
    ) -> Result<(), PanelId> {
        self.register_with_context(id, factory)
    }

    /// Removes a registered panel factory from the active render registry.
    pub fn unregister(&mut self, id: &str) -> bool {
        if let Some(index) = self
            .entries
            .iter()
            .position(|(registered, _)| registered == id)
        {
            drop(self.entries.remove(index));
            return true;
        }
        if let Some(index) = self
            .fallible_entries
            .iter()
            .position(|(registered, _)| registered == id)
        {
            drop(self.fallible_entries.remove(index));
            return true;
        }
        if let Some(index) = self
            .contextual_entries
            .iter()
            .position(|(registered, _)| registered == id)
        {
            drop(self.contextual_entries.remove(index));
            return true;
        }
        false
    }

    /// Returns the infallible factory registered for `id`, if present.
    ///
    /// Fallible factories are intentionally excluded from this accessor; use
    /// [`Self::build`] when the render path must handle both factory kinds.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&ViewFactory> {
        self.entries
            .iter()
            .find(|(registered, _)| registered == id)
            .map(|(_, factory)| factory)
    }

    fn build(&self, id: &str, context: &ViewFactoryContext<'_>) -> Result<Element, DockViewError> {
        if let Some((_, factory)) = self
            .contextual_entries
            .iter()
            .find(|(registered, _)| registered == id)
        {
            return factory(context).map_err(|error| DockViewError::Factory {
                id: id.into(),
                error,
            });
        }
        if let Some((_, factory)) = self
            .fallible_entries
            .iter()
            .find(|(registered, _)| registered == id)
        {
            return factory().map_err(|error| DockViewError::Factory {
                id: id.into(),
                error,
            });
        }
        self.entries
            .iter()
            .find(|(registered, _)| registered == id)
            .map(|(_, factory)| factory())
            .ok_or_else(|| DockViewError::MissingPanelFactory(id.into()))
    }
}

/// Renderer-facing factories for document IDs.
pub struct DocumentViewRegistry {
    entries: Vec<(DocumentId, ViewFactory)>,
    fallible_entries: Vec<(DocumentId, FallibleViewFactory)>,
    contextual_entries: Vec<(DocumentId, ContextualViewFactory)>,
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
            fallible_entries: Vec::new(),
            contextual_entries: Vec::new(),
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
        if self
            .fallible_entries
            .iter()
            .any(|(registered, _)| registered == &id)
            || self
                .contextual_entries
                .iter()
                .any(|(registered, _)| registered == &id)
        {
            return Err(id);
        }
        self.entries.push((id, Box::new(factory)));
        Ok(())
    }

    /// Registers a fallible document factory.
    pub fn register_result(
        &mut self,
        id: impl Into<DocumentId>,
        factory: impl Fn() -> Result<Element, ViewFactoryError> + Send + Sync + 'static,
    ) -> Result<(), DocumentId> {
        let id = id.into();
        if self.entries.iter().any(|(registered, _)| registered == &id)
            || self
                .fallible_entries
                .iter()
                .any(|(registered, _)| registered == &id)
            || self
                .contextual_entries
                .iter()
                .any(|(registered, _)| registered == &id)
        {
            return Err(id);
        }
        self.fallible_entries.push((id, Box::new(factory)));
        Ok(())
    }

    /// Registers a context-aware fallible document factory.
    pub fn register_with_context(
        &mut self,
        id: impl Into<DocumentId>,
        factory: impl for<'a> Fn(&ViewFactoryContext<'a>) -> Result<Element, ViewFactoryError>
            + Send
            + Sync
            + 'static,
    ) -> Result<(), DocumentId> {
        let id = id.into();
        if self.entries.iter().any(|(registered, _)| registered == &id)
            || self
                .fallible_entries
                .iter()
                .any(|(registered, _)| registered == &id)
            || self
                .contextual_entries
                .iter()
                .any(|(registered, _)| registered == &id)
        {
            return Err(id);
        }
        self.contextual_entries.push((id, Box::new(factory)));
        Ok(())
    }

    /// Explicit alias for the context-aware registration API.
    pub fn register_result_with_context(
        &mut self,
        id: impl Into<DocumentId>,
        factory: impl for<'a> Fn(&ViewFactoryContext<'a>) -> Result<Element, ViewFactoryError>
            + Send
            + Sync
            + 'static,
    ) -> Result<(), DocumentId> {
        self.register_with_context(id, factory)
    }

    /// Removes a registered document factory from the active render registry.
    pub fn unregister(&mut self, id: &str) -> bool {
        if let Some(index) = self
            .entries
            .iter()
            .position(|(registered, _)| registered == id)
        {
            drop(self.entries.remove(index));
            return true;
        }
        if let Some(index) = self
            .fallible_entries
            .iter()
            .position(|(registered, _)| registered == id)
        {
            drop(self.fallible_entries.remove(index));
            return true;
        }
        if let Some(index) = self
            .contextual_entries
            .iter()
            .position(|(registered, _)| registered == id)
        {
            drop(self.contextual_entries.remove(index));
            return true;
        }
        false
    }

    fn build(&self, id: &str, context: &ViewFactoryContext<'_>) -> Result<Element, DockViewError> {
        if let Some((_, factory)) = self
            .contextual_entries
            .iter()
            .find(|(registered, _)| registered == id)
        {
            return factory(context).map_err(|error| DockViewError::Factory {
                id: id.into(),
                error,
            });
        }
        if let Some((_, factory)) = self
            .fallible_entries
            .iter()
            .find(|(registered, _)| registered == id)
        {
            return factory().map_err(|error| DockViewError::Factory {
                id: id.into(),
                error,
            });
        }
        self.entries
            .iter()
            .find(|(registered, _)| registered == id)
            .map(|(_, factory)| factory())
            .ok_or_else(|| DockViewError::MissingDocumentFactory(id.into()))
    }
}

/// Converts a validated [`DockLayout`] into an Element tree.
pub struct DockView<'a> {
    layout: &'a DockLayout,
    panels: &'a PanelRegistry,
    panel_views: &'a PanelViewRegistry,
    document_views: &'a DocumentViewRegistry,
    options: DockViewOptions,
}

impl<'a> DockView<'a> {
    /// Creates an Element assembler for a layout and its registered views.
    #[must_use]
    pub fn new(
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
            options: DockViewOptions::default(),
        }
    }

    /// Returns a copy configured to materialize only the visible tab window.
    #[must_use]
    pub const fn with_tab_window(
        mut self,
        scroll_px: u32,
        viewport_px: u32,
        tab_width_px: u32,
        overscan: usize,
    ) -> Self {
        self.options = DockViewOptions {
            tab_scroll_px: scroll_px,
            tab_viewport_px: Some(viewport_px),
            tab_width_px,
            tab_overscan: overscan,
        };
        self
    }

    /// Builds the complete main, floating, and auto-hide surfaces.
    pub fn render(&self) -> Result<Element, DockViewError> {
        self.render_if_generation(0, 0)
    }

    /// Builds an Element tree from an already solved [`RenderPlan`].
    ///
    /// This is the authoritative geometry seam for a host that has already
    /// solved layout: active content is positioned from plan rectangles so
    /// renderer layout cannot silently diverge from the deterministic solver.
    pub fn render_plan(&self, plan: &RenderPlan) -> Result<Element, DockViewError> {
        self.render_plan_with_context(plan, 0, 0)
    }

    /// Builds a plan-driven Element tree with factory request metadata.
    pub fn render_plan_with_context(
        &self,
        plan: &RenderPlan,
        generation: u64,
        layout_revision: u64,
    ) -> Result<Element, DockViewError> {
        self.layout
            .validate()
            .map_err(DockViewError::InvalidLayout)?;
        self.validate_plan_identity(plan)?;

        let mut main_children = Vec::new();
        let mut autohide_children = Vec::new();
        let mut floating_layers = Vec::new();

        for panel in &plan.panels {
            if !panel.active || panel.visibility == PanelVisibility::AutoHideCollapsed {
                continue;
            }
            let element = self.render_plan_panel(plan, panel, generation, layout_revision)?;
            match panel.surface {
                SurfaceId::Main => main_children.push(element),
                SurfaceId::AutoHide(_) => autohide_children.push(element),
                SurfaceId::Floating(_) => {}
            }
        }
        for document in &plan.documents {
            if !document.active {
                continue;
            }
            let element = self.render_plan_document(plan, document, generation, layout_revision)?;
            match document.surface {
                SurfaceId::Main => main_children.push(element),
                SurfaceId::AutoHide(_) => autohide_children.push(element),
                SurfaceId::Floating(_) => {}
            }
        }

        for window in &plan.floating {
            let mut children = Vec::new();
            for panel in &plan.panels {
                if panel.active
                    && panel.visibility != PanelVisibility::AutoHideCollapsed
                    && panel.surface == window.surface
                {
                    children.push(self.render_plan_panel(
                        plan,
                        panel,
                        generation,
                        layout_revision,
                    )?);
                }
            }
            for document in &plan.documents {
                if document.active && document.surface == window.surface {
                    children.push(self.render_plan_document(
                        plan,
                        document,
                        generation,
                        layout_revision,
                    )?);
                }
            }
            if !children.is_empty() {
                floating_layers.push(
                    Element::box_()
                        .class("pk-dock-plan-floating")
                        .key_int(i64::from(window.z))
                        .children(children),
                );
            }
        }

        for bar in &plan.autohide.bars {
            let buttons = bar
                .buttons
                .iter()
                .filter_map(|button| {
                    self.panels.resolve(&button.panel).map(|descriptor| {
                        Element::text(descriptor.title.clone())
                            .class("pk-dock-autohide-tab")
                            .class(format!("pk-dock-autohide-tab--{}", side_name(button.side)))
                            .key_str(button.panel.clone())
                    })
                })
                .collect::<Vec<_>>();
            autohide_children.push(absolute_box(
                Element::box_()
                    .class("pk-dock-plan-autohide-bar")
                    .class(format!(
                        "pk-dock-plan-autohide-bar--{}",
                        side_name(bar.side)
                    ))
                    .children(buttons),
                bar.rect,
            ));
        }

        let mut children = vec![
            Element::box_()
                .class("pk-dock-plan-main")
                .children(main_children),
            Element::box_()
                .class("pk-dock-plan-autohide")
                .children(autohide_children),
        ];
        children.extend(floating_layers);
        Ok(Element::box_()
            .class("pk-dock-plan-root")
            .style(StyleProp::Position, StyleValue::keyword(Keyword::Relative))
            .style(StyleProp::Width, StyleValue::px(plan.viewport.width as f32))
            .style(
                StyleProp::Height,
                StyleValue::px(plan.viewport.height as f32),
            )
            .children(children))
    }

    /// Builds the Element tree only when the request generation is current.
    /// Hosts that construct views asynchronously can use this guard before
    /// committing the returned tree.
    pub fn render_if_generation(
        &self,
        requested_generation: u64,
        current_generation: u64,
    ) -> Result<Element, DockViewError> {
        if requested_generation != 0 && requested_generation != current_generation {
            return Err(DockViewError::StaleGeneration {
                requested: requested_generation,
                current: current_generation,
            });
        }
        let generation = if requested_generation == 0 {
            current_generation
        } else {
            requested_generation
        };
        self.render_if_generation_with_context(generation, 0)
    }

    /// Builds the legacy layout-driven tree with explicit factory metadata.
    pub fn render_with_context(
        &self,
        generation: u64,
        layout_revision: u64,
    ) -> Result<Element, DockViewError> {
        self.render_if_generation_with_context(generation, layout_revision)
    }

    fn render_if_generation_with_context(
        &self,
        generation: u64,
        layout_revision: u64,
    ) -> Result<Element, DockViewError> {
        self.layout
            .validate()
            .map_err(DockViewError::InvalidLayout)?;
        self.validate_panel_registrations(&self.layout.root)?;
        for window in &self.layout.floating {
            self.validate_panel_registrations(&window.root)?;
        }
        if let Some(snapshot) = &self.layout.maximized {
            self.validate_panel_registrations(&snapshot.root)?;
            for window in &snapshot.floating {
                self.validate_panel_registrations(&window.root)?;
            }
        }
        for entry in &self.layout.autohide.entries {
            if self.panels.resolve(&entry.panel).is_none() {
                return Err(DockViewError::UnknownPanel(entry.panel.clone()));
            }
        }

        let mut root = Element::box_().class("pk-dock-root");
        root = root.child(self.render_node(
            &self.layout.root,
            DockSide::Left,
            generation,
            layout_revision,
        )?);
        root = root.child(self.render_floating(generation, layout_revision)?);
        root = root.child(self.render_autohide(generation, layout_revision)?);
        Ok(root)
    }

    fn render_plan_panel(
        &self,
        plan: &RenderPlan,
        panel: &PanelLayout,
        generation: u64,
        layout_revision: u64,
    ) -> Result<Element, DockViewError> {
        let descriptor = self
            .panels
            .resolve(&panel.id)
            .ok_or_else(|| DockViewError::UnknownPanel(panel.id.clone()))?;
        let content = self
            .panel_views
            .build(
                &panel.id,
                &ViewFactoryContext::panel(&panel.id, generation, layout_revision),
            )?
            .key_str(format_key("plan-panel", &panel.id));
        let tabs = self
            .materialize_plan_tabs(plan, panel.surface, &panel.node)
            .into_iter()
            .map(|tab| {
                let (id, title) = match &tab.id {
                    TabId::Panel(id) => {
                        let title = self
                            .panels
                            .resolve(id)
                            .map(|item| item.title.clone())
                            .unwrap_or_else(|| id.clone());
                        (id.clone(), title)
                    }
                    TabId::Document(id) => (id.clone(), id.clone()),
                };
                let class = if tab.active {
                    "pk-dock-tab pk-dock-tab--active"
                } else {
                    "pk-dock-tab"
                };
                Element::text(title).class(class).key_str(id)
            })
            .collect::<Vec<_>>();
        let rendered = DockPanel.render(
            &DockPanelProps::new(descriptor.title.clone())
                .side(plan_side(panel.surface))
                .children([
                    Element::box_().class("pk-dock-tab-bar").children(tabs),
                    content,
                ]),
        );
        Ok(absolute_box(
            rendered
                .class("pk-dock-plan-panel")
                .key_str(format_key("plan", &panel.id)),
            panel.rect,
        ))
    }

    fn render_plan_document(
        &self,
        plan: &RenderPlan,
        document: &DocumentLayout,
        generation: u64,
        layout_revision: u64,
    ) -> Result<Element, DockViewError> {
        let content = self
            .document_views
            .build(
                &document.id,
                &ViewFactoryContext::document(&document.id, generation, layout_revision),
            )?
            .key_str(format_key("plan-document", &document.id));
        let tabs = self
            .materialize_plan_tabs(plan, document.surface, &document.node)
            .into_iter()
            .map(|tab| {
                let id = match &tab.id {
                    TabId::Document(id) | TabId::Panel(id) => id.clone(),
                };
                let class = if tab.active {
                    "pk-dock-tab pk-dock-tab--active"
                } else {
                    "pk-dock-tab"
                };
                Element::text(id.clone()).class(class).key_str(id)
            })
            .collect::<Vec<_>>();
        Ok(absolute_box(
            Element::box_()
                .class("pk-dock-document pk-dock-plan-document")
                .key_str(format_key("plan-document", &document.id))
                .child(Element::box_().class("pk-dock-tab-bar").children(tabs))
                .child(content),
            document.rect,
        ))
    }

    fn materialize_plan_tabs<'b>(
        &self,
        plan: &'b RenderPlan,
        surface: SurfaceId,
        node: &crate::NodePath,
    ) -> Vec<&'b crate::TabHeader> {
        let matching = plan
            .tabs
            .iter()
            .filter(|tab| tab.surface == surface && tab.node == *node)
            .collect::<Vec<_>>();
        let active = matching.iter().position(|tab| tab.active).unwrap_or(0);
        let range = self.options.visible_tab_range(matching.len(), active);
        matching
            .into_iter()
            .skip(range.start)
            .take(range.end.saturating_sub(range.start))
            .collect()
    }

    fn validate_panel_registrations(&self, node: &DockNode) -> Result<(), DockViewError> {
        match node {
            DockNode::Empty | DockNode::Document { .. } => Ok(()),
            DockNode::TabStack { tabs, .. } => {
                for id in tabs {
                    if self.panels.resolve(id).is_none() {
                        return Err(DockViewError::UnknownPanel(id.clone()));
                    }
                }
                Ok(())
            }
            DockNode::Split { children, .. } => {
                for child in children {
                    self.validate_panel_registrations(&child.node)?;
                }
                Ok(())
            }
        }
    }

    fn validate_plan_identity(&self, plan: &RenderPlan) -> Result<(), DockViewError> {
        let mut expected_panels = Vec::new();
        let mut expected_documents = Vec::new();
        collect_node_ids(
            &self.layout.root,
            &mut expected_panels,
            &mut expected_documents,
        );
        for window in &self.layout.floating {
            collect_node_ids(&window.root, &mut expected_panels, &mut expected_documents);
        }
        for entry in &self.layout.autohide.entries {
            expected_panels.push(entry.panel.clone());
        }

        let mut actual_panels = plan
            .panels
            .iter()
            .map(|panel| panel.id.clone())
            .collect::<Vec<_>>();
        let mut actual_documents = plan
            .documents
            .iter()
            .map(|document| document.id.clone())
            .collect::<Vec<_>>();
        expected_panels.sort();
        actual_panels.sort();
        expected_documents.sort();
        actual_documents.sort();
        if expected_panels != actual_panels {
            return Err(DockViewError::PlanLayoutMismatch { part: "panels" });
        }
        if expected_documents != actual_documents {
            return Err(DockViewError::PlanLayoutMismatch { part: "documents" });
        }
        if plan.floating.len() != self.layout.floating.len()
            || plan.floating.iter().enumerate().any(|(index, window)| {
                window.index != index
                    || window.rect != self.layout.floating[index].rect
                    || window.z != self.layout.floating[index].z
            })
        {
            return Err(DockViewError::PlanLayoutMismatch { part: "floating" });
        }
        Ok(())
    }

    fn render_node(
        &self,
        node: &DockNode,
        side: DockSide,
        generation: u64,
        layout_revision: u64,
    ) -> Result<Element, DockViewError> {
        match node {
            DockNode::Empty => Ok(Element::box_().class("pk-dock-empty")),
            DockNode::TabStack { tabs, active } => {
                let active_id = &tabs[*active];
                let descriptor = self
                    .panels
                    .resolve(active_id)
                    .ok_or_else(|| DockViewError::UnknownPanel(active_id.clone()))?;
                let range = self.tab_range(tabs.len(), *active);
                let tabs = tabs[range.clone()].iter().enumerate().map(|(offset, id)| {
                    let index = offset + range.start;
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
                let content = self
                    .panel_views
                    .build(
                        active_id,
                        &ViewFactoryContext::panel(active_id, generation, layout_revision),
                    )
                    .map(|element| element.key_str(format_key("panel", active_id)))?;
                Ok(DockPanel.render(
                    &DockPanelProps::new(descriptor.title.clone())
                        .side(side)
                        .children([tab_bar, content]),
                ))
            }
            DockNode::Document { tabs, active } => {
                let id = &tabs[*active];
                let range = self.tab_range(tabs.len(), *active);
                let content = self.document_views.build(
                    id,
                    &ViewFactoryContext::document(id, generation, layout_revision),
                )?;
                let tab_bar = Element::box_().class("pk-dock-tab-bar").children(
                    tabs[range.clone()].iter().enumerate().map(|(offset, id)| {
                        let index = offset + range.start;
                        let class = if index == *active {
                            "pk-dock-tab pk-dock-tab--active"
                        } else {
                            "pk-dock-tab"
                        };
                        Element::text(id.clone()).class(class).key_str(id.clone())
                    }),
                );
                Ok(Element::box_()
                    .class("pk-dock-document")
                    .key_str(format_key("document", id))
                    .child(tab_bar)
                    .child(content))
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
                        self.render_node(&child.node, child_side, generation, layout_revision)
                            .map(|node| {
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

    fn tab_range(&self, total: usize, active: usize) -> Range<usize> {
        self.options.visible_tab_range(total, active)
    }

    fn render_floating(
        &self,
        generation: u64,
        layout_revision: u64,
    ) -> Result<Element, DockViewError> {
        let mut ordered_windows = self.layout.floating.iter().collect::<Vec<_>>();
        ordered_windows.sort_by_key(|window| window.z);
        let windows = ordered_windows
            .into_iter()
            .map(|window| {
                self.render_node(
                    &window.root,
                    DockSide::Floating,
                    generation,
                    layout_revision,
                )
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

    fn render_autohide(
        &self,
        generation: u64,
        layout_revision: u64,
    ) -> Result<Element, DockViewError> {
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
                let tab = Element::text(descriptor.title.clone())
                    .class("pk-dock-autohide-tab")
                    .key_str(entry.panel.clone())
                    .class(format!("pk-dock-autohide-tab--{side}"));
                Ok(tab)
            })
            .collect::<Result<Vec<_>, DockViewError>>()?;
        let mut children = vec![Element::box_()
            .class("pk-dock-autohide-bar")
            .children(entries)];
        if let Some(active) = &self.layout.autohide.active {
            let entry = self
                .layout
                .autohide
                .entries
                .iter()
                .find(|entry| &entry.panel == active)
                .ok_or_else(|| DockViewError::UnknownPanel(active.clone()))?;
            let descriptor = self
                .panels
                .resolve(active)
                .ok_or_else(|| DockViewError::UnknownPanel(active.clone()))?;
            let side = match entry.side {
                Side::Left => DockSide::Left,
                Side::Right => DockSide::Right,
                Side::Top => DockSide::Top,
                Side::Bottom => DockSide::Bottom,
            };
            let content = self
                .panel_views
                .build(
                    active,
                    &ViewFactoryContext::panel(active, generation, layout_revision),
                )
                .map(|element| element.key_str(format_key("autohide", active)))?;
            let panel = DockPanel.render(
                &DockPanelProps::new(descriptor.title.clone())
                    .side(side)
                    .children([content]),
            );
            children.push(
                Element::box_()
                    .class("pk-dock-autohide-reveal")
                    .child(panel),
            );
        }
        Ok(Element::box_().class("pk-dock-autohide").children(children))
    }
}

fn collect_node_ids(node: &DockNode, panels: &mut Vec<PanelId>, documents: &mut Vec<DocumentId>) {
    match node {
        DockNode::Empty => {}
        DockNode::TabStack { tabs, .. } => panels.extend(tabs.iter().cloned()),
        DockNode::Document { tabs, .. } => documents.extend(tabs.iter().cloned()),
        DockNode::Split { children, .. } => {
            for child in children {
                collect_node_ids(&child.node, panels, documents);
            }
        }
    }
}

impl crate::DockHost {
    /// Solves the host state and assembles the plan-driven Element tree in one
    /// render-feature adapter call.
    pub fn render_element(
        &mut self,
        viewport: crate::Viewport,
        metrics: crate::LayoutMetrics,
        panels: &PanelRegistry,
        panel_views: &PanelViewRegistry,
        document_views: &DocumentViewRegistry,
    ) -> Result<Element, crate::DockHostError> {
        let plan = self.plan(viewport, metrics)?.clone();
        let revision = self.revision();
        DockView::new(self.layout(), panels, panel_views, document_views)
            .render_plan_with_context(&plan, revision, revision)
            .map_err(crate::DockHostError::View)
    }
}

fn absolute_box(element: Element, rect: crate::Rect) -> Element {
    element
        .style(StyleProp::Position, StyleValue::keyword(Keyword::Absolute))
        .style(StyleProp::Left, StyleValue::px(rect.x as f32))
        .style(StyleProp::Top, StyleValue::px(rect.y as f32))
        .style(StyleProp::Width, StyleValue::px(rect.width as f32))
        .style(StyleProp::Height, StyleValue::px(rect.height as f32))
}

fn plan_side(surface: SurfaceId) -> DockSide {
    match surface {
        SurfaceId::AutoHide(side) => match side {
            Side::Left => DockSide::Left,
            Side::Right => DockSide::Right,
            Side::Top => DockSide::Top,
            Side::Bottom => DockSide::Bottom,
        },
        SurfaceId::Main | SurfaceId::Floating(_) => DockSide::Floating,
    }
}

fn side_name(side: Side) -> &'static str {
    match side {
        Side::Left => "left",
        Side::Right => "right",
        Side::Top => "top",
        Side::Bottom => "bottom",
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
    fn render_plan_uses_solver_rectangles_for_absolute_content() {
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
        let plan = crate::solve(&layout, crate::Viewport::new(1200, 480)).unwrap();
        let element = DockView::new(&layout, &panels, &views, &documents)
            .render_plan(&plan)
            .unwrap();
        assert_eq!(element.class_names(), ["pk-dock-plan-root"]);
        let main = &element.child_elements()[0];
        assert_eq!(main.class_names(), ["pk-dock-plan-main"]);
        assert_eq!(main.child_elements().len(), 2);
        assert_eq!(
            main.child_elements()[0].inline_pairs()[0],
            (StyleProp::Position, StyleValue::keyword(Keyword::Absolute))
        );
        assert_eq!(
            main.child_elements()[0].inline_pairs()[1],
            (StyleProp::Left, StyleValue::px(0.0))
        );
    }

    #[test]
    fn render_plan_rejects_a_plan_from_another_layout() {
        let (panels, views, documents) = registries();
        let layout = DockLayout {
            root: DockNode::panel("outliner"),
            ..DockLayout::default()
        };
        let foreign = crate::solve(
            &DockLayout {
                root: DockNode::panel("inspector"),
                ..DockLayout::default()
            },
            crate::Viewport::new(640, 480),
        )
        .unwrap();
        assert_eq!(
            DockView::new(&layout, &panels, &views, &documents)
                .render_plan(&foreign)
                .unwrap_err(),
            DockViewError::PlanLayoutMismatch { part: "panels" }
        );
    }

    #[test]
    fn autohide_active_panel_renders_reveal_and_valid_side_class() {
        let (panels, views, documents) = registries();
        let layout = DockLayout {
            root: DockNode::Empty,
            autohide: crate::AutoHideBars {
                entries: vec![crate::AutoHideEntry {
                    side: Side::Right,
                    panel: String::from("inspector"),
                }],
                active: Some(String::from("inspector")),
            },
            ..DockLayout::default()
        };
        let element = DockView::new(&layout, &panels, &views, &documents)
            .render()
            .unwrap();
        let autohide = &element.child_elements()[2];
        assert_eq!(autohide.class_names(), ["pk-dock-autohide"]);
        assert_eq!(autohide.child_elements().len(), 2);
        assert_eq!(
            autohide.child_elements()[0].child_elements()[0].class_names(),
            ["pk-dock-autohide-tab", "pk-dock-autohide-tab--right"]
        );
        assert_eq!(
            autohide.child_elements()[1].class_names(),
            ["pk-dock-autohide-reveal"]
        );
        assert_eq!(
            autohide.child_elements()[1].child_elements()[0].class_names(),
            ["pk-dock-panel", "pk-dock-panel--right"]
        );
    }

    #[test]
    fn autohide_active_panel_requires_only_reveal_factory() {
        let (panels, _views, documents) = registries();
        let layout = DockLayout {
            root: DockNode::Empty,
            autohide: crate::AutoHideBars {
                entries: vec![crate::AutoHideEntry {
                    side: Side::Left,
                    panel: String::from("outliner"),
                }],
                active: Some(String::from("outliner")),
            },
            ..DockLayout::default()
        };
        assert_eq!(
            DockView::new(&layout, &panels, &PanelViewRegistry::new(), &documents)
                .render()
                .unwrap_err(),
            DockViewError::MissingPanelFactory(String::from("outliner"))
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
    fn floating_windows_are_painted_in_ascending_z_order() {
        let (panels, views, documents) = registries();
        let layout = DockLayout {
            root: DockNode::Empty,
            floating: vec![
                crate::FloatingWindow {
                    rect: crate::Rect {
                        x: 0,
                        y: 0,
                        width: 100,
                        height: 100,
                    },
                    root: DockNode::panel("outliner"),
                    z: 10,
                },
                crate::FloatingWindow {
                    rect: crate::Rect {
                        x: 10,
                        y: 10,
                        width: 100,
                        height: 100,
                    },
                    root: DockNode::panel("inspector"),
                    z: 1,
                },
            ],
            ..DockLayout::default()
        };
        let element = DockView::new(&layout, &panels, &views, &documents)
            .render()
            .unwrap();
        let windows = element.child_elements()[1].child_elements();
        assert_eq!(windows.len(), 2);
        assert_eq!(
            windows[0].inline_pairs()[1],
            (StyleProp::Left, StyleValue::px(10.0))
        );
        assert_eq!(
            windows[1].inline_pairs()[1],
            (StyleProp::Left, StyleValue::px(0.0))
        );
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

    #[test]
    fn fallible_factory_errors_are_structured_and_generation_is_guarded() {
        let mut panels = PanelRegistry::new();
        panels
            .register(PanelDescriptor::new("panel", "Panel"))
            .unwrap();
        let mut views = PanelViewRegistry::new();
        views
            .register_result("panel", || {
                Err(ViewFactoryError::new("loading", "panel data unavailable"))
            })
            .unwrap();
        let layout = DockLayout {
            root: DockNode::panel("panel"),
            ..DockLayout::default()
        };
        let documents = DocumentViewRegistry::new();
        let dock = DockView::new(&layout, &panels, &views, &documents);
        assert_eq!(
            dock.render().unwrap_err(),
            DockViewError::Factory {
                id: "panel".into(),
                error: ViewFactoryError::new("loading", "panel data unavailable"),
            }
        );
        assert_eq!(
            dock.render_if_generation(4, 5).unwrap_err(),
            DockViewError::StaleGeneration {
                requested: 4,
                current: 5,
            }
        );
    }

    #[test]
    fn tab_window_materializes_only_visible_headers_and_active_tab() {
        let mut panels = PanelRegistry::new();
        let mut views = PanelViewRegistry::new();
        let mut tabs = Vec::new();
        for index in 0..10 {
            let id = format!("panel-{index}");
            panels
                .register(PanelDescriptor::new(id.clone(), id.clone()))
                .unwrap();
            views.register(id.clone(), Element::box_).unwrap();
            tabs.push(id);
        }
        let layout = DockLayout {
            root: DockNode::TabStack { tabs, active: 7 },
            ..DockLayout::default()
        };
        let documents = DocumentViewRegistry::new();
        let element = DockView::new(&layout, &panels, &views, &documents)
            .with_tab_window(500, 200, 100, 0)
            .render()
            .unwrap();
        fn count_class(element: &Element, class: &str) -> usize {
            let own = usize::from(element.class_names().iter().any(|name| *name == class));
            own + element
                .child_elements()
                .iter()
                .map(|child| count_class(child, class))
                .sum::<usize>()
        }
        assert_eq!(count_class(&element, "pk-dock-tab"), 2);
    }

    #[test]
    fn plan_render_materializes_only_visible_headers_and_active_tab() {
        let mut panels = PanelRegistry::new();
        let mut views = PanelViewRegistry::new();
        let mut tabs = Vec::new();
        for index in 0..10 {
            let id = format!("panel-{index}");
            panels
                .register(PanelDescriptor::new(id.clone(), id.clone()))
                .unwrap();
            views.register(id.clone(), Element::box_).unwrap();
            tabs.push(id);
        }
        let layout = DockLayout {
            root: DockNode::TabStack { tabs, active: 7 },
            ..DockLayout::default()
        };
        let plan = crate::solve(&layout, crate::Viewport::new(1200, 480)).unwrap();
        let element = DockView::new(&layout, &panels, &views, &DocumentViewRegistry::new())
            .with_tab_window(500, 200, 100, 0)
            .render_plan(&plan)
            .unwrap();
        fn find_tab_bar(element: &Element) -> Option<&Element> {
            if element
                .class_names()
                .iter()
                .any(|class| class == "pk-dock-tab-bar")
            {
                return Some(element);
            }
            element.child_elements().iter().find_map(find_tab_bar)
        }
        let tab_bar = find_tab_bar(&element).expect("plan render should contain a tab bar");
        assert_eq!(tab_bar.class_names(), ["pk-dock-tab-bar"]);
        assert_eq!(tab_bar.child_elements().len(), 3);
        assert!(tab_bar.child_elements().iter().any(|tab| {
            tab.class_names()
                .iter()
                .any(|class| class.contains("pk-dock-tab--active"))
        }));
    }

    #[test]
    fn context_aware_factories_receive_ids_generation_and_revision() {
        let (panels, _views, _documents) = registries();
        let mut panel_views = PanelViewRegistry::new();
        panel_views
            .register_with_context("outliner", |context| {
                assert_eq!(context.panel_id(), Some("outliner"));
                assert_eq!(context.document_id(), None);
                assert_eq!(context.generation(), 17);
                assert_eq!(context.layout_revision(), 23);
                Ok(Element::box_().class("context-panel"))
            })
            .unwrap();
        let panel_layout = DockLayout {
            root: DockNode::panel("outliner"),
            ..DockLayout::default()
        };
        DockView::new(
            &panel_layout,
            &panels,
            &panel_views,
            &DocumentViewRegistry::new(),
        )
        .render_with_context(17, 23)
        .unwrap();

        let mut document_views = DocumentViewRegistry::new();
        document_views
            .register_with_context("document", |context| {
                assert_eq!(context.panel_id(), None);
                assert_eq!(context.document_id(), Some("document"));
                assert_eq!(context.generation(), 17);
                assert_eq!(context.layout_revision(), 23);
                Ok(Element::box_().class("context-document"))
            })
            .unwrap();
        let document_layout = DockLayout {
            root: DockNode::Document {
                tabs: vec![String::from("document")],
                active: 0,
            },
            ..DockLayout::default()
        };
        DockView::new(
            &document_layout,
            &panels,
            &PanelViewRegistry::new(),
            &document_views,
        )
        .render_with_context(17, 23)
        .unwrap();
    }
}
