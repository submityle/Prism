//! Deterministic, renderer-independent dock layout and render planning.
//!
//! [`solve`] is the M1 vertical slice between the pure dock model and a future
//! renderer.  It deliberately emits data only: no `Element`, GPU handle, font
//! measurement, or window-system object is created here.  The solver consumes
//! the fixed-point ratios from [`crate::DockNode`] and performs all rounding
//! with integers, so equal inputs produce byte-for-byte equivalent geometry.
//!
//! The solver is intentionally strict.  A malformed model, an impossible
//! minimum-size constraint, or an inconsistent maximize state is an error;
//! ratios and rectangles are never silently repaired.

use crate::{Axis, DockLayout, DockNode, LayoutError, PanelId, Ratio, Side, ValidationIssue};
use alloc::{string::String, vec, vec::Vec};

/// The viewport supplied to the dock host, in logical pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Viewport {
    /// Width in logical pixels.
    pub width: u32,
    /// Height in logical pixels.
    pub height: u32,
}

impl Viewport {
    /// Creates a viewport from its logical extent.
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    fn rect(self) -> crate::Rect {
        crate::Rect {
            x: 0,
            y: 0,
            width: self.width,
            height: self.height,
        }
    }
}

/// The minimum and chrome metrics used by the deterministic solver.
///
/// These are layout policy, not theme values.  A renderer may style the
/// resulting boxes differently, but changing these values changes geometry and
/// therefore belongs in the layout input/cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutMetrics {
    /// Minimum width of a visible leaf, excluding no additional chrome.
    pub min_pane_width: u32,
    /// Minimum content height of a visible leaf.
    pub min_pane_height: u32,
    /// Height reserved for a tab header row.
    pub tab_bar_height: u32,
    /// Minimum width reserved for each tab header.
    pub tab_min_width: u32,
    /// Width of a visible split line and the space it consumes.
    pub splitter_width: u32,
    /// Extra hit-test padding on each side of a split line.
    pub splitter_hit_slop: u32,
    /// Width/height of an auto-hide edge bar.
    pub autohide_bar_thickness: u32,
    /// Width of a left/right auto-hide reveal.
    pub autohide_reveal_width: u32,
    /// Height of a top/bottom auto-hide reveal.
    pub autohide_reveal_height: u32,
}

impl Default for LayoutMetrics {
    fn default() -> Self {
        Self {
            min_pane_width: 160,
            min_pane_height: 96,
            tab_bar_height: 28,
            tab_min_width: 96,
            splitter_width: 1,
            splitter_hit_slop: 4,
            autohide_bar_thickness: 28,
            autohide_reveal_width: 320,
            autohide_reveal_height: 240,
        }
    }
}

/// Stable identity of a rendered dock surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SurfaceId {
    /// The primary dock host surface.
    Main,
    /// A top-level floating window, addressed by its model index.
    Floating(usize),
    /// The edge surface belonging to an auto-hide bar.
    AutoHide(Side),
}

/// Root-relative node path in a dock surface.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct NodePath {
    /// Child indexes from the surface root to the node.
    pub indices: Vec<usize>,
}

impl NodePath {
    fn root() -> Self {
        Self::default()
    }

    fn child(&self, index: usize) -> Self {
        let mut indices = self.indices.clone();
        indices.push(index);
        Self { indices }
    }
}

/// A tab identity independent of its renderer/widget implementation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TabId {
    /// A tool-panel tab.
    Panel(PanelId),
    /// A document tab.
    Document(String),
}

/// Visibility state of a tool panel in the render plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelVisibility {
    /// The panel is in a visible dock or floating leaf.
    Visible,
    /// The panel is represented by an auto-hide button only.
    AutoHideCollapsed,
    /// The panel is shown in the auto-hide reveal surface.
    AutoHideRevealed,
    /// The panel owns the maximized main surface.
    Maximized,
}

/// Geometry and state for one tool panel tab.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelLayout {
    /// Stable panel identity.
    pub id: PanelId,
    /// Surface containing the panel.
    pub surface: SurfaceId,
    /// Path of the leaf containing the panel, if it is in a dock tree.
    pub node: NodePath,
    /// Outer leaf rectangle.  Inactive tabs share their leaf's outer rect.
    pub rect: crate::Rect,
    /// Content rectangle when the tab is active and rendered.
    pub content_rect: Option<crate::Rect>,
    /// Header/button rectangle for this panel.
    pub tab_rect: crate::Rect,
    /// Whether this is the active tab in its leaf.
    pub active: bool,
    /// Whether the host's deterministic focus target is this panel.
    pub focused: bool,
    /// Panel visibility state.
    pub visibility: PanelVisibility,
}

/// Geometry and state for one document tab.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentLayout {
    /// Stable document identity.
    pub id: String,
    /// Surface containing the document.
    pub surface: SurfaceId,
    /// Path of the document leaf.
    pub node: NodePath,
    /// Outer leaf rectangle.  Inactive documents share their leaf's outer rect.
    pub rect: crate::Rect,
    /// Content rectangle when the document is active and rendered.
    pub content_rect: Option<crate::Rect>,
    /// Header rectangle for this document.
    pub tab_rect: crate::Rect,
    /// Whether this is the active document in its leaf.
    pub active: bool,
    /// Whether the host's deterministic focus target is this document.
    pub focused: bool,
}

/// A tab header primitive for a renderer to paint and hit-test.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TabHeader {
    /// Tab identity.
    pub id: TabId,
    /// Surface and leaf owning the header.
    pub surface: SurfaceId,
    /// Path of the owning leaf.
    pub node: NodePath,
    /// Header rectangle.
    pub rect: crate::Rect,
    /// Whether this tab is selected.
    pub active: bool,
    /// Whether this tab is the focus target.
    pub focused: bool,
}

/// A split line plus an expanded hit-test rectangle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitterLayout {
    /// Surface containing the split.
    pub surface: SurfaceId,
    /// Path of the split node.
    pub node: NodePath,
    /// Split orientation.
    pub axis: Axis,
    /// Visible separator line.
    pub line_rect: crate::Rect,
    /// Hit-test rectangle, clipped to the split's parent rect.
    pub hit_rect: crate::Rect,
    /// Child before the line in the split order.
    pub before_child: usize,
    /// Child after the line in the split order.
    pub after_child: usize,
}

/// The geometry of one floating window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FloatingLayout {
    /// Model index of the floating window.
    pub index: usize,
    /// Window rectangle exactly as supplied by the model.
    pub rect: crate::Rect,
    /// Stable stacking order.
    pub z: u32,
    /// Surface identity used by all descendants.
    pub surface: SurfaceId,
}

/// One auto-hide button in an edge bar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutoHideButton {
    /// Panel represented by this button.
    pub panel: PanelId,
    /// Edge hosting the button.
    pub side: Side,
    /// Button rectangle.
    pub rect: crate::Rect,
    /// Whether the button is the revealed entry.
    pub active: bool,
    /// Whether the button is focused.
    pub focused: bool,
}

/// One edge auto-hide bar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutoHideBarLayout {
    /// Edge of the viewport.
    pub side: Side,
    /// Bar rectangle.
    pub rect: crate::Rect,
    /// Buttons in deterministic model order.
    pub buttons: Vec<AutoHideButton>,
}

/// The optional revealed auto-hide surface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutoHideReveal {
    /// Revealed panel.
    pub panel: PanelId,
    /// Edge from which it was revealed.
    pub side: Side,
    /// Outer panel rectangle.
    pub rect: crate::Rect,
    /// Content rectangle below its single tab header.
    pub content_rect: crate::Rect,
    /// Header rectangle.
    pub tab_rect: crate::Rect,
}

/// All auto-hide geometry for the viewport.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AutoHideLayout {
    /// Non-empty edge bars, in left/right/top/bottom order.
    pub bars: Vec<AutoHideBarLayout>,
    /// The active reveal, if any.
    pub revealed: Option<AutoHideReveal>,
}

/// Maximized-surface metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaximizedLayout {
    /// Maximized panel identity.
    pub panel: PanelId,
    /// Full main dock rectangle occupied by the panel.
    pub rect: crate::Rect,
    /// Whether a restore snapshot is available.
    pub can_restore: bool,
}

/// The target selected by the deterministic focus policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FocusTarget {
    /// A tool panel.
    Panel(PanelId),
    /// A document.
    Document(String),
}

/// Why the focus policy selected its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FocusReason {
    /// The maximized panel owns focus.
    Maximized,
    /// The revealed auto-hide panel owns focus.
    AutoHide,
    /// The topmost floating window owns focus.
    Floating,
    /// The main tree's first active leaf owns focus.
    Main,
}

/// Active/focus metadata emitted with the geometry plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FocusMetadata {
    /// The deterministic focus target, if the layout has any focusable leaf.
    pub target: Option<FocusTarget>,
    /// Reason for choosing the target.
    pub reason: Option<FocusReason>,
}

/// Complete renderer-independent M1 plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderPlan {
    /// Original viewport as a rectangle.
    pub viewport: crate::Rect,
    /// Main dock rectangle after reserving auto-hide bars.
    pub main_rect: crate::Rect,
    /// All tool panels, including collapsed auto-hide entries.
    pub panels: Vec<PanelLayout>,
    /// All documents in visible dock/floating trees.
    pub documents: Vec<DocumentLayout>,
    /// All visible leaf tab headers, including the revealed auto-hide header.
    pub tabs: Vec<TabHeader>,
    /// All split lines and hit regions.
    pub splitters: Vec<SplitterLayout>,
    /// Floating window rectangles and z metadata.
    pub floating: Vec<FloatingLayout>,
    /// Auto-hide bars and optional reveal.
    pub autohide: AutoHideLayout,
    /// Maximized metadata, when the model is maximized.
    pub maximized: Option<MaximizedLayout>,
    /// Active/focus metadata.
    pub focus: FocusMetadata,
}

/// A size requirement failure with enough context for diagnostics/UI logging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeRequirement {
    /// Available width.
    pub available_width: u32,
    /// Available height.
    pub available_height: u32,
    /// Required width.
    pub required_width: u32,
    /// Required height.
    pub required_height: u32,
}

/// Structured failure from [`solve`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderPlanError {
    /// The source model failed its own invariant validation.
    InvalidLayout(LayoutError),
    /// The viewport has no usable extent.
    InvalidViewport(Viewport),
    /// A solver metric is not meaningful.
    InvalidMetrics(&'static str),
    /// A tree cannot fit its assigned rectangle at the configured minimums.
    MinimumSize {
        /// Surface containing the node.
        surface: SurfaceId,
        /// Node path within that surface.
        node: NodePath,
        /// Available and required dimensions.
        requirement: SizeRequirement,
    },
    /// A floating window cannot contain its root tree.
    FloatingTooSmall {
        /// Floating window model index.
        index: usize,
        /// Required dimensions.
        requirement: SizeRequirement,
    },
    /// An auto-hide bar cannot fit all of its buttons.
    AutoHideBarOverflow {
        /// Edge whose bar is too short.
        side: Side,
        /// Available and required bar lengths.
        available: u32,
        /// Required total button length.
        required: u32,
    },
    /// An auto-hide reveal cannot fit the configured minimum pane.
    AutoHideRevealTooSmall {
        /// Revealed panel.
        panel: PanelId,
        /// Edge hosting the panel.
        side: Side,
        /// Available and required dimensions.
        requirement: SizeRequirement,
    },
    /// The current model does not match its maximize snapshot contract.
    InvalidMaximizedState(ValidationIssue),
    /// A checked geometry operation exceeded the integer coordinate range.
    GeometryOverflow,
}

/// Solves a legal [`DockLayout`] into a renderer-independent plan using the
/// default geometry policy.
pub fn solve(layout: &DockLayout, viewport: Viewport) -> Result<RenderPlan, RenderPlanError> {
    solve_with(layout, viewport, LayoutMetrics::default())
}

/// Solves a legal [`DockLayout`] with explicit geometry policy.
pub fn solve_with(
    layout: &DockLayout,
    viewport: Viewport,
    metrics: LayoutMetrics,
) -> Result<RenderPlan, RenderPlanError> {
    layout.validate().map_err(RenderPlanError::InvalidLayout)?;
    validate_metrics(metrics)?;
    if viewport.width == 0 || viewport.height == 0 {
        return Err(RenderPlanError::InvalidViewport(viewport));
    }
    // The model's logical rectangle uses signed 32-bit origins and unsigned
    // extents; reject viewports that cannot be represented without wrapping.
    checked_i32(viewport.width)?;
    checked_i32(viewport.height)?;

    validate_maximized_state(layout)?;
    let viewport_rect = viewport.rect();
    let (main_rect, bars) = solve_autohide_bars(layout, viewport_rect, metrics)?;

    let mut plan = RenderPlan {
        viewport: viewport_rect,
        main_rect,
        panels: Vec::new(),
        documents: Vec::new(),
        tabs: Vec::new(),
        splitters: Vec::new(),
        floating: Vec::new(),
        autohide: bars,
        maximized: layout.maximized.as_ref().map(|snapshot| MaximizedLayout {
            panel: snapshot.panel.clone(),
            rect: main_rect,
            can_restore: true,
        }),
        focus: FocusMetadata {
            target: None,
            reason: None,
        },
    };

    solve_node(
        &layout.root,
        main_rect,
        SurfaceId::Main,
        &NodePath::root(),
        &mut plan,
        metrics,
    )?;

    for (index, window) in layout.floating.iter().enumerate() {
        checked_add_i32(window.rect.x, checked_i32(window.rect.width)?)?;
        checked_add_i32(window.rect.y, checked_i32(window.rect.height)?)?;
        let surface = SurfaceId::Floating(index);
        let requirement = minimum(&window.root, metrics)?;
        if window.rect.width < requirement.width || window.rect.height < requirement.height {
            return Err(RenderPlanError::FloatingTooSmall {
                index,
                requirement: SizeRequirement {
                    available_width: window.rect.width,
                    available_height: window.rect.height,
                    required_width: requirement.width,
                    required_height: requirement.height,
                },
            });
        }
        plan.floating.push(FloatingLayout {
            index,
            rect: window.rect,
            z: window.z,
            surface,
        });
        solve_node(
            &window.root,
            window.rect,
            surface,
            &NodePath::root(),
            &mut plan,
            metrics,
        )?;
    }

    append_autohide_items(layout, viewport_rect, metrics, &mut plan)?;
    let (target, reason) = focus_target(layout, &plan.autohide);
    plan.focus = FocusMetadata { target, reason };
    mark_focus(&mut plan);
    Ok(plan)
}

fn validate_metrics(metrics: LayoutMetrics) -> Result<(), RenderPlanError> {
    if metrics.min_pane_width == 0 {
        return Err(RenderPlanError::InvalidMetrics("min_pane_width"));
    }
    if metrics.min_pane_height == 0 {
        return Err(RenderPlanError::InvalidMetrics("min_pane_height"));
    }
    if metrics.tab_bar_height == 0 {
        return Err(RenderPlanError::InvalidMetrics("tab_bar_height"));
    }
    if metrics.tab_min_width == 0 {
        return Err(RenderPlanError::InvalidMetrics("tab_min_width"));
    }
    if metrics.autohide_bar_thickness == 0 {
        return Err(RenderPlanError::InvalidMetrics("autohide_bar_thickness"));
    }
    if metrics.autohide_reveal_width == 0 {
        return Err(RenderPlanError::InvalidMetrics("autohide_reveal_width"));
    }
    if metrics.autohide_reveal_height == 0 {
        return Err(RenderPlanError::InvalidMetrics("autohide_reveal_height"));
    }
    Ok(())
}

fn validate_maximized_state(layout: &DockLayout) -> Result<(), RenderPlanError> {
    let Some(snapshot) = &layout.maximized else {
        return Ok(());
    };
    let valid_current = matches!(
        &layout.root,
        DockNode::TabStack { tabs, active: 0 }
            if tabs.len() == 1 && tabs.first() == Some(&snapshot.panel)
    ) && layout.floating.is_empty()
        && layout.autohide.entries.is_empty()
        && layout.autohide.active.is_none();
    if !valid_current {
        return Err(RenderPlanError::InvalidMaximizedState(
            ValidationIssue::MaximizeSnapshotInvalid,
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MinSize {
    width: u32,
    height: u32,
}

fn minimum(node: &DockNode, metrics: LayoutMetrics) -> Result<MinSize, RenderPlanError> {
    match node {
        DockNode::Empty => Ok(MinSize::default()),
        DockNode::TabStack { tabs, .. } | DockNode::Document { tabs, .. } => {
            let tab_width = checked_mul(tabs.len() as u32, metrics.tab_min_width)?;
            Ok(MinSize {
                width: metrics.min_pane_width.max(tab_width),
                height: checked_add(metrics.min_pane_height, metrics.tab_bar_height)?,
            })
        }
        DockNode::Split { axis, children } => {
            let mut result = MinSize::default();
            let separator_total = checked_mul(
                children.len().saturating_sub(1) as u32,
                metrics.splitter_width,
            )?;
            for child in children {
                let child_min = minimum(&child.node, metrics)?;
                match axis {
                    Axis::Horizontal => {
                        result.width = checked_add(result.width, child_min.width)?;
                        result.height = result.height.max(child_min.height);
                    }
                    Axis::Vertical => {
                        result.height = checked_add(result.height, child_min.height)?;
                        result.width = result.width.max(child_min.width);
                    }
                }
            }
            match axis {
                Axis::Horizontal => result.width = checked_add(result.width, separator_total)?,
                Axis::Vertical => result.height = checked_add(result.height, separator_total)?,
            }
            Ok(result)
        }
    }
}

fn solve_node(
    node: &DockNode,
    rect: crate::Rect,
    surface: SurfaceId,
    path: &NodePath,
    plan: &mut RenderPlan,
    metrics: LayoutMetrics,
) -> Result<MinSize, RenderPlanError> {
    let required = minimum(node, metrics)?;
    if rect.width < required.width || rect.height < required.height {
        return Err(RenderPlanError::MinimumSize {
            surface,
            node: path.clone(),
            requirement: SizeRequirement {
                available_width: rect.width,
                available_height: rect.height,
                required_width: required.width,
                required_height: required.height,
            },
        });
    }
    match node {
        DockNode::Empty => Ok(required),
        DockNode::TabStack { tabs, active } => {
            let (tab_bar, content) = leaf_rects(rect, metrics);
            let tabs_rects = tab_rects(tab_bar, tabs.len())?;
            for (index, id) in tabs.iter().enumerate() {
                let is_active = index == *active;
                let tab_rect = tabs_rects[index];
                plan.panels.push(PanelLayout {
                    id: id.clone(),
                    surface,
                    node: path.clone(),
                    rect,
                    content_rect: is_active.then_some(content),
                    tab_rect,
                    active: is_active,
                    focused: false,
                    visibility: if plan.maximized.is_some() {
                        PanelVisibility::Maximized
                    } else {
                        PanelVisibility::Visible
                    },
                });
                plan.tabs.push(TabHeader {
                    id: TabId::Panel(id.clone()),
                    surface,
                    node: path.clone(),
                    rect: tab_rect,
                    active: is_active,
                    focused: false,
                });
            }
            Ok(required)
        }
        DockNode::Document { tabs, active } => {
            let (tab_bar, content) = leaf_rects(rect, metrics);
            let tabs_rects = tab_rects(tab_bar, tabs.len())?;
            for (index, id) in tabs.iter().enumerate() {
                let is_active = index == *active;
                let tab_rect = tabs_rects[index];
                plan.documents.push(DocumentLayout {
                    id: id.clone(),
                    surface,
                    node: path.clone(),
                    rect,
                    content_rect: is_active.then_some(content),
                    tab_rect,
                    active: is_active,
                    focused: false,
                });
                plan.tabs.push(TabHeader {
                    id: TabId::Document(id.clone()),
                    surface,
                    node: path.clone(),
                    rect: tab_rect,
                    active: is_active,
                    focused: false,
                });
            }
            Ok(required)
        }
        DockNode::Split { axis, children } => {
            let mins = children
                .iter()
                .map(|child| minimum(&child.node, metrics))
                .collect::<Result<Vec<_>, _>>()?;
            let (main_extent, cross_extent) = match axis {
                Axis::Horizontal => (rect.width, rect.height),
                Axis::Vertical => (rect.height, rect.width),
            };
            let separator_total = checked_mul(
                children.len().saturating_sub(1) as u32,
                metrics.splitter_width,
            )?;
            let usable = main_extent
                .checked_sub(separator_total)
                .ok_or(RenderPlanError::GeometryOverflow)?;
            let min_main = mins
                .iter()
                .map(|min| match axis {
                    Axis::Horizontal => min.width,
                    Axis::Vertical => min.height,
                })
                .collect::<Vec<_>>();
            let ratios = children.iter().map(|child| child.ratio).collect::<Vec<_>>();
            let extents = allocate_weighted(usable, &ratios, &min_main)?;
            let mut cursor = match axis {
                Axis::Horizontal => rect.x,
                Axis::Vertical => rect.y,
            };
            for (index, extent) in extents.iter().copied().enumerate() {
                let child_rect = match axis {
                    Axis::Horizontal => crate::Rect {
                        x: cursor,
                        y: rect.y,
                        width: extent,
                        height: cross_extent,
                    },
                    Axis::Vertical => crate::Rect {
                        x: rect.x,
                        y: cursor,
                        width: cross_extent,
                        height: extent,
                    },
                };
                solve_node(
                    &children[index].node,
                    child_rect,
                    surface,
                    &path.child(index),
                    plan,
                    metrics,
                )?;
                cursor = checked_add_i32(cursor, extent as i32)?;
                if index + 1 < extents.len() {
                    let line_rect = match axis {
                        Axis::Horizontal => crate::Rect {
                            x: cursor,
                            y: rect.y,
                            width: metrics.splitter_width,
                            height: rect.height,
                        },
                        Axis::Vertical => crate::Rect {
                            x: rect.x,
                            y: cursor,
                            width: rect.width,
                            height: metrics.splitter_width,
                        },
                    };
                    plan.splitters.push(SplitterLayout {
                        surface,
                        node: path.clone(),
                        axis: *axis,
                        line_rect,
                        hit_rect: inflate_and_clip(
                            line_rect,
                            rect,
                            *axis,
                            metrics.splitter_hit_slop,
                        )?,
                        before_child: index,
                        after_child: index + 1,
                    });
                    cursor = checked_add_i32(cursor, metrics.splitter_width as i32)?;
                }
            }
            Ok(required)
        }
    }
}

fn leaf_rects(rect: crate::Rect, metrics: LayoutMetrics) -> (crate::Rect, crate::Rect) {
    let header_height = metrics.tab_bar_height.min(rect.height);
    (
        crate::Rect {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: header_height,
        },
        crate::Rect {
            x: rect.x,
            y: rect.y.saturating_add(header_height as i32),
            width: rect.width,
            height: rect.height - header_height,
        },
    )
}

fn tab_rects(bar: crate::Rect, count: usize) -> Result<Vec<crate::Rect>, RenderPlanError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let widths = allocate_equal(bar.width, count)?;
    let mut result = Vec::with_capacity(count);
    let mut x = bar.x;
    for width in widths {
        result.push(crate::Rect {
            x,
            y: bar.y,
            width,
            height: bar.height,
        });
        x = checked_add_i32(x, width as i32)?;
    }
    Ok(result)
}

fn allocate_equal(total: u32, count: usize) -> Result<Vec<u32>, RenderPlanError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let count = count as u32;
    let base = total / count;
    let remainder = total % count;
    let mut result = Vec::with_capacity(count as usize);
    for index in 0..count {
        result.push(base + u32::from(index < remainder));
    }
    Ok(result)
}

fn allocate_weighted(
    total: u32,
    ratios: &[Ratio],
    mins: &[u32],
) -> Result<Vec<u32>, RenderPlanError> {
    if ratios.len() != mins.len() || ratios.is_empty() {
        return Err(RenderPlanError::GeometryOverflow);
    }
    let min_total = mins.iter().try_fold(0u32, |sum, min| sum.checked_add(*min));
    let Some(min_total) = min_total else {
        return Err(RenderPlanError::GeometryOverflow);
    };
    if min_total > total {
        return Err(RenderPlanError::MinimumSize {
            surface: SurfaceId::Main,
            node: NodePath::root(),
            requirement: SizeRequirement {
                available_width: total,
                available_height: 0,
                required_width: min_total,
                required_height: 0,
            },
        });
    }

    let mut result = vec![0; ratios.len()];
    let mut fixed = vec![false; ratios.len()];
    let mut fixed_total = 0u32;
    loop {
        let remaining_total = total - fixed_total;
        let remaining_weight = ratios
            .iter()
            .enumerate()
            .filter(|(index, _)| !fixed[*index])
            .try_fold(0u64, |sum, (_, ratio)| {
                sum.checked_add(ratio.millionths() as u64)
            });
        let Some(remaining_weight) = remaining_weight else {
            return Err(RenderPlanError::GeometryOverflow);
        };
        if remaining_weight == 0 {
            break;
        }
        let mut newly_fixed = false;
        for (index, ratio) in ratios.iter().enumerate() {
            if fixed[index] {
                continue;
            }
            let ideal =
                ((remaining_total as u64 * ratio.millionths() as u64) / remaining_weight) as u32;
            if ideal < mins[index] {
                result[index] = mins[index];
                fixed[index] = true;
                fixed_total = fixed_total
                    .checked_add(mins[index])
                    .ok_or(RenderPlanError::GeometryOverflow)?;
                newly_fixed = true;
            }
        }
        if !newly_fixed {
            let mut allocations = Vec::new();
            let mut used = 0u32;
            for (index, ratio) in ratios.iter().enumerate() {
                if fixed[index] {
                    continue;
                }
                let numerator = remaining_total as u64 * ratio.millionths() as u64;
                let base = (numerator / remaining_weight) as u32;
                let remainder = numerator % remaining_weight;
                result[index] = base;
                used = used
                    .checked_add(base)
                    .ok_or(RenderPlanError::GeometryOverflow)?;
                allocations.push((index, remainder));
            }
            let mut leftover = remaining_total - used;
            allocations.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
            let mut cursor = 0usize;
            while leftover > 0 {
                let index = allocations[cursor % allocations.len()].0;
                result[index] = result[index]
                    .checked_add(1)
                    .ok_or(RenderPlanError::GeometryOverflow)?;
                leftover -= 1;
                cursor += 1;
            }
            break;
        }
        if fixed_total > total {
            return Err(RenderPlanError::MinimumSize {
                surface: SurfaceId::Main,
                node: NodePath::root(),
                requirement: SizeRequirement {
                    available_width: total,
                    available_height: 0,
                    required_width: fixed_total,
                    required_height: 0,
                },
            });
        }
    }
    Ok(result)
}

fn solve_autohide_bars(
    layout: &DockLayout,
    viewport: crate::Rect,
    metrics: LayoutMetrics,
) -> Result<(crate::Rect, AutoHideLayout), RenderPlanError> {
    let mut left = false;
    let mut right = false;
    let mut top = false;
    let mut bottom = false;
    for entry in &layout.autohide.entries {
        match entry.side {
            Side::Left => left = true,
            Side::Right => right = true,
            Side::Top => top = true,
            Side::Bottom => bottom = true,
        }
    }
    let left_size = if left {
        metrics.autohide_bar_thickness
    } else {
        0
    };
    let right_size = if right {
        metrics.autohide_bar_thickness
    } else {
        0
    };
    let top_size = if top {
        metrics.autohide_bar_thickness
    } else {
        0
    };
    let bottom_size = if bottom {
        metrics.autohide_bar_thickness
    } else {
        0
    };
    let main_width = viewport
        .width
        .checked_sub(left_size)
        .and_then(|value| value.checked_sub(right_size))
        .ok_or(RenderPlanError::MinimumSize {
            surface: SurfaceId::Main,
            node: NodePath::root(),
            requirement: SizeRequirement {
                available_width: viewport.width,
                available_height: viewport.height,
                required_width: left_size + right_size,
                required_height: 0,
            },
        })?;
    let main_height = viewport
        .height
        .checked_sub(top_size)
        .and_then(|value| value.checked_sub(bottom_size))
        .ok_or(RenderPlanError::MinimumSize {
            surface: SurfaceId::Main,
            node: NodePath::root(),
            requirement: SizeRequirement {
                available_width: viewport.width,
                available_height: viewport.height,
                required_width: 0,
                required_height: top_size + bottom_size,
            },
        })?;
    let main_rect = crate::Rect {
        x: checked_i32(left_size)?,
        y: checked_i32(top_size)?,
        width: main_width,
        height: main_height,
    };
    let mut result = AutoHideLayout::default();
    for side in [Side::Left, Side::Right, Side::Top, Side::Bottom] {
        if !layout
            .autohide
            .entries
            .iter()
            .any(|entry| entry.side == side)
        {
            continue;
        }
        let entries = layout
            .autohide
            .entries
            .iter()
            .filter(|entry| entry.side == side)
            .collect::<Vec<_>>();
        let cross = match side {
            Side::Left | Side::Right => main_rect.height,
            Side::Top | Side::Bottom => main_rect.width,
        };
        let required = checked_mul(entries.len() as u32, metrics.autohide_bar_thickness)?;
        if required > cross {
            return Err(RenderPlanError::AutoHideBarOverflow {
                side,
                available: cross,
                required,
            });
        }
        let bar = match side {
            Side::Left => crate::Rect {
                x: 0,
                y: main_rect.y,
                width: metrics.autohide_bar_thickness,
                height: main_rect.height,
            },
            Side::Right => crate::Rect {
                x: checked_i32(viewport.width - metrics.autohide_bar_thickness)?,
                y: main_rect.y,
                width: metrics.autohide_bar_thickness,
                height: main_rect.height,
            },
            Side::Top => crate::Rect {
                x: checked_i32(left_size)?,
                y: 0,
                width: main_width,
                height: metrics.autohide_bar_thickness,
            },
            Side::Bottom => crate::Rect {
                x: checked_i32(left_size)?,
                y: checked_i32(viewport.height - metrics.autohide_bar_thickness)?,
                width: main_rect.width,
                height: metrics.autohide_bar_thickness,
            },
        };
        let lengths = allocate_equal(cross, entries.len())?;
        let mut buttons = Vec::with_capacity(entries.len());
        let mut cursor = match side {
            Side::Left | Side::Right => bar.y,
            Side::Top | Side::Bottom => bar.x,
        };
        for (entry, length) in entries.into_iter().zip(lengths) {
            let rect = match side {
                Side::Left | Side::Right => crate::Rect {
                    x: bar.x,
                    y: cursor,
                    width: bar.width,
                    height: length,
                },
                Side::Top | Side::Bottom => crate::Rect {
                    x: cursor,
                    y: bar.y,
                    width: length,
                    height: bar.height,
                },
            };
            buttons.push(AutoHideButton {
                panel: entry.panel.clone(),
                side,
                rect,
                active: layout.autohide.active.as_deref() == Some(entry.panel.as_str()),
                focused: false,
            });
            cursor = checked_add_i32(cursor, length as i32)?;
        }
        result.bars.push(AutoHideBarLayout {
            side,
            rect: bar,
            buttons,
        });
    }
    Ok((main_rect, result))
}

fn append_autohide_items(
    layout: &DockLayout,
    viewport: crate::Rect,
    metrics: LayoutMetrics,
    plan: &mut RenderPlan,
) -> Result<(), RenderPlanError> {
    let Some(active) = &layout.autohide.active else {
        for entry in &layout.autohide.entries {
            let button =
                find_button(plan, &entry.panel).ok_or(RenderPlanError::GeometryOverflow)?;
            plan.panels.push(PanelLayout {
                id: entry.panel.clone(),
                surface: SurfaceId::AutoHide(entry.side),
                node: NodePath::root(),
                rect: button.rect,
                content_rect: None,
                tab_rect: button.rect,
                active: false,
                focused: false,
                visibility: PanelVisibility::AutoHideCollapsed,
            });
        }
        return Ok(());
    };
    let Some(entry) = layout.autohide.entries.iter().find(|e| &e.panel == active) else {
        return Err(RenderPlanError::InvalidLayout(LayoutError {
            issue: ValidationIssue::InvalidAutohideActive,
            path: Vec::new(),
        }));
    };
    for hidden in &layout.autohide.entries {
        let button = find_button(plan, &hidden.panel).ok_or(RenderPlanError::GeometryOverflow)?;
        if hidden.panel != *active {
            plan.panels.push(PanelLayout {
                id: hidden.panel.clone(),
                surface: SurfaceId::AutoHide(hidden.side),
                node: NodePath::root(),
                rect: button.rect,
                content_rect: None,
                tab_rect: button.rect,
                active: false,
                focused: false,
                visibility: PanelVisibility::AutoHideCollapsed,
            });
        }
    }
    let reveal = reveal_rect(&entry.panel, entry.side, viewport, plan.main_rect, metrics)?;
    let required = MinSize {
        width: metrics.min_pane_width,
        height: checked_add(metrics.min_pane_height, metrics.tab_bar_height)?,
    };
    if reveal.width < required.width || reveal.height < required.height {
        return Err(RenderPlanError::AutoHideRevealTooSmall {
            panel: active.clone(),
            side: entry.side,
            requirement: SizeRequirement {
                available_width: reveal.width,
                available_height: reveal.height,
                required_width: required.width,
                required_height: required.height,
            },
        });
    }
    let (tab_rect, content_rect) = leaf_rects(reveal, metrics);
    plan.panels.push(PanelLayout {
        id: active.clone(),
        surface: SurfaceId::AutoHide(entry.side),
        node: NodePath::root(),
        rect: reveal,
        content_rect: Some(content_rect),
        tab_rect,
        active: true,
        focused: false,
        visibility: PanelVisibility::AutoHideRevealed,
    });
    plan.tabs.push(TabHeader {
        id: TabId::Panel(active.clone()),
        surface: SurfaceId::AutoHide(entry.side),
        node: NodePath::root(),
        rect: tab_rect,
        active: true,
        focused: false,
    });
    plan.autohide.revealed = Some(AutoHideReveal {
        panel: active.clone(),
        side: entry.side,
        rect: reveal,
        content_rect,
        tab_rect,
    });
    Ok(())
}

fn find_button<'a>(plan: &'a RenderPlan, panel: &str) -> Option<&'a AutoHideButton> {
    plan.autohide
        .bars
        .iter()
        .flat_map(|bar| bar.buttons.iter())
        .find(|button| button.panel == panel)
}

fn reveal_rect(
    panel: &str,
    side: Side,
    viewport: crate::Rect,
    main: crate::Rect,
    metrics: LayoutMetrics,
) -> Result<crate::Rect, RenderPlanError> {
    let rect = match side {
        Side::Left => crate::Rect {
            x: main.x,
            y: main.y,
            width: metrics.autohide_reveal_width,
            height: main.height,
        },
        Side::Right => crate::Rect {
            x: checked_i32(
                viewport
                    .width
                    .checked_sub(metrics.autohide_bar_thickness)
                    .and_then(|value| value.checked_sub(metrics.autohide_reveal_width))
                    .ok_or(RenderPlanError::AutoHideRevealTooSmall {
                        panel: panel.into(),
                        side,
                        requirement: SizeRequirement {
                            available_width: viewport.width,
                            available_height: viewport.height,
                            required_width: metrics.autohide_reveal_width,
                            required_height: 0,
                        },
                    })?,
            )?,
            y: main.y,
            width: metrics.autohide_reveal_width,
            height: main.height,
        },
        Side::Top => crate::Rect {
            x: main.x,
            y: main.y,
            width: main.width,
            height: metrics.autohide_reveal_height,
        },
        Side::Bottom => crate::Rect {
            x: main.x,
            y: checked_i32(
                viewport
                    .height
                    .checked_sub(metrics.autohide_bar_thickness)
                    .and_then(|value| value.checked_sub(metrics.autohide_reveal_height))
                    .ok_or(RenderPlanError::AutoHideRevealTooSmall {
                        panel: panel.into(),
                        side,
                        requirement: SizeRequirement {
                            available_width: viewport.width,
                            available_height: viewport.height,
                            required_width: 0,
                            required_height: metrics.autohide_reveal_height,
                        },
                    })?,
            )?,
            width: main.width,
            height: metrics.autohide_reveal_height,
        },
    };
    let right = i64::from(rect.x) + i64::from(rect.width);
    let bottom = i64::from(rect.y) + i64::from(rect.height);
    if rect.x < 0
        || rect.y < 0
        || right > i64::from(viewport.width)
        || bottom > i64::from(viewport.height)
    {
        return Err(RenderPlanError::AutoHideRevealTooSmall {
            panel: panel.into(),
            side,
            requirement: SizeRequirement {
                available_width: viewport.width,
                available_height: viewport.height,
                required_width: rect.width,
                required_height: rect.height,
            },
        });
    }
    Ok(rect)
}

fn focus_target(
    layout: &DockLayout,
    autohide: &AutoHideLayout,
) -> (Option<FocusTarget>, Option<FocusReason>) {
    if let Some(snapshot) = &layout.maximized {
        return (
            Some(FocusTarget::Panel(snapshot.panel.clone())),
            Some(FocusReason::Maximized),
        );
    }
    if let Some(reveal) = &autohide.revealed {
        return (
            Some(FocusTarget::Panel(reveal.panel.clone())),
            Some(FocusReason::AutoHide),
        );
    }
    if let Some((_index, window)) = layout
        .floating
        .iter()
        .enumerate()
        .max_by_key(|(_, window)| window.z)
        && let Some(target) = active_target(&window.root)
    {
        return (Some(target), Some(FocusReason::Floating));
    }
    (
        active_target(&layout.root),
        active_target(&layout.root).map(|_| FocusReason::Main),
    )
}

fn active_target(node: &DockNode) -> Option<FocusTarget> {
    match node {
        DockNode::Empty => None,
        DockNode::TabStack { tabs, active } => tabs.get(*active).cloned().map(FocusTarget::Panel),
        DockNode::Document { tabs, active } => {
            tabs.get(*active).cloned().map(FocusTarget::Document)
        }
        DockNode::Split { children, .. } => {
            children.iter().find_map(|child| active_target(&child.node))
        }
    }
}

fn mark_focus(plan: &mut RenderPlan) {
    for panel in &mut plan.panels {
        panel.focused =
            matches!(&plan.focus.target, Some(FocusTarget::Panel(id)) if id == &panel.id);
    }
    for document in &mut plan.documents {
        document.focused =
            matches!(&plan.focus.target, Some(FocusTarget::Document(id)) if id == &document.id);
    }
    for tab in &mut plan.tabs {
        tab.focused = match (&plan.focus.target, &tab.id) {
            (Some(FocusTarget::Panel(target)), TabId::Panel(id))
            | (Some(FocusTarget::Document(target)), TabId::Document(id)) => target == id,
            _ => false,
        };
    }
    for bar in &mut plan.autohide.bars {
        for button in &mut bar.buttons {
            button.focused =
                matches!(&plan.focus.target, Some(FocusTarget::Panel(id)) if id == &button.panel);
        }
    }
}

fn inflate_and_clip(
    line: crate::Rect,
    parent: crate::Rect,
    axis: Axis,
    slop: u32,
) -> Result<crate::Rect, RenderPlanError> {
    let (start, end) = match axis {
        Axis::Horizontal => (line.x as i64, line.x as i64 + line.width as i64),
        Axis::Vertical => (line.y as i64, line.y as i64 + line.height as i64),
    };
    let parent_start = match axis {
        Axis::Horizontal => parent.x as i64,
        Axis::Vertical => parent.y as i64,
    };
    let parent_end = match axis {
        Axis::Horizontal => parent.x as i64 + parent.width as i64,
        Axis::Vertical => parent.y as i64 + parent.height as i64,
    };
    let start = (start - slop as i64).max(parent_start);
    let end = (end + slop as i64).min(parent_end);
    if end < start {
        return Err(RenderPlanError::GeometryOverflow);
    }
    let mut result = line;
    match axis {
        Axis::Horizontal => {
            result.x = checked_i32_signed(start)?;
            result.width = checked_u32(end - start)?;
        }
        Axis::Vertical => {
            result.y = checked_i32_signed(start)?;
            result.height = checked_u32(end - start)?;
        }
    }
    Ok(result)
}

fn checked_add(left: u32, right: u32) -> Result<u32, RenderPlanError> {
    left.checked_add(right)
        .ok_or(RenderPlanError::GeometryOverflow)
}

fn checked_mul(left: u32, right: u32) -> Result<u32, RenderPlanError> {
    left.checked_mul(right)
        .ok_or(RenderPlanError::GeometryOverflow)
}

fn checked_i32(value: u32) -> Result<i32, RenderPlanError> {
    i32::try_from(value).map_err(|_| RenderPlanError::GeometryOverflow)
}

fn checked_i32_signed(value: i64) -> Result<i32, RenderPlanError> {
    i32::try_from(value).map_err(|_| RenderPlanError::GeometryOverflow)
}

fn checked_u32(value: i64) -> Result<u32, RenderPlanError> {
    u32::try_from(value).map_err(|_| RenderPlanError::GeometryOverflow)
}

fn checked_add_i32(left: i32, right: i32) -> Result<i32, RenderPlanError> {
    left.checked_add(right)
        .ok_or(RenderPlanError::GeometryOverflow)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::field_reassign_with_default,
        reason = "test fixtures mutate defaults to emphasize the single field under test"
    )]

    use super::*;
    use crate::{AutoHideBars, AutoHideEntry, FloatingWindow, SplitChild};
    use alloc::boxed::Box;
    use alloc::vec;

    fn split_layout() -> DockLayout {
        DockLayout {
            root: DockNode::Split {
                axis: Axis::Horizontal,
                children: vec![
                    SplitChild {
                        ratio: Ratio::from_millionths(300_000),
                        node: DockNode::panel("left"),
                    },
                    SplitChild {
                        ratio: Ratio::from_millionths(700_000),
                        node: DockNode::Split {
                            axis: Axis::Vertical,
                            children: vec![
                                SplitChild {
                                    ratio: Ratio::HALF,
                                    node: DockNode::document("doc"),
                                },
                                SplitChild {
                                    ratio: Ratio::HALF,
                                    node: DockNode::panel("right"),
                                },
                            ],
                        },
                    },
                ],
            },
            ..DockLayout::default()
        }
    }

    #[test]
    fn nested_split_is_deterministic_and_emits_all_geometry() {
        let layout = split_layout();
        let first = solve(&layout, Viewport::new(1_200, 800)).unwrap();
        let second = solve(&layout, Viewport::new(1_200, 800)).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.panels.len(), 2);
        assert_eq!(first.documents.len(), 1);
        assert_eq!(first.tabs.len(), 3);
        assert_eq!(first.splitters.len(), 2);
        assert_eq!(first.main_rect.width, 1_200);
        assert_eq!(first.focus.target, Some(FocusTarget::Panel("left".into())));
        assert!(first
            .panels
            .iter()
            .any(|panel| panel.active && panel.content_rect.is_some()));
    }

    #[test]
    fn weighted_rounding_preserves_extent_and_largest_remainder_order() {
        let layout = DockLayout {
            root: DockNode::Split {
                axis: Axis::Horizontal,
                children: vec![
                    SplitChild {
                        ratio: Ratio::from_millionths(333_333),
                        node: DockNode::panel("a"),
                    },
                    SplitChild {
                        ratio: Ratio::from_millionths(333_333),
                        node: DockNode::panel("b"),
                    },
                    SplitChild {
                        ratio: Ratio::from_millionths(333_334),
                        node: DockNode::panel("c"),
                    },
                ],
            },
            ..DockLayout::default()
        };
        let plan = solve(&layout, Viewport::new(503, 200)).unwrap();
        let widths = plan
            .panels
            .iter()
            .map(|panel| panel.rect.width)
            .collect::<Vec<_>>();
        assert_eq!(widths.iter().sum::<u32>() + 2, 503);
        assert_eq!(widths, vec![167, 167, 167]);
    }

    #[test]
    fn minimum_size_is_an_error_instead_of_ratio_clamping() {
        let layout = split_layout();
        let error = solve(&layout, Viewport::new(300, 150)).unwrap_err();
        assert!(matches!(error, RenderPlanError::MinimumSize { .. }));
    }

    #[test]
    fn invalid_ratio_is_not_silently_normalized() {
        let mut layout = DockLayout::default();
        layout.root = DockNode::Split {
            axis: Axis::Horizontal,
            children: vec![
                SplitChild {
                    ratio: Ratio::HALF,
                    node: DockNode::panel("a"),
                },
                SplitChild {
                    ratio: Ratio::from_millionths(499_999),
                    node: DockNode::panel("b"),
                },
            ],
        };
        assert!(matches!(
            solve(&layout, Viewport::new(800, 600)),
            Err(RenderPlanError::InvalidLayout(_))
        ));
    }

    #[test]
    fn floating_windows_are_separate_surfaces_and_keep_z_order() {
        let mut layout = DockLayout::default();
        layout.root = DockNode::panel("main");
        layout.floating = vec![FloatingWindow {
            rect: crate::Rect {
                x: -40,
                y: 20,
                width: 400,
                height: 300,
            },
            root: DockNode::panel("float"),
            z: 7,
        }];
        let plan = solve(&layout, Viewport::new(800, 600)).unwrap();
        assert_eq!(plan.floating[0].rect.x, -40);
        assert_eq!(
            plan.panels
                .iter()
                .find(|p| p.id == "float")
                .unwrap()
                .surface,
            SurfaceId::Floating(0)
        );
        assert_eq!(plan.focus.target, Some(FocusTarget::Panel("float".into())));
        assert_eq!(plan.focus.reason, Some(FocusReason::Floating));
    }

    #[test]
    fn autohide_reserves_bars_and_emits_collapsed_and_revealed_geometry() {
        let mut layout = DockLayout::default();
        layout.root = DockNode::panel("main");
        layout.autohide = AutoHideBars {
            entries: vec![
                AutoHideEntry {
                    side: Side::Left,
                    panel: "hidden".into(),
                },
                AutoHideEntry {
                    side: Side::Bottom,
                    panel: "revealed".into(),
                },
            ],
            active: Some("revealed".into()),
        };
        let plan = solve(&layout, Viewport::new(1_000, 700)).unwrap();
        assert_eq!(plan.main_rect.x, 28);
        assert_eq!(plan.main_rect.height, 672);
        assert!(plan.autohide.revealed.is_some());
        assert_eq!(plan.focus.reason, Some(FocusReason::AutoHide));
        assert_eq!(plan.panels.iter().filter(|p| p.id == "revealed").count(), 1);
        assert!(plan
            .panels
            .iter()
            .any(|p| p.id == "hidden" && p.content_rect.is_none()));
    }

    #[test]
    fn autohide_button_overflow_is_structured() {
        let mut layout = DockLayout::default();
        layout.root = DockNode::panel("main");
        layout.autohide.entries = (0..30)
            .map(|index| AutoHideEntry {
                side: Side::Left,
                panel: alloc::format!("p{index}"),
            })
            .collect();
        let error = solve(&layout, Viewport::new(800, 500)).unwrap_err();
        assert!(matches!(
            error,
            RenderPlanError::AutoHideBarOverflow {
                side: Side::Left,
                ..
            }
        ));
    }

    #[test]
    fn maximized_state_is_explicit_and_focuses_the_maximized_panel() {
        let mut layout = DockLayout::default();
        layout.root = DockNode::panel("max");
        layout.maximized = Some(Box::new(crate::MaximizeRestore {
            panel: "max".into(),
            root: DockNode::panel("old"),
            floating: Vec::new(),
            autohide: AutoHideBars::default(),
        }));
        let plan = solve(&layout, Viewport::new(800, 600)).unwrap();
        assert_eq!(plan.maximized.as_ref().unwrap().panel, "max");
        assert_eq!(plan.focus.reason, Some(FocusReason::Maximized));
        assert_eq!(plan.panels[0].visibility, PanelVisibility::Maximized);
    }

    #[test]
    fn inconsistent_maximize_surface_is_rejected() {
        let mut layout = DockLayout::default();
        layout.root = DockNode::panel("other");
        layout.maximized = Some(Box::new(crate::MaximizeRestore {
            panel: "max".into(),
            root: DockNode::panel("old"),
            floating: Vec::new(),
            autohide: AutoHideBars::default(),
        }));
        assert!(matches!(
            solve(&layout, Viewport::new(800, 600)),
            Err(RenderPlanError::InvalidMaximizedState(_))
        ));
    }

    #[test]
    fn split_hit_rect_is_clipped_to_parent() {
        let layout = split_layout();
        let plan = solve_with(
            &layout,
            Viewport::new(1_200, 800),
            LayoutMetrics {
                splitter_hit_slop: 1_000,
                ..LayoutMetrics::default()
            },
        )
        .unwrap();
        let splitter = &plan.splitters[0];
        assert!(splitter.hit_rect.x >= plan.main_rect.x);
        assert!(
            i64::from(splitter.hit_rect.x) + i64::from(splitter.hit_rect.width)
                <= i64::from(plan.main_rect.x) + i64::from(plan.main_rect.width)
        );
    }
}
