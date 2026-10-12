use alloc::{boxed::Box, string::String, vec::Vec};

/// Stable panel identity.
///
/// The reducer treats IDs as opaque, case-sensitive keys. Applications should
/// use a namespaced value when panels are supplied by plugins.
pub type PanelId = String;
/// Stable document identity, separate from tool-panel identity.
pub type DocumentId = String;

/// Fixed-point share with one million units representing 100%.
///
/// Integer storage avoids platform-dependent floating-point layout decisions
/// and makes serialized layouts deterministic across platforms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ratio(u32);

impl Ratio {
    /// Number of fixed-point units representing a complete split.
    pub const SCALE: u32 = 1_000_000;
    /// A 50/50 split ratio.
    pub const HALF: Self = Self(500_000);
    /// Creates a ratio from millionths of the available space.
    pub const fn from_millionths(value: u32) -> Self {
        Self(value)
    }
    /// Returns the raw millionths value.
    pub const fn millionths(self) -> u32 {
        self.0
    }
}

/// Axis along which a split allocates child nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// Allocate children from left to right.
    Horizontal,
    /// Allocate children from top to bottom.
    Vertical,
}

/// Edge used when placing a node beside a target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    /// Place before the target on the horizontal axis.
    Left,
    /// Place after the target on the horizontal axis.
    Right,
    /// Place before the target on the vertical axis.
    Top,
    /// Place after the target on the vertical axis.
    Bottom,
}

impl Side {
    /// Returns the split axis implied by this edge.
    pub const fn axis(self) -> Axis {
        match self {
            Self::Left | Self::Right => Axis::Horizontal,
            Self::Top | Self::Bottom => Axis::Vertical,
        }
    }
    /// Returns whether this edge is ordered before the target.
    pub const fn is_before(self) -> bool {
        matches!(self, Self::Left | Self::Top)
    }
}

/// Logical rectangle used for a floating dock window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    /// Horizontal origin in logical pixels.
    pub x: i32,
    /// Vertical origin in logical pixels.
    pub y: i32,
    /// Width in logical pixels.
    pub width: u32,
    /// Height in logical pixels.
    pub height: u32,
}

/// A child of a split node and its normalized share of the parent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitChild {
    /// Share of the parent's extent.
    pub ratio: Ratio,
    /// Child layout subtree.
    pub node: DockNode,
}

/// Every pane is represented in the tree; documents cannot be tabified with tools.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DockNode {
    /// Allocates space among two or more child nodes.
    Split {
        /// Axis along which child extents are allocated.
        axis: Axis,
        /// Ordered children and their normalized extent shares.
        children: Vec<SplitChild>,
    },
    /// A leaf containing one or more tool panels.
    TabStack {
        /// Panel IDs in visual tab order.
        tabs: Vec<PanelId>,
        /// Index of the selected panel.
        active: usize,
    },
    /// A document leaf. Documents may be split beside tools but not tabified with them.
    Document {
        /// Document IDs in visual tab order.
        tabs: Vec<DocumentId>,
        /// Index of the selected document.
        active: usize,
    },
    /// An empty root placeholder used before the first panel is installed.
    Empty,
}

impl DockNode {
    /// Creates a one-panel tab stack.
    pub fn panel(id: impl Into<PanelId>) -> Self {
        Self::TabStack {
            tabs: alloc::vec![id.into()],
            active: 0,
        }
    }
    /// Creates a one-document document leaf.
    pub fn document(id: impl Into<DocumentId>) -> Self {
        Self::Document {
            tabs: alloc::vec![id.into()],
            active: 0,
        }
    }
    /// Tests whether a panel is present in this subtree.
    pub fn contains_panel(&self, id: &str) -> bool {
        match self {
            Self::TabStack { tabs, .. } => tabs.iter().any(|p| p == id),
            Self::Split { children, .. } => children.iter().any(|c| c.node.contains_panel(id)),
            _ => false,
        }
    }
}

/// A top-level floating dock window and its independent layout tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FloatingWindow {
    /// Window rectangle in logical pixels.
    pub rect: Rect,
    /// Root of the floating window's dock subtree.
    pub root: DockNode,
    /// Stable stacking order among floating windows.
    pub z: u32,
}

/// A panel collapsed into an edge auto-hide bar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutoHideEntry {
    /// Edge hosting the collapsed panel.
    pub side: Side,
    /// Panel stored in the bar.
    pub panel: PanelId,
}

/// All auto-hide entries and the currently revealed entry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AutoHideBars {
    /// Collapsed panels in deterministic insertion order.
    pub entries: Vec<AutoHideEntry>,
    /// Panel currently revealed by the auto-hide interaction, if any.
    pub active: Option<PanelId>,
}

/// Complete pure-data layout state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockLayout {
    /// Main dock tree.
    pub root: DockNode,
    /// Independent floating window trees.
    pub floating: Vec<FloatingWindow>,
    /// Collapsed edge panels.
    pub autohide: AutoHideBars,
    /// Saved state while one panel is maximized.
    pub maximized: Option<Box<MaximizeRestore>>,
}

/// Snapshot retained while a panel is maximized.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaximizeRestore {
    /// Panel that was maximized.
    pub panel: PanelId,
    /// Main tree before maximization.
    pub root: DockNode,
    /// Floating windows before maximization.
    pub floating: Vec<FloatingWindow>,
    /// Auto-hide state before maximization.
    pub autohide: AutoHideBars,
}

impl Default for DockLayout {
    fn default() -> Self {
        Self {
            root: DockNode::Empty,
            floating: Vec::new(),
            autohide: AutoHideBars::default(),
            maximized: None,
        }
    }
}
