//! `containers/` controls — the layout and surface family. See the kit design
//! doc, section 5.
//!
//! Each control emits a data-only [`Element`](prism_ui::Element) carrying only
//! kit class names; this module's [`register_styles`] owns the token-backed
//! [`Class`](prism_ui_style::Class) rules for those names. Controls stay plain
//! [`Component`](prism_ui_component::Component)s so they compose and unit-test
//! without a running runtime.
//!
//! This family ships the structural containers: [`Card`], [`Stack`], [`Grid`],
//! [`AspectRatio`], [`ScrollView`], [`Accordion`], [`Tabs`] (with [`TabList`]
//! and [`TabPanel`]), [`Sheet`] and [`Panel`]. (`Divider` lives in `basics/`.)

use crate::preset::StyleSheet;

pub mod accordion;
pub mod aspect_ratio;
pub mod card;
pub mod grid;
pub mod panel;
pub mod scroll_view;
pub mod sheet;
pub mod stack;
pub mod tabs;
pub mod infinite_scroll;
pub mod scroll_area;
pub mod split_view;

pub use accordion::{Accordion, AccordionItem, AccordionProps};
pub use aspect_ratio::{AspectRatio, AspectRatioProps};
pub use card::{Card, CardProps};
pub use grid::{Grid, GridProps};
pub use panel::{Panel, PanelProps};
pub use scroll_view::{ScrollView, ScrollViewProps};
pub use sheet::{Sheet, SheetProps, SheetSide};
pub use stack::{Stack, StackAlign, StackDirection, StackProps};
pub use tabs::{TabItem, TabList, TabListProps, TabPanel, TabPanelProps, Tabs, TabsProps};
pub use infinite_scroll::{InfiniteScroll, InfiniteScrollProps, LoadMore};
pub use scroll_area::{ScrollArea, ScrollAreaProps};
pub use split_view::{SplitOrientation, SplitView, SplitViewProps};

/// Registers every `containers/` control's token-backed classes into `sheet`.
pub fn register_styles(sheet: &mut StyleSheet) {
    card::register_styles(sheet);
    stack::register_styles(sheet);
    grid::register_styles(sheet);
    aspect_ratio::register_styles(sheet);
    scroll_view::register_styles(sheet);
    accordion::register_styles(sheet);
    tabs::register_styles(sheet);
    sheet::register_styles(sheet);
    panel::register_styles(sheet);
    scroll_area::register_styles(sheet);
    split_view::register_styles(sheet);
    infinite_scroll::register_styles(sheet);
}
