//! `display/` controls — the simpler presentational family. See the kit design
//! doc, section 5.4.
//!
//! Each control emits a data-only [`Element`](prism_ui::Element) carrying only
//! kit class names; this module's [`register_styles`] owns the token-backed
//! [`Class`](prism_ui_style::Class) rules for those names. Controls stay plain
//! [`Component`](prism_ui_component::Component)s so they compose and unit-test
//! without a running runtime.
//!
//! This family currently ships the lightweight presentational controls:
//! [`Badge`], [`Stat`], [`Section`], [`EmptyState`], [`List`]/[`ListRow`],
//! [`Timeline`], [`Rating`] and [`Descriptions`]. The data-heavy controls
//! (`Table`/`DataGrid`/`TreeView` and virtualized variants) land in a later pass.

use crate::preset::StyleSheet;

pub mod badge;
pub mod calendar;
pub mod descriptions;
pub mod empty_state;
pub mod list;
pub mod rating;
pub mod section;
pub mod stat;
pub mod timeline;
pub mod carousel;
pub mod data_grid;
pub mod image_list;
pub mod number_ticker;
pub mod table;
pub mod tree_view;

pub use badge::{Badge, BadgeProps};
pub use calendar::{Calendar, CalendarProps};
pub use descriptions::{Descriptions, DescriptionsProps};
pub use empty_state::{EmptyState, EmptyStateProps};
pub use list::{List, ListProps, ListRow, ListRowProps};
pub use rating::{Rating, RatingProps};
pub use section::{Section, SectionProps};
pub use stat::{Kpi, Stat, StatProps};
pub use timeline::{Timeline, TimelineItem, TimelineProps};
pub use carousel::{Carousel, CarouselProps};
pub use data_grid::{DataColumn, DataGrid, DataGridProps};
pub use image_list::{ImageItem, ImageList, ImageListProps, Masonry};
pub use number_ticker::{NumberTicker, NumberTickerProps};
pub use table::{Table, TableProps};
pub use tree_view::{TreeNode, TreeView, TreeViewProps};

/// Registers every `display/` control's token-backed classes into `sheet`.
pub fn register_styles(sheet: &mut StyleSheet) {
    badge::register_styles(sheet);
    calendar::register_styles(sheet);
    stat::register_styles(sheet);
    section::register_styles(sheet);
    empty_state::register_styles(sheet);
    list::register_styles(sheet);
    timeline::register_styles(sheet);
    rating::register_styles(sheet);
    descriptions::register_styles(sheet);
    table::register_styles(sheet);
    data_grid::register_styles(sheet);
    tree_view::register_styles(sheet);
    carousel::register_styles(sheet);
    image_list::register_styles(sheet);
    number_ticker::register_styles(sheet);
}
