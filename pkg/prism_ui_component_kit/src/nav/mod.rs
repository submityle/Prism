//! `nav/` controls — navigation chrome and wayfinding. See the kit design doc,
//! section 5.
//!
//! Each control emits a data-only [`Element`](prism_ui::Element) carrying only
//! kit class names; this module's [`register_styles`] owns the token-backed
//! [`Class`](prism_ui_style::Class) rules for those names. Controls stay plain
//! [`Component`](prism_ui_component::Component)s so they compose and unit-test
//! without a running runtime.
//!
//! This family ships the navigation controls: [`NavBar`], [`TabBar`],
//! [`Toolbar`], [`Breadcrumb`], [`Pagination`], [`Drawer`], [`Steps`],
//! [`Menu`] and [`SegmentedControl`].

use crate::preset::StyleSheet;

pub mod breadcrumb;
pub mod drawer;
pub mod menu;
pub mod nav_bar;
pub mod pagination;
pub mod segmented;
pub mod steps;
pub mod tab_bar;
pub mod toolbar;
pub mod affix;
pub mod anchor;
pub mod back_top;
pub mod sidebar;
pub mod status_bar;

pub use breadcrumb::{Breadcrumb, BreadcrumbItem, BreadcrumbProps};
pub use drawer::{Drawer, DrawerProps, DrawerSide};
pub use menu::{Menu, MenuEntry, MenuProps};
pub use nav_bar::{NavBar, NavBarProps};
pub use pagination::{Pagination, PaginationProps};
pub use segmented::{SegmentedControl, SegmentedControlProps};
pub use steps::{StepItem, Steps, StepsProps};
pub use tab_bar::{TabBar, TabBarItem, TabBarProps};
pub use toolbar::{Toolbar, ToolbarProps};
pub use affix::{Affix, AffixProps};
pub use anchor::{Anchor, AnchorLink, AnchorProps, ScrollSpy};
pub use back_top::{BackTop, BackTopProps};
pub use sidebar::{NavRail, Sidebar, SidebarItem, SidebarProps};
pub use status_bar::{StatusBar, StatusBarProps};

/// Registers every `nav/` control's token-backed classes into `sheet`.
pub fn register_styles(sheet: &mut StyleSheet) {
    nav_bar::register_styles(sheet);
    tab_bar::register_styles(sheet);
    toolbar::register_styles(sheet);
    breadcrumb::register_styles(sheet);
    pagination::register_styles(sheet);
    drawer::register_styles(sheet);
    steps::register_styles(sheet);
    menu::register_styles(sheet);
    segmented::register_styles(sheet);
    status_bar::register_styles(sheet);
    sidebar::register_styles(sheet);
    anchor::register_styles(sheet);
    affix::register_styles(sheet);
    back_top::register_styles(sheet);
}
