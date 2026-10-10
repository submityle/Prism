//! Gallery instances for the `nav` family. Dev-only; see `gallery/mod.rs`.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::mount_component;

use super::Showcase;
use crate::nav::{
    Affix, AffixProps, Anchor, AnchorLink, AnchorProps, BackTop, BackTopProps, Breadcrumb,
    BreadcrumbItem, BreadcrumbProps, Drawer, DrawerProps, DrawerSide, Menu, MenuEntry, MenuProps,
    NavBar, NavBarProps, Pagination, PaginationProps, SegmentedControl, SegmentedControlProps,
    Sidebar, SidebarItem, SidebarProps, StatusBar, StatusBarProps, StepItem, Steps, StepsProps,
    TabBar, TabBarItem, TabBarProps, Toolbar, ToolbarProps,
};

/// Real, named instances of every `nav` control.
#[must_use]
pub fn instances() -> Vec<Showcase> {
    alloc::vec![
        Showcase::new(
            "NavBar / Title",
            mount_component(
                &NavBar,
                NavBarProps::new()
                    .leading(Element::text("<"))
                    .title("Loom")
                    .trailing_item(Element::text("Share"))
                    .glass(true),
            ),
        ),
        Showcase::new(
            "TabBar / Selected",
            mount_component(
                &TabBar,
                TabBarProps::new()
                    .items([
                        TabBarItem::new("Home"),
                        TabBarItem::new("Search"),
                        TabBarItem::new("Profile"),
                    ])
                    .selected(0),
            ),
        ),
        Showcase::new(
            "Toolbar / Glass",
            mount_component(
                &Toolbar,
                ToolbarProps::new()
                    .items([
                        Element::text("Cut"),
                        Element::text("Copy"),
                        Element::text("Paste"),
                    ])
                    .glass(true),
            ),
        ),
        Showcase::new(
            "Breadcrumb / Trail",
            mount_component(
                &Breadcrumb,
                BreadcrumbProps::new().items([
                    BreadcrumbItem::new("Home"),
                    BreadcrumbItem::new("Library"),
                    BreadcrumbItem::new("Nav"),
                ]),
            ),
        ),
        Showcase::new(
            "Pagination / Page 2",
            mount_component(&Pagination, PaginationProps::new(10).current(2)),
        ),
        Showcase::new(
            "Drawer / Open",
            mount_component(
                &Drawer,
                DrawerProps::new()
                    .open(true)
                    .side(DrawerSide::Left)
                    .child(Element::text("Drawer content")),
            ),
        ),
        Showcase::new(
            "Steps / Current",
            mount_component(
                &Steps,
                StepsProps::new()
                    .steps([
                        StepItem::new("Account").done(true),
                        StepItem::new("Profile"),
                        StepItem::new("Done"),
                    ])
                    .current(1),
            ),
        ),
        Showcase::new(
            "Menu / Items",
            mount_component(
                &Menu,
                MenuProps::new().items([
                    MenuEntry::new("New"),
                    MenuEntry::new("Open"),
                    MenuEntry::new("Save").disabled(true),
                ]),
            ),
        ),
        Showcase::new(
            "SegmentedControl / Selected",
            mount_component(
                &SegmentedControl,
                SegmentedControlProps::new()
                    .segments(["Day", "Week", "Month"])
                    .selected(1),
            ),
        ),
        Showcase::new(
            "StatusBar / Message",
            mount_component(
                &StatusBar,
                StatusBarProps::new()
                    .leading_item(Element::text("main"))
                    .message("Ready")
                    .trailing_item(Element::text("UTF-8")),
            ),
        ),
        Showcase::new(
            "Sidebar / Selected",
            mount_component(
                &Sidebar,
                SidebarProps::new()
                    .items([
                        SidebarItem::new("Inbox").selected(true),
                        SidebarItem::new("Drafts"),
                        SidebarItem::new("Sent"),
                    ])
                    .collapsed(false),
            ),
        ),
        Showcase::new(
            "Anchor / Links",
            mount_component(
                &Anchor,
                AnchorProps::new().links([
                    AnchorLink::new("Intro", "#intro").active(true),
                    AnchorLink::new("Usage", "#usage"),
                ]),
            ),
        ),
        Showcase::new(
            "Affix / Pinned",
            mount_component(
                &Affix,
                AffixProps::new()
                    .pinned(true)
                    .offset(16.0)
                    .child(Element::text("Pinned toolbar")),
            ),
        ),
        Showcase::new(
            "BackTop / Visible",
            mount_component(
                &BackTop,
                BackTopProps::new().visible(true).icon(Element::text("^")),
            ),
        ),
    ]
}
