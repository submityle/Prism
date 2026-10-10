//! Gallery instances for the `containers` family. Dev-only; see `gallery/mod.rs`.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::mount_component;

use super::Showcase;
use crate::containers::{
    Accordion, AccordionItem, AccordionProps, AspectRatio, AspectRatioProps, Card, CardProps, Grid,
    GridProps, InfiniteScroll, InfiniteScrollProps, Panel, PanelProps, ScrollArea, ScrollAreaProps,
    ScrollView, ScrollViewProps, Sheet, SheetProps, SheetSide, SplitOrientation, SplitView,
    SplitViewProps, Stack, StackAlign, StackDirection, StackProps, TabItem, TabList, TabListProps,
    TabPanel, TabPanelProps, Tabs, TabsProps,
};
use crate::ControlSize;

/// Real, named instances of every `containers` control.
#[must_use]
pub fn instances() -> Vec<Showcase> {
    alloc::vec![
        // Card — header/body/footer surface.
        Showcase::new(
            "Card / Header + Footer",
            mount_component(
                &Card,
                CardProps::new()
                    .header(Element::text("Account"))
                    .child(Element::text("Manage your profile and billing."))
                    .footer(Element::text("Updated just now")),
            ),
        ),
        // Panel — titled grouping surface.
        Showcase::new(
            "Panel / Titled",
            mount_component(
                &Panel,
                PanelProps::new()
                    .title("Details")
                    .child(Element::text("Panel body content.")),
            ),
        ),
        // Stack — a horizontal, centered row.
        Showcase::new(
            "Stack / Horizontal",
            mount_component(
                &Stack,
                StackProps::new()
                    .direction(StackDirection::Horizontal)
                    .align(StackAlign::Center)
                    .gap(ControlSize::Medium)
                    .child(Element::text("One"))
                    .child(Element::text("Two"))
                    .child(Element::text("Three")),
            ),
        ),
        // Grid — two columns.
        Showcase::new(
            "Grid / 2 Columns",
            mount_component(
                &Grid,
                GridProps::new()
                    .columns(2)
                    .gap(ControlSize::Small)
                    .child(Element::text("Cell A"))
                    .child(Element::text("Cell B"))
                    .child(Element::text("Cell C"))
                    .child(Element::text("Cell D")),
            ),
        ),
        // ScrollView — vertical scroll region.
        Showcase::new(
            "ScrollView / Vertical",
            mount_component(
                &ScrollView,
                ScrollViewProps::new()
                    .child(Element::text("Line 1"))
                    .child(Element::text("Line 2"))
                    .child(Element::text("Line 3")),
            ),
        ),
        // ScrollArea — scroll region with a visible scrollbar.
        Showcase::new(
            "ScrollArea / Scrollbar",
            mount_component(
                &ScrollArea,
                ScrollAreaProps::new()
                    .show_scrollbar(true)
                    .child(Element::text("Scroll area body content.")),
            ),
        ),
        // SplitView — resizable two-pane layout.
        Showcase::new(
            "SplitView / Horizontal",
            mount_component(
                &SplitView,
                SplitViewProps::new()
                    .orientation(SplitOrientation::Horizontal)
                    .ratio(0.4)
                    .resizable(true)
                    .primary(Element::text("Sidebar"))
                    .secondary(Element::text("Content")),
            ),
        ),
        // Accordion — one expanded disclosure item.
        Showcase::new(
            "Accordion / Expanded",
            mount_component(
                &Accordion,
                AccordionProps::new()
                    .item(
                        AccordionItem::new(
                            Element::text("Section One"),
                            Element::text("First section body."),
                        )
                        .open(true),
                    )
                    .item(AccordionItem::new(
                        Element::text("Section Two"),
                        Element::text("Second section body."),
                    )),
            ),
        ),
        // TabList — the row of tab labels.
        Showcase::new(
            "TabList / Row",
            mount_component(
                &TabList,
                TabListProps::new()
                    .labels([
                        Element::text("Overview"),
                        Element::text("Details"),
                        Element::text("History"),
                    ])
                    .selected(0),
            ),
        ),
        // TabPanel — a single tab's content region.
        Showcase::new(
            "TabPanel / Content",
            mount_component(
                &TabPanel,
                TabPanelProps::new().content(Element::text("Panel content.")),
            ),
        ),
        // Tabs — labels plus panels, first selected.
        Showcase::new(
            "Tabs / Selected",
            mount_component(
                &Tabs,
                TabsProps::new()
                    .tab(TabItem::new(
                        Element::text("Overview"),
                        Element::text("Overview content."),
                    ))
                    .tab(TabItem::new(
                        Element::text("Details"),
                        Element::text("Details content."),
                    ))
                    .selected(0),
            ),
        ),
        // AspectRatio — a 16:9 framed child.
        Showcase::new(
            "AspectRatio / 16:9",
            mount_component(
                &AspectRatio,
                AspectRatioProps::new(16.0 / 9.0).child(Element::text("16:9")),
            ),
        ),
        // InfiniteScroll — loaded rows with more pending.
        Showcase::new(
            "InfiniteScroll / Loading",
            mount_component(
                &InfiniteScroll,
                InfiniteScrollProps::new()
                    .item(Element::text("Row 1"))
                    .item(Element::text("Row 2"))
                    .item(Element::text("Row 3"))
                    .loading(true)
                    .has_more(true),
            ),
        ),
        // Sheet — an open right-anchored sheet.
        Showcase::new(
            "Sheet / Open Right",
            mount_component(
                &Sheet,
                SheetProps::new()
                    .open(true)
                    .side(SheetSide::Right)
                    .child(Element::text("Sheet content.")),
            ),
        ),
    ]
}
