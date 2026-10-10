//! Gallery instances for the `display` family. Dev-only; see `gallery/mod.rs`.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::mount_component;

use super::Showcase;
use crate::display::{
    Badge, BadgeProps, Calendar, CalendarProps, Carousel, CarouselProps, DataColumn, DataGrid,
    DataGridProps, Descriptions, DescriptionsProps, EmptyState, EmptyStateProps, ImageItem,
    ImageList, ImageListProps, List, ListProps, ListRow, ListRowProps, NumberTicker,
    NumberTickerProps, Rating, RatingProps, Section, SectionProps, Stat, StatProps, Table,
    TableProps, Timeline, TimelineItem, TimelineProps, TreeNode, TreeView, TreeViewProps,
};
use crate::pickers::Date;
use crate::Tone;

/// Real, named instances of every `display` control.
#[must_use]
pub fn instances() -> Vec<Showcase> {
    alloc::vec![
        Showcase::new(
            "Badge / Count",
            mount_component(&Badge, BadgeProps::count(3).tone(Tone::Danger)),
        ),
        Showcase::new(
            "Badge / Dot",
            mount_component(&Badge, BadgeProps::dot().tone(Tone::Success)),
        ),
        Showcase::new(
            "Stat / KPI",
            mount_component(
                &Stat,
                StatProps::new("Revenue", "$1.2M").delta("+12%", Tone::Success),
            ),
        ),
        Showcase::new(
            "Section",
            mount_component(
                &Section,
                SectionProps::new()
                    .header(Element::text("Overview"))
                    .child(Element::text("Body content lives here.")),
            ),
        ),
        Showcase::new(
            "EmptyState",
            mount_component(
                &EmptyState,
                EmptyStateProps::new("No results")
                    .icon(Element::text("📭"))
                    .description("Try adjusting your filters.")
                    .action(Element::text("Reset")),
            ),
        ),
        Showcase::new(
            "List",
            mount_component(
                &List,
                ListProps::new().children([
                    mount_component(
                        &ListRow,
                        ListRowProps::new()
                            .leading(Element::text("•"))
                            .content(Element::text("First item"))
                            .trailing(Element::text("›")),
                    ),
                    mount_component(
                        &ListRow,
                        ListRowProps::new()
                            .leading(Element::text("•"))
                            .content(Element::text("Second item"))
                            .trailing(Element::text("›")),
                    ),
                ]),
            ),
        ),
        Showcase::new(
            "ListRow",
            mount_component(
                &ListRow,
                ListRowProps::new()
                    .leading(Element::text("★"))
                    .content(Element::text("Standalone row"))
                    .trailing(Element::text("›")),
            ),
        ),
        Showcase::new(
            "Timeline",
            mount_component(
                &Timeline,
                TimelineProps::new().items([
                    TimelineItem::new(Element::text("Created")).tone(Tone::Neutral),
                    TimelineItem::new(Element::text("Deployed")).tone(Tone::Success),
                    TimelineItem::new(Element::text("Failed")).tone(Tone::Danger),
                ]),
            ),
        ),
        Showcase::new(
            "Rating",
            mount_component(&Rating, RatingProps::new(3.0).max(5)),
        ),
        Showcase::new(
            "Descriptions",
            mount_component(
                &Descriptions,
                DescriptionsProps::new()
                    .row("Name", "Ada Lovelace")
                    .row("Role", "Engineer")
                    .row("Status", "Active"),
            ),
        ),
        Showcase::new(
            "Table",
            mount_component(
                &Table,
                TableProps::new()
                    .columns(["Name", "Email", "Role"])
                    .row(["Ada", "ada@x.io", "Lead"])
                    .row(["Grace", "grace@x.io", "Eng"])
                    .striped(true)
                    .bordered(true),
            ),
        ),
        Showcase::new(
            "DataGrid",
            mount_component(
                &DataGrid,
                DataGridProps::new()
                    .columns([
                        DataColumn::new("Name").sortable(true),
                        DataColumn::new("Status"),
                    ])
                    .row(["Ada", "Active"])
                    .row(["Grace", "Away"])
                    .selectable(true),
            ),
        ),
        Showcase::new(
            "TreeView",
            mount_component(
                &TreeView,
                TreeViewProps::new().root(
                    TreeNode::new("src")
                        .expanded(true)
                        .child(TreeNode::new("main.rs"))
                        .child(TreeNode::new("lib.rs")),
                ),
            ),
        ),
        Showcase::new(
            "Calendar",
            mount_component(
                &Calendar,
                CalendarProps::new(2026, 10)
                    .selected(Date::new(2026, 10, 10))
                    .today(Date::new(2026, 10, 10)),
            ),
        ),
        Showcase::new(
            "Carousel",
            mount_component(
                &Carousel,
                CarouselProps::new()
                    .items([
                        Element::box_().child(Element::text("Slide 1")),
                        Element::box_().child(Element::text("Slide 2")),
                        Element::box_().child(Element::text("Slide 3")),
                    ])
                    .active(0)
                    .show_indicators(true),
            ),
        ),
        Showcase::new(
            "ImageList",
            mount_component(
                &ImageList,
                ImageListProps::new()
                    .items([
                        ImageItem::new("a.png").span(2),
                        ImageItem::new("b.png"),
                        ImageItem::new("c.png"),
                    ])
                    .columns(3),
            ),
        ),
        Showcase::new(
            "NumberTicker",
            mount_component(
                &NumberTicker,
                NumberTickerProps::new(1234.5)
                    .precision(1)
                    .prefix("$")
                    .suffix("k"),
            ),
        ),
    ]
}
