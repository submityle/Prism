//! Gallery instances for the `feedback` family. Dev-only; see `gallery/mod.rs`.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::mount_component;

use super::Showcase;
use crate::feedback::{
    Alert, AlertProps, Banner, BannerProps, Callout, CalloutProps, CommandItem, CommandPalette,
    CommandPaletteProps, ContextMenu, ContextMenuItem, ContextMenuProps, Dialog, DialogProps,
    FloatButton, FloatButtonProps, HoverCard, HoverCardProps, Placement, Popconfirm,
    PopconfirmProps, Popover, PopoverProps, ProgressBar, ProgressBarProps, ResultView, ResultProps, Skeleton, SkeletonProps, Spinner, SpinnerProps, Toast, ToastProps, Tooltip,
    TooltipProps, Tour, TourProps, TourStep,
};
use crate::{ControlSize, Tone};

/// Real, named instances of every `feedback` control.
#[must_use]
pub fn instances() -> Vec<Showcase> {
    alloc::vec![
        // Alert — a titled success status surface.
        Showcase::new(
            "Alert / Success",
            mount_component(
                &Alert,
                AlertProps::new("Your changes have been saved.")
                    .title("Saved")
                    .tone(Tone::Success),
            ),
        ),
        // Banner — a dismissible accent banner.
        Showcase::new(
            "Banner / Accent",
            mount_component(
                &Banner,
                BannerProps::new("A new version is available.")
                    .tone(Tone::Accent)
                    .dismissible(true),
            ),
        ),
        // Callout — a titled accent callout with body.
        Showcase::new(
            "Callout / Accent",
            mount_component(
                &Callout,
                CalloutProps::new()
                    .title("Note")
                    .body_child(Element::text("Callout body content."))
                    .tone(Tone::Accent),
            ),
        ),
        // Dialog — an open modal with title, body and action.
        Showcase::new(
            "Dialog / Open",
            mount_component(
                &Dialog,
                DialogProps::new()
                    .open(true)
                    .title("Confirm")
                    .body_child(Element::text("Are you sure you want to continue?"))
                    .action(Element::text("Continue")),
            ),
        ),
        // Popover — an open frosted-glass overlay anchored below.
        Showcase::new(
            "Popover / Open",
            mount_component(
                &Popover,
                PopoverProps::new()
                    .open(true)
                    .anchor(Element::text("Open menu"))
                    .child(Element::text("Popover content."))
                    .placement(Placement::Bottom),
            ),
        ),
        // ProgressBar — determinate at 60%.
        Showcase::new(
            "ProgressBar / 60%",
            mount_component(
                &ProgressBar,
                ProgressBarProps::new(0.6).tone(Tone::Accent),
            ),
        ),
        // Skeleton — three shimmering placeholder lines.
        Showcase::new(
            "Skeleton / 3 Lines",
            mount_component(&Skeleton, SkeletonProps::new(3)),
        ),
        // Spinner — a large indeterminate spinner.
        Showcase::new(
            "Spinner / Large",
            mount_component(&Spinner, SpinnerProps::new().size(ControlSize::Large)),
        ),
        // Toast — a transient success toast.
        Showcase::new(
            "Toast / Success",
            mount_component(
                &Toast,
                ToastProps::new("File uploaded.").tone(Tone::Success),
            ),
        ),
        // Tooltip — a labelled hover target.
        Showcase::new(
            "Tooltip / Top",
            mount_component(
                &Tooltip,
                TooltipProps::new("More information")
                    .child(Element::text("Hover me"))
                    .placement(Placement::Top),
            ),
        ),
        // HoverCard — an open profile overlay.
        Showcase::new(
            "HoverCard / Open",
            mount_component(
                &HoverCard,
                HoverCardProps::new()
                    .anchor(Element::text("@kiro"))
                    .content(Element::text("Profile preview card."))
                    .open(true),
            ),
        ),
        // ContextMenu — an open menu with a destructive item.
        Showcase::new(
            "ContextMenu / Open",
            mount_component(
                &ContextMenu,
                ContextMenuProps::new()
                    .item(ContextMenuItem::new("Copy"))
                    .item(ContextMenuItem::new("Rename").disabled(true))
                    .item(ContextMenuItem::new("Delete").danger(true))
                    .open(true),
            ),
        ),
        // CommandPalette — an open palette with queried commands.
        Showcase::new(
            "CommandPalette / Open",
            mount_component(
                &CommandPalette,
                CommandPaletteProps::new()
                    .query("op")
                    .command(CommandItem::new("Open File").hint("Cmd O"))
                    .command(CommandItem::new("Save").hint("Cmd S"))
                    .open(true),
            ),
        ),
        // Popconfirm — an open destructive confirmation.
        Showcase::new(
            "Popconfirm / Open",
            mount_component(
                &Popconfirm,
                PopconfirmProps::new("Delete this item?")
                    .confirm_label("Delete")
                    .cancel_label("Cancel")
                    .tone(Tone::Danger)
                    .open(true),
            ),
        ),
        // Result — a success status page with an action.
        Showcase::new(
            "Result / Success",
            mount_component(
                &ResultView,
                ResultProps::new("Payment complete")
                    .description("Your order has shipped.")
                    .tone(Tone::Success)
                    .action(Element::text("Continue")),
            ),
        ),
        // Tour — an open coachmark on its first step.
        Showcase::new(
            "Tour / Step 1",
            mount_component(
                &Tour,
                TourProps::new()
                    .step(TourStep::new("Welcome", "This is the toolbar."))
                    .step(TourStep::new("Files", "Here are your files."))
                    .current(0)
                    .open(true),
            ),
        ),
        // FloatButton — an open speed-dial with one action.
        Showcase::new(
            "FloatButton / Open",
            mount_component(
                &FloatButton,
                FloatButtonProps::new()
                    .icon(Element::text("+"))
                    .action(Element::text("New"))
                    .open(true),
            ),
        ),
    ]
}
