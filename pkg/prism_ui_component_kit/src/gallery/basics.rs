//! Gallery instances for the `basics` family. Dev-only; see `gallery/mod.rs`.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::mount_component;

use super::Showcase;
use crate::basics::{
    AsyncImage, AsyncImageProps, Avatar, AvatarProps, Blockquote, BlockquoteProps, Button,
    ButtonProps, Code, CodeProps, Divider, DividerOrientation, DividerProps, Heading, HeadingLevel,
    HeadingProps, Highlight, HighlightProps, Icon, IconProps, Image, ImageFit, ImageProps, Kbd,
    KbdProps, Label, LabelProps, Link, LinkProps, LoadPhase, Tag, TagProps, Text, TextProps,
    TextRole, TextTone, Video, VideoProps,
};
use crate::{ButtonVariant, ControlSize, Tone};

/// Real, named instances of every `basics` control.
#[must_use]
pub fn instances() -> Vec<Showcase> {
    alloc::vec![
        // Button — a few key variants.
        Showcase::new(
            "Button / Filled",
            mount_component(&Button, ButtonProps::new("Save").variant(ButtonVariant::Filled)),
        ),
        Showcase::new(
            "Button / Tinted",
            mount_component(&Button, ButtonProps::new("Edit").variant(ButtonVariant::Tinted)),
        ),
        Showcase::new(
            "Button / Plain",
            mount_component(&Button, ButtonProps::new("Cancel").variant(ButtonVariant::Plain)),
        ),
        // Text — a couple of roles/tones.
        Showcase::new(
            "Text / Body",
            mount_component(&Text, TextProps::new("The quick brown fox.")),
        ),
        Showcase::new(
            "Text / Headline",
            mount_component(
                &Text,
                TextProps::new("Headline").role(TextRole::Headline).tone(TextTone::Secondary),
            ),
        ),
        // Heading — two levels.
        Showcase::new(
            "Heading / L1",
            mount_component(&Heading, HeadingProps::new("Large Title")),
        ),
        Showcase::new(
            "Heading / L3",
            mount_component(
                &Heading,
                HeadingProps::new("Title 2").level(HeadingLevel::L3),
            ),
        ),
        // Label.
        Showcase::new(
            "Label / Required",
            mount_component(&Label, LabelProps::new("Email").required(true)),
        ),
        // Link.
        Showcase::new(
            "Link / Inline",
            mount_component(&Link, LinkProps::new("Documentation", "/docs")),
        ),
        // Icon.
        Showcase::new(
            "Icon / Large",
            mount_component(&Icon, IconProps::new("gear").size(ControlSize::Large)),
        ),
        // Divider — both orientations.
        Showcase::new(
            "Divider / Horizontal",
            mount_component(&Divider, DividerProps::new()),
        ),
        Showcase::new(
            "Divider / Vertical",
            mount_component(
                &Divider,
                DividerProps::new().orientation(DividerOrientation::Vertical),
            ),
        ),
        // Avatar — initials.
        Showcase::new(
            "Avatar / Initials",
            mount_component(
                &Avatar,
                AvatarProps::new().initials("WK").size(ControlSize::Large),
            ),
        ),
        // Tag — accent + removable success.
        Showcase::new(
            "Tag / Accent",
            mount_component(&Tag, TagProps::new("New").tone(Tone::Accent)),
        ),
        Showcase::new(
            "Tag / Success",
            mount_component(
                &Tag,
                TagProps::new("Done").tone(Tone::Success).removable(true),
            ),
        ),
        // Highlight.
        Showcase::new(
            "Highlight / Warning",
            mount_component(
                &Highlight,
                HighlightProps::new("caution").tone(Tone::Warning),
            ),
        ),
        // Kbd.
        Showcase::new(
            "Kbd / Enter",
            mount_component(&Kbd, KbdProps::new("Enter")),
        ),
        // Code — inline + block.
        Showcase::new(
            "Code / Inline",
            mount_component(&Code, CodeProps::new("let x = 1;")),
        ),
        Showcase::new(
            "Code / Block",
            mount_component(&Code, CodeProps::new("fn main() {}").block(true)),
        ),
        // Image.
        Showcase::new(
            "Image / Contain",
            mount_component(
                &Image,
                ImageProps::new("photo.png").alt("A photo").fit(ImageFit::Contain),
            ),
        ),
        // AsyncImage — loaded phase paints the resolved src.
        Showcase::new(
            "AsyncImage / Loaded",
            mount_component(
                &AsyncImage,
                AsyncImageProps::new("avatar.png")
                    .alt("Avatar")
                    .phase(LoadPhase::Loaded),
            ),
        ),
        // Blockquote.
        Showcase::new(
            "Blockquote / Cited",
            mount_component(
                &Blockquote,
                BlockquoteProps::new()
                    .child(Element::text("Design is how it works."))
                    .cite("— Steve Jobs"),
            ),
        ),
        // Video (MediaPlayer) — mid-playback with controls.
        Showcase::new(
            "Video / Playing",
            mount_component(
                &Video,
                VideoProps::new()
                    .source("clip.mp4")
                    .playing(true)
                    .current(30.0)
                    .duration(120.0),
            ),
        ),
    ]
}
