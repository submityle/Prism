//! Basic, structural controls: the primitives every other family builds on.
//!
//! Everything here emits a data-only [`Element`](prism_ui::Element) and attaches
//! only kit class names (styled by this module's [`register_styles`]). Controls
//! are plain [`Component`](prism_ui_component::Component)s, so they compose and
//! test without a running runtime.

use crate::preset::StyleSheet;

pub mod async_image;
pub mod avatar;
pub mod blockquote;
pub mod button;
pub mod code;
pub mod divider;
pub mod heading;
pub mod highlight;
pub mod icon;
pub mod image;
pub mod kbd;
pub mod label;
pub mod link;
pub mod spacer;
pub mod tag;
pub mod text;
pub mod video;

pub use async_image::{AsyncImage, AsyncImageProps, LoadPhase};
pub use avatar::{Avatar, AvatarContent, AvatarProps};
pub use blockquote::{Blockquote, BlockquoteProps};
pub use button::{Button, ButtonProps};
pub use code::{Code, CodeProps};
pub use divider::{Divider, DividerOrientation, DividerProps};
pub use heading::{Heading, HeadingLevel, HeadingProps};
pub use highlight::{Highlight, HighlightProps};
pub use icon::{Icon, IconProps};
pub use image::{Image, ImageFit, ImageProps};
pub use kbd::{Kbd, KbdProps};
pub use label::{Label, LabelProps};
pub use link::{Link, LinkProps};
pub use spacer::{Spacer, SpacerProps};
pub use tag::{Tag, TagProps};
pub use text::{Text, TextProps, TextRole, TextTone};
pub use video::{MediaPlayer, Video, VideoProps};

/// Registers every `basics/` control's token-backed classes into `sheet`.
pub fn register_styles(sheet: &mut StyleSheet) {
    button::register_styles(sheet);
    text::register_styles(sheet);
    heading::register_styles(sheet);
    label::register_styles(sheet);
    link::register_styles(sheet);
    icon::register_styles(sheet);
    divider::register_styles(sheet);
    spacer::register_styles(sheet);
    avatar::register_styles(sheet);
    tag::register_styles(sheet);
    kbd::register_styles(sheet);
    code::register_styles(sheet);
    image::register_styles(sheet);
    async_image::register_styles(sheet);
    blockquote::register_styles(sheet);
    highlight::register_styles(sheet);
    video::register_styles(sheet);
}
