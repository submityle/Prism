//! Virtual shadow-map feature boundary.
//!
//! This module owns the CPU-side decision layer for Nanite-style virtual
//! shadow maps: which physical pages a light needs resident and, for
//! directional lights, how the clipmap stack selects a level and page for a
//! receiver. Physical page storage and the depth raster live in the backend.
//!
//! * [`clipmap`] — directional-light clipmap level and page selection.

pub mod clipmap;

pub use clipmap::{ClipmapConfig, MAX_CLIP_LEVELS};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ShadowPageKey {
    pub light: u32,
    pub level: u16,
    pub x: u16,
    pub y: u16,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VirtualShadowSettings {
    pub physical_pages: u32,
    pub page_size: u16,
    pub max_clip_levels: u8,
}
