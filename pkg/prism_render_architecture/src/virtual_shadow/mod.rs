//! Virtual shadow-map feature boundary.
//!
//! This module owns the CPU-side decision layer for Nanite-style virtual
//! shadow maps: which physical pages a light needs resident and, for
//! directional lights, how the clipmap stack selects a level and page for a
//! receiver. Physical page storage and the depth raster live in the backend.
//!
//! * [`clipmap`] — directional-light clipmap level and page selection.
//! * [`residency`] — page residency table and per-frame request coalescing.
//! * [`coverage`] — light-space receiver footprints to overlapped clip pages.
//! * [`light_space`] — world-space bounds projected onto the light plane.
//! * [`frame`] — per-frame world-space casters to coalesced clip-page requests.
//! * [`quality`] — screen-adaptive shadow texel-size policy.

pub mod clipmap;
pub mod coverage;
pub mod frame;
pub mod light_space;
pub mod quality;
pub mod residency;

pub use clipmap::{ClipmapConfig, MAX_CLIP_LEVELS};
pub use coverage::mark_receiver_footprint;
pub use frame::{plan_shadow_frame, ShadowCaster, ShadowFramePlan};
pub use light_space::DirectionalLightBasis;
pub use quality::{caster_for_receiver, ShadowQuality};
pub use residency::{ShadowRequestBatch, ShadowResidencyTable};

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
