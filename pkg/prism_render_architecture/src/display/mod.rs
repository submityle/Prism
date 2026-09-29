//! Scene-linear to display-output pipeline metadata and negotiation.
//!
//! This module describes how scene-linear radiance is mapped to a concrete
//! display signal, on the `CPU` and independent of any `GPU` backend. It models:
//!
//! - [`DisplayOutput`] — the target signal encoding (`sRGB`, `scRGB`, `HDR10`
//!   `PQ`) and its associated [`TransferMetadata`] (transfer function, color
//!   gamut primaries, and reference/peak `nits`).
//! - [`ToneMapTarget`] — tone-mapping luminance parameters, with `NaN`-safe
//!   clamping against the selected output's capabilities.
//! - [`DisplayCapabilities`] — a sink's advertised abilities, used to negotiate
//!   a viable output and downgrade gracefully when `HDR` is unavailable.
//!
//! Color-gamut primaries and `nits` metadata are real, table-driven values.
//! Non-linear transfer curves that would require transcendental math (notably
//! the `PQ` curve) are represented as metadata with a [`CurveStatus::Pending`]
//! marker rather than being evaluated here.

mod negotiation;
mod tonemap;
mod transfer;

pub use negotiation::{DisplayCapabilities, NegotiationResult};
pub use tonemap::ToneMapTarget;
pub use transfer::{Chromaticity, ColorGamut, CurveStatus, TransferFunction, TransferMetadata};

/// Target display signal encoding for the final present pass.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DisplayOutput {
    SdrSrgb,
    ScRgb,
    Hdr10Pq,
}

/// Runtime display configuration selected for a swapchain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplaySettings {
    pub output: DisplayOutput,
    pub paper_white_nits: f32,
    pub peak_nits: f32,
    pub local_exposure: bool,
}
