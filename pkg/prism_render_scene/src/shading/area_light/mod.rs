//! `LTC` (Linearly Transformed Cosines, Heitz et al. 2016) area-light support
//! for the scene render world.
//!
//! The `CPU` golden lives in [`prism_render_shading::gi::area_light`]; this
//! module is the render-world plumbing that will drive polygonal area lights
//! through the clustered-lighting resolve path. Block 1 lands the
//! self-contained, non-invasive pieces that build and verify independently of
//! the resolve hot path:
//!
//! * [`abi`] — the `repr(C)` std430 twins ([`GpuAreaLight`](abi::GpuAreaLight)
//!   storage record and the [`GpuAreaLightLtcParams`](abi::GpuAreaLightLtcParams)
//!   immediate block) shared with `shaders/area_light_ltc.wesl`.
//! * [`settings`] — the opt-in [`PrismAreaLightSettings`] resource folding the
//!   golden `LTC`-`LUT` bake tunables.
//! * [`resources`] — the `CPU` bake of the `LTC` look-up table and its upload to
//!   two `Rgba32Float` textures ([`init_area_light_ltc_lut`]).
//!
//! The scene `WESL` kernel `shaders/area_light_ltc.wesl` inlines the golden
//! polygon maths (horizon clip, edge integral, form factor, `M-inverse`
//! transform); [`shader_tests`] compiles it and pins a `CPU` mirror of the
//! kernel against the golden reference.
//!
//! Wiring the baked `LUT` and an `AreaLight` storage buffer into the resolve
//! bind groups and `BRDF` accumulation is deferred to block 2; nothing here
//! touches the lighting hot path.

mod abi;
mod resources;
mod settings;

pub(crate) use resources::init_area_light_ltc_lut;
pub(crate) use settings::PrismAreaLightSettings;

#[cfg(test)]
mod shader_tests;
