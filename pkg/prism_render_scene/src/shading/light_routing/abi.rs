//! ABI shared between the light-routing cull pass and
//! `shaders/light_routing.wesl`.
//!
//! Two GPU structs cross the boundary, each mirroring its shader twin
//! byte-for-byte so machines with and without a GPU agree with the CPU golden
//! in [`prism_render_shading::light_routing`]:
//!
//! * [`GpuLightRouting`] — one per light, the storage-buffer record the
//!   channel/layer gates read. Mirrors the golden [`LightRouting`], flattening
//!   its `LightingChannelMask` / `LightLayerMask` newtypes to their raw bits.
//! * [`GpuLightRoutingParams`] — the single `var<immediate>` (push-constant)
//!   block the `light_routing_cull_main` entry consumes.
//!
//! The shader refines the cluster active-light bitset one 32-light *word* at a
//! time (WGSL core has no 64-bit integer), so the dispatch and the buffer sizes
//! are expressed in [`word_count`] words rather than raw lights.

use bytemuck::{Pod, Zeroable};
use prism_render_shading::LightRouting;

use super::settings::PrismLightRoutingSettings;

/// Lighting channels in a mask, mirroring the golden
/// `prism_render_shading::MAX_LIGHTING_CHANNELS` (not re-exported from the
/// crate root; guarded against drift by [`tests`]).
///
/// Only the drift-guard test consumes this locally (the runtime path packs
/// channel bits through the golden newtypes), so it is `cfg(test)`-scoped to
/// stay a live, warning-free guard.
#[cfg(test)]
pub(crate) const MAX_LIGHTING_CHANNELS: u32 = 8;

/// NPR light layers in a mask, mirroring the golden
/// `prism_render_shading::MAX_LIGHT_LAYERS` and the shader's `MAX_LIGHT_LAYERS`
/// (guarded against drift by [`tests`]).
pub(crate) const MAX_LIGHT_LAYERS: u32 = 4;

/// Lights folded per cluster word. WGSL core has no 64-bit integer, so the
/// shader refines the active-light bitset one `u32` word at a time; this is the
/// bit width of that word.
pub(crate) const LIGHTS_PER_WORD: u32 = 32;

/// Workgroup size (x) of the `light_routing_cull_main` compute entry point.
///
/// Must match `@workgroup_size(N, 1, 1)` in `light_routing.wesl`; the dispatch
/// rounds [`word_count`] up to a multiple of this and the shader bounds-checks
/// every invocation against `word_count`.
pub(crate) const LIGHT_ROUTING_WORKGROUP_SIZE: u32 = 64;

/// Number of 32-light cluster words spanning `light_count` lights.
///
/// The shader keys `visible_lights` on this and lays `layer_lights` out
/// `word_count * MAX_LIGHT_LAYERS` entries deep, so the buffer sizes and the
/// dispatch extent all derive from it.
pub(crate) const fn word_count(light_count: u32) -> u32 {
    light_count.div_ceil(LIGHTS_PER_WORD)
}

/// One routing record per light: the storage-buffer twin of the shader's
/// `LightRouting` struct and the golden [`LightRouting`].
///
/// Two `u32` bitmasks — the lighting-channel visibility gate and the NPR
/// light-layer routing mask — 8 bytes with no implicit padding (`u32` array
/// stride).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub(crate) struct GpuLightRouting {
    /// Lighting-channel bitmask (golden `LightRouting::channels`): the light
    /// affects a primitive iff their channel masks intersect.
    pub channels: u32,
    /// NPR light-layer bitmask (golden `LightRouting::layers`): the light feeds
    /// every stylized accumulation bin whose bit is set.
    pub layers: u32,
}

impl From<LightRouting> for GpuLightRouting {
    fn from(routing: LightRouting) -> Self {
        Self {
            channels: routing.channels.bits(),
            layers: routing.layers.bits(),
        }
    }
}

/// Immediate (push-constant) block consumed by the `light_routing_cull_main`
/// entry point.
///
/// Mirrors the shader's `LightRoutingParams`: the valid record count, the
/// receiving primitive's channel mask, the cluster-word count and an explicit
/// pad — 16 bytes, a multiple of 16 with no implicit padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub(crate) struct GpuLightRoutingParams {
    /// Number of valid routing records in the `routing_records` buffer; the
    /// shader fails records at/above this index safe (never lit).
    pub light_count: u32,
    /// The receiving primitive's lighting-channel bitmask.
    pub primitive_channels: u32,
    /// Number of 32-light cluster words = [`word_count`]`(light_count)`.
    pub word_count: u32,
    /// Explicit tail pad so the block is a multiple of 16 bytes, matching the
    /// shader struct's `_pad`.
    pub _pad: u32,
}

impl GpuLightRoutingParams {
    /// Builds the immediate block from the live [`PrismLightRoutingSettings`].
    ///
    /// `light_count` is the real record count so the shader bounds-checks
    /// exactly the records the buffer carries; `word_count` follows
    /// [`word_count`].
    pub(crate) fn from_settings(settings: &PrismLightRoutingSettings) -> Self {
        let light_count = settings.records.len() as u32;
        Self {
            light_count,
            primitive_channels: settings.primitive_channels.bits(),
            word_count: word_count(light_count),
            _pad: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_shading::{LightLayerMask, LightingChannelMask};

    #[test]
    fn gpu_light_routing_is_the_eight_byte_record() {
        // Two `u32` bitmasks, no implicit padding.
        assert_eq!(size_of::<GpuLightRouting>(), 8);
        assert_eq!(align_of::<GpuLightRouting>(), 4);
    }

    #[test]
    fn gpu_params_is_the_16_byte_immediate_block() {
        assert_eq!(size_of::<GpuLightRoutingParams>(), 16);
        assert_eq!(align_of::<GpuLightRoutingParams>(), 4);
    }

    #[test]
    fn workgroup_and_layer_constants_match_the_golden_and_shader() {
        // `MAX_LIGHT_LAYERS` is the width of the golden `LightLayerMask::all`.
        assert_eq!(LightLayerMask::all().bits(), (1u32 << MAX_LIGHT_LAYERS) - 1);
        // `MAX_LIGHTING_CHANNELS` is the width of the golden channel window: a
        // saturated `from_bits` keeps exactly that many low bits.
        assert_eq!(
            LightingChannelMask::from_bits(u32::MAX).bits(),
            (1u32 << MAX_LIGHTING_CHANNELS) - 1
        );
        assert_eq!(LIGHTS_PER_WORD, 32);
        assert_eq!(LIGHT_ROUTING_WORKGROUP_SIZE, 64);
    }

    #[test]
    fn word_count_rounds_up_per_32_lights() {
        assert_eq!(word_count(0), 0);
        assert_eq!(word_count(1), 1);
        assert_eq!(word_count(32), 1);
        assert_eq!(word_count(33), 2);
        assert_eq!(word_count(64), 2);
        assert_eq!(word_count(65), 3);
    }

    #[test]
    fn gpu_record_flattens_the_golden_masks() {
        let routing = LightRouting {
            channels: LightingChannelMask::from_index(1).with_channel(3),
            layers: LightLayerMask::from_index(2),
        };
        let gpu = GpuLightRouting::from(routing);
        assert_eq!(gpu.channels, routing.channels.bits());
        assert_eq!(gpu.layers, routing.layers.bits());
        assert_eq!(gpu.channels, 0b1010);
        assert_eq!(gpu.layers, 0b100);
    }

    #[test]
    fn params_pack_the_record_count_and_primitive_mask() {
        let mut settings = PrismLightRoutingSettings::default();
        settings.primitive_channels = LightingChannelMask::from_index(2);
        settings.records = vec![LightRouting::default(); 40];
        let params = GpuLightRoutingParams::from_settings(&settings);
        assert_eq!(params.light_count, 40);
        assert_eq!(params.word_count, 2);
        assert_eq!(params.primitive_channels, 0b100);
        assert_eq!(params._pad, 0);
    }
}
