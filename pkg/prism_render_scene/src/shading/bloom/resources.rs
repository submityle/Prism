//! Per-view bloom pyramid textures.
//!
//! Bloom needs three sets of transient targets, all `Rgba16Float` (matching
//! `scene_color`):
//!
//! 1. `base` — a full-res copy of `scene_color` the combine reads from, so the
//!    final pass never reads and writes `scene_color` in one dispatch (the
//!    same read/write-aliasing avoidance the SSR composite uses).
//! 2. `down` — the downsample mip pyramid. `down[0]` is the half-res
//!    Karis-prefiltered mip; each subsequent level is an energy-preserving COD
//!    13-tap downsample of the previous.
//! 3. `up` — the progressive upsample chain accumulating the halo from coarse
//!    to fine. `up[i]` (for `i < levels - 1`) holds `down[i] +
//!    tent(coarser)`; the coarsest level reuses `down[last]` directly, so `up`
//!    only needs `levels - 1` targets and `up[0]` is the final accumulated
//!    bloom the combine samples.
//!
//! All targets are rebuilt every frame from the current framebuffer extent (the
//! `TextureCache` recycles the allocations), and are gated on `enable_bloom`
//! so a disabled frame allocates nothing.

use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{TextureDescriptor, TextureDimension, TextureUsages},
    renderer::RenderDevice,
    texture::{CachedTexture, TextureCache},
};

use super::super::resources::{ViewVisibilityBuffer, SCENE_COLOR_FORMAT};
use super::super::runtime::PrismShadingSettings;
use super::abi::{bloom_mip_sizes, BloomMipChain};

/// The bloom pyramid targets for one view. Present only when `enable_bloom` is
/// set and the view has a resident `scene_color`; rebuilt every frame from the
/// current extent.
#[derive(Component)]
pub(crate) struct ViewBloomTextures {
    /// Full-res copy of `scene_color`, read as the combine base.
    base: CachedTexture,
    /// Downsample mip pyramid; `down[0]` is the half-res prefiltered mip.
    down: Vec<CachedTexture>,
    /// Progressive upsample accumulation; `up[0]` is the final bloom. Length
    /// `levels - 1` (the coarsest upsample reads `down.last()` directly).
    up: Vec<CachedTexture>,
    /// Mip extents (full-res base + half-res-and-down chain) the pass node
    /// derives every dispatch's source/destination extent from.
    chain: BloomMipChain,
}

impl ViewBloomTextures {
    /// The full-res base copy target (combine input / copy destination).
    pub(crate) fn base(&self) -> &CachedTexture {
        &self.base
    }

    /// The downsample mip pyramid, `down[0]` being the half-res prefiltered mip.
    pub(crate) fn down(&self) -> &[CachedTexture] {
        &self.down
    }

    /// The upsample accumulation chain, `up[0]` being the final bloom.
    pub(crate) fn up(&self) -> &[CachedTexture] {
        &self.up
    }

    /// The mip extents driving the dispatch grid and sampling scales.
    pub(crate) fn chain(&self) -> &BloomMipChain {
        &self.chain
    }
}

/// A single `Rgba16Float` bloom target usable as both a sampled input and a
/// storage output, sized to `size`.
fn make_target(
    device: &RenderDevice,
    cache: &mut TextureCache,
    label: &'static str,
    size: UVec2,
) -> CachedTexture {
    cache.get(
        device,
        TextureDescriptor {
            label: Some(label),
            size: size.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: SCENE_COLOR_FORMAT,
            // STORAGE_BINDING: written by the copy/downsample/upsample passes.
            // TEXTURE_BINDING: sampled by the subsequent pass in the chain.
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    )
}

/// Allocates (or recycles) the bloom pyramid for every view with a resident
/// `scene_color`, gated on `enable_bloom`. A disabled frame — or an extent too
/// small to build even one mip — removes any stale targets so nothing is bound.
pub(crate) fn prepare_bloom_textures(
    mut commands: Commands,
    settings: Res<PrismShadingSettings>,
    device: Res<RenderDevice>,
    mut cache: ResMut<TextureCache>,
    views: Query<(Entity, &ExtractedCamera), With<ViewVisibilityBuffer>>,
) {
    for (entity, camera) in &views {
        let Some(size) = camera.physical_viewport_size else {
            commands.entity(entity).remove::<ViewBloomTextures>();
            continue;
        };
        let chain = bloom_mip_sizes(size);
        if !settings.enable_bloom || chain.level_count() == 0 {
            commands.entity(entity).remove::<ViewBloomTextures>();
            continue;
        }

        let base = make_target(&device, &mut cache, "prism bloom base", chain.full);
        let down: Vec<CachedTexture> = chain
            .mips
            .iter()
            .map(|&s| make_target(&device, &mut cache, "prism bloom down", s))
            .collect();
        // Upsample accumulation targets for every level but the coarsest.
        let up: Vec<CachedTexture> = chain.mips[..chain.mips.len() - 1]
            .iter()
            .map(|&s| make_target(&device, &mut cache, "prism bloom up", s))
            .collect();

        commands.entity(entity).insert(ViewBloomTextures {
            base,
            down,
            up,
            chain,
        });
    }
}
