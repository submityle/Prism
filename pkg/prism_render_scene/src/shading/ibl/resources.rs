//! Global device resource backing the split-sum environment BRDF ("DFG") table.
//!
//! Unlike the per-view GTAO textures, the DFG table is view-independent: it
//! encodes `(scale, bias)` as a function of `n_dot_v` and `roughness` only, so
//! a single square `Rg16Float` texture is allocated once at `RenderStartup` and
//! reused for every view and every frame.  The compute pass writes it through a
//! storage view; the shading resolve (a following slice) samples it through the
//! same texture with a linear clamp-to-edge sampler.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{
        AddressMode, Extent3d, FilterMode, Sampler, SamplerDescriptor, Texture, TextureDescriptor,
        TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
    },
    renderer::RenderDevice,
};

/// Texel format of the DFG table.  Two 16-bit float channels hold the split-sum
/// `(scale, bias)` pair, matching `texture_storage_2d<rg16float, write>` in
/// `shaders/brdf_lut.wesl` and the CPU golden `DfgLut`.
pub(crate) const DFG_LUT_FORMAT: TextureFormat = TextureFormat::Rg16Float;

/// Side length of the square DFG table.  128 texels is the standard resolution
/// for the environment BRDF: the integrand is smooth in both axes, so a small
/// table sampled bilinearly reproduces it well within the error of the GGX
/// importance-sample count.
pub(crate) const DFG_LUT_RESOLUTION: u32 = 128;

/// The global DFG lookup-table texture plus the sampler the resolve stage reads
/// it through.  Allocated once and kept resident for the lifetime of the app.
#[derive(Resource)]
pub(crate) struct DfgLutTexture {
    /// The `Rg16Float` storage/sampled texture.  Retained so the resource owns
    /// the GPU allocation for as long as it lives.
    #[expect(dead_code, reason = "owns the GPU allocation the views borrow from")]
    texture: Texture,
    /// Default 2D view, bound both as the compute pass's write-only storage
    /// target and as the resolve stage's sampled input.
    view: TextureView,
    /// Linear clamp-to-edge sampler used by the resolve stage to read the
    /// table; taps are clamped so grazing `n_dot_v` never wraps.
    sampler: Sampler,
    /// Side length in texels, exposed so the dispatch can size its workgroups.
    pub(crate) resolution: u32,
}

impl DfgLutTexture {
    /// Allocates the DFG table texture, its default view, and the sampling
    /// sampler at [`DFG_LUT_RESOLUTION`].
    fn create(device: &RenderDevice) -> Self {
        let resolution = DFG_LUT_RESOLUTION;
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("prism DFG LUT"),
            size: Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: DFG_LUT_FORMAT,
            // STORAGE_BINDING: the compute pass writes the table.
            // TEXTURE_BINDING: the resolve stage samples it.
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor {
            label: Some("prism DFG LUT view"),
            ..Default::default()
        });
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("prism DFG LUT sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        Self {
            texture,
            view,
            sampler,
            resolution,
        }
    }

    /// The default view, used as the compute write target and the resolve
    /// sampled input.
    pub(crate) fn view(&self) -> &TextureView {
        &self.view
    }

    /// The linear clamp-to-edge sampler the resolve stage reads the table with.
    #[expect(dead_code, reason = "consumed by the resolve slice that follows")]
    pub(crate) fn sampler(&self) -> &Sampler {
        &self.sampler
    }
}

/// `RenderStartup` initializer that allocates the global [`DfgLutTexture`].
pub(crate) fn init_dfg_lut_texture(mut commands: Commands, device: Res<RenderDevice>) {
    commands.insert_resource(DfgLutTexture::create(&device));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dfg_lut_format_matches_the_shader_binding() {
        // `(scale, bias)` is two float channels; Rg16Float is the storage
        // format both the golden and brdf_lut.wesl agree on.
        assert_eq!(DFG_LUT_FORMAT, TextureFormat::Rg16Float);
    }

    #[test]
    fn dfg_lut_resolution_is_a_multiple_of_the_workgroup_size() {
        // A resolution divisible by the 8x8 workgroup keeps the dispatch exact,
        // though the shader also bounds-checks each invocation.
        assert_eq!(DFG_LUT_RESOLUTION % super::super::abi::BRDF_LUT_WORKGROUP_SIZE, 0);
    }
}
