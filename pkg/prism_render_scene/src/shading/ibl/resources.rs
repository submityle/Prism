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
        AddressMode, Extent3d, FilterMode, MipmapFilterMode, Sampler, SamplerDescriptor, Texture,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
        TextureViewDescriptor, TextureViewDimension,
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
    pub(crate) fn sampler(&self) -> &Sampler {
        &self.sampler
    }
}

/// `RenderStartup` initializer that allocates the global [`DfgLutTexture`].
pub(crate) fn init_dfg_lut_texture(mut commands: Commands, device: Res<RenderDevice>) {
    commands.insert_resource(DfgLutTexture::create(&device));
}

/// Texel format of the prefiltered radiance cube.  Four 16-bit float channels
/// (`rgba16float`) hold the convolved radiance, matching
/// `texture_storage_2d_array<rgba16float, write>` in `shaders/env_prefilter.wesl`
/// and the CPU golden `PrefilteredEnvMap`.
pub(crate) const PREFILTERED_ENV_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Edge length of the prefiltered cube's base (roughness `0`) mip.  128 texels
/// matches the golden's default and is ample for glossy reflections, since
/// rougher mips are progressively coarser and blurrier.
pub(crate) const PREFILTERED_ENV_BASE_RESOLUTION: u32 = 128;

/// Number of roughness mips in the prefiltered cube: 128, 64, 32, 16, 8.  Five
/// levels span perceptual roughness `0..=1` finely enough that the trilinear
/// roughness-to-LOD mapping in the resolve reproduces the golden without
/// visible banding between levels.
pub(crate) const PREFILTERED_ENV_MIP_COUNT: u32 = 5;

/// The global prefiltered radiance cube-map plus the sampler the resolve stage
/// reads it through.  Allocated once at `RenderStartup`: the source environment
/// is convolved into it by the prefilter pass whenever the active probe
/// changes, but the target allocation itself is fixed-size and reused.
#[derive(Resource)]
pub(crate) struct PrefilteredEnvironmentMap {
    /// The `rgba16float` cube texture (six faces, [`PREFILTERED_ENV_MIP_COUNT`]
    /// mips).  Retained so the resource owns the GPU allocation the views
    /// borrow from.
    #[expect(dead_code, reason = "owns the GPU allocation the views borrow from")]
    texture: Texture,
    /// Cube-dimension view spanning every mip, bound by the resolve stage and
    /// sampled at a roughness-derived LOD along the reflection vector.
    cube_view: TextureView,
    /// One `D2Array` write view per mip (all six faces), bound as the compute
    /// pass's storage target when convolving that roughness level.
    mip_write_views: Vec<TextureView>,
    /// Trilinear clamp-to-edge sampler the resolve reads the cube with; the mip
    /// filter blends adjacent roughness levels.
    sampler: Sampler,
    /// Base mip edge length in texels.
    base_resolution: u32,
    /// Number of roughness mips.
    mip_count: u32,
}

impl PrefilteredEnvironmentMap {
    /// Allocates the prefiltered cube, its per-mip storage views, the cube
    /// sampling view, and the trilinear sampler.
    fn create(device: &RenderDevice) -> Self {
        let base_resolution = PREFILTERED_ENV_BASE_RESOLUTION;
        let mip_count = PREFILTERED_ENV_MIP_COUNT;
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("prism prefiltered env map"),
            size: Extent3d {
                width: base_resolution,
                height: base_resolution,
                depth_or_array_layers: 6,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: PREFILTERED_ENV_FORMAT,
            // STORAGE_BINDING: the prefilter pass writes each mip.
            // TEXTURE_BINDING: the resolve stage samples the cube.
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let cube_view = texture.create_view(&TextureViewDescriptor {
            label: Some("prism prefiltered env map cube view"),
            dimension: Some(TextureViewDimension::Cube),
            base_mip_level: 0,
            mip_level_count: Some(mip_count),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        });
        let mip_write_views = (0..mip_count)
            .map(|mip| {
                texture.create_view(&TextureViewDescriptor {
                    label: Some("prism prefiltered env map mip write view"),
                    dimension: Some(TextureViewDimension::D2Array),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                    ..Default::default()
                })
            })
            .collect();
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("prism prefiltered env map sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        });
        Self {
            texture,
            cube_view,
            mip_write_views,
            sampler,
            base_resolution,
            mip_count,
        }
    }

    /// The cube-dimension sampling view spanning every mip, bound by the
    /// resolve stage.
    pub(crate) fn cube_view(&self) -> &TextureView {
        &self.cube_view
    }

    /// The trilinear clamp-to-edge sampler the resolve reads the cube with.
    pub(crate) fn sampler(&self) -> &Sampler {
        &self.sampler
    }

    /// The per-mip `D2Array` storage write view for `mip`, or [`None`] when the
    /// index is out of range.
    pub(crate) fn mip_write_view(&self, mip: u32) -> Option<&TextureView> {
        self.mip_write_views.get(mip as usize)
    }

    /// Number of roughness mips.
    pub(crate) fn mip_count(&self) -> u32 {
        self.mip_count
    }

    /// Edge length in texels of `mip`, halving per level down to a floor of one.
    pub(crate) fn mip_size(&self, mip: u32) -> u32 {
        (self.base_resolution >> mip).max(1)
    }
}

/// `RenderStartup` initializer that allocates the global
/// [`PrefilteredEnvironmentMap`].
pub(crate) fn init_prefiltered_env_map(mut commands: Commands, device: Res<RenderDevice>) {
    commands.insert_resource(PrefilteredEnvironmentMap::create(&device));
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
    fn prefiltered_env_format_is_rgba16float() {
        // Four float channels of prefiltered radiance, matching the golden and
        // env_prefilter.wesl's storage target.
        assert_eq!(PREFILTERED_ENV_FORMAT, TextureFormat::Rgba16Float);
    }

    #[test]
    fn prefiltered_env_base_resolution_is_a_multiple_of_the_workgroup_size() {
        assert_eq!(
            PREFILTERED_ENV_BASE_RESOLUTION % super::super::abi::ENV_PREFILTER_WORKGROUP_SIZE,
            0
        );
    }

    #[test]
    fn prefiltered_env_mip_chain_halves_down_to_eight() {
        // 128, 64, 32, 16, 8 across five mips, each still divisible by the 8x8
        // workgroup so the per-mip dispatch stays exact.
        assert_eq!(PREFILTERED_ENV_MIP_COUNT, 5);
        let mut size = PREFILTERED_ENV_BASE_RESOLUTION;
        for _ in 0..PREFILTERED_ENV_MIP_COUNT {
            assert_eq!(size % super::super::abi::ENV_PREFILTER_WORKGROUP_SIZE, 0);
            size >>= 1;
        }
        // The smallest mip (index 4) is 8 texels.
        assert_eq!(
            PREFILTERED_ENV_BASE_RESOLUTION >> (PREFILTERED_ENV_MIP_COUNT - 1),
            8
        );
    }

    #[test]
    fn dfg_lut_resolution_is_a_multiple_of_the_workgroup_size() {
        // A resolution divisible by the 8x8 workgroup keeps the dispatch exact,
        // though the shader also bounds-checks each invocation.
        assert_eq!(
            DFG_LUT_RESOLUTION % super::super::abi::BRDF_LUT_WORKGROUP_SIZE,
            0
        );
    }
}
