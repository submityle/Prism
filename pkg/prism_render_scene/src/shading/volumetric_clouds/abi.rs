//! ABI shared between the volumetric-cloud compute passes and the sibling
//! `WESL` shader `shaders/volumetric_clouds.wesl`.
//!
//! Two kinds of host record live here, both laid out byte-for-byte against
//! their on-device counterparts so a plain `bytemuck` cast can move data across
//! the boundary without a per-field marshal:
//!
//! * The eight per-dispatch **immediate param blocks** (`var<immediate>` push
//!   constants in the shader — one per `@compute` entry), mirrored as
//!   `#[repr(C)]` records. Every field is a 4-byte scalar (`u32` / `f32`), so
//!   the packed `#[repr(C)]` layout equals the `WGSL` push-constant layout
//!   (`align == 4`, no trailing round-up) exactly.
//! * The **resident buffer element** mirrors — one representative host type per
//!   persistent device resource the [`prism_render_architecture::volumetric::gpu`]
//!   scheduler sizes (the baked density cache, the weather map, the low-res
//!   ray-march target, the reprojection history, the light-space `AVSM`
//!   cloud-shadow map, and the multiple-scatter `LUT`).
//!
//! The `size_of` contract tests below pin each immediate block to its own field
//! count and each resident element to the golden stride constant exported by
//! [`prism_render_architecture::volumetric::gpu::buffers`], so a drift between
//! the host allocation, the shader `struct` and the golden buffer sizing fails
//! the build rather than corrupting a dispatch at run time.
//!
//! `WESL`/`WGSL` layout rules mirrored here:
//!
//! * A `var<immediate>` push-constant block of all-scalar fields packs at
//!   4-byte offsets with no 16-byte round-up, so the `#[repr(C)]` mirror packs
//!   identically.
//! * A `var<storage>` / storage-texture element uses its `std430` size, which
//!   for these packed scalar / `rgba16f` / `RGBA8` records equals the packed
//!   `#[repr(C)]` size.

use bytemuck::{Pod, Zeroable};

/// Voxel-brick tile edge of every per-voxel volumetric compute entry point.
/// Must match every `@workgroup_size(4, 4, 4)` in `volumetric_clouds.wesl`
/// (`volumetric_noise_bake`, `volumetric_modeling`,
/// `volumetric_multiscatter_lut_bake`) and the `4x4x4` brick the architecture
/// crate's `gpu::kernels` contract launches those 3D dispatches with.
#[cfg(test)]
pub(crate) const VC_VOXEL_BRICK: u32 = 4;

/// Planar tile edge of every per-texel / per-pixel volumetric compute entry
/// point. Must match every `@workgroup_size(8, 8, 1)` in
/// `volumetric_clouds.wesl` (`volumetric_weather_advect`,
/// `volumetric_raymarch`, `volumetric_scatter_resolve`,
/// `volumetric_shadow_march`, `volumetric_upsample`) and the `8x8x1` tile the
/// architecture crate's `gpu::kernels` contract launches those 2D dispatches
/// with.
#[cfg(test)]
pub(crate) const VC_SCREEN_TILE: u32 = 8;

// ===========================================================================
// Resident buffer element mirrors (sized by
// `prism_render_architecture::volumetric::gpu::buffers`)
// ===========================================================================

/// One baked density-cache voxel: a single packed scalar density (`4` bytes)
/// the modelling pass writes and the ray-march samples trilinearly. Byte-sized
/// against the golden `DENSITY_VOXEL_STRIDE`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuDensityVoxel {
    /// Packed density scalar (base shape composed with detail erosion).
    pub density: f32,
}

/// One weather-map texel: coverage, cloud-type, precipitation and wetness as an
/// `RGBA8` quad (`4` bytes), advected in place by the semi-Lagrangian pass.
/// Byte-sized against the golden `WEATHER_TEXEL_STRIDE`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWeatherTexel {
    /// Cloud coverage `[0, 255]` in the red channel.
    pub coverage: u8,
    /// Cloud type `[0, 255]` (stratus..cumulonimbus) in the green channel.
    pub cloud_type: u8,
    /// Precipitation intensity `[0, 255]` in the blue channel.
    pub precipitation: u8,
    /// Ground wetness `[0, 255]` in the alpha channel.
    pub wetness: u8,
}

/// One low-resolution ray-march tile sample: scattering `RGB` plus
/// transmittance as an `rgba16f` quad (`8` bytes, stored as four packed
/// half-float lanes). Byte-sized against the golden `RAYMARCH_TILE_STRIDE`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuRaymarchTile {
    /// Scattering `RGB` and transmittance packed as four `f16` bit patterns.
    pub scatter_transmittance: [u16; 4],
}

/// One full-resolution reprojection-history pixel: resolved scattering `RGB`
/// plus transmittance as an `rgba16f` quad (`8` bytes). Byte-sized against the
/// golden `HISTORY_PIXEL_STRIDE`; double-buffered so the temporal upsample
/// reads the previous frame while writing the next.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuHistoryPixel {
    /// Resolved scattering `RGB` and transmittance packed as four `f16` bit
    /// patterns.
    pub scatter_transmittance: [u16; 4],
}

/// One node of the light-space `AVSM` deep-shadow curve: a `(depth,
/// transmittance)` pair (`8` bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuAvsmNode {
    /// Light-space depth of the curve knot.
    pub depth: f32,
    /// Transmittance at that depth (monotonically non-increasing along the
    /// curve).
    pub transmittance: f32,
}

/// One light-space cloud-shadow texel: the fixed four-node `AVSM` deep-shadow
/// curve (`32` bytes) the shared virtual shadow map and god-ray injection
/// consume. Byte-sized against the golden `SHADOW_TEXEL_STRIDE`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuShadowTexel {
    /// The four compressed `(depth, transmittance)` knots of the curve.
    pub nodes: [GpuAvsmNode; 4],
}

/// One multiple-scatter `LUT` cell: pre-integrated scattering `RGB` plus an
/// energy-normalisation term as an `rgba16f` quad (`8` bytes). Byte-sized
/// against the golden `MULTISCATTER_CELL_STRIDE`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuMultiscatterCell {
    /// Pre-integrated scattering `RGB` and the energy term packed as four `f16`
    /// bit patterns.
    pub scatter_energy: [u16; 4],
}

// ===========================================================================
// Per-dispatch immediate param blocks (push constants in
// `volumetric_clouds.wesl`)
// ===========================================================================

/// Push constants for `volumetric_weather_advect`. Byte-compatible with
/// `VcWeatherAdvectParams` (five 4-byte scalars = `20` bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWeatherAdvectParams {
    /// Weather-map width in texels.
    pub width: u32,
    /// Weather-map height in texels.
    pub height: u32,
    /// Wind velocity, x component (texels/step) used to back-trace.
    pub wind_x: f32,
    /// Wind velocity, y component (texels/step).
    pub wind_y: f32,
    /// Advection time step (`s`).
    pub dt: f32,
}

/// Push constants for `volumetric_noise_bake`. Byte-compatible with
/// `VcNoiseBakeParams` (six 4-byte scalars = `24` bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuNoiseBakeParams {
    /// Density-cache dimension, x (voxels).
    pub dim_x: u32,
    /// Density-cache dimension, y (voxels).
    pub dim_y: u32,
    /// Density-cache dimension, z (voxels).
    pub dim_z: u32,
    /// Base Perlin-Worley shape frequency.
    pub base_freq: f32,
    /// Detail-erosion noise frequency.
    pub detail_freq: f32,
    /// Deterministic noise seed.
    pub seed: u32,
}

/// Push constants for `volumetric_modeling`. Byte-compatible with
/// `VcModelingParams` (seven 4-byte scalars = `28` bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuModelingParams {
    /// Density-cache dimension, x (voxels).
    pub dim_x: u32,
    /// Density-cache dimension, y (voxels).
    pub dim_y: u32,
    /// Density-cache dimension, z (voxels).
    pub dim_z: u32,
    /// Global coverage remap threshold `[0, 1]`.
    pub coverage: f32,
    /// Cloud-type interpolant `[0, 1]` (stratus..cumulonimbus).
    pub cloud_type: f32,
    /// Detail-erosion strength `[0, 1]`.
    pub erosion_strength: f32,
    /// Cloud-kind discriminant driving the height-gradient profile.
    pub kind: u32,
}

/// Push constants for `volumetric_multiscatter_lut_bake`. Byte-compatible with
/// `VcMsLutParams` (seven 4-byte scalars = `28` bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuMsLutParams {
    /// `LUT` cosine-axis resolution (outer).
    pub dim_cos: u32,
    /// `LUT` optical-depth-axis resolution (middle).
    pub dim_depth: u32,
    /// `LUT` albedo-axis resolution (inner).
    pub dim_albedo: u32,
    /// Per-octave attenuation factor `a`.
    pub attenuation: f32,
    /// Per-octave contribution factor `b`.
    pub contribution: f32,
    /// Henyey-Greenstein eccentricity `g` for the octave sum.
    pub eccentricity: f32,
    /// Number of scattering octaves summed per cell.
    pub octave_count: u32,
}

/// Push constants for `volumetric_raymarch`. Byte-compatible with
/// `VcRaymarchParams` (fifteen 4-byte scalars = `60` bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuRaymarchParams {
    /// Full-resolution screen width (pixels), for the low-res tile mapping.
    pub screen_w: u32,
    /// Full-resolution screen height (pixels).
    pub screen_h: u32,
    /// Density-cache grid dimension, x (voxels).
    pub grid_x: u32,
    /// Density-cache grid dimension, y (voxels).
    pub grid_y: u32,
    /// Density-cache grid dimension, z (voxels).
    pub grid_z: u32,
    /// Nominal march step outside cloud (world units).
    pub base_step: f32,
    /// Maximum adaptive step in empty space.
    pub max_step: f32,
    /// Minimum adaptive step inside dense cloud.
    pub min_step: f32,
    /// Density below which a sample is treated as empty space.
    pub density_threshold: f32,
    /// Transmittance below which the march early-outs.
    pub transmittance_cutoff: f32,
    /// Hard cap on the number of march iterations.
    pub max_steps: u32,
    /// Extinction coefficient `sigma_t`.
    pub sigma_t: f32,
    /// Single-scatter albedo `[0, 1]`.
    pub albedo: f32,
    /// Henyey-Greenstein phase eccentricity `g`.
    pub phase_g: f32,
    /// Nubis `powder` dark-edge intensity `[0, 1]`; `0` disables the term.
    pub powder_strength: f32,
}

/// Push constants for `volumetric_scatter_resolve`. Byte-compatible with
/// `VcScatterResolveParams` (twelve 4-byte scalars = `48` bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuScatterResolveParams {
    /// Full-resolution screen width (pixels).
    pub screen_w: u32,
    /// Full-resolution screen height (pixels).
    pub screen_h: u32,
    /// Multi-scatter `LUT` cosine-axis resolution.
    pub lut_cos: u32,
    /// Multi-scatter `LUT` optical-depth-axis resolution.
    pub lut_depth: u32,
    /// Multi-scatter `LUT` albedo-axis resolution.
    pub lut_albedo: u32,
    /// Forward phase lobe eccentricity `g_f`.
    pub forward_g: f32,
    /// Backward phase lobe eccentricity `g_b`.
    pub backward_g: f32,
    /// Dual-lobe blend weight `[0, 1]`.
    pub lobe_blend: f32,
    /// Single-scatter albedo `[0, 1]`.
    pub albedo: f32,
    /// View-to-light cosine `cos(theta)` for the phase evaluation.
    pub cos_theta: f32,
    /// `Draine` forward-peak shape parameter `alpha >= 0`; `0` reduces the
    /// forward lobe to `HG` and the phase to the pure dual-lobe `HG`.
    pub draine_alpha: f32,
    /// Forward-lobe `Draine` mix weight `[0, 1]`; `1` selects the sharp
    /// pure-`Draine` peak, `0` the softer pure-`HG` forward lobe.
    pub draine_weight: f32,
}

/// Push constants for `volumetric_shadow_march`. Byte-compatible with
/// `VcShadowMarchParams` (seven 4-byte scalars = `28` bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuShadowMarchParams {
    /// Light-space shadow-map width in texels.
    pub shadow_w: u32,
    /// Light-space shadow-map height in texels.
    pub shadow_h: u32,
    /// Density-cache grid dimension, x (voxels).
    pub grid_x: u32,
    /// Density-cache grid dimension, y (voxels).
    pub grid_y: u32,
    /// Density-cache grid dimension, z (voxels).
    pub grid_z: u32,
    /// Depth-slice march step toward the light (world units).
    pub step: f32,
    /// Density-to-extinction scale for the accumulated optical depth.
    pub density_scale: f32,
}

/// Push constants for `volumetric_upsample`. Byte-compatible with
/// `VcUpsampleParams` (seven 4-byte scalars = `28` bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuUpsampleParams {
    /// Full-resolution output width (pixels).
    pub screen_w: u32,
    /// Full-resolution output height (pixels).
    pub screen_h: u32,
    /// Low-resolution ray-march source width (pixels).
    pub lowres_w: u32,
    /// Low-resolution ray-march source height (pixels).
    pub lowres_h: u32,
    /// Frame index driving the temporal reprojection sample rotation.
    pub frame_index: u32,
    /// Reconstruction mode discriminant (full re-solve vs. history clamp).
    pub mode: u32,
    /// Variance-clip gamma for the history neighbourhood clamp.
    pub variance_gamma: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::volumetric::gpu::buffers::{
        DENSITY_VOXEL_STRIDE, HISTORY_PIXEL_STRIDE, MULTISCATTER_CELL_STRIDE, RAYMARCH_TILE_STRIDE,
        SHADOW_TEXEL_STRIDE, WEATHER_TEXEL_STRIDE,
    };
    use prism_render_architecture::volumetric::gpu::kernels::VolumetricKernel;

    // --- resident buffer element strides ---------------------------------

    /// The baked density-cache voxel equals the golden scalar-density stride.
    #[test]
    fn density_voxel_matches_golden_stride() {
        assert_eq!(size_of::<GpuDensityVoxel>() as u32, DENSITY_VOXEL_STRIDE);
    }

    /// The weather-map texel equals the golden `RGBA8` stride.
    #[test]
    fn weather_texel_matches_golden_stride() {
        assert_eq!(size_of::<GpuWeatherTexel>() as u32, WEATHER_TEXEL_STRIDE);
    }

    /// The low-res ray-march tile sample equals the golden `rgba16f` stride.
    #[test]
    fn raymarch_tile_matches_golden_stride() {
        assert_eq!(size_of::<GpuRaymarchTile>() as u32, RAYMARCH_TILE_STRIDE);
    }

    /// The reprojection-history pixel equals the golden `rgba16f` stride.
    #[test]
    fn history_pixel_matches_golden_stride() {
        assert_eq!(size_of::<GpuHistoryPixel>() as u32, HISTORY_PIXEL_STRIDE);
    }

    /// The light-space cloud-shadow texel equals the golden four-node `AVSM`
    /// stride. This pins the full four `(depth, transmittance)` knots (`32`
    /// bytes), so a mirror truncated to fewer knots fails the build.
    #[test]
    fn shadow_texel_matches_golden_stride() {
        assert_eq!(size_of::<GpuShadowTexel>() as u32, SHADOW_TEXEL_STRIDE);
        assert_eq!(size_of::<GpuShadowTexel>(), 32);
        assert_eq!(size_of::<GpuAvsmNode>(), 8);
    }

    /// The multiple-scatter `LUT` cell equals the golden `rgba16f` stride.
    #[test]
    fn multiscatter_cell_matches_golden_stride() {
        assert_eq!(
            size_of::<GpuMultiscatterCell>() as u32,
            MULTISCATTER_CELL_STRIDE
        );
    }

    // --- per-dispatch immediate param blocks -----------------------------
    //
    // Each block is all-scalar (`u32` / `f32`), so the packed `#[repr(C)]`
    // size equals `4 * field_count` and matches the `WESL` push-constant
    // layout byte-for-byte. Pinning the sizes catches a field added or dropped
    // on either side of the boundary.

    /// `volumetric_weather_advect`: five 4-byte scalars.
    #[test]
    fn weather_advect_params_size() {
        assert_eq!(size_of::<GpuWeatherAdvectParams>(), 20);
        assert_eq!(align_of::<GpuWeatherAdvectParams>(), 4);
    }

    /// `volumetric_noise_bake`: six 4-byte scalars.
    #[test]
    fn noise_bake_params_size() {
        assert_eq!(size_of::<GpuNoiseBakeParams>(), 24);
        assert_eq!(align_of::<GpuNoiseBakeParams>(), 4);
    }

    /// `volumetric_modeling`: seven 4-byte scalars.
    #[test]
    fn modeling_params_size() {
        assert_eq!(size_of::<GpuModelingParams>(), 28);
        assert_eq!(align_of::<GpuModelingParams>(), 4);
    }

    /// `volumetric_multiscatter_lut_bake`: seven 4-byte scalars.
    #[test]
    fn ms_lut_params_size() {
        assert_eq!(size_of::<GpuMsLutParams>(), 28);
        assert_eq!(align_of::<GpuMsLutParams>(), 4);
    }

    /// `volumetric_raymarch`: fifteen 4-byte scalars.
    #[test]
    fn raymarch_params_size() {
        assert_eq!(size_of::<GpuRaymarchParams>(), 60);
        assert_eq!(align_of::<GpuRaymarchParams>(), 4);
    }

    /// `volumetric_scatter_resolve`: twelve 4-byte scalars.
    #[test]
    fn scatter_resolve_params_size() {
        assert_eq!(size_of::<GpuScatterResolveParams>(), 48);
        assert_eq!(align_of::<GpuScatterResolveParams>(), 4);
    }

    /// `volumetric_shadow_march`: seven 4-byte scalars.
    #[test]
    fn shadow_march_params_size() {
        assert_eq!(size_of::<GpuShadowMarchParams>(), 28);
        assert_eq!(align_of::<GpuShadowMarchParams>(), 4);
    }

    /// `volumetric_upsample`: seven 4-byte scalars.
    #[test]
    fn upsample_params_size() {
        assert_eq!(size_of::<GpuUpsampleParams>(), 28);
        assert_eq!(align_of::<GpuUpsampleParams>(), 4);
    }

    // --- workgroup tiles --------------------------------------------------

    /// The ABI tile constants match the `@workgroup_size` the architecture
    /// crate's kernel descriptors launch each dispatch with (a `4x4x4` voxel
    /// brick for the 3D bakes, an `8x8x1` tile for the 2D screen/shadow passes),
    /// so a retiling on either side fails the build.
    #[test]
    fn workgroup_tiles_match_kernel_descriptors() {
        for kernel in VolumetricKernel::ALL {
            let wg = kernel.descriptor().workgroup;
            match kernel {
                VolumetricKernel::NoiseBake
                | VolumetricKernel::Modeling
                | VolumetricKernel::MultiscatterLutBake => {
                    assert_eq!(
                        (wg.x, wg.y, wg.z),
                        (VC_VOXEL_BRICK, VC_VOXEL_BRICK, VC_VOXEL_BRICK),
                        "{kernel:?} should launch the voxel brick"
                    );
                }
                VolumetricKernel::WeatherAdvect
                | VolumetricKernel::Raymarch
                | VolumetricKernel::ScatterResolve
                | VolumetricKernel::ShadowMarch
                | VolumetricKernel::Upsample => {
                    assert_eq!(
                        (wg.x, wg.y, wg.z),
                        (VC_SCREEN_TILE, VC_SCREEN_TILE, 1),
                        "{kernel:?} should launch the screen tile"
                    );
                }
            }
        }
    }
}
