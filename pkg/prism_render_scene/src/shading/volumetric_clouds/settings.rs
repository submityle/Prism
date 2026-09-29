//! Frame-constant tunables for Prism's volumetric-cloud subsystem.
//!
//! [`PrismVolumetricCloudsSettings`] is the single render-world resource the
//! eight cloud compute passes read to size their resident resources and to
//! parameterise the medium. Every field is a plain tunable mirrored into the
//! golden [`prism_render_architecture::volumetric`] reduction: the domain
//! resolutions size the resident textures ([`super::resources`]), and the
//! per-dispatch immediate blocks the passes upload are built here through the
//! `*_params` constructors so the sizing the layout binds and the sizing the
//! shader reads can never drift.
//!
//! There is no architecture-boundary contract resource for the cloud front (the
//! architecture crate owns numerics, not the render-world enable toggle), so —
//! mirroring the sibling froxel [`super::super::volumetrics`] fog — the
//! subsystem owns its enable flag here rather than in the shared shading front,
//! keeping the clouds opt-in without coupling to `runtime.rs`. A game can
//! overwrite the resource to retune globally without touching any pass code.

use bevy_ecs::prelude::Resource;
use bevy_math::{UVec2, UVec3};

use super::abi::{
    GpuModelingParams, GpuMsLutParams, GpuNoiseBakeParams, GpuRaymarchParams,
    GpuScatterResolveParams, GpuShadowMarchParams, GpuUpsampleParams, GpuWeatherAdvectParams,
};

/// Global volumetric-cloud settings consumed by the eight cloud compute passes.
///
/// The immediate blocks the passes upload are built from these through the
/// `*_params` constructors, which fold in the per-frame runtime values (screen
/// extent, sun/view cosine, frame parity) the resource itself cannot know.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismVolumetricCloudsSettings {
    /// Master enable. Clouds are opt-in: `false` skips every prepare/dispatch
    /// step and drops the cached resident textures so nothing lingers resident.
    pub enabled: bool,

    /// Density-cache voxel grid `[x, y, z]`: the world-space cloud domain the
    /// noise bake and modelling compose and the ray-march / shadow march
    /// sample.
    pub density_dim: UVec3,
    /// Weather-map resolution `[w, h]`: the 2D coverage/type/precip field the
    /// semi-Lagrangian advection evolves.
    pub weather_dim: UVec2,
    /// Multiple-scatter `LUT` resolution `[cos, depth, albedo]`.
    pub mslut_dim: UVec3,
    /// Light-space cloud-shadow (`AVSM`) map resolution `[w, h]`.
    pub shadow_dim: UVec2,
    /// Divisor from the full-resolution framebuffer to the low-resolution
    /// ray-march / scatter target (quarter-res at `2`, sixteenth at `4`).
    pub lowres_scale: u32,

    /// Wind velocity `[x, y]` (texels/step) the advection back-traces along.
    pub wind: [f32; 2],
    /// Advection time step (`s`).
    pub advect_dt: f32,

    /// Base Perlin-Worley shape frequency.
    pub base_freq: f32,
    /// Detail-erosion Worley frequency.
    pub detail_freq: f32,
    /// Deterministic noise seed.
    pub seed: u32,

    /// Global coverage bias `[0, 1]` (more cloud at higher values).
    pub coverage: f32,
    /// Cloud-type morph `[0, 1]` (stratus..cumulonimbus).
    pub cloud_type: f32,
    /// Detail-erosion strength `[0, 1]`.
    pub erosion_strength: f32,
    /// Cloud-kind tag (`0`=Cumulus, `1`=Stratus, `2`=Cirrus, `3`=Cumulonimbus).
    pub kind: u32,

    /// Multi-scatter octave attenuation per octave `[0, 1]`.
    pub ms_attenuation: f32,
    /// Multi-scatter octave contribution per octave `[0, 1]`.
    pub ms_contribution: f32,
    /// Draine/HG eccentricity used to pre-integrate the `LUT`.
    pub ms_eccentricity: f32,
    /// Number of scattering octaves the `LUT` folds.
    pub ms_octaves: u32,

    /// Nominal ray-march step in world units.
    pub base_step: f32,
    /// Maximum adaptive step (empty-space skip).
    pub max_step: f32,
    /// Minimum adaptive step (dense-medium refinement).
    pub min_step: f32,
    /// Density below which a sample is treated as empty space.
    pub density_threshold: f32,
    /// Transmittance below which the march early-outs.
    pub transmittance_cutoff: f32,
    /// Hard cap on ray-march iterations.
    pub max_steps: u32,
    /// Extinction coefficient `sigma_t`.
    pub sigma_t: f32,
    /// Single-scatter albedo.
    pub albedo: f32,
    /// Primary Henyey-Greenstein anisotropy `g`.
    pub phase_g: f32,

    /// Forward HG lobe anisotropy for the dual-lobe resolve.
    pub forward_g: f32,
    /// Backward HG lobe anisotropy for the dual-lobe resolve.
    pub backward_g: f32,
    /// Forward/backward lobe blend weight `[0, 1]`.
    pub lobe_blend: f32,

    /// Light-space shadow-march step in world units.
    pub shadow_step: f32,
    /// Density scale applied along the shadow march.
    pub shadow_density_scale: f32,

    /// Temporal-upsample reconstruction mode (`0`=history-clamp blend).
    pub upsample_mode: u32,
    /// Neighbourhood-variance clamp gamma for the history rectification.
    pub variance_gamma: f32,
}

impl Default for PrismVolumetricCloudsSettings {
    fn default() -> Self {
        Self {
            // Opt-in, matching every other shading front's enable default.
            enabled: false,
            density_dim: UVec3::new(128, 32, 128),
            weather_dim: UVec2::new(512, 512),
            mslut_dim: UVec3::new(32, 32, 32),
            shadow_dim: UVec2::new(1024, 1024),
            lowres_scale: 2,

            wind: [1.0, 0.0],
            advect_dt: 1.0 / 60.0,

            base_freq: 4.0,
            detail_freq: 16.0,
            seed: 0x5eed_c10d,

            coverage: 0.5,
            cloud_type: 0.5,
            erosion_strength: 0.5,
            kind: 0,

            ms_attenuation: 0.5,
            ms_contribution: 0.5,
            ms_eccentricity: 0.6,
            ms_octaves: 4,

            base_step: 8.0,
            max_step: 64.0,
            min_step: 2.0,
            density_threshold: 0.01,
            transmittance_cutoff: 0.01,
            max_steps: 128,
            sigma_t: 0.1,
            albedo: 0.9,
            phase_g: 0.2,

            forward_g: 0.8,
            backward_g: -0.3,
            lobe_blend: 0.5,

            shadow_step: 16.0,
            shadow_density_scale: 1.0,

            upsample_mode: 0,
            variance_gamma: 1.0,
        }
    }
}

impl PrismVolumetricCloudsSettings {
    /// The low-resolution ray-march / scatter target extent derived from the
    /// full framebuffer extent and [`Self::lowres_scale`]. Clamped to at least
    /// one texel on each axis so a degenerate viewport still allocates.
    pub(crate) fn lowres_size(&self, full: UVec2) -> UVec2 {
        let scale = self.lowres_scale.max(1);
        UVec2::new((full.x / scale).max(1), (full.y / scale).max(1))
    }

    /// The `volumetric_weather_advect` immediate block.
    pub(crate) fn weather_advect_params(&self) -> GpuWeatherAdvectParams {
        GpuWeatherAdvectParams {
            width: self.weather_dim.x,
            height: self.weather_dim.y,
            wind_x: self.wind[0],
            wind_y: self.wind[1],
            dt: self.advect_dt,
        }
    }

    /// The `volumetric_noise_bake` immediate block.
    pub(crate) fn noise_bake_params(&self) -> GpuNoiseBakeParams {
        GpuNoiseBakeParams {
            dim_x: self.density_dim.x,
            dim_y: self.density_dim.y,
            dim_z: self.density_dim.z,
            base_freq: self.base_freq,
            detail_freq: self.detail_freq,
            seed: self.seed,
        }
    }

    /// The `volumetric_modeling` immediate block.
    pub(crate) fn modeling_params(&self) -> GpuModelingParams {
        GpuModelingParams {
            dim_x: self.density_dim.x,
            dim_y: self.density_dim.y,
            dim_z: self.density_dim.z,
            coverage: self.coverage,
            cloud_type: self.cloud_type,
            erosion_strength: self.erosion_strength,
            kind: self.kind,
        }
    }

    /// The `volumetric_multiscatter_lut_bake` immediate block.
    pub(crate) fn ms_lut_params(&self) -> GpuMsLutParams {
        GpuMsLutParams {
            dim_cos: self.mslut_dim.x,
            dim_depth: self.mslut_dim.y,
            dim_albedo: self.mslut_dim.z,
            attenuation: self.ms_attenuation,
            contribution: self.ms_contribution,
            eccentricity: self.ms_eccentricity,
            octave_count: self.ms_octaves,
        }
    }

    /// The `volumetric_raymarch` immediate block over the low-res target
    /// `lowres`.
    pub(crate) fn raymarch_params(&self, lowres: UVec2) -> GpuRaymarchParams {
        GpuRaymarchParams {
            screen_w: lowres.x,
            screen_h: lowres.y,
            grid_x: self.density_dim.x,
            grid_y: self.density_dim.y,
            grid_z: self.density_dim.z,
            base_step: self.base_step,
            max_step: self.max_step,
            min_step: self.min_step,
            density_threshold: self.density_threshold,
            transmittance_cutoff: self.transmittance_cutoff,
            max_steps: self.max_steps,
            sigma_t: self.sigma_t,
            albedo: self.albedo,
            phase_g: self.phase_g,
        }
    }

    /// The `volumetric_scatter_resolve` immediate block over the low-res target
    /// `lowres`, with the sun/view `cos_theta` folded in for the phase eval.
    pub(crate) fn scatter_resolve_params(
        &self,
        lowres: UVec2,
        cos_theta: f32,
    ) -> GpuScatterResolveParams {
        GpuScatterResolveParams {
            screen_w: lowres.x,
            screen_h: lowres.y,
            lut_cos: self.mslut_dim.x,
            lut_depth: self.mslut_dim.y,
            lut_albedo: self.mslut_dim.z,
            forward_g: self.forward_g,
            backward_g: self.backward_g,
            lobe_blend: self.lobe_blend,
            albedo: self.albedo,
            cos_theta,
        }
    }

    /// The `volumetric_shadow_march` immediate block.
    pub(crate) fn shadow_march_params(&self) -> GpuShadowMarchParams {
        GpuShadowMarchParams {
            shadow_w: self.shadow_dim.x,
            shadow_h: self.shadow_dim.y,
            grid_x: self.density_dim.x,
            grid_y: self.density_dim.y,
            grid_z: self.density_dim.z,
            step: self.shadow_step,
            density_scale: self.shadow_density_scale,
        }
    }

    /// The `volumetric_upsample` immediate block reconstructing the full-res
    /// `full` buffer from the low-res `lowres` resolve and the reprojected
    /// history, tagged with the current `frame_index` for the parity walk.
    pub(crate) fn upsample_params(
        &self,
        full: UVec2,
        lowres: UVec2,
        frame_index: u32,
    ) -> GpuUpsampleParams {
        GpuUpsampleParams {
            screen_w: full.x,
            screen_h: full.y,
            lowres_w: lowres.x,
            lowres_h: lowres.y,
            frame_index,
            mode: self.upsample_mode,
            variance_gamma: self.variance_gamma,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PrismVolumetricCloudsSettings;
    use bevy_math::UVec2;

    /// The low-res target divides the framebuffer by `lowres_scale` and never
    /// collapses below one texel per axis.
    #[test]
    fn lowres_size_divides_and_clamps() {
        let mut s = PrismVolumetricCloudsSettings {
            lowres_scale: 4,
            ..Default::default()
        };
        assert_eq!(s.lowres_size(UVec2::new(1920, 1080)), UVec2::new(480, 270));
        s.lowres_scale = 0; // guarded to 1, never divides by zero.
        assert_eq!(s.lowres_size(UVec2::new(1, 1)), UVec2::new(1, 1));
        s.lowres_scale = 8;
        assert_eq!(s.lowres_size(UVec2::new(4, 4)), UVec2::new(1, 1));
    }

    /// Every `*_params` constructor forwards the settings dimensions verbatim so
    /// the immediate block the pass uploads matches the resident texture sizing.
    #[test]
    fn params_forward_dimensions() {
        let s = PrismVolumetricCloudsSettings::default();
        let n = s.noise_bake_params();
        assert_eq!(
            [n.dim_x, n.dim_y, n.dim_z],
            [s.density_dim.x, s.density_dim.y, s.density_dim.z]
        );
        let w = s.weather_advect_params();
        assert_eq!([w.width, w.height], [s.weather_dim.x, s.weather_dim.y]);
        let lowres = s.lowres_size(UVec2::new(1920, 1080));
        let rm = s.raymarch_params(lowres);
        assert_eq!([rm.screen_w, rm.screen_h], [lowres.x, lowres.y]);
        let up = s.upsample_params(UVec2::new(1920, 1080), lowres, 7);
        assert_eq!(up.frame_index, 7);
        assert_eq!([up.lowres_w, up.lowres_h], [lowres.x, lowres.y]);
    }
}
