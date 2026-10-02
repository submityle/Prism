//! The water-surface pass's **world-space global-illumination** `@group(8)`
//! plumbing: the `CPU` half that lets the §5 image-based ambient term pick up
//! the engine's `Lumen`-style *far-field* diffuse indirect, gathered from the
//! shared screen-probe buffer the opaque
//! [`ViewWorldSpaceGi`](crate::shading::ViewWorldSpaceGi) prepass already
//! filled this frame.
//!
//! ## Why the water consumes world-space GI (and not just `SSGI`)
//!
//! The sibling [`super::surface_ssgi`] slice gathers a single *near-field*
//! bounce out of the composited `scene_color` — it is blind to any indirect
//! light whose source is off-screen or behind the camera. The opaque
//! world-space GI pass closes that gap: `probe_update` seeds one L1 spherical
//! harmonic radiance probe per screen tile (`view`-space `SH`), persisting a
//! `Lumen`-style radiance field the resolve pass then interpolates. This slice
//! lets the transparent water fragment consume that *same* probe buffer:
//! it reconstructs its own view-space normal/depth (reusing the `@group(3)`
//! [`super::surface_ssr::GpuWaterSsrConfig`] `view_from_world` matrix), gathers
//! the four screen probes around its pixel, and blends their `SH` irradiance
//! with the exact bilinear-plus-geometry weights the golden
//! [`prism_render_shading::interpolate_irradiance`] encodes. No ray tracer, no
//! extra pass, and no buffer of its own — the probe buffer is already resident.
//!
//! ## Why a storage-buffer fallback resource
//!
//! Mirroring [`super::surface_ssr::WaterSsrFallback`]: the
//! [`WaterSurfacePipelines`](super::surface_pipeline::WaterSurfacePipelines)
//! specializer is a pure-`CPU` descriptor bag with no [`RenderDevice`], so the
//! draw node must bind a real, correctly-sized probe storage buffer even on a
//! view with no resident [`ViewWorldSpaceGi`](crate::shading::ViewWorldSpaceGi)
//! (feature off or first frame). [`WaterWorldSpaceGiFallback`] holds a single
//! dummy probe's worth of storage, built once at `RenderStartup` by
//! [`init_water_world_space_gi_fallback`]; the draw node clears this config's
//! `sample_enable` bit whenever it binds the fallback, so the shader skips the
//! gather and adds nothing — but the bind group stays structurally complete.
//!
//! The sibling [`super::surface_pipeline`] slice owns the raster pipelines;
//! this slice adds the ninth (`@group(8)`) bind group, built once per view by
//! [`super::surface_draw`].

use bevy_ecs::prelude::*;
use bevy_material::bind_group_layout_entries::{
    binding_types::{storage_buffer_read_only_sized, uniform_buffer_sized},
    BindGroupLayoutEntries,
};
use bevy_math::UVec2;
use bevy_render::{
    render_resource::{Buffer, BufferDescriptor, BufferUsages, ShaderStages},
    renderer::RenderDevice,
};
use bytemuck::{Pod, Zeroable};

/// Bytes per stored screen probe, matching the opaque
/// [`PROBE_STRIDE`](crate::shading::world_space_gi) layout (`6 x vec4<f32>` =
/// 96 B). The fallback buffer allocates exactly one probe so the bind group is
/// valid on a view with no resident probe field.
const PROBE_STRIDE: u64 = 96;

/// `GPU`-side mirror of the shader's `WaterWorldSpaceGiConfig` uniform (the
/// water `@group(8) @binding(1)` block).
///
/// Layout matches the `WGSL` struct byte-for-byte: a leading `vec2<u32>`
/// (`probe_grid`, 8-byte aligned) precedes the `tile` count, three `f32`
/// tunables, the enable bit, and one trailing `u32` pad — 32 bytes, a multiple
/// of the 16-byte `WGSL` uniform block alignment. The projection / world->view
/// basis is **not** duplicated here: the shader reads it from the `@group(3)`
/// [`super::surface_ssr::GpuWaterSsrConfig`] uniform, since the probe gather
/// shares the `SSR`/`GTAO` view-space reconstruction.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWaterWorldSpaceGiConfig {
    /// Probe-grid dimensions in probes (`div_ceil(size, tile)` per axis), the
    /// row-major extent the gather indexes. Floored to `1` per axis so a
    /// disabled/garbage grid can never produce a zero-width index space.
    pub probe_grid: [u32; 2],
    /// Tile size in pixels per probe, floored to `1` by [`Self::new`]; the
    /// gather maps a shading pixel to probe space as `pixel / tile - 0.5`.
    pub tile: u32,
    /// Minimum `dot(n_probe, n_point)` for a probe to be accepted. Mirrors the
    /// opaque golden
    /// [`InterpolationConfig::normal_threshold`](prism_render_shading::InterpolationConfig).
    pub normal_threshold: f32,
    /// Maximum relative depth difference for a probe to be accepted. Mirrors the
    /// opaque golden
    /// [`InterpolationConfig::depth_rel_threshold`](prism_render_shading::InterpolationConfig).
    pub depth_rel_threshold: f32,
    /// Artistic gain applied to the resolved irradiance before the Lambert
    /// combine (`albedo * E / PI`). `0` disables the contribution without
    /// touching the enable bit.
    pub intensity: f32,
    /// `1` runs the gather; `0` skips it and adds no indirect diffuse. The draw
    /// node clears it when world-space GI is off or no probe field is resident
    /// this frame.
    pub sample_enable: u32,
    /// Padding to the 16-byte uniform block alignment; never read.
    pub _pad0: u32,
}

impl GpuWaterWorldSpaceGiConfig {
    /// Build the uniform from the per-view probe grid and the shared
    /// world-space GI settings, packing the enable bit into
    /// [`Self::sample_enable`].
    ///
    /// `probe_grid` is floored at `1` per axis and `tile` at `1` so a disabled
    /// or garbage setting still yields a structurally valid gather the shader
    /// can early-out of via the enable bit.
    pub(crate) fn new(
        sample_enable: bool,
        probe_grid: UVec2,
        tile: u32,
        normal_threshold: f32,
        depth_rel_threshold: f32,
        intensity: f32,
    ) -> Self {
        Self {
            probe_grid: [probe_grid.x.max(1), probe_grid.y.max(1)],
            tile: tile.max(1),
            normal_threshold,
            depth_rel_threshold,
            intensity,
            sample_enable: u32::from(sample_enable),
            _pad0: 0,
        }
    }
}

/// Builds the water-surface `@group(8)` world-space GI layout entries (two
/// bindings), in the exact `@binding(n)` order `water_surface_raster.wesl`
/// declares:
///
/// 0. the shared screen-probe storage buffer (read-only; the gather reads it
///    with an index, so it needs no sampler), and
/// 1. the [`GpuWaterWorldSpaceGiConfig`] uniform.
///
/// Declared [`ShaderStages::FRAGMENT`]: the water surface gathers the probes in
/// its fragment stage, reusing the `@group(3)` view-space reconstruction, so it
/// needs no texture or sampler of its own.
pub(crate) fn wsgi_layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::FRAGMENT,
        (
            storage_buffer_read_only_sized(false, None),
            uniform_buffer_sized(false, None),
        ),
    )
}

/// Correctly-sized fallback probe buffer bound into the surface pass's
/// `@group(8)` on any view without a resident
/// [`ViewWorldSpaceGi`](crate::shading::ViewWorldSpaceGi).
///
/// Built once at `RenderStartup` by [`init_water_world_space_gi_fallback`];
/// read by the raster draw node ([`super::surface_draw`]). The draw node clears
/// the config's `sample_enable` bit whenever it binds this buffer, so the
/// shader never actually reads it — but the bind group is still structurally
/// complete.
#[derive(Resource)]
pub(crate) struct WaterWorldSpaceGiFallback {
    /// A single dummy probe's worth of storage ([`PROBE_STRIDE`] bytes),
    /// `STORAGE`-usable so the read-only binding is valid when no probe field
    /// is resident.
    pub(crate) probe_buffer: Buffer,
}

/// `RenderStartup` initializer that builds the surface pass's world-space GI
/// fallback probe buffer and inserts [`WaterWorldSpaceGiFallback`].
pub(crate) fn init_water_world_space_gi_fallback(
    mut commands: Commands,
    device: Res<RenderDevice>,
) {
    let probe_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("prism water surface wsgi dummy probes"),
        size: PROBE_STRIDE,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    commands.insert_resource(WaterWorldSpaceGiFallback { probe_buffer });
}

/// Physical diffuse combine for the resolved world-space irradiance: a `CPU`
/// twin of the shader's final `albedo * E / PI`. Each channel multiplies the
/// clamped-cosine irradiance `E` (golden
/// [`prism_render_shading::interpolate_irradiance`]) by the surface albedo and
/// the Lambert `1/PI` normalisation, clamped non-negative so a stray negative
/// input cannot subtract light. Matches the image-based ambient diffuse term
/// (`water_ibl`'s `base_color * irradiance * INV_PI`) by construction, so the
/// world-space gather folds into the same ambient budget without double
/// normalisation.
#[cfg(test)]
fn gi_diffuse(irradiance: [f32; 3], albedo: [f32; 3]) -> [f32; 3] {
    let inv_pi = core::f32::consts::FRAC_1_PI;
    let chan = |e: f32, a: f32| (e.max(0.0) * a.max(0.0) * inv_pi).max(0.0);
    [
        chan(irradiance[0], albedo[0]),
        chan(irradiance[1], albedo[1]),
        chan(irradiance[2], albedo[2]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};
    use prism_render_shading::{
        interpolate_irradiance, InterpolationConfig, ProbeNeighbor, ShL1Rgb,
    };

    use bevy_math::Vec3;

    const EPS: f32 = 1.0e-6;

    #[test]
    fn config_matches_the_wgsl_uniform_block_size_and_alignment() {
        // `vec2<u32>` (8 B) + `tile` (4 B) + three `f32` (12 B) + enable (4 B) +
        // one 4-byte pad = 32 B, a multiple of the 16-byte `WGSL` uniform block
        // alignment.
        assert_eq!(size_of::<GpuWaterWorldSpaceGiConfig>(), 32);
        assert_eq!(align_of::<GpuWaterWorldSpaceGiConfig>(), 4);
    }

    #[test]
    fn new_packs_the_enable_bit_grid_and_thresholds() {
        let enabled =
            GpuWaterWorldSpaceGiConfig::new(true, UVec2::new(120, 68), 16, 0.9, 0.1, 1.25);
        assert_eq!(enabled.sample_enable, 1);
        assert_eq!(enabled.probe_grid, [120, 68]);
        assert_eq!(enabled.tile, 16);
        assert_eq!(enabled._pad0, 0);
        assert!((enabled.normal_threshold - 0.9).abs() < EPS);
        assert!((enabled.depth_rel_threshold - 0.1).abs() < EPS);
        assert!((enabled.intensity - 1.25).abs() < EPS);

        let disabled = GpuWaterWorldSpaceGiConfig::new(false, UVec2::new(1, 1), 16, 0.9, 0.1, 1.0);
        assert_eq!(disabled.sample_enable, 0);
    }

    #[test]
    fn new_floors_a_zero_grid_and_tile() {
        let cfg = GpuWaterWorldSpaceGiConfig::new(true, UVec2::ZERO, 0, 0.9, 0.1, 1.0);
        assert_eq!(cfg.probe_grid, [1, 1]);
        assert_eq!(cfg.tile, 1);
    }

    #[test]
    fn layout_declares_two_fragment_bindings() {
        let entries = wsgi_layout_entries();
        assert_eq!(entries.len(), 2);
        for entry in entries.iter() {
            assert!(entry.visibility.contains(ShaderStages::FRAGMENT));
        }
    }

    #[test]
    fn gi_diffuse_is_lambert_scaled_and_non_negative() {
        let irradiance = [2.0, 1.0, 0.5];
        let albedo = [0.5, 0.5, 0.5];
        let out = gi_diffuse(irradiance, albedo);
        let inv_pi = core::f32::consts::FRAC_1_PI;
        assert!((out[0] - 2.0 * 0.5 * inv_pi).abs() < EPS);
        assert!((out[1] - 1.0 * 0.5 * inv_pi).abs() < EPS);
        // Negative inputs cannot subtract light.
        let neg = gi_diffuse([-5.0, 1.0, 1.0], [1.0, -1.0, 1.0]);
        assert!(neg[0].abs() < EPS);
        assert!(neg[1].abs() < EPS);
        assert!(neg[2] > 0.0);
    }

    #[test]
    fn gi_diffuse_scales_linearly_with_irradiance() {
        let albedo = [0.8, 0.8, 0.8];
        let a = gi_diffuse([1.0, 1.0, 1.0], albedo);
        let b = gi_diffuse([2.0, 2.0, 2.0], albedo);
        assert!((b[0] - 2.0 * a[0]).abs() < EPS);
    }

    #[test]
    fn end_to_end_gather_matches_the_golden_interpolation() {
        // Drive the shared golden interpolation the shader twins arm-for-arm,
        // then apply the water Lambert combine: four identical in-range probes
        // facing the shading normal must resolve to their own irradiance,
        // diffuse-weighted by albedo and `1/PI`.
        let mut sh = ShL1Rgb::ZERO;
        sh.add_directional_radiance(Vec3::Y, [1.0, 0.8, 0.6], 1.0);
        let normal = Vec3::Y;
        let depth = 10.0;
        let neighbors = [
            ProbeNeighbor::new(sh, normal, depth),
            ProbeNeighbor::new(sh, normal, depth),
            ProbeNeighbor::new(sh, normal, depth),
            ProbeNeighbor::new(sh, normal, depth),
        ];
        let cfg = InterpolationConfig::default();
        let e = interpolate_irradiance(&neighbors, 0.5, 0.5, normal, depth, &cfg);
        let albedo = [0.5, 0.4, 0.3];
        let out = gi_diffuse([e.x, e.y, e.z], albedo);
        let inv_pi = core::f32::consts::FRAC_1_PI;
        assert!((out[0] - e.x.max(0.0) * albedo[0] * inv_pi).abs() < EPS);
        assert!((out[1] - e.y.max(0.0) * albedo[1] * inv_pi).abs() < EPS);
        assert!((out[2] - e.z.max(0.0) * albedo[2] * inv_pi).abs() < EPS);
        // A valid probe field facing the normal yields a positive bounce.
        assert!(out[0] > 0.0);
    }
}
