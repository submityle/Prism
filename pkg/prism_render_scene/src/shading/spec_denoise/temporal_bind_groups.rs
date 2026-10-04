//! `PrepareBindGroups` systems building the specular-GI *temporal denoise*
//! passes' group-0 bind groups per view, plus the per-frame config-uniform
//! uploads.
//!
//! Two dispatches each read an 11-binding group 0 (mirroring the frozen
//! `spec_denoise_reproject.wesl` / `spec_denoise_history_clamp.wesl` contracts):
//!
//! * **reproject** — the 304-byte [`GpuSpecDenoiseReprojectConfig`] uniform this
//!   module uploads (0), the current-frame SSR normal/roughness (1), SSR hit
//!   distance (2) and reverse-Z depth (3), the four persistent previous-frame
//!   planes (history 4, depth 5, meta 6, luma 7) and the three transient
//!   write-only reprojected targets (8, 9, 10);
//! * **history-clamp** — the 48-byte [`GpuSpecDenoiseHistoryClampConfig`] uniform
//!   (0), the three reprojected reads (history 1, meta 2, luma 3), the `spec_gi`
//!   noisy resolve (4), the SSR normal/roughness (5) and reverse-Z depth (6),
//!   and the four write-only targets (next-frame history 7, meta 8, luma 9 and
//!   the denoised specular 10 the spatial pass filters).
//!
//! Like [`super::bind_groups`] and [`super::super::spec_gi::bind_groups`], each
//! kernel reads its config from a *bound uniform*, so these systems allocate a
//! small per-view uniform buffer, write the frozen config into it via the render
//! queue and bind it at `@binding(0)`. The reproject config is built from the
//! reprojection transforms and camera positions [`super::temporal_resources`]
//! already stashed on [`ViewSpecDenoiseTemporal`]; both configs fold in the
//! denoiser tunables hardcoded from the CPU golden
//! `prism_render_shading::gi::spec_denoise` defaults (there is no per-view
//! runtime tunable yet).
//!
//! The bind groups are present exactly when the view has resident SSR textures,
//! resident `spec_gi` reuse resources and resident temporal planes; the
//! subsystem gate guarantees the three appear and disappear together.

use bevy_ecs::prelude::*;
use bevy_math::{Mat4, UVec2, Vec3};
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries, BufferDescriptor, BufferUsages},
    renderer::{RenderDevice, RenderQueue},
};

use super::super::spec_gi::ViewSpecGiReuse;
use super::super::ssr::ViewSsrTextures;
use super::temporal_abi::{GpuSpecDenoiseHistoryClampConfig, GpuSpecDenoiseReprojectConfig};
use super::temporal_pipeline::{SpecDenoiseHistoryClampPipeline, SpecDenoiseReprojectPipeline};
use super::temporal_resources::ViewSpecDenoiseTemporal;

/// Reprojection tunables, mirroring `prism_render_shading::gi::spec_denoise`
/// `ReprojectParams::default()` plus the GPU-only world-space disocclusion
/// tolerance. Hardcoded here because there is no per-view runtime tunable for
/// the specular denoiser yet; the WESL `sdr_*` library and the CPU golden read
/// the first three verbatim.
const REPROJECT_PARALLAX_SENSITIVITY: f32 = 8.0;
const REPROJECT_VIRTUAL_EXPONENT: f32 = 2.0;
const REPROJECT_MIN_CONFIDENCE: f32 = 0.0;
/// Relative world-space disocclusion tolerance (fraction of the camera→surface
/// distance); the reproject guard rejects history whose rebuilt previous surface
/// has drifted further than this. GPU-side knob with no CPU-golden twin.
const REPROJECT_DEPTH_REJECTION: f32 = 0.05;

/// History-clamp tunables, mirroring `prism_render_shading::gi::spec_denoise`
/// `HistoryClampParams::default()` plus the dual-rate EMA blend floors.
const CLAMP_SIGMA: f32 = 2.0;
const CLAMP_LOBE_TOLERANCE: f32 = 2.0;
const CLAMP_FAST_SENSITIVITY: f32 = 4.0;
/// Lower bound on the responsive (fast) luminance EMA blend. GPU-side knob.
const CLAMP_FAST_RATE: f32 = 0.5;
/// Lower bound on the stable (slow) luminance EMA blend. GPU-side knob.
const CLAMP_SLOW_RATE: f32 = 0.08;
const CLAMP_MIN_FRAMES: u32 = 2;
const CLAMP_MAX_FRAMES: u32 = 32;

/// Builds the reproject dispatch's config uniform from the reprojection
/// transforms, camera positions and framebuffer extent stashed on the view's
/// temporal state, folding in the hardcoded reprojection tunables.
///
/// Factored out of the ECS system so the config packing is unit-testable without
/// a render world.
#[allow(clippy::too_many_arguments)]
fn reproject_config(
    world_from_clip: Mat4,
    prev_clip_from_world: Mat4,
    prev_world_from_clip: Mat4,
    world_from_view: Mat4,
    curr_cam_pos: Vec3,
    prev_cam_pos: Vec3,
    size: UVec2,
) -> GpuSpecDenoiseReprojectConfig {
    GpuSpecDenoiseReprojectConfig::new(
        world_from_clip,
        prev_clip_from_world,
        prev_world_from_clip,
        world_from_view,
        curr_cam_pos,
        REPROJECT_PARALLAX_SENSITIVITY,
        prev_cam_pos,
        REPROJECT_VIRTUAL_EXPONENT,
        size.x,
        size.y,
        REPROJECT_MIN_CONFIDENCE,
        REPROJECT_DEPTH_REJECTION,
    )
}

/// Builds the history-clamp dispatch's config uniform from the framebuffer
/// extent and the hardcoded clamp/EMA tunables.
fn history_clamp_config(size: UVec2) -> GpuSpecDenoiseHistoryClampConfig {
    GpuSpecDenoiseHistoryClampConfig::new(
        CLAMP_SIGMA,
        CLAMP_LOBE_TOLERANCE,
        CLAMP_FAST_SENSITIVITY,
        CLAMP_FAST_RATE,
        CLAMP_SLOW_RATE,
        CLAMP_MIN_FRAMES,
        CLAMP_MAX_FRAMES,
        size.x,
        size.y,
    )
}

/// The reproject dispatch's group-0 bind group for a single view.
#[derive(Component)]
pub(crate) struct ViewSpecDenoiseReprojectBindGroup {
    /// group 0 for `spec_denoise_reproject`: config uniform (0), the three
    /// current-frame SSR reads (1, 2, 3), the four persistent previous-frame
    /// planes (4, 5, 6, 7) and the three write-only reprojected targets
    /// (8, 9, 10).
    group: BindGroup,
}

impl ViewSpecDenoiseReprojectBindGroup {
    /// group-0 bind group the reproject dispatch node records against.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// The history-clamp dispatch's group-0 bind group for a single view.
#[derive(Component)]
pub(crate) struct ViewSpecDenoiseHistoryClampBindGroup {
    /// group 0 for `spec_denoise_history_clamp`: config uniform (0), the three
    /// reprojected reads (1, 2, 3), the `spec_gi` resolve (4), the SSR
    /// normal/roughness (5) and reverse-Z depth (6), and the four write-only
    /// targets (7, 8, 9, 10).
    group: BindGroup,
}

impl ViewSpecDenoiseHistoryClampBindGroup {
    /// group-0 bind group the history-clamp dispatch node records against.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewSpecDenoiseReprojectBindGroup`] for
/// every view with resident SSR textures and resident temporal planes.
///
/// Must run after `prepare_spec_denoise_temporal_resources` (which allocates the
/// persistent + transient planes and stashes the reprojection transforms).
/// Uploads the per-view config uniform via the render queue; the `wgpu` bind
/// group keeps the buffer alive after the local handle drops.
pub(crate) fn prepare_spec_denoise_reproject_bind_groups(
    mut commands: Commands,
    pipeline: Res<SpecDenoiseReprojectPipeline>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    views: Query<(Entity, &ViewSsrTextures, &ViewSpecDenoiseTemporal)>,
) {
    for (entity, ssr, temporal) in &views {
        let config = reproject_config(
            temporal.world_from_clip,
            temporal.prev_clip_from_world,
            temporal.prev_world_from_clip,
            temporal.world_from_view,
            temporal.curr_cam_pos,
            temporal.prev_cam_pos,
            temporal.size,
        );

        let config_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("prism spec_denoise reproject config"),
            size: size_of::<GpuSpecDenoiseReprojectConfig>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&config_buffer, 0, bytemuck::bytes_of(&config));

        // Sequential group 0 per the frozen reproject contract: config (0), SSR
        // normal/roughness (1), SSR hit (2), reverse-Z depth (3), prev history
        // (4), prev depth (5), prev meta (6), prev luma (7), reprojected (8),
        // reprojected meta (9), reprojected luma (10).
        let group = device.create_bind_group(
            "prism spec_denoise reproject",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                config_buffer.as_entire_binding(),
                ssr.normal_roughness_view(),
                ssr.ssr_hit_view(),
                ssr.scene_depth_sampled(),
                temporal.prev_history_view(),
                temporal.prev_depth_view(),
                temporal.prev_meta_view(),
                temporal.prev_luma_view(),
                temporal.reprojected_view(),
                temporal.reprojected_meta_view(),
                temporal.reprojected_luma_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewSpecDenoiseReprojectBindGroup { group });
    }
}

/// `PrepareBindGroups` system building [`ViewSpecDenoiseHistoryClampBindGroup`]
/// for every view with resident SSR textures, resident `spec_gi` reuse
/// resources and resident temporal planes.
///
/// Must run after `prepare_spec_denoise_reproject_bind_groups` (so the
/// reprojected planes it reads are this frame's) and after
/// `prepare_spec_gi_reuse_bind_groups` (so the resolve it clamps against is this
/// frame's).
pub(crate) fn prepare_spec_denoise_history_clamp_bind_groups(
    mut commands: Commands,
    pipeline: Res<SpecDenoiseHistoryClampPipeline>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    views: Query<(
        Entity,
        &ViewSsrTextures,
        &ViewSpecGiReuse,
        &ViewSpecDenoiseTemporal,
    )>,
) {
    for (entity, ssr, spec_gi, temporal) in &views {
        let config = history_clamp_config(temporal.size);

        let config_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("prism spec_denoise history_clamp config"),
            size: size_of::<GpuSpecDenoiseHistoryClampConfig>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&config_buffer, 0, bytemuck::bytes_of(&config));

        // Sequential group 0 per the frozen history-clamp contract: config (0),
        // reprojected history (1), reprojected meta (2), reprojected luma (3),
        // `spec_gi` resolve (4), SSR normal/roughness (5), reverse-Z depth (6),
        // out history (7), out meta (8), out luma (9), denoised (10).
        let group = device.create_bind_group(
            "prism spec_denoise history_clamp",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                config_buffer.as_entire_binding(),
                temporal.reprojected_view(),
                temporal.reprojected_meta_view(),
                temporal.reprojected_luma_view(),
                spec_gi.resolved_view(),
                ssr.normal_roughness_view(),
                ssr.scene_depth_sampled(),
                temporal.out_history_view(),
                temporal.out_meta_view(),
                temporal.out_luma_view(),
                temporal.denoised_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewSpecDenoiseHistoryClampBindGroup { group });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproject_config_maps_extent_matrices_and_hardcoded_params() {
        let world_from_clip = Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let prev_clip_from_world = Mat4::from_scale(bevy_math::Vec3::splat(2.0));
        let prev_world_from_clip = prev_clip_from_world.inverse();
        let world_from_view = Mat4::IDENTITY;
        let cfg = reproject_config(
            world_from_clip,
            prev_clip_from_world,
            prev_world_from_clip,
            world_from_view,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            UVec2::new(1920, 1080),
        );
        assert_eq!((cfg.width, cfg.height), (1920, 1080));
        // Matrices upload column-major, exactly as stashed.
        assert_eq!(cfg.world_from_clip, world_from_clip.to_cols_array());
        assert_eq!(
            cfg.prev_clip_from_world,
            prev_clip_from_world.to_cols_array()
        );
        assert_eq!(
            cfg.prev_world_from_clip,
            prev_world_from_clip.to_cols_array()
        );
        assert_eq!(cfg.world_from_view, world_from_view.to_cols_array());
        assert_eq!(cfg.curr_cam_pos, [1.0, 2.0, 3.0]);
        assert_eq!(cfg.prev_cam_pos, [4.0, 5.0, 6.0]);
        // Hardcoded reprojection tunables match the CPU golden defaults.
        assert_eq!(cfg.parallax_sensitivity, REPROJECT_PARALLAX_SENSITIVITY);
        assert_eq!(cfg.virtual_exponent, REPROJECT_VIRTUAL_EXPONENT);
        assert_eq!(cfg.min_confidence, REPROJECT_MIN_CONFIDENCE);
        assert_eq!(cfg.depth_rejection, REPROJECT_DEPTH_REJECTION);
    }

    #[test]
    fn history_clamp_config_maps_extent_and_hardcoded_params() {
        let cfg = history_clamp_config(UVec2::new(1280, 720));
        assert_eq!((cfg.width, cfg.height), (1280, 720));
        assert_eq!(cfg.clamp_sigma, CLAMP_SIGMA);
        assert_eq!(cfg.lobe_tolerance, CLAMP_LOBE_TOLERANCE);
        assert_eq!(cfg.fast_sensitivity, CLAMP_FAST_SENSITIVITY);
        assert_eq!(cfg.fast_rate, CLAMP_FAST_RATE);
        assert_eq!(cfg.slow_rate, CLAMP_SLOW_RATE);
        assert_eq!(
            (cfg.min_frames, cfg.max_frames),
            (CLAMP_MIN_FRAMES, CLAMP_MAX_FRAMES)
        );
        // Pad words stay zero so the WGSL uniform stride matches.
        assert_eq!((cfg._pad0, cfg._pad1, cfg._pad2), (0, 0, 0));
    }

    #[test]
    fn hardcoded_reproject_params_match_cpu_golden_defaults() {
        // Pin the three CPU-golden-shared knobs so a drift in either side is a
        // build-time failure rather than a silent parity break.
        let golden = prism_render_shading::gi::spec_denoise::reproject::ReprojectParams::default();
        assert_eq!(golden.parallax_sensitivity, REPROJECT_PARALLAX_SENSITIVITY);
        assert_eq!(golden.virtual_exponent, REPROJECT_VIRTUAL_EXPONENT);
        assert_eq!(golden.min_confidence, REPROJECT_MIN_CONFIDENCE);
    }

    #[test]
    fn hardcoded_clamp_params_match_cpu_golden_defaults() {
        let golden =
            prism_render_shading::gi::spec_denoise::history_clamp::HistoryClampParams::default();
        assert_eq!(golden.clamp_sigma, CLAMP_SIGMA);
        assert_eq!(golden.lobe_tolerance, CLAMP_LOBE_TOLERANCE);
        assert_eq!(golden.fast_sensitivity, CLAMP_FAST_SENSITIVITY);
        assert_eq!(golden.min_frames, CLAMP_MIN_FRAMES);
        assert_eq!(golden.max_frames, CLAMP_MAX_FRAMES);
    }
}
