//! `PrepareBindGroups` system building the specular-GI *spatial denoise* pass's
//! group-0 bind group per view, plus the per-frame config-uniform upload.
//!
//! The spatial dispatch reads six group-0 bindings (mirroring the frozen
//! `spec_denoise_spatial.wesl` contract): the 112-byte
//! [`GpuSpecDenoiseSpatialConfig`] uniform this system uploads (0), the
//! `spec_gi` resolved specular+confidence estimate it filters (1), the SSR
//! trace's per-pixel world-space hit distance (2), the packed `normal_roughness`
//! G-buffer (3), the SSR prepass reverse-Z `scene_depth` (4) and the write-only
//! filtered target (5).
//!
//! Like [`super::super::spec_gi::bind_groups`], the kernel reads its config from
//! a *bound uniform*, so this system allocates a small per-view uniform buffer,
//! writes the frozen 112-byte config into it via the render queue, and binds it
//! at `@binding(0)`. The spatial filter parameters are hardcoded from the CPU
//! golden's `SpatialParams` defaults (there is no per-view runtime tunable for
//! them yet); the near plane is recovered from the inverse projection exactly as
//! the `spec_gi` reuse pass does so both rebuild view-space positions from the
//! shared SSR depth identically.
//!
//! The bind group is present exactly when the view has resident SSR textures,
//! resident `spec_gi` reuse resources (its filter input) and a resident
//! spatial-denoise target; the subsystem's gate guarantees the three appear and
//! disappear together.

use bevy_ecs::prelude::*;
use bevy_math::{Mat4, UVec2, Vec4};
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries, BufferDescriptor, BufferUsages},
    renderer::{RenderDevice, RenderQueue},
    view::ExtractedView,
};

use super::super::spec_gi::ViewSpecGiReuse;
use super::super::ssr::ViewSsrTextures;
use super::abi::GpuSpecDenoiseSpatialConfig;
use super::pipeline::SpecDenoiseSpatialPipeline;
use super::resources::ViewSpecDenoise;
use super::temporal_resources::ViewSpecDenoiseTemporal;

/// Spatial filter parameters, mirroring `prism_render_shading::gi::spec_denoise`
/// `SpatialParams::default()`. Hardcoded here because there is no per-view
/// runtime tunable for the specular denoiser yet; the WESL `sds_*` library and
/// the CPU golden both read these exact values.
const SPATIAL_MAX_RADIUS: f32 = 32.0;
const SPATIAL_PHI_DEPTH: f32 = 0.5;
const SPATIAL_PHI_NORMAL: f32 = 128.0;
const SPATIAL_PHI_ROUGHNESS: f32 = 0.08;
const SPATIAL_CONTACT_HARDENING: f32 = 1.0;
const SPATIAL_MAX_ANISOTROPY: f32 = 3.0;

/// Builds the spatial dispatch's config uniform from the view's inverse
/// projection and framebuffer extent, folding in the hardcoded spatial filter
/// parameters.
///
/// Factored out of the ECS system so the inverse-projection near-plane recovery
/// (the only non-trivial maths here) is unit-testable without a render world.
fn spatial_config(clip_from_view: Mat4, size: UVec2) -> GpuSpecDenoiseSpatialConfig {
    let view_from_clip = clip_from_view.inverse();
    // Recover the positive near-plane distance from the inverse projection: in
    // Prism's reverse-Z convention device depth 1.0 is the near plane, so
    // inverse-projecting clip (0, 0, 1, 1) yields a view-space point at `-near`
    // along the camera's `-Z`. Robust for both finite and infinite reverse-Z.
    // Copied verbatim from `spec_gi::bind_groups::reuse_config` so both passes
    // rebuild view-space positions from the shared SSR depth identically.
    let near_view = view_from_clip * Vec4::new(0.0, 0.0, 1.0, 1.0);
    let near = if near_view.w.abs() > f32::EPSILON {
        (near_view.z / near_view.w).abs().max(1.0e-3)
    } else {
        1.0e-3
    };

    GpuSpecDenoiseSpatialConfig::new(
        view_from_clip,
        size.x,
        size.y,
        SPATIAL_MAX_RADIUS,
        SPATIAL_PHI_DEPTH,
        SPATIAL_PHI_NORMAL,
        SPATIAL_PHI_ROUGHNESS,
        SPATIAL_CONTACT_HARDENING,
        SPATIAL_MAX_ANISOTROPY,
        near,
    )
}

/// The spatial dispatch's group-0 bind group for a single view. Present only
/// when the view has resident SSR textures, resident `spec_gi` reuse resources
/// and a resident spatial-denoise target.
#[derive(Component)]
pub(crate) struct ViewSpecDenoiseBindGroup {
    /// group 0 for `spec_denoise_spatial`: the config uniform (0), the `spec_gi`
    /// resolve (1), the SSR hit distance (2), the packed normal/roughness (3),
    /// the reverse-Z depth (4) and the write-only filtered target (5).
    group: BindGroup,
}

impl ViewSpecDenoiseBindGroup {
    /// group-0 bind group the dispatch node records against.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewSpecDenoiseBindGroup`] for every
/// view that has resident SSR textures, resident `spec_gi` reuse resources and a
/// resident spatial-denoise target.
///
/// Must run after `prepare_spec_denoise_resources` (which allocates the filtered
/// target) and after `prepare_spec_gi_reuse_bind_groups` (so the `spec_gi`
/// resolve it reads is this frame's). Uploads the per-view config uniform via
/// the render queue; the `wgpu` bind group keeps the buffer alive after the
/// local handle drops.
pub(crate) fn prepare_spec_denoise_bind_groups(
    mut commands: Commands,
    pipeline: Res<SpecDenoiseSpatialPipeline>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    views: Query<(
        Entity,
        &ViewSsrTextures,
        &ViewSpecGiReuse,
        &ViewSpecDenoise,
        &ExtractedView,
        Option<&ViewSpecDenoiseTemporal>,
    )>,
) {
    for (entity, ssr, spec_gi, denoise, extracted, temporal) in &views {
        let config = spatial_config(extracted.clip_from_view, denoise.size);

        // Per-view config uniform. `COPY_DST` so the queue write below lands;
        // the bind group retains the buffer past this handle's scope.
        let config_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("prism spec_denoise spatial config"),
            size: size_of::<GpuSpecDenoiseSpatialConfig>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&config_buffer, 0, bytemuck::bytes_of(&config));

        // Binding 1 is the specular estimate the spatial filter cleans. When the
        // temporal accumulator ran (its denoised plane is resident), filter its
        // anti-ghosted output; otherwise fall back to the raw `spec_gi` resolve
        // so the spatial pass still runs standalone before the temporal path is
        // gated in.
        let filter_input = temporal
            .map(|t| t.denoised_view())
            .unwrap_or_else(|| spec_gi.resolved_view());

        // Sequential group 0: config uniform (0), the `spec_gi` resolved
        // specular+confidence estimate (1), the SSR trace's per-pixel world-space
        // hit distance (2), the packed normal/roughness G-buffer (3), the SSR
        // reverse-Z depth (4) and the write-only filtered target (5).
        let group = device.create_bind_group(
            "prism spec_denoise spatial",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                config_buffer.as_entire_binding(),
                filter_input,
                ssr.ssr_hit_view(),
                ssr.normal_roughness_view(),
                ssr.scene_depth_sampled(),
                denoise.filtered_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewSpecDenoiseBindGroup { group });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Mat4;

    /// Prism reverse-Z infinite perspective (depth 1.0 = near): the standard
    /// `clip_from_view` the SSR prepass and this pass share. `near` is recovered
    /// from its inverse, so the config build must round-trip it. Copied from
    /// `spec_gi::bind_groups`' harness so both passes test the identical maths.
    fn reverse_z_infinite(near: f32, aspect: f32, fov_y: f32) -> Mat4 {
        let f = 1.0 / (fov_y * 0.5).tan();
        Mat4::from_cols_array(&[
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            near,
            0.0,
        ])
    }

    #[test]
    fn config_recovers_near_plane_from_reverse_z_projection() {
        let near = 0.1;
        let clip_from_view = reverse_z_infinite(near, 16.0 / 9.0, core::f32::consts::FRAC_PI_2);
        let cfg = spatial_config(clip_from_view, UVec2::new(1920, 1080));
        assert!(
            (cfg.near - near).abs() < 1.0e-4,
            "recovered near {} should match projection near {near}",
            cfg.near
        );
    }

    #[test]
    fn config_maps_extent_and_hardcoded_spatial_params() {
        let clip_from_view = reverse_z_infinite(0.25, 1.0, core::f32::consts::FRAC_PI_3);
        let cfg = spatial_config(clip_from_view, UVec2::new(1280, 720));
        assert_eq!((cfg.width, cfg.height), (1280, 720));
        // The spatial filter parameters match the CPU golden's defaults.
        assert_eq!(cfg.max_radius, SPATIAL_MAX_RADIUS);
        assert_eq!(cfg.phi_depth, SPATIAL_PHI_DEPTH);
        assert_eq!(cfg.phi_normal, SPATIAL_PHI_NORMAL);
        assert_eq!(cfg.phi_roughness, SPATIAL_PHI_ROUGHNESS);
        assert_eq!(cfg.contact_hardening, SPATIAL_CONTACT_HARDENING);
        assert_eq!(cfg.max_anisotropy, SPATIAL_MAX_ANISOTROPY);
        // The uploaded matrix is exactly the inverse projection, column-major.
        assert_eq!(cfg.view_from_clip, clip_from_view.inverse().to_cols_array());
    }
}
