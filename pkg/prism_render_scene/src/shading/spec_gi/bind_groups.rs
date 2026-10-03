//! `PrepareBindGroups` system building the glossy-specular ReSTIR *reuse* pass's
//! group-0 bind group per view, plus the per-frame config-uniform upload.
//!
//! The reuse dispatch reads seven group-0 bindings (mirroring the frozen
//! `spec_gi_reuse.wesl` contract): the 96-byte [`GpuSpecGiReuseConfig`] uniform
//! this system uploads (0), the ping-pong reservoir pair the subsystem owns
//! (prior read-only 1, out read-write 2), the three SSR-rebuilt screen-space
//! reads (reverse-Z `scene_depth` 3, packed `normal_roughness` 4, and the SSR
//! trace's screen-space glossy candidate radiance 5), and the write-only
//! resolved specular + confidence target (6).
//!
//! Unlike [`super::super::ssgi::trace`] and [`super::super::world_restir`] — both
//! of which push their dispatch config in an immediate (push-constant) block —
//! the reuse kernel reads its config from a *bound uniform*, so this system
//! allocates a small per-view uniform buffer, writes the frozen 96-byte
//! [`GpuSpecGiReuseConfig`] into it via the render queue, and binds it at
//! `@binding(0)`. The `wgpu` bind group retains the buffer, so the local handle
//! can drop at the end of the system without freeing the GPU allocation.
//!
//! Like the world-space `ReSTIR` fill group, the bind group is rebuilt every
//! frame because [`ViewSpecGiReuse`]'s ping-pong `src`/`dst` selection flips
//! each frame; it is present exactly when both the view's SSR textures and its
//! resident reuse resources are live (the subsystem's gate guarantees the two
//! appear and disappear together).
//!
//! The temporal reuse toggle is hardcoded on and the spatial toggle off this
//! slice: the kernel reads the reprojected *same-pixel* prior reservoir only.
//! The neighbour (spatial) reuse path and the motion-vector reprojection that
//! generalise this to full screen-space ReSTIR land in a follow-up slice; the
//! WESL already carries the `spatial_enabled` branch so enabling it later is a
//! config flip, not a shader change.

use bevy_ecs::prelude::*;
use bevy_math::{Mat4, UVec2, Vec4};
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries, BufferDescriptor, BufferUsages},
    renderer::{RenderDevice, RenderQueue},
    view::ExtractedView,
};

use super::super::runtime::PrismShadingSettings;
use super::super::ssr::ViewSsrTextures;
use super::abi::GpuSpecGiReuseConfig;
use super::pipeline::SpecGiReusePipeline;
use super::resources::ViewSpecGiReuse;

/// Builds the reuse dispatch's config uniform from the view's inverse
/// projection, framebuffer extent and the runtime reuse tunables.
///
/// Factored out of the ECS system so the inverse-projection near-plane recovery
/// (the only non-trivial maths here) is unit-testable without a render world.
/// `temporal_enabled` is forced on and `spatial_enabled` off: this slice reuses
/// the reprojected same-pixel prior reservoir only (see the module header).
fn reuse_config(
    clip_from_view: Mat4,
    size: UVec2,
    m_cap: f32,
    roughness_cap_base: f32,
    sigma_roughness: f32,
) -> GpuSpecGiReuseConfig {
    let view_from_clip = clip_from_view.inverse();
    // Recover the positive near-plane distance from the inverse projection: in
    // Prism's reverse-Z convention device depth 1.0 is the near plane, so
    // inverse-projecting clip (0, 0, 1, 1) yields a view-space point at `-near`
    // along the camera's `-Z`. Robust for both finite and infinite reverse-Z.
    // Copied verbatim from `ssgi::trace::ssgi_trace_pass` so both passes rebuild
    // view-space positions from the shared SSR depth identically.
    let near_view = view_from_clip * Vec4::new(0.0, 0.0, 1.0, 1.0);
    let near = if near_view.w.abs() > f32::EPSILON {
        (near_view.z / near_view.w).abs().max(1.0e-3)
    } else {
        1.0e-3
    };

    GpuSpecGiReuseConfig::new(
        view_from_clip,
        size.x,
        size.y,
        m_cap,
        roughness_cap_base,
        sigma_roughness,
        near,
        true,
        false,
    )
}

/// The reuse dispatch's group-0 bind group for a single view. Present only when
/// both the view's SSR textures and its resident reuse resources are live.
#[derive(Component)]
#[allow(dead_code)] // consumed by the dispatch node (next slice) once wired in the plugin.
pub(crate) struct ViewSpecGiReuseBindGroup {
    /// group 0 for `spec_gi_reuse`: the config uniform (0), the ping-pong
    /// reservoir pair (prior read-only 1, out read-write 2), the SSR-rebuilt
    /// depth/normal/candidate reads (3-5) and the write-only resolved target
    /// (6). Rebuilt every frame to follow the reservoir ping-pong flip.
    group: BindGroup,
}

impl ViewSpecGiReuseBindGroup {
    /// group-0 bind group the dispatch node records against (next slice).
    #[allow(dead_code)] // recorded by the dispatch node once wired in the plugin.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewSpecGiReuseBindGroup`] for every
/// view that has both resident SSR textures and resident reuse resources.
///
/// Must run after `prepare_spec_gi_reuse_resources` (which advances the
/// ping-pong flip) so the uniform's `width`/`height` and the bound `src`/`dst`
/// reservoirs agree with this frame's resident allocation. The group is rebuilt
/// every frame to follow that flip. Uploads the per-view config uniform via the
/// render queue; the `wgpu` bind group keeps the buffer alive after the local
/// handle drops.
#[allow(dead_code)] // registered in `PrepareBindGroups` by the plugin slice (next).
pub(crate) fn prepare_spec_gi_reuse_bind_groups(
    mut commands: Commands,
    pipeline: Res<SpecGiReusePipeline>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    settings: Res<PrismShadingSettings>,
    views: Query<(Entity, &ViewSsrTextures, &ViewSpecGiReuse, &ExtractedView)>,
) {
    for (entity, ssr, spec_gi, extracted) in &views {
        let config = reuse_config(
            extracted.clip_from_view,
            spec_gi.size,
            settings.spec_gi_temporal_m_cap,
            settings.spec_gi_roughness_cap_base,
            settings.spec_gi_sigma_roughness,
        );

        // Per-view config uniform. `COPY_DST` so the queue write below lands;
        // the bind group retains the buffer past this handle's scope.
        let config_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("prism spec_gi reuse config"),
            size: size_of::<GpuSpecGiReuseConfig>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&config_buffer, 0, bytemuck::bytes_of(&config));

        // Sequential group 0: config uniform (0), prior read-only reservoirs
        // (1), out read-write reservoirs (2), SSR reverse-Z depth (3), packed
        // normal/roughness (4), the SSR trace's screen-space glossy candidate
        // radiance (5), and the write-only resolved specular + confidence (6).
        let group = device.create_bind_group(
            "prism spec_gi reuse",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                config_buffer.as_entire_binding(),
                spec_gi.src_buffer().as_entire_binding(),
                spec_gi.dst_buffer().as_entire_binding(),
                ssr.scene_depth_sampled(),
                ssr.normal_roughness_view(),
                ssr.ssr_out_view(),
                spec_gi.resolved_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewSpecGiReuseBindGroup { group });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Mat4;

    /// Prism reverse-Z infinite perspective (depth 1.0 = near): the standard
    /// `clip_from_view` the SSR prepass and this pass share. `near` is recovered
    /// from its inverse, so the config build must round-trip it.
    fn reverse_z_infinite(near: f32, aspect: f32, fov_y: f32) -> Mat4 {
        let f = 1.0 / (fov_y * 0.5).tan();
        // Column-major; row 2 = (0,0,0,-1) maps clip.w = -view.z, row 3 places
        // the near plane at device depth 1.0 (reverse-Z, infinite far -> 0).
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
        let cfg = reuse_config(clip_from_view, UVec2::new(1920, 1080), 32.0, 32.0, 0.25);
        assert!(
            (cfg.near - near).abs() < 1.0e-4,
            "recovered near {} should match projection near {near}",
            cfg.near
        );
    }

    #[test]
    fn config_maps_extent_and_tunables_and_forces_temporal_only() {
        let clip_from_view = reverse_z_infinite(0.25, 1.0, core::f32::consts::FRAC_PI_3);
        let cfg = reuse_config(clip_from_view, UVec2::new(1280, 720), 24.0, 16.0, 0.3);
        assert_eq!((cfg.width, cfg.height), (1280, 720));
        assert_eq!(cfg.m_cap, 24.0);
        assert_eq!(cfg.roughness_cap_base, 16.0);
        assert_eq!(cfg.sigma_roughness, 0.3);
        // This slice reuses the same-pixel prior reservoir only.
        assert_eq!((cfg.temporal_enabled, cfg.spatial_enabled), (1, 0));
        // The uploaded matrix is exactly the inverse projection, column-major.
        assert_eq!(cfg.view_from_clip, clip_from_view.inverse().to_cols_array());
    }
}
