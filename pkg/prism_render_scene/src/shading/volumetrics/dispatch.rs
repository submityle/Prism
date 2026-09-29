//! `Core3d` scheduling-system pass recording the three froxel-fog compute
//! dispatches for every view.
//!
//! Mirrors [`super::super::taa::dispatch`] / [`super::super::virtual_shadow`]'s
//! dispatch node, but records *three* chained compute passes in one command
//! encoder: the scatter pass fills every froxel's medium + source radiance +
//! extinction, the integrate pass marches each column front-to-back into the
//! integrated in-scattering + transmittance volumes, then the apply pass
//! resolves the fog per screen pixel and composites it over the lit scene colour
//! into the `fog_applied` target — which is finally blitted back over
//! `scene_color` so the downstream composite sees the fog. The passes are
//! recorded as separate `begin_compute_pass` blocks so the implicit
//! storage-write / texture-read barriers between them order the integrate reads
//! after the scatter writes and the apply reads after the integrate writes.
//!
//! Every entry bounds-checks its invocations, so a partially-filled edge
//! workgroup is safe. Gated on [`PrismVolumetricsSettings::enabled`]; the
//! per-view components only exist on views the prepare/bind-group steps ran, so
//! a disabled frame records nothing. The apply pass + copy-back only run when
//! the view carries the SSR depth + visibility scene-colour buffers the
//! composite reads (the apply bind group is `Some` exactly then).

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{
        ComputePassDescriptor, Extent3d, Origin3d, PipelineCache, TexelCopyTextureInfo,
        TextureAspect,
    },
    renderer::{RenderContext, ViewQuery},
};

use super::super::resources::ViewVisibilityBuffer;
use super::abi::{
    VOLUMETRICS_APPLY_WORKGROUP_SIZE, VOLUMETRICS_INTEGRATE_WORKGROUP_SIZE,
    VOLUMETRICS_SCATTER_WORKGROUP_SIZE,
};
use super::bind_groups::ViewVolumetricsBindGroups;
use super::pipeline::VolumetricsPipeline;
use super::resources::ViewVolumetrics;
use super::settings::PrismVolumetricsSettings;

/// `Core3d` scheduling-system pass recording the scatter -> integrate -> apply
/// froxel dispatches for every view, then blitting the fog-composited result
/// back over `scene_color`.
pub(crate) fn volumetrics_pass(
    settings: Res<PrismVolumetricsSettings>,
    view: ViewQuery<(
        &ViewVolumetrics,
        &ViewVolumetricsBindGroups,
        Option<&ViewVisibilityBuffer>,
    )>,
    pipeline: Res<VolumetricsPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (volumetrics, groups, visibility) = view.into_inner();

    // The scatter + integrate pipelines must be resident before either pass
    // records; the integrate pass reads what the scatter pass writes, so it is
    // pointless to run one without the other. The apply pipeline is gated
    // separately alongside its bind group + scene-colour target below.
    let (Some(scatter), Some(integrate)) = (
        cache.get_compute_pipeline(pipeline.scatter()),
        cache.get_compute_pipeline(pipeline.integrate()),
    ) else {
        return;
    };

    let grid = volumetrics.grid;
    if grid.x == 0 || grid.y == 0 || grid.z == 0 {
        return;
    }

    let encoder = ctx.command_encoder();

    // Scatter: one invocation per froxel over the full 3D grid.
    {
        let scatter_x = grid.x.div_ceil(VOLUMETRICS_SCATTER_WORKGROUP_SIZE);
        let scatter_y = grid.y.div_ceil(VOLUMETRICS_SCATTER_WORKGROUP_SIZE);
        let scatter_z = grid.z.div_ceil(VOLUMETRICS_SCATTER_WORKGROUP_SIZE);

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism volumetrics scatter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(scatter);
        pass.set_bind_group(0, groups.scatter(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&volumetrics.scatter_params));
        pass.dispatch_workgroups(scatter_x, scatter_y, scatter_z);
    }

    // Integrate: one invocation per froxel *column* (all Z slices), so the grid
    // is dispatched over X/Y only and each invocation marches Z internally.
    {
        let integrate_x = grid.x.div_ceil(VOLUMETRICS_INTEGRATE_WORKGROUP_SIZE);
        let integrate_y = grid.y.div_ceil(VOLUMETRICS_INTEGRATE_WORKGROUP_SIZE);

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism volumetrics integrate"),
            timestamp_writes: None,
        });
        pass.set_pipeline(integrate);
        pass.set_bind_group(0, groups.integrate(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&volumetrics.integrate_params));
        pass.dispatch_workgroups(integrate_x, integrate_y, 1);
    }

    // Apply: one invocation per screen pixel, compositing the fog over the lit
    // scene colour into `fog_applied`, then copying that back over
    // `scene_color`. Only runs when the apply pipeline is resident *and* the
    // apply bind group + visibility scene-colour target exist (both gated on the
    // SSR depth + visibility buffers being present this frame).
    let size = volumetrics.size;
    if size.x == 0 || size.y == 0 {
        return;
    }
    let (Some(apply), Some(apply_group), Some(visibility)) = (
        cache.get_compute_pipeline(pipeline.apply()),
        groups.apply(),
        visibility,
    ) else {
        return;
    };

    {
        let apply_x = size.x.div_ceil(VOLUMETRICS_APPLY_WORKGROUP_SIZE);
        let apply_y = size.y.div_ceil(VOLUMETRICS_APPLY_WORKGROUP_SIZE);

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism volumetrics apply"),
            timestamp_writes: None,
        });
        pass.set_pipeline(apply);
        pass.set_bind_group(0, apply_group, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&volumetrics.apply_params));
        pass.dispatch_workgroups(apply_x, apply_y, 1);
    }

    // Copy the fog-composited result back over `scene_color` so the downstream
    // composite (which samples `scene_color`, not `fog_applied`) reads the
    // fogged image. The apply pass cannot write `scene_color` in place: it
    // samples `scene_color` per pixel, so it composites into a dedicated target
    // and then blits back. Same extent + same format (SCENE_COLOR_FORMAT),
    // single mip/layer.
    encoder.copy_texture_to_texture(
        TexelCopyTextureInfo {
            texture: volumetrics.fog_applied_texture(),
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyTextureInfo {
            texture: visibility.scene_color_texture(),
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
    );
}
