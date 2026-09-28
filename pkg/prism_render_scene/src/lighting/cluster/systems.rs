//! Render-schedule systems that build, upload, and bind the clustered tables.
//!
//! The build step is where the render world meets the CPU golden: it feeds the
//! extracted punctual lights and the active camera fit into
//! [`build_cluster_data`], then stages the result into the GPU buffers.  The
//! upload and bind systems mirror the sibling light buffers' lifecycle exactly.

use bevy_ecs::prelude::*;
use bevy_render::renderer::{RenderDevice, RenderQueue};

use crate::lighting::extract::ExtractedLights;

use super::bindings::ClusterBindGroup;
use super::buffers::ClusterGpuBuffers;
use super::build::{build_cluster_data, ClusterConfig};
use super::extract::ExtractedClusterView;

/// Builds the clustered tables for the active view and stages them.
///
/// Runs in `PrepareResources`.  When no perspective camera is active this frame
/// (or the light assignment cannot be built), the buffers fall back to a
/// neutral single-cluster grid so the bindings stay valid.
pub(crate) fn rebuild_cluster_buffers(
    lights: Res<ExtractedLights>,
    view: Res<ExtractedClusterView>,
    config: Res<ClusterConfig>,
    mut buffers: ResMut<ClusterGpuBuffers>,
) {
    let data = view
        .view
        .and_then(|fit| {
            build_cluster_data(
                &lights.punctuals,
                &fit.view_from_world,
                &fit.clip_from_view,
                fit.screen_size,
                fit.near,
                fit.far,
                *config,
            )
        })
        .unwrap_or_default();
    buffers.rebuild(&data);
}

/// Flushes the staged clustered arrays to the GPU.
pub(crate) fn write_cluster_buffers(
    mut buffers: ResMut<ClusterGpuBuffers>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    buffers.upload(&device, &queue);
}

/// (Re)builds the clustered bind group once the buffers have uploaded.
pub(crate) fn prepare_cluster_bind_group(
    buffers: Res<ClusterGpuBuffers>,
    mut bindings: ResMut<ClusterBindGroup>,
    device: Res<RenderDevice>,
) {
    bindings.prepare(&device, &buffers);
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::build::ClusterCpuData;
    use crate::lighting::abi::GpuPunctualLight;
    use crate::lighting::cluster::extract::ClusterViewFit;
    use prism_render_shading::PunctualLight;

    fn identity() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn perspective() -> [f32; 16] {
        let near = 0.1;
        let far = 100.0;
        let h = 1.0 / bevy_math::ops::tan(0.5);
        let r = far / (near - far);
        [
            h, 0.0, 0.0, 0.0, //
            0.0, h, 0.0, 0.0, //
            0.0, 0.0, r, -1.0, //
            0.0, 0.0, r * near, 0.0,
        ]
    }

    #[test]
    fn missing_view_falls_back_to_a_single_cluster() {
        // Directly exercise the build fallback the system relies on.
        let lights = ExtractedLights::default();
        let view = ExtractedClusterView::default();
        let config = ClusterConfig::default();
        let data = view
            .view
            .and_then(|fit| {
                build_cluster_data(
                    &lights.punctuals,
                    &fit.view_from_world,
                    &fit.clip_from_view,
                    fit.screen_size,
                    fit.near,
                    fit.far,
                    config,
                )
            })
            .unwrap_or_default();
        assert_eq!(data, ClusterCpuData::default());
    }

    #[test]
    fn present_view_builds_a_full_grid_from_the_lights() {
        let mut lights = ExtractedLights::default();
        lights
            .punctuals
            .push(GpuPunctualLight::from(PunctualLight::point(
                [0.0, 0.0, -2.0],
                [40.0; 3],
                8.0,
            )));
        let fit = ClusterViewFit {
            view_from_world: identity(),
            clip_from_view: perspective(),
            screen_size: [256, 256],
            near: 0.1,
            far: 100.0,
        };
        let config = ClusterConfig::default();
        let data = build_cluster_data(
            &lights.punctuals,
            &fit.view_from_world,
            &fit.clip_from_view,
            fit.screen_size,
            fit.near,
            fit.far,
            config,
        )
        .expect("finite projection");
        assert_eq!(data.cluster_count(), data.grid.cluster_count as usize);
        assert!(data.index_count() > 0, "the light must touch a cluster");
    }
}
