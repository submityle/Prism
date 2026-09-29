//! The `Core3d` scheduling-system pass recording one frame of volumetric-cloud
//! compute.
//!
//! This node is a *dumb executor*: every ordering, tiling and dispatch-domain
//! decision was already made — and unit-tested — in the float-free golden
//! contract [`prism_render_architecture::volumetric::gpu::kernels`]. It walks
//! the eight kernels in [`VolumetricKernel::ALL`] frame order and, for each,
//! reads the kernel's [`KernelDescriptor::workgroup`] tile and
//! [`KernelDescriptor::domain`] shape, sizes the launch against the resident
//! extent with [`linear_group_count`], binds the matching group, pushes the
//! kernel's `var<immediate>` block, and records the dispatch.
//!
//! The five view-independent domain kernels (weather advection, the
//! Perlin-Worley bake, the density-field modelling, the multi-scatter `LUT` bake
//! and the light-space shadow march) are recorded once against the shared
//! [`VolumetricCloudDomain`]; the three per-view kernels (ray-march, scatter
//! resolve, upsample) are recorded once per view carrying its
//! [`ViewVolumetricClouds`] screen targets. Walking [`VolumetricKernel::ALL`]
//! keeps the shadow march after the ray-march, so the ray-march samples the
//! previous frame's resident cloud-shadow map while this frame's is being
//! rebuilt — the deliberate one-frame cloud-shadow latency, consistent with the
//! single-buffered shadow texture.
//!
//! Each dispatch is recorded in its own `begin_compute_pass` block so the
//! implicit storage-write → sampled-read barrier between passes orders every
//! consumer after its producer (modelling's density write before the
//! ray-march / shadow read, the scatter write before the upsample read, and so
//! on). Readiness is all-or-nothing: if any of the eight pipelines is still
//! compiling this frame the node records nothing rather than a partial,
//! non-deterministic frame. Gated on the resident [`VolumetricCloudDomain`] and
//! its bind groups, both of which the prepare/bind-group steps only produce
//! when the clouds are enabled, so a disabled frame records nothing.

use bevy_ecs::prelude::*;
use bevy_math::{UVec2, UVec3};
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::RenderContext,
};
use prism_render_architecture::volumetric::gpu::kernels::{
    linear_group_count, VolumetricKernel, WorkgroupSize,
};

use super::bind_groups::{ViewVolumetricCloudBindGroups, VolumetricCloudDomainBindGroups};
use super::pipeline::VolumetricCloudPipelines;
use super::resources::{ViewVolumetricClouds, VolumetricCloudDomain};

/// Per-axis workgroup counts covering `extent` at the kernel's `workgroup`
/// tile, via the golden saturating ceil-div. A 2D domain passes `extent.z == 1`
/// against a `z == 1` tile so the Z group count is `1`.
fn group_count(extent: UVec3, workgroup: WorkgroupSize) -> (u32, u32, u32) {
    (
        linear_group_count(extent.x, workgroup.x),
        linear_group_count(extent.y, workgroup.y),
        linear_group_count(extent.z, workgroup.z),
    )
}

/// The 2D→3D dispatch extent for a view-independent domain kernel.
fn domain_extent(kernel: VolumetricKernel, domain: &VolumetricCloudDomain) -> UVec3 {
    let as_3d = |d: UVec2| UVec3::new(d.x, d.y, 1);
    match kernel {
        VolumetricKernel::WeatherAdvect => as_3d(domain.weather_dim()),
        VolumetricKernel::NoiseBake | VolumetricKernel::Modeling => domain.density_dim(),
        VolumetricKernel::MultiscatterLutBake => domain.mslut_dim(),
        VolumetricKernel::ShadowMarch => as_3d(domain.shadow_dim()),
        VolumetricKernel::Raymarch
        | VolumetricKernel::ScatterResolve
        | VolumetricKernel::Upsample => UVec3::ZERO,
    }
}

/// The `var<immediate>` push-constant bytes for a view-independent domain
/// kernel.
fn domain_immediate<'a>(kernel: VolumetricKernel, domain: &'a VolumetricCloudDomain) -> &'a [u8] {
    match kernel {
        VolumetricKernel::WeatherAdvect => bytemuck::bytes_of(&domain.weather_advect_params),
        VolumetricKernel::NoiseBake => bytemuck::bytes_of(&domain.noise_bake_params),
        VolumetricKernel::Modeling => bytemuck::bytes_of(&domain.modeling_params),
        VolumetricKernel::MultiscatterLutBake => bytemuck::bytes_of(&domain.ms_lut_params),
        VolumetricKernel::ShadowMarch => bytemuck::bytes_of(&domain.shadow_march_params),
        VolumetricKernel::Raymarch
        | VolumetricKernel::ScatterResolve
        | VolumetricKernel::Upsample => &[],
    }
}

/// The 2D→3D dispatch extent for a per-view kernel.
fn view_extent(kernel: VolumetricKernel, view: &ViewVolumetricClouds) -> UVec3 {
    let as_3d = |d: UVec2| UVec3::new(d.x, d.y, 1);
    match kernel {
        VolumetricKernel::Raymarch | VolumetricKernel::ScatterResolve => as_3d(view.lowres_size),
        VolumetricKernel::Upsample => as_3d(view.full_size),
        VolumetricKernel::WeatherAdvect
        | VolumetricKernel::NoiseBake
        | VolumetricKernel::Modeling
        | VolumetricKernel::MultiscatterLutBake
        | VolumetricKernel::ShadowMarch => UVec3::ZERO,
    }
}

/// The `var<immediate>` push-constant bytes for a per-view kernel.
fn view_immediate<'a>(kernel: VolumetricKernel, view: &'a ViewVolumetricClouds) -> &'a [u8] {
    match kernel {
        VolumetricKernel::Raymarch => bytemuck::bytes_of(&view.raymarch_params),
        VolumetricKernel::ScatterResolve => bytemuck::bytes_of(&view.scatter_resolve_params),
        VolumetricKernel::Upsample => bytemuck::bytes_of(&view.upsample_params),
        VolumetricKernel::WeatherAdvect
        | VolumetricKernel::NoiseBake
        | VolumetricKernel::Modeling
        | VolumetricKernel::MultiscatterLutBake
        | VolumetricKernel::ShadowMarch => &[],
    }
}

/// `Core3d` scheduling-system pass recording the eight volumetric-cloud
/// dispatches in [`VolumetricKernel::ALL`] frame order — the five domain passes
/// once and the three per-view passes for every sized view. No-ops when the
/// clouds are disabled (no resident domain) or while any pipeline is still
/// compiling.
pub(crate) fn dispatch_volumetric_clouds(
    pipelines: Res<VolumetricCloudPipelines>,
    cache: Res<PipelineCache>,
    domain: Option<Res<VolumetricCloudDomain>>,
    domain_groups: Option<Res<VolumetricCloudDomainBindGroups>>,
    views: Query<(&ViewVolumetricClouds, &ViewVolumetricCloudBindGroups)>,
    mut ctx: RenderContext,
) {
    let (Some(domain), Some(domain_groups)) = (domain, domain_groups) else {
        return;
    };

    // All-or-nothing readiness: bail before opening any pass if a kernel is
    // still compiling, so a frame never records a partial (non-deterministic)
    // cloud solve. Every view shares this one pipeline set.
    for kernel in VolumetricKernel::ALL {
        if cache
            .get_compute_pipeline(pipelines.pipeline(kernel))
            .is_none()
        {
            return;
        }
    }

    // Snapshot the view targets + bind groups before borrowing the encoder so
    // the per-view passes can be recorded without holding the query across the
    // mutable `RenderContext` borrow.
    let views: Vec<(&ViewVolumetricClouds, &ViewVolumetricCloudBindGroups)> =
        views.iter().collect();

    let encoder = ctx.command_encoder();

    for kernel in VolumetricKernel::ALL {
        let workgroup = kernel.descriptor().workgroup;
        // Readiness was gated above and the pipeline set is shared by every
        // pass, so the lookup is guaranteed to resolve.
        let pipeline = cache
            .get_compute_pipeline(pipelines.pipeline(kernel))
            .expect("volumetric-cloud pipelines were all checked ready above");

        match domain_groups.group(kernel) {
            // A view-independent domain pass: record exactly once.
            Some(group) => {
                let (gx, gy, gz) = group_count(domain_extent(kernel, &domain), workgroup);
                if gx == 0 || gy == 0 || gz == 0 {
                    continue;
                }
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("prism volumetric-cloud domain pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.set_immediates(0, domain_immediate(kernel, &domain));
                pass.dispatch_workgroups(gx, gy, gz);
            }
            // A per-view pass: record once per sized view.
            None => {
                for (view, groups) in &views {
                    let Some(group) = groups.group(kernel) else {
                        continue;
                    };
                    let (gx, gy, gz) = group_count(view_extent(kernel, view), workgroup);
                    if gx == 0 || gy == 0 || gz == 0 {
                        continue;
                    }
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("prism volumetric-cloud view pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, group, &[]);
                    pass.set_immediates(0, view_immediate(kernel, view));
                    pass.dispatch_workgroups(gx, gy, gz);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-axis group count is the golden saturating ceil-div on every axis,
    /// so a partially filled edge workgroup still launches and an empty extent
    /// launches nothing.
    #[test]
    fn group_count_is_per_axis_ceil_div() {
        let tile = WorkgroupSize { x: 8, y: 8, z: 1 };
        assert_eq!(group_count(UVec3::new(16, 16, 1), tile), (2, 2, 1));
        assert_eq!(group_count(UVec3::new(17, 1, 1), tile), (3, 1, 1));
        assert_eq!(group_count(UVec3::new(0, 8, 1), tile), (0, 1, 1));

        let brick = WorkgroupSize { x: 4, y: 4, z: 4 };
        assert_eq!(group_count(UVec3::new(8, 8, 8), brick), (2, 2, 2));
        assert_eq!(group_count(UVec3::new(9, 4, 3), brick), (3, 1, 1));
    }

    /// Exactly the five view-independent kernels carry a domain extent + a
    /// domain immediate; the three per-view kernels carry a view extent + a view
    /// immediate. The two sets partition [`VolumetricKernel::ALL`], so the
    /// dispatch loop never records a pass with an empty binding.
    #[test]
    fn domain_and_view_kernels_partition_all() {
        let mut domain_kernels = 0;
        let mut view_kernels = 0;
        for kernel in VolumetricKernel::ALL {
            let is_domain = matches!(
                kernel,
                VolumetricKernel::WeatherAdvect
                    | VolumetricKernel::NoiseBake
                    | VolumetricKernel::Modeling
                    | VolumetricKernel::MultiscatterLutBake
                    | VolumetricKernel::ShadowMarch
            );
            if is_domain {
                domain_kernels += 1;
            } else {
                view_kernels += 1;
            }
        }
        assert_eq!(domain_kernels, 5);
        assert_eq!(view_kernels, 3);
    }
}
