//! Host orchestration of the Radiance Cascades `gather`/`merge`/`resolve`
//! kernels and the typed buffers they exchange.
//!
//! [`GpuRadianceCascades`] compiles the three `shaders/*.wgsl` kernels once and
//! drives a full device solve: it gathers every cascade from the pure-rational
//! scene sampler, folds the hierarchy top-down with the bilinear/angular merge,
//! and resolves cascade 0 into per-probe mean radiance. Every stage reproduces
//! the [`prism_render_architecture`] CPU golden
//! ([`radiance_cascades`](prism_render_architecture::lighting::radiance_cascades))
//! operation-for-operation, with directions precomputed on the host so the two
//! sides consume byte-identical direction vectors.
//!
//! The three kernels share one bind-group signature — a uniform parameter
//! block, a read-only storage input, and a read-write storage output — so a
//! single [`wgpu::BindGroupLayout`] backs all of them.
//!
//! Provenance: Sannikov, *Radiance Cascades* (2023); Porter-Duff "over" (1984).
//! Standard `wgpu` compute orchestration. No Unreal Engine source or derived
//! code.

use core::f32::consts::TAU;

use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::lighting::radiance_cascades::CascadeHierarchy;

use crate::buffer;
use crate::context::GpuContext;

/// A single `probes × directions` radiance interval, matching the WGSL
/// `Interval` struct (`vec3` radiance + scalar transmittance, 16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuInterval {
    /// Premultiplied radiance accumulated across the interval.
    pub radiance: [f32; 3],
    /// Fraction of light surviving the interval, in `[0, 1]`.
    pub transmittance: f32,
}

/// Per-probe mean radiance written by the resolve kernel (`vec3` + pad).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuMean {
    value: [f32; 3],
    pad: f32,
}

/// Uniform block shared with `GatherParams` in `gather.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GatherParams {
    n: u32,
    cols: u32,
    rows: u32,
    angular: u32,
    origin_x: f32,
    origin_y: f32,
    spacing: f32,
    t0: f32,
    t1: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// Uniform block shared with `MergeParams` in `merge.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MergeParams {
    n: u32,
    cols: u32,
    rows: u32,
    angular: u32,
    pcols: u32,
    prows: u32,
    pangular: u32,
    pad0: u32,
    origin_x: f32,
    origin_y: f32,
    child_spacing: f32,
    parent_spacing: f32,
}

/// Uniform block shared with `ResolveParams` in `resolve.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResolveParams {
    n: u32,
    angular: u32,
    pad0: u32,
    pad1: u32,
}

/// Workgroup size shared by all three kernels (`@workgroup_size(256)`).
const WORKGROUP: u32 = 256;

/// Number of workgroups needed to cover `n` invocations.
fn groups(n: u32) -> u32 {
    n.div_ceil(WORKGROUP).max(1)
}

/// Host twin of the golden's `bin_angle` + `Vec2::from_angle`, producing the
/// exact direction vectors cascade `level` fans out.
///
/// Computed with the same `glam` build as the golden so the uploaded directions
/// are byte-identical to the ones `Cascade::gather` would evaluate internally.
#[must_use]
fn directions_for(hierarchy: &CascadeHierarchy, level: u32) -> Vec<[f32; 2]> {
    let count = hierarchy.angular_count(level);
    (0..count)
        .map(|dir| {
            let angle = TAU * (dir as f32 + 0.5) / (count as f32);
            let d = Vec2::from_angle(angle);
            [d.x, d.y]
        })
        .collect()
}

/// A compiled, reusable Radiance Cascades `GPU` pipeline set.
pub struct GpuRadianceCascades {
    /// Kept alive so the gather pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    gather_module: ShaderModule,
    /// Kept alive so the merge pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    merge_module: ShaderModule,
    /// Kept alive so the resolve pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    resolve_module: ShaderModule,
    /// Shared `[uniform, storage-read, storage-rw]` layout for all three.
    layout: BindGroupLayout,
    /// Fills one cascade's radiance intervals from the scene sampler.
    gather: ComputePipeline,
    /// Composites a coarse parent cascade into its finer child.
    merge: ComputePipeline,
    /// Integrates cascade 0 over direction into per-probe mean radiance.
    resolve: ComputePipeline,
}

impl GpuRadianceCascades {
    /// Compiles the gather, merge, and resolve kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRadianceCascades {
        let device = ctx.device();

        let gather_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_rc_gather"),
            source: ShaderSource::Wgsl(include_str!("../shaders/gather.wgsl").into()),
        });
        let merge_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_rc_merge"),
            source: ShaderSource::Wgsl(include_str!("../shaders/merge.wgsl").into()),
        });
        let resolve_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_rc_resolve"),
            source: ShaderSource::Wgsl(include_str!("../shaders/resolve.wgsl").into()),
        });

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_rc_layout"),
            entries: &[
                buffer_layout(0, BufferBindingType::Uniform),
                buffer_layout(1, BufferBindingType::Storage { read_only: true }),
                buffer_layout(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_rc_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let make = |module: &ShaderModule, label: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let gather = make(&gather_module, "prism_rc_gather_pipeline");
        let merge = make(&merge_module, "prism_rc_merge_pipeline");
        let resolve = make(&resolve_module, "prism_rc_resolve_pipeline");

        GpuRadianceCascades {
            gather_module,
            merge_module,
            resolve_module,
            layout,
            gather,
            merge,
            resolve,
        }
    }

    /// Gathers cascade `level` on the device and returns its radiance intervals
    /// in probe-major (`((row*cols)+col)*angular + dir`) order.
    #[must_use]
    pub fn gather(
        &self,
        ctx: &GpuContext,
        hierarchy: &CascadeHierarchy,
        level: u32,
    ) -> Vec<GpuInterval> {
        let device = ctx.device();
        let out = self.gather_buffer(ctx, hierarchy, level);
        let rays = hierarchy.rays(level);
        let bytes = u64::from(rays) * size_of::<GpuInterval>() as u64;
        let stage = buffer::staging(device, "prism_rc_gather_stage", bytes);
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_rc_gather_readback"),
        });
        buffer::copy(&mut enc, &out, &stage, bytes);
        ctx.queue().submit([enc.finish()]);
        buffer::read_back::<GpuInterval>(ctx, &stage)
    }

    /// Runs the full device solve and returns the merged cascade-0 intervals in
    /// probe-major order (the twin of
    /// [`radiance_cascades::solve`](prism_render_architecture::lighting::radiance_cascades::solve)).
    #[must_use]
    pub fn solve(&self, ctx: &GpuContext, hierarchy: &CascadeHierarchy) -> Vec<GpuInterval> {
        let device = ctx.device();
        let acc = self.solve_buffer(ctx, hierarchy);
        let rays = hierarchy.rays(0);
        let bytes = u64::from(rays) * size_of::<GpuInterval>() as u64;
        let stage = buffer::staging(device, "prism_rc_solve_stage", bytes);
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_rc_solve_readback"),
        });
        buffer::copy(&mut enc, &acc, &stage, bytes);
        ctx.queue().submit([enc.finish()]);
        buffer::read_back::<GpuInterval>(ctx, &stage)
    }

    /// Runs the full solve then resolves cascade 0 into per-probe mean radiance
    /// (the twin of
    /// [`radiance_cascades::resolve::mean_radiance`](prism_render_architecture::lighting::radiance_cascades::resolve::mean_radiance)),
    /// returned in row-major probe order.
    #[must_use]
    pub fn resolve(&self, ctx: &GpuContext, hierarchy: &CascadeHierarchy) -> Vec<[f32; 3]> {
        let device = ctx.device();
        let cascade0 = self.solve_buffer(ctx, hierarchy);
        let (cols, rows) = hierarchy.probe_dims(0);
        let angular = hierarchy.angular_count(0);
        let probes = cols * rows;

        let params = buffer::uniform(
            device,
            "prism_rc_resolve_params",
            &ResolveParams {
                n: probes,
                angular,
                pad0: 0,
                pad1: 0,
            },
        );
        let out = buffer::storage_rw_zeroed(
            device,
            "prism_rc_means",
            u64::from(probes) * size_of::<GpuMean>() as u64,
        );
        let bind = self.bind(device, "prism_rc_resolve_bind", &params, &cascade0, &out);
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_rc_resolve_encoder"),
        });
        dispatch(
            &mut enc,
            "prism_rc_resolve_pass",
            &self.resolve,
            &bind,
            groups(probes),
        );
        let bytes = u64::from(probes) * size_of::<GpuMean>() as u64;
        let stage = buffer::staging(device, "prism_rc_resolve_stage", bytes);
        buffer::copy(&mut enc, &out, &stage, bytes);
        ctx.queue().submit([enc.finish()]);
        buffer::read_back::<GpuMean>(ctx, &stage)
            .into_iter()
            .map(|m| m.value)
            .collect()
    }

    /// Gathers cascade `level` into a fresh device storage buffer.
    fn gather_buffer(&self, ctx: &GpuContext, hierarchy: &CascadeHierarchy, level: u32) -> Buffer {
        let device = ctx.device();
        let (cols, rows) = hierarchy.probe_dims(level);
        let angular = hierarchy.angular_count(level);
        let rays = hierarchy.rays(level);

        let dirs = directions_for(hierarchy, level);
        let dir_buf = buffer::storage_read(device, "prism_rc_directions", &dirs);
        let out = buffer::storage_rw_zeroed(
            device,
            "prism_rc_gather_out",
            u64::from(rays) * size_of::<GpuInterval>() as u64,
        );
        let params = buffer::uniform(
            device,
            "prism_rc_gather_params",
            &GatherParams {
                n: rays,
                cols,
                rows,
                angular,
                origin_x: hierarchy.origin.x,
                origin_y: hierarchy.origin.y,
                spacing: hierarchy.spacing(level),
                t0: hierarchy.interval_start(level),
                t1: hierarchy.interval_end(level),
                pad0: 0.0,
                pad1: 0.0,
                pad2: 0.0,
            },
        );
        let bind = self.bind(device, "prism_rc_gather_bind", &params, &dir_buf, &out);
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_rc_gather_encoder"),
        });
        dispatch(
            &mut enc,
            "prism_rc_gather_pass",
            &self.gather,
            &bind,
            groups(rays),
        );
        ctx.queue().submit([enc.finish()]);
        out
    }

    /// Full device solve: gather the top cascade, then fold each lower cascade
    /// in with a merge dispatch, returning the merged cascade-0 buffer.
    fn solve_buffer(&self, ctx: &GpuContext, hierarchy: &CascadeHierarchy) -> Buffer {
        let device = ctx.device();
        let top = hierarchy.levels.saturating_sub(1);
        let mut acc = self.gather_buffer(ctx, hierarchy, top);
        let mut level = top;
        while level > 0 {
            level -= 1;
            let child = self.gather_buffer(ctx, hierarchy, level);
            self.merge_dispatch(ctx, hierarchy, level, &child, &acc);
            acc = child;
        }
        let _ = device;
        acc
    }

    /// Records and submits one merge of `parent` (cascade `level + 1`) into
    /// `child` (cascade `level`), compositing in place.
    fn merge_dispatch(
        &self,
        ctx: &GpuContext,
        hierarchy: &CascadeHierarchy,
        level: u32,
        child: &Buffer,
        parent: &Buffer,
    ) {
        let device = ctx.device();
        let (cols, rows) = hierarchy.probe_dims(level);
        let angular = hierarchy.angular_count(level);
        let (pcols, prows) = hierarchy.probe_dims(level + 1);
        let pangular = hierarchy.angular_count(level + 1);
        let rays = hierarchy.rays(level);

        let params = buffer::uniform(
            device,
            "prism_rc_merge_params",
            &MergeParams {
                n: rays,
                cols,
                rows,
                angular,
                pcols,
                prows,
                pangular,
                pad0: 0,
                origin_x: hierarchy.origin.x,
                origin_y: hierarchy.origin.y,
                child_spacing: hierarchy.spacing(level),
                parent_spacing: hierarchy.spacing(level + 1),
            },
        );
        let bind = self.bind(device, "prism_rc_merge_bind", &params, parent, child);
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_rc_merge_encoder"),
        });
        dispatch(
            &mut enc,
            "prism_rc_merge_pass",
            &self.merge,
            &bind,
            groups(rays),
        );
        ctx.queue().submit([enc.finish()]);
    }

    /// Builds the shared `[uniform, read, read_write]` bind group.
    fn bind(
        &self,
        device: &wgpu::Device,
        label: &str,
        params: &Buffer,
        input: &Buffer,
        output: &Buffer,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some(label),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        })
    }
}

/// Builds a compute-visible buffer binding layout entry for `binding`.
fn buffer_layout(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Records one 1-D compute dispatch of `pipeline` bound to `bind`.
fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &ComputePipeline,
    bind: &BindGroup,
    group_count: u32,
) {
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[]);
    pass.dispatch_workgroups(group_count, 1, 1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use prism_render_architecture::lighting::radiance_cascades::RadianceInterval;
    use prism_render_architecture::lighting::radiance_cascades::{Cascade, SceneSampler};

    /// CPU twin of the WGSL `sample_interval`: the exact same pure-rational
    /// arithmetic in the exact same operation order, so a device gather can be
    /// compared against a golden `Cascade::gather` driven by this sampler.
    pub(crate) struct RationalMedium;

    impl SceneSampler for RationalMedium {
        fn sample_interval(&self, o: Vec2, d: Vec2, t0: f32, t1: f32) -> RadianceInterval {
            let m = o.x * 0.5 + o.y * 0.25 + d.x * 2.0 - d.y * 1.5 + t0 * 0.1;
            let base = m * m;
            let denom = 1.0 + base;
            let tr_raw = 1.0 / denom;
            let tr = tr_raw.clamp(0.0, 1.0);
            let k = (t1 - t0) * 0.05;
            RadianceInterval::new(Vec3::new(m + k, m * 0.5 + d.x, base * 0.25 + k), tr)
        }
    }

    #[test]
    fn directions_match_the_golden_gather_vectors() {
        let h = CascadeHierarchy {
            origin: Vec2::new(1.0, -2.0),
            base_spacing: 1.5,
            base_cols: 8,
            base_rows: 8,
            base_angular: 4,
            base_interval: 1.0,
            levels: 3,
        };
        for level in 0..h.levels {
            let dirs = directions_for(&h, level);
            for (dir, got) in dirs.iter().enumerate() {
                let expected = Vec2::from_angle(h.bin_angle(level, dir as u32));
                assert_eq!(got[0], expected.x);
                assert_eq!(got[1], expected.y);
            }
        }
    }

    #[test]
    fn groups_cover_every_invocation() {
        assert_eq!(groups(0), 1);
        assert_eq!(groups(1), 1);
        assert_eq!(groups(256), 1);
        assert_eq!(groups(257), 2);
    }

    /// A clear parent cannot survive a device gather driven by `RationalMedium`
    /// unless the golden sampler agrees — a cheap CPU-only self-consistency
    /// check of the twin sampler against the golden's `over` identity.
    #[test]
    fn rational_medium_composites_like_the_golden() {
        let near = RationalMedium.sample_interval(Vec2::new(0.5, 0.5), Vec2::X, 0.0, 1.0);
        let composed = near.over(RadianceInterval::CLEAR);
        assert_eq!(composed, near);
        let _ = Cascade::cleared;
    }
}
