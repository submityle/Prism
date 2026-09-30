//! Real-device `wgpu` compute implementation of the per-fragment bounds builder.
//!
//! [`GpuFragmentBounds`] compiles `shaders/fracture_bounds.wgsl` once and
//! exposes [`GpuFragmentBounds::compute`], which primes the extrema, scatters
//! every point into its fragment's box, finalises each box centre, folds the
//! squared radius, finalises the radius, and reads the proxies back. The result
//! matches the [`cpu_bounds_fragments`](super::cpu::cpu_bounds_fragments) golden
//! twin: the integer extrema are bit-identical and the finalised sphere radius
//! agrees within a tight floating-point tolerance.
//!
//! Provenance: axis-aligned extrema and a box-centred bounding sphere are
//! elementary geometry and fixed-point atomic reduction is a standard `GPU`
//! technique. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoder, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;

use super::config::BoundsConfig;
use super::cpu::FragmentBounds;
use super::layout::{buffer_entry, entry};

/// Uniform parameters shared with `Params` in `shaders/fracture_bounds.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of fragment cells.
    n_cells: u32,
    /// Number of query points.
    n_points: u32,
    /// Padding to keep the scalar block aligned.
    pad0: u32,
    /// Padding to keep the scalar block aligned.
    pad1: u32,
    /// Fixed-point scale for a quantised coordinate.
    position_scale: f32,
    /// Fixed-point scale for a quantised squared distance.
    radius_sq_scale: f32,
    /// Padding to a 16-byte boundary.
    pad2: f32,
    /// Padding to a 16-byte boundary.
    pad3: f32,
}

/// The finalised per-fragment proxy as written by the kernels, packed into vec4
/// lanes to match `BoundsOut` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BoundsOutRaw {
    /// xyz = box minimum corner, w unused.
    aabb_min: [f32; 4],
    /// xyz = box maximum corner, w unused.
    aabb_max: [f32; 4],
    /// xyz = sphere centre, w = sphere radius.
    sphere: [f32; 4],
}

/// A compiled, reusable `GPU` per-fragment bounds pipeline set.
pub struct GpuFragmentBounds {
    /// Kept alive so the pipelines it produced stay valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring params, inputs, accumulators, and outputs.
    layout: BindGroupLayout,
    /// The clear kernel: one invocation per cell, primes the extrema.
    clear: ComputePipeline,
    /// The box scatter kernel: one invocation per point.
    scatter_aabb: ComputePipeline,
    /// The centre finalise kernel: one invocation per cell.
    finalize_center: ComputePipeline,
    /// The radius scatter kernel: one invocation per point.
    scatter_radius: ComputePipeline,
    /// The radius finalise kernel: one invocation per cell.
    finalize_radius: ComputePipeline,
}

impl GpuFragmentBounds {
    /// Compiles the five bounds kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFragmentBounds {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_fracture_bounds"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/fracture_bounds.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_fracture_bounds_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_fracture_bounds_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |entry_point: &str, label: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let clear = build("clear", "prism_fracture_bounds_clear");
        let scatter_aabb = build("scatter_aabb", "prism_fracture_bounds_scatter_aabb");
        let finalize_center = build("finalize_center", "prism_fracture_bounds_finalize_center");
        let scatter_radius = build("scatter_radius", "prism_fracture_bounds_scatter_radius");
        let finalize_radius = build("finalize_radius", "prism_fracture_bounds_finalize_radius");
        GpuFragmentBounds {
            module,
            layout,
            clear,
            scatter_aabb,
            finalize_center,
            scatter_radius,
            finalize_radius,
        }
    }

    /// Builds one bounding box and sphere per fragment for the `n_cells`
    /// fragments gathering `points` under the `cells` assignment.
    ///
    /// `points` and `cells` must share the same length. A point tagged
    /// [`NO_CELL`](super::super::config::NO_CELL) or out of range for `n_cells`
    /// is ignored. The returned vector has exactly `n_cells` entries in
    /// cell-index order; when `n_cells` is zero the vector is empty and no work
    /// is dispatched.
    ///
    /// # Panics
    ///
    /// Panics if `points` and `cells` do not have the same length.
    #[must_use]
    pub fn compute(
        &self,
        ctx: &GpuContext,
        n_cells: usize,
        points: &[Vec3],
        cells: &[u32],
        config: &BoundsConfig,
    ) -> Vec<FragmentBounds> {
        assert!(
            points.len() == cells.len(),
            "points and cells must have equal length"
        );
        if n_cells == 0 {
            return Vec::new();
        }

        let device = ctx.device();
        let n_points = points.len();

        let params = Params {
            n_cells: u32::try_from(n_cells).unwrap_or(u32::MAX),
            n_points: u32::try_from(n_points).unwrap_or(u32::MAX),
            pad0: 0,
            pad1: 0,
            position_scale: config.position_scale,
            radius_sq_scale: config.radius_sq_scale,
            pad2: 0.0,
            pad3: 0.0,
        };
        let params_buf = buffer::uniform(device, "prism_fracture_bounds_params", &params);

        // A storage buffer must never be zero-sized, so an empty point cloud
        // uploads one padded entry the kernels never read (n_points == 0).
        let point_packed: Vec<[f32; 4]> = points.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let point_upload: &[[f32; 4]] = if point_packed.is_empty() {
            &[[0.0; 4]]
        } else {
            &point_packed
        };
        let cell_upload: &[u32] = if cells.is_empty() { &[0] } else { cells };

        let points_buf = buffer::storage_read(device, "prism_fracture_bounds_points", point_upload);
        let cells_buf = buffer::storage_read(device, "prism_fracture_bounds_cells", cell_upload);

        let extent_bytes = (n_cells * 3 * size_of::<i32>()) as u64;
        let radius_bytes = (n_cells * size_of::<i32>()) as u64;
        let out_bytes = (n_cells * size_of::<BoundsOutRaw>()) as u64;

        let acc_min = buffer::storage_rw_zeroed(device, "prism_fracture_bounds_min", extent_bytes);
        let acc_max = buffer::storage_rw_zeroed(device, "prism_fracture_bounds_max", extent_bytes);
        let acc_radius =
            buffer::storage_rw_zeroed(device, "prism_fracture_bounds_radius", radius_bytes);
        let out_buf = buffer::storage_rw_zeroed(device, "prism_fracture_bounds_out", out_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_fracture_bounds_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &points_buf),
                entry(2, &cells_buf),
                entry(3, &acc_min),
                entry(4, &acc_max),
                entry(5, &acc_radius),
                entry(6, &out_buf),
            ],
        });

        let out_stage = buffer::staging(device, "prism_fracture_bounds_out_stage", out_bytes);

        let cell_groups = u32::try_from(n_cells.div_ceil(64)).unwrap_or(u32::MAX);
        let point_groups = u32::try_from(n_points.div_ceil(64)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_fracture_bounds_encoder"),
        });
        // Each stage is its own pass so the next kernel observes the prior
        // writes; wgpu orders passes and makes prior storage writes visible.
        self.pass(&mut encoder, "clear", &self.clear, &bind, cell_groups);
        if point_groups > 0 {
            self.pass(
                &mut encoder,
                "scatter_aabb",
                &self.scatter_aabb,
                &bind,
                point_groups,
            );
        }
        self.pass(
            &mut encoder,
            "finalize_center",
            &self.finalize_center,
            &bind,
            cell_groups,
        );
        if point_groups > 0 {
            self.pass(
                &mut encoder,
                "scatter_radius",
                &self.scatter_radius,
                &bind,
                point_groups,
            );
        }
        self.pass(
            &mut encoder,
            "finalize_radius",
            &self.finalize_radius,
            &bind,
            cell_groups,
        );
        buffer::copy(&mut encoder, &out_buf, &out_stage, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        let raw = buffer::read_back::<BoundsOutRaw>(ctx, &out_stage);
        raw.into_iter().map(decode).collect()
    }

    /// Records one compute pass dispatching `pipeline` over `groups` workgroups.
    fn pass(
        &self,
        encoder: &mut CommandEncoder,
        label: &str,
        pipeline: &ComputePipeline,
        bind: &BindGroup,
        groups: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

/// Rebuilds a [`FragmentBounds`] from the packed lanes the kernels wrote.
fn decode(raw: BoundsOutRaw) -> FragmentBounds {
    FragmentBounds {
        aabb_min: Vec3::new(raw.aabb_min[0], raw.aabb_min[1], raw.aabb_min[2]),
        aabb_max: Vec3::new(raw.aabb_max[0], raw.aabb_max[1], raw.aabb_max[2]),
        sphere_center: Vec3::new(raw.sphere[0], raw.sphere[1], raw.sphere[2]),
        sphere_radius: raw.sphere[3],
    }
}
