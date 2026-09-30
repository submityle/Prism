//! Real-device `wgpu` compute implementation of the per-fragment aggregator.
//!
//! [`GpuFragmentAggregate`] compiles `shaders/fracture_aggregate.wgsl` once and
//! exposes [`GpuFragmentAggregate::aggregate`], which scatters every point's
//! mass and moments into its fragment's fixed-point accumulators, finalises one
//! rigid-body seed per fragment, and reads the seeds back. The result matches
//! the [`cpu_aggregate_fragments`](super::cpu::cpu_aggregate_fragments) golden
//! twin: the integer accumulation is bit-identical and the finalised centroid
//! and inertia agree within a tight floating-point tolerance.
//!
//! Provenance: rigid-body mass/centroid/inertia formulas are textbook mechanics
//! and fixed-point atomic accumulation is a standard `GPU` reduction. No Unreal
//! Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec3};
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;

use super::config::AggregateConfig;
use super::cpu::FragmentAggregate;
use super::layout::{buffer_entry, entry};

/// Uniform parameters shared with `Params` in `shaders/fracture_aggregate.wgsl`.
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
    /// Fixed-point scale for accumulated mass.
    mass_scale: f32,
    /// Fixed-point scale for the accumulated first moment.
    moment_scale: f32,
    /// Fixed-point scale for the accumulated second moment.
    second_moment_scale: f32,
    /// Padding to a 16-byte boundary.
    pad2: f32,
}

/// The finalised per-fragment seed as written by the `finalize` kernel, packed
/// into vec4 lanes to match `FragmentOut` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FragmentOutRaw {
    /// xyz = centre of mass, w = total mass.
    centroid_mass: [f32; 4],
    /// `Ixx`, `Iyy`, `Izz`, `Ixy`.
    inertia0: [f32; 4],
    /// `Ixz`, `Iyz`, and two unused padding lanes.
    inertia1: [f32; 4],
}

/// A compiled, reusable `GPU` per-fragment aggregation pipeline pair.
pub struct GpuFragmentAggregate {
    /// Kept alive so the pipelines it produced stay valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring params, inputs, accumulators, and outputs.
    layout: BindGroupLayout,
    /// The scatter kernel: one invocation per point.
    scatter: ComputePipeline,
    /// The finalise kernel: one invocation per fragment cell.
    finalize: ComputePipeline,
}

impl GpuFragmentAggregate {
    /// Compiles the scatter and finalise kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFragmentAggregate {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_fracture_aggregate"),
            source: ShaderSource::Wgsl(
                include_str!("../../shaders/fracture_aggregate.wgsl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_fracture_aggregate_layout"),
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
            label: Some("prism_fracture_aggregate_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let scatter = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_fracture_aggregate_scatter"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("scatter"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let finalize = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_fracture_aggregate_finalize"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("finalize"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFragmentAggregate {
            module,
            layout,
            scatter,
            finalize,
        }
    }

    /// Aggregates the per-fragment mass, centroid, and inertia for `n_cells`
    /// fragments from `points`, their `masses`, and their `cells` assignment.
    ///
    /// The three slices must share the same length. A point whose cell is
    /// [`NO_CELL`](super::super::config::NO_CELL) or is out of range for
    /// `n_cells` is skipped. The returned vector has exactly `n_cells` entries
    /// in cell-index order; an empty `n_cells` returns an empty vector without
    /// dispatching.
    ///
    /// # Panics
    ///
    /// Panics if `points`, `masses`, and `cells` do not all have the same
    /// length.
    #[must_use]
    pub fn aggregate(
        &self,
        ctx: &GpuContext,
        n_cells: usize,
        points: &[Vec3],
        masses: &[f32],
        cells: &[u32],
        config: &AggregateConfig,
    ) -> Vec<FragmentAggregate> {
        assert!(
            points.len() == masses.len() && masses.len() == cells.len(),
            "points, masses, and cells must have equal length"
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
            mass_scale: config.mass_scale,
            moment_scale: config.moment_scale,
            second_moment_scale: config.second_moment_scale,
            pad2: 0.0,
        };
        let params_buf = buffer::uniform(device, "prism_fracture_aggregate_params", &params);

        // Pack position and mass into a single vec4 lane. A storage buffer must
        // never be zero-sized, so an empty point cloud uploads one padded entry
        // the kernel never reads (n_points == 0).
        let point_packed: Vec<[f32; 4]> = points
            .iter()
            .zip(masses)
            .map(|(p, &m)| [p.x, p.y, p.z, m])
            .collect();
        let point_upload: &[[f32; 4]] = if point_packed.is_empty() {
            &[[0.0; 4]]
        } else {
            &point_packed
        };
        let cell_upload: &[u32] = if cells.is_empty() { &[0] } else { cells };

        let points_buf =
            buffer::storage_read(device, "prism_fracture_aggregate_points", point_upload);
        let cells_buf = buffer::storage_read(device, "prism_fracture_aggregate_cells", cell_upload);

        let mass_bytes = (n_cells * size_of::<i32>()) as u64;
        let moment_bytes = (n_cells * 3 * size_of::<i32>()) as u64;
        let second_bytes = (n_cells * 6 * size_of::<i32>()) as u64;
        let out_bytes = (n_cells * size_of::<FragmentOutRaw>()) as u64;

        let acc_mass =
            buffer::storage_rw_zeroed(device, "prism_fracture_aggregate_mass", mass_bytes);
        let acc_moment =
            buffer::storage_rw_zeroed(device, "prism_fracture_aggregate_moment", moment_bytes);
        let acc_second =
            buffer::storage_rw_zeroed(device, "prism_fracture_aggregate_second", second_bytes);
        let out_buf = buffer::storage_rw_zeroed(device, "prism_fracture_aggregate_out", out_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_fracture_aggregate_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &points_buf),
                entry(2, &cells_buf),
                entry(3, &acc_mass),
                entry(4, &acc_moment),
                entry(5, &acc_second),
                entry(6, &out_buf),
            ],
        });

        let out_stage = buffer::staging(device, "prism_fracture_aggregate_out_stage", out_bytes);

        let scatter_groups = u32::try_from(n_points.div_ceil(64)).unwrap_or(u32::MAX);
        let finalize_groups = u32::try_from(n_cells.div_ceil(64)).unwrap_or(u32::MAX);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_fracture_aggregate_encoder"),
        });
        if scatter_groups > 0 {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_fracture_aggregate_scatter_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(scatter_groups, 1, 1);
        }
        {
            // A separate pass so the finalise kernel observes every scatter
            // write; wgpu orders passes and makes prior storage writes visible.
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_fracture_aggregate_finalize_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.finalize);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(finalize_groups, 1, 1);
        }
        buffer::copy(&mut encoder, &out_buf, &out_stage, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        let raw = buffer::read_back::<FragmentOutRaw>(ctx, &out_stage);
        raw.into_iter().map(decode).collect()
    }
}

/// Rebuilds a [`FragmentAggregate`] from the packed lanes the `finalize` kernel
/// wrote, reconstructing the symmetric inertia tensor from its unique entries.
fn decode(raw: FragmentOutRaw) -> FragmentAggregate {
    let centroid = Vec3::new(
        raw.centroid_mass[0],
        raw.centroid_mass[1],
        raw.centroid_mass[2],
    );
    let mass = raw.centroid_mass[3];
    let i_xx = raw.inertia0[0];
    let i_yy = raw.inertia0[1];
    let i_zz = raw.inertia0[2];
    let i_xy = raw.inertia0[3];
    let i_xz = raw.inertia1[0];
    let i_yz = raw.inertia1[1];
    let inertia = Mat3::from_cols(
        Vec3::new(i_xx, i_xy, i_xz),
        Vec3::new(i_xy, i_yy, i_yz),
        Vec3::new(i_xz, i_yz, i_zz),
    );
    FragmentAggregate {
        mass,
        centroid,
        inertia,
    }
}
