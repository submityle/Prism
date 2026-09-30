//! Real-device `wgpu` compute implementation of the Voronoi fragment
//! classifier.
//!
//! [`GpuVoronoiAssign`] compiles `shaders/fracture_voronoi.wgsl` once and
//! exposes [`GpuVoronoiAssign::assign`], which uploads the seed sites and query
//! points, dispatches one thread per point, and reads back the owning cell
//! index and wall clearance for every point. The result matches the
//! [`cpu_assign_cells`](super::super::cpu::cpu_assign_cells) golden twin: the
//! cell indices are integer-exact whenever the nearest site is unambiguous, and
//! the clearances agree within a tight floating-point tolerance.
//!
//! Provenance: nearest-site Voronoi membership and perpendicular-bisector cell
//! walls are standard, publicly documented computational-geometry results. No
//! Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;

use super::super::config::VoronoiAssignConfig;
use super::super::cpu::CellAssignment;
use super::layout::{buffer_entry, entry};

/// Uniform parameters shared with `Params` in `shaders/fracture_voronoi.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of seed sites.
    n_sites: u32,
    /// Number of query points.
    n_points: u32,
    /// Squared-length degeneracy threshold for rival bisectors.
    degenerate_eps: f32,
    /// Padding to a 16-byte boundary.
    _pad: f32,
}

/// A compiled, reusable `GPU` Voronoi fragment-assignment pipeline.
pub struct GpuVoronoiAssign {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring params, inputs, and outputs.
    layout: BindGroupLayout,
    /// The single classification kernel.
    assign: ComputePipeline,
}

impl GpuVoronoiAssign {
    /// Compiles the classifier kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVoronoiAssign {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_fracture_voronoi"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/fracture_voronoi.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_fracture_voronoi_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_fracture_voronoi_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let assign = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_fracture_voronoi_assign"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("assign"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVoronoiAssign {
            module,
            layout,
            assign,
        }
    }

    /// Bins every point in `points` into the Voronoi cell of the nearest site
    /// in `sites`, returning one [`CellAssignment`] per point in input order.
    ///
    /// An empty `points` slice returns an empty vector without dispatching. An
    /// empty `sites` slice reports every point as
    /// [`NO_CELL`](super::super::config::NO_CELL) with a saturated clearance.
    #[must_use]
    pub fn assign(
        &self,
        ctx: &GpuContext,
        sites: &[Vec3],
        points: &[Vec3],
        config: &VoronoiAssignConfig,
    ) -> Vec<CellAssignment> {
        if points.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let n_points = points.len();

        let params = Params {
            n_sites: u32::try_from(sites.len()).unwrap_or(u32::MAX),
            n_points: u32::try_from(n_points).unwrap_or(u32::MAX),
            degenerate_eps: config.degenerate_eps,
            _pad: 0.0,
        };
        let params_buf = buffer::uniform(device, "prism_fracture_voronoi_params", &params);

        // A storage buffer must never be zero-sized, so an empty site cloud
        // uploads a single padded element the kernel never reads (n_sites == 0).
        let site_packed: Vec<[f32; 4]> = sites.iter().map(|s| [s.x, s.y, s.z, 0.0]).collect();
        let site_upload: &[[f32; 4]] = if site_packed.is_empty() {
            &[[0.0; 4]]
        } else {
            &site_packed
        };
        let point_packed: Vec<[f32; 4]> = points.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();

        let sites_buf = buffer::storage_read(device, "prism_fracture_voronoi_sites", site_upload);
        let points_buf =
            buffer::storage_read(device, "prism_fracture_voronoi_points", &point_packed);
        let cell_bytes = (n_points * size_of::<u32>()) as u64;
        let clearance_bytes = (n_points * size_of::<f32>()) as u64;
        let cell_buf = buffer::storage_rw_zeroed(device, "prism_fracture_voronoi_cell", cell_bytes);
        let clearance_buf =
            buffer::storage_rw_zeroed(device, "prism_fracture_voronoi_clearance", clearance_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_fracture_voronoi_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &sites_buf),
                entry(2, &points_buf),
                entry(3, &cell_buf),
                entry(4, &clearance_buf),
            ],
        });

        let cell_stage = buffer::staging(device, "prism_fracture_voronoi_cell_stage", cell_bytes);
        let clearance_stage = buffer::staging(
            device,
            "prism_fracture_voronoi_clearance_stage",
            clearance_bytes,
        );

        let groups = u32::try_from(n_points.div_ceil(64)).unwrap_or(u32::MAX);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_fracture_voronoi_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_fracture_voronoi_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.assign);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        buffer::copy(&mut encoder, &cell_buf, &cell_stage, cell_bytes);
        buffer::copy(
            &mut encoder,
            &clearance_buf,
            &clearance_stage,
            clearance_bytes,
        );
        ctx.queue().submit([encoder.finish()]);

        let cells = buffer::read_back::<u32>(ctx, &cell_stage);
        let clearances = buffer::read_back::<f32>(ctx, &clearance_stage);
        cells
            .into_iter()
            .zip(clearances)
            .map(|(cell, clearance)| CellAssignment { cell, clearance })
            .collect()
    }
}
