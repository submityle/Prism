//! `wgpu` compute twin of Prism's per-strand geometry metrics
//! ([`strand_arc_length`](prism_render_architecture::hair::decimation::strand_arc_length)
//! and
//! [`strand_curvature`](prism_render_architecture::hair::decimation::strand_curvature)).
//!
//! Continuous-LOD density thinning ranks render strands by how much silhouette
//! each carries, and the two cheap per-strand scalars that feed that ranking are
//! a strand's arc length and its transcendental-free turning (a curvature
//! proxy). Both are pure per-strand reductions with no cross-strand dependency,
//! so they map cleanly to one `GPU` thread per strand over the strand-major
//! fixed-stride point pool the resampled groom already stores. The `CPU` goldens
//! are
//! [`strand_arc_length`](prism_render_architecture::hair::decimation::strand_arc_length)
//! and
//! [`strand_curvature`](prism_render_architecture::hair::decimation::strand_curvature);
//! this crate is the on-device twin that walks the same reductions so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same metrics as the reference — not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuStrandMetrics::eval`] takes a flat strand-major point pool and the fixed
//! points-per-strand stride, and returns one [`GpuStrandMetric`] per strand.
//! Arc length sums the segment lengths root → tip (`0` below two points).
//! Curvature sums `1 - dot(t_in, t_out)` over the interior vertices (`0` below
//! three points), each tangent guarded by the same zero-length fallback the
//! reference's
//! [`normalize_or`](prism_render_architecture::hair::interpolation) uses, so
//! coincident control points contribute a defined zero-tangent term rather than
//! dividing by zero. Both accumulate in the reference's ascending index order.
//!
//! # Portability
//!
//! The kernel uses only `sqrt`, `dot` and multiply/add in the portable
//! core-`WGSL` subset — no `exp`, `pow` or optional device feature — so the twin
//! runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The reductions contain no transcendental call (the reference restricts itself
//! to `sqrt` via vector length), so `CPU` and `GPU` evaluate the same closed-form
//! geometry in the same ascending order. They are not bit-exact: a `GPU` may fuse
//! a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits. The parity test therefore asserts a per-component tolerance
//! rather than exact equality.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard polyline arc-length / turning-angle geometry metrics
//! plus `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One strand's geometry metrics, mirroring the `CPU` goldens
/// [`strand_arc_length`](prism_render_architecture::hair::decimation::strand_arc_length)
/// and
/// [`strand_curvature`](prism_render_architecture::hair::decimation::strand_curvature).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuStrandMetric {
    /// Accumulated arc length of the strand polyline (`0` below two points).
    pub arc_length: f32,
    /// Accumulated turning `Σ (1 - dot(t_in, t_out))` (`0` below three points).
    pub curvature: f32,
}

/// Uniform parameters for one metrics dispatch. Layout matches `Params` in
/// `shaders/strand_metrics.wesl`: the strand count and the fixed
/// points-per-strand stride, padded to one `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    strand_count: u32,
    points_per_strand: u32,
    pad0: u32,
    pad1: u32,
}

/// A compiled, reusable per-strand metrics pipeline.
pub struct GpuStrandMetrics {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuStrandMetrics {
    /// Compiles the per-strand metrics kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuStrandMetrics {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_strand_metrics"),
            source: ShaderSource::Wgsl(include_str!("../shaders/strand_metrics.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_strand_metrics_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_strand_metrics_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_strand_metrics_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuStrandMetrics {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the arc length and curvature of every strand in a strand-major
    /// fixed-stride point pool, returning one [`GpuStrandMetric`] per strand.
    ///
    /// `points` is the flat pool of every strand's control points in order, and
    /// `points_per_strand` is the fixed stride (the resampled groom's uniform
    /// point count). The strand count is `points.len() / points_per_strand`;
    /// only the whole strands that fit are processed. The metric for strand `s`
    /// equals the `CPU` goldens applied to that strand's `points_per_strand`
    /// points, to within the fused-multiply-add tolerance documented on this
    /// module. An empty pool or a zero stride yields an empty vector without a
    /// dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        points: &[[f32; 3]],
        points_per_strand: usize,
    ) -> Vec<GpuStrandMetric> {
        if points_per_strand == 0 {
            return Vec::new();
        }
        let strand_count = points.len() / points_per_strand;
        if strand_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            strand_count: strand_count as u32,
            points_per_strand: points_per_strand as u32,
            pad0: 0,
            pad1: 0,
        };

        // Flatten the whole-strand prefix of the pool to stride-3 f32.
        let used_points = strand_count * points_per_strand;
        let mut positions: Vec<f32> = Vec::with_capacity(used_points * 3);
        for p in &points[..used_points] {
            positions.extend_from_slice(p);
        }

        let out_bytes = (strand_count as u64) * 2 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_strand_metrics_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_strand_metrics_points"),
            contents: bytemuck::cast_slice(&positions),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_strand_metrics_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_strand_metrics_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_strand_metrics_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: points_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_strand_metrics_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_strand_metrics_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (strand_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        flat.chunks_exact(2)
            .map(|c| GpuStrandMetric {
                arc_length: c[0],
                curvature: c[1],
            })
            .collect()
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
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
