//! `wgpu` compute twin of Prism's density-LOD importance fold
//! ([`compute_importance`](prism_render_architecture::hair::decimation::compute_importance)).
//!
//! Continuous density LOD keeps fewer render strands as a groom recedes, and to
//! pick which strands survive to the lowest counts the reference folds three
//! per-strand metrics into a single importance in `0..=1`: normalized arc length
//! (longer strands read more on the silhouette), normalized accumulated
//! curvature (curls / flyaways matter) and an already-normalized authored artist
//! priority. Each metric is normalized by the groom-wide maximum so scene scale
//! drops out, blended by artist [`ImportanceWeights`], then renormalized by the
//! weight sum — the density-LOD bias `UE5` Groom and `HairWorks` apply. The
//! folded importance then feeds
//! [`decimation_priority`](prism_render_architecture::hair::decimation::decimation_priority)
//! and the host ranking sort that builds the nested, pop-free decimation order;
//! this kernel is only the per-strand fold, the part that maps cleanly to one
//! `GPU` thread per strand. The `CPU` golden is
//! [`compute_importance`](prism_render_architecture::hair::decimation::compute_importance);
//! this crate is the on-device twin that runs the identical fold so a passing
//! real-device parity test is direct evidence the ported kernel weighs strands
//! the same way the reference does — not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairImportance::eval`] takes the parallel `lengths`, `curvatures` and
//! `authored` slices plus the artist [`ImportanceWeights`], and returns one
//! folded importance per strand. The strand index is simply the invocation id.
//!
//! # Groom-level reductions stay on the host
//!
//! The two maxima `max_len` / `max_curv` are computed host-side (bit-for-bit the
//! golden's own reduction) and passed as uniform params, the same split the
//! sibling `hair_importance.wesl` compile-time twin documents. A maximum over
//! finite `f32` is exact and order-independent, so the host reduction is
//! bit-identical to any `GPU` reduction; the kernel only performs the naturally
//! parallel per-strand fold.
//!
//! # Portability
//!
//! The kernel uses only `clamp`, division and multiply/add in the portable
//! core-`WGSL` subset — no `exp`, `pow`, `sin` or optional device feature — so
//! the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The maxima, the weight sum and both normalizations are plain IEEE operations
//! the `CPU` and `GPU` evaluate identically; only the final blend
//! `w_l * len_n + w_c * curv_n + w_a * auth` is a multiply-add chain a `GPU` may
//! fuse, so `CPU` and `GPU` agree to within the documented fma tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`) rather than bit-for-bit. Degenerate
//! inputs (a non-positive weight sum, a zero maximum) collapse to the same
//! exact `0` the golden emits.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard importance-weighted density-LOD metric fold plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::decimation::ImportanceWeights;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters for one importance-fold dispatch. Layout matches `Params`
/// in `shaders/importance.wesl`: the three artist weights, the two groom-level
/// maxima and the strand count, padded to two `16`-byte uniform slots.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w_length: f32,
    w_curvature: f32,
    w_authored: f32,
    max_len: f32,
    max_curv: f32,
    strand_count: u32,
    pad0: u32,
    pad1: u32,
}

/// A compiled, reusable per-strand importance-fold pipeline.
pub struct GpuHairImportance {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairImportance {
    /// Compiles the per-strand importance-fold kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairImportance {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_importance"),
            source: ShaderSource::Wgsl(include_str!("../shaders/importance.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_importance_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_importance_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_importance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairImportance {
            module,
            layout,
            pipeline,
        }
    }

    /// Folds the three per-strand metrics into one importance per strand,
    /// returning one f32 in `0..=1` per strand.
    ///
    /// `lengths`, `curvatures` and `authored` are the parallel per-strand raw
    /// arc length, accumulated curvature and authored priority; they must share
    /// a length. `weights` are the artist blend weights. The importance for
    /// strand `i` equals the `CPU` golden
    /// [`compute_importance`](prism_render_architecture::hair::decimation::compute_importance)
    /// at index `i` to within the module's documented fma tolerance. The two
    /// groom-level maxima are reduced on the host (bit-for-bit the golden's own
    /// reduction) and passed as uniforms. A length mismatch or an empty batch
    /// yields an empty vector without a dispatch — storage buffers cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        lengths: &[f32],
        curvatures: &[f32],
        authored: &[f32],
        weights: ImportanceWeights,
    ) -> Vec<f32> {
        let strand_count = lengths.len();
        if strand_count == 0 || curvatures.len() != strand_count || authored.len() != strand_count {
            return Vec::new();
        }

        // Groom-wide maxima, reduced exactly as the golden does (start at 0,
        // strict `>` update) so the host reduction is bit-identical.
        let mut max_len = 0.0_f32;
        let mut max_curv = 0.0_f32;
        for i in 0..strand_count {
            if lengths[i] > max_len {
                max_len = lengths[i];
            }
            if curvatures[i] > max_curv {
                max_curv = curvatures[i];
            }
        }

        let device = ctx.device();

        let uniforms = Params {
            w_length: weights.length,
            w_curvature: weights.curvature,
            w_authored: weights.authored,
            max_len,
            max_curv,
            strand_count: strand_count as u32,
            pad0: 0,
            pad1: 0,
        };

        let out_bytes = (strand_count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_importance_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let lengths_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_importance_lengths"),
            contents: bytemuck::cast_slice(lengths),
            usage: BufferUsages::STORAGE,
        });
        let curvatures_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_importance_curvatures"),
            contents: bytemuck::cast_slice(curvatures),
            usage: BufferUsages::STORAGE,
        });
        let authored_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_importance_authored"),
            contents: bytemuck::cast_slice(authored),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_importance_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_importance_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_importance_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: lengths_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: curvatures_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: authored_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_importance_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_importance_pass"),
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
        let importances = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        importances
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
