//! `wgpu` compute twin of Prism's density-LOD binding-importance propagation
//! ([`binding_importances`](prism_render_architecture::hair::density_lod::binding_importances)).
//!
//! [`compute_importance`](prism_render_architecture::hair::decimation::compute_importance)
//! (the sibling [`GpuHairImportance`](crate::importance)) folds *already-blended*
//! per-strand metrics into one importance. But a render strand does not carry its
//! own arc length or curvature — it is skinned to up to four guide strands, and
//! the density-LOD driver must first *propagate* the per-guide metrics onto every
//! render strand through its binding weights before it can rank which strands
//! survive as a groom recedes. This kernel is the on-device twin of that whole
//! propagate-and-fold step: one thread per binding gathers the weight-blended
//! length, curvature and authored thickness of the guides it is bound to (skipping
//! out-of-range guides and non-positive weights, exactly like the reference) and
//! then folds the blended triple into an importance in `0..=1` — the guide->render
//! density-LOD bias `UE5` Groom applies. A passing real-device parity test is
//! direct evidence the ported kernel propagates and weighs render strands the same
//! way the reference does, not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairBindingImportance::eval`] takes the per-binding [`RenderStrandBinding`]
//! slice, the per-guide [`GuideMetrics`] and the artist [`ImportanceWeights`], and
//! returns one folded importance per binding. The binding index is simply the
//! invocation id.
//!
//! # Groom-level reductions stay on the host
//!
//! The two maxima `max_len` / `max_curv` are reduced over the *blended* per-binding
//! metrics host-side (bit-for-bit the golden's own reduction inside
//! `compute_importance`) and passed as uniform params, the same split the sibling
//! importance twin documents. A maximum over finite `f32` is exact and
//! order-independent, so the host reduction is bit-identical to any `GPU`
//! reduction; the kernel performs the naturally parallel per-binding gather + fold.
//!
//! # Portability
//!
//! The kernel uses only `clamp`, division and multiply/add in the portable
//! core-`WGSL` subset — no `exp`, `pow`, `sin` or optional device feature — so the
//! twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The maxima, the weight sum and both normalizations are plain IEEE operations
//! the `CPU` and `GPU` evaluate identically; only the weighted gather sums and the
//! final blend are multiply-add chains a `GPU` may fuse, so `CPU` and `GPU` agree
//! to within the documented fma tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`)
//! rather than bit-for-bit. Degenerate inputs (a non-positive weight sum, a zero
//! maximum, an empty guide set) collapse to the same exact `0` the golden emits.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and `bytemuck`
//! surfaces.
//!
//! Provenance: standard guide->render metric propagation plus importance-weighted
//! density-LOD fold plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::decimation::ImportanceWeights;
use prism_render_architecture::hair::density_lod::GuideMetrics;
use prism_render_architecture::hair::interpolation::RenderStrandBinding;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters for one binding-importance dispatch. Layout matches
/// `Params` in `shaders/binding_importance.wesl`: the three artist weights, the
/// two groom-level maxima, the binding count and the guide count, packed into two
/// `16`-byte uniform slots.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w_length: f32,
    w_curvature: f32,
    w_authored: f32,
    max_len: f32,
    max_curv: f32,
    binding_count: u32,
    guide_count: u32,
    pad0: u32,
}

/// A compiled, reusable per-binding importance-propagation pipeline.
pub struct GpuHairBindingImportance {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairBindingImportance {
    /// Compiles the per-binding importance-propagation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairBindingImportance {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_binding_importance"),
            source: ShaderSource::Wgsl(include_str!("../shaders/binding_importance.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_binding_importance_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_binding_importance_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_binding_importance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairBindingImportance {
            module,
            layout,
            pipeline,
        }
    }

    /// Propagates the per-guide `metrics` onto each render-strand `binding` through
    /// its guide weights and folds the blended triple into one importance per
    /// binding.
    ///
    /// The importance for binding `t` equals the `CPU` golden
    /// [`binding_importances`](prism_render_architecture::hair::density_lod::binding_importances)
    /// at index `t` to within the module's documented fma tolerance. The two
    /// groom-level maxima are reduced host-side over the blended metrics
    /// (bit-for-bit the golden's own reduction) and passed as uniforms. An empty
    /// binding list yields an empty vector without a dispatch — storage buffers
    /// cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        bindings: &[RenderStrandBinding],
        metrics: &GuideMetrics,
        weights: ImportanceWeights,
    ) -> Vec<f32> {
        let binding_count = bindings.len();
        if binding_count == 0 {
            return Vec::new();
        }
        let guide_count = metrics.len();

        // Reproduce the golden's weighted gather so the two maxima are reduced
        // over the *blended* per-binding metrics in the identical order (start at
        // 0, strict `>` update), keeping the host reduction bit-identical.
        let mut guides: Vec<[u32; 4]> = Vec::with_capacity(binding_count);
        let mut bind_weights: Vec<[f32; 4]> = Vec::with_capacity(binding_count);
        let mut max_len = 0.0_f32;
        let mut max_curv = 0.0_f32;
        for binding in bindings {
            let mut len_acc = 0.0_f32;
            let mut curv_acc = 0.0_f32;
            for (&guide, &weight) in binding.guides.iter().zip(binding.weights.iter()) {
                let idx = guide as usize;
                if weight > 0.0 && idx < guide_count {
                    len_acc += weight * metrics.lengths[idx];
                    curv_acc += weight * metrics.curvatures[idx];
                }
            }
            if len_acc > max_len {
                max_len = len_acc;
            }
            if curv_acc > max_curv {
                max_curv = curv_acc;
            }
            guides.push(binding.guides);
            bind_weights.push(binding.weights);
        }

        // Storage buffers cannot be zero-sized: pad empty metric pools with one
        // dummy entry. The `guide_count == 0` guard keeps every gather from ever
        // reading it.
        let mut metric_lengths = metrics.lengths.clone();
        let mut metric_curvatures = metrics.curvatures.clone();
        let mut metric_authored = metrics.authored.clone();
        if metric_lengths.is_empty() {
            metric_lengths.push(0.0);
        }
        if metric_curvatures.is_empty() {
            metric_curvatures.push(0.0);
        }
        if metric_authored.is_empty() {
            metric_authored.push(0.0);
        }

        let device = ctx.device();

        let uniforms = Params {
            w_length: weights.length,
            w_curvature: weights.curvature,
            w_authored: weights.authored,
            max_len,
            max_curv,
            binding_count: binding_count as u32,
            guide_count: guide_count as u32,
            pad0: 0,
        };

        let out_bytes = (binding_count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_binding_importance_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let guides_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_binding_importance_guides"),
            contents: bytemuck::cast_slice(&guides),
            usage: BufferUsages::STORAGE,
        });
        let weights_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_binding_importance_weights"),
            contents: bytemuck::cast_slice(&bind_weights),
            usage: BufferUsages::STORAGE,
        });
        let lengths_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_binding_importance_lengths"),
            contents: bytemuck::cast_slice(&metric_lengths),
            usage: BufferUsages::STORAGE,
        });
        let curvatures_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_binding_importance_curvatures"),
            contents: bytemuck::cast_slice(&metric_curvatures),
            usage: BufferUsages::STORAGE,
        });
        let authored_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_binding_importance_authored"),
            contents: bytemuck::cast_slice(&metric_authored),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_binding_importance_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_binding_importance_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_binding_importance_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: guides_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: weights_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: lengths_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: curvatures_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: authored_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_binding_importance_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_binding_importance_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (binding_count as u32).div_ceil(64);
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
