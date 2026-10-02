//! `wgpu` compute twin of Prism's adaptive transmittance curve *accumulation*
//! ([`accumulate`](prism_render_architecture::hair::adaptive_transmittance::accumulate)).
//!
//! A deep-shadow groom turns each light ray/texel's overlapping strand samples
//! into a transmittance curve: a list of `(depth, transmittance)` control nodes
//! whose running composite `T = product(1 - alpha)` a shading pass decodes. This
//! kernel is the *build* side of that pipeline — the one that produces the full
//! (uncompressed) [`TransmittanceCurve`](prism_render_architecture::hair::adaptive_transmittance::TransmittanceCurve)
//! from raw samples — as opposed to the sibling
//! [`GpuHairAdaptiveTransmittance`](crate::adaptive_transmittance::GpuHairAdaptiveTransmittance),
//! which is the receiver-query *lookup* on an already-compressed curve.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairAdaptiveAccumulate::eval`] takes a batch of rays, each a slice of
//! [`TransmittanceSample`](prism_render_architecture::hair::adaptive_transmittance::TransmittanceSample),
//! and returns one [`TransmittanceCurve`] per ray in input order — the
//! array-in/array-out form used to accumulate a tile of light texels at once.
//! Each curve is monotone non-increasing in transmittance and sorted by depth,
//! exactly like the golden.
//!
//! # The sort and grouping live on the host, the running product on the device
//!
//! The golden re-sanitises every sample through
//! [`TransmittanceSample::new`](prism_render_architecture::hair::adaptive_transmittance::TransmittanceSample::new),
//! stably sorts it by light-space depth, and folds every sample sharing a depth
//! (within `DEPTH_EPS`) into a single control node. The sort and the same-depth
//! grouping are host/sequential concerns (the golden documents the device-side
//! build as separate scheduling), so `eval` reproduces them on the `CPU` and
//! uploads each ray's node list as a contiguous run of `(alpha_start,
//! alpha_count)` descriptors into a shared, per-node-grouped `alphas` pool. The
//! kernel owns exactly the per-ray running product: one thread per ray walks its
//! nodes in depth order, composites each node's alphas into the running
//! `product(1 - alpha)`, clamps at the node boundary, and writes one
//! transmittance per global node. The node depths are host passthrough and are
//! reattached after readback.
//!
//! # Distinct from the deep-opacity twin
//!
//! This is **not** the fixed equal-width layer pack of
//! [`GpuHairDeepOpacity`](crate::deep_opacity::GpuHairDeepOpacity): there every
//! texel emits the same `layer_count` rows over a sliced depth span. Here each
//! ray emits a *variable* number of nodes — one per distinct sample depth — so
//! the output length follows the signal rather than a constant stride.
//!
//! # Correctness model
//!
//! The composite is a closed-form running product with no transcendental call,
//! so `CPU` and `GPU` evaluate the same arithmetic in the same order. They are
//! not bit-exact: a `GPU` may fuse the `running * (1 - alpha)` multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! `ULP`. The parity test asserts a per-value tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`) on transmittance; node depths and node counts match
//! exactly because they are host passthrough.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard alpha-composite running product (per-ray depth sort then
//! cumulative `product(1 - alpha)`) plus `wgpu` compute dispatch; no Unreal
//! Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::adaptive_transmittance::{
    TransmittanceCurve, TransmittanceNode, TransmittanceSample,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Mirrors `adaptive_transmittance::DEPTH_EPS`: samples closer together than
/// this in light-space depth fold into the same control node. The constant is
/// private to the golden, so the twin re-declares the identical value to
/// reproduce the grouping bit-for-bit.
const DEPTH_EPS: f32 = 1e-6;

/// Uniform ray count uploaded to the kernel. A single `u32` padded to the
/// `16`-byte uniform block, matching `Params` in
/// `shaders/adaptive_accumulate.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    ray_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One control node's alpha slice descriptor: `(start, count)` into the shared
/// `alphas` pool. `8`-byte `repr(C)` matching the flat `u32` pair layout in the
/// shader's `nodes` buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuNode {
    alpha_start: u32,
    alpha_count: u32,
}

/// One ray's node slice descriptor: `(start, count)` into the shared `nodes`
/// pool. `8`-byte `repr(C)` matching the flat `u32` pair layout in the shader's
/// `rays` buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuRay {
    node_start: u32,
    node_count: u32,
}

/// A compiled, reusable adaptive transmittance accumulation pipeline.
pub struct GpuHairAdaptiveAccumulate {
    #[expect(
        dead_code,
        reason = "the shader module must outlive the pipeline that borrows it"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairAdaptiveAccumulate {
    /// Compiles the accumulation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairAdaptiveAccumulate {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_adaptive_accumulate"),
            source: ShaderSource::Wgsl(include_str!("../shaders/adaptive_accumulate.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_adaptive_accumulate_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_adaptive_accumulate_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_adaptive_accumulate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairAdaptiveAccumulate {
            module,
            layout,
            pipeline,
        }
    }

    /// Accumulates each ray's strand `samples` into a full
    /// [`TransmittanceCurve`], returning one curve per ray in input order.
    ///
    /// The result equals
    /// [`accumulate`](prism_render_architecture::hair::adaptive_transmittance::accumulate)
    /// per ray to within the fused-multiply-add tolerance documented on this
    /// module. The host reproduces the golden's re-sanitise, stable `total_cmp`
    /// depth sort and same-depth grouping before upload (the kernel does neither
    /// sorting nor grouping), so node depths and node counts match exactly and
    /// only the running-product transmittance carries fma drift. Empty input, or
    /// a batch where every ray is empty, returns the empty (fully transmissive)
    /// curves without a dispatch (storage buffers cannot be zero-sized).
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        rays: &[&[TransmittanceSample]],
    ) -> Vec<TransmittanceCurve> {
        let ray_count = rays.len();
        if ray_count == 0 {
            return Vec::new();
        }

        // Reproduce the golden's host-side re-sanitise + stable depth sort +
        // same-depth grouping, flattening every node's alphas into a shared pool
        // the kernel composites. `node_depths` records each global node's depth
        // (host passthrough) so curves can be reassembled after readback.
        let mut alphas: Vec<f32> = Vec::new();
        let mut gpu_nodes: Vec<GpuNode> = Vec::new();
        let mut gpu_rays: Vec<GpuRay> = Vec::with_capacity(ray_count);
        let mut node_depths: Vec<f32> = Vec::new();
        for ray in rays {
            let node_start = gpu_nodes.len() as u32;
            let mut sorted: Vec<TransmittanceSample> = ray
                .iter()
                .map(|s| TransmittanceSample::new(s.depth, s.alpha))
                .collect();
            sorted.sort_by(|a, b| a.depth.total_cmp(&b.depth));

            let mut i = 0usize;
            while i < sorted.len() {
                let depth = sorted[i].depth;
                let alpha_start = alphas.len() as u32;
                let mut alpha_count = 0u32;
                while i < sorted.len() && (sorted[i].depth - depth).abs() < DEPTH_EPS {
                    alphas.push(sorted[i].alpha);
                    alpha_count += 1;
                    i += 1;
                }
                gpu_nodes.push(GpuNode {
                    alpha_start,
                    alpha_count,
                });
                node_depths.push(depth);
            }
            let node_count = gpu_nodes.len() as u32 - node_start;
            gpu_rays.push(GpuRay {
                node_start,
                node_count,
            });
        }

        let total_nodes = gpu_nodes.len();
        // Every ray empty -> no nodes to composite; mirror the golden's empty
        // curves without touching the device (buffers cannot be zero-sized).
        if total_nodes == 0 {
            return (0..ray_count)
                .map(|_| TransmittanceCurve::default())
                .collect();
        }

        let uniform = Params {
            ray_count: ray_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let device = ctx.device();
        let out_bytes = (total_nodes * size_of::<f32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_adaptive_accumulate_params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM,
        });
        let alphas_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_adaptive_accumulate_alphas"),
            contents: bytemuck::cast_slice(&alphas),
            usage: BufferUsages::STORAGE,
        });
        let nodes_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_adaptive_accumulate_nodes"),
            contents: bytemuck::cast_slice(&gpu_nodes),
            usage: BufferUsages::STORAGE,
        });
        let rays_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_adaptive_accumulate_rays"),
            contents: bytemuck::cast_slice(&gpu_rays),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_adaptive_accumulate_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_adaptive_accumulate_out_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_adaptive_accumulate_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: alphas_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: nodes_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: rays_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_adaptive_accumulate_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_adaptive_accumulate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (ray_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let transmittance = read_f32(&out_stage);

        // Reattach host-passthrough depths to the device-computed running
        // products, one curve per ray in input order.
        let mut curves = Vec::with_capacity(ray_count);
        for ray in &gpu_rays {
            let mut nodes = Vec::with_capacity(ray.node_count as usize);
            for n in 0..ray.node_count {
                let gi = (ray.node_start + n) as usize;
                nodes.push(TransmittanceNode {
                    depth: node_depths[gi],
                    transmittance: transmittance[gi],
                });
            }
            curves.push(TransmittanceCurve { nodes });
        }
        curves
    }
}

/// Reads a mapped staging buffer back into an owned `f32` vector, then unmaps.
fn read_f32(stage: &wgpu::Buffer) -> Vec<f32> {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    stage.unmap();
    out
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
