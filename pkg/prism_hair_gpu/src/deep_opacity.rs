//! `wgpu` compute twin of Prism's host deep opacity map packing
//! ([`build_deep_opacity_map`](prism_render_architecture::hair::deep_opacity_layout::build_deep_opacity_map)).
//!
//! A groom cannot self-shadow from a single shadow-map depth: thousands of thin
//! strands overlap along every light ray, so — like `UE5` Groom and AMD
//! `TressFX` self-shadow bakes — each light texel accumulates strand opacity
//! into a fixed number of depth layers whose cumulative transmittance
//! `T = product(1 - alpha)` a shading pass decodes with a constant stride. The
//! reference builds this slab on the host by, per texel, stably sorting the
//! samples by light-space depth, slicing `layer_count` equal-width layers over
//! `shallowest + start_offset ..= deepest`, and compositing the running product
//! `product(1 - clamp(opacity))`. This crate is the on-device twin: one thread
//! per light texel walks the identical slice/composite in the identical order,
//! so a passing real-device parity test is direct evidence the ported kernel
//! packs the same slab as the reference — not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairDeepOpacity::eval`] takes the per-texel [`TransmittanceBins`], a
//! fixed `layer_count` and a near-bias `start_offset`, and returns the packed
//! [`DeepOpacityMap`] (per-texel `near_depth`/`layer_step` plus the flat
//! texel-major transmittance grid). The *sort* stays on the host — the golden
//! documents the device-side sort as separate scheduling — so `eval` reproduces
//! `pack_bucket`'s stable `total_cmp` ordering on the `CPU`, flattens each
//! texel's already-sorted slice into a shared `samples` pool with a compacted
//! `texel_ranges` array, and the kernel only slices layers and composites.
//!
//! # Portability
//!
//! The kernel uses only `max`, `clamp` and multiply/add in the portable
//! core-`WGSL` subset — no `exp`, `pow` or optional device feature — so the
//! twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The composite is a closed-form running product with no transcendental call,
//! so `CPU` and `GPU` evaluate the same arithmetic. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few `ULP`. The parity test therefore
//! asserts a per-value tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`)
//! rather than exact equality.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard deep opacity map packing (per-texel depth sort, fixed
//! equal-width layers, `alpha`-composite product) plus `wgpu` compute dispatch;
//! no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::deep_opacity_layout::DeepOpacityMap;
use prism_render_architecture::hair::deep_transmittance::TransmittanceBins;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform packing parameters uploaded to the kernel. `16`-byte scalar-packed
/// `repr(C)` matching `HairDeepOpacityParams` in `shaders/deep_opacity.wesl`
/// (padded to a multiple of 16 for the uniform block).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    texel_count: u32,
    layer_count: u32,
    start_offset: f32,
    pad: u32,
}

/// One compacted per-texel slice descriptor uploaded to the kernel. `8`-byte
/// `repr(C)` matching `HairTexelRange` in `shaders/deep_opacity.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTexelRange {
    start: u32,
    count: u32,
}

/// A compiled, reusable deep opacity packing pipeline.
pub struct GpuHairDeepOpacity {
    #[expect(
        dead_code,
        reason = "the shader module must outlive the pipeline that borrows it"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairDeepOpacity {
    /// Compiles the deep opacity packing kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairDeepOpacity {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_deep_opacity"),
            source: ShaderSource::Wgsl(include_str!("../shaders/deep_opacity.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_deep_opacity_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_deep_opacity_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_deep_opacity_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairDeepOpacity {
            module,
            layout,
            pipeline,
        }
    }

    /// Packs a [`DeepOpacityMap`] from the per-texel strand `bins`, sliced into
    /// `layer_count` depth layers with near bias `start_offset`.
    ///
    /// The result equals
    /// [`build_deep_opacity_map`](prism_render_architecture::hair::deep_opacity_layout::build_deep_opacity_map)
    /// to within the fused-multiply-add tolerance documented on this module.
    /// The host reproduces `pack_bucket`'s stable `total_cmp` depth sort before
    /// upload (the kernel does no sorting), so the packed slab decodes
    /// bit-for-bit like the golden. When every bucket is empty the map is still
    /// dispatched so empty texels pack their fully transmissive rows; a
    /// degenerate map with zero texels returns the empty golden without a
    /// dispatch (storage buffers cannot be zero-sized).
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        bins: &TransmittanceBins,
        layer_count: u32,
        start_offset: f32,
    ) -> DeepOpacityMap {
        let layer_count = layer_count.max(1) as usize;
        let texel_count = bins.len();

        // A zero-texel map has no rows to pack; mirror the golden's empty map
        // without touching the device (buffers cannot be zero-sized).
        if texel_count == 0 {
            return DeepOpacityMap {
                texel_count: 0,
                layer_count,
                near_depth: Vec::new(),
                layer_step: Vec::new(),
                transmittance: Vec::new(),
            };
        }

        // Flatten each texel's samples into a shared pool, reproducing
        // `pack_bucket`'s stable ascending `total_cmp` sort on the host so the
        // kernel receives an already-sorted slice per texel.
        let mut samples: Vec<[f32; 2]> = Vec::with_capacity(bins.total());
        let mut ranges: Vec<GpuTexelRange> = Vec::with_capacity(texel_count);
        for texel in 0..texel_count {
            let bucket = bins.bucket(texel as u32).unwrap_or(&[]);
            let start = samples.len() as u32;
            let mut sorted = bucket.to_vec();
            sorted.sort_by(|a, b| a.depth.total_cmp(&b.depth));
            for s in &sorted {
                samples.push([s.depth, s.opacity]);
            }
            ranges.push(GpuTexelRange {
                start,
                count: bucket.len() as u32,
            });
        }

        // Storage buffers cannot be zero-sized: pad the sample pool with one
        // dummy entry when every bucket was empty. The `count == 0` ranges keep
        // every texel on the empty-row path, so the dummy is never read.
        if samples.is_empty() {
            samples.push([0.0, 0.0]);
        }

        let uniform = Params {
            texel_count: texel_count as u32,
            layer_count: layer_count as u32,
            start_offset,
            pad: 0,
        };

        let device = ctx.device();
        let near_bytes = (texel_count * size_of::<f32>()) as u64;
        let trans_bytes = (texel_count * layer_count * size_of::<f32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_deep_opacity_params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM,
        });
        let samples_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_deep_opacity_samples"),
            contents: bytemuck::cast_slice(&samples),
            usage: BufferUsages::STORAGE,
        });
        let ranges_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_deep_opacity_ranges"),
            contents: bytemuck::cast_slice(&ranges),
            usage: BufferUsages::STORAGE,
        });
        let near_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_deep_opacity_near"),
            size: near_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let step_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_deep_opacity_step"),
            size: near_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let trans_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_deep_opacity_trans"),
            size: trans_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let near_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_deep_opacity_near_stage"),
            size: near_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let step_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_deep_opacity_step_stage"),
            size: near_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let trans_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_deep_opacity_trans_stage"),
            size: trans_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_deep_opacity_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: samples_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: ranges_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: near_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: step_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: trans_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_deep_opacity_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_deep_opacity_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (texel_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&near_buf, 0, &near_stage, 0, near_bytes);
        encoder.copy_buffer_to_buffer(&step_buf, 0, &step_stage, 0, near_bytes);
        encoder.copy_buffer_to_buffer(&trans_buf, 0, &trans_stage, 0, trans_bytes);
        ctx.queue().submit([encoder.finish()]);

        near_stage.slice(..).map_async(MapMode::Read, |_| {});
        step_stage.slice(..).map_async(MapMode::Read, |_| {});
        trans_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let near_depth = read_f32(&near_stage);
        let layer_step = read_f32(&step_stage);
        let transmittance = read_f32(&trans_stage);

        DeepOpacityMap {
            texel_count,
            layer_count,
            near_depth,
            layer_step,
            transmittance,
        }
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
