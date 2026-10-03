//! Real-device `GPU` displaced-micro-map baker.
//!
//! [`GpuDmmBaker`] compiles the four `DMM` kernels once and exposes
//! [`GpuDmmBaker::bake`], which samples, reduces, quantizes, and bit-packs one
//! base triangle's per-micro-vertex displacement entirely on the device. A
//! passing parity test against
//! [`bake_triangle`](prism_dmm::bake_triangle) is direct evidence that the
//! ported kernels agree with the `CPU` golden element for element.
//!
//! # Host orchestration
//!
//! The bake is four dispatches recorded into a single encoder:
//!
//! 1. the *sample* pass runs one invocation per micro-vertex, decoding the
//!    canonical lattice index, interpolating the micro-vertex `UV`, and
//!    bilinearly sampling the displacement height into a per-vertex `f32`
//!    buffer;
//! 2. the *reduce* pass runs a single invocation that scans the sampled
//!    heights for the per-triangle `[min, max]` range (or copies the
//!    caller-provided fixed range);
//! 3. the *quantize* pass runs one invocation per micro-vertex, normalising
//!    each height and rounding it to an `11-bit` unorm code; and
//! 4. the *pack* pass runs one invocation per output `u32` word, folding the
//!    codes into the raw little-endian `11-bit` bitstream.
//!
//! Only the final truncation to the exact packed byte length happens on the
//! host.
//!
//! Provenance: classical barycentric interpolation, bilinear filtering,
//! round-to-nearest quantization, and bit packing; no Unreal Engine source or
//! derived code, and no AI/ML.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_dmm::{
    packed_len, DmmSubdivisionLevel, ScaleBiasMode, TextureDisplacementMap, WrapMode,
};

use crate::buffer;
use crate::context::GpuContext;

/// Uniform parameters shared with `Params` in the sample, reduce, and quantize
/// kernels.
///
/// Laid out as a `16`-word (`64`-byte) block whose field order matches the
/// `WGSL` `Params` struct exactly.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// `U` of base-triangle vertex `0`.
    ux0: f32,
    /// `V` of base-triangle vertex `0`.
    uy0: f32,
    /// `U` of base-triangle vertex `1`.
    ux1: f32,
    /// `V` of base-triangle vertex `1`.
    uy1: f32,
    /// `U` of base-triangle vertex `2`.
    ux2: f32,
    /// `V` of base-triangle vertex `2`.
    uy2: f32,
    /// Edge-segment count `n == 2^level`.
    n: u32,
    /// Height-texture width in texels.
    width: u32,
    /// Height-texture height in texels.
    height: u32,
    /// Wrap mode: `0` repeat, `1` clamp.
    wrap: u32,
    /// Scale/bias mode: `0` per-triangle, `1` fixed.
    mode: u32,
    /// Number of micro-vertices to process.
    vertex_count: u32,
    /// Fixed-mode lower height bound (ignored in per-triangle mode).
    fixed_min: f32,
    /// Fixed-mode upper height bound (ignored in per-triangle mode).
    fixed_max: f32,
    /// Padding to a `16`-word uniform block.
    pad0: u32,
    /// Padding to a `16`-word uniform block.
    pad1: u32,
}

/// Uniform parameters shared with `PackParams` in the pack kernel.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackParams {
    /// Number of `11-bit` codes to pack.
    count: u32,
    /// Number of output `u32` words.
    word_count: u32,
    /// Padding to a `4`-word uniform block.
    pad0: u32,
    /// Padding to a `4`-word uniform block.
    pad1: u32,
}

/// A baked displaced micro-map produced on the `GPU`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuBakedDmm {
    /// Per-micro-vertex `11-bit` codes in canonical order.
    pub codes: Vec<u16>,
    /// The raw packed little-endian `11-bit` bitstream, truncated to the exact
    /// length.
    pub data: Vec<u8>,
}

/// A compiled, reusable `GPU` displaced-micro-map pipeline set.
pub struct GpuDmmBaker {
    /// Kept alive so the sample pipeline it produced stays valid.
    #[expect(dead_code, reason = "kept alive so the pipeline it produced stays valid")]
    sample_module: ShaderModule,
    /// Kept alive so the reduce pipeline it produced stays valid.
    #[expect(dead_code, reason = "kept alive so the pipeline it produced stays valid")]
    reduce_module: ShaderModule,
    /// Kept alive so the quantize pipeline it produced stays valid.
    #[expect(dead_code, reason = "kept alive so the pipeline it produced stays valid")]
    quantize_module: ShaderModule,
    /// Kept alive so the pack pipeline it produced stays valid.
    #[expect(dead_code, reason = "kept alive so the pipeline it produced stays valid")]
    pack_module: ShaderModule,
    /// Layout wiring params, the height texture, and the sampled heights.
    sample_layout: BindGroupLayout,
    /// Layout wiring params, the heights, and the scale/bias output.
    reduce_layout: BindGroupLayout,
    /// Layout wiring params, the heights, the scale/bias, and the codes.
    quantize_layout: BindGroupLayout,
    /// Layout wiring pack params, the codes, and the packed words.
    pack_layout: BindGroupLayout,
    /// The compiled sample pipeline.
    sample: ComputePipeline,
    /// The compiled reduce pipeline.
    reduce: ComputePipeline,
    /// The compiled quantize pipeline.
    quantize: ComputePipeline,
    /// The compiled pack pipeline.
    pack: ComputePipeline,
}

/// Builds a compute-visible bind-group-layout entry for `binding` of type `ty`.
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

/// Builds a bind-group entry binding all of `buffer` to `binding`.
fn entry(binding: u32, buffer: &Buffer) -> BindGroupEntry<'_> {
    BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

impl GpuDmmBaker {
    /// Compiles the sample, reduce, quantize, and pack kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDmmBaker {
        let device = ctx.device();

        let sample_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_dmm_sample"),
            source: ShaderSource::Wgsl(include_str!("../shaders/dmm_sample.wgsl").into()),
        });
        let reduce_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_dmm_reduce"),
            source: ShaderSource::Wgsl(include_str!("../shaders/dmm_reduce.wgsl").into()),
        });
        let quantize_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_dmm_quantize"),
            source: ShaderSource::Wgsl(include_str!("../shaders/dmm_quantize.wgsl").into()),
        });
        let pack_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_dmm_pack"),
            source: ShaderSource::Wgsl(include_str!("../shaders/dmm_pack.wgsl").into()),
        });

        let sample_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_dmm_sample_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let reduce_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_dmm_reduce_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let quantize_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_dmm_quantize_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pack_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_dmm_pack_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });

        let sample_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_dmm_sample_pipeline_layout"),
            bind_group_layouts: &[Some(&sample_layout)],
            immediate_size: 0,
        });
        let reduce_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_dmm_reduce_pipeline_layout"),
            bind_group_layouts: &[Some(&reduce_layout)],
            immediate_size: 0,
        });
        let quantize_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_dmm_quantize_pipeline_layout"),
            bind_group_layouts: &[Some(&quantize_layout)],
            immediate_size: 0,
        });
        let pack_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_dmm_pack_pipeline_layout"),
            bind_group_layouts: &[Some(&pack_layout)],
            immediate_size: 0,
        });

        let sample = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_dmm_sample_pipeline"),
            layout: Some(&sample_pipeline_layout),
            module: &sample_module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let reduce = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_dmm_reduce_pipeline"),
            layout: Some(&reduce_pipeline_layout),
            module: &reduce_module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let quantize = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_dmm_quantize_pipeline"),
            layout: Some(&quantize_pipeline_layout),
            module: &quantize_module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let pack = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_dmm_pack_pipeline"),
            layout: Some(&pack_pipeline_layout),
            module: &pack_module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        GpuDmmBaker {
            sample_module,
            reduce_module,
            quantize_module,
            pack_module,
            sample_layout,
            reduce_layout,
            quantize_layout,
            pack_layout,
            sample,
            reduce,
            quantize,
            pack,
        }
    }

    /// Bakes one base triangle into a [`GpuBakedDmm`] on the `GPU`.
    ///
    /// `uv` are the three base-triangle texture coordinates, `level` the
    /// subdivision level, `scale_bias_mode` whether the quantization range is
    /// derived per triangle or fixed, and `map` the dense height texture. The
    /// returned codes and packed bytes equal
    /// [`bake_triangle`](prism_dmm::bake_triangle) element for element.
    #[must_use]
    pub fn bake(
        &self,
        ctx: &GpuContext,
        uv: [[f32; 2]; 3],
        level: DmmSubdivisionLevel,
        scale_bias_mode: ScaleBiasMode,
        map: &TextureDisplacementMap,
    ) -> GpuBakedDmm {
        let device = ctx.device();
        let vertex_count = level.micro_vertex_count();
        let width = map.width();
        let height = map.height();

        let wrap_code = match map.wrap() {
            WrapMode::Repeat => 0u32,
            WrapMode::Clamp => 1u32,
        };
        let (mode, fixed_min, fixed_max) = match scale_bias_mode {
            ScaleBiasMode::PerTriangle => (0u32, 0.0, 0.0),
            ScaleBiasMode::Fixed(sb) => (1u32, sb.min, sb.max),
        };

        let params = Params {
            ux0: uv[0][0],
            uy0: uv[0][1],
            ux1: uv[1][0],
            uy1: uv[1][1],
            ux2: uv[2][0],
            uy2: uv[2][1],
            n: level.segments(),
            width,
            height,
            wrap: wrap_code,
            mode,
            vertex_count,
            fixed_min,
            fixed_max,
            pad0: 0,
            pad1: 0,
        };

        // One output word carries 32 raw bits of the 11-bit stream.
        let bits = vertex_count * 11;
        let word_count = bits.div_ceil(32);

        let params_buf = buffer::uniform(device, "prism_dmm_params", &params);
        let heightmap_buf = buffer::storage_read(device, "prism_dmm_heightmap", map.data());
        let heights_buf =
            buffer::storage_rw_zeroed(device, "prism_dmm_heights", u64::from(vertex_count) * 4);
        // Two f32 scale/bias slots, padded to a 16-byte block.
        let scalebias_buf = buffer::storage_rw_zeroed(device, "prism_dmm_scalebias", 16);
        let codes_buf =
            buffer::storage_rw_zeroed(device, "prism_dmm_codes", u64::from(vertex_count) * 4);
        let out_buf =
            buffer::storage_rw_zeroed(device, "prism_dmm_packed", u64::from(word_count) * 4);

        let pack_params = PackParams {
            count: vertex_count,
            word_count,
            pad0: 0,
            pad1: 0,
        };
        let pack_params_buf = buffer::uniform(device, "prism_dmm_pack_params", &pack_params);

        let sample_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_dmm_sample_bind"),
            layout: &self.sample_layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &heightmap_buf),
                entry(2, &heights_buf),
            ],
        });
        let reduce_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_dmm_reduce_bind"),
            layout: &self.reduce_layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &heights_buf),
                entry(2, &scalebias_buf),
            ],
        });
        let quantize_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_dmm_quantize_bind"),
            layout: &self.quantize_layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &heights_buf),
                entry(2, &scalebias_buf),
                entry(3, &codes_buf),
            ],
        });
        let pack_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_dmm_pack_bind"),
            layout: &self.pack_layout,
            entries: &[
                entry(0, &pack_params_buf),
                entry(1, &codes_buf),
                entry(2, &out_buf),
            ],
        });

        let codes_stage =
            buffer::staging(device, "prism_dmm_codes_stage", u64::from(vertex_count) * 4);
        let out_stage =
            buffer::staging(device, "prism_dmm_packed_stage", u64::from(word_count) * 4);

        let vertex_groups = vertex_count.div_ceil(64);
        let pack_groups = word_count.div_ceil(64);
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_dmm_bake_encoder"),
        });
        dispatch(&mut enc, "prism_dmm_sample_pass", &self.sample, &sample_bind, vertex_groups);
        dispatch(&mut enc, "prism_dmm_reduce_pass", &self.reduce, &reduce_bind, 1);
        dispatch(
            &mut enc,
            "prism_dmm_quantize_pass",
            &self.quantize,
            &quantize_bind,
            vertex_groups,
        );
        dispatch(&mut enc, "prism_dmm_pack_pass", &self.pack, &pack_bind, pack_groups);
        buffer::copy(&mut enc, &codes_buf, &codes_stage, u64::from(vertex_count) * 4);
        buffer::copy(&mut enc, &out_buf, &out_stage, u64::from(word_count) * 4);
        ctx.queue().submit([enc.finish()]);

        let code_words = buffer::read_back::<u32>(ctx, &codes_stage);
        let packed_words = buffer::read_back::<u32>(ctx, &out_stage);

        let codes: Vec<u16> = code_words
            .iter()
            .take(vertex_count as usize)
            .map(|&w| w as u16)
            .collect();

        let mut data: Vec<u8> = Vec::with_capacity(packed_words.len() * 4);
        for word in &packed_words {
            data.extend_from_slice(&word.to_le_bytes());
        }
        data.truncate(packed_len(vertex_count as usize));

        GpuBakedDmm { codes, data }
    }
}

/// Records one one-dimensional dispatch of `pipeline` bound to `bind`.
fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
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
    pass.dispatch_workgroups(groups.max(1), 1, 1);
}
