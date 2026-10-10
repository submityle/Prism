//! Real-device `GPU` opacity-micromap baker.
//!
//! [`GpuOmmBaker`] compiles the `omm_classify.wgsl` and `omm_pack.wgsl` kernels
//! once and exposes [`GpuOmmBaker::bake`], which classifies every
//! micro-triangle of one base triangle against a nearest-texel alpha mask and
//! packs the result into the `DXR` micromap byte layout entirely on the device.
//! A passing parity test against
//! [`bake_triangle`](prism_micromap::omm::bake_triangle) is direct evidence
//! that the ported kernels agree with the `CPU` golden element for element.
//!
//! # Host orchestration
//!
//! The bake is two dispatches. The classify pass runs one invocation per
//! micro-triangle, decoding the canonical index into row-strip coordinates,
//! sampling six barycentric anchors plus a uniform centroid grid, and writing a
//! normalised `2-bit` opacity code to a per-micro-triangle `u32` state buffer.
//! The pack pass then runs one invocation per output `u32` word, folding the
//! state codes into the little-endian within-byte micromap layout. Only the
//! final truncation to the exact packed byte length happens on the host.
//!
//! Provenance: classical conservative coverage classification and bit packing;
//! no Unreal Engine source or derived code, and no AI/ML.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_micromap::omm::{packed_len, OmmFormat, SubdivisionLevel, TextureAlphaMask, WrapMode};

use crate::buffer;
use crate::context::GpuContext;

/// Uniform parameters shared with `Params` in `omm_classify.wgsl`.
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
    /// Output format: `0` two-state, `1` four-state.
    format: u32,
    /// Uniform sub-subdivision segments per micro-triangle edge.
    samples: u32,
    /// Mask width in texels.
    width: u32,
    /// Mask height in texels.
    height: u32,
    /// Wrap mode: `0` repeat, `1` clamp.
    wrap: u32,
    /// Alpha cutoff: samples with `alpha >= threshold` are opaque.
    threshold: f32,
    /// Number of micro-triangles (`4^level`).
    micro_count: u32,
    /// Padding to a 16-word uniform block.
    pad0: u32,
    /// Padding to a 16-word uniform block.
    pad1: u32,
}

/// Uniform parameters shared with `PackParams` in `omm_pack.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackParams {
    /// Output format: `0` two-state, `1` four-state.
    format: u32,
    /// Number of micro-triangles to pack.
    count: u32,
    /// Padding to a 4-word uniform block.
    pad0: u32,
    /// Padding to a 4-word uniform block.
    pad1: u32,
}

/// A baked opacity micromap produced on the `GPU`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuBakedOmm {
    /// Per-micro-triangle opacity codes in canonical order, normalised to the
    /// output format (`0` transparent, `1` opaque, `2` unknown-transparent,
    /// `3` unknown-opaque).
    pub states: Vec<u8>,
    /// The packed `DXR`-layout micromap bytes, truncated to the exact length.
    pub data: Vec<u8>,
}

/// A compiled, reusable `GPU` opacity-micromap pipeline pair.
pub struct GpuOmmBaker {
    /// Kept alive so the classify pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    classify_module: ShaderModule,
    /// Kept alive so the pack pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    pack_module: ShaderModule,
    /// Layout wiring params, the mask, and the state output.
    classify_layout: BindGroupLayout,
    /// Layout wiring pack params, the states, and the packed output.
    pack_layout: BindGroupLayout,
    /// Classifies one micro-triangle per invocation.
    classify: ComputePipeline,
    /// Packs one output word per invocation.
    pack: ComputePipeline,
}

/// Builds a compute-visible buffer binding layout entry for `binding`.
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

impl GpuOmmBaker {
    /// Compiles the classify and pack kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuOmmBaker {
        let device = ctx.device();

        let classify_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_omm_classify"),
            source: ShaderSource::Wgsl(include_str!("../shaders/omm_classify.wgsl").into()),
        });
        let pack_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_omm_pack"),
            source: ShaderSource::Wgsl(include_str!("../shaders/omm_pack.wgsl").into()),
        });

        let classify_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_omm_classify_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pack_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_omm_pack_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });

        let classify_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_omm_classify_pipeline_layout"),
            bind_group_layouts: &[Some(&classify_layout)],
            immediate_size: 0,
        });
        let pack_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_omm_pack_pipeline_layout"),
            bind_group_layouts: &[Some(&pack_layout)],
            immediate_size: 0,
        });

        let classify = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_omm_classify_pipeline"),
            layout: Some(&classify_pipeline_layout),
            module: &classify_module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let pack = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_omm_pack_pipeline"),
            layout: Some(&pack_pipeline_layout),
            module: &pack_module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        GpuOmmBaker {
            classify_module,
            pack_module,
            classify_layout,
            pack_layout,
            classify,
            pack,
        }
    }

    /// Bakes one base triangle into a [`GpuBakedOmm`] on the `GPU`.
    ///
    /// `uv` are the three base-triangle texture coordinates, `level` the
    /// subdivision level, `format` the output encoding, `samples_per_edge` the
    /// uniform centroid-grid density (treated as `1` when `0`), and `mask` the
    /// nearest-texel alpha coverage. The returned codes and packed bytes equal
    /// [`bake_triangle`](prism_micromap::omm::bake_triangle) element for
    /// element.
    #[must_use]
    pub fn bake(
        &self,
        ctx: &GpuContext,
        uv: [[f32; 2]; 3],
        level: SubdivisionLevel,
        format: OmmFormat,
        samples_per_edge: u32,
        mask: &TextureAlphaMask,
    ) -> GpuBakedOmm {
        let device = ctx.device();
        let micro_count = level.micro_triangle_count();
        let width = mask.width();
        let height = mask.height();

        // Flatten the mask alpha row-major so `mask[y * width + x]` matches the
        // host's nearest-texel fetch.
        let mut alpha = Vec::with_capacity((width as usize) * (height as usize));
        for y in 0..height {
            for x in 0..width {
                alpha.push(mask.texel(x, y).unwrap_or(0.0));
            }
        }

        let format_code = match format {
            OmmFormat::TwoState => 0u32,
            OmmFormat::FourState => 1u32,
        };
        let wrap_code = match mask.wrap() {
            WrapMode::Repeat => 0u32,
            WrapMode::Clamp => 1u32,
        };

        let params = Params {
            ux0: uv[0][0],
            uy0: uv[0][1],
            ux1: uv[1][0],
            uy1: uv[1][1],
            ux2: uv[2][0],
            uy2: uv[2][1],
            n: level.segments(),
            format: format_code,
            samples: samples_per_edge,
            width,
            height,
            wrap: wrap_code,
            threshold: AlphaThreshold::of(mask),
            micro_count,
            pad0: 0,
            pad1: 0,
        };

        let params_buf = buffer::uniform(device, "prism_omm_params", &params);
        let mask_buf = buffer::storage_read(device, "prism_omm_mask", &alpha);
        let states_buf =
            buffer::storage_rw_zeroed(device, "prism_omm_states", u64::from(micro_count) * 4);

        let classify_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_omm_classify_bind"),
            layout: &self.classify_layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &mask_buf),
                entry(2, &states_buf),
            ],
        });

        let classify_groups = micro_count.div_ceil(64);
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_omm_classify_encoder"),
        });
        dispatch(
            &mut enc,
            "prism_omm_classify_pass",
            &self.classify,
            &classify_bind,
            classify_groups,
        );
        ctx.queue().submit([enc.finish()]);

        // One output word packs 16 four-state or 32 two-state micro-triangles.
        let word_count = match format {
            OmmFormat::FourState => micro_count.div_ceil(16),
            OmmFormat::TwoState => micro_count.div_ceil(32),
        };
        let pack_params = PackParams {
            format: format_code,
            count: micro_count,
            pad0: 0,
            pad1: 0,
        };
        let pack_params_buf = buffer::uniform(device, "prism_omm_pack_params", &pack_params);
        let out_buf =
            buffer::storage_rw_zeroed(device, "prism_omm_packed", u64::from(word_count) * 4);

        let pack_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_omm_pack_bind"),
            layout: &self.pack_layout,
            entries: &[
                entry(0, &pack_params_buf),
                entry(1, &states_buf),
                entry(2, &out_buf),
            ],
        });

        let states_stage =
            buffer::staging(device, "prism_omm_states_stage", u64::from(micro_count) * 4);
        let out_stage =
            buffer::staging(device, "prism_omm_packed_stage", u64::from(word_count) * 4);

        let pack_groups = word_count.div_ceil(64);
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_omm_pack_encoder"),
        });
        dispatch(
            &mut enc,
            "prism_omm_pack_pass",
            &self.pack,
            &pack_bind,
            pack_groups,
        );
        buffer::copy(
            &mut enc,
            &states_buf,
            &states_stage,
            u64::from(micro_count) * 4,
        );
        buffer::copy(&mut enc, &out_buf, &out_stage, u64::from(word_count) * 4);
        ctx.queue().submit([enc.finish()]);

        let state_words = buffer::read_back::<u32>(ctx, &states_stage);
        let packed_words = buffer::read_back::<u32>(ctx, &out_stage);

        let states: Vec<u8> = state_words
            .iter()
            .take(micro_count as usize)
            .map(|&w| w as u8)
            .collect();

        let mut data: Vec<u8> = Vec::with_capacity(packed_words.len() * 4);
        for word in &packed_words {
            data.extend_from_slice(&word.to_le_bytes());
        }
        data.truncate(packed_len(micro_count, format));

        GpuBakedOmm { states, data }
    }
}

/// Reads the alpha threshold of a mask through the `AlphaMask` trait.
struct AlphaThreshold;

impl AlphaThreshold {
    /// Returns the mask's alpha cutoff.
    fn of(mask: &TextureAlphaMask) -> f32 {
        use prism_micromap::omm::AlphaMask;
        mask.threshold()
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
