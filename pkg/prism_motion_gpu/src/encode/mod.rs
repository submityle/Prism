//! Host orchestration of the compact motion-vector encode/pack kernel.
//!
//! [`GpuEncodeMotion`] compiles `shaders/encode_motion.wgsl` once and runs the
//! per-pixel quantiser that mirrors the CPU golden
//! [`encode_sample`](prism_render_architecture::motion::encode::encode_sample):
//! each pixel's velocity is normalised by the encoding full-scale, clamped, and
//! quantised to a signed 16-bit pair, while the reactive / transparency /
//! confidence signals are quantised to `unorm8` and packed into a `u32` whose
//! top byte carries the per-pixel flags (with `TRANSPARENT` forced on when the
//! sample's transparency is positive, exactly as the golden does).
//!
//! Because the kernel reproduces the golden's branch-on-`NaN`-first clamps and
//! its explicit half-step rounding bias (truncate toward zero, never
//! round-half-even), and the clamp bounds keep every truncating cast inside the
//! integer range, the device output equals the golden bit-for-bit: the parity
//! test compares whole [`EncodedMotion`](prism_render_architecture::motion::encode::EncodedMotion)
//! records for exact equality rather than a tolerance.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine source or derived
//! code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::motion::MotionSample;
use prism_render_architecture::motion::encode::{
    EncodedMotion, PackedMasks, VelocityEncoding, flags,
};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

/// Compute workgroup length (the kernel is `@workgroup_size(64, 1, 1)`).
pub const GROUP: u32 = 64;

/// `Pod` mirror of the shader's per-pixel velocity (`vec2<f32>`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVec2 {
    x: f32,
    y: f32,
}

/// `Pod` mirror of the shader's per-pixel mask inputs
/// (`reactive`, `transparency`, `confidence`, unused), a `vec4<f32>`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuMasks {
    reactive: f32,
    transparency: f32,
    confidence: f32,
    unused: f32,
}

/// Uniform block shared with `Params` in `encode_motion.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    max_velocity_pixels: f32,
    pad0: u32,
    pad1: u32,
}

/// `Pod` mirror of the shader's `Encoded` output record.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuEncoded {
    vx: i32,
    vy: i32,
    masks: u32,
    pad: u32,
}

impl GpuEncoded {
    /// Narrows the widened `i32` velocity pair back to the golden
    /// [`EncodedMotion`] record. The kernel's clamp bounds guarantee each
    /// component is inside the `i16` range.
    fn to_encoded_motion(self) -> EncodedMotion {
        EncodedMotion {
            velocity: [
                i16::try_from(self.vx).expect("snorm16 x in range by clamp"),
                i16::try_from(self.vy).expect("snorm16 y in range by clamp"),
            ],
            masks: PackedMasks(self.masks),
        }
    }
}

/// Compiled encode pipeline and its bind-group layout.
pub struct GpuEncodeMotion {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuEncodeMotion {
    /// Compiles the motion-vector encode kernel on `ctx`'s device.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuEncodeMotion {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_motion_encode"),
            source: ShaderSource::Wgsl(include_str!("../shaders/encode_motion.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_motion_encode_layout"),
            entries: &[
                buffer_layout(0, BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
                buffer_layout(1, storage_read_ty()),
                buffer_layout(2, storage_read_ty()),
                buffer_layout(3, storage_read_ty()),
                buffer_layout(4, BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_motion_encode_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_motion_encode_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuEncodeMotion { pipeline, layout }
    }

    /// Encodes `samples` on the device, returning one [`EncodedMotion`] per
    /// pixel in order.
    ///
    /// `extra_flags` supplies the caller's per-pixel flag byte and must have the
    /// same length as `samples`; a length mismatch returns [`None`]. An empty
    /// input returns an empty vector. The result equals mapping the golden
    /// [`encode_sample`](prism_render_architecture::motion::encode::encode_sample)
    /// over the samples bit-for-bit.
    #[must_use]
    pub fn encode(
        &self,
        ctx: &GpuContext,
        samples: &[MotionSample],
        encoding: VelocityEncoding,
        extra_flags: &[u8],
    ) -> Option<Vec<EncodedMotion>> {
        if samples.len() != extra_flags.len() {
            return None;
        }
        let count = samples.len();
        if count == 0 {
            return Some(Vec::new());
        }

        let device = ctx.device();

        let velocity: Vec<GpuVec2> = samples
            .iter()
            .map(|s| GpuVec2 {
                x: s.velocity_pixels[0],
                y: s.velocity_pixels[1],
            })
            .collect();
        let masks_in: Vec<GpuMasks> = samples
            .iter()
            .map(|s| GpuMasks {
                reactive: s.reactive,
                transparency: s.transparency,
                confidence: s.reprojection_confidence,
                unused: 0.0,
            })
            .collect();
        let flags_in: Vec<u32> = extra_flags.iter().map(|&f| u32::from(f)).collect();

        let params = Params {
            count: u32::try_from(count).expect("pixel count fits in u32"),
            max_velocity_pixels: encoding.max_velocity_pixels,
            pad0: 0,
            pad1: 0,
        };

        let out_bytes = (count * size_of::<GpuEncoded>()) as u64;
        let params_buf = buffer::uniform(device, "prism_motion_encode_params", &params);
        let vel_buf = buffer::storage_read(device, "prism_motion_encode_velocity", &velocity);
        let masks_buf = buffer::storage_read(device, "prism_motion_encode_masks", &masks_in);
        let flags_buf = buffer::storage_read(device, "prism_motion_encode_flags", &flags_in);
        let out_buf = buffer::storage_rw_zeroed(device, "prism_motion_encode_out", out_bytes);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_motion_encode_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: vel_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: masks_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: flags_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_motion_encode_encoder"),
        });
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_motion_encode_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(params.count.div_ceil(GROUP), 1, 1);
        }

        let stage = buffer::staging(device, "prism_motion_encode_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        let words = buffer::read_back::<GpuEncoded>(ctx, &stage);
        Some(
            words
                .iter()
                .take(count)
                .map(|&w| w.to_encoded_motion())
                .collect(),
        )
    }
}

/// A read-only storage buffer layout entry.
fn storage_read_ty() -> BindingType {
    BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

/// One storage/uniform bind-group-layout entry visible to the compute stage.
fn buffer_layout(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

// Pull `flags` into scope for documentation links above; the module constant
// `flags::TRANSPARENT` is mirrored in the shader as `FLAG_TRANSPARENT`.
#[expect(unused_imports, reason = "imported for the rustdoc link to flags::TRANSPARENT")]
use flags as _flags_doc_anchor;
