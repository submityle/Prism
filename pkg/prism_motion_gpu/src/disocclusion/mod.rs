//! Host orchestration of the per-pixel disocclusion (history-rejection) kernel.
//!
//! [`GpuDisocclusion`] compiles `shaders/disocclusion.wgsl` once and runs the
//! per-pixel classifier that mirrors the CPU golden
//! [`classify`](prism_render_architecture::motion::disocclusion::classify):
//! each reprojected history sample is tested against the current surface by
//! surface-id identity, depth continuity, and normal continuity, producing a
//! graded confidence in `[0, 1]`, a hard accept/reject verdict, and the
//! rejection reason bits that drive the
//! [`DISOCCLUDED`](prism_render_architecture::motion::encode::flags::DISOCCLUDED)
//! flag.
//!
//! The host sanitises the thresholds through
//! [`DisocclusionParams`](prism_render_architecture::motion::disocclusion::DisocclusionParams)
//! before upload, so the kernel's `max`/`clamp` flooring reproduces the golden
//! arithmetic on finite inputs. The discrete verdict (accepted flag and
//! rejection reason bits) matches the golden exactly; the graded confidence
//! matches to a tight tolerance because a fast-math backend may contract the
//! normal dot product to an FMA (the parity test asserts both accordingly).
//!
//! The golden [`RejectionReasons`](prism_render_architecture::motion::disocclusion::RejectionReasons)
//! exposes no public multi-bit constructor, so the device result is returned as
//! [`DisocclusionResult`], a small value type carrying the accepted flag, the
//! confidence, and the raw reason bits (comparable to
//! `RejectionReasons::bits`).
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine source or derived
//! code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::motion::disocclusion::{DisocclusionParams, SurfacePoint};
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

/// The device verdict for one pixel.
///
/// Mirrors the golden
/// [`DisocclusionVerdict`](prism_render_architecture::motion::disocclusion::DisocclusionVerdict)
/// but carries the rejection reasons as the raw `u32` bit set, since the golden
/// `RejectionReasons` has no public multi-bit constructor. The bits match
/// `RejectionReasons::bits` exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisocclusionResult {
    /// `true` when the reprojected history should be blended in.
    pub accepted: bool,
    /// Graded validity in `[0, 1]`, usable as a history blend weight.
    pub confidence: f32,
    /// Raw rejection reason bits (see `RejectionReasons`).
    pub reasons: u32,
}

/// `Pod` mirror of the shader's `Surface` input record (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSurface {
    nx: f32,
    ny: f32,
    nz: f32,
    depth: f32,
    sid_lo: u32,
    sid_hi: u32,
    pad0: u32,
    pad1: u32,
}

impl GpuSurface {
    fn from_point(p: SurfacePoint) -> GpuSurface {
        GpuSurface {
            nx: p.normal[0],
            ny: p.normal[1],
            nz: p.normal[2],
            depth: p.depth,
            sid_lo: (p.surface_id & 0xFFFF_FFFF) as u32,
            sid_hi: (p.surface_id >> 32) as u32,
            pad0: 0,
            pad1: 0,
        }
    }
}

/// Uniform block shared with `Params` in `disocclusion.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    require_surface_match: u32,
    depth_relative_tolerance: f32,
    normal_cos_threshold: f32,
    accept_threshold: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// `Pod` mirror of the shader's `Verdict` output record.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVerdict {
    accepted: u32,
    confidence: f32,
    reasons: u32,
    pad: u32,
}

impl GpuVerdict {
    fn to_result(self) -> DisocclusionResult {
        DisocclusionResult {
            accepted: self.accepted != 0,
            confidence: self.confidence,
            reasons: self.reasons,
        }
    }
}

/// Compiled disocclusion pipeline and its bind-group layout.
pub struct GpuDisocclusion {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuDisocclusion {
    /// Compiles the disocclusion classifier on `ctx`'s device.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDisocclusion {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_motion_disocclusion"),
            source: ShaderSource::Wgsl(include_str!("../shaders/disocclusion.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_motion_disocclusion_layout"),
            entries: &[
                buffer_layout(0, BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
                buffer_layout(1, storage_read_ty()),
                buffer_layout(2, storage_read_ty()),
                buffer_layout(3, BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_motion_disocclusion_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_motion_disocclusion_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDisocclusion { pipeline, layout }
    }

    /// Classifies each `(current, history)` pair on the device, returning one
    /// [`DisocclusionResult`] per pixel in order.
    ///
    /// `current` and `history` must have the same length; a mismatch returns
    /// [`None`]. An empty input returns an empty vector. The discrete verdict
    /// (accepted flag and reason bits) equals mapping the golden
    /// [`classify`](prism_render_architecture::motion::disocclusion::classify)
    /// over the pairs; the confidence matches to a tight tolerance.
    #[must_use]
    pub fn classify(
        &self,
        ctx: &GpuContext,
        current: &[SurfacePoint],
        history: &[SurfacePoint],
        params: DisocclusionParams,
    ) -> Option<Vec<DisocclusionResult>> {
        if current.len() != history.len() {
            return None;
        }
        let count = current.len();
        if count == 0 {
            return Some(Vec::new());
        }

        let device = ctx.device();

        let cur: Vec<GpuSurface> = current.iter().map(|&p| GpuSurface::from_point(p)).collect();
        let hist: Vec<GpuSurface> = history.iter().map(|&p| GpuSurface::from_point(p)).collect();

        let params_gpu = Params {
            count: u32::try_from(count).expect("pixel count fits in u32"),
            require_surface_match: u32::from(params.require_surface_match),
            depth_relative_tolerance: params.depth_relative_tolerance,
            normal_cos_threshold: params.normal_cos_threshold,
            accept_threshold: params.accept_threshold,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (count * size_of::<GpuVerdict>()) as u64;
        let params_buf = buffer::uniform(device, "prism_motion_disocc_params", &params_gpu);
        let cur_buf = buffer::storage_read(device, "prism_motion_disocc_current", &cur);
        let hist_buf = buffer::storage_read(device, "prism_motion_disocc_history", &hist);
        let out_buf = buffer::storage_rw_zeroed(device, "prism_motion_disocc_out", out_bytes);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_motion_disocc_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: cur_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: hist_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_motion_disocc_encoder"),
        });
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_motion_disocc_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(params_gpu.count.div_ceil(GROUP), 1, 1);
        }

        let stage = buffer::staging(device, "prism_motion_disocc_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        let words = buffer::read_back::<GpuVerdict>(ctx, &stage);
        Some(words.iter().take(count).map(|&w| w.to_result()).collect())
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
