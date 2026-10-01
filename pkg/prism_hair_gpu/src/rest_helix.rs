//! `wgpu` compute twin of Prism's per-strand curly-hair rest-state builder
//! ([`build_rest_helix`](prism_render_architecture::hair::rest_helix::build_rest_helix)).
//!
//! Type-3/type-4 textured hair stores a helical rest shape (not a straight
//! line) and bends anisotropically: tightening the curl along the fibre tangent
//! is far cheaper than bending out of the coil plane. The `CPU` golden builds,
//! per strand, the coil vertices, their per-segment rest lengths, the
//! per-interior-joint discrete `darboux` (curvature binormal) and the two
//! per-segment compliances. This crate is the on-device twin: each `GPU` thread
//! owns one strand and walks the same body value-for-value, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same rest state as the reference — not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuRestHelix::eval`] takes a batch of [`GpuRestHelixStrand`] roots and a
//! single coil-segment count shared by the whole dispatch (the fixed output
//! stride), and returns one [`GpuRestHelixOut`] per strand. The coil axis passes
//! through the root along the normalized tangent; starting from a transverse
//! offset of `radius`, each segment advances `pitch_per_segment` along the axis
//! while the offset is rotated by a unit-complex step. The per-segment rest
//! lengths, per-joint discrete `darboux` curvature and the anisotropic
//! tangential/normal compliances are derived from the resulting vertices,
//! exactly as [`build_rest_helix`](prism_render_architecture::hair::rest_helix::build_rest_helix).
//!
//! # Portability
//!
//! The kernel is trig-free: the transverse offset is stepped by a unit-complex
//! multiply, so no `sin`, `cos` or `pow` is ever called. It uses only `sqrt`,
//! `dot`, `cross`, `clamp` and multiply/add in the portable core-`WGSL` subset —
//! no optional device feature — so the twin runs unmodified on Metal, Vulkan and
//! DX12.
//!
//! # Correctness model
//!
//! The host re-normalizes the unit-complex rotation step and clamps the stiffness
//! ratio exactly as the golden's `sanitized` does; segment counts are clamped
//! into `1..=MAX_SEGMENTS` before the dispatch. Inputs are then assumed finite
//! and legal, on which domain the golden's `NaN`/infinity sanitation is
//! idempotent, so the twin mirrors the arithmetic path without the non-finite
//! guard core-`WGSL` lacks. `CPU` and `GPU` evaluate the same closed-form
//! geometry in the same order but are not bit-exact: a `GPU` may fuse a
//! multiply-add, perturbing the low mantissa bits, so the parity test asserts a
//! per-component tolerance rather than exact equality.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard discrete-elastic-rod helical rest construction plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::hair::rest_helix::MAX_SEGMENTS;

use crate::context::GpuContext;

/// Number of `f32` lanes one strand occupies in the packed input pool:
/// `[root.xyz, tangent.xyz, radius, pitch, rot_cos, rot_sin, ratio]`.
const STRAND_STRIDE: usize = 11;

/// One strand's coil-root parameters, mirroring the arguments of the `CPU`
/// golden [`build_rest_helix`](prism_render_architecture::hair::rest_helix::build_rest_helix).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRestHelixStrand {
    /// World-space root (coil vertex `0` is anchored here on the coil axis).
    pub root: [f32; 3],
    /// Growth direction; the coil axis is this normalized (zero/non-finite
    /// falls back to `+Y` on both sides).
    pub root_tangent: [f32; 3],
    /// Coil radius (distance of the fibre from the coil axis); `<= 0`
    /// degenerates to straight hair.
    pub radius: f32,
    /// Advance along the coil axis per segment (the `helix` pitch per step).
    pub pitch_per_segment: f32,
    /// Real part of the per-segment unit-complex rotation step.
    pub rot_cos_step: f32,
    /// Imaginary part of the per-segment unit-complex rotation step.
    pub rot_sin_step: f32,
    /// Ratio of tangential bending stiffness to normal bending stiffness
    /// (clamped into the golden's `RATIO_MIN..=RATIO_MAX`).
    pub bend_stiffness_ratio: f32,
}

/// One strand's helical rest state, mirroring the `CPU` golden
/// [`RestHelix`](prism_render_architecture::hair::rest_helix::RestHelix).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuRestHelixOut {
    /// Rest control-point positions along the `helix` (`segments + 1` of them).
    pub positions: Vec<[f32; 3]>,
    /// Rest length of each coil segment (`segments` of them).
    pub rest_lengths: Vec<f32>,
    /// Per-interior-joint discrete curvature binormal (`segments - 1` of them,
    /// empty for a single segment).
    pub rest_darboux: Vec<[f32; 3]>,
    /// Per-segment bending compliance about the coil tangent (one value, equal
    /// for every segment: `BASE_BEND_COMPLIANCE / ratio`).
    pub tangential_compliance: f32,
    /// Per-segment bending compliance about the coil normal (one value, equal
    /// for every segment: `BASE_BEND_COMPLIANCE`).
    pub normal_compliance: f32,
}

/// Uniform parameters for one rest-helix dispatch. Layout matches `Params` in
/// `shaders/rest_helix.wesl`: the strand count and the shared coil-segment count,
/// padded to one `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    strand_count: u32,
    segments: u32,
    pad0: u32,
    pad1: u32,
}

/// A compiled, reusable per-strand rest-helix pipeline.
pub struct GpuRestHelix {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRestHelix {
    /// Compiles the per-strand rest-helix kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRestHelix {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_rest_helix"),
            source: ShaderSource::Wgsl(include_str!("../shaders/rest_helix.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_rest_helix_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_rest_helix_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_rest_helix_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRestHelix {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the helical rest state of every strand in `strands`, all sharing
    /// the coil-segment count `segments`, returning one [`GpuRestHelixOut`] per
    /// strand.
    ///
    /// `segments` is clamped into `1..=MAX_SEGMENTS` before the dispatch (the
    /// vertex count is then `segments + 1` and the interior-joint count is
    /// `segments - 1`), matching the golden's `sanitized`. The result for strand
    /// `s` equals [`build_rest_helix`](prism_render_architecture::hair::rest_helix::build_rest_helix)
    /// applied to that strand's parameters at the clamped segment count, to
    /// within the fused-multiply-add tolerance documented on this module. An
    /// empty strand batch yields an empty vector without a dispatch — storage
    /// buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        strands: &[GpuRestHelixStrand],
        segments: usize,
    ) -> Vec<GpuRestHelixOut> {
        let strand_count = strands.len();
        if strand_count == 0 {
            return Vec::new();
        }
        let segments = segments.clamp(1, MAX_SEGMENTS);
        let vertex_count = segments + 1;
        let joint_count = segments - 1;

        let device = ctx.device();

        let uniforms = Params {
            strand_count: strand_count as u32,
            segments: segments as u32,
            pad0: 0,
            pad1: 0,
        };

        // Pack the strand roots strand-major at stride `STRAND_STRIDE`.
        let mut packed: Vec<f32> = Vec::with_capacity(strand_count * STRAND_STRIDE);
        for s in strands {
            packed.extend_from_slice(&s.root);
            packed.extend_from_slice(&s.root_tangent);
            packed.push(s.radius);
            packed.push(s.pitch_per_segment);
            packed.push(s.rot_cos_step);
            packed.push(s.rot_sin_step);
            packed.push(s.bend_stiffness_ratio);
        }

        let f32_bytes = size_of::<f32>() as u64;
        let count = strand_count as u64;
        let positions_len = count * (vertex_count as u64) * 3;
        let lengths_len = count * (segments as u64);
        let darboux_len = count * (segments as u64) * 3;
        let compliance_len = count * 2;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_rest_helix_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let strands_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_rest_helix_strands"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let positions_buf = storage_out(device, "positions", positions_len * f32_bytes);
        let lengths_buf = storage_out(device, "lengths", lengths_len * f32_bytes);
        let darboux_buf = storage_out(device, "darboux", darboux_len * f32_bytes);
        let compliance_buf = storage_out(device, "compliance", compliance_len * f32_bytes);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_rest_helix_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: strands_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: positions_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: lengths_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: darboux_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: compliance_buf.as_entire_binding(),
                },
            ],
        });

        let positions_stage = stage_buffer(device, "positions", positions_len * f32_bytes);
        let lengths_stage = stage_buffer(device, "lengths", lengths_len * f32_bytes);
        let darboux_stage = stage_buffer(device, "darboux", darboux_len * f32_bytes);
        let compliance_stage = stage_buffer(device, "compliance", compliance_len * f32_bytes);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_rest_helix_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_rest_helix_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (strand_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &positions_buf,
            0,
            &positions_stage,
            0,
            positions_len * f32_bytes,
        );
        encoder.copy_buffer_to_buffer(&lengths_buf, 0, &lengths_stage, 0, lengths_len * f32_bytes);
        encoder.copy_buffer_to_buffer(&darboux_buf, 0, &darboux_stage, 0, darboux_len * f32_bytes);
        encoder.copy_buffer_to_buffer(
            &compliance_buf,
            0,
            &compliance_stage,
            0,
            compliance_len * f32_bytes,
        );
        ctx.queue().submit([encoder.finish()]);

        positions_stage.slice(..).map_async(MapMode::Read, |_| {});
        lengths_stage.slice(..).map_async(MapMode::Read, |_| {});
        darboux_stage.slice(..).map_async(MapMode::Read, |_| {});
        compliance_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let positions_flat = read_mapped(&positions_stage);
        let lengths_flat = read_mapped(&lengths_stage);
        let darboux_flat = read_mapped(&darboux_stage);
        let compliance_flat = read_mapped(&compliance_stage);

        let mut out = Vec::with_capacity(strand_count);
        for s in 0..strand_count {
            let pos_base = s * vertex_count * 3;
            let mut positions = Vec::with_capacity(vertex_count);
            for v in 0..vertex_count {
                let b = pos_base + v * 3;
                positions.push([
                    positions_flat[b],
                    positions_flat[b + 1],
                    positions_flat[b + 2],
                ]);
            }

            let len_base = s * segments;
            let rest_lengths = lengths_flat[len_base..len_base + segments].to_vec();

            // The kernel strides `darboux` by `segments * 3` but writes only the
            // first `segments - 1` joints; take exactly those.
            let dar_base = s * segments * 3;
            let mut rest_darboux = Vec::with_capacity(joint_count);
            for j in 0..joint_count {
                let b = dar_base + j * 3;
                rest_darboux.push([darboux_flat[b], darboux_flat[b + 1], darboux_flat[b + 2]]);
            }

            let comp_base = s * 2;
            out.push(GpuRestHelixOut {
                positions,
                rest_lengths,
                rest_darboux,
                tangential_compliance: compliance_flat[comp_base],
                normal_compliance: compliance_flat[comp_base + 1],
            });
        }
        out
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

/// Allocates a device-local storage output buffer that can be copied back.
fn storage_out(device: &wgpu::Device, name: &str, size: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(&format!("prism_hair_rest_helix_{name}_out")),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Allocates a mappable staging buffer for readback.
fn stage_buffer(device: &wgpu::Device, name: &str, size: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(&format!("prism_hair_rest_helix_{name}_stage")),
        size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Reads a mapped staging buffer back into an owned `f32` vector and unmaps it.
fn read_mapped(stage: &Buffer) -> Vec<f32> {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    stage.unmap();
    flat
}
