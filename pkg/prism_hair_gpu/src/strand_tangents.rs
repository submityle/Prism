//! `wgpu` compute twin of Prism's per-control-point strand tangents
//! ([`strand_tangents`](prism_render_architecture::hair::frames::strand_tangents)).
//!
//! Ribbon/card expansion, anisotropic hair shading (Kajiya-Kay / Marschner /
//! Chiang) and the rotation-minimizing frame transport all seed from a unit
//! tangent at every control point. The `CPU` golden
//! [`strand_tangents`](prism_render_architecture::hair::frames::strand_tangents)
//! walks a strand polyline with a forward difference at interior/root points and
//! a backward difference at the tip, normalizing each raw direction and falling
//! back to a fixed unit direction for a single-point strand or a numerically
//! zero-length segment. This crate is the on-device twin that evaluates the same
//! per-point rule, one thread per control point over a flattened batch of
//! strands, so a passing real-device parity test is direct evidence the ported
//! kernel computes the same tangents as the reference — not merely that its
//! shader compiles.
//!
//! # Relationship to the sibling frame twin
//!
//! This is deliberately a different dispatch from
//! [`GpuStrandFrames`](crate::frames::GpuStrandFrames), which runs one thread
//! per strand and transports a full orthonormal frame (tangent, normal,
//! bitangent) along each strand with the double-reflection method. Here each
//! thread owns one control point and emits only its tangent, so the batch is
//! embarrassingly parallel over the flattened point stream with no per-strand
//! serial transport; the frame twin never exposes the bare tangent `Vec`.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairStrandTangents::eval`] takes a batch of strands (each a slice of
//! control points) and returns one `Vec<[f32; 3]>` of unit tangents per strand,
//! in input order. The point index is the invocation id (`@compute
//! @workgroup_size(64)`, a one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the flattened point count
//! early-return. Each point reads a per-point `meta` of (strand offset, strand
//! count) to recover its local index and branch exactly as the golden does.
//!
//! # Correctness model
//!
//! The forward/backward difference is an exact subtraction, so the only `CPU`
//! vs `GPU` divergence is legal fused-multiply-add contraction in the
//! normalize (`sqrt` + reciprocal). Each tangent component is matched against a
//! tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`), not bit-for-bit. The
//! single-point / empty-strand fallback, the forward-vs-backward branch at the
//! tip, and `Vec3::normalize_or`'s zero-length guard (which compares the squared
//! length against `f32::EPSILON`, not the solver's `EPS_LEN_SQ`) are all
//! reproduced exactly, so a swapped branch or a missing guard still fails the
//! parity test.
//!
//! # Portability
//!
//! The kernel uses only compares, subtract, `dot` and a `sqrt` + reciprocal
//! multiply in the portable core-`WGSL` subset — no `exp`, `pow` or optional
//! device feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard finite-difference polyline tangents plus a `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::hair::frames::strand_tangents;
use prism_render_architecture::hair::interpolation::Vec3;

use crate::context::GpuContext;

/// Uniform parameters for one tangent dispatch. Layout matches `Params` in
/// `shaders/strand_tangents.wesl`: the flattened control-point count, padded to
/// one `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    point_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// Per control point: the global point index of its strand's first point
/// (`offset`) and the strand's control-point count (`count`). `8`-byte stride
/// matching the shader's `array<vec2<u32>>`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PointMeta {
    offset: u32,
    count: u32,
}

/// A compiled, reusable per-control-point strand-tangent pipeline.
pub struct GpuHairStrandTangents {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairStrandTangents {
    /// Compiles the per-control-point tangent kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairStrandTangents {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_strand_tangents"),
            source: ShaderSource::Wgsl(include_str!("../shaders/strand_tangents.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_strand_tangents_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_strand_tangents_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_strand_tangents_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairStrandTangents {
            module,
            layout,
            pipeline,
        }
    }

    /// Computes forward/backward-difference unit tangents for every strand,
    /// returning one `Vec<[f32; 3]>` per input strand (same length as
    /// `strands`), each with one tangent per control point in root → tip order.
    ///
    /// The tangents for strand `s` equal
    /// [`strand_tangents`](prism_render_architecture::hair::frames::strand_tangents)
    /// applied to `strands[s]`, to within the fused-multiply-add tolerance
    /// documented on this module (`abs_diff < 1e-4` or `rel_diff < 1e-3`): a
    /// single-point strand yields the fallback direction, interior/root points
    /// take the forward difference, the tip takes the backward difference, and a
    /// zero-length segment falls back. An empty strand yields an empty inner
    /// `Vec`. A wholly empty batch (no strands, or every strand empty) is
    /// handled without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, strands: &[&[[f32; 3]]]) -> Vec<Vec<[f32; 3]>> {
        if strands.is_empty() {
            return Vec::new();
        }

        // Flatten points and build per-point (offset, count) metadata sharing
        // one buffer. The offset is the global index of each strand's first
        // point, so a point recovers its local index as `global - offset`.
        let mut flat_points: Vec<f32> = Vec::new();
        let mut meta: Vec<PointMeta> = Vec::new();
        for strand in strands {
            let offset = (flat_points.len() / 3) as u32;
            let count = strand.len() as u32;
            for p in *strand {
                flat_points.extend_from_slice(p);
                meta.push(PointMeta { offset, count });
            }
        }

        let total_points = meta.len();
        if total_points == 0 {
            // Every strand is empty: no tangents to compute, no dispatch.
            return strands.iter().map(|_| Vec::new()).collect();
        }

        let device = ctx.device();
        let params = Params {
            point_count: total_points as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Three f32 (tangent) per control point.
        let out_bytes = (total_points as u64) * 3 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_strand_tangents_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_strand_tangents_points"),
            contents: bytemuck::cast_slice(&flat_points),
            usage: BufferUsages::STORAGE,
        });
        let meta_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_strand_tangents_meta"),
            contents: bytemuck::cast_slice(&meta),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_strand_tangents_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_strand_tangents_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_strand_tangents_bind_group"),
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
                    resource: meta_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_strand_tangents_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_strand_tangents_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (total_points as u32).div_ceil(64);
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
        debug_assert_eq!(flat.len(), total_points * 3);

        // Re-split the flat tangent stream back into per-strand vectors using
        // the same offsets the metadata carries.
        let mut cursor = 0usize;
        strands
            .iter()
            .map(|strand| {
                let mut out = Vec::with_capacity(strand.len());
                for _ in 0..strand.len() {
                    let base = cursor * 3;
                    out.push([flat[base], flat[base + 1], flat[base + 2]]);
                    cursor += 1;
                }
                out
            })
            .collect()
    }
}

/// The `CPU` golden per-control-point strand tangents, re-exported so the parity
/// test can assert the device twin against the identical reference it mirrors.
///
/// Runs [`strand_tangents`](prism_render_architecture::hair::frames::strand_tangents)
/// on `points` and returns the unit tangents as flat `[x, y, z]` triples in
/// order.
#[must_use]
pub fn reference_strand_tangents(points: &[Vec3]) -> Vec<[f32; 3]> {
    strand_tangents(points)
        .iter()
        .map(|t| [t.x, t.y, t.z])
        .collect()
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
