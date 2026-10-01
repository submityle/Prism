//! `wgpu` compute twin of Prism's per-edge rest `Darboux` builder
//! ([`rest_darboux_from_frames`](prism_render_architecture::hair::cosserat::rest_darboux_from_frames)).
//!
//! Given a strand's material-frame chain (the unit quaternions produced by
//! [`parallel_transport_frames`](prism_render_architecture::hair::cosserat::parallel_transport_frames)
//! / `build_strand_frames`), the rest `Darboux` vector between adjacent frames
//! `a` and `b` is the imaginary part of the relative rotation
//! `conjugate(a) * b` (a single Hamilton product). This is the rest companion
//! the `Cosserat` bend-twist constraint ([`simulate_strand_cosserat`](prism_render_architecture::hair::cosserat))
//! reads back, so the on-device twin must match the scalar reference term for
//! term.
//!
//! # Why one thread per edge
//!
//! Unlike the serial parallel-transport recurrence that builds the frames
//! ([`GpuParallelTransport`](crate::parallel_transport::GpuParallelTransport)),
//! each rest `Darboux` vector depends only on one adjacent frame pair, so this
//! is embarrassingly parallel: one thread owns one edge, reads `frames[a]` and
//! `frames[a + 1]` and writes one vector. The host lays out a flat per-edge
//! base index so edges never straddle a strand boundary, exactly the per-edge
//! shape of the [`GpuHairStrandTangents`](crate::strand_tangents::GpuHairStrandTangents)
//! dispatch.
//!
//! # What the kernel evaluates
//!
//! [`GpuRestDarboux::eval`] takes a batch of strands (each a slice of material
//! frames) and returns one `Vec<Vec3>` per strand of `frames.len() - 1` rest
//! `Darboux` vectors (empty when a strand has fewer than two frames).
//! Quaternions travel through the storage buffer as `[w, x, y, z]`; outputs as
//! `[x, y, z, 0]`.
//!
//! # Correctness model
//!
//! The reference performs no `normalize` and no `sanitize` — it is a pure
//! quaternion product — so the only `CPU` vs `GPU` divergence is the legal
//! fused multiply-add contraction inside the single sum-of-four-products per
//! component. Parity is asserted per component to within `abs_diff < 1e-4` or
//! `rel_diff < 1e-3`; identity frame pairs produce a bit-exact zero vector.
//!
//! # Portability
//!
//! The kernel uses only multiply/add in the portable core-`WGSL` subset — no
//! `sqrt`, `exp`, `pow` or optional device feature — so the twin runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Cosserat`/`Kirchhoff` rod rest-`Darboux` construction
//! plus a `wgpu` compute dispatch; no Unreal Engine source or derived code.

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

use prism_render_architecture::hair::cosserat::{rest_darboux_from_frames, Quat, Vec3};

use crate::context::GpuContext;

/// Uniform parameters for one rest-`Darboux` dispatch. Layout matches `Params`
/// in `shaders/rest_darboux.wesl`: the total edge count padded to one `16`-byte
/// uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    edge_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One strand's rest-`Darboux` input: its material-frame chain (the unit
/// quaternions produced by parallel transport, in root → tip order).
#[derive(Clone, Copy)]
pub struct DarbouxStrand<'a> {
    /// Material frames in root → tip order.
    pub frames: &'a [Quat],
}

/// Compiled per-edge rest-`Darboux` compute twin: the shader module (kept alive
/// so its pipeline stays valid), the bind-group layout and the pipeline.
pub struct GpuRestDarboux {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRestDarboux {
    /// Compiles the per-edge rest-`Darboux` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRestDarboux {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_rest_darboux"),
            source: ShaderSource::Wgsl(include_str!("../shaders/rest_darboux.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_rest_darboux_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_rest_darboux_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_rest_darboux_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRestDarboux {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the rest `Darboux` vectors for every strand, returning one
    /// `Vec<Vec3>` per input strand (same length as `strands`), each with
    /// `frames.len() - 1` vectors in root → tip order (empty for a strand with
    /// fewer than two frames).
    ///
    /// The vectors for strand `s` equal
    /// [`rest_darboux_from_frames`](prism_render_architecture::hair::cosserat::rest_darboux_from_frames)
    /// applied to `strands[s].frames`, to within the single-product tolerance
    /// documented on this module (`abs_diff < 1e-4` or `rel_diff < 1e-3`). A
    /// wholly edgeless batch (no strands, or every strand under two frames) is
    /// handled without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, strands: &[DarbouxStrand<'_>]) -> Vec<Vec<Vec3>> {
        if strands.is_empty() {
            return Vec::new();
        }

        // Flatten frames and build the per-edge base-index table. An edge `e`
        // owns frame `edge_base[e]` as `a` and `edge_base[e] + 1` as `b`; the
        // base is the absolute index into `flat_frames`, so edges never straddle
        // a strand boundary.
        let mut flat_frames: Vec<[f32; 4]> = Vec::new();
        let mut edge_base: Vec<u32> = Vec::new();
        for strand in strands {
            let frame_offset = flat_frames.len() as u32;
            for q in strand.frames {
                flat_frames.push([q.w, q.x, q.y, q.z]);
            }
            let count = strand.frames.len();
            if count >= 2 {
                for e in 0..(count - 1) {
                    edge_base.push(frame_offset + e as u32);
                }
            }
        }

        let total_edges = edge_base.len();
        if total_edges == 0 {
            // No strand has two or more frames: nothing to evaluate.
            return strands.iter().map(|_| Vec::new()).collect();
        }

        let device = ctx.device();
        let params = Params {
            edge_count: total_edges as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let out_bytes = (total_edges as u64) * 4 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_rest_darboux_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let frames_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_rest_darboux_frames"),
            contents: bytemuck::cast_slice(&flat_frames),
            usage: BufferUsages::STORAGE,
        });
        let edge_base_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_rest_darboux_edge_base"),
            contents: bytemuck::cast_slice(&edge_base),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_rest_darboux_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_rest_darboux_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_rest_darboux_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: frames_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: edge_base_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_rest_darboux_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_rest_darboux_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (total_edges as u32).div_ceil(64);
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
        debug_assert_eq!(flat.len(), total_edges * 4);

        // Re-split the flat vector stream back into per-strand vectors: each
        // strand with at least two frames contributes `frames.len() - 1`
        // vectors, consumed in the same order they were pushed above.
        let mut out: Vec<Vec<Vec3>> = Vec::with_capacity(strands.len());
        let mut edge_cursor = 0usize;
        for strand in strands {
            let count = strand.frames.len();
            let mut darboux = Vec::new();
            if count >= 2 {
                for _ in 0..(count - 1) {
                    let b = edge_cursor * 4;
                    darboux.push(Vec3::new(flat[b], flat[b + 1], flat[b + 2]));
                    edge_cursor += 1;
                }
            }
            out.push(darboux);
        }
        out
    }
}

/// The `CPU` golden rest `Darboux` vectors, re-exported so the parity test can
/// assert the device twin against the identical reference it mirrors.
///
/// Runs [`rest_darboux_from_frames`](prism_render_architecture::hair::cosserat::rest_darboux_from_frames)
/// on `frames`.
#[must_use]
pub fn reference_rest_darboux(frames: &[Quat]) -> Vec<Vec3> {
    rest_darboux_from_frames(frames)
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
