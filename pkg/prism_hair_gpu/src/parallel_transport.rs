//! `wgpu` compute twin of Prism's parallel-transport material-frame builder
//! ([`parallel_transport_frames`](prism_render_architecture::hair::cosserat::parallel_transport_frames)).
//!
//! A torsion-free (Bishop / rotation-minimizing) frame is transported along a
//! strand poly-line so it never twists: the first segment rotates the frame's
//! local `+z` axis (`d3`) onto the first edge, and every later frame is the
//! previous one rotated by the minimal rotation carrying the previous edge
//! direction onto the current one ([`Quat::from_min_rotation`](prism_render_architecture::hair::cosserat::Quat),
//! a pure half-vector cross/dot construction with no trigonometric call). This
//! frame chain seeds the `Cosserat` rest `Darboux` vectors, ribbon/card
//! expansion and the anisotropic hair shading frame, so the on-device twin must
//! match the `CPU` reference term for term.
//!
//! # Why one thread per strand
//!
//! The transport is a strict serial recurrence along a strand (`frame[i]`
//! depends on `frame[i - 1]`), so unlike the per-control-point
//! [`GpuHairStrandTangents`](crate::strand_tangents::GpuHairStrandTangents) this
//! is *not* an embarrassingly-parallel per-edge dispatch. One thread owns a
//! whole strand and walks its edge list in order, exactly mirroring the scalar
//! reference loop; independent strands are the parallel axis, the same shape as
//! the per-strand [`GpuCosserat`](crate::cosserat::GpuCosserat) solver twin.
//!
//! # What the kernel evaluates
//!
//! [`GpuParallelTransport::eval`] takes a batch of strands (each a slice of
//! control points plus an initial frame) and returns one `Vec<Quat>` per strand
//! of `points.len() - 1` transported frames in root → tip order (empty when the
//! strand has fewer than two points). Quaternions travel through the storage
//! buffers as `[w, x, y, z]`; points as `[x, y, z, 0]`.
//!
//! # Correctness model
//!
//! Every reference branch is reproduced: the sub-two-point early return, the
//! degenerate (zero-length) edge that reuses the running frame without
//! advancing `prev_dir`, the first-edge seed from the transported `+z` axis
//! versus the chained later edges, and `from_min_rotation`'s antiparallel and
//! degenerate fallbacks. The host supplies finite inputs, so the kernel omits
//! the reference `is_finite` guards (core-`WGSL` has no `isFinite`) but keeps
//! the `EPS_LEN_SQ` zero-length thresholds verbatim. The transport chains a
//! re-normalize per edge, so `CPU` vs `GPU` divergence (legal fused multiply-add
//! in the quaternion product and the `sqrt` + reciprocal normalize) compounds
//! along the strand; parity is asserted per component to within `abs_diff <
//! 3e-3` or `rel_diff < 1e-2`, tight enough to fail a swapped branch, a missing
//! normalize or a wrong seed, loose enough to admit the compounded contraction.
//!
//! # Portability
//!
//! The kernel uses only `dot`, `cross`, `sqrt`, `min`/`max` and multiply-add in
//! the portable core-`WGSL` subset — no `exp`, `pow` or optional device
//! feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard rotation-minimizing / parallel-transport frame (Bishop
//! frame) construction plus a `wgpu` compute dispatch; no Unreal Engine source
//! or derived code.

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

use prism_render_architecture::hair::cosserat::{parallel_transport_frames, Quat, Vec3};

use crate::context::GpuContext;

/// Uniform parameters for one transport dispatch. Layout matches `Params` in
/// `shaders/parallel_transport.wesl`: the strand count padded to one `16`-byte
/// uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    strand_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// Per strand: the global index of its first point (`point_offset`), its
/// control-point count (`point_count`), and the base index of its frame output
/// region (`frame_offset`). `16`-byte stride matching the shader's
/// `array<vec4<u32>>`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StrandMeta {
    point_offset: u32,
    point_count: u32,
    frame_offset: u32,
    pad: u32,
}

/// One strand's transport input: its control-point poly-line and the initial
/// material frame transported from the root.
#[derive(Clone, Copy)]
pub struct TransportStrand<'a> {
    /// Control points in root → tip order.
    pub points: &'a [Vec3],
    /// The initial frame (normalized on device before transport begins).
    pub initial: Quat,
}

/// A compiled, reusable per-strand parallel-transport pipeline.
pub struct GpuParallelTransport {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuParallelTransport {
    /// Compiles the per-strand transport kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuParallelTransport {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_parallel_transport"),
            source: ShaderSource::Wgsl(include_str!("../shaders/parallel_transport.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_parallel_transport_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_parallel_transport_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_parallel_transport_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuParallelTransport {
            module,
            layout,
            pipeline,
        }
    }

    /// Transports a material frame along every strand, returning one `Vec<Quat>`
    /// per input strand (same length as `strands`), each with
    /// `points.len() - 1` frames in root → tip order (empty for a strand with
    /// fewer than two points).
    ///
    /// The frames for strand `s` equal
    /// [`parallel_transport_frames`](prism_render_architecture::hair::cosserat::parallel_transport_frames)
    /// applied to `strands[s].points` and `strands[s].initial`, to within the
    /// chained-transport tolerance documented on this module (`abs_diff < 3e-3`
    /// or `rel_diff < 1e-2`). A wholly frameless batch (no strands, or every
    /// strand under two points) is handled without a dispatch — storage buffers
    /// cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, strands: &[TransportStrand<'_>]) -> Vec<Vec<Quat>> {
        if strands.is_empty() {
            return Vec::new();
        }

        // Flatten points, build per-strand metadata, and lay out the output
        // frame regions (one per strand, `count - 1` frames for strands with at
        // least two points, nothing otherwise).
        let mut flat_points: Vec<[f32; 4]> = Vec::new();
        let mut meta: Vec<StrandMeta> = Vec::with_capacity(strands.len());
        let mut initials: Vec<[f32; 4]> = Vec::with_capacity(strands.len());
        let mut frame_cursor: u32 = 0;
        for strand in strands {
            let point_offset = flat_points.len() as u32;
            let point_count = strand.points.len() as u32;
            for p in strand.points {
                flat_points.push([p.x, p.y, p.z, 0.0]);
            }
            let frame_offset = frame_cursor;
            if point_count >= 2 {
                frame_cursor += point_count - 1;
            }
            meta.push(StrandMeta {
                point_offset,
                point_count,
                frame_offset,
                pad: 0,
            });
            let q = strand.initial;
            initials.push([q.w, q.x, q.y, q.z]);
        }

        let total_frames = frame_cursor as usize;
        if total_frames == 0 {
            // No strand has two or more points: nothing to transport.
            return strands.iter().map(|_| Vec::new()).collect();
        }

        let device = ctx.device();
        let params = Params {
            strand_count: strands.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let out_bytes = (total_frames as u64) * 4 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_parallel_transport_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_parallel_transport_points"),
            contents: bytemuck::cast_slice(&flat_points),
            usage: BufferUsages::STORAGE,
        });
        let meta_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_parallel_transport_meta"),
            contents: bytemuck::cast_slice(&meta),
            usage: BufferUsages::STORAGE,
        });
        let initials_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_parallel_transport_initials"),
            contents: bytemuck::cast_slice(&initials),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_parallel_transport_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_parallel_transport_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_parallel_transport_bind_group"),
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
                    resource: initials_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_parallel_transport_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_parallel_transport_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (strands.len() as u32).div_ceil(64);
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
        debug_assert_eq!(flat.len(), total_frames * 4);

        // Re-split the flat frame stream back into per-strand vectors using the
        // frame regions laid out above.
        let mut out: Vec<Vec<Quat>> = Vec::with_capacity(strands.len());
        for strand in strands {
            let count = strand.points.len();
            let mut frames = Vec::new();
            if count >= 2 {
                let base = out.iter().map(Vec::len).sum::<usize>();
                for e in 0..(count - 1) {
                    let b = (base + e) * 4;
                    frames.push(Quat::new(flat[b], flat[b + 1], flat[b + 2], flat[b + 3]));
                }
            }
            out.push(frames);
        }
        out
    }
}

/// The `CPU` golden parallel-transport frames, re-exported so the parity test
/// can assert the device twin against the identical reference it mirrors.
///
/// Runs [`parallel_transport_frames`](prism_render_architecture::hair::cosserat::parallel_transport_frames)
/// on `points` with the given `initial` frame.
#[must_use]
pub fn reference_parallel_transport(points: &[Vec3], initial: Quat) -> Vec<Quat> {
    parallel_transport_frames(points, initial)
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
