//! `wgpu` compute twin of Prism's rotation-minimizing strand-frame transport
//! ([`build_strand_frames`](prism_render_architecture::hair::frames::build_strand_frames)).
//!
//! Ribbon/card expansion and anisotropic hair shading (Kajiya-Kay / Marschner /
//! Chiang) need a coherent orthonormal frame at every control point, not just a
//! tangent. Production hair engines (`UE5` Groom, AMD `TressFX`) carry one
//! reference normal along the strand with minimal twist rather than rebuilding
//! an arbitrary basis per point (which flips and shimmers). The `CPU` golden for
//! that transport is
//! [`build_strand_frames`](prism_render_architecture::hair::frames::build_strand_frames),
//! implementing the double-reflection rotation-minimizing frame (Wang et al.
//! 2008); this crate is the on-device twin that walks the same transport, one
//! thread per strand, so a passing real-device parity test is direct evidence
//! the ported kernel computes the same per-vertex frames as the reference — not
//! merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuStrandFrames::eval`] takes a batch of strands (each a slice of control
//! points) and returns one [`GpuStrandFrame`] per control point per strand, in
//! input order. Each strand is processed by a single thread that seeds the root
//! frame from an arbitrary orthonormal reference, then propagates the reference
//! normal point-to-point with two reflections, re-orthonormalizing against the
//! next tangent for numerical hygiene. Strands are independent, so the batch is
//! embarrassingly parallel.
//!
//! Every guard is reproduced: an empty strand yields no frames, a single point
//! yields one frame around the fallback tangent, and coincident points reuse the
//! previous reference direction rather than dividing by zero.
//!
//! # Portability
//!
//! The kernel uses only `sqrt`, `dot`, `cross` and multiply/add in the portable
//! core-`WGSL` subset — no `exp`, `pow` or optional device feature — so it runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The transport contains no transcendental call (the reference restricts itself
//! to `sqrt`), so `CPU` and `GPU` evaluate the same closed-form geometry. They
//! are **not** bit-exact: a `GPU` may fuse a multiply-add or reassociate a dot
//! product the scalar reference leaves separate, perturbing the low mantissa
//! bits by a few `ULP`. The parity test therefore asserts a tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`) per component and keeps inputs away
//! from the degenerate `len_sq <= f32::EPSILON` thresholds so `CPU` and `GPU`
//! never diverge on a branch.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard double-reflection rotation-minimizing frame plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One control point's orthonormal frame, mirroring the `CPU`
/// [`StrandFrame`](prism_render_architecture::hair::frames::StrandFrame).
///
/// The three axes are mutually perpendicular unit vectors with
/// `bitangent == tangent × normal`, forming a right-handed basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuStrandFrame {
    /// Unit direction along the strand at this control point (root → tip).
    pub tangent: [f32; 3],
    /// Rotation-minimizing reference normal, perpendicular to `tangent`.
    pub normal: [f32; 3],
    /// `tangent × normal`; the direction a ribbon widens along.
    pub bitangent: [f32; 3],
}

/// Uniform parameters for one frame-transport dispatch. Layout matches `Params`
/// in `shaders/frames.wesl`: the strand count then three pad words for the
/// `16`-byte uniform alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    strand_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One strand descriptor uploaded to the kernel. `16`-byte `repr(C)` matching
/// `Strand` in `shaders/frames.wesl`: the base index of this strand's first
/// control point (and first output frame) in the flat point/frame arrays, then
/// its point count.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameStrand {
    point_offset: u32,
    point_count: u32,
    pad0: u32,
    pad1: u32,
}

/// A compiled, reusable strand-frame transport pipeline.
pub struct GpuStrandFrames {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuStrandFrames {
    /// Compiles the frame-transport kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuStrandFrames {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_frames"),
            source: ShaderSource::Wgsl(include_str!("../shaders/frames.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_frames_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_frames_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_frames_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuStrandFrames {
            module,
            layout,
            pipeline,
        }
    }

    /// Builds rotation-minimizing frames for every strand, returning one
    /// `Vec<GpuStrandFrame>` per input strand (same length as `strands`), each
    /// with one frame per control point in root → tip order.
    ///
    /// The frames for strand `s` equal
    /// [`build_strand_frames`](prism_render_architecture::hair::frames::build_strand_frames)
    /// applied to `strands[s]`, to within the fused-multiply-add tolerance
    /// documented on this module. An empty strand yields an empty inner `Vec`.
    /// A wholly empty batch (no strands, or every strand empty) is handled
    /// without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, strands: &[&[[f32; 3]]]) -> Vec<Vec<GpuStrandFrame>> {
        if strands.is_empty() {
            return Vec::new();
        }

        // Flatten points and build per-strand descriptors sharing one buffer.
        let mut flat_points: Vec<f32> = Vec::new();
        let mut descriptors: Vec<FrameStrand> = Vec::with_capacity(strands.len());
        for strand in strands {
            let offset = (flat_points.len() / 3) as u32;
            descriptors.push(FrameStrand {
                point_offset: offset,
                point_count: strand.len() as u32,
                pad0: 0,
                pad1: 0,
            });
            for p in *strand {
                flat_points.extend_from_slice(p);
            }
        }

        let total_points = flat_points.len() / 3;
        if total_points == 0 {
            // Every strand is empty: no frames to compute, no dispatch possible.
            return strands.iter().map(|_| Vec::new()).collect();
        }

        let device = ctx.device();
        let params = Params {
            strand_count: strands.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Nine f32 (tangent, normal, bitangent) per control point.
        let out_bytes = (total_points as u64) * 9 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_frames_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let strands_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_frames_strands"),
            contents: bytemuck::cast_slice(&descriptors),
            usage: BufferUsages::STORAGE,
        });
        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_frames_points"),
            contents: bytemuck::cast_slice(&flat_points),
            usage: BufferUsages::STORAGE,
        });
        let frames_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_frames_values"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let frames_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_frames_values_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_frames_bind_group"),
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
                    resource: points_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: frames_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_frames_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_frames_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (strands.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&frames_buf, 0, &frames_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        frames_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = frames_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        frames_stage.unmap();
        debug_assert_eq!(flat.len(), total_points * 9);

        // Re-split the flat frame stream back into per-strand vectors using the
        // same offsets the descriptors carry.
        strands
            .iter()
            .zip(descriptors.iter())
            .map(|(strand, desc)| {
                let mut out = Vec::with_capacity(strand.len());
                for local in 0..(desc.point_count as usize) {
                    let base = (desc.point_offset as usize + local) * 9;
                    out.push(GpuStrandFrame {
                        tangent: [flat[base], flat[base + 1], flat[base + 2]],
                        normal: [flat[base + 3], flat[base + 4], flat[base + 5]],
                        bitangent: [flat[base + 6], flat[base + 7], flat[base + 8]],
                    });
                }
                out
            })
            .collect()
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
