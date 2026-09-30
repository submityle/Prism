//! `wgpu` compute twin of Prism's guide-to-render strand interpolation
//! ([`interpolate_render_strand`](prism_render_architecture::hair::interpolation::interpolate_render_strand)).
//!
//! A groom simulates only its sparse set of *guide* strands; the many *render*
//! strands that actually draw are interpolated from them every frame. The
//! reference expands one render strand at a time from its binding (up to four
//! nearest guides with barycentric-style weights, a per-strand seed) through a
//! fixed transform chain — weighted blend, length jitter, clump pull toward the
//! representative guide, a seed-stable curl helix framed on the clumped
//! tangent, and per-point position jitter. This crate is the on-device twin:
//! one thread per render strand walks the identical chain in the identical
//! order, so a passing real-device parity test is direct evidence the ported
//! kernel expands the groom to the same control points as the reference — not
//! merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairInterp::eval`] takes the guide polylines, the render-strand
//! bindings and the shared [`InterpolationParams`], and returns one control
//! point list per binding. Each strand's output length is the shortest
//! contributing guide's control-point count (so mismatched guide resolutions
//! never index out of bounds); a binding with no in-range, positive-weight,
//! non-empty guide yields an empty list, exactly as the reference leaves `out`
//! empty.
//!
//! Determinism is the whole point: the only entropy source is a hand-written
//! `splitmix64` finalizer, reproduced on-device bit-for-bit with a `vec2<u32>`
//! emulation because baseline `WGSL` has no native 64-bit integer. `f32::sin`
//! is banned for the same reproducibility reason, so curl uses the same
//! range-reduced Taylor `sin_turns` the reference uses.
//!
//! # Portability
//!
//! The kernel uses only `sqrt` (via `length`/`normalize`), `min`, `max`,
//! `floor`, `sign`, `abs`, `dot`, `cross` and multiply/add in the portable
//! core-`WGSL` subset — no `exp`, `pow` or optional device feature — so the
//! twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The blend, clump and jitter are closed-form, and the hash and `sin_turns`
//! are integer/polynomial, so `CPU` and `GPU` evaluate the same arithmetic.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few `ULP`. The parity
//! test therefore asserts a per-component tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`) rather than exact equality.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard guide-to-render interpolation (blend + clump + curl +
//! jitter) plus `wgpu` compute dispatch; no Unreal Engine source or derived
//! code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::interpolation::{
    InterpolationParams, RenderStrandBinding, Vec3, GUIDE_INFLUENCE_COUNT,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform interpolation parameters uploaded to the kernel. `32`-byte
/// scalar-packed `repr(C)` matching `HairInterpParams` in `shaders/interp.wesl`
/// (padded to a multiple of 16 for the uniform block).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    clump_count: u32,
    clump_strength: f32,
    curl_frequency: f32,
    curl_amplitude: f32,
    position_jitter: f32,
    length_jitter: f32,
    strand_count: u32,
    pad: u32,
}

/// One guide slice descriptor uploaded to the kernel. `8`-byte `repr(C)`
/// matching `HairGuideRange` in `shaders/interp.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GuideRange {
    offset: u32,
    count: u32,
}

/// One render-strand binding uploaded to the kernel. `48`-byte `repr(C)`
/// matching `HairRenderBinding` in `shaders/interp.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBinding {
    guides: [u32; 4],
    weights: [f32; 4],
    root_uv: [f32; 2],
    seed: u32,
    out_offset: u32,
}

/// Upper bound on control points per strand mirrored from the kernel's
/// `MAX_HAIR_STRAND_POINTS`. Callers keep guide resolutions within this so the
/// device scratch slab never overflows and the twin stays a faithful mirror of
/// the (uncapped) reference.
const MAX_HAIR_STRAND_POINTS: usize = 256;

/// A compiled, reusable interpolation pipeline.
pub struct GpuHairInterp {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairInterp {
    /// Compiles the interpolation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairInterp {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_interp"),
            source: ShaderSource::Wgsl(include_str!("../shaders/interp.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_interp_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_interp_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_interp_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairInterp {
            module,
            layout,
            pipeline,
        }
    }

    /// Interpolates every render strand from its guides, returning one control
    /// point list per binding (same length and order as `bindings`).
    ///
    /// The result equals
    /// [`interpolate_render_strand`](prism_render_architecture::hair::interpolation::interpolate_render_strand)
    /// applied per binding, to within the fused-multiply-add tolerance
    /// documented on this module. A binding with no in-range, positive-weight,
    /// non-empty guide yields an empty list. When no strand produces any output
    /// (empty bindings, or every binding degenerate) the call returns the
    /// per-binding empty lists without a dispatch — storage buffers cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        guides: &[&[Vec3]],
        bindings: &[RenderStrandBinding],
        params: InterpolationParams,
    ) -> Vec<Vec<Vec3>> {
        // Host-side sanitize once, so the uploaded uniform matches what the
        // reference derives internally via `InterpolationParams::sanitized`.
        let params = params.sanitized();

        // Precompute each strand's output length and flat offset by replicating
        // the reference's contribution gather (the device kernel derives the
        // same `len`, so the layout must agree). A degenerate strand takes zero
        // slots and reads back as an empty list.
        let mut out_offsets: Vec<u32> = Vec::with_capacity(bindings.len());
        let mut out_lens: Vec<usize> = Vec::with_capacity(bindings.len());
        let mut total_points = 0usize;
        for binding in bindings {
            let len = strand_output_len(guides, binding);
            out_offsets.push(total_points as u32);
            out_lens.push(len);
            total_points += len;
        }

        if total_points == 0 {
            return out_lens.iter().map(|_| Vec::new()).collect();
        }

        // Flat guide-point pool (`xyz` used, `w` padding) plus per-guide ranges.
        let mut guide_points: Vec<[f32; 4]> = Vec::new();
        let mut guide_ranges: Vec<GuideRange> = Vec::with_capacity(guides.len());
        for points in guides {
            let offset = guide_points.len() as u32;
            for p in *points {
                guide_points.push([p.x, p.y, p.z, 0.0]);
            }
            guide_ranges.push(GuideRange {
                offset,
                count: points.len() as u32,
            });
        }
        if guide_points.is_empty() {
            guide_points.push([0.0; 4]);
        }
        if guide_ranges.is_empty() {
            guide_ranges.push(GuideRange {
                offset: 0,
                count: 0,
            });
        }

        // Per render strand binding, with the flat output offset appended.
        let gpu_bindings: Vec<GpuBinding> = bindings
            .iter()
            .zip(out_offsets.iter())
            .map(|(b, &offset)| GpuBinding {
                guides: b.guides,
                weights: b.weights,
                root_uv: [b.root_uv.0, b.root_uv.1],
                seed: b.seed,
                out_offset: offset,
            })
            .collect();

        let uniform = Params {
            clump_count: params.clump_count,
            clump_strength: params.clump_strength,
            curl_frequency: params.curl_frequency,
            curl_amplitude: params.curl_amplitude,
            position_jitter: params.position_jitter,
            length_jitter: params.length_jitter,
            strand_count: bindings.len() as u32,
            pad: 0,
        };

        let device = ctx.device();
        let out_bytes = (total_points * 4 * size_of::<f32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_interp_params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM,
        });
        let guide_points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_interp_guide_points"),
            contents: bytemuck::cast_slice(&guide_points),
            usage: BufferUsages::STORAGE,
        });
        let guide_ranges_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_interp_guide_ranges"),
            contents: bytemuck::cast_slice(&guide_ranges),
            usage: BufferUsages::STORAGE,
        });
        let bindings_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_interp_bindings"),
            contents: bytemuck::cast_slice(&gpu_bindings),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_interp_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_interp_out_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_interp_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: guide_points_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: guide_ranges_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: bindings_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_interp_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_interp_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (bindings.len() as u32).div_ceil(64);
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

        // Slice the flat output back into one list per binding using the
        // offsets/lengths computed up front.
        out_offsets
            .iter()
            .zip(out_lens.iter())
            .map(|(&offset, &len)| {
                let base = offset as usize * 4;
                (0..len)
                    .map(|i| {
                        let b = base + i * 4;
                        Vec3::new(flat[b], flat[b + 1], flat[b + 2])
                    })
                    .collect()
            })
            .collect()
    }
}

/// Replicates the reference's contribution gather to derive one render strand's
/// output length: the shortest contributing guide's control-point count, or `0`
/// when no in-range, positive-weight, non-empty guide contributes. Capped at
/// [`MAX_HAIR_STRAND_POINTS`] to match the device scratch bound.
fn strand_output_len(guides: &[&[Vec3]], binding: &RenderStrandBinding) -> usize {
    let mut contrib_len = 0usize;
    let mut min_len = usize::MAX;
    for k in 0..GUIDE_INFLUENCE_COUNT {
        let raw = binding.weights[k].max(0.0);
        if raw <= 0.0 {
            continue;
        }
        let index = binding.guides[k] as usize;
        let Some(points) = guides.get(index) else {
            continue;
        };
        if points.is_empty() {
            continue;
        }
        contrib_len += 1;
        if points.len() < min_len {
            min_len = points.len();
        }
    }
    if contrib_len == 0 {
        return 0;
    }
    min_len.min(MAX_HAIR_STRAND_POINTS)
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
