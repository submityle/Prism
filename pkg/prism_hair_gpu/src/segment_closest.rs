//! `wgpu` compute twin of Prism's segment-segment closest-point query
//! ([`segment_segment_closest`](prism_render_architecture::hair::barrier_contact::segment_segment_closest)).
//!
//! For each pair of segments `[p1, q1]` and `[p2, q2]` the kernel returns the
//! closest point on each segment plus the distance between them, following the
//! clamped parametric solve from Ericson, *Real-Time Collision Detection*. This
//! closed-form closest-pair query is the geometric kernel the strand
//! self-collision / barrier contact pass evaluates for every candidate
//! segment-segment pair, so the on-device twin must match the scalar reference
//! branch for branch.
//!
//! # Why one thread per pair
//!
//! Each pair's result depends only on its own four endpoints, so this is
//! embarrassingly parallel: one thread owns one pair, reads its four points and
//! writes its two closest points plus the scalar distance. The host lays the
//! pairs out flat (four `vec4` lanes per pair) so threads never alias.
//!
//! # What the kernel evaluates
//!
//! [`GpuSegmentClosest::eval`] takes a batch of [`SegmentPair`]s and returns a
//! `(closest_on_first, closest_on_second, distance)` triple per pair, in input
//! order. Points travel through the storage buffers as `[x, y, z, 0]`; the
//! distance rides in the first output point's `w` lane.
//!
//! # Correctness model
//!
//! The reference sanitizes its inputs (replacing non-finite components with 0);
//! the host supplies finite endpoints, so the kernel omits that guard but keeps
//! the `EPS_LEN_SQ` degeneracy thresholds and the `EPS_DENOM` parallel guard
//! verbatim. The solve is a single closed-form evaluation (a handful of
//! dot/clamp/divide/multiply-add plus one `sqrt` for the distance), with no
//! chained recurrence, so the only `CPU` vs `GPU` divergence is legal
//! fused-multiply-add contraction; parity is asserted per component to within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3`. Test pairs stay clear of the branch
//! boundaries so the branch taken is identical on both sides.
//!
//! # Portability
//!
//! The kernel uses only `dot`, `clamp`, `length`, divide and multiply-add in
//! the portable core-`WGSL` subset — no `exp`, `pow` or optional device
//! feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard Ericson segment-segment closest-point construction plus
//! a `wgpu` compute dispatch; no Unreal Engine source or derived code.

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

use prism_render_architecture::hair::barrier_contact::{segment_segment_closest, Vec3};

use crate::context::GpuContext;

/// Uniform parameters for one closest-pair dispatch. Layout matches `Params`
/// in `shaders/segment_closest.wesl`: the pair count padded to one `16`-byte
/// uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    pair_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One segment-segment query: the first segment `[p1, q1]` and the second
/// segment `[p2, q2]`.
#[derive(Clone, Copy)]
pub struct SegmentPair {
    /// First endpoint of segment 1.
    pub p1: Vec3,
    /// Second endpoint of segment 1.
    pub q1: Vec3,
    /// First endpoint of segment 2.
    pub p2: Vec3,
    /// Second endpoint of segment 2.
    pub q2: Vec3,
}

/// Compiled per-pair closest-point compute twin: the shader module (kept alive
/// so its pipeline stays valid), the bind-group layout and the pipeline.
pub struct GpuSegmentClosest {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSegmentClosest {
    /// Compiles the per-pair closest-point kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSegmentClosest {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_segment_closest"),
            source: ShaderSource::Wgsl(include_str!("../shaders/segment_closest.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_segment_closest_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_segment_closest_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_segment_closest_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSegmentClosest {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the closest-point query for every pair, returning one
    /// `(closest_on_first, closest_on_second, distance)` triple per input pair,
    /// in input order.
    ///
    /// The triple for pair `i` equals
    /// [`segment_segment_closest`](prism_render_architecture::hair::barrier_contact::segment_segment_closest)
    /// applied to `pairs[i]`, to within the single-evaluation tolerance
    /// documented on this module (`abs_diff < 1e-4` or `rel_diff < 1e-3`). The
    /// empty batch is handled without a dispatch — storage buffers cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, pairs: &[SegmentPair]) -> Vec<(Vec3, Vec3, f32)> {
        if pairs.is_empty() {
            return Vec::new();
        }

        // Flatten four endpoints per pair into one `vec4` stream.
        let mut flat: Vec<[f32; 4]> = Vec::with_capacity(pairs.len() * 4);
        for pair in pairs {
            flat.push([pair.p1.x, pair.p1.y, pair.p1.z, 0.0]);
            flat.push([pair.q1.x, pair.q1.y, pair.q1.z, 0.0]);
            flat.push([pair.p2.x, pair.p2.y, pair.p2.z, 0.0]);
            flat.push([pair.q2.x, pair.q2.y, pair.q2.z, 0.0]);
        }

        let device = ctx.device();
        let params = Params {
            pair_count: pairs.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let out_bytes = (pairs.len() as u64) * 4 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_segment_closest_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let segs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_segment_closest_segs"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE,
        });
        let out_c1_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_segment_closest_out_c1"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_c2_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_segment_closest_out_c2"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let c1_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_segment_closest_c1_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let c2_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_segment_closest_c2_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_segment_closest_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: segs_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_c1_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_c2_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_segment_closest_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_segment_closest_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (pairs.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_c1_buf, 0, &c1_stage, 0, out_bytes);
        encoder.copy_buffer_to_buffer(&out_c2_buf, 0, &c2_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        c1_stage.slice(..).map_async(MapMode::Read, |_| {});
        c2_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let c1_view = c1_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let c1_flat = bytemuck::cast_slice::<u8, f32>(&c1_view).to_vec();
        drop(c1_view);
        c1_stage.unmap();

        let c2_view = c2_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let c2_flat = bytemuck::cast_slice::<u8, f32>(&c2_view).to_vec();
        drop(c2_view);
        c2_stage.unmap();

        debug_assert_eq!(c1_flat.len(), pairs.len() * 4);
        debug_assert_eq!(c2_flat.len(), pairs.len() * 4);

        let mut out: Vec<(Vec3, Vec3, f32)> = Vec::with_capacity(pairs.len());
        for i in 0..pairs.len() {
            let b = i * 4;
            let c1 = Vec3::new(c1_flat[b], c1_flat[b + 1], c1_flat[b + 2]);
            let dist = c1_flat[b + 3];
            let c2 = Vec3::new(c2_flat[b], c2_flat[b + 1], c2_flat[b + 2]);
            out.push((c1, c2, dist));
        }
        out
    }
}

/// The `CPU` golden closest-pair triple, re-exported so the parity test can
/// assert the device twin against the identical reference it mirrors.
///
/// Runs [`segment_segment_closest`](prism_render_architecture::hair::barrier_contact::segment_segment_closest)
/// on the pair `[p1, q1]`, `[p2, q2]`.
#[must_use]
pub fn reference_segment_closest(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3, f32) {
    segment_segment_closest(p1, q1, p2, q2)
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
