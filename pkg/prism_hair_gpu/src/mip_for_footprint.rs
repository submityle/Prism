//! `wgpu` compute twin of Prism's card-chain mip classifier
//! ([`mip_for_footprint`](prism_render_architecture::hair::card_bake::mip_for_footprint)),
//! the power-of-two mip level a baked card footprint sits at.
//!
//! When a strand cluster is baked into an atlas card, its footprint rarely
//! lands exactly at the card's full `base_texels` resolution: a distant cluster
//! covers only a few texels and belongs at a coarser mip. Like `UE5` Groom's
//! card atlas, the baker classifies every footprint into the card chain so the
//! upload picks the matching mip, and a whole tile of footprints is classified
//! in one dispatch.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairMipForFootprint::eval`] takes a batch of [`MipQuery`] triples — the
//! footprint's longest side `max_dim`, the full-resolution card side
//! `base_texels`, and the coarsest level `max_mip` — and returns one mip index
//! per query, preserving input order. The query index is the invocation id
//! (`@compute @workgroup_size(64)`, one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the query count early-return.
//!
//! # A data-dependent doubling loop, not `log2`
//!
//! The mip index is an explicit integer doubling loop (`size = max_dim`,
//! doubled while the doubled value still fits within `base_texels` and the mip
//! counter is below `max_mip`) exactly like the golden, **not** `log2`: a card
//! baked at `base_texels` is mip `0`, half that side is mip `1`, a quarter is
//! mip `2`, and so on. The doubling is saturating (`u32::saturating_mul(2)`
//! semantics: a value past half of `u32::MAX` saturates instead of wrapping),
//! so a huge footprint can never alias back under `base`. A zero footprint
//! reports `max_mip` (the coarsest level). The trip count is read per-thread
//! from the query, so the loop diverges between invocations in the same
//! workgroup.
//!
//! # Portability
//!
//! The kernel uses only core-`WGSL` integer arithmetic — no `exp`, `pow`,
//! `log2`, `sqrt` or optional device feature — so the twin runs unmodified on
//! Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The kernel is pure integer arithmetic with no float anywhere, so the device
//! reproduces the scalar reference's result identically — there is no rounding
//! or fma to diverge. The parity test therefore compares the integer mip levels
//! directly ([`assert_eq!`]) rather than within a tolerance. The zero-footprint
//! early return and the `base_texels.max(1)` floor mirror the golden branch for
//! branch, so degenerate inputs produce the same level.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard power-of-two mip-chain classification plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::card_bake::mip_for_footprint;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One mip-classification query: how many times the footprint's longest side
/// `max_dim` can be doubled before reaching the full-resolution card side
/// `base_texels`, clamped to `max_mip`.
///
/// `max_dim = 0` reports `max_mip` (the coarsest level). `base_texels` is
/// floored to `1` before the comparison, matching the golden. `max_mip` is a
/// `u32` here for `std430` friendliness and is at most `255` (a `u8` mip index)
/// in practice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MipQuery {
    /// The footprint's longest side, in texels.
    pub max_dim: u32,
    /// The full-resolution card side the chain is baked at, in texels.
    pub base_texels: u32,
    /// The coarsest mip index the result is clamped to.
    pub max_mip: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/mip_for_footprint.wesl`: the query count in a single `16`-byte
/// uniform slot (one `u32` plus padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    element_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One query uploaded to the kernel. `12`-byte `repr(C)` matching `MipQuery` in
/// `shaders/mip_for_footprint.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuMipQuery {
    max_dim: u32,
    base_texels: u32,
    max_mip: u32,
}

/// A compiled, reusable per-query card-chain mip classifier pipeline.
pub struct GpuHairMipForFootprint {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairMipForFootprint {
    /// Compiles the mip-classifier compute pipeline on `ctx`'s device.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairMipForFootprint {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_mip_for_footprint"),
            source: ShaderSource::Wgsl(include_str!("../shaders/mip_for_footprint.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_mip_for_footprint_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_mip_for_footprint_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_mip_for_footprint_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairMipForFootprint {
            module,
            layout,
            pipeline,
        }
    }

    /// Classifies each query into its card-chain mip level, returning one level
    /// per query in input order.
    ///
    /// The value for query `i` equals the `CPU` golden
    /// [`mip_for_footprint`](prism_render_architecture::hair::card_bake::mip_for_footprint)
    /// of `(max_dim, base_texels, max_mip)` exactly — the kernel is pure integer
    /// arithmetic with no rounding to diverge. An empty batch yields an empty
    /// vector without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[MipQuery]) -> Vec<u32> {
        let element_count = queries.len();
        if element_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            element_count: element_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let gpu_queries: Vec<GpuMipQuery> = queries
            .iter()
            .map(|q| GpuMipQuery {
                max_dim: q.max_dim,
                base_texels: q.base_texels,
                max_mip: q.max_mip,
            })
            .collect();

        // Output is one u32 (4 bytes) per query.
        let out_bytes = (element_count as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_mip_for_footprint_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_mip_for_footprint_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_mip_for_footprint_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_mip_for_footprint_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_mip_for_footprint_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_mip_for_footprint_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_mip_for_footprint_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (element_count as u32).div_ceil(64);
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
        let out = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        out
    }
}

/// The `CPU` golden card-chain mip level for one query, re-exported so the
/// parity test can assert the device twin against the identical reference it
/// mirrors. `max_mip` is narrowed to the golden's `u8` mip index.
#[must_use]
pub fn reference_mip_for_footprint(query: MipQuery) -> u32 {
    let max_mip = query.max_mip.min(u32::from(u8::MAX)) as u8;
    u32::from(mip_for_footprint(query.max_dim, query.base_texels, max_mip))
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
