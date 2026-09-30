//! `wgpu` compute twin of Prism's froxel voxel forward-scatter decode
//! ([`voxel_forward_scatter`](prism_render_architecture::hair::dual_scattering::voxel_forward_scatter)).
//!
//! Where [`GpuHairVoxelTransmittance`](crate::voxel_transmittance::GpuHairVoxelTransmittance)
//! composites the froxel density slab into the running self-shadow
//! transmittance `T = product(1 - sigma_j)`, this kernel is its additive
//! sibling: it sums the same slab into the cumulative coverage-weighted
//! crossing count `n = sum(sigma_j)` over voxels `0..=index`, the monotonically
//! non-decreasing curve a dual-scattering shading pass samples for the `a_f^n`
//! forward-scatter exponent. Both decode the *same*
//! [`accumulate_voxel_density`](prism_render_architecture::hair::deep_transmittance::accumulate_voxel_density)
//! slab but carry independent information — the product discards how coverage
//! splits across strands while the sum preserves the raw count — so the voxel
//! self-shadow path can hand the shading side both `T` (attenuation) and `n`
//! (the exponent) from one froxel volume. Uniform voxels give a constant
//! stride, so the whole curve is a single prefix sum per light texel — no sort,
//! no transcendental. This crate is the on-device twin: one thread per light
//! texel walks its own disjoint density row and writes the prefix crossing
//! count for every voxel, so a passing real-device parity test is direct
//! evidence the ported kernel accumulates the same count curve as the reference
//! — not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairVoxelForwardScatter::eval`] takes one per-texel density row per
//! light texel (the froxel slab, e.g. the output of the density accumulation
//! twin) and returns one prefix-count curve per texel: output voxel `v` holds
//! [`voxel_forward_scatter`](prism_render_architecture::hair::dual_scattering::voxel_forward_scatter)
//! of that row at index `v`, so the whole emitted curve equals the golden
//! evaluated at every index. The host flattens the rows into a shared
//! `densities` pool with a compacted `texel_ranges` array; the kernel keeps a
//! running sum per row.
//!
//! # Portability
//!
//! The kernel uses only `max` and add in the portable core-`WGSL` subset — no
//! `exp`, `pow` or optional device feature — so the twin runs unmodified on
//! Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The accumulation is a closed-form running sum with no transcendental call,
//! and each thread owns a disjoint output row so the per-voxel add order
//! matches the reference `..=end` fold. `CPU` and `GPU` evaluate the same
//! arithmetic; they are not bit-exact only because a `GPU` may fuse a
//! multiply-add, perturbing the low mantissa bits by a few `ULP`. The parity
//! test therefore asserts a per-value tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`) rather than exact equality.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard uniform-voxel forward-scatter count plus `wgpu`
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

/// Uniform decode parameters uploaded to the kernel. `16`-byte scalar-packed
/// `repr(C)` matching `HairVoxelForwardScatterParams` in
/// `shaders/voxel_forward_scatter.wesl` (already a multiple of 16 for the
/// uniform block).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    texel_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One compacted per-texel slice descriptor uploaded to the kernel. `8`-byte
/// `repr(C)` matching `HairTexelRange` in `shaders/voxel_forward_scatter.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTexelRange {
    start: u32,
    count: u32,
}

/// A compiled, reusable voxel forward-scatter decode pipeline.
pub struct GpuHairVoxelForwardScatter {
    #[expect(
        dead_code,
        reason = "the shader module must outlive the pipeline that borrows it"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairVoxelForwardScatter {
    /// Compiles the voxel forward-scatter kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairVoxelForwardScatter {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_voxel_forward_scatter"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/voxel_forward_scatter.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_voxel_forward_scatter_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_voxel_forward_scatter_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_voxel_forward_scatter_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairVoxelForwardScatter {
            module,
            layout,
            pipeline,
        }
    }

    /// Accumulates one prefix-count curve per texel from the per-texel froxel
    /// density rows, returning one curve per texel (same order and length as
    /// the input rows).
    ///
    /// Output voxel `v` of row `t` equals
    /// [`voxel_forward_scatter`](prism_render_architecture::hair::dual_scattering::voxel_forward_scatter)
    /// applied to `densities[t]` at index `v`, to within the fused-multiply-add
    /// tolerance documented on this module. An empty input row yields an empty
    /// output row (the golden's `is_empty()` → `0.0` scalar case has no
    /// per-voxel slot). A zero-texel batch returns no rows without a dispatch
    /// (storage buffers cannot be zero-sized).
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, densities: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let texel_count = densities.len();

        if texel_count == 0 {
            return Vec::new();
        }

        // Flatten each texel's density row into a shared pool, with a compacted
        // range per texel.
        let total: usize = densities.iter().map(Vec::len).sum();
        let mut flat: Vec<f32> = Vec::with_capacity(total);
        let mut ranges: Vec<GpuTexelRange> = Vec::with_capacity(texel_count);
        for row in densities {
            let start = flat.len() as u32;
            flat.extend_from_slice(row);
            ranges.push(GpuTexelRange {
                start,
                count: row.len() as u32,
            });
        }

        // Storage buffers cannot be zero-sized: pad the density pool (and, by
        // the shared layout, the output pool) with one dummy entry when every
        // row was empty. The `count == 0` ranges keep every texel from touching
        // it.
        let padded = flat.is_empty();
        if padded {
            flat.push(0.0);
        }

        let uniform = Params {
            texel_count: texel_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let device = ctx.device();
        let pool_bytes = (flat.len() * size_of::<f32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_voxel_forward_scatter_params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM,
        });
        let densities_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_voxel_forward_scatter_densities"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE,
        });
        let ranges_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_voxel_forward_scatter_ranges"),
            contents: bytemuck::cast_slice(&ranges),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_voxel_forward_scatter_out"),
            size: pool_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_voxel_forward_scatter_out_stage"),
            size: pool_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_voxel_forward_scatter_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: densities_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: ranges_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_voxel_forward_scatter_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_voxel_forward_scatter_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (texel_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, pool_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let out_flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        // Slice the flat pool back into one curve per texel (empty rows stay
        // empty; the dummy pad, if any, is never sliced out).
        let mut curves: Vec<Vec<f32>> = Vec::with_capacity(texel_count);
        for range in &ranges {
            let start = range.start as usize;
            let count = range.count as usize;
            curves.push(out_flat[start..start + count].to_vec());
        }
        curves
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
