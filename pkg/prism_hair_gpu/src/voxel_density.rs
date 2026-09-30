//! `wgpu` compute twin of Prism's per-ray voxel opacity accumulation
//! ([`accumulate_voxel_density`](prism_render_architecture::hair::deep_transmittance::accumulate_voxel_density)).
//!
//! The froxel self-shadow tier bins strand opacity into a *uniform* slab of
//! voxels along each light ray — the coarse, sort-free sibling of the layered
//! deep opacity packing — forming a per-voxel optical density `sigma` a shading
//! pass composites into transmittance `T = product(1 - sigma_j)`. The reference
//! divides the light-space range `slab_start..slab_end` into `voxel_count`
//! equal voxels and adds each in-slab sample's clamped opacity into the voxel
//! its depth falls in, skipping samples outside the slab. This crate is the
//! on-device twin: one thread per light texel scatter-adds its own sample slice
//! into its own disjoint density row, so a passing real-device parity test is
//! direct evidence the ported kernel accumulates the same froxel densities as
//! the reference — not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairVoxelDensity::eval`] takes the per-texel [`TransmittanceBins`], the
//! slab range and `voxel_count`, and returns one density slab per texel (the
//! per-texel [`accumulate_voxel_density`](prism_render_architecture::hair::deep_transmittance::accumulate_voxel_density)).
//! Uniform voxels need no depth sort, so the host flattens each bucket into a
//! shared `samples` pool *in bin order* (matching the golden's input-order
//! scatter) with a compacted `texel_ranges` array, and the kernel only skips
//! out-of-slab samples and scatter-adds into its own row.
//!
//! # Portability
//!
//! The kernel uses only `clamp` and multiply/add/divide in the portable
//! core-`WGSL` subset — no `exp`, `pow` or optional device feature — so the
//! twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The accumulation is a closed-form clamped sum with no transcendental call,
//! and each thread owns a disjoint density row so the per-voxel add order
//! matches the reference bucket order. `CPU` and `GPU` evaluate the same
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
//! Provenance: standard uniform-voxel opacity accumulation plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::deep_transmittance::TransmittanceBins;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform accumulation parameters uploaded to the kernel. `16`-byte
/// scalar-packed `repr(C)` matching `HairVoxelParams` in
/// `shaders/voxel_density.wesl` (already a multiple of 16 for the uniform
/// block).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    slab_start: f32,
    slab_end: f32,
    voxel_count: u32,
    texel_count: u32,
}

/// One compacted per-texel slice descriptor uploaded to the kernel. `8`-byte
/// `repr(C)` matching `HairTexelRange` in `shaders/voxel_density.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTexelRange {
    start: u32,
    count: u32,
}

/// A compiled, reusable voxel density accumulation pipeline.
pub struct GpuHairVoxelDensity {
    #[expect(
        dead_code,
        reason = "the shader module must outlive the pipeline that borrows it"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairVoxelDensity {
    /// Compiles the voxel density kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairVoxelDensity {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_voxel_density"),
            source: ShaderSource::Wgsl(include_str!("../shaders/voxel_density.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_voxel_density_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_voxel_density_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_voxel_density_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairVoxelDensity {
            module,
            layout,
            pipeline,
        }
    }

    /// Accumulates one uniform voxel density slab per texel from the per-texel
    /// strand `bins`, returning one `voxel_count`-long slab per texel (same
    /// order as the bins).
    ///
    /// Each slab equals
    /// [`accumulate_voxel_density`](prism_render_architecture::hair::deep_transmittance::accumulate_voxel_density)
    /// applied to that texel's bucket, to within the fused-multiply-add
    /// tolerance documented on this module. Uniform voxels need no sort, so the
    /// host preserves bin order and the kernel scatter-adds in that same order.
    /// A degenerate slab (`slab_end <= slab_start`) yields all-zero rows. A
    /// zero-texel map returns no rows without a dispatch (storage buffers cannot
    /// be zero-sized).
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        bins: &TransmittanceBins,
        slab_start: f32,
        slab_end: f32,
        voxel_count: u32,
    ) -> Vec<Vec<f32>> {
        let voxel_count = voxel_count.max(1) as usize;
        let texel_count = bins.len();

        if texel_count == 0 {
            return Vec::new();
        }

        // Flatten each texel's samples into a shared pool in bin order (no sort
        // for uniform voxels), with a compacted range per texel.
        let mut samples: Vec<[f32; 2]> = Vec::with_capacity(bins.total());
        let mut ranges: Vec<GpuTexelRange> = Vec::with_capacity(texel_count);
        for texel in 0..texel_count {
            let bucket = bins.bucket(texel as u32).unwrap_or(&[]);
            let start = samples.len() as u32;
            for s in bucket {
                samples.push([s.depth, s.opacity]);
            }
            ranges.push(GpuTexelRange {
                start,
                count: bucket.len() as u32,
            });
        }

        // Storage buffers cannot be zero-sized: pad the sample pool with one
        // dummy entry when every bucket was empty. The `count == 0` ranges keep
        // every texel from reading it.
        if samples.is_empty() {
            samples.push([0.0, 0.0]);
        }

        let uniform = Params {
            slab_start,
            slab_end,
            voxel_count: voxel_count as u32,
            texel_count: texel_count as u32,
        };

        let device = ctx.device();
        let density_bytes = (texel_count * voxel_count * size_of::<f32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_voxel_density_params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM,
        });
        let samples_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_voxel_density_samples"),
            contents: bytemuck::cast_slice(&samples),
            usage: BufferUsages::STORAGE,
        });
        let ranges_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_voxel_density_ranges"),
            contents: bytemuck::cast_slice(&ranges),
            usage: BufferUsages::STORAGE,
        });
        let density_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_voxel_density_out"),
            size: density_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let density_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_voxel_density_out_stage"),
            size: density_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_voxel_density_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: samples_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: ranges_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: density_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_voxel_density_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_voxel_density_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (texel_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&density_buf, 0, &density_stage, 0, density_bytes);
        ctx.queue().submit([encoder.finish()]);

        density_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = density_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        density_stage.unmap();

        // Slice the flat texel-major slab back into one row per texel.
        (0..texel_count)
            .map(|texel| {
                let base = texel * voxel_count;
                flat[base..base + voxel_count].to_vec()
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
