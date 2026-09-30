//! `wgpu` compute twin of Prism's per-texel transmittance-sample binning
//! ([`bin_samples`](prism_render_architecture::hair::deep_transmittance::bin_samples)).
//!
//! Both self-shadow paths — the layered deep opacity map and the froxel voxel
//! slab — start from the same fan-out: a flat stream of strand samples, each
//! tagged with the light texel/ray it lands on, must be routed into one bucket
//! per texel before either accumulation can run. The reference
//! [`bin_samples`](prism_render_architecture::hair::deep_transmittance::bin_samples)
//! is the pure, deterministic partition pass that does this: it preserves input
//! order inside every bucket and silently skips any sample whose texel index is
//! out of range. This crate is the on-device twin: one thread per destination
//! texel walks the indexed sample stream and appends every matching sample into
//! its own disjoint output slice, so a passing real-device parity test is
//! direct evidence the ported routing lands each sample in the same bucket, in
//! the same order, as the reference — not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairBinSamples::eval`] takes the indexed sample stream and a texel
//! count and returns one `(depth, opacity)` row per texel (the bucket contents
//! in input order), matching
//! [`TransmittanceBins::bucket`](prism_render_architecture::hair::deep_transmittance::TransmittanceBins::bucket)
//! for every texel. The host pre-counts each texel's population (skipping
//! out-of-range tags exactly like the golden), builds an exclusive-prefix
//! `texel_ranges` layout, and each thread copies its samples into its slice.
//!
//! # Portability
//!
//! The kernel performs no arithmetic at all — it only compares texel tags and
//! copies `vec2<f32>` payloads — so it uses the most portable core-`WGSL`
//! subset and runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Because the kernel only copies payloads, the routing is exact: the parity
//! test asserts bit-identical `(depth, opacity)` pairs and bucket order, not a
//! tolerance. Each thread owns a disjoint output slice, so there is no
//! cross-thread contention and no atomics; walking the stream in ascending
//! index and appending in encounter order reproduces the reference's stable
//! push order.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard per-bucket scatter/gather partition plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::hair::deep_transmittance::TexelSample;

use crate::context::GpuContext;

/// Uniform binning parameters uploaded to the kernel. `16`-byte scalar-packed
/// `repr(C)` matching `HairBinParams` in `shaders/bin_samples.wesl` (already a
/// multiple of 16 for the uniform block).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    sample_count: u32,
    texel_count: u32,
    pad0: u32,
    pad1: u32,
}

/// One compacted per-texel slice descriptor uploaded to the kernel. `8`-byte
/// `repr(C)` matching `HairTexelRange` in `shaders/bin_samples.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTexelRange {
    start: u32,
    count: u32,
}

/// A compiled, reusable per-texel sample binning pipeline.
pub struct GpuHairBinSamples {
    #[expect(
        dead_code,
        reason = "the shader module must outlive the pipeline that borrows it"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairBinSamples {
    /// Compiles the sample-binning kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairBinSamples {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_bin_samples"),
            source: ShaderSource::Wgsl(include_str!("../shaders/bin_samples.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_bin_samples_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_bin_samples_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_bin_samples_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairBinSamples {
            module,
            layout,
            pipeline,
        }
    }

    /// Routes the indexed sample stream into one bucket per texel, returning one
    /// `(depth, opacity)` row per texel in input (encounter) order.
    ///
    /// Row `t` equals
    /// [`TransmittanceBins::bucket`](prism_render_architecture::hair::deep_transmittance::TransmittanceBins::bucket)
    /// of the golden [`bin_samples`](prism_render_architecture::hair::deep_transmittance::bin_samples)
    /// applied to the same input: same members, same order, bit-for-bit (the
    /// kernel only copies payloads). A sample whose `texel` falls outside
    /// `0..effective` is skipped exactly like the reference. `texel_count` is
    /// clamped to at least `1` (matching `TransmittanceBins::new`), so the
    /// result always has `texel_count.max(1)` rows; empty buckets stay empty.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        samples: &[TexelSample],
        texel_count: u32,
    ) -> Vec<Vec<[f32; 2]>> {
        // `TransmittanceBins::new` clamps the bucket count to at least 1.
        let effective = texel_count.max(1) as usize;

        // Pre-count each texel's population in input order, skipping any tag
        // that falls outside `0..effective` exactly like the golden's `push`.
        let mut counts = vec![0u32; effective];
        for entry in samples {
            let texel = entry.texel as usize;
            if texel < effective {
                counts[texel] += 1;
            }
        }

        // Exclusive-prefix offsets → compacted per-texel output ranges + total.
        let mut ranges: Vec<GpuTexelRange> = Vec::with_capacity(effective);
        let mut running = 0u32;
        for &count in &counts {
            ranges.push(GpuTexelRange {
                start: running,
                count,
            });
            running += count;
        }
        let total = running as usize;

        // Flatten the input stream into the two parallel upload pools (tags and
        // payloads), in input order. Out-of-range tags stay in the pool but are
        // never matched by any thread `t < effective`.
        let sample_count = samples.len();
        let mut sample_texels: Vec<u32> = Vec::with_capacity(sample_count);
        let mut sample_values: Vec<[f32; 2]> = Vec::with_capacity(sample_count);
        for entry in samples {
            sample_texels.push(entry.texel);
            sample_values.push([entry.sample.depth, entry.sample.opacity]);
        }

        // Storage buffers cannot be zero-sized. Pad the empty input pools with a
        // single dummy entry; `sample_count` still carries the true `0`, so the
        // kernel never reads the dummy. Pad the output pool likewise when no
        // sample landed anywhere (`count == 0` ranges keep it untouched).
        if sample_texels.is_empty() {
            sample_texels.push(0);
            sample_values.push([0.0, 0.0]);
        }
        let mut out_len = total;
        if out_len == 0 {
            out_len = 1;
        }

        let uniform = Params {
            sample_count: sample_count as u32,
            texel_count: effective as u32,
            pad0: 0,
            pad1: 0,
        };

        let device = ctx.device();
        let out_bytes = (out_len * size_of::<[f32; 2]>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_bin_samples_params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM,
        });
        let texels_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_bin_samples_texels"),
            contents: bytemuck::cast_slice(&sample_texels),
            usage: BufferUsages::STORAGE,
        });
        let values_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_bin_samples_values"),
            contents: bytemuck::cast_slice(&sample_values),
            usage: BufferUsages::STORAGE,
        });
        let ranges_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_bin_samples_ranges"),
            contents: bytemuck::cast_slice(&ranges),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_bin_samples_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_bin_samples_out_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_bin_samples_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: texels_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: values_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: ranges_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_bin_samples_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_bin_samples_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (effective as u32).div_ceil(64);
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
        let out_flat = bytemuck::cast_slice::<u8, [f32; 2]>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        // Slice the flat pool back into one bucket per texel (empty buckets stay
        // empty; the dummy pad, if any, is never sliced out).
        let mut buckets: Vec<Vec<[f32; 2]>> = Vec::with_capacity(effective);
        for range in &ranges {
            let start = range.start as usize;
            let count = range.count as usize;
            buckets.push(out_flat[start..start + count].to_vec());
        }
        buckets
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
