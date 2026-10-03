//! Real-device `wgpu` compute implementation of the SER 64-bit `LSD` radix
//! sort.
//!
//! [`GpuRayReorder`] compiles `shaders/reorder_count.wgsl` and
//! `shaders/reorder_scatter.wgsl` once and exposes [`GpuRayReorder::reorder`],
//! a stable ascending sort of 64-bit
//! [`CoherenceKey`](prism_render_architecture::ray_scene::reorder::CoherenceKey)
//! values that returns the ray permutation. Because reordering rays is a pure
//! permutation of indices, a passing real-device test is direct bit-for-bit
//! evidence that the ported kernels produce the same ordering as the
//! [`radix_order`](prism_render_architecture::ray_scene::reorder::radix_order)
//! `CPU` golden.
//!
//! # Host orchestration
//!
//! The sort runs [`PASSES`] least-significant-digit passes over
//! [`RADIX_BITS`]-bit digits, ping-ponging between two key buffers (low word,
//! high word) and two payload buffers so each pass reads the previous pass's
//! stable output. A `WGSL` kernel cannot address a native 64-bit integer, so
//! the key is split into two `u32` words and the first [`LOW_WORD_PASSES`]
//! passes read digits from the low word while the rest read from the high word
//! — identical in ordering to one logical 64-bit `LSD` sort.
//!
//! Each pass is three steps: a `count` dispatch builds every block's digit
//! histogram in digit-major order on the device; the host exclusive-scans that
//! small `RADIX * num_blocks` histogram into per-`(digit, block)` output bases;
//! and a `scatter` dispatch reprocesses each block, derives a stable in-block
//! rank, and writes every key word and payload to its scanned base on the
//! device. The two O(n) data-movement steps (count, scatter) therefore run on
//! the `GPU`; only the tiny digit-major prefix sum runs on the host. Moving that
//! scan on-device (as `prism_physics_gpu::scan` does for composability) is a
//! documented future optimisation; it does not change the sorted result, which
//! this module validates against the `CPU` golden element-for-element.
//!
//! Provenance: the `LSD` radix sort with a per-block, digit-major histogram
//! scanned to output offsets is a classical, openly published `GPU` technique
//! (Blelloch 1990; Satish, Harris, Garland 2009). No Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor,
    BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource,
};

use prism_render_architecture::ray_scene::reorder::CoherenceKey;

use crate::buffer;
use crate::context::GpuContext;

use super::config::{LOW_WORD_PASSES, PASSES, RADIX, RADIX_BITS, TILE};
use super::layout::{buffer_entry, entry};

/// Uniform parameters shared with `Params` in both reorder shaders.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of keys.
    n: u32,
    /// Bit offset of the digit within its 32-bit word (0, 8, 16, or 24).
    digit_shift: u32,
    /// 0 selects the low word, 1 selects the high word.
    use_hi: u32,
    /// Number of blocks the keys are tiled into.
    num_blocks: u32,
}

/// A compiled, reusable `GPU` SER radix-sort pipeline set.
pub struct GpuRayReorder {
    /// Kept alive so the count pipeline it produced stays valid.
    #[expect(dead_code, reason = "kept alive so the pipeline it produced stays valid")]
    count_module: ShaderModule,
    /// Kept alive so the scatter pipeline it produced stays valid.
    #[expect(dead_code, reason = "kept alive so the pipeline it produced stays valid")]
    scatter_module: ShaderModule,
    /// Layout wiring params, the two key words, and the per-block histogram.
    count_layout: BindGroupLayout,
    /// Layout wiring params, the input key/payload, offsets, and the output.
    scatter_layout: BindGroupLayout,
    /// Builds each block's digit-major histogram for one pass.
    count: ComputePipeline,
    /// Scatters each block's keys to their scanned, stable output bases.
    scatter: ComputePipeline,
}

impl GpuRayReorder {
    /// Compiles the count and scatter kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayReorder {
        let device = ctx.device();

        let count_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_reorder_count"),
            source: ShaderSource::Wgsl(include_str!("../shaders/reorder_count.wgsl").into()),
        });
        let scatter_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_reorder_scatter"),
            source: ShaderSource::Wgsl(include_str!("../shaders/reorder_scatter.wgsl").into()),
        });

        let count_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_reorder_count_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let scatter_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_reorder_scatter_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
                buffer_entry(7, BufferBindingType::Storage { read_only: false }),
            ],
        });

        let count_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_reorder_count_pipeline_layout"),
            bind_group_layouts: &[Some(&count_layout)],
            immediate_size: 0,
        });
        let scatter_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_reorder_scatter_pipeline_layout"),
            bind_group_layouts: &[Some(&scatter_layout)],
            immediate_size: 0,
        });

        let count = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_reorder_count_pipeline"),
            layout: Some(&count_pipeline_layout),
            module: &count_module,
            entry_point: Some("count"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let scatter = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_reorder_scatter_pipeline"),
            layout: Some(&scatter_pipeline_layout),
            module: &scatter_module,
            entry_point: Some("scatter"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        GpuRayReorder {
            count_module,
            scatter_module,
            count_layout,
            scatter_layout,
            count,
            scatter,
        }
    }

    /// Stably sorts `keys` ascending on the `GPU` and returns the ray
    /// permutation.
    ///
    /// `order[i]` is the source ray index placed at reordered slot `i`, equal to
    /// [`radix_order`](prism_render_architecture::ray_scene::reorder::radix_order)
    /// bit-for-bit. Equal keys keep ascending source order (stable). An empty
    /// input yields an empty vector without touching the device.
    #[must_use]
    pub fn reorder(&self, ctx: &GpuContext, keys: &[CoherenceKey]) -> Vec<u32> {
        let n = keys.len();
        if n == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        // Split the 64-bit keys into low/high 32-bit words and seed the payload
        // with the identity permutation; the sort carries each ray's source
        // index so the final payload *is* the permutation.
        let lo: Vec<u32> = keys.iter().map(|k| k.raw() as u32).collect();
        let hi: Vec<u32> = keys.iter().map(|k| (k.raw() >> 32) as u32).collect();
        let vals: Vec<u32> = (0..u32::try_from(n).unwrap_or(u32::MAX)).collect();

        let bytes = (n * size_of::<u32>()) as u64;
        let num_blocks = n.div_ceil(TILE as usize);
        let hist_len = (RADIX as usize) * num_blocks;
        let groups = u32::try_from(num_blocks).unwrap_or(u32::MAX);

        // Ping-pong buffers: each pass reads `src_*` and writes `dst_*`.
        let mut src_lo = buffer::storage_rw_init(device, "prism_reorder_lo_a", &lo);
        let mut dst_lo = buffer::storage_rw_zeroed(device, "prism_reorder_lo_b", bytes);
        let mut src_hi = buffer::storage_rw_init(device, "prism_reorder_hi_a", &hi);
        let mut dst_hi = buffer::storage_rw_zeroed(device, "prism_reorder_hi_b", bytes);
        let mut src_vals = buffer::storage_rw_init(device, "prism_reorder_vals_a", &vals);
        let mut dst_vals = buffer::storage_rw_zeroed(device, "prism_reorder_vals_b", bytes);

        for pass in 0..PASSES {
            let use_hi = u32::from(pass >= LOW_WORD_PASSES);
            let digit_shift = (pass % LOW_WORD_PASSES) * RADIX_BITS;
            let params = buffer::uniform(
                device,
                "prism_reorder_params",
                &Params {
                    n: u32::try_from(n).unwrap_or(u32::MAX),
                    digit_shift,
                    use_hi,
                    num_blocks: u32::try_from(num_blocks).unwrap_or(u32::MAX),
                },
            );

            // --- count: build the per-(digit, block) histogram on device. ---
            let block_hist = buffer::storage_rw_zeroed(
                device,
                "prism_reorder_block_hist",
                (hist_len * size_of::<u32>()) as u64,
            );
            let count_bind = device.create_bind_group(&BindGroupDescriptor {
                label: Some("prism_reorder_count_bind"),
                layout: &self.count_layout,
                entries: &[
                    entry(0, &params),
                    entry(1, &src_lo),
                    entry(2, &src_hi),
                    entry(3, &block_hist),
                ],
            });
            let hist_stage =
                buffer::staging(device, "prism_reorder_hist_stage", (hist_len * 4) as u64);
            let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("prism_reorder_count_encoder"),
            });
            Self::dispatch(&mut enc, "prism_reorder_count_pass", &self.count, &count_bind, groups);
            buffer::copy(&mut enc, &block_hist, &hist_stage, (hist_len * 4) as u64);
            ctx.queue().submit([enc.finish()]);
            let hist = buffer::read_back::<u32>(ctx, &hist_stage);

            // --- host exclusive scan of the digit-major histogram. ---
            let offsets = exclusive_scan(&hist);
            let offsets_buf = buffer::storage_read(device, "prism_reorder_offsets", &offsets);

            // --- scatter: stable write to scanned bases on device. ---
            let scatter_bind = device.create_bind_group(&BindGroupDescriptor {
                label: Some("prism_reorder_scatter_bind"),
                layout: &self.scatter_layout,
                entries: &[
                    entry(0, &params),
                    entry(1, &src_lo),
                    entry(2, &src_hi),
                    entry(3, &src_vals),
                    entry(4, &offsets_buf),
                    entry(5, &dst_lo),
                    entry(6, &dst_hi),
                    entry(7, &dst_vals),
                ],
            });
            let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("prism_reorder_scatter_encoder"),
            });
            Self::dispatch(
                &mut enc,
                "prism_reorder_scatter_pass",
                &self.scatter,
                &scatter_bind,
                groups,
            );
            ctx.queue().submit([enc.finish()]);

            std::mem::swap(&mut src_lo, &mut dst_lo);
            std::mem::swap(&mut src_hi, &mut dst_hi);
            std::mem::swap(&mut src_vals, &mut dst_vals);
        }

        // After `PASSES` (even) swaps, `src_vals` names the sorted payload.
        let vals_stage = buffer::staging(device, "prism_reorder_vals_stage", bytes);
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_reorder_readback_encoder"),
        });
        buffer::copy(&mut enc, &src_vals, &vals_stage, bytes);
        ctx.queue().submit([enc.finish()]);
        buffer::read_back::<u32>(ctx, &vals_stage)
    }

    /// Records one block-granular dispatch of `pipeline` bound to `bind`.
    fn dispatch(
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &ComputePipeline,
        bind: &BindGroup,
        groups: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

/// Exclusive prefix sum of `hist`; `out[i] = sum(hist[0..i])`.
///
/// This is the host twin of the digit-major histogram scan the scatter pass
/// consumes. It is exact `u32` accumulation (the total is the key count, which
/// fits in `u32`), so it matches the `CPU` golden's bucket layout.
#[must_use]
fn exclusive_scan(hist: &[u32]) -> Vec<u32> {
    let mut out = vec![0u32; hist.len()];
    let mut running = 0u32;
    for (slot, &count) in out.iter_mut().zip(hist.iter()) {
        *slot = running;
        running = running.wrapping_add(count);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radix_bits_and_passes_cover_a_u64() {
        assert_eq!(RADIX_BITS * PASSES, u64::BITS);
    }

    #[test]
    fn exclusive_scan_matches_running_sum() {
        assert_eq!(exclusive_scan(&[]), Vec::<u32>::new());
        assert_eq!(exclusive_scan(&[3, 0, 2, 5]), vec![0, 3, 3, 5]);
    }
}
