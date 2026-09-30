//! Real-device `wgpu` compute implementation of the `LSD` radix sort.
//!
//! [`GpuRadixSort`] compiles `shaders/radix_count.wgsl` and
//! `shaders/radix_scatter.wgsl` once and exposes [`GpuRadixSort::sort_keys`] and
//! [`GpuRadixSort::sort_pairs`], stable ascending sorts of 32-bit keys (and an
//! optional 32-bit payload) built on the sibling [`GpuScan`]. Sorting `u32`
//! keys is a pure integer permutation, so a passing real-device parity test is
//! direct bit-for-bit evidence that the ported kernels produce the same ordered
//! stream as the [`cpu_radix_sort_keys`](super::cpu::cpu_radix_sort_keys) and
//! [`cpu_radix_sort_pairs`](super::cpu::cpu_radix_sort_pairs) golden twins.
//!
//! # Host orchestration
//!
//! The sort runs [`PASSES`] least-significant-digit passes over
//! [`RADIX_BITS`]-bit digits, ping-ponging between two key buffers (and two
//! payload buffers) so each pass reads the previous pass's stable output. A
//! single pass is three recorded steps: a `count` dispatch builds each block's
//! digit histogram in digit-major order; the sibling [`GpuScan`] exclusive-scans
//! that histogram in place on the device, turning it into per-`(digit, block)`
//! output bases; and a `scatter` dispatch reprocesses each block, deriving a
//! stable in-block rank and writing every key to its scanned base. All
//! [`PASSES`] passes are recorded into one encoder and submitted together, so
//! `wgpu` orders the dispatches and only the final key (and payload) buffer is
//! read back.
//!
//! # Keys-only variant
//!
//! [`GpuRadixSort::sort_keys`] routes through the pairs path with an identity
//! payload `0..n` and discards the sorted payload. The single scatter kernel
//! always moves a key and its payload, so keys-only sorting spends a little
//! extra bandwidth carrying an unused payload; this keeps one code path correct
//! rather than two.
//!
//! Provenance: the `LSD` radix sort with a per-block, digit-major histogram
//! scanned to output offsets is a classical, openly published `GPU` technique
//! (Blelloch 1990; Satish, Harris, Garland 2009). No Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, Buffer,
    BufferBindingType, CommandEncoder, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;
use crate::scan::GpuScan;

use super::config::{PASSES, RADIX, TILE};
use super::layout::{buffer_entry, entry};

/// Uniform parameters shared with `Params` in both radix shaders.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of keys.
    n: u32,
    /// Which digit this pass extracts (0 = least significant).
    pass: u32,
    /// Number of blocks the keys are tiled into.
    num_blocks: u32,
    /// Padding to a 16-byte boundary.
    pad0: u32,
}

/// A compiled, reusable `GPU` `LSD` radix sort pipeline set.
pub struct GpuRadixSort {
    /// Kept alive so the count pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    count_module: ShaderModule,
    /// Kept alive so the scatter pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    scatter_module: ShaderModule,
    /// Layout wiring params, the keys, and the per-block digit histogram.
    count_layout: BindGroupLayout,
    /// Layout wiring params, the input key/payload, the offsets, and the output.
    scatter_layout: BindGroupLayout,
    /// Builds each block's digit-major histogram for one pass.
    count: ComputePipeline,
    /// Scatters each block's keys to their scanned, stable output bases.
    scatter: ComputePipeline,
    /// The sibling scan used to turn each pass's histogram into output offsets.
    scan: GpuScan,
}

impl GpuRadixSort {
    /// Compiles the count and scatter kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRadixSort {
        let device = ctx.device();

        let count_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_radix_count"),
            source: ShaderSource::Wgsl(include_str!("../shaders/radix_count.wgsl").into()),
        });
        let scatter_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_radix_scatter"),
            source: ShaderSource::Wgsl(include_str!("../shaders/radix_scatter.wgsl").into()),
        });

        let count_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_radix_count_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let scatter_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_radix_scatter_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
            ],
        });

        let count_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_radix_count_pipeline_layout"),
            bind_group_layouts: &[Some(&count_layout)],
            immediate_size: 0,
        });
        let scatter_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_radix_scatter_pipeline_layout"),
            bind_group_layouts: &[Some(&scatter_layout)],
            immediate_size: 0,
        });

        let count = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_radix_count_pipeline"),
            layout: Some(&count_pipeline_layout),
            module: &count_module,
            entry_point: Some("count"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let scatter = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_radix_scatter_pipeline"),
            layout: Some(&scatter_pipeline_layout),
            module: &scatter_module,
            entry_point: Some("scatter"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        GpuRadixSort {
            count_module,
            scatter_module,
            count_layout,
            scatter_layout,
            count,
            scatter,
            scan: GpuScan::new(ctx),
        }
    }

    /// Sorts `keys` ascending on the `GPU` with a stable `LSD` radix sort.
    ///
    /// Returns a new vector; the input is left untouched. Matches
    /// [`cpu_radix_sort_keys`](super::cpu::cpu_radix_sort_keys) bit-for-bit. An
    /// empty input yields an empty vector without touching the device.
    #[must_use]
    pub fn sort_keys(&self, ctx: &GpuContext, keys: &[u32]) -> Vec<u32> {
        let identity: Vec<u32> = (0..u32::try_from(keys.len()).unwrap_or(u32::MAX)).collect();
        self.sort_pairs(ctx, keys, &identity).0
    }

    /// Sorts `keys` ascending on the `GPU`, carrying each entry of `values`
    /// alongside its key.
    ///
    /// Returns the sorted keys and their reordered payloads. Equal keys keep
    /// input order, so the payloads follow their keys stably, matching
    /// [`cpu_radix_sort_pairs`](super::cpu::cpu_radix_sort_pairs) bit-for-bit.
    /// An empty input yields two empty vectors without touching the device.
    ///
    /// # Panics
    ///
    /// Panics if `keys` and `values` do not have the same length.
    #[must_use]
    pub fn sort_pairs(
        &self,
        ctx: &GpuContext,
        keys: &[u32],
        values: &[u32],
    ) -> (Vec<u32>, Vec<u32>) {
        assert!(
            keys.len() == values.len(),
            "keys and values must have equal length"
        );
        let n = keys.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }

        let device = ctx.device();
        let bytes = size_of_val(keys) as u64;
        let num_blocks = n.div_ceil(TILE as usize);
        let hist_len = (RADIX as usize) * num_blocks;

        // Ping-pong key and payload buffers: each pass reads `src` and writes
        // `dst`, then the roles swap. After an even number of passes `src` again
        // names the initially uploaded buffer, which by then holds the sorted
        // result.
        let mut src_keys = buffer::storage_rw_init(device, "prism_radix_keys_a", keys);
        let mut dst_keys = buffer::storage_rw_zeroed(device, "prism_radix_keys_b", bytes);
        let mut src_vals = buffer::storage_rw_init(device, "prism_radix_vals_a", values);
        let mut dst_vals = buffer::storage_rw_zeroed(device, "prism_radix_vals_b", bytes);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_radix_encoder"),
        });

        // Everything recorded into the encoder must outlive submission.
        let mut params_bufs: Vec<Buffer> = Vec::new();
        let mut binds: Vec<BindGroup> = Vec::new();
        let mut levels = Vec::new();

        let groups = u32::try_from(num_blocks).unwrap_or(u32::MAX);
        for pass in 0..PASSES {
            let params = buffer::uniform(
                device,
                "prism_radix_params",
                &Params {
                    n: u32::try_from(n).unwrap_or(u32::MAX),
                    pass,
                    num_blocks: u32::try_from(num_blocks).unwrap_or(u32::MAX),
                    pad0: 0,
                },
            );

            // The per-block, digit-major histogram this pass fills, then scans.
            let block_hist = buffer::storage_rw_zeroed(
                device,
                "prism_radix_block_hist",
                (hist_len * size_of::<u32>()) as u64,
            );

            let count_bind = device.create_bind_group(&BindGroupDescriptor {
                label: Some("prism_radix_count_bind"),
                layout: &self.count_layout,
                entries: &[
                    entry(0, &params),
                    entry(1, &src_keys),
                    entry(2, &block_hist),
                ],
            });
            Self::dispatch(
                &mut encoder,
                "prism_radix_count_pass",
                &self.count,
                &count_bind,
                groups,
            );

            // Scan the histogram in place on the device; `result()` then holds
            // the per-(digit, block) output bases the scatter writes into. The
            // count bind group keeps a strong reference to `block_hist`, so it
            // stays valid after the handle moves into the scan.
            let scanned = self
                .scan
                .record_scan(device, &mut encoder, block_hist, hist_len);

            let scatter_bind = device.create_bind_group(&BindGroupDescriptor {
                label: Some("prism_radix_scatter_bind"),
                layout: &self.scatter_layout,
                entries: &[
                    entry(0, &params),
                    entry(1, &src_keys),
                    entry(2, &src_vals),
                    entry(3, scanned.result()),
                    entry(4, &dst_keys),
                    entry(5, &dst_vals),
                ],
            });
            Self::dispatch(
                &mut encoder,
                "prism_radix_scatter_pass",
                &self.scatter,
                &scatter_bind,
                groups,
            );

            params_bufs.push(params);
            binds.push(count_bind);
            binds.push(scatter_bind);
            levels.push(scanned);

            std::mem::swap(&mut src_keys, &mut dst_keys);
            std::mem::swap(&mut src_vals, &mut dst_vals);
        }

        // After `PASSES` swaps `src_*` names the buffers holding the result.
        let keys_stage = buffer::staging(device, "prism_radix_keys_stage", bytes);
        let vals_stage = buffer::staging(device, "prism_radix_vals_stage", bytes);
        buffer::copy(&mut encoder, &src_keys, &keys_stage, bytes);
        buffer::copy(&mut encoder, &src_vals, &vals_stage, bytes);
        ctx.queue().submit([encoder.finish()]);

        let sorted_keys = buffer::read_back::<u32>(ctx, &keys_stage);
        let sorted_vals = buffer::read_back::<u32>(ctx, &vals_stage);
        drop(levels);
        drop(binds);
        drop(params_bufs);
        (sorted_keys, sorted_vals)
    }

    /// Records one block-granular dispatch of `pipeline` bound to `bind`.
    fn dispatch(
        encoder: &mut CommandEncoder,
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

#[cfg(test)]
mod tests {
    use super::PASSES;
    use crate::radix::config::RADIX_BITS;

    #[test]
    fn radix_bits_and_passes_cover_a_u32() {
        assert_eq!(RADIX_BITS * PASSES, u32::BITS);
    }
}
