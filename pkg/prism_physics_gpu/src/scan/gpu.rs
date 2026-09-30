//! Real-device `wgpu` compute implementation of the exclusive scan and stream
//! compaction.
//!
//! [`GpuScan`] compiles `shaders/scan.wgsl` and `shaders/scan_scatter.wgsl`
//! once and exposes [`GpuScan::exclusive_scan`], a work-efficient parallel
//! prefix sum over an arbitrarily long `u32` stream, and [`GpuScan::compact`],
//! a flag-driven stream compaction built on that scan. Both match their golden
//! twins ([`cpu_exclusive_scan`](super::cpu::cpu_exclusive_scan) and
//! [`cpu_compact`](super::cpu::cpu_compact)) bit-for-bit: the fold is `u32`
//! addition, which is associative and wraps identically on host and device, so
//! parity is exact rather than within a tolerance.
//!
//! # Host orchestration
//!
//! One workgroup scans one [`BLOCK`](super::config::BLOCK)-wide slice, so a
//! stream longer than a block cannot be scanned in a single dispatch. The host
//! walks a pyramid of levels: `scan_block` runs down the pyramid, leaving each
//! level block-local exclusive and emitting the next, `BLOCK`-times smaller
//! level of per-block totals; `add_offsets` runs back up, folding each level's
//! globally scanned block offsets into the level below. The top level is a
//! single block whose scan is already global, and its lone total is the grand
//! total. Each dispatch is its own compute pass, so `wgpu` orders them and
//! makes each pass's storage writes visible to the next.
//!
//! Provenance: the Blelloch work-efficient scan and flag-driven stream
//! compaction are classical, openly published parallel primitives (Blelloch
//! 1990; Harris, Sengupta, Owens, GPU Gems 3, 2007). No Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, Buffer,
    BufferBindingType, CommandEncoder, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipeline, ComputePipelineDescriptor, Device, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;

use super::config::{BLOCK, WORKGROUP};
use super::layout::{buffer_entry, entry};

/// Uniform parameters shared with `Params` in both scan shaders.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of elements in the level (or input) this dispatch touches.
    n: u32,
    /// Padding to a 16-byte boundary.
    pad0: u32,
    /// Padding to a 16-byte boundary.
    pad1: u32,
    /// Padding to a 16-byte boundary.
    pad2: u32,
}

/// The device buffers and per-dispatch bindings produced while recording a
/// full multi-level scan into a command encoder.
///
/// Ownership is returned to the caller so every buffer and bind group outlives
/// the submitted encoder. `buffers[0]` is scanned in place and holds the final
/// exclusive scan on completion; the last entry is the length-one grand total.
/// It is `pub(crate)` so sibling primitives (for example the radix sort) can
/// scan an on-device histogram in place and read the offsets back without a
/// host round-trip.
pub(crate) struct ScanLevels {
    /// Per-level device buffers, coarsest last; `buffers[0]` is the input level.
    buffers: Vec<Buffer>,
    /// Per-dispatch uniform parameter buffers, kept alive for submission.
    #[expect(
        dead_code,
        reason = "kept alive so the recorded dispatches keep valid bindings until submit"
    )]
    params: Vec<Buffer>,
    /// Per-dispatch bind groups, kept alive for submission.
    #[expect(
        dead_code,
        reason = "kept alive so the recorded dispatches keep valid bindings until submit"
    )]
    binds: Vec<BindGroup>,
}

impl ScanLevels {
    /// The buffer holding the exclusive scan of the input once the recorded
    /// encoder has been submitted (the in-place scanned input level).
    #[must_use]
    pub(crate) fn result(&self) -> &Buffer {
        &self.buffers[0]
    }
}

/// A compiled, reusable `GPU` scan and compaction pipeline set.
pub struct GpuScan {
    /// Kept alive so the scan pipelines it produced stay valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    scan_module: ShaderModule,
    /// Kept alive so the scatter pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    scatter_module: ShaderModule,
    /// Layout wiring params, the level values, and per-block sums.
    scan_layout: BindGroupLayout,
    /// Layout wiring params, flags, offsets, payload, and the dense output.
    scatter_layout: BindGroupLayout,
    /// Scans one block per workgroup and emits per-block totals.
    scan_block: ComputePipeline,
    /// Folds globally scanned block offsets back into a level.
    add_offsets: ComputePipeline,
    /// Gathers flagged payload entries into a dense output.
    scatter: ComputePipeline,
}

impl GpuScan {
    /// Compiles the scan and scatter kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuScan {
        let device = ctx.device();

        let scan_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_scan"),
            source: ShaderSource::Wgsl(include_str!("../shaders/scan.wgsl").into()),
        });
        let scatter_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_scan_scatter"),
            source: ShaderSource::Wgsl(include_str!("../shaders/scan_scatter.wgsl").into()),
        });

        let scan_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_scan_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: false }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let scatter_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_scan_scatter_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });

        let scan_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_scan_pipeline_layout"),
            bind_group_layouts: &[Some(&scan_layout)],
            immediate_size: 0,
        });
        let scatter_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_scan_scatter_pipeline_layout"),
            bind_group_layouts: &[Some(&scatter_layout)],
            immediate_size: 0,
        });

        let scan_block = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_scan_block_pipeline"),
            layout: Some(&scan_pipeline_layout),
            module: &scan_module,
            entry_point: Some("scan_block"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let add_offsets = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_scan_add_offsets_pipeline"),
            layout: Some(&scan_pipeline_layout),
            module: &scan_module,
            entry_point: Some("add_offsets"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let scatter = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_scan_scatter_pipeline"),
            layout: Some(&scatter_pipeline_layout),
            module: &scatter_module,
            entry_point: Some("scatter"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        GpuScan {
            scan_module,
            scatter_module,
            scan_layout,
            scatter_layout,
            scan_block,
            add_offsets,
            scatter,
        }
    }

    /// Computes the exclusive prefix sum of `values` on the `GPU` and returns it
    /// alongside the grand total.
    ///
    /// Matches [`cpu_exclusive_scan`](super::cpu::cpu_exclusive_scan)
    /// bit-for-bit. An empty input yields an empty vector and a zero total
    /// without touching the device.
    #[must_use]
    pub fn exclusive_scan(&self, ctx: &GpuContext, values: &[u32]) -> (Vec<u32>, u32) {
        let n = values.len();
        if n == 0 {
            return (Vec::new(), 0);
        }

        let device = ctx.device();
        let values_buf = buffer::storage_rw_init(device, "prism_scan_values", values);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_scan_encoder"),
        });
        let levels = self.record_scan(device, &mut encoder, values_buf, n);

        let bytes = size_of_val(values) as u64;
        let scan_stage = buffer::staging(device, "prism_scan_result_stage", bytes);
        let total_stage =
            buffer::staging(device, "prism_scan_total_stage", size_of::<u32>() as u64);
        buffer::copy(&mut encoder, &levels.buffers[0], &scan_stage, bytes);
        buffer::copy(
            &mut encoder,
            levels.buffers.last().expect("scan has a grand-total level"),
            &total_stage,
            size_of::<u32>() as u64,
        );
        ctx.queue().submit([encoder.finish()]);

        let scan = buffer::read_back::<u32>(ctx, &scan_stage);
        let total = buffer::read_back::<u32>(ctx, &total_stage)[0];
        drop(levels);
        (scan, total)
    }

    /// Gathers the entries of `data` whose corresponding `flags` entry is
    /// non-zero into a dense vector, preserving input order.
    ///
    /// Internally scans the flags to destination offsets and scatters on the
    /// `GPU`. Matches [`cpu_compact`](super::cpu::cpu_compact) bit-for-bit.
    ///
    /// # Panics
    ///
    /// Panics if `data` and `flags` do not have the same length.
    #[must_use]
    pub fn compact(&self, ctx: &GpuContext, data: &[u32], flags: &[u32]) -> Vec<u32> {
        assert!(
            data.len() == flags.len(),
            "data and flags must have equal length"
        );
        let n = data.len();
        if n == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        // One flags copy is scanned in place into destination offsets; a second,
        // read-only copy feeds the scatter's keep test.
        let offsets_buf = buffer::storage_rw_init(device, "prism_scan_compact_offsets", flags);
        let flags_buf = buffer::storage_read(device, "prism_scan_compact_flags", flags);
        let data_buf = buffer::storage_read(device, "prism_scan_compact_data", data);
        // Worst case keeps every element, so the dense output is sized to `n`.
        let out_bytes = size_of_val(data) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_scan_compact_out", out_bytes);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_scan_compact_encoder"),
        });
        let levels = self.record_scan(device, &mut encoder, offsets_buf, n);

        let scatter_params = buffer::uniform(
            device,
            "prism_scan_compact_params",
            &Params {
                n: u32::try_from(n).unwrap_or(u32::MAX),
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );
        let scatter_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_scan_compact_bind"),
            layout: &self.scatter_layout,
            entries: &[
                entry(0, &scatter_params),
                entry(1, &flags_buf),
                entry(2, &levels.buffers[0]),
                entry(3, &data_buf),
                entry(4, &out_buf),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_scan_scatter_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter);
            pass.set_bind_group(0, &scatter_bind, &[]);
            let groups = u32::try_from(n.div_ceil(WORKGROUP as usize)).unwrap_or(u32::MAX);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let out_stage = buffer::staging(device, "prism_scan_compact_out_stage", out_bytes);
        let total_stage = buffer::staging(
            device,
            "prism_scan_compact_total_stage",
            size_of::<u32>() as u64,
        );
        buffer::copy(&mut encoder, &out_buf, &out_stage, out_bytes);
        buffer::copy(
            &mut encoder,
            levels.buffers.last().expect("scan has a grand-total level"),
            &total_stage,
            size_of::<u32>() as u64,
        );
        ctx.queue().submit([encoder.finish()]);

        let total = buffer::read_back::<u32>(ctx, &total_stage)[0] as usize;
        let mut out = buffer::read_back::<u32>(ctx, &out_stage);
        out.truncate(total);
        drop(levels);
        out
    }

    /// Records a full multi-level exclusive scan of `values_buf` (`n` elements)
    /// into `encoder`, taking ownership of `values_buf` as the input level.
    ///
    /// Returns the level buffers and per-dispatch bindings so they outlive the
    /// submitted encoder; on completion `buffers[0]` holds the exclusive scan
    /// and the last buffer holds the length-one grand total.
    pub(crate) fn record_scan(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        values_buf: Buffer,
        n: usize,
    ) -> ScanLevels {
        let lens = level_lens(n);

        let mut buffers = Vec::with_capacity(lens.len());
        buffers.push(values_buf);
        for (level, &len) in lens.iter().enumerate().skip(1) {
            buffers.push(buffer::storage_rw_zeroed(
                device,
                &format!("prism_scan_level_{level}"),
                (len * size_of::<u32>()) as u64,
            ));
        }

        let mut params = Vec::new();
        let mut binds = Vec::new();

        // Down the pyramid: scan each level in place, emitting the next level of
        // per-block totals.
        for level in 0..lens.len() - 1 {
            self.record_level_pass(
                device,
                encoder,
                &self.scan_block,
                "prism_scan_block_pass",
                &buffers[level],
                &buffers[level + 1],
                lens[level],
                &mut params,
                &mut binds,
            );
        }

        // Up the pyramid: fold each coarser level's globally scanned offsets
        // back into the level below. The top level is already global, and the
        // grand-total slot needs no fold.
        for level in (0..lens.len().saturating_sub(2)).rev() {
            self.record_level_pass(
                device,
                encoder,
                &self.add_offsets,
                "prism_scan_add_offsets_pass",
                &buffers[level],
                &buffers[level + 1],
                lens[level],
                &mut params,
                &mut binds,
            );
        }

        ScanLevels {
            buffers,
            params,
            binds,
        }
    }

    /// Records one block-granular dispatch of `pipeline` over a level of `len`
    /// elements, wiring `values` and `block_sums`, and stashes the per-dispatch
    /// param buffer and bind group into `params` and `binds`.
    #[expect(
        clippy::too_many_arguments,
        reason = "records one dispatch from level buffers plus caller-owned keepalive stores"
    )]
    fn record_level_pass(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        pipeline: &ComputePipeline,
        label: &str,
        values: &Buffer,
        block_sums: &Buffer,
        len: usize,
        params: &mut Vec<Buffer>,
        binds: &mut Vec<BindGroup>,
    ) {
        let param_buf = buffer::uniform(
            device,
            "prism_scan_level_params",
            &Params {
                n: u32::try_from(len).unwrap_or(u32::MAX),
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );
        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_scan_level_bind"),
            layout: &self.scan_layout,
            entries: &[entry(0, &param_buf), entry(1, values), entry(2, block_sums)],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let groups = u32::try_from(len.div_ceil(BLOCK as usize)).unwrap_or(u32::MAX);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        params.push(param_buf);
        binds.push(bind);
    }
}

/// Builds the element counts of each scan level, coarsest last.
///
/// `lens[0]` is the input length `n`; each subsequent level is the per-block
/// total count `ceil(prev / BLOCK)`, appended until a level fits in one block,
/// after which one final length-one grand-total level is appended. For example
/// `n = 1_000_000` yields `[1_000_000, 1954, 4, 1]` and `n = 100` yields
/// `[100, 1]`.
fn level_lens(n: usize) -> Vec<usize> {
    let block = BLOCK as usize;
    let mut lens = vec![n];
    while *lens.last().expect("level list is never empty") > block {
        let last = *lens.last().expect("level list is never empty");
        lens.push(last.div_ceil(block));
    }
    let last = *lens.last().expect("level list is never empty");
    lens.push(last.div_ceil(block));
    lens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_block_input_has_one_scan_level_and_a_total() {
        assert_eq!(level_lens(100), vec![100, 1]);
    }

    #[test]
    fn just_over_one_block_adds_a_second_level() {
        assert_eq!(level_lens(BLOCK as usize + 1), vec![513, 2, 1]);
    }

    #[test]
    fn million_element_input_forms_a_four_level_pyramid() {
        assert_eq!(level_lens(1_000_000), vec![1_000_000, 1954, 4, 1]);
    }

    #[test]
    fn exactly_one_block_still_appends_a_total_level() {
        assert_eq!(level_lens(BLOCK as usize), vec![512, 1]);
    }
}
