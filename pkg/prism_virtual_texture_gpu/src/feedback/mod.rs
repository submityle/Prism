//! Host orchestration of the sampler-feedback min-mip grid decode kernel.
//!
//! [`GpuFeedbackDecode`] compiles `shaders/feedback_decode.wgsl` once and turns
//! a dense min-mip feedback grid into per-page streaming demand, mirroring the
//! device-free CPU golden
//! [`decode_feedback`](prism_render_architecture::texture_streaming::decode_feedback).
//!
//! # Split of work
//!
//! The hot, embarrassingly parallel part — mapping each base-level grid cell to
//! the collapsed [`TexturePageKey`] compare words and clamped desired mip — runs
//! on the `GPU` ([`GpuFeedbackDecode::map_cells`]). The inherently serial part —
//! deduplicating colliding cells into one demand in ascending key order and
//! stamping each with its current `resident_mip` — stays on the host inside
//! [`GpuFeedbackDecode::decode`], because a `BTreeMap` fold is ordered and serial
//! and residency is a host callback. The host folds the identical first-writer-
//! wins map over the row-major cell order the golden uses, so `decode` returns a
//! [`PageDemand`] list equal to `decode_feedback` for the same inputs.
//!
//! # Exact parity
//!
//! The per-cell map is **integer-only** — clamp, shift, and the same compare-word
//! packing the golden [`GpuPageTable::compare_words`] uses — so the device output
//! equals the golden arithmetic bit-for-bit and the parity tests assert exact
//! equality rather than a tolerance.
//!
//! Provenance: DX12 Sampler-Feedback-style min-mip readback → per-page residency
//! demand. Classical, data-oblivious integer work; no neural, learned, or
//! data-driven components. No Unreal Engine source or derived code.

use alloc::collections::BTreeMap;

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::texture_streaming::{
    decode_feedback, FeedbackTextureDesc, GpuPageTable, PageDemand, TexturePageKey, NOT_REQUESTED,
};

use crate::buffer;
use crate::context::GpuContext;

/// Desired-mip sentinel written by the kernel for a not-requested cell, matching
/// the shader's `REQ_NONE`. Distinguishable from any real mip, which is a `u8`.
pub const REQ_NONE: u32 = u32::MAX;

/// Words emitted per cell by the map kernel: three compare words plus the mip.
const OUT_WORDS: usize = 4;

/// Uniform block shared with `Params` in `feedback_decode.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    texture: u32,
    layer: u32,
    base_mip: u32,
    max_mip: u32,
    pages_x: u32,
    pages_y: u32,
    cell_count: u32,
    pad0: u32,
}

/// One decoded grid cell as the kernel emits it: the three compare words of the
/// collapsed page key and the clamped desired mip, or [`REQ_NONE`] in `desired`
/// when the cell was not requested.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CellMap {
    /// Compare word 0 (`texture`).
    pub w0: u32,
    /// Compare word 1 (`(mip << 24) | (layer << 8)`).
    pub w1: u32,
    /// Compare word 2 (`(x << 16) | y`).
    pub w2: u32,
    /// Clamped desired mip, or [`REQ_NONE`] for a not-requested cell.
    pub desired: u32,
}

impl CellMap {
    /// Whether the source cell was sampled this frame (not [`REQ_NONE`]).
    #[must_use]
    pub fn is_requested(&self) -> bool {
        self.desired != REQ_NONE
    }
}

/// Compiled feedback-decode map pipeline and its bind-group layout.
pub struct GpuFeedbackDecode {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuFeedbackDecode {
    /// Compiles the feedback-decode map kernel on `ctx`'s device.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFeedbackDecode {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_vt_feedback_decode"),
            source: ShaderSource::Wgsl(include_str!("../shaders/feedback_decode.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_vt_feedback_layout"),
            entries: &[
                buffer_layout(0, BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
                buffer_layout(1, BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
                buffer_layout(2, BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_vt_feedback_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_vt_feedback_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFeedbackDecode { pipeline, layout }
    }

    /// Runs the per-cell map stage on the device.
    ///
    /// Returns one [`CellMap`] per grid cell in row-major order (cell `i` is the
    /// base page `(i % pages_x, i / pages_x)`). A grid whose length does not
    /// match `desc.grid_len()`, a zero `mip_count`, or an empty grid yields an
    /// empty vector, mirroring the golden `decode_feedback` early-out.
    #[must_use]
    pub fn map_cells(&self, ctx: &GpuContext, desc: &FeedbackTextureDesc, grid: &[u8]) -> Vec<CellMap> {
        if grid.len() != desc.grid_len() || desc.mip_count == 0 || grid.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        // Expand each min-mip byte to its own word so the shader indexes cells
        // directly without unpacking; `storage_read` is happy with a non-empty
        // slice (guaranteed, since an empty grid returned above).
        let cell_words: Vec<u32> = grid.iter().map(|&b| u32::from(b)).collect();
        let cell_count = cell_words.len() as u32;

        let params = buffer::uniform(
            device,
            "prism_vt_feedback_params",
            &Params {
                texture: desc.texture,
                layer: u32::from(desc.layer),
                base_mip: u32::from(desc.base_mip),
                max_mip: u32::from(desc.max_mip()),
                pages_x: u32::from(desc.pages_x),
                pages_y: u32::from(desc.pages_y),
                cell_count,
                pad0: 0,
            },
        );
        let grid_buf = buffer::storage_read(device, "prism_vt_feedback_grid", &cell_words);
        let out_bytes = u64::from(cell_count) * (OUT_WORDS as u64) * size_of::<u32>() as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_vt_feedback_out", out_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_vt_feedback_bind"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: grid_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
            ],
        });

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_vt_feedback_encoder"),
        });
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_vt_feedback_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups(cell_count), 1, 1);
        }

        let stage = buffer::staging(device, "prism_vt_feedback_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        let words = buffer::read_back::<u32>(ctx, &stage);
        words
            .chunks_exact(OUT_WORDS)
            .take(cell_count as usize)
            .map(|c| CellMap { w0: c[0], w1: c[1], w2: c[2], desired: c[3] })
            .collect()
    }

    /// Decodes the feedback grid into deduplicated per-page demand on the device.
    ///
    /// Runs [`map_cells`](Self::map_cells) on the `GPU`, then folds the identical
    /// first-writer-wins `BTreeMap` the golden `decode_feedback` uses over the
    /// row-major cell order, stamping each surviving demand with `resident_mip`
    /// and `frame`. The result equals `decode_feedback(desc, grid, resident_mip,
    /// frame)` for the same inputs.
    #[must_use]
    pub fn decode(
        &self,
        ctx: &GpuContext,
        desc: &FeedbackTextureDesc,
        grid: &[u8],
        mut resident_mip: impl FnMut(TexturePageKey) -> Option<u8>,
        frame: u64,
    ) -> Vec<PageDemand> {
        let cells = self.map_cells(ctx, desc, grid);
        let mut merged: BTreeMap<TexturePageKey, PageDemand> = BTreeMap::new();
        for cell in cells {
            if cell.desired == REQ_NONE {
                continue;
            }
            // Reconstruct the key from the kernel's compare words — the inverse
            // of the packing the shader performed, shared with the golden.
            let key = GpuPageTable::unpack_key(cell.w0, cell.w1, cell.w2);
            let demand = PageDemand {
                key,
                semantic: desc.semantic,
                desired_mip: cell.desired as u8,
                resident_mip: resident_mip(key),
                screen_importance: desc.screen_importance,
                byte_cost: desc.page_byte_cost,
                frame,
            };
            merged.entry(key).or_insert(demand);
        }
        merged.into_values().collect()
    }
}

/// Reference per-cell map replicated on the host, independent of the kernel.
///
/// Mirrors the shader arithmetic with ordinary integer ops so parity tests can
/// bind the device output cell-by-cell against a from-scratch oracle rather than
/// trusting the kernel to describe itself. `NOT_REQUESTED` cells map to a
/// [`REQ_NONE`] `desired`.
#[must_use]
pub fn reference_map(desc: &FeedbackTextureDesc, grid: &[u8]) -> Vec<CellMap> {
    if grid.len() != desc.grid_len() || desc.mip_count == 0 || grid.is_empty() {
        return Vec::new();
    }
    let base = desc.base_mip;
    let max_mip = desc.max_mip();
    (0..grid.len())
        .map(|i| {
            let cell = grid[i];
            if cell == NOT_REQUESTED {
                return CellMap { w0: 0, w1: 0, w2: 0, desired: REQ_NONE };
            }
            let desired = cell.clamp(base, max_mip);
            let shift = desired - base;
            let x = (i % desc.pages_x as usize) as u16;
            let y = (i / desc.pages_x as usize) as u16;
            let key = TexturePageKey {
                texture: desc.texture,
                mip: desired,
                layer: desc.layer,
                x: x >> shift,
                y: y >> shift,
            };
            let [w0, w1, w2] = GpuPageTable::compare_words(key);
            CellMap { w0, w1, w2, desired: u32::from(desired) }
        })
        .collect()
}

/// Golden demand list for the same inputs, re-exported for parity assertions.
#[must_use]
pub fn golden_decode(
    desc: &FeedbackTextureDesc,
    grid: &[u8],
    resident_mip: impl FnMut(TexturePageKey) -> Option<u8>,
    frame: u64,
) -> Vec<PageDemand> {
    decode_feedback(desc, grid, resident_mip, frame)
}

/// One storage/uniform bind-group-layout entry visible to the compute stage.
fn buffer_layout(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

/// Workgroups needed to cover `n` invocations at `@workgroup_size(256)`.
fn groups(n: u32) -> u32 {
    n.div_ceil(256).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::texture_streaming::TextureSemantic;

    #[expect(
        clippy::print_stderr,
        reason = "surface a clear skip message when no GPU adapter is present"
    )]
    fn with_gpu(body: impl FnOnce(&GpuContext)) {
        match GpuContext::try_headless() {
            Some(ctx) => body(&ctx),
            None => eprintln!("skipping GPU feedback-decode test: no usable adapter"),
        }
    }

    fn desc() -> FeedbackTextureDesc {
        FeedbackTextureDesc {
            texture: 7,
            layer: 0,
            semantic: TextureSemantic::Color,
            base_mip: 0,
            mip_count: 4,
            pages_x: 4,
            pages_y: 4,
            page_byte_cost: 65_536,
            screen_importance: 500,
        }
    }

    fn cell(x: usize, y: usize) -> usize {
        y * 4 + x
    }

    #[test]
    fn groups_cover_every_cell() {
        assert_eq!(groups(0), 1);
        assert_eq!(groups(1), 1);
        assert_eq!(groups(256), 1);
        assert_eq!(groups(257), 2);
    }

    #[test]
    fn req_none_matches_u32_max() {
        assert_eq!(REQ_NONE, u32::MAX);
    }

    #[test]
    fn map_cells_matches_reference_oracle() {
        with_gpu(|ctx| {
            let d = desc();
            let mut grid = [NOT_REQUESTED; 16];
            grid[cell(2, 1)] = 0; // finest request at its own cell
            grid[cell(0, 0)] = 1; // 2x2 collapse at mip 1
            grid[cell(1, 0)] = 1;
            grid[cell(0, 1)] = 1;
            grid[cell(1, 1)] = 1;
            grid[cell(3, 3)] = 200; // clamp up to max mip 3

            let kernel = GpuFeedbackDecode::new(ctx);
            let got = kernel.map_cells(ctx, &d, &grid);
            let want = reference_map(&d, &grid);
            assert_eq!(got.len(), 16);
            assert_eq!(got, want, "device per-cell map must equal the host oracle");

            // Anti-vacuous guards: the fixture must exercise every branch so a
            // kernel that only ever emits REQ_NONE cannot pass.
            assert!(got.iter().any(|c| !c.is_requested()), "has a not-requested cell");
            assert!(got.iter().any(CellMap::is_requested), "has a requested cell");
            assert!(
                got[cell(3, 3)].desired == 3,
                "coarse request clamps to max streamable mip"
            );
            // The 2x2 at mip 1 must collapse to page (0, 0): w2 == 0.
            assert_eq!(got[cell(1, 1)].w2, 0, "mip-1 cell collapses to page (0,0)");
            assert_eq!(got[cell(1, 1)].desired, 1);
        });
    }

    #[test]
    fn decode_matches_golden_decode_feedback() {
        with_gpu(|ctx| {
            let d = desc();
            let mut grid = [NOT_REQUESTED; 16];
            grid[cell(2, 1)] = 0;
            grid[cell(0, 0)] = 1;
            grid[cell(1, 0)] = 1;
            grid[cell(0, 1)] = 1;
            grid[cell(1, 1)] = 1;
            grid[cell(3, 3)] = 200;
            grid[cell(3, 0)] = 2;

            let kernel = GpuFeedbackDecode::new(ctx);
            // A non-trivial residency closure to exercise the host fold.
            let resident = |k: TexturePageKey| if k.mip == 0 { Some(2) } else { None };
            let got = kernel.decode(ctx, &d, &grid, resident, 9);
            let want = golden_decode(&d, &grid, resident, 9);
            assert!(!got.is_empty(), "fixture produces demand");
            assert_eq!(got, want, "device decode must equal the CPU golden exactly");
        });
    }

    #[test]
    fn decode_dedups_collapsed_cells() {
        with_gpu(|ctx| {
            let d = desc();
            let mut grid = [NOT_REQUESTED; 16];
            // Whole top-left 2x2 at mip 1 collapses to one page.
            grid[cell(0, 0)] = 1;
            grid[cell(1, 0)] = 1;
            grid[cell(0, 1)] = 1;
            grid[cell(1, 1)] = 1;
            let kernel = GpuFeedbackDecode::new(ctx);
            let got = kernel.decode(ctx, &d, &grid, |_| None, 1);
            assert_eq!(got.len(), 1, "four collapsing cells dedup to one demand");
            assert_eq!(got[0].key.mip, 1);
            assert_eq!(got[0].key.x, 0);
            assert_eq!(got[0].key.y, 0);
            assert_eq!(got, golden_decode(&d, &grid, |_| None, 1));
        });
    }

    #[test]
    fn wrong_length_grid_yields_no_cells() {
        with_gpu(|ctx| {
            let d = desc();
            let kernel = GpuFeedbackDecode::new(ctx);
            assert!(kernel.map_cells(ctx, &d, &[0u8; 3]).is_empty());
            assert!(kernel.decode(ctx, &d, &[0u8; 3], |_| None, 1).is_empty());
        });
    }
}
