//! Host orchestration of the `GDeflate` word-transpose de-interleave kernel.
//!
//! [`GpuDeinterleave`] compiles `shaders/deinterleave.wgsl` once and reverses
//! the 32-lane warp interleave of a tile payload produced by the CPU golden
//! [`gdeflate_compress`](prism_render_architecture::compression::gdeflate_compress).
//! The device recovers the linear `DEFLATE` word stream; the serial Huffman
//! [`inflate`](prism_render_architecture::compression::inflate) stays on the
//! host (it is inherently variable-length and not a good `GPU` fit).
//!
//! The transpose is a pure permutation of whole 32-bit words, so the device
//! output equals the golden de-interleave bit-for-bit; [`reference_deinterleave`]
//! is an independent host oracle replicating the identical arithmetic, and the
//! parity tests compare with exact equality, not tolerance.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or NVIDIA `GDeflate`
//! source or derived code.

use alloc::vec;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

/// Number of warp lanes the bitstream is interleaved across (matches the golden
/// container's `LANES`).
pub const LANES: u32 = 32;
/// Width of one interleave word in bytes (matches the golden container's
/// `WORD`).
pub const WORD: usize = 4;
/// Bytes consumed by one interleave round across all lanes (`LANES * WORD`,
/// matches the golden container's `GROUP`).
pub const GROUP: usize = LANES as usize * WORD;

/// Uniform block shared with `Params` in `deinterleave.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    word_count: u32,
    rounds: u32,
    pad0: u32,
    pad1: u32,
}

/// Reads a lane-major interleaved payload as little-endian 32-bit words.
fn to_words_le(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(WORD)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Writes recovered words back out as little-endian bytes.
fn from_words_le(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * WORD);
    for &word in words {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

/// Independent host oracle for the de-interleave transpose.
///
/// Replicates the golden `compression::gdeflate` word transpose exactly:
/// linear word `r * LANES + lane` is gathered from the lane-major source word
/// `lane * rounds + r`. `interleaved.len()` is expected to be a whole number of
/// lane groups ([`GROUP`] bytes); any trailing partial group is left as zero in
/// the output. Returns a buffer the same length as `interleaved`.
#[must_use]
pub fn reference_deinterleave(interleaved: &[u8]) -> Vec<u8> {
    let padded_len = interleaved.len();
    if padded_len < WORD {
        return Vec::new();
    }
    let words = padded_len / WORD;
    let rounds = words / LANES as usize;
    let mut out = vec![0u8; padded_len];
    for lane in 0..LANES as usize {
        for r in 0..rounds {
            let logical = r * LANES as usize + lane;
            let stored = lane * rounds + r;
            out[logical * WORD..logical * WORD + WORD]
                .copy_from_slice(&interleaved[stored * WORD..stored * WORD + WORD]);
        }
    }
    out
}

/// Compiled de-interleave pipeline and its bind-group layout.
pub struct GpuDeinterleave {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuDeinterleave {
    /// Compiles the de-interleave kernel on `ctx`'s device.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDeinterleave {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_gdeflate_deinterleave"),
            source: ShaderSource::Wgsl(include_str!("../shaders/deinterleave.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_gdeflate_deinterleave_layout"),
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
            label: Some("prism_gdeflate_deinterleave_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_gdeflate_deinterleave_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDeinterleave { pipeline, layout }
    }

    /// Reverses the warp interleave of `interleaved` on the device.
    ///
    /// `interleaved` is a lane-major tile payload — a whole number of lane
    /// groups ([`GROUP`] bytes). The returned buffer is the recovered linear
    /// `DEFLATE` word stream, the same length as `interleaved`; the caller
    /// truncates it to the tile's `compressed_size` and inflates it with the
    /// golden core exactly as `gdeflate_decompress` does. An empty input yields
    /// an empty output.
    #[must_use]
    pub fn deinterleave(&self, ctx: &GpuContext, interleaved: &[u8]) -> Vec<u8> {
        if interleaved.len() < WORD {
            return Vec::new();
        }
        let device = ctx.device();

        let src_words = to_words_le(interleaved);
        let words = src_words.len();
        let rounds = words / LANES as usize;
        // Only the whole-lane-group prefix participates in the permutation; any
        // trailing partial group stays zero, matching `reference_deinterleave`.
        let active = rounds * LANES as usize;

        let params = buffer::uniform(
            device,
            "prism_gdeflate_params",
            &Params {
                word_count: active as u32,
                rounds: rounds as u32,
                pad0: 0,
                pad1: 0,
            },
        );
        let src_buf = buffer::storage_read(device, "prism_gdeflate_src", &src_words);
        let out_bytes = (words.max(1) * WORD) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_gdeflate_dst", out_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_gdeflate_deinterleave_bind"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: src_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_gdeflate_deinterleave_encoder"),
        });
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_gdeflate_deinterleave_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups(active as u32), 1, 1);
        }

        let stage = buffer::staging(device, "prism_gdeflate_deinterleave_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        let mut out_words = buffer::read_back::<u32>(ctx, &stage);
        out_words.truncate(words);
        from_words_le(&out_words)
    }
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

    /// Interleaves a linear word stream into lane-major order, replicating the
    /// golden forward transpose so the de-interleave has a known inverse input.
    fn interleave(linear: &[u8]) -> Vec<u8> {
        let padded_len = linear.len().div_ceil(GROUP) * GROUP;
        if padded_len == 0 {
            return Vec::new();
        }
        let mut padded = vec![0u8; padded_len];
        padded[..linear.len()].copy_from_slice(linear);
        let words = padded_len / WORD;
        let rounds = words / LANES as usize;
        let mut out = vec![0u8; padded_len];
        for lane in 0..LANES as usize {
            for r in 0..rounds {
                let logical = r * LANES as usize + lane;
                let stored = lane * rounds + r;
                out[stored * WORD..stored * WORD + WORD]
                    .copy_from_slice(&padded[logical * WORD..logical * WORD + WORD]);
            }
        }
        out
    }

    fn lcg_bytes(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed;
        let mut data = vec![0u8; len];
        for byte in &mut data {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *byte = (state >> 24) as u8;
        }
        data
    }

    #[test]
    fn groups_cover_every_word() {
        assert_eq!(groups(0), 1);
        assert_eq!(groups(1), 1);
        assert_eq!(groups(256), 1);
        assert_eq!(groups(257), 2);
    }

    #[test]
    fn constants_match_golden_container() {
        assert_eq!(LANES, 32);
        assert_eq!(WORD, 4);
        assert_eq!(GROUP, 128);
    }

    #[test]
    fn reference_inverts_interleave() {
        // A multi-round, lane-group-aligned linear stream round-trips through
        // the forward interleave and the reference de-interleave exactly.
        let linear = lcg_bytes(GROUP * 7, 0x51ed_c0de);
        let interleaved = interleave(&linear);
        assert_eq!(interleaved.len(), linear.len());
        let recovered = reference_deinterleave(&interleaved);
        assert_eq!(recovered, linear);
    }

    #[test]
    fn reference_empty_is_empty() {
        assert!(reference_deinterleave(&[]).is_empty());
        assert!(reference_deinterleave(&[0, 0, 0]).is_empty());
    }
}
