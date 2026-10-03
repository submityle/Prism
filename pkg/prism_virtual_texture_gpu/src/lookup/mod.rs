//! Host orchestration of the page-table indirection-lookup kernel.
//!
//! [`GpuPageLookup`] compiles `shaders/page_lookup.wgsl` once and resolves a
//! batch of sampled page coordinates against a flat page table produced by the
//! CPU golden
//! [`GpuPageTable`](prism_render_architecture::texture_streaming::GpuPageTable).
//! Each query is pre-reduced on the host to its three compare words via the
//! golden [`GpuPageTable::compare_words`], so the shader performs the identical
//! unsigned lexicographic binary search as
//! [`GpuPageTable::lookup`](prism_render_architecture::texture_streaming::GpuPageTable::lookup)
//! and returns the identical physical slot. A miss is reported as [`MISS`].
//!
//! The work is integer-only, so device output equals the golden output
//! bit-for-bit; parity tests compare with exact equality, not tolerance.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine source or derived
//! code.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::texture_streaming::{GpuPageTable, TexturePageKey};

use crate::buffer;
use crate::context::GpuContext;

/// Sentinel slot written for a page coordinate with no resident entry, matching
/// the shader's `MISS` constant. Equals `None` from the golden `lookup`.
pub const MISS: u32 = u32::MAX;

/// Uniform block shared with `Params` in `page_lookup.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    entry_count: u32,
    query_count: u32,
    pad0: u32,
    pad1: u32,
}

/// Compiled page-lookup pipeline and its bind-group layout.
pub struct GpuPageLookup {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuPageLookup {
    /// Compiles the lookup kernel on `ctx`'s device.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPageLookup {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_vt_page_lookup"),
            source: ShaderSource::Wgsl(include_str!("../shaders/page_lookup.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_vt_lookup_layout"),
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
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
                buffer_layout(3, BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_vt_lookup_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_vt_lookup_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPageLookup { pipeline, layout }
    }

    /// Resolves each key in `keys` to its physical slot on the device.
    ///
    /// The returned vector is parallel to `keys`: entry `i` is the slot the
    /// golden `GpuPageTable::lookup(keys[i])` would return, or [`MISS`] for a
    /// page coordinate with no resident entry.
    #[must_use]
    pub fn lookup(&self, ctx: &GpuContext, table: &GpuPageTable, keys: &[TexturePageKey]) -> Vec<u32> {
        let device = ctx.device();

        // Pre-reduce each query to its three compare words via the golden
        // packer, so the shader and golden search identical keys.
        let mut query_words: Vec<u32> = Vec::with_capacity(keys.len() * 3);
        for &key in keys {
            let [w0, w1, w2] = GpuPageTable::compare_words(key);
            query_words.push(w0);
            query_words.push(w1);
            query_words.push(w2);
        }

        let entry_count = table.len() as u32;
        let query_count = keys.len() as u32;

        // `storage_read` rejects empty slices; pad degenerate inputs to one word
        // so a device with zero resident entries or zero queries still binds.
        let table_words: &[u32] = if table.words().is_empty() {
            &[0]
        } else {
            table.words()
        };
        let query_src: &[u32] = if query_words.is_empty() {
            &[0]
        } else {
            &query_words
        };

        let params = buffer::uniform(
            device,
            "prism_vt_lookup_params",
            &Params {
                entry_count,
                query_count,
                pad0: 0,
                pad1: 0,
            },
        );
        let table_buf = buffer::storage_read(device, "prism_vt_table", table_words);
        let query_buf = buffer::storage_read(device, "prism_vt_queries", query_src);
        let out_bytes = u64::from(query_count.max(1)) * size_of::<u32>() as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_vt_slots", out_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_vt_lookup_bind"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: table_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: query_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_vt_lookup_encoder"),
        });
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_vt_lookup_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups(query_count), 1, 1);
        }

        let stage = buffer::staging(device, "prism_vt_lookup_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        let mut slots = buffer::read_back::<u32>(ctx, &stage);
        slots.truncate(keys.len());
        slots
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

    #[test]
    fn groups_cover_every_query() {
        assert_eq!(groups(0), 1);
        assert_eq!(groups(1), 1);
        assert_eq!(groups(256), 1);
        assert_eq!(groups(257), 2);
    }

    #[test]
    fn miss_matches_u32_max() {
        assert_eq!(MISS, u32::MAX);
    }
}
