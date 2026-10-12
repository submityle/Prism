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

use prism_render_architecture::texture_streaming::{
    GpuPageTable, PageResolution, TexturePageKey, PAGE_TABLE_ENTRY_WORDS,
};

use crate::buffer;
use crate::context::GpuContext;

/// Sentinel slot written for a page coordinate with no resident entry, matching
/// the shader's `MISS` constant. Equals `None` from the golden `lookup`.
pub const MISS: u32 = u32::MAX;

/// Resolved virtual page plus the original page to feed back for refinement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuPageResolution {
    /// Nearest resident page, including mip-tail fallback; `None` means that
    /// the material must use its ordinary fallback texture for this sample.
    pub resolution: Option<PageResolution>,
    /// Original request when it missed or used a coarser page. Feed this key to
    /// the streaming-demand accumulator; exact hits produce no feedback.
    pub feedback: Option<TexturePageKey>,
}

/// Uniform block shared with `Params` in `page_lookup.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    entry_count: u32,
    query_count: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResolveParams {
    entry_count: u32,
    query_count: u32,
    coarsest_mip: u32,
    pad0: u32,
}

/// Compiled page-lookup pipeline and its bind-group layout.
pub struct GpuPageLookup {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
    resolve_pipeline: ComputePipeline,
    resolve_layout: BindGroupLayout,
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
                buffer_layout(
                    0,
                    BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    1,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    2,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    3,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
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
        let resolve_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_vt_page_resolve"),
            source: ShaderSource::Wgsl(include_str!("../shaders/page_resolve.wgsl").into()),
        });
        let resolve_layout = lookup_layout(device, "prism_vt_resolve_layout");
        let resolve_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_vt_resolve_pipeline_layout"),
            bind_group_layouts: &[Some(&resolve_layout)],
            immediate_size: 0,
        });
        let resolve_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_vt_resolve_pipeline"),
            layout: Some(&resolve_pipeline_layout),
            module: &resolve_module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPageLookup {
            pipeline,
            layout,
            resolve_pipeline,
            resolve_layout,
        }
    }

    /// Resolves each key in `keys` to its physical slot on the device.
    ///
    /// The returned vector is parallel to `keys`: entry `i` is the slot the
    /// golden `GpuPageTable::lookup(keys[i])` would return, or [`MISS`] for a
    /// page coordinate with no resident entry.
    #[must_use]
    pub fn lookup(
        &self,
        ctx: &GpuContext,
        table: &GpuPageTable,
        keys: &[TexturePageKey],
    ) -> Vec<u32> {
        assert_valid_page_table_words(table.words());
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

        let entry_count = u32::try_from(table.len())
            .expect("virtual-texture page table exceeds the GPU u32 entry-count ABI");
        let query_count = u32::try_from(keys.len())
            .expect("virtual-texture query batch exceeds the GPU u32 query-count ABI");

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

    /// Resolves one requested page on the GPU, probing progressively coarser
    /// covering pages through `coarsest_mip` and returning the first resident
    /// physical slot. Missing results remain `None`; callers must use their
    /// ordinary material fallback and submit the original key as feedback.
    ///
    /// The binary-search dispatch remains the authoritative exact lookup. The
    /// host only expands the deterministic mip-cover candidates and selects
    /// the first hit, matching `GpuPageTable::resolve` exactly.
    #[must_use]
    pub fn resolve(
        &self,
        ctx: &GpuContext,
        table: &GpuPageTable,
        key: TexturePageKey,
        coarsest_mip: u8,
    ) -> Option<PageResolution> {
        self.resolve_with_feedback(ctx, table, key, coarsest_mip)
            .resolution
    }

    /// GPU page lookup with explicit quality-refinement feedback.
    ///
    /// This is the material-facing closed loop: lookup the requested page,
    /// fall back to the nearest resident mip-tail page, and retain the original
    /// virtual key whenever the sample was not exact. A full miss retains the
    /// same request, so callers never lose demand when falling back to a
    /// non-virtual texture.
    #[must_use]
    pub fn resolve_with_feedback(
        &self,
        ctx: &GpuContext,
        table: &GpuPageTable,
        key: TexturePageKey,
        coarsest_mip: u8,
    ) -> GpuPageResolution {
        self.resolve_with_feedback_batch(ctx, table, &[key], coarsest_mip)
            .into_iter()
            .next()
            .expect("single-page resolve must return one result")
    }

    /// Resolves a batch of requested pages in one GPU dispatch and readback.
    ///
    /// The result is parallel to `keys`. Reusing one page-table upload and one
    /// command submission is important for material workloads, where a frame
    /// commonly resolves many page samples. An empty batch returns immediately
    /// without touching the device.
    #[must_use]
    pub fn resolve_with_feedback_batch(
        &self,
        ctx: &GpuContext,
        table: &GpuPageTable,
        keys: &[TexturePageKey],
        coarsest_mip: u8,
    ) -> Vec<GpuPageResolution> {
        assert_valid_page_table_words(table.words());
        if keys.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let entry_count = u32::try_from(table.len())
            .expect("virtual-texture page table exceeds the GPU u32 entry-count ABI");
        let query_count = u32::try_from(keys.len())
            .expect("virtual-texture resolve batch exceeds the GPU u32 query-count ABI");
        let query_word_count = keys
            .len()
            .checked_mul(3)
            .expect("virtual-texture resolve query word count overflowed usize");
        let mut query_words = Vec::with_capacity(query_word_count);
        for &key in keys {
            query_words.extend_from_slice(&GpuPageTable::compare_words(key));
        }
        let params = buffer::uniform(
            device,
            "prism_vt_resolve_params",
            &ResolveParams {
                entry_count,
                query_count,
                coarsest_mip: u32::from(coarsest_mip),
                pad0: 0,
            },
        );
        let table_words = if table.words().is_empty() {
            &[0u32][..]
        } else {
            table.words()
        };
        let table_buf = buffer::storage_read(device, "prism_vt_resolve_table", table_words);
        let query_buf = buffer::storage_read(device, "prism_vt_resolve_query", &query_words);
        let out_bytes = u64::from(query_count)
            .checked_mul(3 * size_of::<u32>() as u64)
            .expect("virtual-texture resolve output byte count overflowed u64");
        let out_buf = buffer::storage_rw_zeroed(device, "prism_vt_resolve_result", out_bytes);
        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_vt_resolve_bind"),
            layout: &self.resolve_layout,
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
            label: Some("prism_vt_resolve_encoder"),
        });
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_vt_resolve_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resolve_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups(query_count), 1, 1);
        }
        let stage = buffer::staging(device, "prism_vt_resolve_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);
        let words = buffer::read_back::<u32>(ctx, &stage);
        keys.iter()
            .enumerate()
            .map(|(index, &key)| {
                let base = index * 3;
                let slot = words.get(base).copied().unwrap_or(MISS);
                let resolved_mip = words.get(base + 1).copied().unwrap_or(MISS);
                let resolution = (slot != MISS && resolved_mip != MISS).then(|| {
                    let delta = resolved_mip.saturating_sub(u32::from(key.mip));
                    let shift = delta.min(16);
                    PageResolution {
                        key: TexturePageKey {
                            texture: key.texture,
                            mip: u8::try_from(resolved_mip).unwrap_or(u8::MAX),
                            layer: key.layer,
                            x: key.x.checked_shr(shift).unwrap_or(0),
                            y: key.y.checked_shr(shift).unwrap_or(0),
                        },
                        slot,
                        mip_bias: u8::try_from(delta).unwrap_or(u8::MAX),
                    }
                });
                GpuPageResolution {
                    feedback: (words.get(base + 2).copied().unwrap_or(1) != 0).then_some(key),
                    resolution,
                }
            })
            .collect()
    }
}

/// Enforces the page-table ABI before its words cross into a GPU shader.
/// `MISS` is the lookup kernel's no-resident-page result, so it cannot also be
/// a physical slot. Keep this check at the consumer boundary even though the
/// CPU golden table already rejects the value in `from_bindings`.
fn assert_valid_page_table_words(words: &[u32]) {
    assert_eq!(
        words.len() % PAGE_TABLE_ENTRY_WORDS,
        0,
        "GPU page table must contain complete entries"
    );
    for entry in words.chunks_exact(PAGE_TABLE_ENTRY_WORDS) {
        assert_ne!(
            entry[PAGE_TABLE_ENTRY_WORDS - 1],
            MISS,
            "GPU page-table physical slots must not use the reserved miss sentinel"
        );
    }
}

fn lookup_layout(device: &wgpu::Device, label: &str) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            buffer_layout(
                0,
                BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ),
            buffer_layout(
                1,
                BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ),
            buffer_layout(
                2,
                BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ),
            buffer_layout(
                3,
                BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ),
        ],
    })
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

    #[test]
    fn page_table_abi_rejects_miss_sentinel_as_a_physical_slot() {
        let result = std::panic::catch_unwind(|| {
            assert_valid_page_table_words(&[1, 0, 0, MISS]);
        });
        assert!(result.is_err());
    }

    #[test]
    fn page_table_abi_accepts_low_and_high_non_sentinel_slots() {
        assert_valid_page_table_words(&[1, 0, 0, 0, 2, 0, 0, MISS - 1]);
    }

    #[test]
    fn page_table_abi_rejects_partial_entries() {
        let result = std::panic::catch_unwind(|| {
            assert_valid_page_table_words(&[1, 0, 0]);
        });
        assert!(result.is_err());
    }

    #[test]
    fn cpu_golden_constructor_rejects_reserved_physical_slot() {
        let key = TexturePageKey {
            texture: 1,
            mip: 0,
            layer: 0,
            x: 0,
            y: 0,
        };
        let result = std::panic::catch_unwind(|| {
            let _ = GpuPageTable::from_bindings(&[(key, MISS)]);
        });
        assert!(result.is_err());
    }
}
