//! `wgpu` compute twin of the physical page-table resolve
//! ([`PagePool::slot_of`](prism_render_architecture::paging::PagePool::slot_of)).
//!
//! Virtualized geometry keeps its resident cluster pages in a bounded physical
//! pool; the CPU golden [`PagePool`](prism_render_architecture::paging::PagePool)
//! owns the slot allocation and exports the resident map as a key-ordered
//! `(key, slot)` list. A GPU-driven pipeline needs the *inverse* lookup on the
//! device: each cluster resolves its [`GeometryPageKey`] to a physical slot so
//! it can address the real page buffer. [`GpuPageTable`] is that resolver - a
//! parallel binary search over the sorted entry table, one thread per query -
//! validated bit-for-bit against the reference.
//!
//! # Portability
//!
//! The kernel is pure 32-bit integer comparison in the portable core-WGSL
//! subset (the `GeometryPageKey` is split into its two `u32` fields and compared
//! lexicographically), so unlike the 64-bit payload twin it needs no optional
//! device feature and runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Both sides resolve the same sorted table with the same lexicographic key
//! order and the same lower-bound-then-exact-match rule, over integer keys, so
//! every resolved slot - hit or the
//! [`UNMAPPED_SLOT`](prism_render_architecture::paging::UNMAPPED_SLOT)
//! sentinel on a miss - is bit-exact.
//!
//! There is no floating-point arithmetic and hence no tolerance.
//!
//! Provenance: standard sorted-table binary search and `wgpu` compute dispatch;
//! no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::virtual_geometry::GeometryPageKey;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters for one resolve dispatch. Layout matches `Params` in
/// `shaders/page_table_resolve.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    entry_count: u32,
    query_count: u32,
    pad0: u32,
    pad1: u32,
}

/// One `(key, slot)` row of the sorted resident table. `16`-byte stride,
/// matching `Entry` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuEntry {
    asset: u32,
    page: u32,
    slot: u32,
    pad: u32,
}

/// One query key. `8`-byte stride, matching `Query` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    asset: u32,
    page: u32,
}

/// Errors returned by [`GpuPageTable::resolve`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolveError {
    /// The resident entry table was not sorted ascending by key, so the
    /// device's binary search would return wrong slots. The reference
    /// [`PagePool::entries`](prism_render_architecture::paging::PagePool::entries)
    /// always yields a sorted list; this guards a hand-built table.
    EntriesNotSorted {
        /// Index of the first entry that is not strictly greater than its
        /// predecessor.
        index: usize,
    },
}

impl core::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ResolveError::EntriesNotSorted { index } => {
                write!(f, "resident entry table is not sorted ascending at index {index}")
            }
        }
    }
}

impl core::error::Error for ResolveError {}

/// A compiled, reusable page-table resolve pipeline.
pub struct GpuPageTable {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPageTable {
    /// Compiles the resolve kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-WGSL subset, so unlike
    /// [`GpuPayloadRaster::new`](crate::GpuPayloadRaster::new) this never
    /// returns [`None`]: it compiles on any backend `ctx` acquired.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPageTable {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_page_table_resolve"),
            source: ShaderSource::Wgsl(include_str!("../shaders/page_table_resolve.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_page_table_resolve_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_page_table_resolve_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_page_table_resolve_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("resolve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPageTable {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves each key in `queries` against the sorted resident `entries`,
    /// returning one slot per query in input order.
    ///
    /// `entries` must be sorted ascending by key - exactly the list
    /// [`PagePool::entries`](prism_render_architecture::paging::PagePool::entries)
    /// exports. A query that matches an entry yields that entry's slot; a miss
    /// yields [`UNMAPPED_SLOT`](prism_render_architecture::paging::UNMAPPED_SLOT), so the result equals
    /// [`PagePool::slot_of`](prism_render_architecture::paging::PagePool::slot_of)
    /// mapped over the queries (`None` denoted by [`UNMAPPED_SLOT`](prism_render_architecture::paging::UNMAPPED_SLOT)).
    ///
    /// # Errors
    ///
    /// [`ResolveError::EntriesNotSorted`] when `entries` is not strictly
    /// ascending, which would make the device binary search unsound.
    pub fn resolve(
        &self,
        ctx: &GpuContext,
        entries: &[(GeometryPageKey, u32)],
        queries: &[GeometryPageKey],
    ) -> Result<Vec<u32>, ResolveError> {
        for i in 1..entries.len() {
            if entries[i].0 <= entries[i - 1].0 {
                return Err(ResolveError::EntriesNotSorted { index: i });
            }
        }
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let device = ctx.device();

        let params = Params {
            entry_count: entries.len() as u32,
            query_count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
        };

        let gpu_entries: Vec<GpuEntry> = entries
            .iter()
            .map(|(key, slot)| GpuEntry {
                asset: key.asset,
                page: key.page,
                slot: *slot,
                pad: 0,
            })
            .collect();
        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|key| GpuQuery {
                asset: key.asset,
                page: key.page,
            })
            .collect();

        // A storage buffer must never be zero-sized. With no entries every
        // query legitimately misses, so a single padded row the shader skips
        // (`entry_count` stays 0) keeps the binding valid.
        let padded_entry = [GpuEntry::zeroed()];
        let entry_upload: &[GpuEntry] = if gpu_entries.is_empty() {
            &padded_entry
        } else {
            &gpu_entries
        };

        let out_bytes = (queries.len() as u64) * 4;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_page_table_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let entry_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_page_table_entries"),
            contents: bytemuck::cast_slice(entry_upload),
            usage: BufferUsages::STORAGE,
        });
        let query_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_page_table_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let slot_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_page_table_slots"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slot_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_page_table_slots_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_page_table_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: entry_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: query_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: slot_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_page_table_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_page_table_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&slot_buf, 0, &slot_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        slot_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = slot_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let slots = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        slot_stage.unmap();
        debug_assert_eq!(slots.len(), queries.len());
        Ok(slots)
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
