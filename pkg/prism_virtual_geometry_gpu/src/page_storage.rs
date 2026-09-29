//! `wgpu` compute twin of the physical page-data placement
//! ([`PageStorage`](prism_render_architecture::paging::PageStorage)).
//!
//! [`GpuPageTable`](crate::GpuPageTable) resolves *which slot* a resident page
//! occupies; this twin moves the page *contents* in and out of that slot on the
//! device. The CPU golden
//! [`PageStorage`](prism_render_architecture::paging::PageStorage) is a flat
//! pool of `capacity` slots, each `page_words` 32-bit words, so slot `s`'s data
//! is the contiguous span `[s * page_words, (s + 1) * page_words)`. Streaming a
//! page in writes one slot's span; the rasterizer later reads a single word by
//! slot and offset.
//!
//! [`GpuPageStorage::round_trip`] reproduces one such upload-then-read cycle on
//! the device in a single command encoder: a *scatter* pass places a batch of
//! page payloads into their slots, then a *gather* pass reads a batch of words
//! back out. The pool storage buffer starts zeroed - exactly as
//! [`PageStorage::new`](prism_render_architecture::paging::PageStorage::new)
//! zeroes its backing store and as `wgpu` zero-initialises a fresh buffer - so a
//! slot no upload targets reads back `0`, matching both a never-written slot and
//! one the reference cleared with
//! [`clear_slot`](prism_render_architecture::paging::PageStorage::clear_slot).
//!
//! # Portability
//!
//! Both kernels are pure 32-bit integer indexing in the portable core-WGSL
//! subset, so like [`GpuPageTable`](crate::GpuPageTable) this twin needs no
//! optional device feature and runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The pool holds raw `u32` words moved by index alone - there is no
//! floating-point arithmetic - so every gathered word is bit-exact against
//! [`PageStorage::fetch`](prism_render_architecture::paging::PageStorage::fetch)
//! with no tolerance.
//!
//! Provenance: standard indexed scatter/gather and `wgpu` compute dispatch; no
//! Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters shared by both passes. Layout matches `Params` in
/// `shaders/page_pool_scatter.wesl` and `shaders/page_pool_gather.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    page_words: u32,
    capacity: u32,
    pad0: u32,
}

/// A compiled, reusable page-storage scatter/gather pipeline pair.
pub struct GpuPageStorage {
    #[expect(
        dead_code,
        reason = "kept alive so the scatter pipeline it produced stays valid"
    )]
    scatter_module: ShaderModule,
    #[expect(
        dead_code,
        reason = "kept alive so the gather pipeline it produced stays valid"
    )]
    gather_module: ShaderModule,
    scatter_layout: BindGroupLayout,
    gather_layout: BindGroupLayout,
    scatter_pipeline: ComputePipeline,
    gather_pipeline: ComputePipeline,
}

impl GpuPageStorage {
    /// Compiles both kernels on `ctx`.
    ///
    /// Both use only the portable core-WGSL subset, so like
    /// [`GpuPageTable::new`](crate::GpuPageTable::new) this never returns
    /// [`None`]: it compiles on any backend `ctx` acquired.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPageStorage {
        let device = ctx.device();

        let scatter_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_page_pool_scatter"),
            source: ShaderSource::Wgsl(include_str!("../shaders/page_pool_scatter.wesl").into()),
        });
        let gather_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_page_pool_gather"),
            source: ShaderSource::Wgsl(include_str!("../shaders/page_pool_gather.wesl").into()),
        });

        let scatter_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_page_pool_scatter_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let gather_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_page_pool_gather_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });

        let scatter_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_page_pool_scatter_pl"),
            bind_group_layouts: &[Some(&scatter_layout)],
            immediate_size: 0,
        });
        let gather_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_page_pool_gather_pl"),
            bind_group_layouts: &[Some(&gather_layout)],
            immediate_size: 0,
        });

        let scatter_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_page_pool_scatter_pipeline"),
            layout: Some(&scatter_pl),
            module: &scatter_module,
            entry_point: Some("scatter"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let gather_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_page_pool_gather_pipeline"),
            layout: Some(&gather_pl),
            module: &gather_module,
            entry_point: Some("gather"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        GpuPageStorage {
            scatter_module,
            gather_module,
            scatter_layout,
            gather_layout,
            scatter_pipeline,
            gather_pipeline,
        }
    }

    /// Uploads `uploads` into a fresh zeroed pool, then reads `fetches` back.
    ///
    /// The pool holds `capacity` slots of `page_words` words. Each upload
    /// `(slot, page)` scatters `page` (which must be `page_words` long) into
    /// `slot`'s span; an out-of-range slot is skipped, matching the reference.
    /// Each fetch `(slot, word)` gathers one word; the returned vector is one
    /// word per fetch in order, with `0` for an out-of-range read.
    ///
    /// With no fetches the returned vector is empty and no dispatch runs.
    ///
    /// # Panics
    ///
    /// Panics if any upload payload's length is not `page_words`, which the
    /// reference would reject as a size mismatch, or if `capacity` or
    /// `page_words` is zero.
    #[must_use]
    pub fn round_trip(
        &self,
        ctx: &GpuContext,
        page_words: u32,
        capacity: u32,
        uploads: &[(u32, &[u32])],
        fetches: &[(u32, u32)],
    ) -> Vec<u32> {
        assert!(page_words > 0, "page_words must be non-zero");
        assert!(capacity > 0, "capacity must be non-zero");
        for (_, page) in uploads {
            assert_eq!(
                page.len(),
                page_words as usize,
                "each upload payload must hold exactly page_words words"
            );
        }

        let device = ctx.device();

        // The flat pool, slot-major, zero-initialised just like the reference.
        let pool_words = (capacity as u64) * (page_words as u64);
        let pool_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_page_pool_store"),
            size: pool_words * 4,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        self.scatter(ctx, page_words, capacity, uploads, &pool_buf);
        self.gather(ctx, page_words, capacity, fetches, &pool_buf)
    }

    /// Runs the scatter pass that places every upload payload into its slot.
    fn scatter(
        &self,
        ctx: &GpuContext,
        page_words: u32,
        capacity: u32,
        uploads: &[(u32, &[u32])],
        pool_buf: &wgpu::Buffer,
    ) {
        let device = ctx.device();

        // A storage buffer must never be zero-sized. With no uploads the pass
        // still runs with `count` = 0 so the shader does nothing; a single
        // padded element in each input keeps the bindings valid.
        let mut slots: Vec<u32> = uploads.iter().map(|(slot, _)| *slot).collect();
        let mut data: Vec<u32> = uploads
            .iter()
            .flat_map(|(_, page)| page.iter().copied())
            .collect();
        if slots.is_empty() {
            slots.push(0);
        }
        if data.is_empty() {
            data.push(0);
        }

        let params = Params {
            count: uploads.len() as u32,
            page_words,
            capacity,
            pad0: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_page_pool_scatter_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let slot_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_page_pool_scatter_slots"),
            contents: bytemuck::cast_slice(&slots),
            usage: BufferUsages::STORAGE,
        });
        let data_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_page_pool_scatter_data"),
            contents: bytemuck::cast_slice(&data),
            usage: BufferUsages::STORAGE,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_page_pool_scatter_bind_group"),
            layout: &self.scatter_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: slot_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: data_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: pool_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_page_pool_scatter_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_page_pool_scatter_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let threads = (uploads.len() as u32) * page_words;
            let groups = threads.div_ceil(64).max(1);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        ctx.queue().submit([encoder.finish()]);
        ctx.wait();
    }

    /// Runs the gather pass that reads one word per fetch back out of the pool.
    fn gather(
        &self,
        ctx: &GpuContext,
        page_words: u32,
        capacity: u32,
        fetches: &[(u32, u32)],
        pool_buf: &wgpu::Buffer,
    ) -> Vec<u32> {
        if fetches.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let fetch_slots: Vec<u32> = fetches.iter().map(|(slot, _)| *slot).collect();
        let fetch_words: Vec<u32> = fetches.iter().map(|(_, word)| *word).collect();

        let params = Params {
            count: fetches.len() as u32,
            page_words,
            capacity,
            pad0: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_page_pool_gather_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let slot_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_page_pool_gather_slots"),
            contents: bytemuck::cast_slice(&fetch_slots),
            usage: BufferUsages::STORAGE,
        });
        let word_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_page_pool_gather_words"),
            contents: bytemuck::cast_slice(&fetch_words),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (fetches.len() as u64) * 4;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_page_pool_gather_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_page_pool_gather_out_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_page_pool_gather_bind_group"),
            layout: &self.gather_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: slot_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: word_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: pool_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_page_pool_gather_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_page_pool_gather_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gather_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (fetches.len() as u32).div_ceil(64);
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
        let out = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        out_stage.unmap();
        debug_assert_eq!(out.len(), fetches.len());
        out
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
