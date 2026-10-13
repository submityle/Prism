//! Production-facing GPU dispatch contract for the texel-space shade writer.
//!
//! The rest of the texel-cache module owns residency and publication.  This
//! module owns the backend boundary: it allocates the writer's atlas and
//! storage buffers, validates one immutable page-table snapshot against its
//! jobs, uploads the exact ABI rows, and records the compute pass.  Keeping
//! this boundary in one type prevents the real render node from having to
//! duplicate the bind-group and dispatch arithmetic that the parity test used
//! to contain.

use core::mem::size_of;

use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource,
        BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoder,
        ComputePassDescriptor, ComputePipeline, PipelineLayoutDescriptor,
        RawComputePipelineDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
        StorageTextureAccess, Texture, TextureView, TextureViewDimension,
    },
    renderer::{RenderDevice, RenderQueue},
};
use bytemuck::cast_slice;

use crate::RenderShadingVertex;

use super::atlas::{ShadeAtlas, ShadeAtlasError, SHADE_ATLAS_FORMAT};

use super::{
    page_table::{GpuShadePageEntry, ShadePageTableError, ShadePageTableSnapshot},
    shade::ShadeJob,
};

/// The workgroup dimensions declared by [`super::shade.wgsl`].
pub const SHADE_WORKGROUP_SIZE: [u32; 2] = [8, 8];

/// The storage format written by the texture-space irradiance shader.
/// Validated allocation limits for one [`ShadeDispatch`].
///
/// Buffers are allocated at these limits once, then reused for every
/// prepared batch.  A caller must create another dispatch object when a
/// scene's upper bound changes; silently reallocating while a render graph
/// still owns a bind group would make resource lifetime and queue ordering
/// ambiguous.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadeDispatchConfig {
    /// Width and height of the writable shade atlas in texels.
    pub atlas_extent: [u32; 2],
    /// Maximum number of jobs in one dispatch.
    pub max_jobs: u32,
    /// Maximum number of indexed page-table entries.
    pub max_pages: u32,
    /// Maximum number of canonical shading vertices referenced by jobs.
    pub max_vertices: u32,
}

impl ShadeDispatchConfig {
    /// Validates limits without allocating GPU resources.
    pub fn validate(self) -> Result<Self, ShadeDispatchError> {
        if self.atlas_extent.contains(&0)
            || self.max_jobs == 0
            || self.max_pages == 0
            || self.max_vertices == 0
        {
            return Err(ShadeDispatchError::InvalidConfig);
        }
        checked_buffer_bytes(self.max_jobs as usize, size_of::<ShadeJob>())?;
        checked_buffer_bytes(self.max_pages as usize, size_of::<GpuShadePageEntry>())?;
        checked_buffer_bytes(self.max_vertices as usize, size_of::<RenderShadingVertex>())?;
        Ok(self)
    }
}

/// Errors raised before a shade dispatch can be recorded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadeDispatchError {
    /// One of the fixed allocation limits is zero or cannot fit in a buffer.
    InvalidConfig,
    /// A multiplication used to calculate a GPU buffer size overflowed.
    BufferSizeOverflow,
    /// The page-table snapshot failed its CPU-side publication invariants.
    InvalidPageTable(ShadePageTableError),
    /// The batch contains more rows than the fixed dispatch allocation.
    CapacityExceeded,
    /// A job refers to a page-table entry that is not present.
    InvalidPageIndex,
    /// A job's generation/epoch/origin proof differs from its page entry.
    PageProofMismatch,
    /// A job would write outside the atlas.  Partial clipping is rejected so
    /// a page can never become apparently resident with missing texels.
    AtlasBoundsExceeded,
    /// A job references a vertex outside the uploaded canonical vertex table.
    InvalidVertexIndex,
    /// A previous `prepare` has not yet been consumed by `record` or canceled.
    BatchAlreadyPrepared,
    /// The caller passed a prepared token belonging to another batch.
    InvalidPreparedBatch,
    /// The requested atlas extent cannot be allocated as a storage texture.
    InvalidAtlas(ShadeAtlasError),
}

impl From<ShadePageTableError> for ShadeDispatchError {
    fn from(error: ShadePageTableError) -> Self {
        Self::InvalidPageTable(error)
    }
}

fn checked_buffer_bytes(count: usize, element_size: usize) -> Result<u64, ShadeDispatchError> {
    count
        .checked_mul(element_size)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(ShadeDispatchError::BufferSizeOverflow)
}

fn non_zero_buffer_size(bytes: u64) -> u64 {
    bytes.max(1)
}

/// A prepared batch that has been uploaded and is ready to record exactly
/// once.  The token is intentionally opaque: callers cannot alter dispatch
/// dimensions or bypass the CPU proof checks after preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedShadeDispatch {
    serial: u64,
    job_count: u32,
    workgroups: [u32; 3],
}

impl PreparedShadeDispatch {
    /// Number of texture-space jobs in this batch.
    #[must_use]
    pub const fn job_count(self) -> u32 {
        self.job_count
    }

    /// Workgroup counts that will be recorded by [`ShadeDispatch::record`].
    #[must_use]
    pub const fn workgroups(self) -> [u32; 3] {
        self.workgroups
    }
}

/// Result of recording one shade writer pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadeDispatchStats {
    /// Number of jobs consumed by the dispatch.
    pub job_count: u32,
    /// Exact workgroup dimensions sent to wgpu.
    pub workgroups: [u32; 3],
}

/// Reusable GPU resources and command-recording API for the shade writer.
///
/// The object is deliberately independent of Bevy's render graph.  A render
/// node may own it as a resource, call [`Self::prepare`] after it has produced
/// the visibility-driven jobs, then call [`Self::record`] on the graph's
/// command encoder before the resolve consumer.  This module does not claim
/// that such an ordering has been installed: that remains an upper-layer
/// integration task.
pub struct ShadeDispatch {
    config: ShadeDispatchConfig,
    atlas: ShadeAtlas,
    jobs_buffer: Buffer,
    page_table_buffer: Buffer,
    vertices_buffer: Buffer,
    bind_group_layout: BindGroupLayout,
    bind_group: BindGroup,
    pipeline: ComputePipeline,
    next_serial: u64,
    prepared: Option<PreparedShadeDispatch>,
}

impl ShadeDispatch {
    /// Creates the complete writer pipeline and its fixed-capacity resources.
    pub fn new(
        device: &RenderDevice,
        config: ShadeDispatchConfig,
    ) -> Result<Self, ShadeDispatchError> {
        let config = config.validate()?;
        let jobs_bytes = non_zero_buffer_size(checked_buffer_bytes(
            config.max_jobs as usize,
            size_of::<ShadeJob>(),
        )?);
        let page_table_bytes = non_zero_buffer_size(checked_buffer_bytes(
            config.max_pages as usize,
            size_of::<GpuShadePageEntry>(),
        )?);
        let vertices_bytes = non_zero_buffer_size(checked_buffer_bytes(
            config.max_vertices as usize,
            size_of::<RenderShadingVertex>(),
        )?);

        let atlas = ShadeAtlas::new(device, config.atlas_extent)
            .map_err(ShadeDispatchError::InvalidAtlas)?;

        let jobs_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("prism texel-cache shade jobs"),
            size: jobs_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let page_table_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("prism texel-cache shade page table"),
            size: page_table_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vertices_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("prism texel-cache shade vertices"),
            size: vertices_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(
            "prism texel-cache shade bindings",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: SHADE_ATLAS_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                storage_binding(1),
                storage_binding(2),
                storage_binding(3),
            ],
        );
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism texel-cache shade pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let shader = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("prism texel-cache shade shader"),
            source: ShaderSource::Wgsl(include_str!("shade.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("prism texel-cache shade pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("shade_pages"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(
            "prism texel-cache shade bind group",
            &bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(atlas.view()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: jobs_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: page_table_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: vertices_buffer.as_entire_binding(),
                },
            ],
        );

        Ok(Self {
            config,
            atlas,
            jobs_buffer,
            page_table_buffer,
            vertices_buffer,
            bind_group_layout,
            bind_group,
            pipeline,
            next_serial: 0,
            prepared: None,
        })
    }

    fn validate_batch(
        &self,
        jobs: &[ShadeJob],
        page_table: &ShadePageTableSnapshot,
        vertices: &[RenderShadingVertex],
    ) -> Result<(), ShadeDispatchError> {
        page_table.validate()?;
        if jobs.len() > self.config.max_jobs as usize
            || page_table.len() > self.config.max_pages as usize
            || vertices.len() > self.config.max_vertices as usize
        {
            return Err(ShadeDispatchError::CapacityExceeded);
        }
        let entries = page_table.gpu_entries();
        for job in jobs {
            let Some(entry) = entries.get(job.page_index as usize) else {
                return Err(ShadeDispatchError::InvalidPageIndex);
            };
            if entry.atlas_x != job.atlas_origin[0]
                || entry.atlas_y != job.atlas_origin[1]
                || entry.generation != job.generation
                || entry.epoch != job.epoch
            {
                return Err(ShadeDispatchError::PageProofMismatch);
            }
            let Some(max_x) = job.atlas_origin[0].checked_add(job.extent[0]) else {
                return Err(ShadeDispatchError::AtlasBoundsExceeded);
            };
            let Some(max_y) = job.atlas_origin[1].checked_add(job.extent[1]) else {
                return Err(ShadeDispatchError::AtlasBoundsExceeded);
            };
            if job.extent.contains(&0)
                || max_x > self.config.atlas_extent[0]
                || max_y > self.config.atlas_extent[1]
            {
                return Err(ShadeDispatchError::AtlasBoundsExceeded);
            }
            if job
                .vertex_indices
                .into_iter()
                .any(|index| index as usize >= vertices.len())
            {
                return Err(ShadeDispatchError::InvalidVertexIndex);
            }
        }
        Ok(())
    }

    /// Uploads one validated batch.  It must be consumed by [`Self::record`]
    /// before another batch is prepared, preventing queue writes for two
    /// dispatches from aliasing the same fixed buffers.
    pub fn prepare(
        &mut self,
        queue: &RenderQueue,
        jobs: &[ShadeJob],
        page_table: &ShadePageTableSnapshot,
        vertices: &[RenderShadingVertex],
    ) -> Result<PreparedShadeDispatch, ShadeDispatchError> {
        if self.prepared.is_some() {
            return Err(ShadeDispatchError::BatchAlreadyPrepared);
        }
        self.validate_batch(jobs, page_table, vertices)?;
        self.next_serial = self
            .next_serial
            .checked_add(1)
            .ok_or(ShadeDispatchError::InvalidPreparedBatch)?;

        if !jobs.is_empty() {
            queue.write_buffer(&self.jobs_buffer, 0, cast_slice(jobs));
        }
        let page_words = page_table.gpu_words();
        if !page_words.is_empty() {
            queue.write_buffer(&self.page_table_buffer, 0, cast_slice(&page_words));
        }
        if !vertices.is_empty() {
            queue.write_buffer(&self.vertices_buffer, 0, cast_slice(vertices));
        }

        let workgroups = [
            jobs.iter()
                .map(|job| job.extent[0])
                .max()
                .unwrap_or(0)
                .div_ceil(SHADE_WORKGROUP_SIZE[0]),
            jobs.iter()
                .map(|job| job.extent[1])
                .max()
                .unwrap_or(0)
                .div_ceil(SHADE_WORKGROUP_SIZE[1]),
            jobs.len() as u32,
        ];
        let prepared = PreparedShadeDispatch {
            serial: self.next_serial,
            job_count: jobs.len() as u32,
            workgroups,
        };
        self.prepared = Some(prepared);
        Ok(prepared)
    }

    /// Records the compute pass and consumes the prepared token exactly once.
    pub fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        prepared: PreparedShadeDispatch,
    ) -> Result<ShadeDispatchStats, ShadeDispatchError> {
        if self.prepared != Some(prepared) {
            return Err(ShadeDispatchError::InvalidPreparedBatch);
        }
        if prepared.job_count != 0 {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism texel-cache shade dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(
                prepared.workgroups[0],
                prepared.workgroups[1],
                prepared.workgroups[2],
            );
        }
        self.prepared = None;
        Ok(ShadeDispatchStats {
            job_count: prepared.job_count,
            workgroups: prepared.workgroups,
        })
    }

    /// Cancels a prepared upload when the surrounding render pass is skipped.
    pub fn cancel(&mut self, prepared: PreparedShadeDispatch) -> Result<(), ShadeDispatchError> {
        if self.prepared == Some(prepared) {
            self.prepared = None;
            Ok(())
        } else {
            Err(ShadeDispatchError::InvalidPreparedBatch)
        }
    }

    /// Fixed allocation limits used by this dispatch.
    #[must_use]
    pub const fn config(&self) -> ShadeDispatchConfig {
        self.config
    }

    /// Writable shade atlas owned by this dispatch.
    #[must_use]
    pub fn atlas(&self) -> &Texture {
        self.atlas.texture()
    }

    /// Atlas view bound at group 0, binding 0.
    #[must_use]
    pub fn atlas_view(&self) -> &TextureView {
        self.atlas.view()
    }

    /// Group-0 layout, exposed for a render-graph integration layer that needs
    /// to assert resource compatibility while assembling a larger pipeline.
    #[must_use]
    pub fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }

    /// The writer pipeline used by [`Self::record`].
    #[must_use]
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }
}

fn storage_binding(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_zero_limits_and_accepts_a_real_bound() {
        let invalid = ShadeDispatchConfig {
            atlas_extent: [64, 64],
            max_jobs: 0,
            max_pages: 4,
            max_vertices: 8,
        };
        assert_eq!(invalid.validate(), Err(ShadeDispatchError::InvalidConfig));
        let valid = ShadeDispatchConfig {
            atlas_extent: [64, 32],
            max_jobs: 16,
            max_pages: 16,
            max_vertices: 64,
        };
        assert_eq!(valid.validate(), Ok(valid));
    }

    #[test]
    fn workgroup_dimensions_cover_the_largest_job_axis_and_each_job_in_z() {
        let groups = [
            17u32.div_ceil(SHADE_WORKGROUP_SIZE[0]),
            9u32.div_ceil(SHADE_WORKGROUP_SIZE[1]),
            3,
        ];
        assert_eq!(groups, [3, 2, 3]);
    }
}
