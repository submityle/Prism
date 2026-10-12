//! RHI plumbing for the surface pass: the bind-group layout, the uniform batch
//! upload, the result storage buffer, and the bind group that ties them
//! together.
//!
//! Binding 0 is the read-only uniform array of [`GpuSurfaceParams`]; binding 1
//! is the read-write storage buffer the shader writes one `vec4<f32>` result
//! into per evaluated surface. Every helper takes the RHI trait objects, so the
//! crate names no backend.

use prism_render_driver::{
    BindGroupDescriptor, BindGroupEntry, BindGroupId, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindGroupLayoutId, BindingResource, BindingType, BufferBindingType,
    BufferDescriptor, BufferId, BufferUsages, RenderDevice, RenderQueue, ShaderStages,
};

use crate::params::{GpuSurfaceParams, BATCH_BYTES, SURFACE_PARAMS_SIZE};

/// The bytes one result occupies on the GPU: a `vec4<f32>`.
pub const RESULT_STRIDE: u64 = 16;

/// A debug label for the bind-group layout.
const LAYOUT_LABEL: &str = "prism_render_shading_gpu::surface layout";
/// A debug label for the uniform batch buffer.
const BATCH_LABEL: &str = "prism_render_shading_gpu::surface uniform batch";
/// A debug label for the result storage buffer.
const RESULT_LABEL: &str = "prism_render_shading_gpu::surface results";
/// A debug label for the bind group.
const BIND_GROUP_LABEL: &str = "prism_render_shading_gpu::surface bind group";

/// Creates the two-entry bind-group layout the shader declares.
#[must_use]
pub fn surface_bind_group_layout<D: RenderDevice>(device: &D) -> BindGroupLayoutId {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(LAYOUT_LABEL.into()),
        entries: vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Uploads `params` into a uniform buffer sized for a full [`MAX_BATCH`] batch.
///
/// The buffer is always [`BATCH_BYTES`] bytes — the size the WGSL
/// `array<SurfaceParams, MAX_BATCH>` declares — with the unused tail
/// zero-padded, so one binding covers any batch up to the capacity.
///
/// [`MAX_BATCH`]: crate::params::MAX_BATCH
///
/// # Panics
///
/// Panics when `params` is longer than the batch capacity.
#[must_use]
pub fn upload_surface_batch<D: RenderDevice, Q: RenderQueue>(
    device: &D,
    queue: &Q,
    params: &[GpuSurfaceParams],
) -> BufferId {
    assert!(
        params.len() <= crate::params::MAX_BATCH as usize,
        "a surface batch holds at most {} entries",
        crate::params::MAX_BATCH,
    );
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some(BATCH_LABEL.into()),
        size: BATCH_BYTES,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut bytes = vec![0u8; BATCH_BYTES as usize];
    for (index, surface) in params.iter().enumerate() {
        surface.write_into(&mut bytes[index * SURFACE_PARAMS_SIZE..]);
    }
    queue.write_buffer(buffer, 0, &bytes);
    buffer
}

/// Creates the storage buffer holding `count` `vec4<f32>` results.
///
/// `STORAGE` lets the shader write it; `COPY_SRC` lets a read-back copy it out
/// for golden comparison. `count` is floored at one so the buffer is never
/// zero-sized.
#[must_use]
pub fn create_result_buffer<D: RenderDevice>(device: &D, count: u32) -> BufferId {
    device.create_buffer(&BufferDescriptor {
        label: Some(RESULT_LABEL.into()),
        size: RESULT_STRIDE * u64::from(count.max(1)),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Binds the uniform batch at `@binding(0)` and the result buffer at
/// `@binding(1)`.
#[must_use]
pub fn surface_bind_group<D: RenderDevice>(
    device: &D,
    layout: BindGroupLayoutId,
    params: BufferId,
    results: BufferId,
) -> BindGroupId {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some(BIND_GROUP_LABEL.into()),
        layout: Some(layout),
        entries: vec![
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: params,
                    offset: 0,
                    size: None,
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer {
                    buffer: results,
                    offset: 0,
                    size: None,
                },
            },
        ],
    })
}
