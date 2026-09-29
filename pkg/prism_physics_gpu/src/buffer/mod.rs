//! Small typed helpers over `wgpu` buffer creation and readback.
//!
//! These wrap the repetitive descriptor plumbing so kernel modules read as a
//! sequence of intent (`upload this slice as read-only storage`, `read this
//! storage buffer back into a Vec`) rather than raw usage flags. They are
//! deliberately minimal and allocation-explicit; nothing here hides a
//! synchronisation point.
//!
//! Provenance: standard `wgpu` buffer usage; no Unreal Engine source or derived
//! code.

use bytemuck::Pod;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, MapMode};

use crate::context::GpuContext;

/// Uploads `data` as a read-only `STORAGE` buffer.
#[must_use]
pub fn storage_read<T: Pod>(device: &Device, label: &str, data: &[T]) -> Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: BufferUsages::STORAGE,
    })
}

/// Uploads `data` as a read-write `STORAGE` buffer that can also be copied out.
#[must_use]
pub fn storage_rw_init<T: Pod>(device: &Device, label: &str, data: &[T]) -> Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    })
}

/// Uploads `value` as a `UNIFORM` buffer.
#[must_use]
pub fn uniform<T: Pod>(device: &Device, label: &str, value: &T) -> Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: BufferUsages::UNIFORM,
    })
}

/// Allocates a zero-initialised, copyable read-write `STORAGE` buffer of
/// `len_bytes` bytes.
#[must_use]
pub fn storage_rw_zeroed(device: &Device, label: &str, len_bytes: u64) -> Buffer {
    // `wgpu` zero-initialises buffers by default, so no explicit clear pass is
    // needed before the first atomic accumulation.
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: len_bytes.max(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Allocates a `MAP_READ | COPY_DST` staging buffer of `len_bytes` bytes.
#[must_use]
pub fn staging(device: &Device, label: &str, len_bytes: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: len_bytes.max(4),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Records a `source` to `staging` copy on `encoder`.
pub fn copy(encoder: &mut CommandEncoder, source: &Buffer, staging: &Buffer, len_bytes: u64) {
    encoder.copy_buffer_to_buffer(source, 0, staging, 0, len_bytes);
}

/// Maps `staging`, waits for the device, and returns its contents as `Vec<T>`.
///
/// The buffer is unmapped before returning so it can be reused.
///
/// # Panics
///
/// Panics if the mapped range is unavailable after the device poll, which would
/// indicate the submitted copy never completed.
#[must_use]
pub fn read_back<T: Pod>(ctx: &GpuContext, staging: &Buffer) -> Vec<T> {
    staging.slice(..).map_async(MapMode::Read, |_| {});
    ctx.wait();
    let view = staging
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let values = bytemuck::cast_slice::<u8, T>(&view).to_vec();
    drop(view);
    staging.unmap();
    values
}
