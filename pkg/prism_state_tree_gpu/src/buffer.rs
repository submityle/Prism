//! Explicit typed buffer allocation and readback helpers.

use bytemuck::Pod;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, MapMode};

use crate::context::GpuContext;

/// Uploads a typed slice as a read-only storage buffer.
#[must_use]
pub fn storage_read<T: Pod>(device: &Device, label: &str, data: &[T]) -> Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: BufferUsages::STORAGE,
    })
}

/// Uploads a typed value as a uniform buffer.
#[must_use]
pub fn uniform<T: Pod>(device: &Device, label: &str, value: &T) -> Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: BufferUsages::UNIFORM,
    })
}

/// Allocates a zero-initialized writable storage buffer and copy source.
#[must_use]
pub fn storage_rw(device: &Device, label: &str, bytes: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Allocates a map-readable staging buffer.
#[must_use]
pub fn staging(device: &Device, label: &str, bytes: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Records a buffer copy into a command encoder.
pub fn copy(encoder: &mut CommandEncoder, source: &Buffer, target: &Buffer, bytes: u64) {
    encoder.copy_buffer_to_buffer(source, 0, target, 0, bytes);
}

/// Waits for and copies a typed staging buffer into host-owned memory.
#[must_use]
pub fn read_back<T: Pod>(context: &GpuContext, staging: &Buffer) -> Vec<T> {
    staging.slice(..).map_async(MapMode::Read, |_| {});
    context.wait();
    let view = staging
        .slice(..)
        .get_mapped_range()
        .expect("StateTree GPU readback mapping should be available after poll");
    let result = bytemuck::cast_slice::<u8, T>(&view).to_vec();
    drop(view);
    staging.unmap();
    result
}
