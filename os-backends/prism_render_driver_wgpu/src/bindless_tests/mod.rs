//! GPU-gated conformance tests for the RHI's bindless descriptor arrays.
//!
//! Each test builds a real bind-group layout whose entry carries an RHI array
//! binding type (`BindingType::BufferArray`, `SamplerArray`, `TextureArray` or
//! `StorageTextureArray`) together with a `count`, binds the matching array
//! `BindingResource`, runs a shader that indexes the array with a value the GPU
//! produced at run time, and reads the result back to prove that the *selected
//! element* — not some other element — determined the output.
//!
//! When no adapter is present (the usual GPU-less CI sandbox) every test
//! returns early and is a no-op, so the suite never requires hardware. Set
//! `PRISM_REQUIRE_GPU=1` to make both a missing adapter and a missing bindless
//! feature bit a hard failure, so a run meant to exercise hardware cannot
//! silently degrade into a no-op.
//!
//! `wgpu` encodes array-ness as `count`, so these tests are also what proves
//! the RHI's array/`count` invariant survives translation: a layout that fails
//! [`prism_render_driver::BindGroupLayoutEntry::validate`] would have panicked
//! before reaching the GPU.

mod buffer_array;
mod sampler_array;
mod storage_texture_array;
mod texture_array;

use prism_render_driver as rhi;
use prism_render_driver_wgpu::{WgpuDevice, WgpuInstance, WgpuQueue};
use rhi::{BufferDescriptor, BufferId, BufferUsages, RenderDevice, RenderQueue};

/// Brings up a device/queue pair, or `None` when no GPU is available.
fn device_or_skip() -> Option<(WgpuDevice, WgpuQueue)> {
    WgpuInstance::new().request_default_device().ok()
}

/// Whether a missing adapter/feature must fail the test instead of skipping it.
fn require_gpu() -> bool {
    std::env::var_os("PRISM_REQUIRE_GPU").is_some()
}

/// Returns `true` when `device` advertises `bit`.
///
/// A missing bit skips the class (its binding-array form cannot be created on
/// this adapter), but under `PRISM_REQUIRE_GPU` the absence is a failure, so the
/// strict run reports exactly which class was unavailable.
fn class_supported(device: &WgpuDevice, bit: rhi::Features, class: &str) -> bool {
    if device.capabilities().features.contains(bit) {
        return true;
    }
    assert!(
        !require_gpu(),
        "PRISM_REQUIRE_GPU is set but the adapter does not advertise {class}"
    );
    false
}

/// Uploads `values` into a freshly created storage buffer.
fn upload_u32s(device: &WgpuDevice, queue: &WgpuQueue, label: &str, values: &[u32]) -> BufferId {
    let size = (values.len() as u64) * 4;
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some(label.into()),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    queue.write_buffer(buffer, 0, &bytes);
    buffer
}

/// Creates a GPU-written storage buffer readable back to the CPU.
fn output_buffer(device: &WgpuDevice, label: &str, u32_count: usize) -> BufferId {
    device.create_buffer(&BufferDescriptor {
        label: Some(label.into()),
        size: (u32_count as u64) * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Reads `count` little-endian `u32`s back from `buffer`.
fn read_u32s(queue: &WgpuQueue, buffer: BufferId, count: usize) -> Vec<u32> {
    let bytes = queue.read_buffer(buffer, (count as u64) * 4);
    assert_eq!(
        bytes.len(),
        count * 4,
        "the readback covers every u32 in the buffer"
    );
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}
