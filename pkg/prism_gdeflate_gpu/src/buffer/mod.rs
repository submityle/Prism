//! Small typed helpers over `wgpu` buffer creation and readback.
//!
//! These wrap the repetitive descriptor plumbing so kernel modules read as a
//! sequence of intent (`upload this slice as read-only storage`, `read this
//! storage buffer back into a Vec`) rather than raw usage flags. They are
//! deliberately minimal and allocation-explicit; nothing here hides a
//! synchronisation point.
//!
//! A deliberate local copy of the identical helpers in
//! `prism_physics_gpu::buffer`, kept so this twin does not couple to another
//! crate's private module tree.
//!
//! Provenance: standard `wgpu` buffer usage; no Unreal Engine source or derived
//! code.

use bytemuck::Pod;
use core::fmt;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, MapMode, PollType, SubmissionIndex};

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

/// A submission, mapping, or host-copy failure. No partial bytes are returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadbackError {
    /// The submission or mapping callback did not complete within the deadline.
    Timeout,
    /// The device rejected the submission index used for polling.
    Poll,
    /// Mapping failed, for example after device loss or buffer destruction.
    Mapping,
    /// The mapping callback was not delivered after submission completion.
    Callback,
    /// The mapped bytes cannot represent the requested element type.
    Layout,
    /// The host could not allocate the readback result.
    Allocation,
}

impl fmt::Display for ReadbackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "GPU readback failed: {self:?}")
    }
}

impl std::error::Error for ReadbackError {}

// Unmap on every exit, including cancellation of a pending map after timeout.
// The callback owns only its channel sender, never a borrowed staging buffer.
struct Unmap<'a>(&'a Buffer);

impl Drop for Unmap<'_> {
    fn drop(&mut self) {
        self.0.unmap();
    }
}

/// Maps `staging` after this call's submission, not unrelated queue work.
///
/// Polling and callback receipt share a 30-second deadline. On every exit the
/// mapping is released or cancelled, so an error cannot leave a reused buffer
/// mapped. Each dispatch must own its staging buffer until this call returns.
pub fn read_back<T: Pod>(
    ctx: &GpuContext,
    staging: &Buffer,
    submission: SubmissionIndex,
) -> Result<Vec<T>, ReadbackError> {
    read_back_with_timeout(ctx, staging, submission, Duration::from_secs(30))
}

fn read_back_with_timeout<T: Pod>(
    ctx: &GpuContext,
    staging: &Buffer,
    submission: SubmissionIndex,
    timeout: Duration,
) -> Result<Vec<T>, ReadbackError> {
    let started = Instant::now();
    let (sender, receiver) = mpsc::sync_channel(1);
    staging.slice(..).map_async(MapMode::Read, move |result| {
        // A timeout may already have dropped the receiver.
        let _ = sender.send(result);
    });
    let _unmap = Unmap(staging);
    ctx.device()
        .poll(PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(timeout),
        })
        .map_err(|error| match error {
            wgpu::PollError::Timeout => ReadbackError::Timeout,
            _ => ReadbackError::Poll,
        })?;
    receiver
        .recv_timeout(timeout.saturating_sub(started.elapsed()))
        .map_err(|error| match error {
            mpsc::RecvTimeoutError::Timeout => ReadbackError::Timeout,
            mpsc::RecvTimeoutError::Disconnected => ReadbackError::Callback,
        })?
        .map_err(|_| ReadbackError::Mapping)?;
    let view = staging
        .slice(..)
        .get_mapped_range()
        .map_err(|_| ReadbackError::Mapping)?;
    let mapped = bytemuck::try_cast_slice::<u8, T>(&view).map_err(|_| ReadbackError::Layout)?;
    let mut values = Vec::new();
    values.try_reserve_exact(mapped.len()).map_err(|_| ReadbackError::Allocation)?;
    values.extend_from_slice(mapped);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copied_staging(ctx: &GpuContext) -> (Buffer, SubmissionIndex) {
        let source = storage_rw_init(ctx.device(), "readback_test_source", &[7u32, 11]);
        let stage = staging(ctx.device(), "readback_test_stage", 8);
        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        copy(&mut encoder, &source, &stage, 8);
        let submission = ctx.queue().submit([encoder.finish()]);
        (stage, submission)
    }

    #[test]
    fn readback_unmaps_after_success_and_host_layout_failure() {
        let Some(ctx) = GpuContext::try_headless() else { return; };
        let (stage, submission) = copied_staging(&ctx);
        assert_eq!(read_back::<u32>(&ctx, &stage, submission.clone()), Ok(vec![7, 11]));
        assert_eq!(
            read_back::<[u8; 3]>(&ctx, &stage, submission.clone()),
            Err(ReadbackError::Layout)
        );
        assert_eq!(read_back::<u32>(&ctx, &stage, submission), Ok(vec![7, 11]));
    }

    #[test]
    fn readback_reports_destroyed_buffer_and_does_not_poison_next_request() {
        let Some(ctx) = GpuContext::try_headless() else { return; };
        let (stage, submission) = copied_staging(&ctx);
        stage.destroy();
        assert_eq!(read_back::<u32>(&ctx, &stage, submission), Err(ReadbackError::Mapping));
        let (stage, submission) = copied_staging(&ctx);
        assert_eq!(read_back::<u32>(&ctx, &stage, submission), Ok(vec![7, 11]));
    }
}
