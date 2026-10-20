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
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, MapMode, PollType,
    SubmissionIndex,
};

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
    read_back_with_poll(ctx, staging, submission, timeout, |poll_type| {
        ctx.device().poll(poll_type)
    })
}

fn read_back_with_poll<T: Pod>(
    ctx: &GpuContext,
    staging: &Buffer,
    submission: SubmissionIndex,
    timeout: Duration,
    poll: impl FnOnce(PollType) -> Result<wgpu::PollStatus, wgpu::PollError>,
) -> Result<Vec<T>, ReadbackError> {
    let started = Instant::now();
    let device = ctx.device();
    // map_async reports validation failures both through its callback and the
    // device error sink. unmap can also fail after destruction or device loss.
    // Keep all three scopes active until mapping, polling, host copy, and
    // cancellation have finished; native scopes belong to this thread.
    let allocation_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal_scope = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let (sender, receiver) = mpsc::sync_channel(1);
        staging.slice(..).map_async(MapMode::Read, move |result| {
            // Never block a poll/unmap callback, including after cancellation.
            // This closure owns no buffer, scope, or borrowed request state.
            let _ = sender.try_send(result);
        });
        let mapped_immediately = match receiver.try_recv() {
            Ok(Ok(())) => true,
            Ok(Err(_)) => return Err(ReadbackError::Mapping),
            Err(mpsc::TryRecvError::Empty) => false,
            Err(mpsc::TryRecvError::Disconnected) => return Err(ReadbackError::Callback),
        };
        // Native wgpu turns fatal poll errors (including device errors) into
        // panics instead of PollError. Contain that boundary as well.
        catch_unwind(AssertUnwindSafe(|| {
            poll(PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(timeout.saturating_sub(started.elapsed())),
            })
        }))
        .map_err(|_| ReadbackError::Poll)?
        .map_err(|error| match error {
            wgpu::PollError::Timeout => ReadbackError::Timeout,
            _ => ReadbackError::Poll,
        })?;
        if !mapped_immediately {
            receiver
                .recv_timeout(timeout.saturating_sub(started.elapsed()))
                .map_err(|error| match error {
                    mpsc::RecvTimeoutError::Timeout => ReadbackError::Timeout,
                    mpsc::RecvTimeoutError::Disconnected => ReadbackError::Callback,
                })?
                .map_err(|_| ReadbackError::Mapping)?;
        }
        let view = staging
            .slice(..)
            .get_mapped_range()
            .map_err(|_| ReadbackError::Mapping)?;
        let mapped = bytemuck::try_cast_slice::<u8, T>(&view).map_err(|_| ReadbackError::Layout)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(mapped.len())
            .map_err(|_| ReadbackError::Allocation)?;
        values.extend_from_slice(mapped);
        Ok(values)
    }))
    .unwrap_or(Err(ReadbackError::Mapping));

    // The view and receiver are gone before cancellation, even on an unwind.
    // A destroyed buffer may reject unmap too, so cleanup stays inside the
    // scopes and cannot panic from a Drop implementation while unwinding.
    let unmap_result =
        catch_unwind(AssertUnwindSafe(|| staging.unmap())).map_err(|_| ReadbackError::Mapping);
    // Pop every scope in reverse order, without early returns. Native wgpu
    // resolves these futures immediately; no additional device poll is needed.
    let validation_error = crate::context::block_on(validation_scope.pop());
    let internal_error = crate::context::block_on(internal_scope.pop());
    let allocation_error = crate::context::block_on(allocation_scope.pop());
    // Preserve the first operational failure, especially Timeout versus Poll.
    let values = result?;
    unmap_result?;
    if validation_error.is_some() || internal_error.is_some() || allocation_error.is_some() {
        return Err(ReadbackError::Mapping);
    }
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
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let (stage, submission) = copied_staging(&ctx);
        assert_eq!(
            read_back::<u32>(&ctx, &stage, submission.clone()),
            Ok(vec![7, 11])
        );
        assert_eq!(
            read_back::<[u8; 3]>(&ctx, &stage, submission.clone()),
            Err(ReadbackError::Layout)
        );
        assert_eq!(read_back::<u32>(&ctx, &stage, submission), Ok(vec![7, 11]));
    }

    #[test]
    fn readback_reports_destroyed_buffer_and_does_not_poison_next_request() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let (stage, submission) = copied_staging(&ctx);
        stage.destroy();
        // Our readback scopes must consume both map and cleanup errors, and
        // leave the caller's scopes intact and empty. Repeat to check reset.
        let allocation_scope = ctx
            .device()
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal_scope = ctx.device().push_error_scope(wgpu::ErrorFilter::Internal);
        let validation_scope = ctx.device().push_error_scope(wgpu::ErrorFilter::Validation);
        assert_eq!(
            read_back::<u32>(&ctx, &stage, submission.clone()),
            Err(ReadbackError::Mapping)
        );
        assert_eq!(
            read_back::<u32>(&ctx, &stage, submission),
            Err(ReadbackError::Mapping)
        );
        let validation_error = crate::context::block_on(validation_scope.pop());
        let internal_error = crate::context::block_on(internal_scope.pop());
        let allocation_error = crate::context::block_on(allocation_scope.pop());
        assert!(validation_error.is_none(), "{validation_error:?}");
        assert!(internal_error.is_none(), "{internal_error:?}");
        assert!(allocation_error.is_none(), "{allocation_error:?}");
        let (stage, submission) = copied_staging(&ctx);
        assert_eq!(read_back::<u32>(&ctx, &stage, submission), Ok(vec![7, 11]));
    }

    #[test]
    fn readback_reports_invalid_map_usage_and_range_without_leaking_errors() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let invalid_usage = storage_rw_zeroed(ctx.device(), "readback_invalid_usage", 8);
        let invalid_range = staging(ctx.device(), "readback_invalid_range", 6);
        let (stage, submission) = copied_staging(&ctx);
        let scope = ctx.device().push_error_scope(wgpu::ErrorFilter::Validation);
        for invalid in [&invalid_usage, &invalid_range] {
            assert_eq!(
                read_back::<u32>(&ctx, invalid, submission.clone()),
                Err(ReadbackError::Mapping)
            );
        }
        let error = crate::context::block_on(scope.pop());
        assert!(error.is_none(), "{error:?}");
        assert_eq!(read_back::<u32>(&ctx, &stage, submission), Ok(vec![7, 11]));
    }

    #[test]
    fn readback_cancels_pending_mapping_after_poll_failure() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        // Inject poll failures so cancellation is exercised independently of
        // GPU speed, without losing the real device needed for the retry.
        for (poll_error, expected) in [
            (wgpu::PollError::Timeout, ReadbackError::Timeout),
            (
                wgpu::PollError::WrongSubmissionIndex(2, 1),
                ReadbackError::Poll,
            ),
        ] {
            let (stage, submission) = copied_staging(&ctx);
            assert_eq!(
                read_back_with_poll::<u32>(
                    &ctx,
                    &stage,
                    submission.clone(),
                    Duration::from_secs(30),
                    |_| Err(poll_error)
                ),
                Err(expected)
            );
            assert_eq!(read_back::<u32>(&ctx, &stage, submission), Ok(vec![7, 11]));
        }
        let (stage, submission) = copied_staging(&ctx);
        assert_eq!(read_back::<u32>(&ctx, &stage, submission), Ok(vec![7, 11]));
    }

    #[test]
    fn readback_cancels_pending_mapping_after_native_poll_unwind() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let (stage, submission) = copied_staging(&ctx);
        assert_eq!(
            read_back_with_poll::<u32>(
                &ctx,
                &stage,
                submission.clone(),
                Duration::from_secs(30),
                |_| {
                    // Model wgpu's fatal-poll unwind without invoking the
                    // process panic hook for this deliberately injected fault.
                    std::panic::resume_unwind(Box::new("simulated native poll failure"))
                }
            ),
            Err(ReadbackError::Poll)
        );
        assert_eq!(read_back::<u32>(&ctx, &stage, submission), Ok(vec![7, 11]));
        let (stage, submission) = copied_staging(&ctx);
        assert_eq!(read_back::<u32>(&ctx, &stage, submission), Ok(vec![7, 11]));
    }

    #[test]
    fn readback_reports_destruction_while_mapping_and_allows_next_request() {
        let Some(ctx) = GpuContext::try_headless() else {
            return;
        };
        let (stage, submission) = copied_staging(&ctx);
        assert_eq!(
            read_back_with_poll::<u32>(
                &ctx,
                &stage,
                submission,
                Duration::from_secs(30),
                |poll_type| {
                    stage.destroy();
                    ctx.device().poll(poll_type)
                }
            ),
            Err(ReadbackError::Mapping)
        );
        let (stage, submission) = copied_staging(&ctx);
        assert_eq!(read_back::<u32>(&ctx, &stage, submission), Ok(vec![7, 11]));
    }
}
