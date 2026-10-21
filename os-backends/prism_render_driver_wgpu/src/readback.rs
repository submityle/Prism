//! Nonblocking, queue-ordered buffer snapshots. Private to the queue backend.
//!
//! A ticket owns one packed staging buffer and native source handles, not RHI
//! IDs. Its callback only publishes map status. Dropping a pending/failed/lost
//! ticket only drops handles: it never explicitly polls, unmaps or destroys.
//! Loss notifications are cooperative gates, not native-device leases. A host
//! must quiesce concurrent work before external teardown; checks cannot exclude
//! an unrelated native destroy between a check and the following operation.

use super::WgpuQueue;
use crate::device::DeviceInner;
use prism_render_driver as rhi;
use std::ops::Range;
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};

const PENDING: u8 = 0;
const READY: u8 = 1;
const FAILED: u8 = 2;

fn mapping_ready(status: u8) -> Result<bool, rhi::BufferReadbackError> {
    match status {
        READY => Ok(true),
        FAILED => Err(rhi::BufferReadbackError::MapFailed),
        _ => Ok(false),
    }
}

struct Layout {
    bytes: u64,
    ranges: Vec<Range<usize>>,
}

/// Validate every integer conversion, alignment and bound before GPU allocation.
/// Empty batches are immediately ready; individual ranges must be nonempty and
/// obey the native buffer-copy alignment (both offset and size).
fn layout(
    ranges: &[rhi::BufferReadbackRange],
    max_bytes: u64,
    max_buffer_size: u64,
) -> Result<Layout, rhi::BufferReadbackError> {
    use rhi::BufferReadbackError as Error;
    let mut bytes = 0_u64;
    // First pass does not allocate, even for a malformed or over-budget batch.
    for (index, range) in ranges.iter().enumerate() {
        if range.size == 0
            || range.offset % wgpu::COPY_BUFFER_ALIGNMENT != 0
            || range.size % wgpu::COPY_BUFFER_ALIGNMENT != 0
            || range.offset.checked_add(range.size).is_none()
            || range.size > max_buffer_size
        {
            return Err(Error::InvalidRange { index });
        }
        bytes = bytes
            .checked_add(range.size)
            .ok_or(Error::InvalidRange { index })?;
    }
    if bytes > max_bytes {
        return Err(Error::BudgetExceeded {
            requested: bytes,
            limit: max_bytes,
        });
    }
    if bytes > max_buffer_size || bytes > isize::MAX as u64 {
        return Err(Error::ResourceUnavailable);
    }
    let mut packed = Vec::new();
    packed
        .try_reserve_exact(ranges.len())
        .map_err(|_| Error::ResourceUnavailable)?;
    let mut offset = 0_usize;
    for (index, range) in ranges.iter().enumerate() {
        let size = usize::try_from(range.size).map_err(|_| Error::InvalidRange { index })?;
        let end = offset
            .checked_add(size)
            .ok_or(Error::InvalidRange { index })?;
        packed.push(offset..end);
        offset = end;
    }
    Ok(Layout {
        bytes,
        ranges: packed,
    })
}

struct WgpuBufferReadback {
    inner: Arc<DeviceInner>,
    staging: Option<wgpu::Buffer>,
    // Retain these even if the caller retires/recycles the table's BufferIds.
    _sources: Vec<wgpu::Buffer>,
    ranges: Vec<Range<usize>>,
    status: Arc<AtomicU8>,
    consumed: bool,
}

pub(super) fn request(
    queue: &WgpuQueue,
    ranges: &[rhi::BufferReadbackRange],
    max_bytes: u64,
) -> Result<Box<dyn rhi::BufferReadback>, rhi::BufferReadbackError> {
    Ok(Box::new(WgpuBufferReadback::begin(
        queue, ranges, max_bytes,
    )?))
}

impl WgpuBufferReadback {
    fn begin(
        queue: &WgpuQueue,
        ranges: &[rhi::BufferReadbackRange],
        max_bytes: u64,
    ) -> Result<Self, rhi::BufferReadbackError> {
        use rhi::BufferReadbackError as Error;
        if queue.inner.is_lost() {
            return Err(Error::DeviceLost);
        }
        let layout = layout(
            ranges,
            max_bytes,
            queue.inner.device.limits().max_buffer_size,
        )?;
        let mut sources = Vec::new();
        sources
            .try_reserve_exact(ranges.len())
            .map_err(|_| Error::ResourceUnavailable)?;
        for (index, range) in ranges.iter().enumerate() {
            let buffer = queue
                .inner
                .try_resolve_buffer(range.buffer)
                .ok_or(Error::ResourceUnavailable)?;
            if !buffer.usage().contains(wgpu::BufferUsages::COPY_SRC) {
                return Err(Error::ResourceUnavailable);
            }
            if range
                .offset
                .checked_add(range.size)
                .is_none_or(|end| end > buffer.size())
            {
                return Err(Error::InvalidRange { index });
            }
            sources.push(buffer);
        }
        // No staging buffer or commands exist until *all* sources are admitted.
        if queue.inner.is_lost() {
            return Err(Error::DeviceLost);
        }
        let status = Arc::new(AtomicU8::new(PENDING));
        let staging = if layout.bytes == 0 {
            None
        } else {
            let staging = queue.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rhi-async-buffer-readback"),
                size: layout.bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if queue.inner.is_lost() {
                return Err(Error::DeviceLost);
            }
            let mut encoder =
                queue
                    .inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("rhi-async-buffer-readback"),
                    });
            for ((range, source), destination) in ranges.iter().zip(&sources).zip(&layout.ranges) {
                if queue.inner.is_lost() {
                    return Err(Error::DeviceLost);
                }
                encoder.copy_buffer_to_buffer(
                    source,
                    range.offset,
                    &staging,
                    destination.start as u64,
                    range.size,
                );
            }
            // Establish K's snapshot now, not on the first try_complete call.
            // Later writes/submits on this queue cannot change these bytes.
            if queue.inner.is_lost() {
                return Err(Error::DeviceLost);
            }
            queue.queue.submit([encoder.finish()]);
            if queue.inner.is_lost() {
                return Err(Error::DeviceLost);
            }
            let callback_status = Arc::clone(&status);
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    callback_status.store(
                        if result.is_ok() { READY } else { FAILED },
                        Ordering::Release,
                    );
                });
            Some(staging)
        };
        if queue.inner.is_lost() {
            return Err(Error::DeviceLost);
        }
        Ok(Self {
            inner: Arc::clone(&queue.inner),
            staging,
            _sources: sources,
            ranges: layout.ranges,
            status,
            consumed: false,
        })
    }

    fn map_ready(&self) -> Result<bool, rhi::BufferReadbackError> {
        mapping_ready(self.status.load(Ordering::Acquire))
    }
}

impl rhi::BufferReadback for WgpuBufferReadback {
    fn try_complete(&mut self) -> Result<Option<Vec<Vec<u8>>>, rhi::BufferReadbackError> {
        use rhi::BufferReadbackError as Error;
        if self.consumed {
            return Err(Error::Consumed);
        }
        if self.inner.is_lost() {
            return Err(Error::DeviceLost);
        }
        let Some(staging) = &self.staging else {
            self.consumed = true;
            return Ok(Some(Vec::new()));
        };
        if !self.map_ready()? {
            // This is the only poll in the asynchronous implementation. Never
            // wait for a submission or for unrelated queue work to complete.
            let polled = self.inner.device.poll(wgpu::PollType::Poll);
            if self.inner.is_lost() {
                return Err(Error::DeviceLost);
            }
            polled.map_err(|_| Error::MapFailed)?;
            if !self.map_ready()? {
                return Ok(None);
            }
        }
        if self.inner.is_lost() {
            return Err(Error::DeviceLost);
        }
        let mapped = staging
            .slice(..)
            .get_mapped_range()
            .map_err(|_| Error::MapFailed)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.ranges.len())
            .map_err(|_| Error::ResourceUnavailable)?;
        for range in &self.ranges {
            let bytes = mapped.get(range.clone()).ok_or(Error::MapFailed)?;
            let mut exact = Vec::new();
            exact
                .try_reserve_exact(bytes.len())
                .map_err(|_| Error::ResourceUnavailable)?;
            exact.extend_from_slice(bytes);
            output.push(exact);
        }
        drop(mapped);
        if self.inner.is_lost() {
            return Err(Error::DeviceLost);
        }
        staging.unmap();
        self.consumed = true;
        self.staging = None;
        self._sources.clear();
        Ok(Some(output))
    }
}

#[cfg(test)]
#[path = "readback_tests.rs"]
mod tests;
