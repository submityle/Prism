//! Authenticated native resident decode/assembly → RHI page-buffer copies.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use prism_asset_container::GpuResidentAssetRange;
use prism_gdeflate_gpu::{
    GpuResidentAssembler, ResidentAssemblyError, ResidentAssemblyLimits, ResidentByteRange,
    ResidentDeflateTile,
};
use prism_render_architecture::gpu_scene::GpuCompletionValue;
use prism_render_driver::BufferId;

use super::{
    FixedPagePool, FixedPagePoolDescriptor, FixedPageReclaimEvent, GpuPrepareBudget,
    PageAllocation, PagePoolError, PrepareCancellation, PrepareContext,
};
use crate::ResidentRenderDevice;

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

/// A page capability bound to one native pool, not merely a table-local ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentPageAllocation {
    pool_id: u64,
    allocation: PageAllocation,
}

impl ResidentPageAllocation {
    /// RHI allocation metadata for consumers using this pool's render device.
    #[must_use]
    pub const fn allocation(self) -> PageAllocation {
        self.allocation
    }
}

/// One successful queue-ordered publication. Prior pages remain quarantined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentPagePublication {
    /// Newly resident range, retaining its exact logical byte length.
    pub allocation: ResidentPageAllocation,
    /// Prior range, if replacing. Never submit new consumers of this range.
    pub previous: Option<ResidentPageAllocation>,
    /// Producer-defined version, opaque to the pool.
    pub revision: u64,
}

/// Admission/ownership failure. No partial allocation is published on error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentPageError {
    /// Fixed-pool descriptor, allocation, or lifecycle rejection.
    Pool(PagePoolError),
    /// Source or pipeline belongs to another logical device or instance.
    DeviceMismatch,
    /// A capability belongs to another pool (including aliased native IDs).
    ForeignAllocation,
    /// Native allocation, command validation, or device polling failed.
    DeviceResource,
    /// GPU-only byte gather admission or dispatch failed.
    Assembly(ResidentAssemblyError),
    /// Backing-buffer size exceeds the requested new-device-buffer budget.
    DeviceBudget {
        /// Requested backing-buffer bytes.
        required: u64,
        /// Caller-supplied admission bound.
        limit: u64,
    },
    /// Backing-buffer size exceeds the enabled logical device limit.
    DeviceLimit {
        /// Requested backing-buffer bytes.
        required: u64,
        /// Enabled logical device maximum buffer size.
        limit: u64,
    },
    /// Pool identity or retirement serial space was exhausted; never wraps.
    IdentityExhausted,
    /// Authorized range metadata did not describe a contiguous exact output.
    InvalidRange,
}

impl From<PagePoolError> for ResidentPageError {
    fn from(value: PagePoolError) -> Self {
        Self::Pool(value)
    }
}

impl core::fmt::Display for ResidentPageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "resident page: {self:?}")
    }
}
impl std::error::Error for ResidentPageError {}

/// Fixed buffer pages on the actual renderer's original device/queue/tables.
///
/// Construct from `ResidentRenderDevice::from_runtime` after successful
/// runtime initialization. Decode using that snapshot's context. No decoded
/// payload is read back; odd entry/block boundaries are gathered on the GPU.
/// Budgets bound this pool and each assembly request, not retained caller
/// source tiles, compiled pipelines, driver overhead or aggregate live VRAM.
///
/// Consumers must use the same native queue and submit their LAST use of an
/// old allocation BEFORE replacing/releasing it. Retirement fences cover all
/// work already submitted on that queue, never future or unsubmitted work.
/// Reclamation accepts no caller-supplied proof: only actual queue callbacks.
/// A failed or panicking native submission preserves the old publication and
/// quarantines the new reservation until pool destruction. Rebuild the pool
/// after a submission failure; its uncertain range is never recycled.
/// At most one callback per quarantined allocation can be outstanding, hence
/// at most `page_count` (without relying on an assumed frame delay).
///
/// A recovery creates a new device/table generation: drop this pool and rebuild
/// it from the recovered runtime snapshot; old capabilities cannot migrate.
pub struct ResidentPagePool {
    shared: ResidentRenderDevice,
    pool: FixedPagePool,
    native: wgpu::Buffer,
    assembler: GpuResidentAssembler,
    id: u64,
    next_retirement: u64,
    completed: Arc<AtomicU64>,
}

impl ResidentPagePool {
    /// Allocates a pool with bounded host bookkeeping and backing-buffer bytes.
    pub fn new(
        shared: ResidentRenderDevice,
        descriptor: &FixedPagePoolDescriptor,
        max_device_bytes: u64,
    ) -> Result<Self, ResidentPageError> {
        let required = descriptor
            .page_size
            .checked_mul(u64::from(descriptor.page_count))
            .ok_or(PagePoolError::SizeOverflow)?;
        if required > max_device_bytes {
            return Err(ResidentPageError::DeviceBudget {
                required,
                limit: max_device_bytes,
            });
        }
        let limit = shared.context().device().limits().max_buffer_size;
        if required > limit {
            return Err(ResidentPageError::DeviceLimit { required, limit });
        }
        let id = NEXT_POOL_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| ResidentPageError::IdentityExhausted)?;
        let mut budget = GpuPrepareBudget::new(max_device_bytes);
        let cancellation = PrepareCancellation::new(false);
        let mut prepare =
            PrepareContext::new(shared.device(), shared.queue(), &mut budget, &cancellation);
        let pool = device_scope(shared.context().device(), || {
            FixedPagePool::new(&mut prepare, descriptor).map_err(Into::into)
        })?;
        let assembler = GpuResidentAssembler::new(shared.context())
            .map_err(|error| ResidentPageError::Assembly(ResidentAssemblyError::Device(error)))?;
        let native = shared.device().wgpu_buffer(pool.buffer());
        prepare.commit();
        drop(prepare);
        Ok(Self {
            shared,
            pool,
            native,
            assembler,
            id,
            next_retirement: 0,
            completed: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Backing ID in the exact RHI table set retained by `device()`.
    #[must_use]
    pub fn buffer(&self) -> BufferId {
        self.pool.buffer()
    }
    /// Original device/queue/context snapshot for downstream consumers.
    #[must_use]
    pub fn device(&self) -> &ResidentRenderDevice {
        &self.shared
    }
    /// Fixed size of each physical page.
    #[must_use]
    pub fn page_size(&self) -> u64 {
        self.pool.page_size()
    }
    /// Pages immediately reusable by subsequent uploads.
    #[must_use]
    pub fn free_page_count(&self) -> u32 {
        self.pool.free_page_count()
    }
    /// Pages unavailable until an actual queue completion is observed.
    #[must_use]
    pub fn pending_reclaim_page_count(&self) -> u32 {
        self.pool.pending_reclaim_page_count()
    }

    /// Copies a validated tile directly to the RHI buffer and publishes it.
    /// Prior consumer commands must already be submitted on this same queue.
    pub fn upload(
        &mut self,
        tile: &ResidentDeflateTile,
        previous: Option<ResidentPageAllocation>,
        revision: u64,
    ) -> Result<ResidentPagePublication, ResidentPageError> {
        if !tile.is_on_context(self.shared.context()) {
            return Err(ResidentPageError::DeviceMismatch);
        }
        let old = self.validate_previous(previous)?;
        let serial = self.next_serial(old.is_some())?;
        let page_size = self.pool.page_size();
        let transaction = self.pool.begin_resident_copy(tile.byte_len(), old)?;
        let allocation = transaction.allocation();
        let commands = encode_copy(&self.shared, &self.native, tile, allocation, page_size)?;
        transaction.submit(GpuCompletionValue(serial), || {
            device_scope(self.shared.context().device(), || {
                self.shared.context().queue().submit([commands]);
                Ok(())
            })
        })?;
        self.finish(previous, serial);
        Ok(self.publication(previous, revision, allocation))
    }

    /// Gathers an authorized container asset range then copies it GPU-to-GPU.
    /// Slice metadata and pool space are admitted BEFORE assembly submission.
    /// The caller's range and all source tiles may be dropped after return.
    pub fn upload_range(
        &mut self,
        range: &GpuResidentAssetRange,
        previous: Option<ResidentPageAllocation>,
        revision: u64,
        limits: ResidentAssemblyLimits,
    ) -> Result<ResidentPagePublication, ResidentPageError> {
        if range.slices().len() > limits.max_slices {
            return Err(ResidentPageError::Assembly(
                ResidentAssemblyError::TooManySlices {
                    requested: range.slices().len(),
                    limit: limits.max_slices,
                },
            ));
        }
        let mut ranges = Vec::new();
        ranges
            .try_reserve_exact(range.slices().len())
            .map_err(|_| ResidentPageError::Pool(PagePoolError::HostAllocation))?;
        let mut output_offset = 0u64;
        for slice in range.slices() {
            if !slice.tile().is_on_context(self.shared.context()) {
                return Err(ResidentPageError::DeviceMismatch);
            }
            let bytes = slice.buffer_range();
            if slice.output_offset() != output_offset || bytes.start > bytes.end {
                return Err(ResidentPageError::InvalidRange);
            }
            output_offset = output_offset
                .checked_add(bytes.end - bytes.start)
                .ok_or(ResidentPageError::InvalidRange)?;
            ranges.push(ResidentByteRange {
                tile: slice.tile(),
                range: bytes,
            });
        }
        let logical = range.range();
        if logical.end.checked_sub(logical.start) != Some(output_offset) {
            return Err(ResidentPageError::InvalidRange);
        }
        let old = self.validate_previous(previous)?;
        let serial = self.next_serial(old.is_some())?;
        let page_size = self.pool.page_size();
        let transaction = self.pool.begin_resident_copy(output_offset, old)?;
        let allocation = transaction.allocation();
        let tile = self
            .assembler
            .assemble(self.shared.context(), &ranges, limits)
            .map_err(ResidentPageError::Assembly)?;
        let commands = encode_copy(&self.shared, &self.native, &tile, allocation, page_size)?;
        transaction.submit(GpuCompletionValue(serial), || {
            device_scope(self.shared.context().device(), || {
                self.shared.context().queue().submit([commands]);
                Ok(())
            })
        })?;
        self.finish(previous, serial);
        Ok(self.publication(previous, revision, allocation))
    }

    /// Retires after all consumers already submitted on the original queue.
    pub fn release(
        &mut self,
        allocation: ResidentPageAllocation,
        revision: u64,
    ) -> Result<(), ResidentPageError> {
        let allocation = self
            .validate_previous(Some(allocation))?
            .ok_or(ResidentPageError::ForeignAllocation)?;
        let serial = self.next_serial(true)?;
        let transaction = self.pool.begin_release(allocation, revision)?;
        device_scope(self.shared.context().device(), || {
            self.shared.context().queue().submit([]);
            Ok(())
        })?;
        transaction.commit_deferred(GpuCompletionValue(serial))?;
        self.next_retirement = serial;
        self.observe_completion(serial);
        Ok(())
    }

    /// Polls without waiting, then reclaims ONLY callback-proven retirements.
    pub fn reclaim_completed(&mut self) -> Result<Vec<FixedPageReclaimEvent>, ResidentPageError> {
        self.shared
            .context()
            .device()
            .poll(wgpu::PollType::Poll)
            .map_err(|_| ResidentPageError::DeviceResource)?;
        self.pool
            .reclaim_completed(GpuCompletionValue(self.completed.load(Ordering::Acquire)))
            .map_err(Into::into)
    }

    fn validate_previous(
        &self,
        previous: Option<ResidentPageAllocation>,
    ) -> Result<Option<PageAllocation>, ResidentPageError> {
        previous
            .map(|allocation| {
                if allocation.pool_id == self.id {
                    Ok(allocation.allocation)
                } else {
                    Err(ResidentPageError::ForeignAllocation)
                }
            })
            .transpose()
    }
    fn next_serial(&self, retiring: bool) -> Result<u64, ResidentPageError> {
        if retiring {
            self.next_retirement
                .checked_add(1)
                .ok_or(ResidentPageError::IdentityExhausted)
        } else {
            Ok(0)
        }
    }
    fn publication(
        &self,
        previous: Option<ResidentPageAllocation>,
        revision: u64,
        allocation: PageAllocation,
    ) -> ResidentPagePublication {
        ResidentPagePublication {
            allocation: ResidentPageAllocation {
                pool_id: self.id,
                allocation,
            },
            previous,
            revision,
        }
    }
    fn finish(&mut self, previous: Option<ResidentPageAllocation>, serial: u64) {
        if previous.is_some() {
            self.next_retirement = serial;
            self.observe_completion(serial);
        }
    }
    fn observe_completion(&self, serial: u64) {
        let completed = Arc::clone(&self.completed);
        self.shared
            .context()
            .queue()
            .on_submitted_work_done(move || {
                completed.fetch_max(serial, Ordering::Release);
            });
    }
}

impl Drop for ResidentPagePool {
    fn drop(&mut self) {
        self.pool.destroy(self.shared.device());
    }
}

fn encode_copy(
    shared: &ResidentRenderDevice,
    destination: &wgpu::Buffer,
    tile: &ResidentDeflateTile,
    allocation: PageAllocation,
    page_size: u64,
) -> Result<wgpu::CommandBuffer, ResidentPageError> {
    let bytes = tile
        .byte_len()
        .checked_add(3)
        .ok_or(PagePoolError::SizeOverflow)?
        & !3;
    device_scope(shared.context().device(), || {
        let mut encoder =
            shared
                .context()
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("prism_resident_page_copy"),
                });
        encoder.copy_buffer_to_buffer(
            tile.buffer(),
            0,
            destination,
            allocation.offset_bytes(page_size),
            bytes,
        );
        Ok(encoder.finish())
    })
}

fn device_scope<T>(
    device: &wgpu::Device,
    operation: impl FnOnce() -> Result<T, ResidentPageError>,
) -> Result<T, ResidentPageError> {
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result =
        catch_unwind(AssertUnwindSafe(operation)).unwrap_or(Err(ResidentPageError::DeviceResource));
    let validation = futures_lite::future::block_on(validation.pop());
    let internal = futures_lite::future::block_on(internal.pop());
    let oom = futures_lite::future::block_on(oom.pop());
    if validation.is_some() || internal.is_some() || oom.is_some() {
        Err(ResidentPageError::DeviceResource)
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_validation_is_captured_before_publication_even_with_an_outer_scope() {
        let Some(context) = prism_gdeflate_gpu::GpuContext::try_headless() else {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "GPU required"
            );
            return;
        };
        let device = context.device();
        let source = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident_submit_failure_source"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let destination = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident_submit_failure_destination"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&source, 0, &destination, 0, 4);
        let commands = encoder.finish();
        // Recording succeeds, but destruction makes submission invalid. The
        // caller's error scope must not absorb this error and mask success.
        source.destroy();
        let outer = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = device_scope(device, || {
            context.queue().submit([commands]);
            Ok(())
        });
        assert_eq!(result, Err(ResidentPageError::DeviceResource));
        assert!(futures_lite::future::block_on(outer.pop()).is_none());
    }
}
