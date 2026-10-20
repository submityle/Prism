//! Exact pass-point previews of real, recorded Prism render-fabric frames.
//!
//! The existing graph records once and the existing wgpu bridge encodes it.
//! One native submission orders graph prefix, readback copy and graph suffix.
//! Physical identity comes from imports or graph allocations, never a matching
//! descriptor supplied by the caller. Graph-owned resources retire after actual
//! queue completion, including when the inspection handle is dropped. Imported
//! persistent/history resources retain their original host ownership.

use crate::{
    capture::CaptureHandle,
    capture_runtime_graph,
    gpu_preview::{
        TexturePreviewBinding, TexturePreviewError, TexturePreviewLimits, TexturePreviewProducer,
        TexturePreviewResult,
    },
    GraphCaptureInputs, GraphCaptureLimits, GraphCaptureReport, ProfilerError,
};
use prism_render_architecture::render_fabric::{
    PhysicalResource, ResourceLayout, RuntimeFeatureGraph,
};
use prism_render_driver::{CommandBuffer, TextureId};
use prism_render_driver_wgpu::{WgpuDevice, WgpuQueue};
use prism_render_graph::{GraphAllocations, LogicalResource, RecordedGraph};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

struct OwnedAllocations {
    device: WgpuDevice,
    allocations: Option<GraphAllocations>,
}
impl Drop for OwnedAllocations {
    fn drop(&mut self) {
        if let Some(allocations) = self.allocations.take() {
            allocations.release(&self.device);
        }
    }
}

/// A rejected request never submits a partial graph.
#[derive(Debug)]
pub enum RuntimePreviewError {
    /// Authoritative capture admission failed before recording.
    Report(ProfilerError),
    /// Selection, layout, format or staging admission failed.
    Texture(TexturePreviewError),
    /// A binding belongs to another exact captured frame.
    StaleReport,
    /// The recorded allocation or pass boundary is absent.
    MissingPhysicalResource,
}
impl std::fmt::Display for RuntimePreviewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "runtime texture preview: {self:?}")
    }
}
impl std::error::Error for RuntimePreviewError {}
impl From<TexturePreviewError> for RuntimePreviewError {
    fn from(error: TexturePreviewError) -> Self {
        Self::Texture(error)
    }
}

/// Unsubmitted graph recording and immutable report of the very same frame.
/// Dropping it discards commands and releases its graph-created allocations.
/// `device` and `queue` must be the pair owning every imported RHI ID.
pub struct RuntimePreviewFrame<'queue> {
    device: WgpuDevice,
    queue: &'queue WgpuQueue,
    report: Arc<GraphCaptureReport>,
    recorded: Option<RecordedGraph>,
    textures: Vec<Option<TextureId>>,
}

/// Owns an unsubmitted frame after a rejected capture, allowing retry or discard.
pub struct RuntimePreviewFailure<'queue> {
    /// Admission or selection error.
    pub error: RuntimePreviewError,
    /// Original frame; dropping it releases all unsubmitted graph allocations.
    pub frame: RuntimePreviewFrame<'queue>,
}
impl std::fmt::Debug for RuntimePreviewFailure<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimePreviewFailure")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}
impl<'queue> RuntimePreviewFrame<'queue> {
    /// Capture the prepared authority, then record its callbacks exactly once.
    /// Does not submit, map, poll, wait, or clone graph commands.
    pub fn record(
        graph: RuntimeFeatureGraph,
        device: &WgpuDevice,
        queue: &'queue WgpuQueue,
        inputs: GraphCaptureInputs<'_>,
        limits: GraphCaptureLimits,
    ) -> Result<Self, RuntimePreviewError> {
        let report = Arc::new(
            capture_runtime_graph(&graph, inputs, limits).map_err(RuntimePreviewError::Report)?,
        );
        let mut textures = vec![None; graph.declaration().resources().len()];
        for (i, resource) in graph.declaration().resources().iter().enumerate() {
            if let Some(binding) = resource.external
                && let PhysicalResource::Texture(id) = binding.physical
            {
                textures[i] = Some(id);
            }
        }
        // Runtime normalization groups texture accesses before buffer accesses,
        // preserving order within each group. The report adapter has already
        // verified this correspondence and every SSA version.
        let mut indices = vec![None; textures.len()];
        for (pass, accesses) in graph
            .declaration()
            .passes()
            .iter()
            .zip(&graph.plan().accesses)
        {
            let mut ordered: Vec<_> = pass.accesses().iter().collect();
            ordered.sort_by_key(|access| {
                !matches!(
                    graph.declaration().resources()[access.resource.0 as usize]
                        .template
                        .layout,
                    ResourceLayout::Texture { .. }
                )
            });
            for (bound, access) in ordered.into_iter().zip(accesses) {
                if let LogicalResource::Texture(index) = access.resource {
                    indices[bound.resource.0 as usize] = Some(index);
                }
            }
        }
        let recorded = graph.record(device);
        for (i, index) in indices.into_iter().enumerate() {
            if textures[i].is_none()
                && let Some(index) = index
            {
                textures[i] = recorded
                    .allocations()
                    .textures()
                    .iter()
                    .find(|(actual, _, _)| *actual == index)
                    .map(|(_, _, realized)| realized.texture);
            }
        }
        Ok(Self {
            device: device.clone(),
            queue,
            report,
            recorded: Some(recorded),
            textures,
        })
    }
    /// The exact immutable prepared-runtime report used for selection and pixels.
    pub fn report(&self) -> &Arc<GraphCaptureReport> {
        &self.report
    }
    /// Select a real pass/resource access in this frame.
    pub fn binding(
        &self,
        pass: u32,
        resource: u32,
    ) -> Result<TexturePreviewBinding, TexturePreviewError> {
        TexturePreviewBinding::new(self.report.clone(), pass, resource)
    }
    /// Copy after the selected actual pass and before every later pass.
    /// A rejected request returns the unsubmitted frame with no GPU work queued.
    pub fn submit(
        mut self,
        binding: TexturePreviewBinding,
        mip: u32,
        layer: u32,
        limits: TexturePreviewLimits,
    ) -> Result<RuntimePreviewSubmission, Box<RuntimePreviewFailure<'queue>>> {
        match self.submit_inner(binding, mip, layer, limits) {
            Ok(submission) => Ok(submission),
            Err(error) => Err(Box::new(RuntimePreviewFailure { error, frame: self })),
        }
    }
    fn submit_inner(
        &mut self,
        binding: TexturePreviewBinding,
        mip: u32,
        layer: u32,
        limits: TexturePreviewLimits,
    ) -> Result<RuntimePreviewSubmission, RuntimePreviewError> {
        if !Arc::ptr_eq(binding.report(), &self.report) {
            return Err(RuntimePreviewError::StaleReport);
        }
        let recorded = self.recorded.as_ref().expect("unsubmitted frame");
        let position = recorded
            .plan()
            .position_by_pass
            .get(binding.pass() as usize)
            .copied()
            .flatten()
            .ok_or(RuntimePreviewError::MissingPhysicalResource)?;
        if recorded.commands().passes.len() != recorded.plan().order.len() {
            return Err(RuntimePreviewError::MissingPhysicalResource);
        }
        let texture = self
            .textures
            .get(binding.resource() as usize)
            .copied()
            .flatten()
            .ok_or(RuntimePreviewError::MissingPhysicalResource)?;
        let mut producer = TexturePreviewProducer::new(
            self.device.wgpu_device(),
            self.queue.wgpu_queue(),
            limits,
        )?;
        let handle = producer.begin(binding.clone())?;
        // Copy encoding and all budget checks happen before any submission.
        producer.copy_texture(handle, &self.device.wgpu_texture(texture), mip, layer)?;
        let (mut prefix, _, allocations) = self
            .recorded
            .take()
            .expect("unsubmitted frame")
            .into_parts();
        // Guard ownership before encoding so validation panics also discard
        // unsubmitted resources rather than leaking backend table entries.
        let retired = OwnedAllocations {
            device: self.device.clone(),
            allocations: Some(allocations),
        };
        let suffix = CommandBuffer {
            label: prefix.label.clone(),
            passes: prefix.passes.split_off(position + 1),
        };
        let before = self.queue.encode(&prefix);
        let after = self.queue.encode(&suffix);
        let submission = producer
            .submit_ordered(handle, [before], [after])
            .expect("owned handle has exactly one validated copy");
        let complete = Arc::new(AtomicBool::new(false));
        let completed = complete.clone();
        self.queue.wgpu_queue().on_submitted_work_done(move || {
            drop(retired);
            completed.store(true, Ordering::Release);
        });
        Ok(RuntimePreviewSubmission {
            producer,
            handle,
            binding,
            mip,
            layer,
            submission,
            complete,
        })
    }
}
impl Drop for RuntimePreviewFrame<'_> {
    fn drop(&mut self) {
        if let Some(recorded) = self.recorded.take() {
            recorded.release(&self.device);
        }
    }
}

/// One submitted actual graph and its bounded asynchronous readback.
/// Poll this object or the creating device to drive map/completion callbacks.
/// Dropping it never destroys in-flight graph allocations: the queue completion
/// callback retains and retires those independently. Imports remain host-owned.
pub struct RuntimePreviewSubmission {
    producer: TexturePreviewProducer,
    handle: CaptureHandle,
    binding: TexturePreviewBinding,
    mip: u32,
    layer: u32,
    submission: wgpu::SubmissionIndex,
    complete: Arc<AtomicBool>,
}
impl RuntimePreviewSubmission {
    /// Exact mip copied from the physical graph texture.
    pub const fn mip(&self) -> u32 {
        self.mip
    }
    /// Exact array layer copied from the physical graph texture.
    pub const fn layer(&self) -> u32 {
        self.layer
    }
    /// Slots still recording, mapping, or awaiting cancellation drainage.
    pub fn occupied_slots(&self) -> usize {
        self.producer.occupied_slots()
    }
    /// Actual retained staging bytes, including cancelled in-flight readbacks.
    pub fn allocated_bytes(&self) -> u64 {
        self.producer.allocated_bytes()
    }
    /// Exact request identity for the native readback result.
    pub const fn handle(&self) -> CaptureHandle {
        self.handle
    }
    /// Report/pass/resource identity, identical to the frame selection.
    pub fn binding(&self) -> &TexturePreviewBinding {
        &self.binding
    }
    /// Native index covers both graph halves and the intervening copy.
    pub fn submission(&self) -> &wgpu::SubmissionIndex {
        &self.submission
    }
    /// Nonblocking real device poll; a readback is returned only once.
    pub fn poll(&mut self) -> Result<Option<TexturePreviewResult>, TexturePreviewError> {
        Ok(self.producer.poll()?.into_iter().next())
    }
    /// Whether the final graph work completed and graph allocations were retired.
    /// This is independent of whether the caller consumed the copied pixels.
    pub fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }
    /// Cancel pixel publication. Full graph execution and safe retirement
    /// continue; cancellation does not undo already submitted rendering.
    pub fn cancel(&mut self) -> Result<(), TexturePreviewError> {
        self.producer.cancel(self.handle)
    }
}
