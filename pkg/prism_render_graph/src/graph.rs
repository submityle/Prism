//! The frame render graph: declare passes, compile, and execute.
//!
//! [`RenderGraph`] is the crate's front door. A renderer builds one per frame:
//! it imports the swapchain image and any resident resources, then declares
//! passes with [`add_raster_pass`](RenderGraph::add_raster_pass) and friends.
//! Each pass is a *setup* closure that declares resource accesses (returning an
//! *execute* closure) plus the execute closure itself, which records real
//! driver commands once resources are realized.
//!
//! [`compile`](RenderGraph::compile) turns the declaration into an immutable
//! [`ExecutionPlan`] without touching the GPU. [`prepare`](RenderGraph::prepare)
//! freezes that plan; [`record`](RenderGraph::record) realizes resources and
//! records commands without submitting. [`execute`](RenderGraph::execute) also
//! submits, returning the same allocation owner for completion-time retirement.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::{Arc, Weak};
use alloc::vec;
use alloc::vec::Vec;
use core::any::Any;
use core::fmt;
use core::mem::ManuallyDrop;
use core::sync::atomic::{AtomicBool, Ordering};

use prism_render_driver::{
    BindGroupDescriptor, BindGroupEntry, BindGroupId, BindGroupLayoutId, BindingResource, BufferId,
    BufferUsages, ColorAttachment, CommandBuffer, CommandEncoder, DepthOperations,
    DepthStencilAttachment, Extent3d, RenderDevice, RenderPassDescriptor, RenderQueue,
    TextureDimension, TextureId, TextureUsages, TextureViewDescriptor, TextureViewDimension,
    TextureViewId,
};

use crate::binding::TextureBindGroup;
use crate::blackboard::Blackboard;
use crate::builder::{ExecuteContext, PassBuilder};
use crate::capacity::GraphLimits;
use crate::compile;
use crate::desc::{BufferDesc, TextureDesc};
use crate::handle::{BufferHandle, ResourceIndex, TextureBindGroupHandle, TextureHandle};
use crate::pass::{BoxedExecute, PassFlags, PassKind, PassNode, ThreadBound};
use crate::plan::{CompileError, ExecutionPlan, RasterAttachments};
use crate::record::{self, PassCommands};
use crate::resource::{
    BufferResource, ImportedBuffer, ImportedTexture, Lifetime, RealizedTexture, TextureResource,
};

#[cfg(feature = "parallel")]
use crate::parallel::{self, RecordUnit};

/// The move-only closure a pass runs during execution to record its commands.
///
/// Re-exported for callers that store execute closures in their own types; most
/// code produces one implicitly by returning a closure from a pass's setup.
pub type ExecuteFn = BoxedExecute;

/// A compiled, move-only frame whose callbacks have not run and resources have
/// not been allocated. Recording consumes this owner without compiling again.
#[must_use = "record the prepared frame, or drop it to cancel before allocation"]
pub struct PreparedGraph {
    graph: RenderGraph,
    plan: ExecutionPlan,
}

/// Submission state for one graph, independent of GPU completion/retirement.
///
/// Keep only a [`Weak`] obtained from [`RenderGraph::submission_token`] when
/// tracking outstanding declarations: a live, unsubmitted token blocks writes
/// to resources those commands will read. Once submitted, subsequent writes on
/// the same ordered queue need not wait for GPU completion. This is not a lock:
/// callers must still serialize resource updates with graph declaration and
/// submission, and must not replay old commands after updating their resources.
#[derive(Debug, Default)]
pub struct GraphSubmissionToken {
    submitted: AtomicBool,
}

impl GraphSubmissionToken {
    /// Whether actual queue submission returned and the allocation owner was
    /// marked. This is monotonic and does **not** imply GPU completion.
    #[must_use]
    pub fn is_submitted(&self) -> bool {
        self.submitted.load(Ordering::Acquire)
    }
}

// Unlike external-owner pins, this CPU-only guard must drop normally even when
// a recorded packet is abandoned without explicit GPU-allocation retirement.
#[derive(Debug, Default)]
struct GraphSubmissionGuard(Arc<GraphSubmissionToken>);

/// The move-only owner of physical allocations created by one graph recording.
///
/// Imported texture/buffer IDs are never included, but frame-local bind groups
/// referencing them and explicitly retained external owners are owned here.
/// Indices retain the texture/buffer namespaces of the compiled plan, including
/// synchronization metadata.
///
/// # Lifetime contract
///
/// Keep this owner until every submission using its resources has completed,
/// then call [`Self::release`] on the creating device. For a cancelled frame,
/// first discard all commands that could still reference these allocations.
/// Dropping this owner does **not** destroy GPU resources or release external
/// owner pins: it cannot observe external submission/completion, and an early
/// destroy would invalidate in-flight commands. Omitting explicit release leaks
/// allocations and pinned strong references; device teardown alone does not
/// recover those references. The CPU submission guard is not leaked: dropping
/// this owner expires its weak submission token. Discard every command copy
/// before abandoning an unsubmitted owner; commands alone do not hold the guard.
#[derive(Default)]
#[must_use = "retain until GPU completion, then explicitly release on the creating device"]
pub struct GraphAllocations {
    texture_bind_groups: Vec<(TextureBindGroupHandle, BindGroupId)>,
    textures: Vec<(ResourceIndex, Lifetime, RealizedTexture)>,
    buffers: Vec<(ResourceIndex, Lifetime, BufferId)>,
    // Match the explicit-release GPU allocation contract: dropping a recorded
    // owner cannot safely run arbitrary imported-resource destructors either.
    external_owners: ManuallyDrop<Vec<Arc<dyn Any + Send + Sync>>>,
    submission_guard: Option<GraphSubmissionGuard>,
}

impl fmt::Debug for GraphAllocations {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GraphAllocations")
            .field("texture_bind_groups", &self.texture_bind_groups)
            .field("textures", &self.textures)
            .field("buffers", &self.buffers)
            .field("external_owner_count", &self.external_owners.len())
            .field("submission_guard", &self.submission_guard)
            .finish()
    }
}

impl GraphAllocations {
    /// Marks this graph submitted, only after actual queue submission returns.
    ///
    /// Idempotent; never call during recording, before submitting, or merely
    /// because a completion callback was registered. This permits later resource
    /// writes only under the submitting queue's ordering; it neither establishes
    /// GPU completion nor releases external owners. Default/empty owners with no
    /// associated graph have no token to mark.
    pub fn mark_submitted(&self) {
        if let Some(guard) = &self.submission_guard {
            guard.0.submitted.store(true, Ordering::Release);
        }
    }

    /// Frame-local bind groups, including groups referencing imported/history
    /// textures. Layouts and imported views remain externally owned.
    #[must_use]
    pub fn texture_bind_groups(&self) -> &[(TextureBindGroupHandle, BindGroupId)] {
        &self.texture_bind_groups
    }

    /// Graph-created textures and default views, keyed by virtual texture index.
    #[must_use]
    pub fn textures(&self) -> &[(ResourceIndex, Lifetime, RealizedTexture)] {
        &self.textures
    }

    /// Graph-created buffers, keyed by virtual buffer index.
    #[must_use]
    pub fn buffers(&self) -> &[(ResourceIndex, Lifetime, BufferId)] {
        &self.buffers
    }

    /// The number of explicitly retained strong references to external owners.
    ///
    /// Repeated retention of the same owner counts once per call. Pins survive
    /// [`Self::release_transients`] and are relinquished by [`Self::release`].
    #[must_use]
    pub fn external_owner_count(&self) -> usize {
        self.external_owners.len()
    }

    /// Whether no graph-created allocations or external owner pins remain.
    /// Submission tracking alone does not count as a physical allocation; keep
    /// even an empty owner while its commands may still be submitted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.texture_bind_groups.is_empty()
            && self.textures.is_empty()
            && self.buffers.is_empty()
            && self.external_owners.is_empty()
    }

    /// Releases completed frame-local allocations, retaining persistent ones.
    ///
    /// The caller must establish completion on the actual submitting queue and
    /// use the creating device. Persistent allocations remain owned here so a
    /// history owner can reimport them in later frames and eventually release
    /// them. External owner pins are conservatively retained alongside history
    /// until [`Self::release`], even if no graph-created allocations remain.
    /// This method does not submit, wait, or infer completion.
    pub fn release_transients<D: RenderDevice>(&mut self, device: &D) {
        // All graph bind groups are frame-local, even those binding history.
        for (_, bind_group) in self.texture_bind_groups.drain(..) {
            device.destroy_bind_group(bind_group);
        }
        self.textures.retain(|(_, lifetime, realized)| {
            if *lifetime != Lifetime::Transient {
                return true;
            }
            device.destroy_texture_view(realized.default_view);
            device.destroy_texture(realized.texture);
            false
        });
        self.buffers.retain(|(_, lifetime, buffer)| {
            if *lifetime != Lifetime::Transient {
                return true;
            }
            device.destroy_buffer(*buffer);
            false
        });
    }

    /// Releases all owned allocations, including persistent resources.
    ///
    /// Call only on the creating device after all GPU users have completed (or
    /// after discarding an unsubmitted frame). Use [`Self::release_transients`]
    /// instead when persistent history must outlive this frame. Imported
    /// resource IDs are never directly destroyed by the graph. Bind groups are
    /// destroyed before any views, and views before their textures. External
    /// owner pins are dropped last, after all graph-owned resources, and their
    /// final destructors may destroy imported resources.
    pub fn release<D: RenderDevice>(self, device: &D) {
        for (_, bind_group) in self.texture_bind_groups {
            device.destroy_bind_group(bind_group);
        }
        for (_, _, realized) in &self.textures {
            device.destroy_texture_view(realized.default_view);
        }
        for (_, _, realized) in self.textures {
            device.destroy_texture(realized.texture);
        }
        for (_, _, buffer) in self.buffers {
            device.destroy_buffer(buffer);
        }
        drop(ManuallyDrop::into_inner(self.external_owners));
    }
}

/// Owns allocations only while recording, when no submission can reference them.
/// Unlike the returned allocation owner, this guard can release on unwinding
/// without guessing GPU completion. It stays on the recording caller's thread.
struct RecordingAllocations<'device, D: RenderDevice> {
    device: &'device D,
    allocations: GraphAllocations,
    // Track the texture immediately, before its default-view creation can panic.
    pending_texture: Option<(TextureId, Option<TextureViewId>)>,
}

impl<D: RenderDevice> RecordingAllocations<'_, D> {
    fn into_allocations(mut self) -> GraphAllocations {
        debug_assert!(self.pending_texture.is_none());
        core::mem::take(&mut self.allocations)
    }
}

impl<D: RenderDevice> Drop for RecordingAllocations<'_, D> {
    fn drop(&mut self) {
        // Realization creates every texture/view before any bind groups.
        // A pending texture can therefore never coexist with a live group.
        debug_assert!(
            self.pending_texture.is_none() || self.allocations.texture_bind_groups.is_empty()
        );
        if let Some((texture, view)) = self.pending_texture.take() {
            if let Some(view) = view {
                self.device.destroy_texture_view(view);
            }
            self.device.destroy_texture(texture);
        }
        core::mem::take(&mut self.allocations).release(self.device);
    }
}

/// Recorded commands, their authoritative plan, and their physical allocations.
///
/// This owner is deliberately neither `Clone` nor `Copy`. Recording never
/// submits; an external owner may encode the commands on its native device,
/// submit them in plan order, and retire the allocations after actual GPU
/// completion. Logical queue batches, waits and barriers remain in the plan;
/// as before, the built-in executor uses one ordered command buffer on an
/// auto-synchronizing queue, not independent physical queue submissions.
///
/// Dropping this packet does not destroy GPU resources. See [`GraphAllocations`]
/// for the explicit completion/retirement contract.
#[derive(Debug)]
#[must_use = "retain the allocation owner until actual GPU completion, then release it"]
pub struct RecordedGraph {
    commands: CommandBuffer,
    plan: ExecutionPlan,
    allocations: GraphAllocations,
}

impl RecordedGraph {
    /// The one ordered command buffer, without submitting or transferring it.
    #[must_use]
    pub fn commands(&self) -> &CommandBuffer {
        &self.commands
    }

    /// The exact compiled plan used to realize and record this frame.
    #[must_use]
    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// Physical graph-owned resources and external owner pins in this packet.
    #[must_use]
    pub fn allocations(&self) -> &GraphAllocations {
        &self.allocations
    }

    /// Transfers commands, plan and allocation ownership to an external runtime.
    ///
    /// Native encoding/submission may consume `commands`; keep `allocations`
    /// alive in the pending/submitted batch until its real completion signal.
    /// External owner pins move with `allocations`, not `commands` or `plan`;
    /// dropping those two values does not release the pins.
    #[must_use]
    pub fn into_parts(self) -> (CommandBuffer, ExecutionPlan, GraphAllocations) {
        (self.commands, self.plan, self.allocations)
    }

    /// Discards commands and releases all allocations on the creating device.
    ///
    /// The caller must first establish completion of every GPU use, or ensure
    /// the commands were never submitted or handed off for later submission.
    /// Persistent allocations are also released; use [`Self::into_parts`] and
    /// [`GraphAllocations::release_transients`] to retain cross-frame history.
    pub fn release<D: RenderDevice>(self, device: &D) {
        self.allocations.release(device);
    }
}

/// A declarative, backend-agnostic frame render graph.
///
/// Owns the frame's virtual-resource tables, declared passes, and the shared
/// [`Blackboard`]. Build it per frame, declare passes, then [`compile`] or
/// [`execute`]. The graph is `unsafe`-free and creates no GPU objects until
/// recording realizes them through the [`RenderDevice`] trait; imports and
/// explicitly retained external owners may already hold resident objects.
///
/// [`compile`]: RenderGraph::compile
/// [`execute`]: RenderGraph::execute
pub struct RenderGraph {
    textures: Vec<TextureResource>,
    buffers: Vec<BufferResource>,
    texture_bind_groups: Vec<TextureBindGroup>,
    passes: Vec<PassNode>,
    blackboard: Blackboard,
    swapchain: Extent3d,
    limits: GraphLimits,
    // After captures so cancelling an unrecorded graph drops callback/blackboard
    // captures before releasing external objects those captures may reference.
    external_owners: Vec<Arc<dyn Any + Send + Sync>>,
    submission_guard: GraphSubmissionGuard,
}

impl RenderGraph {
    /// Creates an empty graph whose screen-relative sizes resolve against the
    /// given `swapchain` extent.
    #[must_use]
    pub fn new(swapchain: Extent3d) -> Self {
        Self::with_limits(swapchain, GraphLimits::default())
    }

    /// Creates a graph with checked resource, pass, access and version limits.
    pub fn with_limits(swapchain: Extent3d, limits: GraphLimits) -> Self {
        Self {
            textures: Vec::new(),
            buffers: Vec::new(),
            texture_bind_groups: Vec::new(),
            passes: Vec::new(),
            blackboard: Blackboard::new(),
            swapchain,
            limits,
            external_owners: Vec::new(),
            submission_guard: GraphSubmissionGuard::default(),
        }
    }

    /// The swapchain extent screen-relative sizes resolve against.
    #[must_use]
    pub fn swapchain(&self) -> Extent3d {
        self.swapchain
    }

    /// Observes this graph without keeping a cancelled graph alive.
    ///
    /// The graph, then its prepared/recorded allocation owner, holds the strong
    /// reference. Cancellation or compile/record failure drops it naturally.
    /// Recording alone never marks submission. Manual submitters must call
    /// [`GraphAllocations::mark_submitted`] after actual submit returns, and keep
    /// the allocations alive for as long as any commands may still be submitted.
    /// A weak observer should upgrade only briefly to check `is_submitted()`;
    /// retaining an upgraded strong reference also keeps the token alive.
    #[must_use]
    pub fn submission_token(&self) -> Weak<GraphSubmissionToken> {
        Arc::downgrade(&self.submission_guard.0)
    }

    /// Resolves a texture's declared base-mip extent against this graph's swapchain.
    ///
    /// Works before compilation or GPU allocation, including for imported textures.
    /// Imports use their declared description; this does not query the backend or
    /// account for a default view selecting a smaller mip. Every declared SSA
    /// version of a texture has the same extent. Returns `None` for an out-of-range
    /// resource index or a version that has not been declared.
    ///
    /// The handle must belong to this graph: handles encode only index/version,
    /// so a coincidentally valid handle from another graph cannot be distinguished.
    #[must_use]
    pub fn texture_extent(&self, handle: TextureHandle) -> Option<Extent3d> {
        let texture = self.textures.get(handle.resource().get() as usize)?;
        if handle.version() > texture.version {
            return None;
        }
        Some(texture.desc.size.resolve(self.swapchain))
    }

    /// Pins an external owner through declaration, recording and GPU retirement.
    ///
    /// Use this for imported resources, pipelines, layouts or other external
    /// objects whose lifetime must cover the graph's commands. Each call takes
    /// one strong reference, without deduplication or inferred resource usage;
    /// culling does not remove pins. The owner must itself prevent destruction
    /// of the relevant resources while a strong reference exists. Retention
    /// cannot prevent explicit destruction through a different handle.
    ///
    /// Dropping a declaration/prepared graph or failing compilation releases
    /// the pins automatically. The graph keeps them during realization, then
    /// transfers them to the recording cleanup guard before callbacks run.
    /// Device-creation/callback unwinding also releases the pins, after
    /// graph-owned resources and remaining callback/blackboard captures.
    /// Successful sequential or parallel recording transfers them into the
    /// returned [`GraphAllocations`], including via [`RecordedGraph::into_parts`].
    /// The caller must explicitly [`release`](GraphAllocations::release) that
    /// owner after actual GPU completion, or after discarding every copy of
    /// unsubmitted commands. Dropping a recorded packet/allocations instead
    /// deliberately leaks pins rather than risking premature destruction.
    /// [`GraphAllocations::release_transients`] does not release these pins.
    ///
    /// This API does not submit, wait, install a completion callback, or assume
    /// that a queue callback covers work that has not yet been submitted.
    pub fn retain_external_owner(&mut self, owner: Arc<dyn Any + Send + Sync>) {
        self.external_owners.push(owner);
    }

    /// Number of strongly retained external owners on this declaration graph.
    ///
    /// This diagnostic is useful for proving that imported resources remain
    /// pinned independently for each output graph. The graph does not dedupe
    /// owners: the count is the number of explicit retention calls.
    #[must_use]
    pub fn external_owner_count(&self) -> usize {
        self.external_owners.len()
    }

    /// Borrows the frame blackboard.
    #[must_use]
    pub fn blackboard(&self) -> &Blackboard {
        &self.blackboard
    }

    /// Mutably borrows the frame blackboard.
    pub fn blackboard_mut(&mut self) -> &mut Blackboard {
        &mut self.blackboard
    }

    /// The number of passes declared so far (before culling).
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Declares a frame-local bind group over graph textures' default views.
    ///
    /// This stores CPU metadata only: no driver allocation occurs until record,
    /// when views are resolved and groups are created on the calling thread,
    /// before any sequential or parallel execute callback. The returned handle
    /// can be cloned and resolved with [`ExecuteContext::texture_bind_group`].
    /// The caller owns `layout` and must keep it valid through group retirement.
    ///
    /// This is NOT a resource access declaration: each pass must still call
    /// [`PassBuilder::read_texture`] (or the appropriate write/access helper).
    /// No usage, dependency, barrier or texture liveness is inferred here. If
    /// any entry's texture is unused after pass culling, this group is not
    /// created and resolving it during execution panics. Otherwise the group is
    /// realized even if no surviving callback resolves it. Texture versions do
    /// not snapshot contents; explicit pass accesses determine ordering.
    ///
    /// All groups are owned by [`GraphAllocations`] until explicit completion-
    /// time release, including groups binding imported or persistent textures.
    /// Cancelling a declaration/prepared graph allocates no driver objects;
    /// cancel an unsubmitted recorded graph with [`RecordedGraph::release`].
    ///
    /// # Panics
    /// Panics on an invalid texture index/version, duplicate binding slots, or
    /// exhaustion of the configured resource/entry limits (`resources` groups,
    /// `accesses_per_pass` entries per group). Texture handles must originate
    /// from this graph, just as for pass access declarations.
    #[must_use]
    pub fn create_texture_bind_group(
        &mut self,
        layout: BindGroupLayoutId,
        entries: Vec<(u32, TextureHandle)>,
        label: Option<String>,
    ) -> TextureBindGroupHandle {
        let index = GraphLimits::next_id(
            "texture bind groups",
            self.texture_bind_groups.len(),
            self.limits.resources,
        )
        .expect("graph texture bind group capacity precondition");
        GraphLimits::check_count(
            "texture bind group entries",
            entries.len(),
            self.limits.accesses_per_pass,
        )
        .expect("graph texture bind group entry capacity precondition");
        for (position, &(binding, handle)) in entries.iter().enumerate() {
            assert!(
                entries[..position].iter().all(|(slot, _)| *slot != binding),
                "duplicate texture bind group binding slot"
            );
            let resource = self
                .textures
                .get(handle.resource().get() as usize)
                .expect("invalid texture bind group texture index");
            assert!(
                handle.version() <= resource.version,
                "invalid texture bind group texture version"
            );
        }
        let handle = TextureBindGroupHandle::new(index);
        self.texture_bind_groups.push(TextureBindGroup {
            handle: handle.clone(),
            layout,
            entries,
            label,
            realized: None,
        });
        handle
    }

    /// Imports an externally-owned texture (swapchain image, resident asset).
    ///
    /// The graph tracks its state transitions but never allocates or frees it;
    /// its realized objects are seeded from `imported` so execution reuses them
    /// verbatim. Returns a version-0 handle to the imported contents.
    ///
    /// # Panics
    /// Panics if the configured texture limit is exhausted. For dynamic input,
    /// use [`Self::try_import_texture`] to handle admission failure.
    pub fn import_texture(
        &mut self,
        name: impl Into<String>,
        desc: TextureDesc,
        imported: ImportedTexture,
    ) -> TextureHandle {
        self.try_import_texture(name, desc, imported)
            .expect("graph import texture capacity precondition")
    }

    /// Checked import, leaving the graph unchanged if the resource table is full.
    pub fn try_import_texture(
        &mut self,
        name: impl Into<String>,
        desc: TextureDesc,
        imported: ImportedTexture,
    ) -> Result<TextureHandle, CompileError> {
        let index = ResourceIndex(GraphLimits::next_id(
            "textures",
            self.textures.len(),
            self.limits.resources,
        )?);
        self.textures.push(TextureResource {
            name: name.into(),
            desc,
            lifetime: Lifetime::Imported,
            imported: Some(imported),
            inferred_usage: TextureUsages::NONE,
            version: 0,
            realized: Some(RealizedTexture {
                texture: imported.texture,
                default_view: imported.default_view,
            }),
        });
        Ok(TextureHandle::new(index, 0))
    }

    /// Imports an externally-owned buffer. See [`import_texture`] for the
    /// ownership contract.
    ///
    /// [`import_texture`]: RenderGraph::import_texture
    ///
    /// # Panics
    /// Panics if the configured buffer limit is exhausted. For dynamic input,
    /// use [`Self::try_import_buffer`] to handle admission failure.
    pub fn import_buffer(
        &mut self,
        name: impl Into<String>,
        desc: BufferDesc,
        imported: ImportedBuffer,
    ) -> BufferHandle {
        self.try_import_buffer(name, desc, imported)
            .expect("graph import buffer capacity precondition")
    }

    /// Checked import, leaving the graph unchanged if the resource table is full.
    pub fn try_import_buffer(
        &mut self,
        name: impl Into<String>,
        desc: BufferDesc,
        imported: ImportedBuffer,
    ) -> Result<BufferHandle, CompileError> {
        let index = ResourceIndex(GraphLimits::next_id(
            "buffers",
            self.buffers.len(),
            self.limits.resources,
        )?);
        self.buffers.push(BufferResource {
            name: name.into(),
            desc,
            lifetime: Lifetime::Imported,
            imported: Some(imported),
            inferred_usage: BufferUsages::NONE,
            version: 0,
            realized: Some(imported.buffer),
        });
        Ok(BufferHandle::new(index, 0))
    }

    /// Declares a raster pass: binds color/depth attachments and records draws.
    ///
    /// # Panics
    /// Panics if the pass limit is exhausted, before calling `setup`. Setup
    /// must also satisfy the preconditions of its infallible builder calls.
    /// Use [`Self::try_add_raster_pass`] for fallible pass admission.
    pub fn add_raster_pass<Setup, Exec>(
        &mut self,
        name: impl Into<String>,
        flags: PassFlags,
        setup: Setup,
    ) where
        Setup: FnOnce(&mut PassBuilder<'_>) -> Exec,
        Exec: FnOnce(&mut ExecuteContext<'_>) + ThreadBound + 'static,
    {
        self.try_add_raster_pass(name, flags, setup)
            .expect("graph pass capacity precondition");
    }

    /// Checks pass admission before invoking setup.
    ///
    /// # Errors
    /// Returns [`CompileError::Capacity`] without invoking `setup` when full.
    /// Once admitted, setup handles its own checked builder errors; this
    /// method does not roll back setup's resource or blackboard mutations.
    pub fn try_add_raster_pass<Setup, Exec>(
        &mut self,
        name: impl Into<String>,
        flags: PassFlags,
        setup: Setup,
    ) -> Result<(), CompileError>
    where
        Setup: FnOnce(&mut PassBuilder<'_>) -> Exec,
        Exec: FnOnce(&mut ExecuteContext<'_>) + ThreadBound + 'static,
    {
        self.try_add_pass(name, PassKind::Raster, flags, setup)
    }

    /// Declares a compute pass: records dispatches.
    ///
    /// # Panics
    /// Panics if the pass limit is exhausted, before calling `setup`. Setup
    /// must also satisfy the preconditions of its infallible builder calls.
    /// Use [`Self::try_add_compute_pass`] for fallible pass admission.
    pub fn add_compute_pass<Setup, Exec>(
        &mut self,
        name: impl Into<String>,
        flags: PassFlags,
        setup: Setup,
    ) where
        Setup: FnOnce(&mut PassBuilder<'_>) -> Exec,
        Exec: FnOnce(&mut ExecuteContext<'_>) + ThreadBound + 'static,
    {
        self.try_add_compute_pass(name, flags, setup)
            .expect("graph pass capacity precondition");
    }

    /// Checks pass admission before invoking setup.
    ///
    /// # Errors
    /// Returns [`CompileError::Capacity`] without invoking `setup` when full.
    /// Once admitted, setup handles its own checked builder errors; this
    /// method does not roll back setup's resource or blackboard mutations.
    pub fn try_add_compute_pass<Setup, Exec>(
        &mut self,
        name: impl Into<String>,
        flags: PassFlags,
        setup: Setup,
    ) -> Result<(), CompileError>
    where
        Setup: FnOnce(&mut PassBuilder<'_>) -> Exec,
        Exec: FnOnce(&mut ExecuteContext<'_>) + ThreadBound + 'static,
    {
        self.try_add_pass(name, PassKind::Compute, flags, setup)
    }

    /// Declares a transfer pass: copies/blits, tracked for synchronization.
    ///
    /// # Panics
    /// Panics if the pass limit is exhausted, before calling `setup`. Setup
    /// must also satisfy the preconditions of its infallible builder calls.
    /// Use [`Self::try_add_transfer_pass`] for fallible pass admission.
    pub fn add_transfer_pass<Setup, Exec>(
        &mut self,
        name: impl Into<String>,
        flags: PassFlags,
        setup: Setup,
    ) where
        Setup: FnOnce(&mut PassBuilder<'_>) -> Exec,
        Exec: FnOnce(&mut ExecuteContext<'_>) + ThreadBound + 'static,
    {
        self.try_add_transfer_pass(name, flags, setup)
            .expect("graph pass capacity precondition");
    }

    /// Checks pass admission before invoking setup.
    ///
    /// # Errors
    /// Returns [`CompileError::Capacity`] without invoking `setup` when full.
    /// Once admitted, setup handles its own checked builder errors; this
    /// method does not roll back setup's resource or blackboard mutations.
    pub fn try_add_transfer_pass<Setup, Exec>(
        &mut self,
        name: impl Into<String>,
        flags: PassFlags,
        setup: Setup,
    ) -> Result<(), CompileError>
    where
        Setup: FnOnce(&mut PassBuilder<'_>) -> Exec,
        Exec: FnOnce(&mut ExecuteContext<'_>) + ThreadBound + 'static,
    {
        self.try_add_pass(name, PassKind::Transfer, flags, setup)
    }

    /// Declares a present pass: hands an imported swapchain image to the
    /// surface. A present pass is always a culling root.
    ///
    /// # Panics
    /// Panics if the pass limit is exhausted, before calling `setup`. Setup
    /// must also satisfy the preconditions of its infallible builder calls.
    /// Use [`Self::try_add_present_pass`] for fallible pass admission.
    pub fn add_present_pass<Setup, Exec>(
        &mut self,
        name: impl Into<String>,
        flags: PassFlags,
        setup: Setup,
    ) where
        Setup: FnOnce(&mut PassBuilder<'_>) -> Exec,
        Exec: FnOnce(&mut ExecuteContext<'_>) + ThreadBound + 'static,
    {
        self.try_add_present_pass(name, flags, setup)
            .expect("graph pass capacity precondition");
    }

    /// Checks pass admission before invoking setup.
    ///
    /// # Errors
    /// Returns [`CompileError::Capacity`] without invoking `setup` when full.
    /// Once admitted, setup handles its own checked builder errors; this
    /// method does not roll back setup's resource or blackboard mutations.
    pub fn try_add_present_pass<Setup, Exec>(
        &mut self,
        name: impl Into<String>,
        flags: PassFlags,
        setup: Setup,
    ) -> Result<(), CompileError>
    where
        Setup: FnOnce(&mut PassBuilder<'_>) -> Exec,
        Exec: FnOnce(&mut ExecuteContext<'_>) + ThreadBound + 'static,
    {
        self.try_add_pass(name, PassKind::Present, flags, setup)
    }

    /// Shared pass-construction core: runs `setup` against a fresh
    /// [`PassBuilder`], captures the recorded accesses and the returned execute
    /// closure, and appends the resulting [`PassNode`].
    fn try_add_pass<Setup, Exec>(
        &mut self,
        name: impl Into<String>,
        kind: PassKind,
        flags: PassFlags,
        setup: Setup,
    ) -> Result<(), CompileError>
    where
        Setup: FnOnce(&mut PassBuilder<'_>) -> Exec,
        Exec: FnOnce(&mut ExecuteContext<'_>) + ThreadBound + 'static,
    {
        self.try_add_pass_with_dependencies(name, kind, flags, Vec::new(), setup)
            .map(|_| ())
    }

    pub(crate) fn add_topology_pass<Setup>(
        &mut self,
        name: impl Into<String>,
        kind: PassKind,
        flags: PassFlags,
        dependencies: Vec<usize>,
        setup: Setup,
    ) -> Result<usize, CompileError>
    where
        Setup: FnOnce(&mut PassBuilder<'_>),
    {
        self.try_add_pass_with_dependencies(name, kind, flags, dependencies, |builder| {
            setup(builder);
            |_| {}
        })
    }

    /// Declares a pass with explicit control prerequisites in addition to SSA
    /// resource dependencies. Returns its stable executable pass-table index.
    ///
    /// Dependencies must name already-added passes in this executable graph,
    /// not startup declaration [`PassId`](crate::frame_graph::PassId) values
    /// or scheduled positions. Forward/self references are rejected before
    /// `setup` runs. Explicit edges order surviving passes; callers must also
    /// keep control-only work alive using the appropriate [`PassFlags`].
    ///
    /// # Errors
    /// Returns [`CompileError::Capacity`] or [`CompileError::InvalidDependency`]
    /// before invoking `setup` on rejected admission. Once admitted, setup
    /// handles its own checked builder errors; resource/blackboard mutations
    /// made during setup are not rolled back by this method.
    pub fn try_add_pass_with_dependencies<Setup, Exec>(
        &mut self,
        name: impl Into<String>,
        kind: PassKind,
        flags: PassFlags,
        dependencies: Vec<usize>,
        setup: Setup,
    ) -> Result<usize, CompileError>
    where
        Setup: FnOnce(&mut PassBuilder<'_>) -> Exec,
        Exec: FnOnce(&mut ExecuteContext<'_>) + ThreadBound + 'static,
    {
        GraphLimits::next_id("passes", self.passes.len(), self.limits.passes)?;
        GraphLimits::check_count("dependencies", dependencies.len(), self.limits.passes)?;
        let pass_id = self.passes.len();
        for &dependency in &dependencies {
            if dependency >= pass_id {
                return Err(CompileError::InvalidDependency {
                    pass: crate::frame_graph::PassId(pass_id as u32),
                    dependency: crate::frame_graph::PassId(dependency as u32),
                });
            }
        }
        let mut builder = PassBuilder::new(
            &mut self.textures,
            &mut self.buffers,
            &mut self.blackboard,
            kind,
            self.limits,
        );
        let exec = setup(&mut builder);
        let (texture_accesses, buffer_accesses) = builder.finish();
        self.passes.push(PassNode {
            name: name.into(),
            kind,
            flags,
            dependencies,
            texture_accesses,
            buffer_accesses,
            execute: Some(Box::new(exec)),
        });
        Ok(pass_id)
    }

    /// Lightweight control-only admission using the same declaration-order
    /// kernel as full compilation. Resource accesses are deliberately ignored.
    pub(crate) fn declaration_dependency_order(
        passes: &[crate::frame_graph::PassDescriptor],
        limits: GraphLimits,
    ) -> Result<Vec<crate::frame_graph::PassId>, CompileError> {
        GraphLimits::check_count("passes", passes.len(), limits.passes)?;
        for (pass_index, pass) in passes.iter().enumerate() {
            GraphLimits::check_count("dependencies", pass.depends_on.len(), limits.passes)?;
            for &dependency in &pass.depends_on {
                if dependency.0 as usize >= passes.len() || dependency.0 as usize == pass_index {
                    return Err(CompileError::InvalidDependency {
                        pass: crate::frame_graph::PassId(pass_index as u32),
                        dependency,
                    });
                }
            }
        }
        let explicit = passes
            .iter()
            .map(|pass| pass.depends_on.iter().map(|id| id.0 as usize).collect())
            .collect::<Vec<_>>();
        Ok(Self::solve_declaration_dependencies(passes, &explicit)?
            .into_iter()
            .map(|id| crate::frame_graph::PassId(id as u32))
            .collect())
    }

    fn solve_declaration_dependencies(
        passes: &[crate::frame_graph::PassDescriptor],
        explicit: &[Vec<usize>],
    ) -> Result<Vec<usize>, CompileError> {
        let names = passes
            .iter()
            .map(|pass| pass.name.clone().into_owned())
            .collect::<Vec<_>>();
        Ok(compile::kernel::solve(
            &vec![Vec::new(); passes.len()],
            &vec![true; passes.len()],
            explicit,
            &names,
        )?
        .order)
    }

    /// Lowers a startup declaration frontend into this graph's authoritative
    /// SSA compiler. The declaration records remain backend-neutral; all
    /// scheduling, hazards, lifetimes, aliasing and barriers are derived by
    /// [`Self::compile`] after lowering. Optional input versions parallel the
    /// original pass/access table; writes always use the checked builder's
    /// version allocator regardless of their input version.
    pub(crate) fn compile_declarations<Input: Copy + Into<crate::frame_graph::AccessInput>>(
        resources: &[crate::frame_graph::ResourceDescriptor],
        passes: &[crate::frame_graph::PassDescriptor],
        limits: GraphLimits,
        inputs: Option<&[Vec<Input>]>,
    ) -> Result<ExecutionPlan, CompileError> {
        use crate::frame_graph::{AccessInput, AccessKind, PassId, QueueClass, ResourceLifetime};
        use crate::resource::{BufferResource, Lifetime, TextureResource};
        use prism_render_driver::{BufferUsages, SubresourceRange, TextureFormat, TextureUsages};

        GraphLimits::check_count("resources", resources.len(), limits.resources)?;
        GraphLimits::check_count("passes", passes.len(), limits.passes)?;
        if let Some(inputs) = inputs {
            if inputs.len() != passes.len() {
                return Err(CompileError::InvalidInputShape {
                    pass: None,
                    expected: passes.len(),
                    actual: inputs.len(),
                });
            }
            for (pass_index, pass) in passes.iter().enumerate() {
                if inputs[pass_index].len() != pass.accesses.len() {
                    return Err(CompileError::InvalidInputShape {
                        pass: Some(PassId(pass_index as u32)),
                        expected: pass.accesses.len(),
                        actual: inputs[pass_index].len(),
                    });
                }
            }
        }
        // Classify once while validating IDs instead of rescanning every
        // access for each resource (O(resources + accesses), not O(R * A)).
        let mut resource_classes = vec![(false, false); resources.len()];
        for (pass_index, pass) in passes.iter().enumerate() {
            GraphLimits::check_count("accesses", pass.accesses.len(), limits.accesses_per_pass)?;
            for access in &pass.accesses {
                if access.resource.0 as usize >= resources.len() {
                    return Err(CompileError::InvalidResource {
                        pass: PassId(pass_index as u32),
                        resource: access.resource,
                    });
                }
                let (texture, depth) = &mut resource_classes[access.resource.0 as usize];
                *texture |= access.kind.requires_texture();
                *depth |= matches!(
                    access.kind,
                    AccessKind::DepthAttachment | AccessKind::DepthAttachmentRead
                );
            }
            for &dependency in &pass.depends_on {
                if dependency.0 as usize >= passes.len() || dependency.0 as usize == pass_index {
                    return Err(CompileError::InvalidDependency {
                        pass: PassId(pass_index as u32),
                        dependency,
                    });
                }
            }
        }

        // Reject incompatible uses in original declaration order, independent
        // of dependency scheduling and the texture/buffer namespace split.
        for (pass_index, pass) in passes.iter().enumerate() {
            for access in &pass.accesses {
                if resource_classes[access.resource.0 as usize].0
                    && access.kind.texture_use().is_none()
                {
                    return Err(CompileError::InvalidAccess {
                        pass: PassId(pass_index as u32),
                        resource: access.resource,
                        kind: access.kind,
                    });
                }
            }
        }

        let mut explicit: Vec<Vec<usize>> = passes
            .iter()
            .map(|pass| pass.depends_on.iter().map(|id| id.0 as usize).collect())
            .collect();
        let mut has_sources = false;
        if let Some(inputs) = inputs {
            for (pass_index, pass) in passes.iter().enumerate() {
                for (access_index, access) in pass.accesses.iter().enumerate() {
                    let AccessInput::ProducedBy {
                        pass: producer,
                        access: producer_access,
                    } = inputs[pass_index][access_index].into()
                    else {
                        continue;
                    };
                    let invalid = |reason| CompileError::InvalidInputSource {
                        pass: PassId(pass_index as u32),
                        access: access_index,
                        producer,
                        producer_access,
                        reason,
                    };
                    if producer.0 as usize == pass_index {
                        return Err(invalid("producer must be a different pass"));
                    }
                    let source = passes
                        .get(producer.0 as usize)
                        .and_then(|pass| pass.accesses.get(producer_access))
                        .ok_or_else(|| invalid("producer access does not exist"))?;
                    if source.resource != access.resource {
                        return Err(invalid("producer uses a different resource"));
                    }
                    if !source.kind.writes() {
                        return Err(invalid("producer access is not a write"));
                    }
                    has_sources = true;
                    if !explicit[pass_index].contains(&(producer.0 as usize)) {
                        explicit[pass_index].push(producer.0 as usize);
                    }
                }
            }
        }
        if has_sources {
            // References supplement (rather than duplicate) explicit edges.
            // This is dependency normalization, not another scheduler.
            for dependencies in &mut explicit {
                dependencies.sort_unstable();
                dependencies.dedup();
            }
        }
        let declaration_order = Self::solve_declaration_dependencies(passes, &explicit)?;
        // Only cache versions returned by checked write accessors. Neither
        // semantic producers nor write counts allocate SSA versions here.
        let mut outputs: Vec<Vec<Option<u32>>> = if has_sources {
            passes
                .iter()
                .map(|pass| vec![None; pass.accesses.len()])
                .collect()
        } else {
            Vec::new()
        };

        let mut graph = Self::with_limits(Extent3d::new_2d(1, 1), limits);

        // A topology resource has no driver object at startup. It is typed as
        // a texture when its declaration requires sampling/attachments and as
        // a buffer otherwise; the type only supplies the unified access/state
        // vocabulary, while the actual scene owner binds the real resource.
        #[derive(Clone, Copy)]
        enum Handle {
            Texture(TextureHandle),
            Buffer(BufferHandle),
        }
        let mut handles = Vec::with_capacity(resources.len());
        let mut texture_count = 0u32;
        let mut buffer_count = 0u32;
        for (index, resource) in resources.iter().enumerate() {
            let (uses_texture, depth) = resource_classes[index];
            let lifetime = match resource.lifetime {
                ResourceLifetime::Transient => Lifetime::Transient,
                ResourceLifetime::Imported => Lifetime::Imported,
                ResourceLifetime::Persistent => Lifetime::Persistent,
            };
            if uses_texture {
                let desc = if depth {
                    TextureDesc::depth(TextureFormat::Depth32Float)
                } else {
                    TextureDesc::color(TextureFormat::Rgba8Unorm)
                };
                // Texture and buffer handles use separate namespaces. The
                // declaration API exposes one ResourceId namespace, so assign
                // dense local indices while lowering and keep the global
                // declaration index only in `handles`.
                let handle = TextureHandle::new(ResourceIndex(texture_count), 0);
                texture_count += 1;
                graph.textures.push(TextureResource {
                    name: resource.name.clone().into_owned(),
                    desc,
                    lifetime,
                    imported: None,
                    inferred_usage: TextureUsages::NONE,
                    version: 0,
                    realized: None,
                });
                handles.push(Handle::Texture(handle));
            } else {
                let handle = BufferHandle::new(ResourceIndex(buffer_count), 0);
                buffer_count += 1;
                graph.buffers.push(BufferResource {
                    name: resource.name.clone().into_owned(),
                    desc: BufferDesc::new(resource.size.max(1), BufferUsages::NONE),
                    lifetime,
                    imported: None,
                    inferred_usage: BufferUsages::NONE,
                    version: 0,
                    realized: None,
                });
                handles.push(Handle::Buffer(handle));
            }
        }

        let mut original_to_graph = vec![usize::MAX; passes.len()];
        for (graph_index, &original_index) in declaration_order.iter().enumerate() {
            original_to_graph[original_index] = graph_index;
        }

        for &original_index in &declaration_order {
            let pass = &passes[original_index];
            let dependencies = explicit[original_index]
                .iter()
                .map(|&id| original_to_graph[id])
                .collect::<Vec<_>>();
            let kind = match pass.queue {
                QueueClass::Transfer => PassKind::Transfer,
                QueueClass::Compute => PassKind::Compute,
                QueueClass::Graphics => {
                    if pass
                        .accesses
                        .iter()
                        .any(|access| access.kind.is_attachment())
                    {
                        PassKind::Raster
                    } else {
                        PassKind::Compute
                    }
                }
            };
            let mut flags = PassFlags::NEVER_CULL;
            if pass.queue == QueueClass::Compute {
                flags = flags.union(PassFlags::ASYNC_COMPUTE);
            }
            let accesses = pass.accesses.clone();
            let mut setup_result: Result<(), CompileError> = Ok(());
            graph.add_topology_pass(pass.name.clone(), kind, flags, dependencies, |builder| {
                // Checked builder accesses are the write-version authority in
                // all input modes. A failure only mutates this disposable local
                // graph; it is returned before any plan can escape.
                setup_result = (|| {
                    for (access_index, access) in accesses.into_iter().enumerate() {
                        let index = access.resource.0 as usize;
                        let input =
                            match inputs.map(|rows| rows[original_index][access_index].into()) {
                                None => None,
                                Some(AccessInput::Version(version)) => Some(version),
                                Some(AccessInput::ProducedBy { pass, access }) => Some(
                                    outputs
                                        .get(pass.0 as usize)
                                        .and_then(|row| row.get(access))
                                        .copied()
                                        .flatten()
                                        .ok_or(CompileError::InvalidInputSource {
                                            pass: PassId(original_index as u32),
                                            access: access_index,
                                            producer: pass,
                                            producer_access: access,
                                            reason: "producer output is unavailable",
                                        })?,
                                ),
                            };
                        let next = match handles[index] {
                            Handle::Texture(handle) => {
                                let usage = access.kind.texture_use().ok_or(
                                    CompileError::InvalidAccess {
                                        pass: PassId(original_index as u32),
                                        resource: access.resource,
                                        kind: access.kind,
                                    },
                                )?;
                                let handle = TextureHandle::new(
                                    handle.resource(),
                                    input.unwrap_or(handle.version()),
                                );
                                Handle::Texture(builder.try_access_texture(
                                    handle,
                                    usage,
                                    SubresourceRange::all(),
                                )?)
                            }
                            Handle::Buffer(handle) => {
                                let usage = access.kind.buffer_use().ok_or(
                                    CompileError::InvalidAccess {
                                        pass: PassId(original_index as u32),
                                        resource: access.resource,
                                        kind: access.kind,
                                    },
                                )?;
                                let handle = BufferHandle::new(
                                    handle.resource(),
                                    input.unwrap_or(handle.version()),
                                );
                                Handle::Buffer(builder.try_use_buffer(handle, usage)?)
                            }
                        };
                        // Retaining an older read input must not roll the
                        // resource's latest handle backwards for later uses.
                        if access.kind.writes() {
                            if has_sources {
                                outputs[original_index][access_index] = Some(match next {
                                    Handle::Texture(handle) => handle.version(),
                                    Handle::Buffer(handle) => handle.version(),
                                });
                            }
                            handles[index] = next;
                        }
                    }
                    Ok(())
                })();
            })?;
            setup_result?;
        }

        // Setup must visit prerequisites first to admit SSA input versions,
        // but its temporary pass indices are not declaration PassIds. Restore
        // the original table (and dependency IDs) before the authoritative
        // compiler derives any pass-indexed or scheduled-position metadata.
        // Access records carry resource versions, not pass indices, so they
        // move unchanged. This is a linear permutation, not another scheduler.
        let mut original_passes: Vec<Option<PassNode>> = (0..passes.len()).map(|_| None).collect();
        for (mut pass, &original_index) in graph.passes.drain(..).zip(&declaration_order) {
            for dependency in &mut pass.dependencies {
                *dependency = declaration_order[*dependency];
            }
            original_passes[original_index] = Some(pass);
        }
        graph.passes = original_passes
            .into_iter()
            .map(|pass| pass.expect("every declaration is lowered exactly once"))
            .collect();
        graph.compile()
    }

    /// Compiles the declared frame into an immutable [`ExecutionPlan`] without
    /// touching the GPU. Errors include cycles and checked memory overflow.
    ///
    /// # Errors
    /// Returns [`CompileError::Cycle`] when the derived dependency graph cannot
    /// be topologically ordered, or another [`CompileError`] for invalid
    /// admission, handles or overflowing footprint arithmetic.
    pub fn compile(&self) -> Result<ExecutionPlan, CompileError> {
        compile::compile(&self.passes, &self.textures, &self.buffers, self.swapchain)
    }

    /// Validates and freezes the frame without allocating or recording work.
    ///
    /// The returned owner is consumed by [`PreparedGraph::record`], which uses
    /// this exact plan without compiling again. A compilation error runs no
    /// execute callbacks and creates no driver objects.
    pub fn prepare(self) -> Result<PreparedGraph, CompileError> {
        let plan = self.compile()?;
        Ok(PreparedGraph { graph: self, plan })
    }

    /// Compiles once, realizes resources and records without submitting.
    ///
    /// On [`CompileError`], no allocation or execute callback has occurred.
    /// Resource creation and callbacks are otherwise infallible trait calls;
    /// their panics are not converted into a recoverable compilation error.
    /// Unwinding releases graph-created, unsubmitted allocations, including a
    /// texture whose default-view creation panicked. Retained external owner
    /// pins are released after graph-owned allocations on recording unwind.
    /// Keep the returned owner until actual GPU completion, then release its
    /// allocations on the creating device.
    pub fn record<D: RenderDevice>(self, device: &D) -> Result<RecordedGraph, CompileError> {
        Ok(self.prepare()?.record(device))
    }

    /// Compiles, realizes resources, records every surviving pass into one
    /// driver [`CommandBuffer`], and submits it on `queue`.
    ///
    /// Consumes the graph: a frame's execute closures are move-only and run
    /// exactly once.
    ///
    /// The recording strategy is selected by the `parallel` feature: without it
    /// passes are recorded in order on the calling thread; with it they are
    /// recorded concurrently on a worker pool and assembled in order. Both
    /// produce the same command buffer.
    ///
    /// Submission is not completion. The returned [`RecordedGraph`] owns the
    /// allocations until the caller observes completion on the real queue and
    /// explicitly releases them. [`RenderQueue`] has no completion API, so
    /// neither this function nor dropping the owner can retire them safely.
    ///
    /// # Errors
    /// Returns [`CompileError`] when compilation fails; no GPU work is issued in
    /// that case.
    pub fn execute<D: RenderDevice, Q: RenderQueue>(
        self,
        device: &D,
        queue: &Q,
    ) -> Result<RecordedGraph, CompileError> {
        #[cfg(feature = "parallel")]
        {
            self.execute_parallel(device, queue)
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.execute_sequential(device, queue)
        }
    }

    /// Records every surviving pass in plan order on the calling thread, then
    /// submits one command buffer.
    ///
    /// This is the reference executor: the parallel executor must produce a
    /// byte-for-byte identical command buffer, so this path stays available
    /// under the `parallel` feature for testing and comparison.
    #[cfg(any(test, not(feature = "parallel")))]
    pub(crate) fn execute_sequential<D: RenderDevice, Q: RenderQueue>(
        self,
        device: &D,
        queue: &Q,
    ) -> Result<RecordedGraph, CompileError> {
        let recorded = self.prepare()?.record_sequential(device);
        queue.submit(core::slice::from_ref(recorded.commands()));
        recorded.allocations().mark_submitted();
        Ok(recorded)
    }

    /// The reference recording path, shared by record-only and submit APIs.
    fn record_commands_sequential(&mut self, plan: &ExecutionPlan) -> CommandBuffer {
        let mut encoder = CommandEncoder::new();
        for (pos, &pidx) in plan.order.iter().enumerate() {
            let kind = self.passes[pidx].kind;
            // Take the move-only closure before the shared borrows below.
            let exec = self.passes[pidx].execute.take();
            let recorded = record::record_pass(
                exec,
                kind,
                &self.textures,
                &self.buffers,
                &self.texture_bind_groups,
            );
            self.encode_recorded(&mut encoder, plan, pos, pidx, recorded);
        }

        encoder.finish()
    }

    /// Records every surviving pass concurrently on a worker pool, then
    /// assembles their commands in plan order and submits one command buffer.
    ///
    /// Only available with the `parallel` feature. See [`Self::execute`] for the
    /// determinism guarantee this upholds.
    #[cfg(feature = "parallel")]
    pub(crate) fn execute_parallel<D: RenderDevice, Q: RenderQueue>(
        self,
        device: &D,
        queue: &Q,
    ) -> Result<RecordedGraph, CompileError> {
        let recorded = self.prepare()?.record_parallel(device);
        queue.submit(core::slice::from_ref(recorded.commands()));
        recorded.allocations().mark_submitted();
        Ok(recorded)
    }

    /// Uses the existing worker pool and assembles in authoritative plan order.
    #[cfg(feature = "parallel")]
    fn record_commands_parallel(&mut self, plan: &ExecutionPlan) -> CommandBuffer {
        // Move every execute closure out of the graph up front so recording
        // reads the (now frozen) resource tables through shared borrows only
        // and each worker owns exactly the closures it records.
        let mut units: Vec<RecordUnit> = Vec::with_capacity(plan.order.len());
        for (pos, &pidx) in plan.order.iter().enumerate() {
            units.push(RecordUnit {
                pos,
                pidx,
                kind: self.passes[pidx].kind,
                execute: self.passes[pidx].execute.take(),
            });
        }

        let recorded = parallel::record(
            &mut units,
            &self.textures,
            &self.buffers,
            &self.texture_bind_groups,
        );

        let mut encoder = CommandEncoder::new();
        for (pos, pidx, commands) in recorded {
            self.encode_recorded(&mut encoder, plan, pos, pidx, commands);
        }

        encoder.finish()
    }

    /// Encodes one pass's recorded commands into `encoder` in plan order.
    ///
    /// Shared by both executors so assembly is identical by construction: the
    /// descriptor, label, and pass kind come from the same `plan` and pass
    /// table regardless of where the commands were recorded.
    fn encode_recorded(
        &self,
        encoder: &mut CommandEncoder,
        plan: &ExecutionPlan,
        pos: usize,
        pidx: usize,
        recorded: PassCommands,
    ) {
        match recorded {
            PassCommands::Render(commands) => {
                let descriptor = self.raster_descriptor(
                    plan.attachments[pos].as_ref(),
                    self.passes[pidx].name.as_str(),
                );
                encoder.push_render_pass(descriptor, commands);
            }
            PassCommands::Compute(commands) => {
                let label = Some(self.passes[pidx].name.clone());
                encoder.push_compute_pass(label, commands);
            }
            PassCommands::Transfer(commands) => {
                let label = Some(self.passes[pidx].name.clone());
                encoder.push_transfer_pass(label, commands);
            }
            // Present contributes only state tracking: handing the swapchain
            // image to the surface is driven by the runtime, not the command
            // buffer.
            PassCommands::None => {}
        }
    }

    /// Creates the driver objects backing every used, not-yet-realized
    /// transient or persistent resource. Imported resources are already
    /// realized at import and are skipped. Guards exactly the allocations
    /// created here, including persistent history, until recording succeeds.
    fn realize<'device, D: RenderDevice>(
        &mut self,
        plan: &ExecutionPlan,
        device: &'device D,
    ) -> RecordingAllocations<'device, D> {
        let mut allocations = RecordingAllocations {
            device,
            // Reserve ownership slots before creating any driver objects, so
            // registering a returned buffer ID cannot allocate or unwind.
            allocations: GraphAllocations {
                texture_bind_groups: Vec::with_capacity(self.texture_bind_groups.len()),
                textures: Vec::with_capacity(self.textures.len()),
                buffers: Vec::with_capacity(self.buffers.len()),
                external_owners: ManuallyDrop::new(Vec::new()),
                // Keep the token continuously alive across realization and
                // graph/callback teardown; only the returned owner survives.
                submission_guard: Some(GraphSubmissionGuard(Arc::clone(&self.submission_guard.0))),
            },
            pending_texture: None,
        };
        for i in 0..self.textures.len() {
            if !plan.used_textures[i] || self.textures[i].realized.is_some() {
                continue;
            }
            let descriptor = self.textures[i]
                .desc
                .lower(self.swapchain, self.textures[i].inferred_usage);
            let texture = device.create_texture(&descriptor);
            allocations.pending_texture = Some((texture, None));
            let view = device.create_texture_view(
                texture,
                &default_view_descriptor(self.textures[i].desc.dimension),
            );
            allocations.pending_texture = Some((texture, Some(view)));
            let realized = RealizedTexture {
                texture,
                default_view: view,
            };
            self.textures[i].realized = Some(realized);
            allocations.allocations.textures.push((
                ResourceIndex(i as u32),
                self.textures[i].lifetime,
                realized,
            ));
            allocations.pending_texture = None;
        }
        for i in 0..self.buffers.len() {
            if !plan.used_buffers[i] || self.buffers[i].realized.is_some() {
                continue;
            }
            let descriptor = self.buffers[i].desc.lower(self.buffers[i].inferred_usage);
            let buffer = device.create_buffer(&descriptor);
            self.buffers[i].realized = Some(buffer);
            allocations.allocations.buffers.push((
                ResourceIndex(i as u32),
                self.buffers[i].lifetime,
                buffer,
            ));
        }
        // Only this serial realization stage touches the device. Workers see
        // the frozen table, and no bind group implicitly keeps textures alive.
        for group in &mut self.texture_bind_groups {
            if group
                .entries
                .iter()
                .any(|(_, texture)| !plan.used_textures[texture.resource().get() as usize])
            {
                continue;
            }
            let descriptor = BindGroupDescriptor {
                label: group.label.clone(),
                layout: Some(group.layout),
                entries: group
                    .entries
                    .iter()
                    .map(|&(binding, texture)| BindGroupEntry {
                        binding,
                        resource: BindingResource::TextureView(
                            self.textures[texture.resource().get() as usize]
                                .realized
                                .expect("used texture must be realized before binding")
                                .default_view,
                        ),
                    })
                    .collect(),
            };
            // Clone before creation: the ownership push below cannot allocate.
            let handle = group.handle.clone();
            let bind_group = device.create_bind_group(&descriptor);
            allocations
                .allocations
                .texture_bind_groups
                .push((handle, bind_group));
            group.realized = Some(bind_group);
        }
        allocations
    }

    /// Builds a driver [`RenderPassDescriptor`] from a raster pass's resolved
    /// attachments, resolving each virtual resource to its realized default
    /// view.
    fn raster_descriptor(
        &self,
        attachments: Option<&RasterAttachments>,
        label: &str,
    ) -> RenderPassDescriptor {
        let mut descriptor = RenderPassDescriptor {
            label: Some(label.into()),
            color_attachments: Vec::new(),
            depth_stencil_attachment: None,
        };
        let Some(attachments) = attachments else {
            return descriptor;
        };

        for color in &attachments.colors {
            let view = self.textures[color.resource.get() as usize]
                .realized
                .expect("color attachment of a surviving pass must be realized")
                .default_view;
            descriptor.color_attachments.push(Some(ColorAttachment {
                view,
                resolve_target: None,
                load: color.load,
                store: color.store,
            }));
        }
        if let Some(depth) = &attachments.depth {
            let view = self.textures[depth.resource.get() as usize]
                .realized
                .expect("depth attachment of a surviving pass must be realized")
                .default_view;
            descriptor.depth_stencil_attachment = Some(DepthStencilAttachment {
                view,
                // No operations means read-only depth in the driver/wgpu;
                // even Load/Store would declare a writable attachment.
                depth: depth.writes.then_some(DepthOperations {
                    load: depth.depth_load,
                    store: depth.depth_store,
                }),
                stencil: None,
            });
        }
        descriptor
    }
}

impl PreparedGraph {
    /// The authoritative plan, available before any driver objects are created.
    #[must_use]
    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// Realizes and records this prepared frame exactly once without submitting.
    ///
    /// No device or queue is sent to recording workers. Imported resources must
    /// belong to this device, and must outlive all submissions that use them.
    /// Explicitly retained external owners move into the allocation owner;
    /// the caller still establishes actual completion before releasing it.
    ///
    /// Resource sizes use the graph's compiled swapchain extent. Recording
    /// cannot change it; resize requires preparing a new graph so the frozen
    /// memory/alias plan and physical allocation sizes remain consistent.
    #[must_use]
    pub fn record<D: RenderDevice>(self, device: &D) -> RecordedGraph {
        #[cfg(feature = "parallel")]
        {
            self.record_parallel(device)
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.record_sequential(device)
        }
    }

    /// Records on the calling thread, including when `parallel` is enabled.
    ///
    /// Shares realization and command assembly with [`Self::record`]; it never
    /// compiles again or submits. The same extent and retirement contracts apply.
    #[must_use]
    pub fn record_sequential<D: RenderDevice>(self, device: &D) -> RecordedGraph {
        self.record_with(device, RenderGraph::record_commands_sequential)
    }

    #[cfg(feature = "parallel")]
    pub(crate) fn record_parallel<D: RenderDevice>(self, device: &D) -> RecordedGraph {
        self.record_with(device, RenderGraph::record_commands_parallel)
    }

    fn record_with<D: RenderDevice>(
        self,
        device: &D,
        record_commands: fn(&mut RenderGraph, &ExecutionPlan) -> CommandBuffer,
    ) -> RecordedGraph {
        // Declare before `graph`: on callback unwind, remaining graph captures
        // must drop before this guard releases external owners they may use.
        let mut allocations;
        let PreparedGraph { mut graph, plan } = self;
        allocations = graph.realize(&plan, device);
        // Realization errors leave pins in `graph`, which drops them last after
        // its captures. Move them only now, while the recording guard is armed.
        allocations.allocations.external_owners =
            ManuallyDrop::new(core::mem::take(&mut graph.external_owners));
        let commands = record_commands(&mut graph, &plan);
        // Dropping unexecuted, culled captures may also panic. Do it before
        // handing off allocations, while the unsubmitted cleanup guard is armed.
        drop(graph);
        RecordedGraph {
            commands,
            plan,
            allocations: allocations.into_allocations(),
        }
    }
}

/// A whole-resource default view descriptor matching a texture's dimension.
fn default_view_descriptor(dimension: TextureDimension) -> TextureViewDescriptor {
    let view_dimension = match dimension {
        TextureDimension::D1 => TextureViewDimension::D1,
        TextureDimension::D2 => TextureViewDimension::D2,
        TextureDimension::D3 => TextureViewDimension::D3,
    };
    TextureViewDescriptor {
        dimension: view_dimension,
        ..TextureViewDescriptor::default()
    }
}

#[cfg(test)]
#[path = "../text/unit/external_owners.rs"]
mod external_owner_tests;
