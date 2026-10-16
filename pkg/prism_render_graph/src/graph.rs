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
use alloc::vec;
use alloc::vec::Vec;

use prism_render_driver::{
    BufferId, BufferUsages, ColorAttachment, CommandBuffer, CommandEncoder, DepthOperations,
    DepthStencilAttachment, Extent3d, RenderDevice, RenderPassDescriptor, RenderQueue,
    TextureDimension, TextureUsages, TextureViewDescriptor, TextureViewDimension,
};

use crate::blackboard::Blackboard;
use crate::builder::{ExecuteContext, PassBuilder};
use crate::capacity::GraphLimits;
use crate::compile;
use crate::desc::{BufferDesc, TextureDesc};
use crate::handle::{BufferHandle, ResourceIndex, TextureHandle};
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

/// The move-only owner of physical allocations created by one graph recording.
///
/// Imported resources are never included. Indices retain the texture/buffer
/// namespaces of the compiled plan, including synchronization metadata.
///
/// # Lifetime contract
///
/// Keep this owner until every submission using its resources has completed,
/// then call [`Self::release`] on the creating device. For a cancelled frame,
/// first discard all commands that could still reference these allocations.
/// Dropping this data-only owner does **not** destroy GPU resources: it cannot
/// observe external submission/completion, and an early destroy would invalidate
/// in-flight commands. Omitting explicit release leaks until device teardown.
#[derive(Debug, Default)]
#[must_use = "retain until GPU completion, then explicitly release on the creating device"]
pub struct GraphAllocations {
    textures: Vec<(ResourceIndex, Lifetime, RealizedTexture)>,
    buffers: Vec<(ResourceIndex, Lifetime, BufferId)>,
}

impl GraphAllocations {
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

    /// Whether every graph-created allocation has been released or transferred.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.textures.is_empty() && self.buffers.is_empty()
    }

    /// Releases completed frame-local allocations, retaining persistent ones.
    ///
    /// The caller must establish completion on the actual submitting queue and
    /// use the creating device. Persistent allocations remain owned here so a
    /// history owner can reimport them in later frames and eventually release
    /// them. This method does not submit, wait, or infer completion.
    pub fn release_transients<D: RenderDevice>(&mut self, device: &D) {
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
    /// resources are never destroyed. Views are destroyed before their textures.
    pub fn release<D: RenderDevice>(self, device: &D) {
        for (_, _, realized) in &self.textures {
            device.destroy_texture_view(realized.default_view);
        }
        for (_, _, realized) in self.textures {
            device.destroy_texture(realized.texture);
        }
        for (_, _, buffer) in self.buffers {
            device.destroy_buffer(buffer);
        }
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

    /// Physical graph-owned resources retained by this packet.
    #[must_use]
    pub fn allocations(&self) -> &GraphAllocations {
        &self.allocations
    }

    /// Transfers commands, plan and allocation ownership to an external runtime.
    ///
    /// Native encoding/submission may consume `commands`; keep `allocations`
    /// alive in the pending/submitted batch until its real completion signal.
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
/// [`execute`]. The graph is `unsafe`-free and holds no GPU objects until
/// execution realizes them through the [`RenderDevice`] trait.
///
/// [`compile`]: RenderGraph::compile
/// [`execute`]: RenderGraph::execute
pub struct RenderGraph {
    textures: Vec<TextureResource>,
    buffers: Vec<BufferResource>,
    passes: Vec<PassNode>,
    blackboard: Blackboard,
    swapchain: Extent3d,
    limits: GraphLimits,
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
            passes: Vec::new(),
            blackboard: Blackboard::new(),
            swapchain,
            limits,
        }
    }

    /// The swapchain extent screen-relative sizes resolve against.
    #[must_use]
    pub fn swapchain(&self) -> Extent3d {
        self.swapchain
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
        use crate::access::{BufferUse, TextureUse};
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
        for (pass_index, pass) in passes.iter().enumerate() {
            GraphLimits::check_count("accesses", pass.accesses.len(), limits.accesses_per_pass)?;
            for access in &pass.accesses {
                if access.resource.0 as usize >= resources.len() {
                    return Err(CompileError::InvalidResource {
                        pass: PassId(pass_index as u32),
                        resource: access.resource,
                    });
                }
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
            let uses_texture = passes.iter().any(|pass| {
                pass.accesses.iter().any(|access| {
                    access.resource.0 as usize == index
                        && matches!(
                            access.kind,
                            AccessKind::SampledRead
                                | AccessKind::ColorAttachment
                                | AccessKind::DepthAttachment
                        )
                })
            });
            let lifetime = match resource.lifetime {
                ResourceLifetime::Transient => Lifetime::Transient,
                ResourceLifetime::Imported => Lifetime::Imported,
                ResourceLifetime::Persistent => Lifetime::Persistent,
            };
            if uses_texture {
                let depth = passes.iter().any(|pass| {
                    pass.accesses.iter().any(|access| {
                        access.resource.0 as usize == index
                            && access.kind == AccessKind::DepthAttachment
                    })
                });
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
                    if pass.accesses.iter().any(|access| {
                        matches!(
                            access.kind,
                            AccessKind::ColorAttachment | AccessKind::DepthAttachment
                        )
                    }) {
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
                                let usage = match access.kind {
                                    AccessKind::SampledRead => TextureUse::Sampled,
                                    AccessKind::StorageRead | AccessKind::IndirectRead => {
                                        TextureUse::StorageRead
                                    }
                                    AccessKind::StorageWrite => TextureUse::StorageWrite,
                                    AccessKind::ColorAttachment => TextureUse::ColorAttachment,
                                    AccessKind::DepthAttachment => TextureUse::DepthAttachment,
                                    AccessKind::TransferRead => TextureUse::CopySrc,
                                    AccessKind::TransferWrite => TextureUse::CopyDst,
                                };
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
                                let usage = match access.kind {
                                    AccessKind::SampledRead | AccessKind::StorageRead => {
                                        BufferUse::StorageRead
                                    }
                                    AccessKind::StorageWrite => BufferUse::StorageWrite,
                                    AccessKind::IndirectRead => BufferUse::Indirect,
                                    AccessKind::TransferRead => BufferUse::CopySrc,
                                    AccessKind::TransferWrite => BufferUse::CopyDst,
                                    AccessKind::ColorAttachment | AccessKind::DepthAttachment => {
                                        unreachable!(
                                            "attachment declarations are lowered as textures"
                                        )
                                    }
                                };
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
        Ok(recorded)
    }

    /// The reference recording path, shared by record-only and submit APIs.
    fn record_commands_sequential(&mut self, plan: &ExecutionPlan) -> CommandBuffer {
        let mut encoder = CommandEncoder::new();
        for (pos, &pidx) in plan.order.iter().enumerate() {
            let kind = self.passes[pidx].kind;
            // Take the move-only closure before the shared borrows below.
            let exec = self.passes[pidx].execute.take();
            let recorded = record::record_pass(exec, kind, &self.textures, &self.buffers);
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

        let recorded = parallel::record(&mut units, &self.textures, &self.buffers);

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
    /// realized at import and are skipped. Returns ownership of exactly the
    /// allocations created here, including persistent history resources.
    fn realize<D: RenderDevice>(&mut self, plan: &ExecutionPlan, device: &D) -> GraphAllocations {
        let mut allocations = GraphAllocations::default();
        for i in 0..self.textures.len() {
            if !plan.used_textures[i] || self.textures[i].realized.is_some() {
                continue;
            }
            let descriptor = self.textures[i]
                .desc
                .lower(self.swapchain, self.textures[i].inferred_usage);
            let texture = device.create_texture(&descriptor);
            let view = device.create_texture_view(
                texture,
                &default_view_descriptor(self.textures[i].desc.dimension),
            );
            let realized = RealizedTexture {
                texture,
                default_view: view,
            };
            self.textures[i].realized = Some(realized);
            allocations.textures.push((
                ResourceIndex(i as u32),
                self.textures[i].lifetime,
                realized,
            ));
        }
        for i in 0..self.buffers.len() {
            if !plan.used_buffers[i] || self.buffers[i].realized.is_some() {
                continue;
            }
            let descriptor = self.buffers[i].desc.lower(self.buffers[i].inferred_usage);
            let buffer = device.create_buffer(&descriptor);
            self.buffers[i].realized = Some(buffer);
            allocations
                .buffers
                .push((ResourceIndex(i as u32), self.buffers[i].lifetime, buffer));
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
                depth: Some(DepthOperations {
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
    /// The caller retains responsibility for external completion and retirement.
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
        mut self,
        device: &D,
        record_commands: fn(&mut RenderGraph, &ExecutionPlan) -> CommandBuffer,
    ) -> RecordedGraph {
        let allocations = self.graph.realize(&self.plan, device);
        let commands = record_commands(&mut self.graph, &self.plan);
        RecordedGraph {
            commands,
            plan: self.plan,
            allocations,
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
