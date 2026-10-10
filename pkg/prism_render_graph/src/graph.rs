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
//! [`ExecutionPlan`] without touching the GPU; [`execute`](RenderGraph::execute)
//! realizes the surviving transients, replays each pass's execute closure into
//! a driver [`CommandBuffer`], and submits it.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use prism_render_driver::{
    BufferUsages, ColorAttachment, CommandEncoder, DepthOperations, DepthStencilAttachment,
    Extent3d, RenderDevice, RenderPassDescriptor, RenderQueue, TextureDimension, TextureUsages,
    TextureViewDescriptor, TextureViewDimension,
};

use crate::blackboard::Blackboard;
use crate::builder::{ExecuteContext, PassBuilder};
use crate::compile;
use crate::desc::{BufferDesc, TextureDesc};
use crate::handle::{BufferHandle, ResourceIndex, TextureHandle};
use crate::pass::{BoxedExecute, PassFlags, PassKind, PassNode};
use crate::plan::{CompileError, ExecutionPlan, RasterAttachments};
use crate::resource::{
    BufferResource, ImportedBuffer, ImportedTexture, Lifetime, RealizedTexture, TextureResource,
};

/// The move-only closure a pass runs during execution to record its commands.
///
/// Re-exported for callers that store execute closures in their own types; most
/// code produces one implicitly by returning a closure from a pass's setup.
pub type ExecuteFn = BoxedExecute;

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
}

impl RenderGraph {
    /// Creates an empty graph whose screen-relative sizes resolve against the
    /// given `swapchain` extent.
    #[must_use]
    pub fn new(swapchain: Extent3d) -> Self {
        Self {
            textures: Vec::new(),
            buffers: Vec::new(),
            passes: Vec::new(),
            blackboard: Blackboard::new(),
            swapchain,
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
    pub fn import_texture(
        &mut self,
        name: impl Into<String>,
        desc: TextureDesc,
        imported: ImportedTexture,
    ) -> TextureHandle {
        let index = ResourceIndex(self.textures.len() as u32);
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
        TextureHandle::new(index, 0)
    }

    /// Imports an externally-owned buffer. See [`import_texture`] for the
    /// ownership contract.
    ///
    /// [`import_texture`]: RenderGraph::import_texture
    pub fn import_buffer(
        &mut self,
        name: impl Into<String>,
        desc: BufferDesc,
        imported: ImportedBuffer,
    ) -> BufferHandle {
        let index = ResourceIndex(self.buffers.len() as u32);
        self.buffers.push(BufferResource {
            name: name.into(),
            desc,
            lifetime: Lifetime::Imported,
            imported: Some(imported),
            inferred_usage: BufferUsages::NONE,
            version: 0,
            realized: Some(imported.buffer),
        });
        BufferHandle::new(index, 0)
    }

    /// Declares a raster pass: binds color/depth attachments and records draws.
    pub fn add_raster_pass<Setup, Exec>(
        &mut self,
        name: impl Into<String>,
        flags: PassFlags,
        setup: Setup,
    ) where
        Setup: FnOnce(&mut PassBuilder<'_>) -> Exec,
        Exec: FnOnce(&mut ExecuteContext<'_>) + 'static,
    {
        self.add_pass(name, PassKind::Raster, flags, setup);
    }

    /// Declares a compute pass: records dispatches.
    pub fn add_compute_pass<Setup, Exec>(
        &mut self,
        name: impl Into<String>,
        flags: PassFlags,
        setup: Setup,
    ) where
        Setup: FnOnce(&mut PassBuilder<'_>) -> Exec,
        Exec: FnOnce(&mut ExecuteContext<'_>) + 'static,
    {
        self.add_pass(name, PassKind::Compute, flags, setup);
    }

    /// Declares a transfer pass: copies/blits, tracked for synchronization.
    pub fn add_transfer_pass<Setup, Exec>(
        &mut self,
        name: impl Into<String>,
        flags: PassFlags,
        setup: Setup,
    ) where
        Setup: FnOnce(&mut PassBuilder<'_>) -> Exec,
        Exec: FnOnce(&mut ExecuteContext<'_>) + 'static,
    {
        self.add_pass(name, PassKind::Transfer, flags, setup);
    }

    /// Declares a present pass: hands an imported swapchain image to the
    /// surface. A present pass is always a culling root.
    pub fn add_present_pass<Setup, Exec>(
        &mut self,
        name: impl Into<String>,
        flags: PassFlags,
        setup: Setup,
    ) where
        Setup: FnOnce(&mut PassBuilder<'_>) -> Exec,
        Exec: FnOnce(&mut ExecuteContext<'_>) + 'static,
    {
        self.add_pass(name, PassKind::Present, flags, setup);
    }

    /// Shared pass-construction core: runs `setup` against a fresh
    /// [`PassBuilder`], captures the recorded accesses and the returned execute
    /// closure, and appends the resulting [`PassNode`].
    fn add_pass<Setup, Exec>(
        &mut self,
        name: impl Into<String>,
        kind: PassKind,
        flags: PassFlags,
        setup: Setup,
    ) where
        Setup: FnOnce(&mut PassBuilder<'_>) -> Exec,
        Exec: FnOnce(&mut ExecuteContext<'_>) + 'static,
    {
        let mut builder = PassBuilder::new(
            &mut self.textures,
            &mut self.buffers,
            &mut self.blackboard,
            kind,
        );
        let exec = setup(&mut builder);
        let (texture_accesses, buffer_accesses) = builder.finish();
        self.passes.push(PassNode {
            name: name.into(),
            kind,
            flags,
            texture_accesses,
            buffer_accesses,
            execute: Some(Box::new(exec)),
        });
    }

    /// Compiles the declared frame into an immutable [`ExecutionPlan`] without
    /// touching the GPU. Fails only on a dependency cycle.
    ///
    /// # Errors
    /// Returns [`CompileError::Cycle`] when the derived dependency graph cannot
    /// be topologically ordered.
    pub fn compile(&self) -> Result<ExecutionPlan, CompileError> {
        compile::compile(&self.passes, &self.textures, &self.buffers, self.swapchain)
    }

    /// Compiles, realizes transients, replays every surviving pass into a
    /// driver [`CommandBuffer`], and submits it on `queue`.
    ///
    /// Consumes the graph: a frame's execute closures are move-only and run
    /// exactly once.
    ///
    /// # Errors
    /// Returns [`CompileError`] when compilation fails; no GPU work is issued in
    /// that case.
    pub fn execute<D: RenderDevice, Q: RenderQueue>(
        mut self,
        device: &D,
        queue: &Q,
    ) -> Result<(), CompileError> {
        let plan = self.compile()?;
        self.realize(&plan, device);

        let mut encoder = CommandEncoder::new();
        for (pos, &pidx) in plan.order.iter().enumerate() {
            let kind = self.passes[pidx].kind;
            // Take the move-only closure before the shared borrows below.
            let exec = self.passes[pidx].execute.take();

            let mut ctx = ExecuteContext::new(&self.textures, &self.buffers, kind);
            if let Some(exec) = exec {
                exec(&mut ctx);
            }

            match kind {
                PassKind::Raster => {
                    let descriptor = self.raster_descriptor(
                        plan.attachments[pos].as_ref(),
                        self.passes[pidx].name.as_str(),
                    );
                    encoder.push_render_pass(descriptor, ctx.take_render_commands());
                }
                PassKind::Compute => {
                    let label = Some(self.passes[pidx].name.clone());
                    encoder.push_compute_pass(label, ctx.take_compute_commands());
                }
                // Transfer/present contribute only state tracking until the
                // driver gains copy/present encoder commands.
                PassKind::Transfer | PassKind::Present => {}
            }
        }

        queue.submit(&[encoder.finish()]);
        Ok(())
    }

    /// Creates the driver objects backing every used, not-yet-realized
    /// transient or persistent resource. Imported resources are already
    /// realized at import and are skipped.
    fn realize<D: RenderDevice>(&mut self, plan: &ExecutionPlan, device: &D) {
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
            self.textures[i].realized = Some(RealizedTexture {
                texture,
                default_view: view,
            });
        }
        for i in 0..self.buffers.len() {
            if !plan.used_buffers[i] || self.buffers[i].realized.is_some() {
                continue;
            }
            let descriptor = self.buffers[i].desc.lower(self.buffers[i].inferred_usage);
            self.buffers[i].realized = Some(device.create_buffer(&descriptor));
        }
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
