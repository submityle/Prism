//! Native shared-device indexed geometry and GPU object-ID debug composition.
//!
//! This module is enabled by the host's `desktop` feature/module wiring. It
//! creates no device, surface, window, material system, or reflected models.
//! Register the selectable models with `EditorController` first, then supply
//! explicit full-identity mappings through `DesktopSceneObject`.
//!
//! `encode_preview` appends an ID/depth pass and a load-preserving debug-color
//! pass to the host encoder. Submit that encoder normally. `request_pick` uses
//! its own small encoder on the same queue, so asynchronous picking needs no
//! after-submit callback from the native host. Pointer positions are viewport-
//! local logical pixels. The newest click supersedes older clicks; camera,
//! geometry, bounds, target and controller-revision changes reject late work.

use alloc::{
    collections::{BTreeMap, BTreeSet},
    string::String,
    vec::Vec,
};

#[path = "desktop_gizmo.rs"]
mod gizmo;
pub use gizmo::{GizmoFrame, SceneTransformAdapter};

use prism_editor_core::SelectionItem;
use prism_editor_panels::OutlinerId;
use prism_editor_ui::{EditorAction, SelectionMode};
use prism_editor_viewport::{
    BoxSelectionCoordinator, EncodedPickId, GpuBoxSelectionLimits, GpuPickingBackend,
    GpuPickingConfig, InFlightBoxSelectionReadback, InFlightPickReadback, PickCoordinator,
    PickOutcome, PickResult, PixelPoint, PixelRect, RenderTargetRegistry, RenderTargetState,
    SceneIdCamera, SceneIdDepth, SceneIdGeometryRenderer, SceneIdMatrix, SceneIdPass,
    SceneIdPassItem, SceneIdPassOperation, SceneObjectId, SelectionOutcome, Viewport,
    ViewportRegistry, SCENE_ID_IDENTITY_MATRIX,
};
use wgpu::{CommandEncoder, TextureView};

use crate::desktop::{
    DesktopError, DesktopGpu, DesktopRenderer, DesktopViewportBounds, DesktopWork,
};
use crate::desktop_ui::EditorDesktopUi;
use crate::EditorController;

pub(crate) const MAX_OBJECTS: usize = 4096;
pub(crate) const MAX_GEOMETRY_BYTES: u64 = 64 * 1024 * 1024;

/// An actual indexed mesh and its explicit controller selection mapping.
/// Multiple mesh/subobject identities may intentionally map to one selection.
#[derive(Clone, Debug)]
pub struct DesktopSceneObject {
    /// Complete 128-bit entity, 128-bit mesh and 64-bit subobject identity.
    pub id: SceneObjectId,
    /// Exact loaded controller identity, never inferred from a truncated token.
    pub selection: SelectionItem,
    /// Optional exact Outliner row for Add/Toggle/Extend semantics, even when
    /// collapsed. Without it, modifier picks search visible matching rows.
    pub outliner: Option<OutlinerId>,
    /// Object-space Float32x3 vertices.
    pub positions: Vec<[f32; 3]>,
    /// Uint32 triangle-list indices.
    pub indices: Vec<u32>,
    /// Column-major object-to-world matrix.
    pub model: SceneIdMatrix,
    /// Whether this object draws and can be picked.
    pub visible: bool,
    /// Stable draw order for equal-depth tie breaking.
    pub sort_key: u64,
}

/// The engine-neutral scene payload produced for one validated document.
/// Geometry and identity remain explicit so the editor never guesses project
/// gameplay types from reflection metadata.
#[derive(Clone, Debug)]
pub struct DesktopSceneDocument {
    /// Complete indexed geometry and controller mappings for this document.
    pub objects: Vec<DesktopSceneObject>,
    /// World-to-clip matrix for the document's initial viewport camera.
    pub camera: SceneIdMatrix,
    /// Depth convention matching `camera`.
    pub depth: SceneIdDepth,
}

/// Host-owned mapping from a validated editor document to render descriptors.
/// Implementations may inspect reflected roots, asset handles, or an external
/// runtime snapshot, but the editor application supplies no business schema.
pub trait SceneDocumentAdapter {
    /// Builds a complete candidate without mutating the active scene.
    fn build(
        &mut self,
        controller: &EditorController,
    ) -> Result<DesktopSceneDocument, DesktopError>;
}

impl DesktopSceneObject {
    /// Describes an identity-mapped indexed mesh with identity transform.
    #[must_use]
    pub fn new(
        id: SceneObjectId,
        selection: SelectionItem,
        positions: Vec<[f32; 3]>,
        indices: Vec<u32>,
    ) -> Self {
        Self {
            id,
            selection,
            outliner: None,
            positions,
            indices,
            model: SCENE_ID_IDENTITY_MATRIX,
            visible: true,
            sort_key: 0,
        }
    }
}

/// Observable adapter state for integration diagnostics and offscreen tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct DesktopSceneStatus {
    /// Current allocation, extent, camera generation and target generation.
    pub target: Option<RenderTargetState>,
    /// Current geometry/scene generation.
    pub scene_generation: u64,
    /// At most one GPU readback is retained (newest click wins).
    pub pending: bool,
    /// At most one GPU rectangle readback is retained.
    pub pending_box: bool,
    /// Number of submitted pick copies.
    pub submitted_picks: u64,
    /// Number of current hit/miss results accepted by the adapter.
    pub accepted_picks: u64,
    /// Number of superseded, cancelled, or stale results discarded.
    pub rejected_picks: u64,
    /// Number of preview bind groups created, including resize/reopen rebinding.
    pub preview_bindings: u64,
    /// Successful vertex/index buffer uploads over this adapter's lifetime.
    pub geometry_uploads: u64,
    /// Vertex/index payload bytes actually uploaded (not allocator capacity).
    pub geometry_uploaded_bytes: u64,
    /// Successful instance-transform uploads, including initial population.
    pub transform_uploads: u64,
    /// Geometry renderer allocations, including initialize/resume and reload.
    pub geometry_renderer_creations: u64,
}

struct PendingClick {
    readback: InFlightPickReadback,
    controller_revision: u64,
    mode: SelectionMode,
}

struct PendingBox {
    readback: InFlightBoxSelectionReadback,
    controller_revision: u64,
    mode: SelectionMode,
}

// Constructed only after a completed readback passes identity/revision checks.
// Keep this separate from polling: even borrowing the UI controller mutably
// invalidates its retained projection and cancels the active editing draft.
enum ReadySelection {
    Replace(SelectionItem),
    Outliner {
        node: OutlinerId,
        mode: SelectionMode,
    },
    Clear,
    Unchanged,
}

impl ReadySelection {
    fn apply(self, controller: &mut EditorController) -> Result<bool, DesktopError> {
        let outcome = match self {
            Self::Replace(selection) => controller.select(selection),
            Self::Outliner { node, mode } => {
                controller.execute(EditorAction::SelectOutliner { node, mode })
            }
            Self::Clear => controller.clear_selection(),
            Self::Unchanged => return Ok(false),
        };
        Ok(outcome.map_err(renderer_error)?.changed)
    }
}

enum ReadyBoxSelection {
    Items(Vec<SelectionItem>),
    Clear,
    Unchanged,
}

impl ReadyBoxSelection {
    fn apply(
        self,
        controller: &mut EditorController,
        mode: SelectionMode,
    ) -> Result<bool, DesktopError> {
        let outcome = match self {
            Self::Items(items) => controller.select_items(&items, mode),
            Self::Clear => controller.clear_selection(),
            Self::Unchanged => return Ok(false),
        };
        Ok(outcome.map_err(renderer_error)?.changed)
    }
}

struct GpuScene {
    gizmo: gizmo::GizmoPainter,
    renderer: SceneIdGeometryRenderer,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    binding: Option<(RenderTargetState, wgpu::BindGroup)>,
}

/// Shared-wgpu native adapter: opaque indexed geometry, real depth-tested IDs,
/// GPU-only object-ID debug colors, and asynchronous controller selection.
///
/// Default camera is identity (geometry in WebGPU clip coordinates). Supply
/// `set_camera` for world geometry; camera math/aspect policy is host-owned.
/// No full material rendering or automatic model-to-mesh extraction is claimed.
pub struct DesktopScene {
    objects: Vec<DesktopSceneObject>,
    object_indices: BTreeMap<SceneObjectId, usize>,
    scene: SceneIdPass,
    viewports: ViewportRegistry,
    viewport: Viewport,
    targets: RenderTargetRegistry,
    picks: PickCoordinator,
    pending: Option<PendingClick>,
    boxes: BoxSelectionCoordinator,
    pending_box: Option<PendingBox>,
    gpu: Option<GpuScene>,
    needs_upload: bool,
    geometry_dirty: BTreeSet<usize>,
    visibility_dirty: BTreeSet<usize>,
    matrix: SceneIdMatrix,
    depth: SceneIdDepth,
    bounds: Option<DesktopViewportBounds>,
    counters: DesktopSceneStatus,
    document_adapter: Option<Box<dyn SceneDocumentAdapter>>,
    pending_document: Option<DesktopSceneDocument>,
    transform_adapter: Option<Box<dyn SceneTransformAdapter>>,
    transform_revision: Option<u64>,
    transform_indices: BTreeMap<SelectionItem, Vec<usize>>,
    gizmo: gizmo::GizmoState,
}

impl DesktopScene {
    /// Validates CPU descriptors only; call `DesktopRenderer::initialize` on
    /// the host device before encoding. Empty scenes are valid and draw misses.
    pub fn new(objects: Vec<DesktopSceneObject>) -> Result<Self, DesktopError> {
        validate_objects(&objects)?;
        let mut viewports = ViewportRegistry::new();
        let viewport = viewports
            .open("Object-ID debug preview")
            .map_err(renderer_error)?;
        Ok(Self {
            object_indices: objects.iter().enumerate().map(|(i, o)| (o.id, i)).collect(),
            objects,
            scene: SceneIdPass::new(),
            viewports,
            viewport,
            targets: RenderTargetRegistry::new(),
            picks: PickCoordinator::new(),
            pending: None,
            boxes: BoxSelectionCoordinator::new(MAX_OBJECTS),
            pending_box: None,
            gpu: None,
            needs_upload: true,
            geometry_dirty: BTreeSet::new(),
            visibility_dirty: BTreeSet::new(),
            matrix: SCENE_ID_IDENTITY_MATRIX,
            depth: SceneIdDepth::Forward,
            bounds: None,
            counters: DesktopSceneStatus::default(),
            document_adapter: None,
            pending_document: None,
            transform_adapter: None,
            transform_revision: None,
            transform_indices: BTreeMap::new(),
            gizmo: gizmo::GizmoState::default(),
        })
    }

    /// Creates an empty scene whose contents are supplied by a host-owned
    /// document adapter before each successful document publication.
    pub fn new_with_document_adapter<A>(adapter: A) -> Result<Self, DesktopError>
    where
        A: SceneDocumentAdapter + 'static,
    {
        let mut scene = Self::new(Vec::new())?;
        scene.document_adapter = Some(Box::new(adapter));
        Ok(scene)
    }

    /// Returns the descriptors currently owned by this adapter.
    #[must_use]
    pub fn objects(&self) -> &[DesktopSceneObject] {
        &self.objects
    }

    pub(crate) fn object(&self, id: SceneObjectId) -> Option<&DesktopSceneObject> {
        self.object_indices
            .get(&id)
            .map(|index| &self.objects[*index])
    }

    /// Applies prevalidated authoring changes without cloning unrelated meshes.
    /// Replacements keep stable identity bindings; property-only batches never
    /// visit vertex/index data. All fallible CPU checks precede publication.
    pub(crate) fn update_document_objects(
        &mut self,
        replacements: Vec<DesktopSceneObject>,
        models: &[(SceneObjectId, SceneIdMatrix)],
        visibility: &[(SceneObjectId, bool)],
    ) -> Result<(), DesktopError> {
        if self.gizmo.drag_active() {
            return Err(message("cancel active gizmo before synchronizing scene"));
        }
        validate_objects(&replacements)?;
        let mut replacement_indices = Vec::with_capacity(replacements.len());
        let mut bytes = 0u64;
        if !replacements.is_empty() {
            bytes = self.objects.iter().map(geometry_bytes).sum();
        }
        for object in &replacements {
            let index = *self
                .object_indices
                .get(&object.id)
                .ok_or_else(|| message("unknown replacement scene identity"))?;
            let old = &self.objects[index];
            if object.selection != old.selection || object.outliner != old.outliner {
                return Err(message("replacement changes document identity bindings"));
            }
            bytes = bytes
                .checked_sub(geometry_bytes(old))
                .and_then(|value| value.checked_add(geometry_bytes(object)))
                .ok_or_else(|| message("geometry byte count overflow"))?;
            replacement_indices.push(index);
        }
        if bytes > MAX_GEOMETRY_BYTES {
            return Err(message("desktop scene exceeds 64 MiB of vertex/index data"));
        }
        for (id, model) in models {
            validate_matrix(model)?;
            if !self.object_indices.contains_key(id) {
                return Err(message("unbound scene transform"));
            }
        }
        let mut visibility_changes = Vec::with_capacity(visibility.len());
        for (id, visible) in visibility {
            let index = *self
                .object_indices
                .get(id)
                .ok_or_else(|| message("unbound scene visibility"))?;
            if self.objects[index].visible != *visible {
                visibility_changes.push((index, *visible));
            }
        }
        self.scene
            .generation()
            .checked_add(models.len() as u64 + 1)
            .ok_or_else(|| message("scene generation exhausted"))?;
        self.apply_models(models)?;
        if !replacements.is_empty() || !visibility_changes.is_empty() {
            self.scene.invalidate_content().map_err(renderer_error)?;
            self.cancel_pending();
            for (index, object) in replacement_indices.into_iter().zip(replacements) {
                self.objects[index] = object;
                self.geometry_dirty.insert(index);
            }
            for (index, visible) in visibility_changes {
                self.objects[index].visible = visible;
                self.visibility_dirty.insert(index);
            }
        }
        Ok(())
    }

    /// Replaces descriptors after complete CPU validation. Invalid inputs leave
    /// current descriptors and pending picks intact. Valid changes immediately
    /// invalidate old requests; resources upload at the next encode. Identity
    /// tokens remain stable, including remove/readd during this scene lifetime.
    pub fn replace_objects(
        &mut self,
        objects: Vec<DesktopSceneObject>,
    ) -> Result<(), DesktopError> {
        if self.gizmo.drag_active() {
            return Err(message("cancel active gizmo before replacing scene"));
        }
        validate_objects(&objects)?;
        self.scene.invalidate_content().map_err(renderer_error)?;
        self.cancel_pending();
        self.object_indices = objects.iter().enumerate().map(|(i, o)| (o.id, i)).collect();
        self.objects = objects;
        self.geometry_dirty.clear();
        self.visibility_dirty.clear();
        self.transform_revision = None;
        self.needs_upload = true;
        Ok(())
    }

    /// Updates one object's model transform without rebuilding unrelated mesh
    /// buffers. When a native device is live, only that instance bind group is
    /// replaced; otherwise the validated CPU change is deferred to resume.
    /// Any pending readback is cancelled because the old pixels no longer
    /// describe the current scene.
    pub fn update_object_model(
        &mut self,
        id: SceneObjectId,
        model: SceneIdMatrix,
    ) -> Result<bool, DesktopError> {
        if self.gizmo.drag_active() {
            return Err(message(
                "cancel active gizmo before external transform updates",
            ));
        }
        validate_matrix(&model)?;
        let index = self
            .object_indices
            .get(&id)
            .copied()
            .ok_or_else(|| message("unknown desktop scene object"))?;
        if self.objects[index].model == model {
            return Ok(false);
        }

        if let Some(gpu) = self.gpu.as_mut().filter(|_| !self.needs_upload) {
            gpu.renderer
                .set_transform(&mut self.scene, index as u64, model)
                .map_err(renderer_error)?;
            self.counters.transform_uploads = self.counters.transform_uploads.saturating_add(1);
        } else {
            self.needs_upload = true;
        }
        self.cancel_pending();
        self.objects[index].model = model;
        Ok(true)
    }

    /// Returns the current model transform for an exact scene identity.
    #[must_use]
    pub fn object_model(&self, id: SceneObjectId) -> Option<SceneIdMatrix> {
        self.object_indices.get(&id).map(|i| self.objects[*i].model)
    }

    /// Sets a column-major world-to-clip matrix and matching `[0,1]` depth
    /// convention. Invalidates camera-correlated requests. On resize the host
    /// must supply a matrix with the desired new aspect; no camera is inferred.
    pub fn set_camera(
        &mut self,
        view_projection: SceneIdMatrix,
        depth: SceneIdDepth,
    ) -> Result<(), DesktopError> {
        if self.gizmo.drag_active() {
            return Err(message("cancel active gizmo before changing camera"));
        }
        validate_matrix(&view_projection)?;
        if self.matrix != view_projection || self.depth != depth {
            self.advance_camera()?;
            self.matrix = view_projection;
            self.depth = depth;
            self.cancel_pending();
        }
        Ok(())
    }

    /// Returns bounded pick/lifecycle diagnostics; no GPU synchronization occurs.
    #[must_use]
    pub fn status(&self) -> DesktopSceneStatus {
        DesktopSceneStatus {
            target: self.targets.state(self.viewport.id()),
            scene_generation: self.scene.generation(),
            pending: self.pending.is_some(),
            pending_box: self.pending_box.is_some(),
            ..self.counters
        }
    }

    /// Fences all GPU work associated with the previously loaded document.
    ///
    /// The host calls this only after the controller has successfully replaced
    /// its document. CPU mesh descriptors and the camera remain owned by this
    /// adapter, while the old async readback, coordinator correlation, ID
    /// target, and debug bind group are discarded. The next encode uploads the
    /// current transforms again and publishes a fresh target/frame generation.
    pub fn document_reloaded(&mut self) {
        self.cancel_pending();
        self.geometry_dirty.clear();
        self.visibility_dirty.clear();
        self.transform_revision = None;

        // These counters are monotonic fences. Exhaustion is unreachable for a
        // live adapter; retain the last valid state if an artificial boundary
        // test forces it rather than panicking from an event-loop callback.
        let _ = self.scene.invalidate_content();
        let _ = self.advance_camera();

        if self.targets.state(self.viewport.id()).is_some() {
            let _ = self.targets.close(self.viewport.id());
        }
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.binding = None;
        }
        self.needs_upload = true;
    }

    fn prepare_document_candidate(
        &mut self,
        controller: &EditorController,
    ) -> Result<(), DesktopError> {
        if self.gizmo.drag_active() || controller.has_active_edit() {
            return Err(message("cancel active gizmo before loading document"));
        }
        self.pending_document = None;
        self.preflight_document_reload()?;
        let Some(adapter) = self.document_adapter.as_mut() else {
            return Ok(());
        };
        let candidate = adapter.build(controller)?;
        self.stage_document(controller, candidate)
    }

    pub(crate) fn stage_document(
        &mut self,
        controller: &EditorController,
        candidate: DesktopSceneDocument,
    ) -> Result<(), DesktopError> {
        self.pending_document = None;
        if self.gizmo.drag_active() || controller.has_active_edit() {
            return Err(message("cancel active gizmo before loading document"));
        }
        self.preflight_document_reload()?;
        validate_matrix(&candidate.camera)?;
        validate_objects(&candidate.objects)?;
        for object in &candidate.objects {
            // Reject identity drift before publication; a late pick must not be
            // the first place that discovers a scene/document mismatch.
            controller
                .history_lengths(&object.selection)
                .map_err(renderer_error)?;
            if let Some(node) = object.outliner {
                let matches = controller
                    .outliner()
                    .node(node)
                    .and_then(|row| row.selection())
                    == Some(&object.selection);
                if !matches {
                    return Err(message(
                        "scene object Outliner identity does not match its document selection",
                    ));
                }
            }
        }
        self.pending_document = Some(candidate);
        Ok(())
    }

    fn preflight_document_reload(&self) -> Result<(), DesktopError> {
        self.scene
            .generation()
            .checked_add(1)
            .ok_or_else(|| message("scene generation exhausted"))?;
        let viewport = self
            .viewports
            .get(self.viewport.id())
            .ok_or_else(|| message("missing viewport"))?;
        if viewport.generation() != self.viewport.generation() {
            return Err(message("desktop viewport state is out of sync"));
        }
        let generation = viewport
            .generation()
            .checked_add(1)
            .ok_or_else(|| message("viewport generation exhausted"))?;
        if self
            .targets
            .state(self.viewport.id())
            .is_some_and(|target| generation < target.viewport_generation)
        {
            return Err(message(
                "render target generation is ahead of the desktop viewport",
            ));
        }
        Ok(())
    }

    fn discard_document_candidate(&mut self) {
        self.pending_document = None;
    }

    fn commit_document_candidate(&mut self) {
        let Some(candidate) = self.pending_document.take() else {
            self.document_reloaded();
            return;
        };
        self.object_indices = candidate
            .objects
            .iter()
            .enumerate()
            .map(|(i, o)| (o.id, i))
            .collect();
        self.objects = candidate.objects;
        self.transform_revision = None;
        self.matrix = candidate.camera;
        self.depth = candidate.depth;
        self.document_reloaded();
    }

    /// Encodes real ID geometry followed by GPU debug-color composition into
    /// the supplied surface/offscreen view. Preserves pixels outside the bounds
    /// using `Load` and scissor. The host submits the encoder, and must pass the
    /// same shared GPU used for initialization and `gpu.format` output views.
    /// Controller selection only controls debug highlighting; it is not mutated.
    pub fn encode_preview(
        &mut self,
        gpu: &DesktopGpu,
        controller: &EditorController,
        encoder: &mut CommandEncoder,
        target: &TextureView,
        bounds: DesktopViewportBounds,
    ) -> Result<(), DesktopError> {
        self.sync_transforms(controller)?;
        let Some(region) = preview_region(&bounds)? else {
            self.cancel_pending();
            self.bounds = None;
            return Ok(());
        };
        if self.gpu.is_none() {
            return Err(message("DesktopScene must be initialized before encode"));
        }
        let bounds_changed = self
            .bounds
            .as_ref()
            .is_none_or(|old| !same_bounds(old, &bounds));
        if bounds_changed {
            self.advance_camera()?;
            self.cancel_pending();
        }
        if self.targets.state(self.viewport.id()).is_none() {
            let backend =
                GpuPickingBackend::new(&gpu.device, &gpu.adapter, &gpu.queue, region.config)
                    .map_err(renderer_error)?;
            self.targets
                .open(self.viewport.id(), self.viewport.generation(), backend)
                .map_err(renderer_error)?;
        } else {
            self.targets
                .resize(
                    self.viewport.id(),
                    self.viewport.generation(),
                    region.config,
                )
                .map_err(renderer_error)?;
        }
        if self.needs_upload || !self.geometry_dirty.is_empty() || !self.visibility_dirty.is_empty()
        {
            self.upload(gpu)?;
        }
        let native = self
            .gpu
            .as_mut()
            .ok_or_else(|| message("missing GPU scene"))?;
        let viewport_target = self
            .targets
            .get(self.viewport.id())
            .ok_or_else(|| message("missing viewport target"))?;
        let state = viewport_target.state();
        // Camera-only generation changes need no rebind; allocation changes do.
        if native.binding.as_ref().is_none_or(|(previous, _)| {
            previous.handle != state.handle || previous.target_generation != state.target_generation
        }) {
            let binding = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("desktop object-ID debug binding"),
                layout: &native.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            viewport_target.backend().target_view(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: native.uniform.as_entire_binding(),
                    },
                ],
            });
            native.binding = Some((state, binding));
            self.counters.preview_bindings = self.counters.preview_bindings.saturating_add(1);
        }
        let selected = controller
            .selection()
            .primary()
            .and_then(|item| {
                self.objects
                    .iter()
                    .find(|object| object.visible && &object.selection == item)
            })
            .and_then(|object| {
                self.scene
                    .entries()
                    .find(|entry| entry.item().id == object.id)
            })
            .map_or(EncodedPickId::none(), |entry| entry.encoded_id());
        let mut bytes = Vec::with_capacity(32);
        for value in region.origin_extent {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in selected.channels() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        gpu.queue.write_buffer(&native.uniform, 0, &bytes);
        let frame = viewport_target.begin_frame(&self.scene);
        native
            .renderer
            .render(
                encoder,
                &frame,
                &self.scene,
                &SceneIdCamera {
                    target: state,
                    view_projection: self.matrix,
                    depth: self.depth,
                },
            )
            .map_err(renderer_error)?;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("desktop GPU object-ID debug composition"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&native.pipeline);
            pass.set_scissor_rect(
                region.scissor[0],
                region.scissor[1],
                region.scissor[2],
                region.scissor[3],
            );
            let (_, binding) = native
                .binding
                .as_ref()
                .ok_or_else(|| message("missing preview binding"))?;
            pass.set_bind_group(0, binding, &[]);
            pass.draw(0..3, 0..1);
        }
        self.bounds = Some(bounds);
        self.paint_gizmo(gpu, controller, encoder, target, bounds)?;
        Ok(())
    }

    /// Queues at most one asynchronous pick using viewport-local logical pixels.
    /// Returns false for outside/hidden/unencoded viewports. Requires objects
    /// already registered in the controller; no reflected model is fabricated.
    /// This rerenders geometry before copying, avoiding reads of an old ID image.
    pub fn request_pick(
        &mut self,
        gpu: &DesktopGpu,
        controller: &EditorController,
        position: [f32; 2],
        mode: SelectionMode,
    ) -> Result<bool, DesktopError> {
        if position.iter().any(|value| !value.is_finite()) {
            return Err(message("pick coordinates must be finite"));
        }
        if self.needs_upload || !self.geometry_dirty.is_empty() || !self.visibility_dirty.is_empty()
        {
            return Ok(false);
        }
        let Some(bounds) = &self.bounds else {
            return Ok(false);
        };
        let Some(viewport_target) = self.targets.get(self.viewport.id()) else {
            return Ok(false);
        };
        let Some(pixel) = local_pixel(position, bounds, viewport_target.state().config) else {
            return Ok(false);
        };
        self.cancel_pending();
        let viewport_target = self
            .targets
            .get(self.viewport.id())
            .ok_or_else(|| message("missing target"))?;
        let state = viewport_target.state();
        let request = self
            .picks
            .submit(&self.viewport, state, &self.scene, pixel)
            .map_err(renderer_error)?;
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("desktop asynchronous geometry pick"),
            });
        let encoded = (|| {
            let native = self
                .gpu
                .as_ref()
                .ok_or_else(|| message("missing GPU scene"))?;
            let frame = viewport_target.begin_frame(&self.scene);
            native
                .renderer
                .render(
                    &mut encoder,
                    &frame,
                    &self.scene,
                    &SceneIdCamera {
                        target: state,
                        view_projection: self.matrix,
                        depth: self.depth,
                    },
                )
                .map_err(renderer_error)?;
            frame
                .encode_readback(&mut encoder, request)
                .map_err(renderer_error)
        })();
        let pending = match encoded {
            Ok(pending) => pending,
            Err(error) => {
                self.picks.cancel_viewport(self.viewport.id());
                return Err(error);
            }
        };
        let readback = viewport_target.backend().submit_readback(encoder, pending);
        self.pending = Some(PendingClick {
            readback,
            controller_revision: controller.revision(),
            mode,
        });
        self.counters.submitted_picks = self.counters.submitted_picks.saturating_add(1);
        Ok(true)
    }

    /// Queues one bounded asynchronous rectangular selection. The rectangle
    /// is copied from the real integer ID target; identity resolution and the
    /// controller commit happen only after correlation checks pass.
    pub fn request_box_pick(
        &mut self,
        gpu: &DesktopGpu,
        controller: &EditorController,
        start: [f32; 2],
        end: [f32; 2],
        mode: SelectionMode,
    ) -> Result<bool, DesktopError> {
        if start
            .iter()
            .chain(end.iter())
            .any(|value| !value.is_finite())
        {
            return Err(message("box-pick coordinates must be finite"));
        }
        if self.needs_upload || !self.geometry_dirty.is_empty() || !self.visibility_dirty.is_empty()
        {
            return Ok(false);
        }
        let Some(bounds) = &self.bounds else {
            return Ok(false);
        };
        let Some(viewport_target) = self.targets.get(self.viewport.id()) else {
            return Ok(false);
        };
        let config = viewport_target.state().config;
        let rectangle = local_rectangle(start, end, bounds, config)
            .ok_or_else(|| message("box-pick must cover a non-empty viewport rectangle"))?;
        self.cancel_pending();
        let viewport_target = self
            .targets
            .get(self.viewport.id())
            .ok_or_else(|| message("missing target"))?;
        let state = viewport_target.state();
        let request = self
            .boxes
            .submit(&self.viewport, state, &self.scene, rectangle)
            .map_err(renderer_error)?;
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("desktop asynchronous geometry box selection"),
            });
        let encoded = (|| {
            let native = self
                .gpu
                .as_ref()
                .ok_or_else(|| message("missing GPU scene"))?;
            let frame = viewport_target.begin_frame(&self.scene);
            native
                .renderer
                .render(
                    &mut encoder,
                    &frame,
                    &self.scene,
                    &SceneIdCamera {
                        target: state,
                        view_projection: self.matrix,
                        depth: self.depth,
                    },
                )
                .map_err(renderer_error)?;
            frame
                .encode_box_readback(
                    &mut encoder,
                    request,
                    GpuBoxSelectionLimits {
                        max_ids: MAX_OBJECTS,
                        ..GpuBoxSelectionLimits::default()
                    },
                )
                .map_err(renderer_error)
        })();
        let pending = match encoded {
            Ok(pending) => pending,
            Err(error) => {
                self.boxes.cancel_viewport(self.viewport.id());
                return Err(error);
            }
        };
        let readback = viewport_target
            .backend()
            .submit_box_readback(encoder, pending);
        self.pending_box = Some(PendingBox {
            readback,
            controller_revision: controller.revision(),
            mode,
        });
        self.counters.submitted_picks = self.counters.submitted_picks.saturating_add(1);
        Ok(true)
    }

    /// Nonblocking headless/controller integration point. Rejects results when
    /// controller revision changed since the click, in addition to viewport,
    /// scene, target allocation and target-generation checks. Miss clears only
    /// in Replace mode; modifier misses leave selection untouched. Add/Toggle/
    /// Extend go through real Outliner actions (and the controller's limits).
    pub fn poll_controller(
        &mut self,
        controller: &mut EditorController,
    ) -> Result<DesktopWork, DesktopError> {
        let selection = self.poll_selection(controller, InFlightPickReadback::poll)?;
        let mut changed = if let Some(selection) = selection {
            let changed = selection.apply(controller)?;
            self.counters.accepted_picks = self.counters.accepted_picks.saturating_add(1);
            changed
        } else {
            false
        };
        if let Some((selection, mode)) = self.poll_box_selection(controller)? {
            changed |= selection.apply(controller, mode)?;
            self.counters.accepted_picks = self.counters.accepted_picks.saturating_add(1);
        }
        Ok(DesktopWork {
            redraw: changed,
            pending: self.pending.is_some() || self.pending_box.is_some(),
        })
    }

    // The reader parameter also lets tests deterministically exercise a pending
    // real GPU readback without racing the device's map completion callback.
    fn poll_selection(
        &mut self,
        controller: &EditorController,
        read: impl FnOnce(
            &mut InFlightPickReadback,
        ) -> Result<Option<PickResult>, prism_editor_viewport::GpuPickingError>,
    ) -> Result<Option<ReadySelection>, DesktopError> {
        let Some(pending) = self.pending.as_mut() else {
            return Ok(None);
        };
        let result = match read(&mut pending.readback) {
            Ok(Some(result)) => result,
            Ok(None) => return Ok(None),
            Err(error) => {
                self.cancel_pending();
                return Err(renderer_error(error));
            }
        };
        let pending = self
            .pending
            .take()
            .ok_or_else(|| message("missing pending click"))?;
        let outcome = self.picks.resolve(
            &self.viewport,
            self.targets.state(self.viewport.id()),
            &self.scene,
            result,
        );
        if pending.controller_revision != controller.revision()
            || matches!(outcome, PickOutcome::Rejected(_))
        {
            self.counters.rejected_picks = self.counters.rejected_picks.saturating_add(1);
            return Ok(None);
        }
        let selection = match outcome {
            PickOutcome::Accepted { object: Some(id) } => {
                let object = self
                    .objects
                    .iter()
                    .find(|object| object.id == id && object.visible)
                    .ok_or_else(|| message("current GPU identity has no controller mapping"))?;
                // Validate the mapped model before granting mutable UI access.
                controller
                    .history_lengths(&object.selection)
                    .map_err(renderer_error)?;
                if pending.mode == SelectionMode::Replace {
                    ReadySelection::Replace(object.selection.clone())
                } else {
                    let node = object
                        .outliner
                        .or_else(|| {
                            controller
                                .outliner()
                                .visible_window(0, usize::MAX)
                                .into_iter()
                                .find(|id| {
                                    controller
                                        .outliner()
                                        .node(*id)
                                        .and_then(|node| node.selection())
                                        == Some(&object.selection)
                                })
                        })
                        .ok_or_else(|| message("modifier pick requires a matching Outliner row"))?;
                    if controller
                        .outliner()
                        .node(node)
                        .and_then(|row| row.selection())
                        != Some(&object.selection)
                    {
                        return Err(message(
                            "Outliner row does not match the full selection identity",
                        ));
                    }
                    ReadySelection::Outliner {
                        node,
                        mode: pending.mode,
                    }
                }
            }
            PickOutcome::Accepted { object: None } if pending.mode == SelectionMode::Replace => {
                ReadySelection::Clear
            }
            _ => ReadySelection::Unchanged,
        };
        Ok(Some(selection))
    }

    fn poll_box_selection(
        &mut self,
        controller: &EditorController,
    ) -> Result<Option<(ReadyBoxSelection, SelectionMode)>, DesktopError> {
        let Some(pending) = self.pending_box.as_mut() else {
            return Ok(None);
        };
        let result = match pending.readback.poll() {
            Ok(Some(result)) => result,
            Ok(None) => return Ok(None),
            Err(error) => {
                self.cancel_pending();
                return Err(renderer_error(error));
            }
        };
        let pending = self
            .pending_box
            .take()
            .ok_or_else(|| message("missing pending box selection"))?;
        let mode = pending.mode;
        let outcome = self.boxes.resolve(
            &self.viewport,
            self.targets.state(self.viewport.id()),
            &self.scene,
            result,
        );
        if pending.controller_revision != controller.revision()
            || matches!(outcome, SelectionOutcome::Rejected(_))
        {
            self.counters.rejected_picks = self.counters.rejected_picks.saturating_add(1);
            return Ok(None);
        }
        let SelectionOutcome::Accepted(ids) = outcome else {
            return Ok(None);
        };
        let mut items = Vec::with_capacity(ids.len());
        for id in ids {
            let Some(object) = self
                .objects
                .iter()
                .find(|object| object.id == id && object.visible)
            else {
                self.counters.rejected_picks = self.counters.rejected_picks.saturating_add(1);
                return Ok(None);
            };
            controller
                .history_lengths(&object.selection)
                .map_err(renderer_error)?;
            if !items.contains(&object.selection) {
                items.push(object.selection.clone());
            }
        }
        let selection = if items.is_empty() && mode == SelectionMode::Replace {
            ReadyBoxSelection::Clear
        } else if items.is_empty() {
            ReadyBoxSelection::Unchanged
        } else {
            ReadyBoxSelection::Items(items)
        };
        Ok(Some((selection, mode)))
    }

    fn poll_ui(
        &mut self,
        ui: &mut EditorDesktopUi,
        read: impl FnOnce(
            &mut InFlightPickReadback,
        ) -> Result<Option<PickResult>, prism_editor_viewport::GpuPickingError>,
    ) -> Result<DesktopWork, DesktopError> {
        let selection = self.poll_selection(ui.controller(), read)?;
        let changed = match selection {
            None => false,
            Some(selection) => {
                let changed = match selection {
                    ReadySelection::Unchanged => false,
                    ReadySelection::Outliner { node, mode } => {
                        ui.execute(EditorAction::SelectOutliner { node, mode })
                            .map_err(renderer_error)?
                            .changed
                    }
                    selection => {
                        let changed = selection.apply(ui.controller_mut())?;
                        let refreshed = ui.refresh().map_err(renderer_error)?;
                        changed || refreshed
                    }
                };
                self.counters.accepted_picks = self.counters.accepted_picks.saturating_add(1);
                changed
            }
        };
        let mut changed = changed;
        if let Some((selection, mode)) = self.poll_box_selection(ui.controller())? {
            let box_changed = selection.apply(ui.controller_mut(), mode)?;
            let refreshed = ui.refresh().map_err(renderer_error)?;
            changed |= box_changed || refreshed;
            self.counters.accepted_picks = self.counters.accepted_picks.saturating_add(1);
        }
        Ok(DesktopWork {
            redraw: changed,
            pending: self.pending.is_some() || self.pending_box.is_some(),
        })
    }

    fn cancel_pending(&mut self) {
        if self.pending.take().is_some() {
            self.counters.rejected_picks = self.counters.rejected_picks.saturating_add(1);
        }
        if self.pending_box.take().is_some() {
            self.counters.rejected_picks = self.counters.rejected_picks.saturating_add(1);
        }
        self.picks.cancel_viewport(self.viewport.id());
        self.boxes.cancel_viewport(self.viewport.id());
    }

    fn advance_camera(&mut self) -> Result<(), DesktopError> {
        self.viewports
            .set_camera(self.viewport.id(), self.viewport.camera())
            .map_err(renderer_error)?;
        self.viewport = self
            .viewports
            .get(self.viewport.id())
            .ok_or_else(|| message("missing viewport"))?
            .clone();
        if self.targets.state(self.viewport.id()).is_some() {
            self.targets
                .set_viewport_generation(self.viewport.id(), self.viewport.generation())
                .map_err(renderer_error)?;
        }
        Ok(())
    }

    fn upload(&mut self, gpu: &DesktopGpu) -> Result<(), DesktopError> {
        if !self.needs_upload && self.geometry_dirty.is_empty() {
            let native = self
                .gpu
                .as_mut()
                .ok_or_else(|| message("missing GPU scene"))?;
            // Patch only draw metadata. Keep the renderer, mesh buffers,
            // transforms and stable per-object ID bindings; never visit geometry.
            let operations = self
                .visibility_dirty
                .iter()
                .map(|index| {
                    let object = &self.objects[*index];
                    let key = *index as u64;
                    SceneIdPassOperation::Upsert(SceneIdPassItem::new(
                        object.id,
                        key,
                        key,
                        object.visible,
                        object.sort_key,
                    ))
                })
                .collect::<Vec<_>>();
            let commit = self.scene.apply_patch(operations).map_err(renderer_error)?;
            self.scene
                .record_commit(&commit, &mut native.renderer)
                .unwrap_or_else(|never| match never {});
            self.visibility_dirty.clear();
            return Ok(());
        }
        let uploads = if self.needs_upload {
            self.objects.len()
        } else {
            self.geometry_dirty.len()
        };
        self.scene
            .generation()
            .checked_add(uploads as u64 * 2 + 1)
            .ok_or_else(|| message("scene generation exhausted"))?;
        let mut fresh = self
            .needs_upload
            .then(|| SceneIdGeometryRenderer::new(&gpu.device));
        if fresh.is_some() {
            self.counters.geometry_renderer_creations =
                self.counters.geometry_renderer_creations.saturating_add(1);
        }
        let renderer = match fresh.as_mut() {
            Some(renderer) => renderer,
            None => {
                &mut self
                    .gpu
                    .as_mut()
                    .ok_or_else(|| message("missing GPU scene"))?
                    .renderer
            }
        };
        let mut items = Vec::with_capacity(self.objects.len());
        for (index, object) in self.objects.iter().enumerate() {
            let key = index as u64;
            if self.needs_upload || self.geometry_dirty.contains(&index) {
                renderer
                    .upload_geometry(&mut self.scene, key, &object.positions, &object.indices)
                    .map_err(renderer_error)?;
                self.counters.geometry_uploads = self.counters.geometry_uploads.saturating_add(1);
                self.counters.geometry_uploaded_bytes = self
                    .counters
                    .geometry_uploaded_bytes
                    .saturating_add(geometry_bytes(object));
                renderer
                    .set_transform(&mut self.scene, key, object.model)
                    .map_err(renderer_error)?;
                self.counters.transform_uploads = self.counters.transform_uploads.saturating_add(1);
            }
            items.push(SceneIdPassItem::new(
                object.id,
                key,
                key,
                object.visible,
                object.sort_key,
            ));
        }
        self.scene.submit(items).map_err(renderer_error)?;
        renderer.synchronize(&self.scene);
        if let Some(renderer) = fresh {
            self.gpu
                .as_mut()
                .ok_or_else(|| message("missing GPU scene"))?
                .renderer = renderer;
        }
        self.needs_upload = false;
        self.geometry_dirty.clear();
        self.visibility_dirty.clear();
        Ok(())
    }
}

impl DesktopRenderer for DesktopScene {
    fn gesture_active(&self) -> bool {
        self.gizmo.drag_active()
    }

    fn gesture_press(
        &mut self,
        ui: &mut EditorDesktopUi,
        position: [f32; 2],
    ) -> Result<bool, DesktopError> {
        if ui.has_uncommitted_edit() {
            return Err(message("finish the active edit before capturing a gizmo"));
        }
        self.gizmo_press(ui.controller_mut(), position)
    }

    fn gesture_move(
        &mut self,
        ui: &mut EditorDesktopUi,
        position: [f32; 2],
    ) -> Result<bool, DesktopError> {
        if !self.gizmo.drag_active() {
            return Ok(false);
        }
        self.gizmo_move(ui.controller_mut(), position)
    }

    fn gesture_release(&mut self, ui: &mut EditorDesktopUi) -> Result<bool, DesktopError> {
        if !self.gizmo.drag_active() {
            return Ok(false);
        }
        self.gizmo_release(ui.controller_mut())
    }

    fn gesture_cancel(&mut self, ui: &mut EditorDesktopUi) -> Result<bool, DesktopError> {
        if !self.gizmo.drag_active() {
            return Ok(false);
        }
        self.gizmo_cancel(ui.controller_mut())
    }

    fn transform_tool(
        &mut self,
        mode: prism_editor_viewport::GizmoMode,
    ) -> Result<bool, DesktopError> {
        self.set_gizmo_tool(mode, self.gizmo.snap)?;
        Ok(self.transform_adapter.is_some())
    }

    fn initialize(&mut self, gpu: &DesktopGpu) -> Result<(), DesktopError> {
        self.suspend();
        self.scene.invalidate_content().map_err(renderer_error)?;
        self.gpu = Some(GpuScene::new(gpu));
        self.counters.geometry_renderer_creations =
            self.counters.geometry_renderer_creations.saturating_add(1);
        self.needs_upload = true;
        Ok(())
    }

    fn encode(
        &mut self,
        gpu: &DesktopGpu,
        ui: &mut EditorDesktopUi,
        encoder: &mut CommandEncoder,
        target: &TextureView,
        bounds: DesktopViewportBounds,
    ) -> Result<(), DesktopError> {
        self.encode_preview(gpu, ui.controller(), encoder, target, bounds)
    }

    fn pointer(
        &mut self,
        gpu: &DesktopGpu,
        ui: &mut EditorDesktopUi,
        position: [f32; 2],
        mode: SelectionMode,
    ) -> Result<bool, DesktopError> {
        self.request_pick(gpu, ui.controller(), position, mode)
    }

    fn box_select(
        &mut self,
        gpu: &DesktopGpu,
        ui: &mut EditorDesktopUi,
        start: [f32; 2],
        end: [f32; 2],
        mode: SelectionMode,
    ) -> Result<bool, DesktopError> {
        self.request_box_pick(gpu, ui.controller(), start, end, mode)
    }

    fn poll(
        &mut self,
        _gpu: &DesktopGpu,
        ui: &mut EditorDesktopUi,
    ) -> Result<DesktopWork, DesktopError> {
        self.poll_ui(ui, InFlightPickReadback::poll)
    }

    fn prepare_document(&mut self, controller: &EditorController) -> Result<(), DesktopError> {
        self.prepare_document_candidate(controller)
    }

    fn discard_document(&mut self) {
        self.discard_document_candidate();
    }

    fn commit_document(&mut self) {
        self.commit_document_candidate();
    }

    fn document_reloaded(&mut self) {
        DesktopScene::document_reloaded(self);
    }

    fn suspend(&mut self) {
        self.cancel_pending();
        if self.targets.state(self.viewport.id()).is_some() {
            // Existence was checked; closing cannot fail for this private registry.
            let _ = self.targets.close(self.viewport.id());
        }
        self.bounds = None;
        self.gpu = None;
        self.needs_upload = true;
    }
}

impl GpuScene {
    fn new(gpu: &DesktopGpu) -> Self {
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("desktop object-ID debug layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(32),
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("desktop object-ID debug pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("desktop GPU object-ID debug shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("desktop_scene.wgsl").into()),
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("desktop GPU object-ID debug pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("desktop preview origin and selected ID"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            renderer: SceneIdGeometryRenderer::new(&gpu.device),
            gizmo: gizmo::GizmoPainter::new(gpu),
            pipeline,
            layout,
            uniform,
            binding: None,
        }
    }
}

struct PreviewRegion {
    config: GpuPickingConfig,
    origin_extent: [f32; 4],
    scissor: [u32; 4],
}

fn preview_region(bounds: &DesktopViewportBounds) -> Result<Option<PreviewRegion>, DesktopError> {
    for rect in [&bounds.logical, &bounds.physical] {
        if [
            rect.left(),
            rect.top(),
            rect.size.width,
            rect.size.height,
            rect.right(),
            rect.bottom(),
        ]
        .iter()
        .any(|value| !value.is_finite())
            || rect.size.width < 0.0
            || rect.size.height < 0.0
        {
            return Err(message(
                "viewport bounds must be finite with nonnegative extents",
            ));
        }
    }
    if bounds.surface_extent.contains(&0)
        || bounds.logical.size.width == 0.0
        || bounds.logical.size.height == 0.0
        || bounds.physical.size.width == 0.0
        || bounds.physical.size.height == 0.0
    {
        return Ok(None);
    }
    let left = bounds.physical.left().floor();
    let top = bounds.physical.top().floor();
    let right = bounds.physical.right().ceil();
    let bottom = bounds.physical.bottom().ceil();
    let width = right - left;
    let height = bottom - top;
    if width <= 0.0 || height <= 0.0 {
        return Ok(None);
    }
    if width > 32768.0 || height > 32768.0 {
        return Err(message("viewport extent exceeds 32768 pixels"));
    }
    let x = left.max(0.0).min(bounds.surface_extent[0] as f32) as u32;
    let y = top.max(0.0).min(bounds.surface_extent[1] as f32) as u32;
    let end_x = right.max(0.0).min(bounds.surface_extent[0] as f32) as u32;
    let end_y = bottom.max(0.0).min(bounds.surface_extent[1] as f32) as u32;
    if end_x <= x || end_y <= y {
        return Ok(None);
    }
    Ok(Some(PreviewRegion {
        config: GpuPickingConfig::new(width as u32, height as u32),
        origin_extent: [left, top, width, height],
        scissor: [x, y, end_x - x, end_y - y],
    }))
}

fn same_bounds(a: &DesktopViewportBounds, b: &DesktopViewportBounds) -> bool {
    a.logical == b.logical && a.physical == b.physical && a.surface_extent == b.surface_extent
}

fn local_pixel(
    position: [f32; 2],
    bounds: &DesktopViewportBounds,
    config: GpuPickingConfig,
) -> Option<PixelPoint> {
    let [x, y] = position;
    let width = bounds.logical.size.width;
    let height = bounds.logical.size.height;
    if !x.is_finite()
        || !y.is_finite()
        || config.width == 0
        || config.height == 0
        || x < 0.0
        || y < 0.0
        || x >= width
        || y >= height
        || width <= 0.0
        || height <= 0.0
    {
        return None;
    }
    Some(PixelPoint {
        x: ((x / width * config.width as f32).floor() as u32).min(config.width - 1),
        y: ((y / height * config.height as f32).floor() as u32).min(config.height - 1),
    })
}

fn local_rectangle(
    start: [f32; 2],
    end: [f32; 2],
    bounds: &DesktopViewportBounds,
    config: GpuPickingConfig,
) -> Option<PixelRect> {
    let width = bounds.logical.size.width;
    let height = bounds.logical.size.height;
    if width <= 0.0 || height <= 0.0 || config.width == 0 || config.height == 0 {
        return None;
    }
    // Rectangle endpoints may reach or cross the viewport edge. Clamp in
    // integer target space: width - EPSILON can round back to width in f32.
    let clamp = |[x, y]: [f32; 2]| PixelPoint {
        x: (((x / width).clamp(0.0, 1.0) * config.width as f32).floor() as u32)
            .min(config.width - 1),
        y: (((y / height).clamp(0.0, 1.0) * config.height as f32).floor() as u32)
            .min(config.height - 1),
    };
    let start = clamp(start);
    let end = clamp(end);
    let min_x = start.x.min(end.x);
    let min_y = start.y.min(end.y);
    let max_x = start.x.max(end.x).checked_add(1)?;
    let max_y = start.y.max(end.y).checked_add(1)?;
    PixelRect::new(
        min_x,
        min_y,
        max_x.min(config.width),
        max_y.min(config.height),
    )
    .ok()
}

fn validate_objects(objects: &[DesktopSceneObject]) -> Result<(), DesktopError> {
    if objects.len() > MAX_OBJECTS {
        return Err(message("desktop scene exceeds 4096 objects"));
    }
    let mut identities = BTreeSet::new();
    let mut bytes = 0u64;
    for object in objects {
        if !identities.insert(object.id) {
            return Err(message("duplicate full scene object identity"));
        }
        validate_matrix(&object.model)?;
        if object.positions.is_empty()
            || object.indices.is_empty()
            || !object.indices.len().is_multiple_of(3)
            || object.indices.len() > u32::MAX as usize
            || object.positions.len() > u32::MAX as usize
        {
            return Err(message(
                "geometry must be a nonempty bounded indexed triangle list",
            ));
        }
        if object
            .positions
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
            || object
                .indices
                .iter()
                .any(|index| *index as usize >= object.positions.len())
        {
            return Err(message(
                "geometry has nonfinite positions or out-of-bounds indices",
            ));
        }
        bytes = bytes
            .checked_add(object.positions.len() as u64 * 12 + object.indices.len() as u64 * 4)
            .ok_or_else(|| message("geometry byte count overflow"))?;
        if bytes > MAX_GEOMETRY_BYTES {
            return Err(message("desktop scene exceeds 64 MiB of vertex/index data"));
        }
    }
    Ok(())
}

fn geometry_bytes(object: &DesktopSceneObject) -> u64 {
    object.positions.len() as u64 * 12 + object.indices.len() as u64 * 4
}

fn validate_matrix(matrix: &SceneIdMatrix) -> Result<(), DesktopError> {
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(message("camera/model matrix must be finite"));
    }
    Ok(())
}

fn message(message: &str) -> DesktopError {
    DesktopError::Renderer(String::from(message))
}

fn renderer_error(error: impl core::fmt::Display) -> DesktopError {
    DesktopError::Renderer(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_editor_panels::OutlinerNode;
    use prism_editor_viewport::{EntityId, MeshId, SubobjectId};
    use prism_reflect::{Reflect, TypeRegistry};
    use prism_ui_layout::{Point, Rect, Size};
    use prism_ui_render_backend::draw::DrawCommand;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    fn bounds() -> DesktopViewportBounds {
        DesktopViewportBounds {
            logical: Rect::new(Point::new(4.0, 4.0), Size::new(32.0, 32.0)),
            physical: Rect::new(Point::new(8.0, 8.0), Size::new(64.0, 64.0)),
            surface_extent: [80, 80],
        }
    }

    #[test]
    fn box_rectangle_includes_edges_and_clamps_captured_pointer_outside() {
        let bounds = bounds();
        let config = GpuPickingConfig::new(64, 64);
        let full = PixelRect::new(0, 0, 64, 64).unwrap();
        assert_eq!(
            local_rectangle([0.0, 0.0], [32.0, 32.0], &bounds, config),
            Some(full)
        );
        assert_eq!(
            local_rectangle([40.0, 40.0], [-8.0, -8.0], &bounds, config),
            Some(full)
        );
        assert_eq!(
            local_rectangle([16.0, 16.0], [16.0, 16.0], &bounds, config),
            PixelRect::new(32, 32, 33, 33).ok()
        );
    }

    fn objects() -> Vec<DesktopSceneObject> {
        // All low 32-bit identity components collide. Full identities and the
        // controller mapping must survive; the selection IDs differ from both.
        let mut result = Vec::new();
        for (index, radius, z, selection) in [
            (0, 0.8, 0.75, SelectionItem::asset((1 << 121) | 19)),
            (
                1,
                0.4,
                0.25,
                SelectionItem::sub_object((1 << 122) | 23, "primitive", u32::MAX),
            ),
        ] {
            let id = SceneObjectId::new(
                EntityId::from_raw((1u128 << (119 + index)) | 9),
                MeshId::from_raw((1u128 << (109 + index)) | 7),
                SubobjectId::from_raw((1u64 << (58 + index)) | 3),
            );
            let mut object = DesktopSceneObject::new(
                id,
                selection,
                vec![
                    [-radius, -radius, z],
                    [radius, -radius, z],
                    [radius, radius, z],
                    [-radius, radius, z],
                ],
                vec![0, 1, 2, 0, 2, 3],
            );
            object.outliner = Some(OutlinerId::from_raw(100 + index as u128));
            // Draw near first and far last: depth, not draw order, wins.
            object.sort_key = 1 - index as u64;
            result.push(object);
        }
        result
    }

    fn controller(objects: &[DesktopSceneObject]) -> EditorController {
        let mut controller = EditorController::new();
        for object in objects {
            controller
                .register_object(
                    object.selection.clone(),
                    Box::new(1u32),
                    TypeRegistry::new(),
                )
                .expect("register actual selectable model");
            let mut row = OutlinerNode::new(object.outliner.expect("row"), None, "indexed mesh");
            row.set_selection(object.selection.clone());
            controller
                .register_outliner_node(row)
                .expect("register mapped row");
        }
        controller
    }

    #[derive(Clone, Debug, PartialEq, Reflect)]
    struct DraftModel {
        label: String,
    }

    fn draft_ui(objects: &[DesktopSceneObject]) -> EditorDesktopUi {
        let mut controller = EditorController::new();
        for object in objects {
            let mut registry = TypeRegistry::new();
            registry.register::<DraftModel>();
            controller
                .register_object(
                    object.selection.clone(),
                    Box::new(DraftModel {
                        label: "original draft model".into(),
                    }),
                    registry,
                )
                .expect("register editable model");
            let mut row = OutlinerNode::new(object.outliner.expect("row"), None, "editable mesh");
            row.set_selection(object.selection.clone());
            controller.register_outliner_node(row).expect("mapped row");
        }
        controller
            .select(objects[0].selection.clone())
            .expect("select editable model");
        EditorDesktopUi::new(controller).expect("real draft UI")
    }

    fn draft_label(ui: &EditorDesktopUi) -> String {
        let mut model = DraftModel {
            label: String::new(),
        };
        model
            .apply(&*ui.controller().primary_snapshot().expect("snapshot"))
            .expect("apply snapshot");
        model.label
    }

    struct StaticSceneAdapter {
        document: Option<DesktopSceneDocument>,
    }

    impl SceneDocumentAdapter for StaticSceneAdapter {
        fn build(
            &mut self,
            _controller: &EditorController,
        ) -> Result<DesktopSceneDocument, DesktopError> {
            self.document
                .take()
                .ok_or_else(|| message("test scene candidate already consumed"))
        }
    }

    #[test]
    fn document_adapter_stages_scene_before_commit() {
        let object = objects().pop().expect("test object");
        let mut scene = DesktopScene::new_with_document_adapter(StaticSceneAdapter {
            document: Some(DesktopSceneDocument {
                objects: vec![object.clone()],
                camera: SCENE_ID_IDENTITY_MATRIX,
                depth: SceneIdDepth::Forward,
            }),
        })
        .expect("adapter scene");
        let controller = controller(&[object.clone()]);

        DesktopRenderer::prepare_document(&mut scene, &controller).expect("stage candidate");
        assert!(
            scene.objects().is_empty(),
            "candidate must not publish early"
        );
        DesktopRenderer::commit_document(&mut scene);
        assert_eq!(scene.objects().len(), 1);
        assert_eq!(scene.objects()[0].id, object.id);
        assert!(scene.status().scene_generation > 0);
    }

    #[test]
    fn document_prepare_preflights_reload_without_a_scene_adapter() {
        let mut scene = DesktopScene::new(Vec::new()).expect("empty scene");
        scene
            .viewports
            .close(scene.viewport.id())
            .expect("private viewport exists");
        let scene_generation = scene.status().scene_generation;
        let viewport_generation = scene.viewport.generation();

        let error = DesktopRenderer::prepare_document(&mut scene, &EditorController::new())
            .expect_err("missing private viewport must reject preparation");

        assert!(error.to_string().contains("missing viewport"));
        assert_eq!(scene.status().scene_generation, scene_generation);
        assert_eq!(scene.viewport.generation(), viewport_generation);
        assert!(scene.pending_document.is_none());
    }

    #[test]
    fn failed_document_reprepare_discards_the_previous_candidate() {
        let object = objects().pop().expect("test object");
        let mut scene = DesktopScene::new_with_document_adapter(StaticSceneAdapter {
            document: Some(DesktopSceneDocument {
                objects: vec![object.clone()],
                camera: SCENE_ID_IDENTITY_MATRIX,
                depth: SceneIdDepth::Forward,
            }),
        })
        .expect("adapter scene");
        let controller = controller(&[object]);

        DesktopRenderer::prepare_document(&mut scene, &controller).expect("stage candidate");
        assert!(scene.pending_document.is_some());
        scene
            .viewports
            .close(scene.viewport.id())
            .expect("private viewport exists");

        let error = DesktopRenderer::prepare_document(&mut scene, &controller)
            .expect_err("missing private viewport must reject reprepare");

        assert!(error.to_string().contains("missing viewport"));
        assert!(scene.pending_document.is_none());
        DesktopRenderer::commit_document(&mut scene);
        assert!(scene.objects().is_empty());
    }

    #[test]
    fn invalid_document_scene_candidate_is_rejected_without_publication() {
        let mut invalid = objects().pop().expect("test object");
        invalid.indices[0] = 99;
        let mut scene = DesktopScene::new_with_document_adapter(StaticSceneAdapter {
            document: Some(DesktopSceneDocument {
                objects: vec![invalid],
                camera: SCENE_ID_IDENTITY_MATRIX,
                depth: SceneIdDepth::Forward,
            }),
        })
        .expect("adapter scene");
        let controller = EditorController::new();

        assert!(DesktopRenderer::prepare_document(&mut scene, &controller).is_err());
        assert!(scene.objects().is_empty());
        DesktopRenderer::discard_document(&mut scene);
        assert!(scene.objects().is_empty());
    }

    #[test]
    fn document_adapter_rejects_unbound_selection_before_publication() {
        let object = objects().pop().expect("test object");
        let mut scene = DesktopScene::new_with_document_adapter(StaticSceneAdapter {
            document: Some(DesktopSceneDocument {
                objects: vec![object],
                camera: SCENE_ID_IDENTITY_MATRIX,
                depth: SceneIdDepth::Forward,
            }),
        })
        .expect("adapter scene");

        assert!(DesktopRenderer::prepare_document(&mut scene, &EditorController::new()).is_err());
        assert!(scene.objects().is_empty());
    }

    #[test]
    fn document_adapter_rejects_mismatched_outliner_before_publication() {
        let mut object = objects().pop().expect("test object");
        let mapped = object.selection.clone();
        object.outliner = Some(OutlinerId::from_raw(999));
        let mut scene = DesktopScene::new_with_document_adapter(StaticSceneAdapter {
            document: Some(DesktopSceneDocument {
                objects: vec![object],
                camera: SCENE_ID_IDENTITY_MATRIX,
                depth: SceneIdDepth::Forward,
            }),
        })
        .expect("adapter scene");
        let mut controller = EditorController::new();
        controller
            .register_object(mapped, Box::new(1u32), TypeRegistry::new())
            .expect("register object");

        assert!(DesktopRenderer::prepare_document(&mut scene, &controller).is_err());
        assert!(scene.objects().is_empty());
    }

    fn begin_draft(ui: &mut EditorDesktopUi, text: &str) {
        // Use the actual rendered field label/hit geometry, not UI internals.
        let rect = ui
            .draw_list()
            .commands()
            .iter()
            .find_map(|command| match command {
                DrawCommand::Glyph(glyph) if ui.draw_list().text(glyph.text) == "label" => {
                    Some(glyph.rect)
                }
                _ => None,
            })
            .expect("drawn editable field label");
        assert!(ui
            .click(
                rect.location.x + rect.size.width * 0.5,
                rect.location.y + rect.size.height * 0.5,
                SelectionMode::Replace,
            )
            .expect("focus actual Inspector field"));
        assert!(ui.text(text).expect("enter uncommitted draft"));
        assert!(ui.text_input_rect().is_some());
    }

    #[test]
    fn desktop_scene_idle_poll_preserves_editing_draft_and_ime_without_redraw() {
        let objects = objects();
        let mut ui = draft_ui(&objects);
        let mut scene = DesktopScene::new(objects).expect("scene");
        begin_draft(&mut ui, "idle draft survives");
        assert!(ui.preedit("组合输入").expect("IME composition"));
        let revision = ui.controller().revision();
        let draw = ui.draw_list().clone();
        let input = ui.text_input_rect();
        for _ in 0..8 {
            // This is the exact UI path used by DesktopRenderer::poll. No GPU
            // exists and an idle poll must not even try reading one.
            let work = scene
                .poll_ui(&mut ui, |_| panic!("idle poll read GPU"))
                .expect("idle UI poll");
            assert!(!work.redraw && !work.pending);
            assert_eq!(ui.text_input_rect(), input);
            assert_eq!(ui.draw_list(), &draw);
            assert_eq!(ui.controller().revision(), revision);
            assert!(!ui.refresh().expect("not invalidated"));
        }
        assert_eq!(draft_label(&ui), "original draft model");
        assert_eq!(ui.controller().document_history_lengths(), (0, 0));
        assert!(ui.preedit("").expect("end IME composition"));
        assert!(ui
            .key(crate::desktop_ui::DesktopKey::Enter)
            .expect("commit surviving draft"));
        assert_eq!(draft_label(&ui), "idle draft survives");
        assert_eq!(ui.controller().document_history_lengths(), (1, 0));
    }

    #[test]
    fn desktop_scene_gpu_pending_stale_and_modifier_miss_preserve_editing_draft() {
        let Some(gpu) = gpu() else { return };
        let objects = objects();
        let mut ui = draft_ui(&objects);
        let mut scene = DesktopScene::new(objects.clone()).expect("scene");
        scene.initialize(&gpu).expect("initialize");
        let output = output(&gpu);
        encode(&mut scene, &gpu, ui.controller(), &output, bounds());
        begin_draft(&mut ui, "pending draft survives");
        assert!(scene
            .pointer(&gpu, &mut ui, [16.0, 16.0], SelectionMode::Replace)
            .expect("real pick"));
        let revision = ui.controller().revision();
        let draw = ui.draw_list().clone();
        let input = ui.text_input_rect();
        for _ in 0..8 {
            // A real submitted copy/map is retained, but deliberately do not
            // observe its callback yet: the unready branch is deterministic.
            let work = scene
                .poll_ui(&mut ui, |_| Ok(None))
                .expect("unready UI poll");
            assert!(!work.redraw && work.pending);
            assert_eq!(ui.controller().revision(), revision);
            assert_eq!(ui.draw_list(), &draw);
            assert_eq!(ui.text_input_rect(), input);
            assert!(!ui.refresh().expect("pending poll not invalidated"));
        }
        assert_eq!(scene.status().accepted_picks, 0);
        assert_eq!(draft_label(&ui), "original draft model");
        assert!(ui
            .key(crate::desktop_ui::DesktopKey::Enter)
            .expect("commit pending draft"));
        assert_eq!(draft_label(&ui), "pending draft survives");
        // The commit changes controller revision, so the old result is stale.
        // Start another draft before observing it to catch mutation on reject.
        begin_draft(&mut ui, "stale result preserves this draft");
        let draw = ui.draw_list().clone();
        let input = ui.text_input_rect();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let work = scene.poll(&gpu, &mut ui).expect("real stale readback poll");
            assert!(!work.redraw);
            assert_eq!(ui.draw_list(), &draw);
            assert_eq!(ui.text_input_rect(), input);
            assert!(!ui.refresh().expect("stale poll not invalidated"));
            if !work.pending {
                break;
            }
            assert!(Instant::now() < deadline, "stale completion deadline");
            std::thread::yield_now();
        }
        assert_eq!(scene.status().accepted_picks, 0);
        assert_eq!(scene.status().rejected_picks, 1);
        assert_eq!(
            ui.controller().selection().primary(),
            Some(&objects[0].selection)
        );
        // A ready, valid modifier miss also carries no selection operation.
        assert!(scene
            .pointer(&gpu, &mut ui, [0.0, 0.0], SelectionMode::Add)
            .expect("modifier miss"));
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let work = scene.poll(&gpu, &mut ui).expect("real modifier miss poll");
            assert!(!work.redraw);
            assert_eq!(ui.draw_list(), &draw);
            assert_eq!(ui.text_input_rect(), input);
            assert!(!ui.refresh().expect("modifier miss not invalidated"));
            if !work.pending {
                break;
            }
            assert!(Instant::now() < deadline, "miss completion deadline");
            std::thread::yield_now();
        }
        assert_eq!(scene.status().accepted_picks, 1);
        assert!(ui
            .key(crate::desktop_ui::DesktopKey::Enter)
            .expect("commit after stale result and miss"));
        assert_eq!(draft_label(&ui), "stale result preserves this draft");
        assert_eq!(ui.controller().document_history_lengths(), (2, 0));
        // Finally prove a current hit still takes the actual selection path.
        assert!(scene
            .pointer(&gpu, &mut ui, [16.0, 16.0], SelectionMode::Replace)
            .expect("current hit"));
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let work = scene.poll(&gpu, &mut ui).expect("apply current hit");
            if !work.pending {
                assert!(work.redraw);
                break;
            }
            assert!(Instant::now() < deadline, "hit completion deadline");
            std::thread::yield_now();
        }
        assert_eq!(
            ui.controller().selection().primary(),
            Some(&objects[1].selection)
        );
        assert_eq!(scene.status().accepted_picks, 2);
    }

    #[test]
    fn desktop_scene_validates_full_identity_and_replacement_atomically() {
        let descriptors = objects();
        let mut scene = DesktopScene::new(descriptors.clone()).expect("valid full identities");
        let before = scene.status().scene_generation;
        let mut invalid = descriptors.clone();
        invalid[1].id = invalid[0].id;
        assert!(scene.replace_objects(invalid).is_err());
        let mut invalid = descriptors.clone();
        invalid[0].indices[0] = 4;
        assert!(scene.replace_objects(invalid).is_err());
        let mut invalid = descriptors.clone();
        invalid[0].positions[0][0] = f32::NAN;
        assert!(scene.replace_objects(invalid).is_err());
        let mut invalid = descriptors.clone();
        invalid[0].indices.pop();
        assert!(scene.replace_objects(invalid).is_err());
        assert_eq!(scene.status().scene_generation, before);
        assert_eq!(scene.objects()[0].id, descriptors[0].id);
        assert_eq!(scene.objects()[1].selection, descriptors[1].selection);
        // Multiple distinct mesh identities may map to the same model.
        let mut shared = descriptors;
        shared[1].selection = shared[0].selection.clone();
        scene.replace_objects(shared).expect("many-to-one mapping");
        assert!(scene.status().scene_generation > before);
        scene
            .replace_objects(Vec::new())
            .expect("valid empty scene");
    }

    #[test]
    fn object_model_updates_are_deferred_without_rebuilding_the_scene() {
        let descriptors = objects();
        let mut scene = DesktopScene::new(descriptors.clone()).expect("valid scene");
        let id = descriptors[0].id;
        let mut model = descriptors[0].model;
        model[3][0] = 0.25;

        assert_eq!(
            scene.update_object_model(id, descriptors[0].model),
            Ok(false)
        );
        assert_eq!(scene.update_object_model(id, model), Ok(true));
        assert_eq!(scene.object_model(id), Some(model));
        assert_eq!(scene.objects()[1].model, descriptors[1].model);

        let mut invalid = model;
        invalid[0][0] = f32::NAN;
        assert!(scene.update_object_model(id, invalid).is_err());
        assert_eq!(scene.object_model(id), Some(model));
    }

    #[test]
    fn desktop_scene_local_coordinates_scaling_clipping_and_hidden_bounds() {
        let mut bounds = bounds();
        let region = preview_region(&bounds).expect("bounds").expect("visible");
        assert_eq!(region.scissor, [8, 8, 64, 64]);
        assert_eq!(
            local_pixel([16.0, 16.0], &bounds, region.config),
            Some(PixelPoint { x: 32, y: 32 })
        );
        assert_eq!(
            local_pixel([31.9, 0.0], &bounds, region.config),
            Some(PixelPoint { x: 63, y: 0 })
        );
        for position in [[32.0, 0.0], [-0.1, 1.0], [1.0, 32.0], [f32::NAN, 0.0]] {
            assert!(local_pixel(position, &bounds, region.config).is_none());
        }
        bounds.physical.location.x = -8.0;
        let clipped = preview_region(&bounds).expect("clipped").expect("visible");
        assert_eq!(clipped.scissor, [0, 8, 56, 64]);
        assert_eq!(clipped.config.width, 64);
        bounds.physical.size.width = 0.0;
        assert!(preview_region(&bounds).expect("hidden").is_none());
        bounds.physical.size.width = -1.0;
        assert!(preview_region(&bounds).is_err());
        bounds.physical.size.width = f32::INFINITY;
        assert!(preview_region(&bounds).is_err());
    }

    #[test]
    fn desktop_scene_camera_validation_does_not_mutate_generation() {
        let mut scene = DesktopScene::new(Vec::new()).expect("empty");
        let before = scene.viewport.generation();
        let mut matrix = SCENE_ID_IDENTITY_MATRIX;
        matrix[3][0] = f32::INFINITY;
        assert!(scene.set_camera(matrix, SceneIdDepth::Forward).is_err());
        assert_eq!(scene.viewport.generation(), before);
        scene
            .set_camera(SCENE_ID_IDENTITY_MATRIX, SceneIdDepth::Forward)
            .expect("unchanged");
        assert_eq!(scene.viewport.generation(), before);
        scene
            .set_camera(SCENE_ID_IDENTITY_MATRIX, SceneIdDepth::Reverse)
            .expect("depth convention");
        assert!(scene.viewport.generation() > before);
    }

    #[test]
    fn desktop_scene_document_reload_fences_generations_and_preserves_camera() {
        let mut scene = DesktopScene::new(Vec::new()).expect("empty");
        let mut camera = SCENE_ID_IDENTITY_MATRIX;
        camera[3][0] = 0.25;
        camera[3][1] = -0.5;
        scene
            .set_camera(camera, SceneIdDepth::Reverse)
            .expect("camera");
        let scene_generation = scene.scene.generation();
        let viewport_generation = scene.viewport.generation();

        scene.document_reloaded();

        assert!(scene.scene.generation() > scene_generation);
        assert!(scene.viewport.generation() > viewport_generation);
        assert_eq!(scene.matrix, camera);
        assert_eq!(scene.depth, SceneIdDepth::Reverse);
        assert!(scene.needs_upload);
        assert!(!scene.status().pending);
    }

    // These are real offscreen GPU integrations, not window or material tests.
    // PRISM_REQUIRE_GPU=1 makes unavailable hardware a failure, never a pass.
    #[expect(
        clippy::print_stderr,
        reason = "explicit required-GPU/skip diagnostics"
    )]
    fn gpu() -> Option<DesktopGpu> {
        let required = std::env::var("PRISM_REQUIRE_GPU").as_deref() == Ok("1");
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            match futures_lite::future::block_on(instance.request_adapter(&Default::default())) {
                Ok(adapter) => adapter,
                Err(error) => {
                    assert!(
                        !required,
                        "PRISM_REQUIRE_GPU=1: desktop scene adapter unavailable: {error}"
                    );
                    eprintln!("SKIP desktop scene GPU integration: {error}");
                    return None;
                }
            };
        eprintln!("desktop scene offscreen GPU: {:?}", adapter.get_info());
        let descriptor = wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        };
        let (device, queue) =
            match futures_lite::future::block_on(adapter.request_device(&descriptor)) {
                Ok(handles) => handles,
                Err(error) => {
                    assert!(
                        !required,
                        "PRISM_REQUIRE_GPU=1: desktop scene device unavailable: {error}"
                    );
                    eprintln!("SKIP desktop scene GPU integration: {error}");
                    return None;
                }
            };
        Some(DesktopGpu {
            adapter,
            device,
            queue,
            format: wgpu::TextureFormat::Rgba8Unorm,
        })
    }

    fn output(gpu: &DesktopGpu) -> wgpu::Texture {
        gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("desktop scene offscreen integration output"),
            size: wgpu::Extent3d {
                width: 80,
                height: 80,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: gpu.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    fn encode(
        scene: &mut DesktopScene,
        gpu: &DesktopGpu,
        controller: &EditorController,
        output: &wgpu::Texture,
        bounds: DesktopViewportBounds,
    ) {
        let view = output.create_view(&Default::default());
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 1.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
        }
        scene
            .encode_preview(gpu, controller, &mut encoder, &view, bounds)
            .expect("encode indexed geometry and GPU preview");
        gpu.queue.submit([encoder.finish()]);
    }

    #[test]
    fn desktop_scene_gpu_box_selection_resolves_all_visible_ids_atomically() {
        let Some(gpu) = gpu() else { return };
        let objects = objects();
        let mut controller = controller(&objects);
        let mut scene = DesktopScene::new(objects.clone()).expect("scene");
        scene.initialize(&gpu).expect("initialize");
        let output = output(&gpu);
        encode(&mut scene, &gpu, &controller, &output, bounds());
        assert!(scene
            .request_box_pick(
                &gpu,
                &controller,
                [0.0, 0.0],
                [32.0, 32.0],
                SelectionMode::Replace,
            )
            .expect("submit GPU box selection"));
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let work = scene
                .poll_controller(&mut controller)
                .expect("poll GPU box selection");
            if !work.pending {
                assert!(work.redraw);
                break;
            }
            assert!(
                Instant::now() < deadline,
                "box selection completion deadline"
            );
            std::thread::yield_now();
        }
        assert_eq!(controller.selection().items().len(), 2);
        assert_eq!(
            controller.selection().primary(),
            Some(&objects[1].selection)
        );
        assert!(!scene.status().pending_box);
        assert_eq!(scene.status().accepted_picks, 1);
    }

    fn pixels(gpu: &DesktopGpu, texture: &wgpu::Texture) -> Vec<u8> {
        // Readback is test evidence only; production presentation stays on GPU.
        const STRIDE: u32 = 512;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offscreen preview assertion readback"),
            size: u64::from(STRIDE) * 80,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(STRIDE),
                    rows_per_image: Some(80),
                },
            },
            wgpu::Extent3d {
                width: 80,
                height: 80,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit([encoder.finish()]);
        let (sender, receiver) = mpsc::sync_channel(1);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU test wait");
        receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("map callback")
            .expect("map success");
        let mapped = buffer.slice(..).get_mapped_range().expect("mapped preview");
        let mut result = Vec::with_capacity(80 * 80 * 4);
        for row in mapped.chunks_exact(STRIDE as usize) {
            result.extend_from_slice(&row[..80 * 4]);
        }
        drop(mapped);
        buffer.unmap();
        result
    }

    fn pixel(bytes: &[u8], x: usize, y: usize) -> [u8; 4] {
        bytes[(y * 80 + x) * 4..(y * 80 + x) * 4 + 4]
            .try_into()
            .expect("RGBA pixel")
    }

    fn expected_color(scene: &DesktopScene, id: SceneObjectId) -> [u8; 4] {
        let channels = scene
            .scene
            .entries()
            .find(|entry| entry.item().id == id)
            .expect("full scene identity")
            .encoded_id()
            .channels();
        let hash = channels[0]
            .wrapping_mul(1_664_525)
            .wrapping_add(channels[1].wrapping_mul(1_013_904_223));
        let color = |shift: u32| {
            ((0.2 + 0.65 * ((hash >> shift) & 255u32) as f32 / 255.0) * 255.0).round() as u8
        };
        [color(0), color(8), color(16), 255]
    }

    fn assert_color(actual: [u8; 4], expected: [u8; 4]) {
        assert!(
            actual.iter().zip(expected).all(|(a, b)| a.abs_diff(b) <= 1),
            "actual {actual:?}, expected {expected:?}"
        );
    }

    fn finish(
        scene: &mut DesktopScene,
        gpu: &DesktopGpu,
        controller: &mut EditorController,
    ) -> DesktopWork {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let work = scene
                .poll_controller(controller)
                .expect("poll correlated selection");
            if !work.pending {
                return work;
            }
            assert!(Instant::now() < deadline, "GPU pick completion deadline");
            gpu.device.poll(wgpu::PollType::Poll).expect("GPU poll");
            std::thread::yield_now();
        }
    }

    fn pick(
        scene: &mut DesktopScene,
        gpu: &DesktopGpu,
        controller: &mut EditorController,
        position: [f32; 2],
        mode: SelectionMode,
    ) {
        assert!(scene
            .request_pick(gpu, controller, position, mode)
            .expect("request"));
        finish(scene, gpu, controller);
    }

    #[test]
    fn desktop_scene_gpu_indexed_occlusion_preview_and_full_selection_mapping() {
        let Some(gpu) = gpu() else { return };
        let objects = objects();
        let mut controller = controller(&objects);
        let mut scene = DesktopScene::new(objects.clone()).expect("scene");
        scene.initialize(&gpu).expect("shared GPU initialize");
        let output = output(&gpu);
        encode(&mut scene, &gpu, &controller, &output, bounds());
        let image = pixels(&gpu, &output);
        assert_eq!(pixel(&image, 0, 0), [0, 255, 0, 255]);
        assert_eq!(pixel(&image, 75, 40), [0, 255, 0, 255]);
        assert_color(pixel(&image, 10, 10), [6, 8, 11, 255]);
        assert_color(pixel(&image, 40, 40), expected_color(&scene, objects[1].id));
        assert_color(pixel(&image, 18, 40), expected_color(&scene, objects[0].id));
        pick(
            &mut scene,
            &gpu,
            &mut controller,
            [16.0, 16.0],
            SelectionMode::Replace,
        );
        assert_eq!(
            controller.selection().primary(),
            Some(&objects[1].selection)
        );
        encode(&mut scene, &gpu, &controller, &output, bounds());
        assert_ne!(pixel(&pixels(&gpu, &output), 40, 40), pixel(&image, 40, 40));
        pick(
            &mut scene,
            &gpu,
            &mut controller,
            [5.0, 16.0],
            SelectionMode::Add,
        );
        assert_eq!(controller.selection().items().len(), 2);
        pick(
            &mut scene,
            &gpu,
            &mut controller,
            [16.0, 16.0],
            SelectionMode::Toggle,
        );
        assert_eq!(
            controller.selection().items(),
            &[objects[0].selection.clone()]
        );
        pick(
            &mut scene,
            &gpu,
            &mut controller,
            [5.0, 16.0],
            SelectionMode::Extend,
        );
        assert_eq!(controller.selection().items().len(), 2);
        pick(
            &mut scene,
            &gpu,
            &mut controller,
            [1.0, 1.0],
            SelectionMode::Add,
        );
        assert_eq!(controller.selection().items().len(), 2);
        pick(
            &mut scene,
            &gpu,
            &mut controller,
            [1.0, 1.0],
            SelectionMode::Replace,
        );
        assert!(controller.selection().items().is_empty());
        assert_eq!(scene.status().accepted_picks, 6);
        assert_eq!(scene.status().preview_bindings, 1);
    }

    #[test]
    fn desktop_scene_gpu_stale_camera_resize_controller_and_reopen() {
        let Some(gpu) = gpu() else { return };
        let objects = objects();
        let mut controller = controller(&objects);
        let mut scene = DesktopScene::new(objects.clone()).expect("scene");
        scene.initialize(&gpu).expect("initialize");
        let output = output(&gpu);
        encode(&mut scene, &gpu, &controller, &output, bounds());
        let original = scene.status().target.expect("target");
        assert!(scene
            .request_pick(&gpu, &controller, [16.0, 16.0], SelectionMode::Replace)
            .expect("request"));
        controller
            .select(objects[0].selection.clone())
            .expect("newer controller action");
        finish(&mut scene, &gpu, &mut controller);
        assert_eq!(
            controller.selection().primary(),
            Some(&objects[0].selection)
        );
        assert_eq!(scene.status().accepted_picks, 0);
        assert_eq!(scene.status().rejected_picks, 1);
        assert!(scene
            .request_pick(&gpu, &controller, [16.0, 16.0], SelectionMode::Replace)
            .expect("old click"));
        pick(
            &mut scene,
            &gpu,
            &mut controller,
            [5.0, 16.0],
            SelectionMode::Replace,
        );
        assert_eq!(
            controller.selection().primary(),
            Some(&objects[0].selection)
        );
        assert_eq!(scene.status().rejected_picks, 2);
        assert!(scene
            .request_pick(&gpu, &controller, [16.0, 16.0], SelectionMode::Replace)
            .expect("camera-correlated request"));
        scene
            .set_camera(SCENE_ID_IDENTITY_MATRIX, SceneIdDepth::Reverse)
            .expect("reverse camera");
        assert!(!scene.status().pending);
        encode(&mut scene, &gpu, &controller, &output, bounds());
        pick(
            &mut scene,
            &gpu,
            &mut controller,
            [16.0, 16.0],
            SelectionMode::Replace,
        );
        assert_eq!(
            controller.selection().primary(),
            Some(&objects[0].selection)
        );
        assert_eq!(scene.status().preview_bindings, 1);
        let mut translated = SCENE_ID_IDENTITY_MATRIX;
        translated[3][0] = 2.0;
        scene
            .set_camera(translated, SceneIdDepth::Forward)
            .expect("translated camera");
        encode(&mut scene, &gpu, &controller, &output, bounds());
        pick(
            &mut scene,
            &gpu,
            &mut controller,
            [16.0, 16.0],
            SelectionMode::Replace,
        );
        assert!(controller.selection().items().is_empty());
        scene
            .set_camera(SCENE_ID_IDENTITY_MATRIX, SceneIdDepth::Forward)
            .expect("restore camera");
        assert!(scene
            .request_pick(&gpu, &controller, [16.0, 16.0], SelectionMode::Replace)
            .expect("resize-correlated request"));
        let mut resized = bounds();
        resized.physical.size = Size::new(32.0, 32.0);
        encode(&mut scene, &gpu, &controller, &output, resized);
        assert!(!scene.status().pending);
        assert!(controller.selection().items().is_empty());
        let new_target = scene.status().target.expect("resized target");
        assert_eq!(new_target.handle, original.handle);
        assert_eq!(new_target.config.width, 32);
        assert!(new_target.target_generation > original.target_generation);
        assert!(new_target.viewport_generation > original.viewport_generation);
        assert_eq!(scene.status().preview_bindings, 2);
        pick(
            &mut scene,
            &gpu,
            &mut controller,
            [16.0, 16.0],
            SelectionMode::Replace,
        );
        assert_eq!(
            controller.selection().primary(),
            Some(&objects[1].selection)
        );
        scene.suspend();
        assert!(scene.status().target.is_none());
        assert!(!scene
            .request_pick(&gpu, &controller, [16.0, 16.0], SelectionMode::Replace)
            .expect("suspended"));
        scene.initialize(&gpu).expect("resume shared device");
        encode(&mut scene, &gpu, &controller, &output, bounds());
        assert_ne!(
            scene.status().target.expect("reopened").handle,
            original.handle
        );
        assert_eq!(scene.status().preview_bindings, 3);
        pick(
            &mut scene,
            &gpu,
            &mut controller,
            [5.0, 16.0],
            SelectionMode::Replace,
        );
        assert_eq!(
            controller.selection().primary(),
            Some(&objects[0].selection)
        );
    }

    #[test]
    fn desktop_scene_gpu_replacement_and_real_ui_adapter_poll() {
        let Some(gpu) = gpu() else { return };
        let objects = objects();
        let mut ui = EditorDesktopUi::new(controller(&objects)).expect("real retained UI");
        let mut scene = DesktopScene::new(objects.clone()).expect("scene");
        scene.initialize(&gpu).expect("initialize");
        let output = output(&gpu);
        let view = output.create_view(&Default::default());
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        scene
            .encode(&gpu, &mut ui, &mut encoder, &view, bounds())
            .expect("native trait encode");
        gpu.queue.submit([encoder.finish()]);
        let token = scene
            .scene
            .entries()
            .find(|entry| entry.item().id == objects[1].id)
            .expect("token")
            .encoded_id();
        assert!(scene
            .pointer(&gpu, &mut ui, [16.0, 16.0], SelectionMode::Replace)
            .expect("native pointer"));
        let mut invalid = objects.clone();
        invalid[0].indices[0] = 99;
        assert!(scene.replace_objects(invalid).is_err());
        assert!(scene.status().pending);
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let work = scene
                .poll(&gpu, &mut ui)
                .expect("native trait poll and UI refresh");
            if !work.pending {
                assert!(work.redraw);
                break;
            }
            assert!(
                Instant::now() < deadline,
                "native adapter completion deadline"
            );
            std::thread::yield_now();
        }
        assert_eq!(
            ui.controller().selection().primary(),
            Some(&objects[1].selection)
        );
        assert!(scene
            .pointer(&gpu, &mut ui, [16.0, 16.0], SelectionMode::Replace)
            .expect("pending replacement"));
        scene
            .replace_objects(vec![objects[0].clone()])
            .expect("remove near mesh");
        assert!(!scene.status().pending);
        assert!(!scene
            .pointer(&gpu, &mut ui, [16.0, 16.0], SelectionMode::Replace)
            .expect("unencoded replacement"));
        encode(&mut scene, &gpu, ui.controller(), &output, bounds());
        pick(
            &mut scene,
            &gpu,
            ui.controller_mut(),
            [16.0, 16.0],
            SelectionMode::Replace,
        );
        assert_eq!(
            ui.controller().selection().primary(),
            Some(&objects[0].selection)
        );
        scene
            .replace_objects(objects.clone())
            .expect("readd near mesh");
        encode(&mut scene, &gpu, ui.controller(), &output, bounds());
        assert_eq!(
            scene
                .scene
                .entries()
                .find(|entry| entry.item().id == objects[1].id)
                .expect("readded token")
                .encoded_id(),
            token
        );
        let mut hidden = bounds();
        hidden.logical.size.width = 0.0;
        encode(&mut scene, &gpu, ui.controller(), &output, hidden);
        assert!(!scene
            .pointer(&gpu, &mut ui, [1.0, 1.0], SelectionMode::Replace)
            .expect("hidden"));
    }

    #[test]
    fn desktop_scene_gpu_document_reload_rejects_old_pick_and_republishes() {
        let Some(gpu) = gpu() else { return };
        let objects = objects();
        let mut controller = controller(&objects);
        let mut scene = DesktopScene::new(objects.clone()).expect("scene");
        scene.initialize(&gpu).expect("initialize");
        let camera = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.125, -0.25, 0.0, 1.0],
        ];
        scene
            .set_camera(camera, SceneIdDepth::Forward)
            .expect("non-identity camera");
        let output = output(&gpu);
        encode(&mut scene, &gpu, &controller, &output, bounds());

        let old_scene_generation = scene.status().scene_generation;
        let old_viewport_generation = scene.viewport.generation();
        let old_target = scene.status().target.expect("published ID target");
        assert!(scene
            .request_pick(&gpu, &controller, [16.0, 16.0], SelectionMode::Replace)
            .expect("queue old-document pick"));

        scene.document_reloaded();

        assert!(!scene.status().pending);
        assert_eq!(scene.status().rejected_picks, 1);
        assert!(scene.status().target.is_none());
        assert!(scene.status().scene_generation > old_scene_generation);
        assert!(scene.viewport.generation() > old_viewport_generation);
        assert_ne!(scene.matrix, SCENE_ID_IDENTITY_MATRIX);
        assert_eq!(scene.matrix, camera);
        assert_eq!(
            scene
                .poll_controller(&mut controller)
                .expect("poll after reload"),
            DesktopWork {
                redraw: false,
                pending: false,
            }
        );
        assert_eq!(controller.selection().primary(), None);

        // A reload keeps the CPU descriptors/camera but requires a fresh
        // target and transform upload before another pick can be submitted.
        encode(&mut scene, &gpu, &controller, &output, bounds());
        let new_target = scene.status().target.expect("republished ID target");
        assert_ne!(new_target.handle, old_target.handle);
        assert_eq!(new_target.target_generation, 0);
        assert!(new_target.viewport_generation > old_target.viewport_generation);
        pick(
            &mut scene,
            &gpu,
            &mut controller,
            [16.0, 16.0],
            SelectionMode::Replace,
        );
        assert_eq!(
            controller.selection().primary(),
            Some(&objects[1].selection)
        );
    }
}
