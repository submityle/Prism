//! Authoring document authority with transactional ECS and scene projections.

use alloc::{collections::BTreeMap, sync::Arc};

use prism_ecs::world::World;
use prism_editor_core::{SelectionItem, SelectionKind};
use prism_editor_ui::SelectionMode;
use prism_editor_viewport::{
    EntityId, GizmoMode, MeshId, SceneIdDepth, SceneIdMatrix, SceneObjectId, SubobjectId,
    SCENE_ID_IDENTITY_MATRIX,
};
use prism_reflect::{FromReflect, Reflect, ReflectRef};

use crate::{
    desktop_scene::{MAX_GEOMETRY_BYTES, MAX_OBJECTS},
    scene_authoring::scene_document_budget,
    scene_mesh::{
        editor_scene_mesh_geometry_bytes, editor_scene_mesh_matrix, editor_scene_mesh_validate,
        EditorSceneMeshTransform,
    },
    ControllerError, DesktopError, DesktopGpu, DesktopRenderer, DesktopScene, DesktopSceneDocument,
    DesktopSceneObject, DesktopViewportBounds, DesktopWork, EditorController, EditorDesktopUi,
    EditorEcsDocumentBridge, EditorSceneMesh,
};

/// Actual synchronization work, excluding UI-only revisions and idle polls.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EditorSceneSyncWork {
    /// Complete scene documents installed (initial load included). Same-owner
    /// structure changes restage the scene but retain the ECS World.
    pub document_rebuilds: u64,
    /// Changed objects published into ECS and scene.
    pub objects_synchronized: u64,
    /// Changed geometry descriptors, never charged for property-only updates.
    pub geometry_replacements: u64,
    /// Matrix updates without geometry replacement (possibly with visibility).
    pub matrix_updates: u64,
    /// Visibility updates without decoding or copying geometry.
    pub visibility_updates: u64,
}

struct Projection {
    world: World,
    bridge: EditorEcsDocumentBridge,
    owner: Arc<()>,
    shape_revision: u64,
    model_revision: u64,
    object_count: usize,
    geometry_bytes: u64,
}

/// Production indexed-mesh authoring profile. The controller is the authority;
/// ECS and GPU scene are read-only projections, never separate edit histories.
/// Additional project schemas use an explicit host adapter instead of silently
/// dropping unknown roots or inventing geometry for them.
pub struct EditorSceneSession {
    scene: DesktopScene,
    projection: Projection,
    pending: Option<Projection>,
    work: EditorSceneSyncWork,
    camera: SceneIdMatrix,
    depth: SceneIdDepth,
}

impl EditorSceneSession {
    /// Materializes the complete document using an identity clip-space camera.
    /// An empty new project remains empty; no demo objects are injected.
    /// Installs the host's aggregate geometry budget before authored history.
    /// An already installed budget permits attaching with existing history.
    pub fn new(controller: &mut EditorController) -> Result<Self, DesktopError> {
        Self::with_camera(controller, SCENE_ID_IDENTITY_MATRIX, SceneIdDepth::Forward)
    }

    /// Materializes a profile with an explicit host-owned world-to-clip camera.
    pub fn with_camera(
        controller: &mut EditorController,
        camera: SceneIdMatrix,
        depth: SceneIdDepth,
    ) -> Result<Self, DesktopError> {
        let (projection, document) = build_projection(controller, camera, depth)?;
        let mut scene = DesktopScene::new(Vec::new())?;
        scene.set_transform_adapter(EditorSceneMeshTransform)?;
        scene.stage_document(controller, document)?;
        // Validate/stage the complete projection before changing host policy.
        // A missing budget cannot be retrofitted across existing undo/redo.
        // Registration is the last fallible step, so failures leave no policy
        // (or authoring commands) installed on the supplied controller.
        controller
            .register_document_budgets([scene_document_budget()])
            .map_err(|cause| match cause {
                ControllerError::EditConflict => error(
                    "scene geometry budget must be installed before authored undo/redo history",
                ),
                cause => error(&cause.to_string()),
            })?;
        DesktopRenderer::commit_document(&mut scene);
        Ok(Self {
            scene,
            projection,
            pending: None,
            work: EditorSceneSyncWork {
                document_rebuilds: 1,
                objects_synchronized: controller.document_objects().len() as u64,
                ..EditorSceneSyncWork::default()
            },
            camera,
            depth,
        })
    }

    /// Read-only live ECS projection; mutations must go through the controller.
    #[must_use]
    pub fn world(&self) -> &World {
        &self.projection.world
    }

    /// Stable authoring identity to generational ECS entity mapping.
    #[must_use]
    pub fn bridge(&self) -> &EditorEcsDocumentBridge {
        &self.projection.bridge
    }

    /// Current scene descriptors and pick diagnostics.
    #[must_use]
    pub fn scene(&self) -> &DesktopScene {
        &self.scene
    }

    /// Work counters permit tests and hosts to detect accidental full rebuilds.
    #[must_use]
    pub const fn work(&self) -> EditorSceneSyncWork {
        self.work
    }

    /// Changes the explicit viewport camera without rebuilding the ECS World.
    pub fn set_camera(
        &mut self,
        camera: SceneIdMatrix,
        depth: SceneIdDepth,
    ) -> Result<(), DesktopError> {
        if self.pending.is_some() {
            return Err(error("cannot change camera during document preparation"));
        }
        self.scene.set_camera(camera, depth)?;
        self.camera = camera;
        self.depth = depth;
        Ok(())
    }

    /// Publishes committed edits only. Uncommitted Gizmo previews remain owned
    /// by `DesktopScene` until release/cancel. Idle and UI-only paths are O(1).
    pub fn synchronize(&mut self, controller: &EditorController) -> Result<bool, DesktopError> {
        // A host may render/poll the old controller between prepare and commit.
        // Never let that path replace or publish over the staged candidate.
        if self.pending.is_some() {
            return Err(error("cannot synchronize during document preparation"));
        }
        if controller.has_active_edit() || self.scene.gesture_active() {
            return Ok(false);
        }
        if !Arc::ptr_eq(&self.projection.owner, &controller.document_identity()) {
            self.prepare_document(controller)?;
            self.commit_document();
            return Ok(true);
        }
        if self.projection.shape_revision != controller.shape_revision() {
            return self.synchronize_structure(controller);
        }
        if self.projection.model_revision == controller.model_revision() {
            return Ok(false);
        }
        let Some(changes) = controller.changed_fields_since(self.projection.model_revision) else {
            // A structural command may have cleared the bounded field journal.
            // Losing delta history must not replace this document's live World.
            return self.synchronize_structure(controller);
        };

        // Reject the final aggregate before decoding or cloning any geometry.
        // Property-only deltas do not inspect unrelated roots or geometry.
        let mut geometry_bytes = self.projection.geometry_bytes;
        for (item, fields) in &changes {
            if property_only(fields) {
                continue;
            }
            let root = controller
                .object_root(item)
                .map_err(|cause| DesktopError::Renderer(cause.to_string()))?;
            let old = self
                .scene
                .object(scene_identity(item)?)
                .ok_or_else(|| error("missing scene mesh descriptor"))?;
            let replacement_bytes = editor_scene_mesh_geometry_bytes(root)?;
            geometry_bytes = geometry_bytes
                .checked_sub(old.positions.len() as u64 * 12 + old.indices.len() as u64 * 4)
                .and_then(|bytes| bytes.checked_add(replacement_bytes))
                .ok_or_else(|| error("invalid scene geometry byte count"))?;
        }
        check_geometry_budget(geometry_bytes)?;

        // Every root/entity/geometry is checked before publishing any object.
        let mut models = Vec::new();
        let mut visibility = Vec::new();
        let mut replacements = Vec::new();
        let mut ecs_updates = Vec::with_capacity(changes.len());
        for (item, fields) in &changes {
            let root = controller
                .object_root(item)
                .map_err(|cause| DesktopError::Renderer(cause.to_string()))?;
            let entity = self
                .projection
                .bridge
                .entity_for_object(item)
                .ok_or_else(|| error("missing authoring entity mapping"))?;
            if self
                .projection
                .world
                .get::<EditorSceneMesh>(entity)
                .is_none()
            {
                return Err(error("missing scene mesh ECS component"));
            }
            let id = scene_identity(item)?;
            if property_only(fields) {
                let matrix = fields
                    .contains("matrix")
                    .then(|| editor_scene_mesh_matrix(root))
                    .transpose()?;
                let visible = fields
                    .contains("visible")
                    .then(|| mesh_visibility(root))
                    .transpose()?;
                if let Some(matrix) = matrix {
                    models.push((id, matrix));
                }
                if let Some(visible) = visible {
                    visibility.push((id, visible));
                }
                ecs_updates.push((entity, MeshUpdate::Properties { matrix, visible }));
            } else {
                let mesh = decode_mesh(root)?;
                let old = self
                    .scene
                    .object(id)
                    .ok_or_else(|| error("missing scene mesh descriptor"))?;
                replacements.push(descriptor(item, mesh.clone(), old.outliner)?);
                ecs_updates.push((entity, MeshUpdate::Mesh(mesh)));
            }
        }
        self.scene
            .update_document_objects(replacements, &models, &visibility)?;
        let mut geometry = 0u64;
        for (entity, update) in ecs_updates {
            // Private World and complete preflight make this infallible.
            let mut mesh = self
                .projection
                .world
                .get_mut::<EditorSceneMesh>(entity)
                .expect("scene session owns prevalidated live ECS entities");
            match update {
                MeshUpdate::Properties { matrix, visible } => {
                    if let Some(matrix) = matrix {
                        mesh.matrix = matrix;
                    }
                    if let Some(visible) = visible {
                        mesh.visible = visible;
                    }
                }
                MeshUpdate::Mesh(value) => {
                    *mesh = value;
                    geometry += 1;
                }
            }
        }
        self.projection.model_revision = controller.model_revision();
        self.projection.geometry_bytes = geometry_bytes;
        self.work.objects_synchronized = self
            .work
            .objects_synchronized
            .saturating_add(changes.len() as u64);
        self.work.geometry_replacements = self.work.geometry_replacements.saturating_add(geometry);
        self.work.matrix_updates = self.work.matrix_updates.saturating_add(models.len() as u64);
        self.work.visibility_updates = self
            .work
            .visibility_updates
            .saturating_add(visibility.len() as u64);
        Ok(true)
    }

    fn synchronize_structure(
        &mut self,
        controller: &EditorController,
    ) -> Result<bool, DesktopError> {
        // Structural commands clear the field journal. Inspect the complete
        // final document so property edits on either side of a command publish
        // in the same transaction. Admit all lengths before touching elements.
        let geometry_bytes = document_geometry_bytes(
            controller
                .document_objects()
                .map(|(item, root, _)| (item, root)),
        )?;
        let mut rows = BTreeMap::new();
        for row in controller.document_outliner_nodes() {
            if let Some(item) = row.selection()
                && rows.insert(item, row.id()).is_some()
            {
                return Err(error("scene object has multiple Outliner identities"));
            }
        }
        let object_count = controller.document_objects().len();
        let mut objects = Vec::with_capacity(object_count);
        let mut updates = Vec::new();
        let mut geometry_updates = 0u64;
        let mut matrix_updates = 0u64;
        let mut visibility_updates = 0u64;
        for (item, root, _) in controller.document_objects() {
            editor_scene_mesh_validate(root)?;
            let id = scene_identity(item)?;
            let outliner = rows.get(item).copied();
            match (
                self.projection.bridge.entity_for_object(item),
                self.scene.object(id),
            ) {
                (Some(entity), Some(old)) => {
                    if self
                        .projection
                        .world
                        .get::<EditorSceneMesh>(entity)
                        .is_none()
                    {
                        return Err(error("missing scene mesh ECS component"));
                    }
                    // Compare with the published authoring projection, not the
                    // runtime model: unchanged authored fields must not clobber
                    // runtime values. No whole-model decode of unchanged data.
                    let matrix = editor_scene_mesh_matrix(root)?;
                    let visible = mesh_visibility(root)?;
                    let matrix_change = (matrix != old.model).then_some(matrix);
                    let visible_change = (visible != old.visible).then_some(visible);
                    let geometry = if geometry_matches(root, old) {
                        let mut object = old.clone();
                        object.outliner = outliner;
                        object.model = matrix;
                        object.visible = visible;
                        objects.push(object);
                        None
                    } else {
                        let mesh = decode_mesh(root)?;
                        objects.push(descriptor(item, mesh.clone(), outliner)?);
                        geometry_updates += 1;
                        Some(mesh)
                    };
                    if geometry.is_none() {
                        matrix_updates += u64::from(matrix_change.is_some());
                        visibility_updates += u64::from(visible_change.is_some());
                    }
                    if geometry.is_some() || matrix_change.is_some() || visible_change.is_some() {
                        updates.push((entity, geometry, matrix_change, visible_change));
                    }
                }
                (None, None) => objects.push(descriptor(item, decode_mesh(root)?, outliner)?),
                _ => return Err(error("inconsistent scene mesh identity mapping")),
            }
        }
        // All fallible scene validation, including descriptor bindings and
        // generation limits, precedes mutation of the retained World/bridge.
        self.scene.stage_document(
            controller,
            DesktopSceneDocument {
                objects,
                camera: self.camera,
                depth: self.depth,
            },
        )?;
        if let Err(cause) = self
            .projection
            .bridge
            .synchronize_structure(&mut self.projection.world, controller)
        {
            self.scene.discard_document();
            return Err(DesktopError::Renderer(cause.to_string()));
        }
        for (entity, geometry, matrix, visible) in updates {
            // The private World exposes no host insertion hooks; the bridge
            // preserves prevalidated survivors and their runtime components.
            let mut mesh = self
                .projection
                .world
                .get_mut::<EditorSceneMesh>(entity)
                .expect("scene session owns prevalidated live ECS entities");
            if let Some(geometry) = geometry {
                mesh.positions = geometry.positions;
                mesh.indices = geometry.indices;
            }
            if let Some(matrix) = matrix {
                mesh.matrix = matrix;
            }
            if let Some(visible) = visible {
                mesh.visible = visible;
            }
        }
        self.scene.commit_document();
        self.projection.shape_revision = controller.shape_revision();
        self.projection.model_revision = controller.model_revision();
        self.projection.object_count = object_count;
        self.projection.geometry_bytes = geometry_bytes;
        self.work.document_rebuilds = self.work.document_rebuilds.saturating_add(1);
        self.work.objects_synchronized = self
            .work
            .objects_synchronized
            .saturating_add(object_count as u64);
        self.work.geometry_replacements = self
            .work
            .geometry_replacements
            .saturating_add(geometry_updates);
        self.work.matrix_updates = self.work.matrix_updates.saturating_add(matrix_updates);
        self.work.visibility_updates = self
            .work
            .visibility_updates
            .saturating_add(visibility_updates);
        Ok(true)
    }
}

enum MeshUpdate {
    Properties {
        matrix: Option<SceneIdMatrix>,
        visible: Option<bool>,
    },
    Mesh(EditorSceneMesh),
}

fn property_only(fields: &alloc::collections::BTreeSet<String>) -> bool {
    !fields.is_empty()
        && fields
            .iter()
            .all(|field| field == "matrix" || field == "visible")
}

fn check_geometry_budget(bytes: u64) -> Result<(), DesktopError> {
    if bytes > MAX_GEOMETRY_BYTES {
        return Err(error("desktop scene exceeds 64 MiB of vertex/index data"));
    }
    Ok(())
}

// Complete admission precedes every reflected conversion, descriptor clone
// and ECS allocation. Keeping the borrowed scan separate makes that ordering
// testable with synthetic lists that reject cloning or reading any element.
pub(crate) fn document_geometry_bytes<'a>(
    roots: impl ExactSizeIterator<Item = (&'a SelectionItem, &'a dyn Reflect)>,
) -> Result<u64, DesktopError> {
    if roots.len() > MAX_OBJECTS {
        return Err(error("desktop scene exceeds 4096 objects"));
    }
    let mut bytes = 0u64;
    for (item, root) in roots {
        scene_identity(item)?;
        bytes = bytes
            .checked_add(editor_scene_mesh_geometry_bytes(root)?)
            .ok_or_else(|| error("scene geometry byte count overflow"))?;
        check_geometry_budget(bytes)?;
    }
    Ok(bytes)
}

// Deliberately inspect only the requested scalar. FromReflect would decode and
// allocate both geometry vectors even when the document changed visibility only.
fn mesh_visibility(root: &dyn Reflect) -> Result<bool, DesktopError> {
    if let Some(mesh) = root.downcast_ref::<EditorSceneMesh>() {
        return Ok(mesh.visible);
    }
    if root.type_name() != core::any::type_name::<EditorSceneMesh>() {
        return Err(error("unexpected scene mesh schema"));
    }
    let ReflectRef::Struct(fields) = root.reflect_ref() else {
        return Err(error("scene mesh must be a reflected struct"));
    };
    fields
        .field("visible")
        .and_then(|field| field.downcast_ref::<bool>())
        .copied()
        .ok_or_else(|| error("scene mesh visibility must be bool"))
}

fn build_projection(
    controller: &EditorController,
    camera: SceneIdMatrix,
    depth: SceneIdDepth,
) -> Result<(Projection, DesktopSceneDocument), DesktopError> {
    if controller.has_active_edit() {
        return Err(error("cannot project an active document edit"));
    }
    let geometry_bytes = document_geometry_bytes(
        controller
            .document_objects()
            .map(|(item, root, _)| (item, root)),
    )?;
    let mut rows = BTreeMap::new();
    for row in controller.document_outliner_nodes() {
        if let Some(item) = row.selection()
            && rows.insert(item, row.id()).is_some()
        {
            return Err(error("scene object has multiple Outliner identities"));
        }
    }
    let mut objects = Vec::with_capacity(controller.document_objects().len());
    for (item, root, _) in controller.document_objects() {
        let mesh = decode_mesh(root)?;
        objects.push(descriptor(item, mesh, rows.get(item).copied())?);
    }
    // Fresh World isolates failed preparation and component lifecycle effects
    // from the currently published document projection.
    let mut world = World::new();
    world.register_reflect_component::<EditorSceneMesh>();
    let bridge = EditorEcsDocumentBridge::materialize(&mut world, controller)
        .map_err(|cause| DesktopError::Renderer(cause.to_string()))?;
    Ok((
        Projection {
            world,
            bridge,
            owner: controller.document_identity(),
            shape_revision: controller.shape_revision(),
            model_revision: controller.model_revision(),
            object_count: controller.document_objects().len(),
            geometry_bytes,
        },
        DesktopSceneDocument {
            objects,
            camera,
            depth,
        },
    ))
}

fn scene_identity(item: &SelectionItem) -> Result<SceneObjectId, DesktopError> {
    match item.kind() {
        SelectionKind::Entity(id) => Ok(SceneObjectId::new(
            EntityId::from_raw(*id),
            MeshId::from_raw(*id),
            SubobjectId::from_raw(0),
        )),
        _ => Err(error(
            "indexed-mesh scene roots require stable entity selections",
        )),
    }
}

fn decode_mesh(root: &dyn Reflect) -> Result<EditorSceneMesh, DesktopError> {
    // Unregistered external hosts must also pass borrowed admission before
    // FromReflect allocates vectors from untrusted dynamic list lengths.
    editor_scene_mesh_validate(root)?;
    let mesh = EditorSceneMesh::from_reflect(root)
        .ok_or_else(|| error("document root is not a registered EditorSceneMesh"))?;
    Ok(mesh)
}

fn descriptor(
    item: &SelectionItem,
    mesh: EditorSceneMesh,
    outliner: Option<prism_editor_panels::OutlinerId>,
) -> Result<DesktopSceneObject, DesktopError> {
    Ok(DesktopSceneObject {
        id: scene_identity(item)?,
        selection: item.clone(),
        outliner,
        positions: mesh.positions,
        indices: mesh.indices,
        model: mesh.matrix,
        visible: mesh.visible,
        sort_key: 0,
    })
}

// Mesh validation has already checked the schema and every element. Compare
// borrowed geometry to the published descriptor without FromReflect allocating
// temporary vectors for unchanged typed or codec-decoded dynamic roots.
fn geometry_matches(root: &dyn Reflect, old: &DesktopSceneObject) -> bool {
    if let Some(mesh) = root.downcast_ref::<EditorSceneMesh>() {
        return mesh.positions == old.positions && mesh.indices == old.indices;
    }
    let ReflectRef::Struct(fields) = root.reflect_ref() else {
        return false;
    };
    fields
        .field("positions")
        .is_some_and(|value| geometry_value_matches(value, &old.positions))
        && fields
            .field("indices")
            .is_some_and(|value| geometry_value_matches(value, &old.indices))
}

fn geometry_value_matches(value: &dyn Reflect, old: &dyn Reflect) -> bool {
    match (value.reflect_ref(), old.reflect_ref()) {
        (ReflectRef::List(value), ReflectRef::List(old)) => {
            value.len() == old.len()
                && (0..value.len()).all(|i| match (value.get(i), old.get(i)) {
                    (Some(value), Some(old)) => geometry_value_matches(value, old),
                    _ => false,
                })
        }
        (ReflectRef::Array(value), ReflectRef::Array(old)) => {
            value.len() == old.len()
                && (0..value.len()).all(|i| match (value.get(i), old.get(i)) {
                    (Some(value), Some(old)) => geometry_value_matches(value, old),
                    _ => false,
                })
        }
        _ => {
            value
                .downcast_ref::<f32>()
                .zip(old.downcast_ref::<f32>())
                .is_some_and(|(value, old)| value == old)
                || value
                    .downcast_ref::<u32>()
                    .zip(old.downcast_ref::<u32>())
                    .is_some_and(|(value, old)| value == old)
        }
    }
}

fn error(message: &str) -> DesktopError {
    DesktopError::Renderer(message.to_owned())
}

impl DesktopRenderer for EditorSceneSession {
    fn gesture_active(&self) -> bool {
        self.scene.gesture_active()
    }

    fn gesture_press(
        &mut self,
        ui: &mut EditorDesktopUi,
        position: [f32; 2],
    ) -> Result<bool, DesktopError> {
        self.synchronize(ui.controller())?;
        self.scene.gesture_press(ui, position)
    }

    fn gesture_move(
        &mut self,
        ui: &mut EditorDesktopUi,
        position: [f32; 2],
    ) -> Result<bool, DesktopError> {
        self.scene.gesture_move(ui, position)
    }

    fn gesture_release(&mut self, ui: &mut EditorDesktopUi) -> Result<bool, DesktopError> {
        let changed = self.scene.gesture_release(ui)?;
        self.synchronize(ui.controller())?;
        Ok(changed)
    }

    fn gesture_cancel(&mut self, ui: &mut EditorDesktopUi) -> Result<bool, DesktopError> {
        self.scene.gesture_cancel(ui)
    }

    fn transform_tool(&mut self, mode: GizmoMode) -> Result<bool, DesktopError> {
        self.scene.transform_tool(mode)
    }

    fn initialize(&mut self, gpu: &DesktopGpu) -> Result<(), DesktopError> {
        self.scene.initialize(gpu)
    }

    fn encode(
        &mut self,
        gpu: &DesktopGpu,
        ui: &mut EditorDesktopUi,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        bounds: DesktopViewportBounds,
    ) -> Result<(), DesktopError> {
        self.synchronize(ui.controller())?;
        self.scene.encode(gpu, ui, encoder, target, bounds)
    }

    fn pointer(
        &mut self,
        gpu: &DesktopGpu,
        ui: &mut EditorDesktopUi,
        position: [f32; 2],
        mode: SelectionMode,
    ) -> Result<bool, DesktopError> {
        self.synchronize(ui.controller())?;
        self.scene.pointer(gpu, ui, position, mode)
    }

    fn box_select(
        &mut self,
        gpu: &DesktopGpu,
        ui: &mut EditorDesktopUi,
        start: [f32; 2],
        end: [f32; 2],
        mode: SelectionMode,
    ) -> Result<bool, DesktopError> {
        self.synchronize(ui.controller())?;
        self.scene.box_select(gpu, ui, start, end, mode)
    }

    fn poll(
        &mut self,
        gpu: &DesktopGpu,
        ui: &mut EditorDesktopUi,
    ) -> Result<DesktopWork, DesktopError> {
        let changed = self.synchronize(ui.controller())?;
        let mut work = self.scene.poll(gpu, ui)?;
        work.redraw |= changed;
        Ok(work)
    }

    fn prepare_document(&mut self, controller: &EditorController) -> Result<(), DesktopError> {
        self.pending = None;
        self.scene.discard_document();
        let (projection, document) = build_projection(controller, self.camera, self.depth)?;
        self.scene.stage_document(controller, document)?;
        self.pending = Some(projection);
        Ok(())
    }

    fn discard_document(&mut self) {
        self.pending = None;
        self.scene.discard_document();
    }

    fn commit_document(&mut self) {
        if let Some(projection) = self.pending.take() {
            self.scene.commit_document();
            self.work.document_rebuilds = self.work.document_rebuilds.saturating_add(1);
            self.work.objects_synchronized = self
                .work
                .objects_synchronized
                .saturating_add(projection.object_count as u64);
            self.projection = projection;
        }
    }

    fn document_reloaded(&mut self) {
        self.scene.document_reloaded();
    }

    fn suspend(&mut self) {
        self.scene.suspend();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs::prelude::{ChildOf, Component};
    use prism_editor_panels::{OutlinerId, OutlinerNode};
    use prism_reflect::DynamicStruct;

    #[derive(Component, Debug, PartialEq)]
    struct RuntimeOnly(u32);

    fn mesh() -> EditorSceneMesh {
        EditorSceneMesh {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            indices: vec![0, 1, 2],
            matrix: SCENE_ID_IDENTITY_MATRIX,
            visible: true,
        }
    }

    fn register(controller: &mut EditorController, id: u128) {
        controller
            .register_object(
                SelectionItem::entity(id),
                Box::new(mesh()),
                crate::editor_scene_mesh_registry(),
            )
            .unwrap();
    }

    #[test]
    fn structural_publication_preserves_runtime_components_and_unauthored_model_values() {
        let mut controller = EditorController::new();
        register(&mut controller, 1);
        let mut session = EditorSceneSession::new(&mut controller).unwrap();
        let world = session.world().id();
        let entity = session
            .bridge()
            .entity_for_object(&SelectionItem::entity(1))
            .unwrap();
        session.projection.world.insert(entity, RuntimeOnly(91));
        let mut runtime = mesh();
        runtime.matrix[3][2] = 0.75;
        runtime.visible = false;
        runtime.positions[0][1] = 0.25;
        *session
            .projection
            .world
            .get_mut::<EditorSceneMesh>(entity)
            .unwrap() = runtime.clone();
        let geometry = session
            .world()
            .get::<EditorSceneMesh>(entity)
            .unwrap()
            .positions
            .as_ptr();
        controller
            .register_outliner_node(OutlinerNode::new(OutlinerId::from_raw(10), None, "Group"))
            .unwrap();
        assert!(session.synchronize(&controller).unwrap());
        assert_eq!(session.world().id(), world);
        assert_eq!(
            session
                .bridge()
                .entity_for_object(&SelectionItem::entity(1)),
            Some(entity)
        );
        assert_eq!(
            session.world().get::<RuntimeOnly>(entity),
            Some(&RuntimeOnly(91))
        );
        assert_eq!(
            session.world().get::<EditorSceneMesh>(entity),
            Some(&runtime)
        );
        assert_eq!(
            session
                .world()
                .get::<EditorSceneMesh>(entity)
                .unwrap()
                .positions
                .as_ptr(),
            geometry
        );

        let edit = controller
            .prepare_edits(vec![crate::ControllerFieldEdit {
                item: SelectionItem::entity(1),
                path: prism_editor_reflect::FieldPath::parse("positions[1][0]").unwrap(),
                value: prism_editor_ui::ActionValue::float(0.5),
            }])
            .unwrap();
        controller.commit_edits(edit).unwrap();
        controller
            .register_outliner_node(OutlinerNode::new(
                OutlinerId::from_raw(11),
                None,
                "Another group",
            ))
            .unwrap();
        assert!(session.synchronize(&controller).unwrap());
        runtime.positions = mesh().positions;
        runtime.positions[1][0] = 0.5;
        assert_eq!(
            session.world().get::<EditorSceneMesh>(entity),
            Some(&runtime)
        );
        assert_eq!(
            session.world().get::<RuntimeOnly>(entity),
            Some(&RuntimeOnly(91))
        );
        assert_eq!(session.world().id(), world);
        assert!(!session.synchronize(&controller).unwrap());
    }

    #[test]
    fn bridge_rejection_discards_staged_scene_and_does_not_publish_properties_or_revisions() {
        let mut controller = EditorController::new();
        register(&mut controller, 1);
        register(&mut controller, 2);
        let mut session = EditorSceneSession::new(&mut controller).unwrap();
        let survivor = session
            .bridge()
            .entity_for_object(&SelectionItem::entity(1))
            .unwrap();
        let removed = session
            .bridge()
            .entity_for_object(&SelectionItem::entity(2))
            .unwrap();
        let foreign = session.projection.world.spawn(());
        session.projection.world.register_child_of();
        session
            .projection
            .world
            .add_relation::<ChildOf>(foreign, removed);
        let world = session.world().id();
        let work = session.work();
        let generation = session.scene().status().scene_generation;
        let shape = session.projection.shape_revision;
        let model = session.projection.model_revision;
        let edit = controller
            .prepare_edits(vec![crate::ControllerFieldEdit {
                item: SelectionItem::entity(1),
                path: prism_editor_reflect::FieldPath::parse("visible").unwrap(),
                value: prism_editor_ui::ActionValue::bool(false),
            }])
            .unwrap();
        controller.commit_edits(edit).unwrap();
        register(&mut controller, 3);
        controller
            .delete_objects(&[SelectionItem::entity(2)])
            .unwrap();
        for _ in 0..2 {
            assert!(session.synchronize(&controller).is_err());
            assert_eq!(session.world().id(), world);
            assert_eq!(session.work(), work);
            assert_eq!(session.projection.shape_revision, shape);
            assert_eq!(session.projection.model_revision, model);
            assert_eq!(
                session
                    .bridge()
                    .entity_for_object(&SelectionItem::entity(3)),
                None
            );
            assert!(session.world().contains(removed));
            assert!(session.world().contains(foreign));
            assert_eq!(
                session.world().get::<EditorSceneMesh>(survivor),
                Some(&mesh())
            );
            assert_eq!(session.scene().status().scene_generation, generation);
            assert_eq!(session.scene().objects().len(), 2);
            assert!(session
                .scene()
                .objects()
                .iter()
                .all(|object| object.visible));
        }
        // A leaked candidate would publish object 3 and the hidden survivor.
        // DesktopScene reloads its generation even on an empty explicit commit,
        // so check descriptor contents, not the generation of that host action.
        session.scene.commit_document();
        assert!(session
            .scene()
            .objects()
            .iter()
            .all(|object| object.visible));
        assert!(session
            .scene()
            .objects()
            .iter()
            .any(|object| object.selection == SelectionItem::entity(2)));
        assert!(!session
            .scene()
            .objects()
            .iter()
            .any(|object| object.selection == SelectionItem::entity(3)));
        session
            .projection
            .world
            .remove_relation::<ChildOf>(foreign, removed);
        assert!(session.synchronize(&controller).unwrap());
        assert_eq!(session.world().id(), world);
        assert_eq!(
            session
                .bridge()
                .entity_for_object(&SelectionItem::entity(1)),
            Some(survivor)
        );
        assert!(
            !session
                .world()
                .get::<EditorSceneMesh>(survivor)
                .unwrap()
                .visible
        );
        assert!(!session.world().contains(removed));
        assert!(session.world().contains(foreign));
        assert_eq!(session.scene().objects().len(), 2);
        assert_eq!(
            session
                .scene()
                .objects()
                .iter()
                .filter(|object| !object.visible)
                .count(),
            1
        );
        assert!(!session.synchronize(&controller).unwrap());
    }

    #[test]
    fn visibility_access_does_not_require_geometry_or_matrix() {
        let mut root = DynamicStruct::new();
        root.set_represented_type_name(core::any::type_name::<EditorSceneMesh>());
        assert!(mesh_visibility(&root).is_err());
        root.insert("visible", false);
        // A full decode/validation cannot succeed on this root. Scalar access
        // must not inspect any absent geometry or matrix fields.
        assert!(!mesh_visibility(&root).unwrap());
        assert!(decode_mesh(&root).is_err());
        root.insert("visible", true);
        assert!(mesh_visibility(&root).unwrap());
        root.insert("visible", 1_u32);
        assert!(mesh_visibility(&root).is_err());
        root.insert("visible", true);
        root.set_represented_type_name("foreign::Mesh");
        assert!(mesh_visibility(&root).is_err());
        assert!(mesh_visibility(&true).is_err());
    }
}
