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
    /// Complete document projections installed (initial load included).
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
        if !Arc::ptr_eq(&self.projection.owner, &controller.document_identity())
            || self.projection.shape_revision != controller.shape_revision()
        {
            self.prepare_document(controller)?;
            self.commit_document();
            return Ok(true);
        }
        if self.projection.model_revision == controller.model_revision() {
            return Ok(false);
        }
        let Some(changes) = controller.changed_fields_since(self.projection.model_revision) else {
            self.prepare_document(controller)?;
            self.commit_document();
            return Ok(true);
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
                replacements.push(descriptor(item, &mesh, old.outliner)?);
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
        objects.push(descriptor(item, &mesh, rows.get(item).copied())?);
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
    mesh: &EditorSceneMesh,
    outliner: Option<prism_editor_panels::OutlinerId>,
) -> Result<DesktopSceneObject, DesktopError> {
    Ok(DesktopSceneObject {
        id: scene_identity(item)?,
        selection: item.clone(),
        outliner,
        positions: mesh.positions.clone(),
        indices: mesh.indices.clone(),
        model: mesh.matrix,
        visible: mesh.visible,
        sort_key: 0,
    })
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
    use prism_reflect::DynamicStruct;

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
