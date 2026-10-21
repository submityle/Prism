//! Explicit object-authoring commands for the optional indexed-mesh profile.
//!
//! Installation registers factories and a document budget only. Geometry is
//! created solely by an executed command, through atomic structural history.

use alloc::{boxed::Box, vec};
use std::sync::OnceLock;

use prism_editor_panels::OutlinerNode;
use prism_editor_viewport::SCENE_ID_IDENTITY_MATRIX;

use crate::{
    desktop_scene::MAX_GEOMETRY_BYTES, editor_scene_mesh_registry,
    scene_mesh::editor_scene_mesh_geometry_bytes, ControllerAuthoringCommand,
    ControllerDocumentBudget, ControllerError, ControllerObject, EditorController, EditorSceneMesh,
};

/// Semantic command ID for an explicit user-created triangle.
pub const SCENE_ADD_TRIANGLE_COMMAND: &str = "scene.add_triangle";
/// Semantic command ID for deleting the current object selection.
pub const SCENE_DELETE_SELECTION_COMMAND: &str = "scene.delete_selection";

/// Shared callback identity makes constructor/profile installation idempotent.
/// Hidden meshes retain their geometry contribution; unrelated schemas cost zero.
pub(crate) fn scene_document_budget() -> ControllerDocumentBudget {
    static BUDGET: OnceLock<ControllerDocumentBudget> = OnceLock::new();
    BUDGET
        .get_or_init(|| {
            ControllerDocumentBudget::new("scene.geometry_bytes", MAX_GEOMETRY_BYTES, |_, root| {
                if root.type_name() == core::any::type_name::<EditorSceneMesh>() {
                    editor_scene_mesh_geometry_bytes(root).map_err(|error| error.to_string())
                } else {
                    Ok(0)
                }
            })
        })
        .clone()
}

/// Opts a controller into the mesh profile's Add triangle / Delete selection UI.
///
/// Does not create bootstrap objects, change selection, or mark the model dirty.
/// Commands use the controller's normal admission, budgets, and Undo/Redo. An
/// installation error leaves all command registrations and document state
/// unchanged. Reload keeps the host-installed factories, not serialized code.
/// The shared 64 MiB geometry budget includes hidden meshes and is installed
/// atomically with both commands. Other root schemas contribute zero; this
/// does not make them renderable by the indexed-mesh scene adapter.
pub fn install_scene_authoring_commands(
    controller: &mut EditorController,
) -> Result<(), ControllerError> {
    type Factory = fn(&EditorController) -> Result<ControllerAuthoringCommand, ControllerError>;
    let create: Factory = |controller| {
        // The controller includes live, historical and discarded-redo IDs in
        // its lifetime high-water mark. Failed creation does not consume it.
        let (item, row_id) = controller.next_entity_identity()?;
        let mut row = OutlinerNode::new(row_id, None, "Triangle");
        row.set_selection(item.clone());
        Ok(ControllerAuthoringCommand::Create(vec![ControllerObject {
            item,
            root: Box::new(EditorSceneMesh {
                positions: vec![[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.0, 0.5, 0.5]],
                indices: vec![0, 1, 2],
                matrix: SCENE_ID_IDENTITY_MATRIX,
                visible: true,
            }),
            registry: editor_scene_mesh_registry(),
            row,
        }]))
    };
    let delete: Factory = |controller| {
        Ok(ControllerAuthoringCommand::Delete(
            controller.selection().items().to_vec(),
        ))
    };
    controller.register_authoring_profile(
        [
            (SCENE_ADD_TRIANGLE_COMMAND.into(), create),
            (SCENE_DELETE_SELECTION_COMMAND.into(), delete),
        ],
        [scene_document_budget()],
    )
}
