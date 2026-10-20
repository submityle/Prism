//! A concrete host schema; none of these fields are assumed by the app.
use prism_editor_app::{
    ControllerFieldEdit, DesktopError, DesktopScene, DesktopSceneObject, EditorController,
    SceneTransformAdapter,
};
use prism_editor_core::SelectionItem;
use prism_editor_panels::{OutlinerId, OutlinerNode};
use prism_editor_reflect::FieldPath;
use prism_editor_ui::{ActionValue, SelectionMode};
use prism_editor_viewport::{
    EntityId, MeshId, SceneIdDepth, SceneIdMatrix, SceneObjectId, SubobjectId, TransformDelta,
    SCENE_ID_IDENTITY_MATRIX,
};
use prism_reflect::{Reflect, TypeRegistry};

#[derive(Clone, Debug, PartialEq, Reflect)]
pub struct ExampleTransform {
    pub matrix: [[f32; 4]; 4],
}

pub fn registry() -> TypeRegistry {
    let mut registry = TypeRegistry::new();
    registry.register::<ExampleTransform>();
    registry.register::<[[f32; 4]; 4]>();
    registry.register::<[f32; 4]>();
    registry.register::<f32>();
    registry
}

pub fn read(root: &dyn Reflect) -> Result<ExampleTransform, DesktopError> {
    let mut value = ExampleTransform {
        matrix: SCENE_ID_IDENTITY_MATRIX,
    };
    value
        .apply(root)
        .map_err(|e| DesktopError::Renderer(e.to_string()))?;
    Ok(value)
}

pub struct ExampleBinding;
impl SceneTransformAdapter for ExampleBinding {
    fn model(
        &self,
        _: &SelectionItem,
        _: SceneObjectId,
        root: &dyn Reflect,
    ) -> Result<SceneIdMatrix, DesktopError> {
        Ok(read(root)?.matrix)
    }

    fn edits(
        &self,
        item: &SelectionItem,
        root: &dyn Reflect,
        delta: TransformDelta,
    ) -> Result<Vec<ControllerFieldEdit>, DesktopError> {
        let mut matrix = read(root)?.matrix;
        // Scale local basis vectors; rotate in world axes about each own pivot.
        for (column, scale) in matrix.iter_mut().take(3).zip(delta.scale_milliunits) {
            if scale <= 0 {
                return Err(DesktopError::Renderer(
                    "example binding requires positive scale".into(),
                ));
            }
            for value in column.iter_mut().take(3) {
                *value *= scale as f32 / 1000.0;
            }
        }
        for axis in 0..3 {
            let angle = (delta.rotation_millidegrees[axis] as f32 / 1000.0).to_radians();
            let (sin, cos) = prism_math::ops::sin_cos(angle);
            let a = (axis + 1) % 3;
            let b = (axis + 2) % 3;
            for column in matrix.iter_mut().take(3) {
                let x = column[a];
                let y = column[b];
                column[a] = x * cos - y * sin;
                column[b] = x * sin + y * cos;
            }
            matrix[3][axis] += delta.translation_milliunits[axis] as f32 / 1000.0;
        }
        let mut edits = Vec::new();
        for (column, values) in matrix.iter().enumerate() {
            for (row, value) in values.iter().enumerate() {
                edits.push(ControllerFieldEdit {
                    item: item.clone(),
                    path: FieldPath::parse(&format!("matrix[{column}][{row}]"))
                        .map_err(|e| DesktopError::Renderer(e.to_string()))?,
                    value: ActionValue::float(f64::from(*value)),
                });
            }
        }
        Ok(edits)
    }
}

pub fn scene() -> Result<(EditorController, DesktopScene), DesktopError> {
    let mut controller = EditorController::new();
    let mut objects = Vec::new();
    for n in 1..=2 {
        let item = SelectionItem::entity(n);
        let mut matrix = SCENE_ID_IDENTITY_MATRIX;
        matrix[3] = [if n == 1 { -0.3 } else { 0.3 }, 0.0, 0.4, 1.0];
        controller
            .register_object(
                item.clone(),
                Box::new(ExampleTransform { matrix }),
                registry(),
            )
            .map_err(|e| DesktopError::Renderer(e.to_string()))?;
        let mut node =
            OutlinerNode::new(OutlinerId::from_raw(n), None, format!("Example mesh {n}"));
        node.set_selection(item.clone());
        controller
            .register_outliner_node(node)
            .map_err(|e| DesktopError::Renderer(e.to_string()))?;
        let mut object = DesktopSceneObject::new(
            SceneObjectId::new(
                EntityId::from_raw(n),
                MeshId::from_raw(n),
                SubobjectId::from_raw(0),
            ),
            item,
            vec![[-0.15, -0.15, 0.0], [0.15, -0.15, 0.0], [0.0, 0.15, 0.0]],
            vec![0, 1, 2],
        );
        object.model = matrix;
        objects.push(object);
    }
    controller
        .select_items(
            &[SelectionItem::entity(1), SelectionItem::entity(2)],
            SelectionMode::Replace,
        )
        .map_err(|e| DesktopError::Renderer(e.to_string()))?;
    let mut scene = DesktopScene::new(objects)?;
    // An oblique camera makes all three world axes visible, including Z.
    let mut camera = SCENE_ID_IDENTITY_MATRIX;
    camera[2][0] = -0.45;
    camera[2][1] = 0.3;
    scene.set_camera(camera, SceneIdDepth::Forward)?;
    scene.set_transform_adapter(ExampleBinding)?;
    Ok((controller, scene))
}
