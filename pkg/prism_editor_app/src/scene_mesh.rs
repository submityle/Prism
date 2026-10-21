//! Explicit, optional indexed-geometry profile for the desktop editor.
//!
//! The host wires this module behind `desktop`. This is not a gameplay schema,
//! scene assembler, or an implicit contract for arbitrary reflected objects.

use alloc::{format, vec::Vec};
use core::any::{type_name, TypeId};

use prism_ecs::prelude::Component;
use prism_editor_core::SelectionItem;
use prism_editor_reflect::FieldPath;
use prism_editor_ui::ActionValue;
use prism_editor_viewport::{SceneIdMatrix, SceneObjectId, TransformDelta};
use prism_reflect::{Access, Array, List, Reflect, ReflectRef, Struct, TypeRegistry};

use crate::{
    ControllerFieldEdit, ControllerObjectValidator, DesktopError, DocumentError,
    DocumentSchemaCatalog, SceneTransformAdapter,
};

// Match DesktopScene's vertex/index payload budget, not allocator capacity or
// serialized-document size. The host still enforces the aggregate scene limit.
const MAX_GEOMETRY_BYTES: u64 = 64 * 1024 * 1024;

/// An opt-in, business-neutral editable triangle mesh and world transform.
#[derive(Clone, Debug, PartialEq, Reflect, Component)]
pub struct EditorSceneMesh {
    /// Object-space Float32x3 vertices; intentionally editable.
    pub positions: Vec<[f32; 3]>,
    /// Uint32 triangle-list indices; intentionally editable.
    pub indices: Vec<u32>,
    /// Column-major, finite, affine, nonsingular object-to-world matrix.
    pub matrix: [[f32; 4]; 4],
    /// Whether this mesh is drawn and pickable; hidden geometry is validated too.
    pub visible: bool,
}

impl EditorSceneMesh {
    /// Checks geometry bounds and the transform without allocating or cloning.
    ///
    /// Positions and indices must be nonempty, fit u32 counts and together fit
    /// 64 MiB. Indices form complete triangles and reference existing finite
    /// vertices. Degenerate triangles, shared vertices, shear and reflection
    /// are permitted: this profile imposes no topology or gameplay policy.
    pub fn validate(&self) -> Result<(), DesktopError> {
        editor_scene_mesh_validate(self)
    }
}

/// Registers the root and every recursive field/container/leaf type.
/// No readonly metadata is installed: geometry is editable by default.
/// Root TypeData admits complete objects and validates affected edit groups.
#[must_use]
pub fn editor_scene_mesh_registry() -> TypeRegistry {
    let mut registry = TypeRegistry::new();
    registry.register::<EditorSceneMesh>();
    registry.register::<Vec<[f32; 3]>>();
    registry.register::<[f32; 3]>();
    registry.register::<Vec<u32>>();
    registry.register::<u32>();
    registry.register::<[[f32; 4]; 4]>();
    registry.register::<[f32; 4]>();
    registry.register::<f32>();
    registry.register::<bool>();
    registry
        .get_mut(TypeId::of::<EditorSceneMesh>())
        .expect("mesh root was registered")
        .insert(ControllerObjectValidator::new(|root, paths| {
            validate_mesh_edits(root, paths).map_err(|e| e.to_string())
        }));
    registry
}

/// Explicit schema allowlist for version 1 of this optional profile.
pub fn editor_scene_mesh_catalog() -> Result<DocumentSchemaCatalog, DocumentError> {
    let mut catalog = DocumentSchemaCatalog::new();
    catalog.register(
        type_name::<EditorSceneMesh>(),
        1,
        editor_scene_mesh_registry,
    )?;
    Ok(catalog)
}

/// Fully validates a borrowed typed or codec-decoded dynamic mesh.
///
/// Checks the complete schema, geometry budget, every finite vertex, every
/// triangle index, visibility type and affine/nonsingular matrix. No root or
/// geometry is cloned and no `FromReflect` conversion is performed.
pub fn editor_scene_mesh_validate(root: &dyn Reflect) -> Result<(), DesktopError> {
    if root.downcast_ref::<EditorSceneMesh>().is_none() && mesh_fields(root)?.field_count() != 4 {
        return Err(error("EditorSceneMesh must have exactly four fields"));
    }
    validate_geometry(root)?;
    validate_visibility(root)?;
    editor_scene_mesh_matrix(root)?;
    Ok(())
}

/// Counts borrowed vertex/index payload bytes before scene-wide allocation.
///
/// Checks the represented root/container schema, nonempty u32 counts, complete
/// triangles and the single-mesh budget. Never reads list elements, matrix or
/// visibility, nor clones/converts the root. This is not full mesh validation:
/// finite vertices, index bounds and the other fields need separate admission.
pub(crate) fn editor_scene_mesh_geometry_bytes(root: &dyn Reflect) -> Result<u64, DesktopError> {
    if let Some(mesh) = root.downcast_ref::<EditorSceneMesh>() {
        return validate_geometry_lengths(mesh.positions.len(), mesh.indices.len());
    }
    let fields = mesh_fields(root)?;
    if fields.field_count() != 4 {
        return Err(error("EditorSceneMesh must have exactly four fields"));
    }
    let positions = typed_list(field(fields, "positions")?, type_name::<Vec<[f32; 3]>>())?;
    let indices = typed_list(field(fields, "indices")?, type_name::<Vec<u32>>())?;
    validate_geometry_lengths(positions.len(), indices.len())
}

// The controller has validated the previous complete root and supplies ALL
// paths in the final isolated candidate. Geometry fields are coupled; matrix
// and visibility are independent. Never return early after the first path.
fn validate_mesh_edits(root: &dyn Reflect, paths: &[FieldPath]) -> Result<(), DesktopError> {
    if paths.is_empty() {
        return editor_scene_mesh_validate(root);
    }
    let (mut geometry, mut matrix, mut visible) = (false, false, false);
    for path in paths {
        match path.segments().first() {
            Some(Access::Field(name)) => match name.as_str() {
                "positions" | "indices" => geometry = true,
                "matrix" => matrix = true,
                "visible" => visible = true,
                _ => return editor_scene_mesh_validate(root),
            },
            // A root replacement or unfamiliar path cannot take a partial fast path.
            _ => return editor_scene_mesh_validate(root),
        }
    }
    if geometry {
        validate_geometry(root)?;
    }
    if matrix {
        editor_scene_mesh_matrix(root)?;
    }
    if visible {
        validate_visibility(root)?;
    }
    Ok(())
}

fn validate_geometry(root: &dyn Reflect) -> Result<(), DesktopError> {
    if let Some(mesh) = root.downcast_ref::<EditorSceneMesh>() {
        validate_geometry_lengths(mesh.positions.len(), mesh.indices.len())?;
        if mesh.positions.iter().flatten().any(|v| !v.is_finite()) {
            return Err(error("positions must be finite"));
        }
        if mesh
            .indices
            .iter()
            .any(|i| *i as usize >= mesh.positions.len())
        {
            return Err(error("triangle index is out of bounds"));
        }
        return Ok(());
    }
    let fields = mesh_fields(root)?;
    let positions = typed_list(field(fields, "positions")?, type_name::<Vec<[f32; 3]>>())?;
    let indices = typed_list(field(fields, "indices")?, type_name::<Vec<u32>>())?;
    // Admit lengths before any element access, even for synthetic reflected lists.
    validate_geometry_lengths(positions.len(), indices.len())?;
    for i in 0..positions.len() {
        let position = positions.get(i).ok_or_else(|| error("missing position"))?;
        let coordinates = typed_array(position, 3, type_name::<[f32; 3]>())?;
        for axis in 0..3 {
            let value = array_element(coordinates, axis)?
                .downcast_ref::<f32>()
                .ok_or_else(|| error("position elements must be f32"))?;
            if !value.is_finite() {
                return Err(error("positions must be finite"));
            }
        }
    }
    for i in 0..indices.len() {
        let value = indices
            .get(i)
            .and_then(|v| v.downcast_ref::<u32>())
            .ok_or_else(|| error("index elements must be u32"))?;
        if *value as usize >= positions.len() {
            return Err(error("triangle index is out of bounds"));
        }
    }
    Ok(())
}

fn validate_visibility(root: &dyn Reflect) -> Result<(), DesktopError> {
    if root.downcast_ref::<EditorSceneMesh>().is_some() {
        return Ok(());
    }
    field(mesh_fields(root)?, "visible")?
        .downcast_ref::<bool>()
        .ok_or_else(|| error("visible must be bool"))?;
    Ok(())
}

/// Copies and validates only the 16 matrix scalars of a typed or decoded root.
///
/// This never calls root `apply`/`FromReflect`/`reflect_clone`, nor reads any
/// geometry or visibility field. Full geometry validation is a separate step.
pub fn editor_scene_mesh_matrix(root: &dyn Reflect) -> Result<SceneIdMatrix, DesktopError> {
    let matrix = if let Some(mesh) = root.downcast_ref::<EditorSceneMesh>() {
        mesh.matrix
    } else {
        let fields = mesh_fields(root)?;
        let columns = typed_array(field(fields, "matrix")?, 4, type_name::<[[f32; 4]; 4]>())?;
        let mut matrix = [[0.0; 4]; 4];
        for (column, values) in matrix.iter_mut().enumerate() {
            let rows = typed_array(array_element(columns, column)?, 4, type_name::<[f32; 4]>())?;
            for (row, value) in values.iter_mut().enumerate() {
                *value = *array_element(rows, row)?
                    .downcast_ref::<f32>()
                    .ok_or_else(|| error("matrix elements must be f32"))?;
            }
        }
        matrix
    };
    validate_matrix(&matrix)?;
    Ok(matrix)
}

/// Stateless cumulative TRS adapter for the explicit EditorSceneMesh profile.
/// The caller supplies the unchanged drag-start root on each preview.
pub struct EditorSceneMeshTransform;

impl SceneTransformAdapter for EditorSceneMeshTransform {
    fn model(
        &self,
        _: &SelectionItem,
        _: SceneObjectId,
        root: &dyn Reflect,
    ) -> Result<SceneIdMatrix, DesktopError> {
        editor_scene_mesh_matrix(root)
    }

    fn edits(
        &self,
        item: &SelectionItem,
        root: &dyn Reflect,
        delta: TransformDelta,
    ) -> Result<Vec<ControllerFieldEdit>, DesktopError> {
        let mut matrix = editor_scene_mesh_matrix(root)?;
        if delta.scale_milliunits.iter().any(|scale| *scale <= 0) {
            return Err(error(
                "scale milliunits must be strictly positive (1000 is identity)",
            ));
        }
        // Same arithmetic and order as examples/native_gizmo/support.rs:
        // local basis scale, world X/Y/Z rotations about the object's own pivot,
        // then world translation. Translation is never scaled or rotated.
        for (column, scale) in matrix.iter_mut().take(3).zip(delta.scale_milliunits) {
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
        // Reject overflow, underflow-to-singularity and invalid source/output
        // before constructing any edits. An Err never exposes a partial batch.
        validate_matrix(&matrix)?;
        let mut edits = Vec::with_capacity(16);
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

fn mesh_fields(root: &dyn Reflect) -> Result<&dyn Struct, DesktopError> {
    if root.type_name() != type_name::<EditorSceneMesh>() {
        return Err(error("root must represent EditorSceneMesh"));
    }
    match root.reflect_ref() {
        ReflectRef::Struct(fields) => Ok(fields),
        _ => Err(error("EditorSceneMesh root must be a struct")),
    }
}

fn field<'a>(fields: &'a dyn Struct, name: &str) -> Result<&'a dyn Reflect, DesktopError> {
    fields
        .field(name)
        .ok_or_else(|| error("missing mesh field"))
}

fn typed_array<'a>(
    value: &'a dyn Reflect,
    length: usize,
    name: &str,
) -> Result<&'a dyn Array, DesktopError> {
    if value.type_name() != name {
        return Err(error("mesh field has an invalid array type"));
    }
    match value.reflect_ref() {
        ReflectRef::Array(array) if array.len() == length => Ok(array),
        _ => Err(error("mesh field has an invalid array shape")),
    }
}

fn array_element(array: &dyn Array, index: usize) -> Result<&dyn Reflect, DesktopError> {
    array
        .get(index)
        .ok_or_else(|| error("missing array element"))
}

fn typed_list<'a>(value: &'a dyn Reflect, name: &str) -> Result<&'a dyn List, DesktopError> {
    if value.type_name() != name {
        return Err(error("mesh field has an invalid list type"));
    }
    match value.reflect_ref() {
        ReflectRef::List(list) => Ok(list),
        _ => Err(error("mesh field must be a list")),
    }
}

fn validate_geometry_lengths(positions: usize, indices: usize) -> Result<u64, DesktopError> {
    if positions == 0
        || indices == 0
        || !indices.is_multiple_of(3)
        || u32::try_from(positions).is_err()
        || u32::try_from(indices).is_err()
    {
        return Err(error(
            "geometry must be a nonempty bounded indexed triangle list",
        ));
    }
    let bytes = (positions as u64)
        .checked_mul(12)
        .and_then(|vertices| (indices as u64).checked_mul(4)?.checked_add(vertices))
        .ok_or_else(|| error("geometry byte count overflow"))?;
    if bytes > MAX_GEOMETRY_BYTES {
        return Err(error("mesh exceeds 64 MiB of vertex/index data"));
    }
    Ok(bytes)
}

fn validate_matrix(matrix: &SceneIdMatrix) -> Result<(), DesktopError> {
    if matrix.iter().flatten().any(|v| !v.is_finite()) {
        return Err(error("matrix must be finite"));
    }
    if matrix[0][3] != 0.0 || matrix[1][3] != 0.0 || matrix[2][3] != 0.0 || matrix[3][3] != 1.0 {
        return Err(error("matrix must be affine with last row [0, 0, 0, 1]"));
    }
    if singular_basis(matrix) {
        return Err(error("matrix must be nonsingular"));
    }
    Ok(())
}

fn singular_basis(matrix: &SceneIdMatrix) -> bool {
    // A plain f64 determinant can round a singular f32 basis to nonzero.
    // Each product of TWO f32 inputs is exact in f64; split the third factor's
    // product with FMA, then sum all twelve parts as a zero-eliminated exact
    // expansion. Finite f32 triple products and their residuals stay well
    // inside f64's exponent range. No epsilon or minimum scale is imposed.
    let a = matrix[0].map(f64::from);
    let b = matrix[1].map(f64::from);
    let c = matrix[2].map(f64::from);
    let terms = [
        (a[0], b[1], c[2]),
        (b[0], c[1], a[2]),
        (c[0], a[1], b[2]),
        (-c[0], b[1], a[2]),
        (-b[0], a[1], c[2]),
        (-a[0], c[1], b[2]),
    ];
    let mut parts = [0.0; 12];
    let mut length = 0;
    for (x, y, z) in terms {
        let xy = x * y;
        let high = xy * z;
        let low = xy.mul_add(z, -high);
        add_exact(&mut parts, &mut length, low);
        add_exact(&mut parts, &mut length, high);
    }
    length == 0
}

fn add_exact(parts: &mut [f64; 12], length: &mut usize, mut carry: f64) {
    let mut written = 0;
    for i in 0..*length {
        let value = parts[i];
        // General TwoSum: sum + residual equals carry + value exactly.
        let sum = carry + value;
        let value_rounded = sum - carry;
        let carry_rounded = sum - value_rounded;
        let residual = (carry - carry_rounded) + (value - value_rounded);
        if residual != 0.0 {
            parts[written] = residual;
            written += 1;
        }
        carry = sum;
    }
    if carry != 0.0 {
        parts[written] = carry;
        written += 1;
    }
    *length = written;
}

fn error(message: &str) -> DesktopError {
    DesktopError::Renderer(format!("editor scene mesh: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ControllerLimits, DocumentCodec, DocumentLimits, EditorController};
    use prism_editor_ui::EditorAction;
    use prism_editor_viewport::{EntityId, MeshId, SubobjectId, SCENE_ID_IDENTITY_MATRIX};
    use prism_reflect::{
        DynamicArray, DynamicList, DynamicStruct, FromReflect, ReflectMut, TypeInfo, Typed,
    };
    use std::any::Any;

    fn triangle() -> EditorSceneMesh {
        EditorSceneMesh {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            indices: vec![0, 1, 2],
            matrix: SCENE_ID_IDENTITY_MATRIX,
            visible: true,
        }
    }

    fn identity_delta() -> TransformDelta {
        TransformDelta {
            scale_milliunits: [1000; 3],
            ..TransformDelta::default()
        }
    }

    fn object_validator() -> ControllerObjectValidator {
        // The same TypeData survives registry cloning and is used by the codec.
        let registry = editor_scene_mesh_registry();
        let registration = registry
            .get_with_name(type_name::<EditorSceneMesh>())
            .unwrap()
            .clone();
        registration
            .data::<ControllerObjectValidator>()
            .unwrap()
            .clone()
    }

    fn paths(names: &[&str]) -> Vec<FieldPath> {
        names
            .iter()
            .map(|name| FieldPath::parse(name).unwrap())
            .collect()
    }

    fn decoded(mesh: &EditorSceneMesh) -> EditorController {
        let mut controller = EditorController::new();
        controller
            .register_object(
                SelectionItem::entity(1),
                Box::new(mesh.clone()),
                editor_scene_mesh_registry(),
            )
            .unwrap();
        let catalog = editor_scene_mesh_catalog().unwrap();
        let bytes =
            DocumentCodec::encode(&controller, &catalog, DocumentLimits::default()).unwrap();
        DocumentCodec::decode(
            &bytes,
            &catalog,
            DocumentLimits::default(),
            ControllerLimits::default(),
        )
        .unwrap()
        .into_controller()
    }

    fn edit_matrix(root: &dyn Reflect, delta: TransformDelta) -> SceneIdMatrix {
        let item = SelectionItem::entity(1);
        let edits = EditorSceneMeshTransform.edits(&item, root, delta).unwrap();
        assert_eq!(edits.len(), 16);
        let mut matrix = [[0.0; 4]; 4];
        for (i, edit) in edits.iter().enumerate() {
            assert_eq!(edit.item, item);
            assert_eq!(
                edit.path,
                FieldPath::parse(&format!("matrix[{}][{}]", i / 4, i % 4)).unwrap()
            );
            matrix[i / 4][i % 4] = edit.value.as_float().unwrap() as f32;
        }
        matrix
    }

    fn assert_close(actual: SceneIdMatrix, expected: SceneIdMatrix) {
        for (a, b) in actual.iter().flatten().zip(expected.iter().flatten()) {
            assert!((a - b).abs() < 1e-5, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn complete_registry_and_component_contract() {
        fn component<T: Component>() {}
        component::<EditorSceneMesh>();
        let registry = editor_scene_mesh_registry();
        for name in [
            type_name::<EditorSceneMesh>(),
            type_name::<Vec<[f32; 3]>>(),
            type_name::<[f32; 3]>(),
            type_name::<Vec<u32>>(),
            type_name::<u32>(),
            type_name::<[[f32; 4]; 4]>(),
            type_name::<[f32; 4]>(),
            type_name::<f32>(),
            type_name::<bool>(),
        ] {
            assert!(registry.get_with_name(name).is_some(), "{name}");
        }
    }

    #[test]
    fn codec_dynamic_root_and_typed_model_have_the_same_matrix_and_geometry() {
        let mut mesh = triangle();
        mesh.matrix[3] = [2.0, -3.0, 4.0, 1.0];
        mesh.visible = false;
        let controller = decoded(&mesh);
        let root = controller.object_root(&SelectionItem::entity(1)).unwrap();
        assert!(root.downcast_ref::<EditorSceneMesh>().is_none());
        assert!(root.downcast_ref::<DynamicStruct>().is_some());
        assert_eq!(editor_scene_mesh_matrix(root).unwrap(), mesh.matrix);
        let id = SceneObjectId::new(
            EntityId::from_raw(1),
            MeshId::from_raw(2),
            SubobjectId::from_raw(0),
        );
        for root in [&mesh as &dyn Reflect, root] {
            assert_eq!(editor_scene_mesh_geometry_bytes(root).unwrap(), 48);
            editor_scene_mesh_validate(root).unwrap();
            object_validator().validate(root, &[]).unwrap();
            assert_eq!(
                EditorSceneMeshTransform
                    .model(&SelectionItem::entity(1), id, root)
                    .unwrap(),
                mesh.matrix
            );
        }
        let restored = EditorSceneMesh::from_reflect(root).unwrap();
        restored.validate().unwrap();
        assert_eq!(restored, mesh);
    }

    #[test]
    fn full_borrowed_validator_rejects_invalid_typed_and_dynamic_meshes() {
        let validator = object_validator();
        for case in 0..9 {
            let mut mesh = triangle();
            mesh.visible = false;
            match case {
                0 => mesh.positions.clear(),
                1 => mesh.indices.clear(),
                2 => mesh.indices.push(0),
                3 => mesh.indices[2] = 3,
                4 => mesh.positions[1][2] = f32::NAN,
                5 => mesh.positions[1][2] = f32::INFINITY,
                6 => mesh.positions[1][2] = f32::NEG_INFINITY,
                7 => mesh.matrix[0][0] = 0.0,
                _ => mesh.matrix[3][3] = 2.0,
            }
            let dynamic = mesh.reflect_clone(); // fixture construction, not validation
            for root in [&mesh as &dyn Reflect, &*dynamic] {
                assert!(editor_scene_mesh_validate(root).is_err(), "case {case}");
                assert!(validator.validate(root, &[]).is_err(), "case {case}");
                assert!(validator.validate(root, &[FieldPath::root()]).is_err());
            }
        }
    }

    #[test]
    fn full_dynamic_validator_checks_schema_and_geometry_leaf_types() {
        let mesh = triangle();
        let mut root = dynamic_root(mesh.matrix);
        root.insert("positions", mesh.positions.clone());
        root.insert("indices", mesh.indices.clone());
        root.insert("visible", true);
        editor_scene_mesh_validate(&root).unwrap();
        root.insert("extra", 0_u32);
        assert!(editor_scene_mesh_validate(&root).is_err());
        root.remove("extra");
        root.insert("visible", 1_u32);
        assert!(editor_scene_mesh_validate(&root).is_err());
        root.insert("visible", true);
        root.insert("positions", vec![[0.0_f64; 3]; 3]);
        assert!(editor_scene_mesh_validate(&root).is_err());
        let mut positions = DynamicList::new();
        positions.set_represented_type_name(type_name::<Vec<[f32; 3]>>());
        let mut position = DynamicArray::new();
        position.set_represented_type_name(type_name::<[f32; 3]>());
        position.push_value(0.0_f32);
        position.push_value(0.0_f64);
        position.push_value(0.0_f32);
        positions.push_value(position);
        root.insert("positions", positions);
        assert!(editor_scene_mesh_validate(&root).is_err());
        let mut positions = DynamicList::new();
        positions.set_represented_type_name(type_name::<Vec<[f32; 3]>>());
        positions.push_value([0.0_f32; 2]);
        root.insert("positions", positions);
        assert!(editor_scene_mesh_validate(&root).is_err());
        root.insert("positions", mesh.positions);
        let mut indices = DynamicList::new();
        indices.set_represented_type_name(type_name::<Vec<u32>>());
        indices.push_value(0_i32);
        indices.push_value(1_u32);
        indices.push_value(2_u32);
        root.insert("indices", indices);
        assert!(editor_scene_mesh_validate(&root).is_err());
        root.remove("indices");
        assert!(editor_scene_mesh_validate(&root).is_err());
        assert!(editor_scene_mesh_validate(&DynamicStruct::new()).is_err());
        assert!(editor_scene_mesh_validate(&mesh.matrix).is_err());
    }

    #[test]
    fn path_callback_checks_all_mixed_fields_and_coupled_geometry() {
        let validator = object_validator();
        let mesh = triangle();
        let controller = decoded(&mesh);
        let decoded_root = controller.object_root(&SelectionItem::entity(1)).unwrap();
        for source in [&mesh as &dyn Reflect, decoded_root] {
            let mut candidate = source.reflect_clone();
            candidate
                .downcast_mut::<DynamicStruct>()
                .unwrap()
                .insert("positions", vec![[0.0_f32; 3]; 2]);
            // Even an index-only edit rechecks positions and bounds against them.
            assert!(validator
                .validate(&*candidate, &paths(&["indices[0]"]))
                .is_err());
            for names in [
                ["matrix[3][0]", "positions[0][0]"],
                ["positions[0][0]", "matrix[3][0]"],
                ["visible", "indices[0]"],
                ["indices[0]", "visible"],
            ] {
                assert!(validator.validate(&*candidate, &paths(&names)).is_err());
            }
            let mut candidate = source.reflect_clone();
            let ReflectMut::Struct(fields) = candidate.reflect_mut() else {
                panic!("mesh struct")
            };
            fields
                .field_mut("matrix")
                .unwrap()
                .apply(&[[0.0_f32; 4]; 4])
                .unwrap();
            assert!(validator
                .validate(
                    &*candidate,
                    &paths(&["positions[0][0]", "visible", "matrix"])
                )
                .is_err());
            assert!(validator
                .validate(&*candidate, &paths(&["matrix", "visible", "indices[0]"]))
                .is_err());
        }
        // Dynamic visibility cannot hide behind a valid earlier matrix path.
        let mut root = dynamic_root(mesh.matrix);
        root.insert("visible", 1_u32);
        assert!(validator
            .validate(&root, &paths(&["matrix[0][0]", "visible"]))
            .is_err());
        assert!(validator
            .validate(&root, &paths(&["visible", "matrix[0][0]"]))
            .is_err());
    }

    #[test]
    fn controller_registration_and_final_candidate_reject_invalid_mesh_atomically() {
        let item = SelectionItem::entity(1);
        let mut invalid = triangle();
        invalid.indices[2] = 3;
        for root in [
            Box::new(invalid.clone()) as Box<dyn Reflect>,
            invalid.reflect_clone(),
        ] {
            let mut controller = EditorController::new();
            assert!(controller
                .register_object(item.clone(), root, editor_scene_mesh_registry())
                .is_err());
            assert!(controller.object_root(&item).is_err());
        }
        let mesh = triangle();
        let controller = decoded(&mesh);
        for bad in [
            ("matrix[0][0]", ActionValue::float(0.0)),
            ("indices[2]", ActionValue::unsigned(3)),
        ] {
            let edits = [
                ("visible", ActionValue::bool(false)),
                ("positions[0][0]", ActionValue::float(0.5)),
                bad,
            ]
            .into_iter()
            .map(|(name, value)| ControllerFieldEdit {
                item: item.clone(),
                path: FieldPath::parse(name).unwrap(),
                value,
            })
            .collect();
            assert!(controller.prepare_edits(edits).is_err());
            let root = controller.object_root(&item).unwrap();
            assert_eq!(editor_scene_mesh_matrix(root).unwrap(), mesh.matrix);
        }
    }

    #[test]
    fn trs_scales_local_basis_rotates_about_own_pivot_and_translates_in_world() {
        let mut mesh = triangle();
        mesh.matrix[1][1] = 2.0;
        mesh.matrix[2][2] = 3.0;
        mesh.matrix[3] = [2.0, -3.0, 4.0, 1.0];
        let controller = decoded(&mesh);
        let dynamic = controller.object_root(&SelectionItem::entity(1)).unwrap();
        let delta = TransformDelta {
            scale_milliunits: [2000, 3000, 4000],
            rotation_millidegrees: [0, 0, 90_000],
            translation_milliunits: [1000, -2000, 500],
        };
        let expected = [
            [0.0, 2.0, 0.0, 0.0],
            [-6.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 12.0, 0.0],
            [3.0, -5.0, 4.5, 1.0],
        ];
        for root in [&mesh as &dyn Reflect, dynamic] {
            assert_close(edit_matrix(root, delta), expected);
            assert_close(edit_matrix(root, delta), expected); // cumulative, not incremental
            assert_eq!(editor_scene_mesh_matrix(root).unwrap(), mesh.matrix);
            assert_eq!(edit_matrix(root, identity_delta()), mesh.matrix);
        }
    }

    #[test]
    fn rotations_apply_in_world_xyz_order() {
        let mesh = triangle();
        let delta = TransformDelta {
            rotation_millidegrees: [30_000, 45_000, 60_000],
            ..identity_delta()
        };
        let (sx, cx) = 30.0_f64.to_radians().sin_cos();
        let (sy, cy) = 45.0_f64.to_radians().sin_cos();
        let (sz, cz) = 60.0_f64.to_radians().sin_cos();
        // Independent closed form of Rz * Ry * Rx, stored column-major.
        let expected = [
            [cz * cy, sz * cy, -sy, 0.0],
            [cz * sy * sx - sz * cx, sz * sy * sx + cz * cx, cy * sx, 0.0],
            [cz * sy * cx + sz * sx, sz * sy * cx - cz * sx, cy * cx, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        .map(|column| column.map(|v| v as f32));
        let controller = decoded(&mesh);
        for root in [
            &mesh as &dyn Reflect,
            controller.object_root(&SelectionItem::entity(1)).unwrap(),
        ] {
            assert_close(edit_matrix(root, delta), expected);
        }
    }

    #[test]
    fn dynamic_edits_commit_atomically_and_undo_redo_without_geometry_changes() {
        let mesh = triangle();
        let mut controller = decoded(&mesh);
        let item = SelectionItem::entity(1);
        let delta = TransformDelta {
            translation_milliunits: [1000, 2000, -3000],
            ..identity_delta()
        };
        let root = controller.object_root(&item).unwrap();
        let edits = EditorSceneMeshTransform.edits(&item, root, delta).unwrap();
        let prepared = controller.prepare_edits(edits).unwrap();
        assert_eq!(
            editor_scene_mesh_matrix(controller.object_root(&item).unwrap()).unwrap(),
            mesh.matrix
        );
        controller.commit_edits(prepared).unwrap();
        let changed = controller.object_root(&item).unwrap();
        assert_eq!(
            editor_scene_mesh_matrix(changed).unwrap()[3],
            [1.0, 2.0, -3.0, 1.0]
        );
        controller.execute(EditorAction::Undo).unwrap();
        assert_eq!(
            editor_scene_mesh_matrix(controller.object_root(&item).unwrap()).unwrap(),
            mesh.matrix
        );
        controller.execute(EditorAction::Redo).unwrap();
        assert_eq!(
            editor_scene_mesh_matrix(controller.object_root(&item).unwrap()).unwrap()[3],
            [1.0, 2.0, -3.0, 1.0]
        );
    }

    #[test]
    fn geometry_and_visibility_remain_editable() {
        let mut controller = decoded(&triangle());
        let item = SelectionItem::entity(1);
        let edits = [
            ("positions[0][0]", ActionValue::float(0.25)),
            ("indices[0]", ActionValue::unsigned(2)),
            ("visible", ActionValue::bool(false)),
        ]
        .into_iter()
        .map(|(path, value)| ControllerFieldEdit {
            item: item.clone(),
            path: FieldPath::parse(path).unwrap(),
            value,
        })
        .collect();
        let prepared = controller.prepare_edits(edits).unwrap();
        controller.commit_edits(prepared).unwrap();
        let mut restored = triangle();
        restored
            .apply(controller.object_root(&item).unwrap())
            .unwrap();
        restored.validate().unwrap();
        assert_eq!(restored.positions[0][0], 0.25);
        assert_eq!(restored.indices[0], 2);
        assert!(!restored.visible);
    }

    fn assert_invalid_matrix(matrix: SceneIdMatrix) {
        let mut mesh = triangle();
        mesh.matrix = matrix;
        assert!(mesh.validate().is_err());
        let dynamic = mesh.reflect_clone();
        for root in [&mesh as &dyn Reflect, &*dynamic] {
            assert!(editor_scene_mesh_matrix(root).is_err());
            assert!(EditorSceneMeshTransform
                .edits(&SelectionItem::entity(1), root, identity_delta())
                .is_err());
        }
    }

    #[test]
    fn every_nonfinite_matrix_entry_is_rejected() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for column in 0..4 {
                for row in 0..4 {
                    let mut matrix = SCENE_ID_IDENTITY_MATRIX;
                    matrix[column][row] = bad;
                    assert_invalid_matrix(matrix);
                }
            }
        }
    }

    #[test]
    fn nonaffine_and_singular_matrices_are_rejected() {
        for column in 0..4 {
            let mut matrix = SCENE_ID_IDENTITY_MATRIX;
            matrix[column][3] = if column == 3 { 0.0 } else { 0.001 };
            assert_invalid_matrix(matrix);
        }
        let mut matrix = SCENE_ID_IDENTITY_MATRIX;
        matrix[1] = matrix[0];
        assert_invalid_matrix(matrix);
        matrix[1] = [2.0, 0.0, 0.0, 0.0];
        assert_invalid_matrix(matrix);
        assert_invalid_matrix([[0.0; 4]; 4]);
    }

    #[test]
    fn exact_singularity_is_not_hidden_by_determinant_cancellation() {
        // c = a + b exactly in f32, but a naive f64 determinant is 4096.
        let mut matrix = [
            [-3_392_361.0, 2_889_348.0, 495_304.0, 0.0],
            [-3_210_380.0, -932_380.0, 888_780.0, 0.0],
            [-6_602_741.0, 1_956_968.0, 1_384_084.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_invalid_matrix(matrix);
        matrix.swap(0, 1);
        assert_invalid_matrix(matrix);
        matrix[2][2] += 1.0; // genuinely nonsingular, no tolerance-based rejection
        let mut mesh = triangle();
        mesh.matrix = matrix;
        mesh.validate().unwrap();
        let controller = decoded(&mesh);
        assert_eq!(
            editor_scene_mesh_matrix(controller.object_root(&SelectionItem::entity(1)).unwrap())
                .unwrap(),
            matrix
        );
    }

    #[test]
    fn scale_policy_and_invalid_output_never_return_partial_edits() {
        let item = SelectionItem::entity(1);
        let mesh = triangle();
        for axis in 0..3 {
            for bad in [0, -1, i64::MIN] {
                let mut delta = identity_delta();
                delta.scale_milliunits[axis] = bad;
                assert!(EditorSceneMeshTransform.edits(&item, &mesh, delta).is_err());
            }
        }
        assert!(EditorSceneMeshTransform
            .edits(&item, &mesh, TransformDelta::default())
            .is_err());
        let mut huge = triangle();
        huge.matrix[0][0] = f32::MAX;
        huge.validate().unwrap();
        let delta = TransformDelta {
            scale_milliunits: [2000, 1000, 1000],
            ..identity_delta()
        };
        assert!(EditorSceneMeshTransform.edits(&item, &huge, delta).is_err());
        assert_eq!(huge.matrix[0][0], f32::MAX);
        let mut tiny = triangle();
        tiny.matrix[0][0] = f32::from_bits(1);
        tiny.validate().unwrap();
        let delta = TransformDelta {
            scale_milliunits: [1, 1000, 1000],
            ..identity_delta()
        };
        assert!(EditorSceneMeshTransform.edits(&item, &tiny, delta).is_err());
    }

    #[test]
    fn finite_shear_mirror_and_tiny_or_huge_nonsingular_bases_are_allowed() {
        for scale in [f32::from_bits(1), f32::MIN_POSITIVE, f32::MAX, -1.0] {
            let mut mesh = triangle();
            for axis in 0..3 {
                mesh.matrix[axis][axis] = scale;
            }
            mesh.validate().unwrap();
            assert_eq!(editor_scene_mesh_matrix(&mesh).unwrap(), mesh.matrix);
        }
        let mut mesh = triangle();
        mesh.matrix[1][0] = 0.5;
        mesh.validate().unwrap();
        assert_eq!(edit_matrix(&mesh, identity_delta()), mesh.matrix);
    }

    #[test]
    fn geometry_checks_nonempty_triangles_finite_vertices_and_index_bounds_even_hidden() {
        let mut mesh = triangle();
        mesh.visible = false;
        mesh.validate().unwrap();
        mesh.indices = vec![2, 2, 2]; // degenerate is not a profile policy error
        mesh.validate().unwrap();
        for indices in [
            vec![],
            vec![0],
            vec![0, 1],
            vec![0, 1, 2, 0],
            vec![0, 1, 3],
            vec![0, 1, u32::MAX],
        ] {
            mesh.indices = indices;
            assert!(mesh.validate().is_err());
        }
        mesh.indices = vec![0, 1, 2];
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for axis in 0..3 {
                mesh.positions[0] = [0.0; 3];
                mesh.positions[0][axis] = bad;
                assert!(mesh.validate().is_err());
            }
        }
        mesh.positions.clear();
        assert!(mesh.validate().is_err());
    }

    #[test]
    fn geometry_budget_boundaries_use_counts_without_large_allocations() {
        // Both vertex records and complete triangles cost 12 bytes. 64 MiB is
        // 4 mod 12, so the closest reachable payload below the cap is cap - 4.
        let triangles = (MAX_GEOMETRY_BYTES as usize - 12) / 12;
        assert!(validate_geometry_lengths(1, triangles * 3).is_ok());
        assert!(validate_geometry_lengths(1, (triangles + 1) * 3).is_err());
        assert!(validate_geometry_lengths(2, triangles * 3).is_err());
        for (positions, indices) in [(0, 3), (1, 0), (1, 2), (usize::MAX, 3), (1, usize::MAX)] {
            assert!(validate_geometry_lengths(positions, indices).is_err());
        }
        if usize::BITS > 32 {
            assert!(validate_geometry_lengths(u32::MAX as usize + 1, 3).is_err());
            assert!(validate_geometry_lengths(1, u32::MAX as usize + 3).is_err());
        }
        // Spare Vec capacity is not uploaded geometry and does not change admission.
        let mut mesh = triangle();
        mesh.positions.reserve(100);
        mesh.indices.reserve(100);
        mesh.validate().unwrap();
    }

    fn dynamic_root(matrix: impl Reflect) -> DynamicStruct {
        let mut root = DynamicStruct::new();
        root.set_represented_type_name(type_name::<EditorSceneMesh>());
        root.insert("matrix", matrix);
        root
    }

    #[test]
    fn malformed_dynamic_matrix_shapes_types_and_wrong_roots_are_rejected() {
        let mut roots = vec![
            dynamic_root(0.0_f32),
            dynamic_root([[0.0_f32; 4]; 3]),
            dynamic_root([[0.0_f32; 3]; 4]),
            dynamic_root([[0.0_f64; 4]; 4]),
            dynamic_root(vec![[0.0_f32; 4]; 4]),
        ];
        let mut absent = dynamic_root(SCENE_ID_IDENTITY_MATRIX);
        absent.remove("matrix");
        roots.push(absent);
        roots.push(DynamicStruct::new());
        for root in roots {
            assert!(editor_scene_mesh_matrix(&root).is_err());
        }
        assert!(editor_scene_mesh_matrix(&SCENE_ID_IDENTITY_MATRIX).is_err());
        let mut ragged = DynamicArray::new();
        ragged.set_represented_type_name(type_name::<[[f32; 4]; 4]>());
        for length in [4, 4, 3, 4] {
            let mut column = DynamicArray::new();
            column.set_represented_type_name(type_name::<[f32; 4]>());
            for _ in 0..length {
                column.push_value(0.0_f32);
            }
            ragged.push_value(column);
        }
        assert!(editor_scene_mesh_matrix(&dynamic_root(ragged)).is_err());
        let mut wrong_type = DynamicArray::new();
        wrong_type.set_represented_type_name(type_name::<[[f64; 4]; 4]>());
        for column in SCENE_ID_IDENTITY_MATRIX {
            wrong_type.push_value(column);
        }
        assert!(editor_scene_mesh_matrix(&dynamic_root(wrong_type)).is_err());
        let mut wrong_leaf = DynamicArray::new();
        wrong_leaf.set_represented_type_name(type_name::<[f32; 4]>());
        for _ in 0..4 {
            wrong_leaf.push_value(0.0_f64);
        }
        let mut matrix = DynamicArray::new();
        matrix.set_represented_type_name(type_name::<[[f32; 4]; 4]>());
        matrix.push_value(wrong_leaf);
        for _ in 0..3 {
            matrix.push_value([0.0_f32; 4]);
        }
        assert!(editor_scene_mesh_matrix(&dynamic_root(matrix)).is_err());
    }

    // A reflection sentinel makes accidental geometry reads/clones fail, without
    // allocating a huge mesh or relying on allocator/timing instrumentation.
    struct Untouched;
    impl Reflect for Untouched {
        fn type_name(&self) -> &'static str {
            panic!("unrelated field type read")
        }
        fn type_info(&self) -> &'static TypeInfo {
            panic!("unrelated field type info read")
        }
        fn as_any(&self) -> &dyn Any {
            panic!("unrelated field downcast")
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn into_any(self: Box<Self>) -> Box<dyn Any> {
            self
        }
        fn as_reflect(&self) -> &dyn Reflect {
            self
        }
        fn as_reflect_mut(&mut self) -> &mut dyn Reflect {
            self
        }
        fn reflect_ref(&self) -> ReflectRef<'_> {
            panic!("unrelated field read")
        }
        fn reflect_mut(&mut self) -> ReflectMut<'_> {
            panic!("unrelated field mutation")
        }
        fn reflect_clone(&self) -> Box<dyn Reflect> {
            panic!("unrelated field clone")
        }
    }

    #[test]
    fn registered_matrix_callback_never_accesses_geometry_or_visibility() {
        let validator = object_validator();
        let mut root = dynamic_root(SCENE_ID_IDENTITY_MATRIX);
        root.insert("positions", Untouched);
        root.insert("indices", Untouched);
        root.insert("visible", Untouched);
        let matrix_paths: Vec<FieldPath> = (0..4)
            .flat_map(|column| {
                (0..4)
                    .map(move |row| FieldPath::parse(&format!("matrix[{column}][{row}]")).unwrap())
            })
            .collect();
        validator.validate(&root, &matrix_paths).unwrap();
        validator.validate(&root, &paths(&["matrix"])).unwrap();
        for bad in [f32::NAN, f32::INFINITY, 0.0] {
            let mut matrix = SCENE_ID_IDENTITY_MATRIX;
            matrix[0][0] = bad;
            root.insert("matrix", matrix);
            assert!(validator.validate(&root, &matrix_paths).is_err());
        }
        let mut matrix = SCENE_ID_IDENTITY_MATRIX;
        matrix[0][3] = 1.0;
        root.insert("matrix", matrix);
        assert!(validator.validate(&root, &matrix_paths).is_err());
        let mut typed = triangle();
        typed.positions.clear();
        typed.indices.clear();
        assert!(validator.validate(&typed, &[]).is_err());
        validator.validate(&typed, &matrix_paths).unwrap();
    }

    #[test]
    fn registered_visibility_callback_never_accesses_geometry_or_matrix() {
        let validator = object_validator();
        let mut root = dynamic_root(Untouched);
        root.insert("positions", Untouched);
        root.insert("indices", Untouched);
        root.insert("visible", false);
        validator.validate(&root, &paths(&["visible"])).unwrap();
        root.insert("visible", 0_u32);
        assert!(validator.validate(&root, &paths(&["visible"])).is_err());
    }

    // Synthetic borrowed lists enforce no clones and no element reads
    // before admission of lengths. Large budgets need no large allocation.
    struct BorrowedGeometry<T> {
        len: usize,
        readable: bool,
        value: T,
    }

    impl<T: Reflect + Typed> List for BorrowedGeometry<T> {
        fn get(&self, index: usize) -> Option<&dyn Reflect> {
            assert!(self.readable, "over-budget geometry element read");
            (index < self.len).then_some(&self.value as &dyn Reflect)
        }
        fn get_mut(&mut self, _: usize) -> Option<&mut dyn Reflect> {
            panic!("borrowed geometry mutation")
        }
        fn len(&self) -> usize {
            self.len
        }
        fn push(&mut self, _: Box<dyn Reflect>) -> Result<(), Box<dyn Reflect>> {
            panic!("borrowed geometry mutation")
        }
        fn as_list(&self) -> &dyn List {
            self
        }
    }

    impl<T: Reflect + Typed> Reflect for BorrowedGeometry<T> {
        fn type_name(&self) -> &'static str {
            type_name::<Vec<T>>()
        }
        fn type_info(&self) -> &'static TypeInfo {
            <Vec<T> as Typed>::type_info()
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn into_any(self: Box<Self>) -> Box<dyn Any> {
            self
        }
        fn as_reflect(&self) -> &dyn Reflect {
            self
        }
        fn as_reflect_mut(&mut self) -> &mut dyn Reflect {
            self
        }
        fn reflect_ref(&self) -> ReflectRef<'_> {
            ReflectRef::List(self)
        }
        fn reflect_mut(&mut self) -> ReflectMut<'_> {
            panic!("borrowed geometry mutation")
        }
        fn reflect_clone(&self) -> Box<dyn Reflect> {
            panic!("borrowed geometry clone")
        }
    }

    #[test]
    fn full_borrowed_validation_checks_budget_before_elements_without_cloning() {
        let validator = object_validator();
        let mut root = dynamic_root(SCENE_ID_IDENTITY_MATRIX);
        root.insert("indices", vec![0_u32, 1, 2]);
        root.insert("visible", true);
        root.insert(
            "positions",
            BorrowedGeometry {
                len: 3,
                readable: true,
                value: [0.0_f32; 3],
            },
        );
        editor_scene_mesh_validate(&root).unwrap();
        validator.validate(&root, &[]).unwrap();
        validator
            .validate(&root, &paths(&["positions[0][0]", "indices[2]"]))
            .unwrap();
        for len in [0, MAX_GEOMETRY_BYTES as usize / 12 + 1, usize::MAX] {
            root.insert(
                "positions",
                BorrowedGeometry {
                    len,
                    readable: false,
                    value: [0.0_f32; 3],
                },
            );
            assert!(editor_scene_mesh_validate(&root).is_err());
            for changed in [
                vec![],
                vec![FieldPath::root()],
                paths(&["indices[0]"]),
                paths(&["unknown"]),
            ] {
                assert!(validator.validate(&root, &changed).is_err());
            }
        }
    }

    fn length_only_root(positions: usize, indices: usize) -> DynamicStruct {
        let mut root = dynamic_root(Untouched);
        root.insert("visible", Untouched);
        root.insert(
            "positions",
            BorrowedGeometry {
                len: positions,
                readable: false,
                value: [f32::NAN; 3],
            },
        );
        root.insert(
            "indices",
            BorrowedGeometry {
                len: indices,
                readable: false,
                value: u32::MAX,
            },
        );
        root
    }

    #[test]
    fn geometry_bytes_reads_only_lengths_without_clones_or_large_allocations() {
        let records = (MAX_GEOMETRY_BYTES as usize - 12) / 12;
        for (positions, indices, bytes) in [
            (3, 3, 48),
            (records, 3, MAX_GEOMETRY_BYTES - 4),
            (1, records * 3, MAX_GEOMETRY_BYTES - 4),
        ] {
            assert_eq!(
                editor_scene_mesh_geometry_bytes(&length_only_root(positions, indices)).unwrap(),
                bytes
            );
        }
        let mut mesh = triangle();
        mesh.positions.reserve(100);
        mesh.indices.reserve(100);
        mesh.positions[0][0] = f32::NAN;
        mesh.indices[0] = u32::MAX;
        mesh.matrix = [[f32::NAN; 4]; 4];
        assert_eq!(editor_scene_mesh_geometry_bytes(&mesh).unwrap(), 48);
        assert!(mesh.validate().is_err()); // bytes are not full mesh admission
    }

    #[test]
    fn geometry_bytes_rejects_invalid_counts_overflow_and_single_mesh_budget() {
        let validator = object_validator();
        let records = (MAX_GEOMETRY_BYTES as usize - 12) / 12;
        for (positions, indices) in [
            (0, 3),
            (1, 0),
            (1, 2),
            (records + 1, 3),
            (1, (records + 1) * 3),
            (records, records * 3),
            (usize::MAX, 3),
            (1, usize::MAX),
            (usize::MAX, usize::MAX),
        ] {
            let root = length_only_root(positions, indices);
            assert!(
                editor_scene_mesh_geometry_bytes(&root).is_err(),
                "positions={positions}, indices={indices}"
            );
            // Full admission and geometry-edit admission must reject the same
            // lengths before the unreadable elements or unrelated sentinels.
            assert!(editor_scene_mesh_validate(&root).is_err());
            assert!(validator.validate(&root, &paths(&["indices[0]"])).is_err());
        }
        if usize::BITS > 32 {
            for (positions, indices) in [(u32::MAX as usize + 1, 3), (1, u32::MAX as usize + 3)] {
                assert!(
                    editor_scene_mesh_geometry_bytes(&length_only_root(positions, indices))
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn document_geometry_bytes_empty_and_bounded_totals_need_no_geometry_reads() {
        use crate::scene_session::document_geometry_bytes;

        assert_eq!(document_geometry_bytes(core::iter::empty()).unwrap(), 0);
        let items = [SelectionItem::entity(1), SelectionItem::entity(2)];
        let small = [length_only_root(3, 3), length_only_root(1, 6)];
        assert_eq!(
            document_geometry_bytes(
                items
                    .iter()
                    .zip(small.iter().map(|root| root as &dyn Reflect))
            )
            .unwrap(),
            84
        );

        // Two large synthetic lists, with no allocation for their elements.
        let positions = MAX_GEOMETRY_BYTES as usize / 24 - 1;
        let large = [
            length_only_root(positions, 3),
            length_only_root(positions, 3),
        ];
        assert_eq!(
            document_geometry_bytes(
                items
                    .iter()
                    .zip(large.iter().map(|root| root as &dyn Reflect))
            )
            .unwrap(),
            MAX_GEOMETRY_BYTES - 16
        );
    }

    #[test]
    fn document_geometry_bytes_rejects_aggregate_before_any_geometry_read_or_clone() {
        use crate::scene_session::document_geometry_bytes;

        let positions = (MAX_GEOMETRY_BYTES as usize - 12) / 12;
        let items = [SelectionItem::entity(1), SelectionItem::entity(2)];
        let roots = [
            length_only_root(positions, 3),
            length_only_root(positions, 3),
        ];
        for root in &roots {
            assert_eq!(
                editor_scene_mesh_geometry_bytes(root).unwrap(),
                MAX_GEOMETRY_BYTES - 4
            );
        }
        assert!(document_geometry_bytes(
            items
                .iter()
                .zip(roots.iter().map(|root| root as &dyn Reflect))
        )
        .is_err());
    }

    #[test]
    fn document_geometry_bytes_rejects_object_count_before_iterating_roots() {
        use crate::{desktop_scene::MAX_OBJECTS, scene_session::document_geometry_bytes};

        let item = SelectionItem::entity(1);
        let root = Untouched;
        let roots = (0..MAX_OBJECTS + 1)
            .inspect(|_| panic!("over-count document iterated a root"))
            .map(|_| (&item, &root as &dyn Reflect));
        assert!(document_geometry_bytes(roots).is_err());
    }

    #[test]
    fn document_geometry_bytes_rejects_nonentity_before_accessing_root() {
        use crate::scene_session::document_geometry_bytes;

        let root = Untouched;
        for item in [
            SelectionItem::asset(1),
            SelectionItem::sub_object(1, "mesh", 0),
        ] {
            assert!(
                document_geometry_bytes(core::iter::once((&item, &root as &dyn Reflect))).is_err()
            );
        }
    }

    #[test]
    fn geometry_bytes_rejects_wrong_root_and_container_schema() {
        assert!(editor_scene_mesh_geometry_bytes(&SCENE_ID_IDENTITY_MATRIX).is_err());
        let mut root = length_only_root(3, 3);
        root.set_represented_type_name("foreign::Mesh");
        assert!(editor_scene_mesh_geometry_bytes(&root).is_err());
        assert!(editor_scene_mesh_geometry_bytes(&dynamic_root(Untouched)).is_err());
        let mut root = length_only_root(3, 3);
        root.insert("extra", Untouched);
        assert!(editor_scene_mesh_geometry_bytes(&root).is_err());
        for field in ["positions", "indices"] {
            let mut root = length_only_root(3, 3);
            root.insert(field, 0_u32);
            assert!(editor_scene_mesh_geometry_bytes(&root).is_err());
        }
        let mut root = length_only_root(3, 3);
        root.insert("positions", vec![0_u32; 3]);
        assert!(editor_scene_mesh_geometry_bytes(&root).is_err());
        let mut root = length_only_root(3, 3);
        root.insert("indices", vec![[0.0_f32; 3]; 3]);
        assert!(editor_scene_mesh_geometry_bytes(&root).is_err());
        let mut root = length_only_root(3, 3);
        let mut wrong_kind = DynamicStruct::new();
        wrong_kind.set_represented_type_name(type_name::<Vec<u32>>());
        root.insert("indices", wrong_kind);
        assert!(editor_scene_mesh_geometry_bytes(&root).is_err());
    }

    #[test]
    fn matrix_access_and_transform_never_read_or_clone_geometry() {
        let mut root = dynamic_root(SCENE_ID_IDENTITY_MATRIX);
        root.insert("positions", Untouched);
        root.insert("indices", Untouched);
        root.insert("visible", Untouched);
        assert_eq!(
            editor_scene_mesh_matrix(&root).unwrap(),
            SCENE_ID_IDENTITY_MATRIX
        );
        assert_eq!(
            edit_matrix(&root, identity_delta()),
            SCENE_ID_IDENTITY_MATRIX
        );
        let mut typed = triangle();
        typed.positions.clear(); // matrix-only access must not run full validate
        assert!(typed.validate().is_err());
        assert_eq!(edit_matrix(&typed, identity_delta()), typed.matrix);
    }
}
