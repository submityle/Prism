//! Integration tests for adapting canonical Mesh IR into physics geometry.

use glam::Vec3;
use prism_mesh::{Mesh, MeshVertexAttribute, VertexAttributeValues, POSITION_ATTRIBUTE};
use prism_physics_core::collider::{TriMeshAssetError, TriMeshData};
use prism_render_driver::{PrimitiveTopology, VertexFormat};

fn triangle_mesh(topology: PrimitiveTopology) -> Mesh {
    let mut mesh = Mesh::new(topology);
    mesh.insert_attribute(
        MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
        VertexAttributeValues::Float32x3(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    .expect("valid position attribute");
    mesh
}

#[test]
fn adapts_validated_mesh_ir_to_physics_triangle_geometry() {
    let mesh = triangle_mesh(PrimitiveTopology::TriangleList);
    let collision = TriMeshData::from_mesh_asset(&mesh).expect("valid collision source");

    assert_eq!(collision.mesh().triangle_count(), 1);
    assert_eq!(
        collision.mesh().triangle(0),
        Some([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ])
    );
}

#[test]
fn rejects_unsupported_topology_before_cooking() {
    let mesh = triangle_mesh(PrimitiveTopology::LineList);
    assert!(matches!(
        TriMeshData::from_mesh_asset(&mesh),
        Err(TriMeshAssetError::UnsupportedTopology(
            PrimitiveTopology::LineList
        ))
    ));
}

#[test]
fn rejects_mesh_without_canonical_positions() {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
    mesh.insert_attribute(
        MeshVertexAttribute::new("Other", VertexFormat::Float32x3, 0),
        VertexAttributeValues::Float32x3(vec![[0.0, 0.0, 0.0]; 3]),
    )
    .expect("valid unrelated attribute");
    assert!(matches!(
        TriMeshData::from_mesh_asset(&mesh),
        Err(TriMeshAssetError::MissingPositions)
    ));
}
