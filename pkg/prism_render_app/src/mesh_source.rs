//! Where a geometry's pipeline inputs come from: the prepared mesh.
//!
//! A [`GeometryRecord`](prism_render_architecture::geometry::GeometryRecord)
//! names a *class* (`vertex_buffer_class`), and the queue needs that class's
//! layout and buffers to specialize a pipeline and bind geometry. Those are
//! properties of the mesh, not of the record: the record says "this is a mesh of
//! class 5", the prepared [`RenderMesh`] says what class 5 *is*.
//!
//! This module is the join between the two. A scene binds a geometry to a mesh
//! asset; each frame the registry syncs the bound classes from the prepared
//! meshes, so a mesh that was created, re-uploaded, or re-laid-out this frame is
//! what the queue draws — without the queue knowing anything about assets.

use alloc::vec;
use alloc::vec::Vec;

use prism_asset::AssetId;
use prism_mesh::Mesh;
use prism_render_architecture::gpu_scene::GeometryHandle;

use crate::mesh::RenderMesh;
use crate::render_asset::RenderAssets;
use crate::scene::{MeshClass, SceneRegistry};

impl SceneRegistry {
    /// Binds `geometry` to the mesh asset it draws.
    ///
    /// Binding is the scene's statement of intent ("this instance's geometry is
    /// that mesh"); the class table is *derived* from it by
    /// [`sync_mesh_classes`](Self::sync_mesh_classes), so there is one authority
    /// for a mesh's layout rather than a hand-registered copy that can drift.
    pub fn bind_geometry_mesh(&mut self, geometry: GeometryHandle, mesh: AssetId<Mesh>) {
        self.mesh_bindings.insert(geometry, mesh);
    }

    /// The mesh asset bound to `geometry`, if any.
    #[must_use]
    pub fn geometry_mesh(&self, geometry: GeometryHandle) -> Option<AssetId<Mesh>> {
        self.mesh_bindings.get(&geometry).copied()
    }

    /// The number of bound geometries.
    #[must_use]
    pub fn bound_geometry_count(&self) -> usize {
        self.mesh_bindings.len()
    }

    /// Brings every bound geometry's mesh class in step with its prepared mesh.
    ///
    /// Returns how many classes were written. A geometry whose record or whose
    /// prepared mesh is not resident yet is skipped — the queue skips its work
    /// items for the same reason, so both agree on what is drawable, rather than
    /// the queue drawing with a class the mesh never declared.
    pub fn sync_mesh_classes(&mut self, meshes: &RenderAssets<RenderMesh>) -> usize {
        let bindings: Vec<(GeometryHandle, AssetId<Mesh>)> = self
            .mesh_bindings
            .iter()
            .map(|(geometry, mesh)| (*geometry, *mesh))
            .collect();
        let mut written = 0;
        for (geometry, mesh) in bindings {
            let Some(record) = self.geometries.get(&geometry) else {
                continue;
            };
            let Some(prepared) = meshes.get(mesh) else {
                continue;
            };
            let class = record.vertex_buffer_class;
            let updated = MeshClass::new(
                vec![prepared.layout.clone()],
                prepared.topology,
                prepared.buffers,
            );
            if self.mesh_classes.get(&class) != Some(&updated) {
                self.mesh_classes.insert(class, updated);
                written += 1;
            }
        }
        written
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use prism_asset::AssetIndex;
    use prism_mesh::{Indices, Mesh, MeshVertexAttribute, VertexAttributeValues};
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_architecture::geometry::{
        GeometryLodRecord, GeometryPrimitiveKind, GeometryRecord,
    };
    use prism_render_driver::{
        DeviceDescriptor, Instance, PrimitiveTopology, RequestAdapterOptions, VertexFormat,
    };
    use prism_render_driver_null::NullInstance;

    use crate::mesh::RenderMesh;
    use crate::render_asset::{RenderAsset, RenderAssets};

    use super::SceneRegistry;

    fn null_device() -> (prism_render_driver_null::NullDevice, impl prism_render_driver::RenderQueue)
    {
        let options = RequestAdapterOptions::default();
        let adapter = NullInstance::new()
            .request_adapter(&options)
            .expect("the null backend always has an adapter");
        NullInstance::new()
            .create_device(&adapter, &DeviceDescriptor::default())
            .expect("the null device always comes up")
    }

    fn quad_mesh() -> Mesh {
        Mesh::new(PrimitiveTopology::TriangleList)
            .with_attribute(
                MeshVertexAttribute::new("Vertex_Position", VertexFormat::Float32x3, 0),
                VertexAttributeValues::Float32x3(vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]),
            )
            .expect("positions fit")
            .indexed(Indices::Uint16(vec![0, 1, 2, 0, 2, 3]))
    }

    /// The class a geometry draws through is *derived* from its prepared mesh:
    /// layout, topology and buffers all come from the mesh, so a mesh that was
    /// re-uploaded this frame is what the queue specializes on.
    #[test]
    fn the_mesh_class_is_synced_from_the_prepared_mesh() {
        let (device, queue) = null_device();
        let geometry = GenerationalHandle::new(1, 1);
        let mesh_id = prism_asset::AssetId::new(AssetIndex::from_parts(0, 1));

        let mut registry = SceneRegistry::new();
        registry.insert_geometry(GeometryRecord {
            handle: geometry,
            revision: 1,
            vertex_buffer_class: 5,
            index_buffer_class: 0,
            primitive_start: 0,
            primitive_count: 1,
            lods: vec![GeometryLodRecord {
                level: 0,
                primitive_kind: GeometryPrimitiveKind::Indexed,
                element_count: 6,
                first_element: 0,
                base_vertex: 0,
                vertex_count: 4,
                screen_error: 0.0,
                resident: true,
                fallback: false,
            }],
        });
        registry.bind_geometry_mesh(geometry, mesh_id);
        assert_eq!(registry.bound_geometry_count(), 1);
        assert_eq!(registry.geometry_mesh(geometry), Some(mesh_id));

        // Not prepared yet: nothing to sync, and the class stays absent — which
        // is exactly why the queue would skip the item.
        let mut meshes = RenderAssets::<RenderMesh>::new();
        assert_eq!(registry.sync_mesh_classes(&meshes), 0);
        assert!(registry.mesh_class(5).is_none());

        let prepared = RenderMesh::prepare(
            &quad_mesh(),
            mesh_id,
            &device,
            &queue,
            None,
        )
        .expect("the quad prepares");
        let expected_layout = prepared.layout.clone();
        let expected_buffers = prepared.buffers;
        meshes.insert(mesh_id, prepared);

        assert_eq!(registry.sync_mesh_classes(&meshes), 1, "the class is written");
        let class = registry.mesh_class(5).expect("class 5 is registered");
        assert_eq!(
            class.layout,
            vec![expected_layout],
            "the layout is the mesh's"
        );
        assert_eq!(class.topology, PrimitiveTopology::TriangleList);
        assert_eq!(class.buffers, expected_buffers);

        // A second sync with nothing changed writes nothing.
        assert_eq!(registry.sync_mesh_classes(&meshes), 0);
    }

    /// A geometry whose record or whose mesh is not resident is skipped, so the
    /// class table never gains an entry nothing backs — the same filter the queue
    /// applies, which is why the two agree on what is drawable.
    #[test]
    fn syncing_skips_unbacked_geometries() {
        let (device, queue) = null_device();
        let geometry = GenerationalHandle::new(1, 1);
        let mesh_id = prism_asset::AssetId::new(AssetIndex::from_parts(0, 1));
        let mut registry = SceneRegistry::new();
        registry.bind_geometry_mesh(geometry, mesh_id);

        let mut meshes = RenderAssets::<RenderMesh>::new();
        meshes.insert(
            mesh_id,
            RenderMesh::prepare(&quad_mesh(), mesh_id, &device, &queue, None).expect("prepares"),
        );

        // The mesh is prepared but the geometry has no record: nothing to sync.
        assert_eq!(registry.sync_mesh_classes(&meshes), 0);
        assert!(registry.mesh_class(5).is_none());

        // With the record present but the mesh gone, the class is still absent.
        registry.insert_geometry(GeometryRecord {
            handle: geometry,
            revision: 1,
            vertex_buffer_class: 5,
            index_buffer_class: 0,
            primitive_start: 0,
            primitive_count: 1,
            lods: vec![GeometryLodRecord {
                level: 0,
                primitive_kind: GeometryPrimitiveKind::Indexed,
                element_count: 6,
                first_element: 0,
                base_vertex: 0,
                vertex_count: 4,
                screen_error: 0.0,
                resident: true,
                fallback: false,
            }],
        });
        let empty = RenderAssets::<RenderMesh>::new();
        assert_eq!(registry.sync_mesh_classes(&empty), 0);
        assert!(registry.mesh_class(5).is_none());
    }
}

