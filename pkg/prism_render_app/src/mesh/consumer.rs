//! Concrete GPU bindings for the engine-neutral mesh consumer contract.

use prism_render_architecture::geometry::{
    MeshConsumer, MeshConsumerBinding, MeshConsumerBindingError,
    MeshConsumerSelection, MeshDerivedArtifact, MeshRepresentation,
};
use prism_render_architecture::gpu_scene::GeometryHandle;
use prism_render_architecture::virtual_geometry::GeometryPageKey;
use prism_render_driver::{BufferId, IndexFormat, PrimitiveTopology, VertexBufferLayout};

use super::deformation::DeformationOutput;
use super::RenderMesh;

/// Concrete source data used by one mesh consumer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuMeshSource {
    /// Static or fallback RenderMesh buffers.
    Static {
        /// Vertex buffers in pipeline binding order.
        vertices: Vec<BufferId>,
        /// Optional index buffer and its format.
        indices: Option<(BufferId, IndexFormat)>,
        /// Vertex layouts matching `vertices`.
        layout: Vec<VertexBufferLayout>,
        /// Primitive topology of the source.
        topology: PrimitiveTopology,
    },
    /// One output of the shared deformation cache.
    Deformed {
        /// Current or previous position/normal output buffer.
        vertex_buffer: BufferId,
        /// Fixed deformation output layout.
        layout: VertexBufferLayout,
        /// Number of valid vertices in the output.
        vertex_count: u32,
    },
    /// A cluster addressed in resident virtual-geometry page storage.
    VirtualCluster {
        /// Resident page identity.
        page: GeometryPageKey,
        /// Cluster descriptor index within the asset.
        cluster: u32,
    },
    /// A versioned cooked product resolved by a non-render consumer.
    DerivedArtifact {
        /// Identity used to resolve the cooked GPU/CPU product.
        artifact: MeshDerivedArtifact,
    },
}

/// A render-app binding carrying semantic identity and concrete GPU sources.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderMeshConsumerBinding {
    /// Engine-neutral identity and consumer selection rules.
    pub contract: MeshConsumerBinding,
    /// Current-frame source.
    pub current: GpuMeshSource,
    /// Previous-frame source.
    pub previous: GpuMeshSource,
    /// Explicit fallback source.
    pub fallback: GpuMeshSource,
    /// Concrete source for collision and other cooked consumers.
    pub derived: Option<GpuMeshSource>,
}

impl RenderMeshConsumerBinding {
    /// Selects a source and returns its semantic selection.
    pub fn select(
        &self,
        consumer: MeshConsumer,
        preferred_resident: bool,
    ) -> Result<(&GpuMeshSource, MeshConsumerSelection), MeshConsumerBindingError> {
        let selection = self
            .contract
            .select_with_fallback(consumer, preferred_resident)?;
        let source = match selection {
            MeshConsumerSelection::Motion { history_valid, .. } if history_valid => &self.previous,
            MeshConsumerSelection::Motion { .. } => &self.current,
            MeshConsumerSelection::Render { used_fallback: true, .. } => &self.fallback,
            MeshConsumerSelection::Render { .. } => &self.current,
            MeshConsumerSelection::Derived { artifact, .. } => match self.derived.as_ref() {
                Some(source @ GpuMeshSource::DerivedArtifact { artifact: bound })
                    if *bound == artifact => source,
                _ => return Err(MeshConsumerBindingError::MissingDerivedArtifact),
            },
        };
        Ok((source, selection))
    }

    /// Attaches a published derived artifact without changing render identity.
    #[must_use]
    pub fn with_derived_artifact(mut self, artifact: MeshDerivedArtifact) -> Self {
        self.contract.derived = Some(artifact);
        self.derived = Some(GpuMeshSource::DerivedArtifact { artifact });
        self
    }
}

impl RenderMesh {
    /// Builds a static source binding. The fallback identity is explicit; the
    /// caller can replace the fallback source with a separately prepared LOD.
    #[must_use]
    pub fn consumer_binding(
        &self,
        geometry: GeometryHandle,
        revision: u32,
        lod: u32,
        fallback_lod: u32,
    ) -> RenderMeshConsumerBinding {
        let source = GpuMeshSource::Static {
            vertices: self.buffers.vertices.clone(),
            indices: self.buffers.indices,
            layout: self.layout.clone(),
            topology: self.topology,
        };
        RenderMeshConsumerBinding {
            contract: MeshConsumerBinding {
                geometry,
                revision,
                current: MeshRepresentation::StaticLod { level: lod },
                previous: MeshRepresentation::StaticLod { level: lod },
                history_valid: false,
                fallback: MeshRepresentation::StaticLod { level: fallback_lod },
                derived: None,
            },
            current: source.clone(),
            previous: source.clone(),
            fallback: source,
            derived: None,
        }
    }
}

/// Builds a binding from one deformation-cache output and a static fallback.
#[must_use]
pub fn deformed_consumer_binding(
    geometry: GeometryHandle,
    revision: u32,
    deformation_handle: prism_render_architecture::deformation::DeformationHandle,
    output: DeformationOutput,
    fallback: &RenderMeshConsumerBinding,
) -> RenderMeshConsumerBinding {
    let layout = DeformationOutput::vertex_layout(0, 1);
    let current = GpuMeshSource::Deformed {
        vertex_buffer: output.current,
        layout: layout.clone(),
        vertex_count: output.vertex_count,
    };
    let previous = GpuMeshSource::Deformed {
        vertex_buffer: output.previous,
        layout,
        vertex_count: output.vertex_count,
    };
    RenderMeshConsumerBinding {
        contract: MeshConsumerBinding {
            geometry,
            revision,
            current: MeshRepresentation::Deformed {
                handle: deformation_handle,
                frame: output.frame,
            },
            previous: MeshRepresentation::Deformed {
                handle: deformation_handle,
                frame: output.frame.saturating_sub(1),
            },
            history_valid: output.history_valid,
            fallback: fallback.contract.fallback,
            derived: fallback.contract.derived,
        },
        current,
        previous,
        fallback: fallback.fallback.clone(),
        derived: fallback.derived.clone(),
    }
}

/// Builds a page-backed binding before a cluster is handed to the raster backend.
#[must_use]
pub fn virtual_cluster_consumer_binding(
    geometry: GeometryHandle,
    revision: u32,
    page: GeometryPageKey,
    cluster: u32,
    fallback: &RenderMeshConsumerBinding,
) -> RenderMeshConsumerBinding {
    let source = GpuMeshSource::VirtualCluster { page, cluster };
    RenderMeshConsumerBinding {
        contract: MeshConsumerBinding {
            geometry,
            revision,
            current: MeshRepresentation::VirtualCluster { page, cluster },
            previous: MeshRepresentation::VirtualCluster { page, cluster },
            history_valid: false,
            fallback: fallback.contract.fallback,
            derived: fallback.contract.derived,
        },
        current: source.clone(),
        previous: source,
        fallback: fallback.fallback.clone(),
        derived: fallback.derived.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_architecture::deformation::DeformationHandle;

    fn static_binding() -> RenderMeshConsumerBinding {
        let source = GpuMeshSource::Static {
            vertices: vec![BufferId::from_parts(1, 1)],
            indices: None,
            layout: vec![],
            topology: PrimitiveTopology::TriangleList,
        };
        RenderMeshConsumerBinding {
            contract: MeshConsumerBinding {
                geometry: GenerationalHandle::new(1, 1),
                revision: 1,
                current: MeshRepresentation::StaticLod { level: 0 },
                previous: MeshRepresentation::StaticLod { level: 0 },
                history_valid: false,
                fallback: MeshRepresentation::StaticLod { level: 1 },
                derived: None,
            },
            current: source.clone(),
            previous: source.clone(),
            fallback: source,
            derived: None,
        }
    }

    #[test]
    fn motion_selection_binds_current_on_history_reset() {
        let binding = static_binding();
        let (source, selection) = binding.select(MeshConsumer::MotionVector, true).unwrap();
        assert_eq!(source, &binding.current);
        assert!(matches!(selection, MeshConsumerSelection::Motion { history_valid: false, .. }));
    }

    #[test]
    fn page_miss_binds_declared_fallback_source() {
        let binding = static_binding();
        let virtual_binding = virtual_cluster_consumer_binding(
            GenerationalHandle::new(1, 1),
            2,
            GeometryPageKey::new(1, 3),
            4,
            &binding,
        );
        let (source, selection) = virtual_binding.select(MeshConsumer::Shadow, false).unwrap();
        assert_eq!(source, &virtual_binding.fallback);
        assert!(matches!(selection, MeshConsumerSelection::Render { used_fallback: true, .. }));
    }

    #[test]
    fn deformed_binding_keeps_current_and_previous_buffers_distinct() {
        let fallback = static_binding();
        let current = BufferId::from_parts(3, 1);
        let previous = BufferId::from_parts(4, 1);
        let binding = deformed_consumer_binding(
            GenerationalHandle::new(1, 1),
            3,
            DeformationHandle(8),
            DeformationOutput {
                current,
                previous,
                history_valid: true,
                frame: 10,
                vertex_count: 4,
            },
            &fallback,
        );
        assert_eq!(binding.select(MeshConsumer::MainView, true).unwrap().0, &binding.current);
        assert_eq!(binding.select(MeshConsumer::MotionVector, true).unwrap().0, &binding.previous);
        assert_ne!(binding.current, binding.previous);
    }

    #[test]
    fn derived_consumer_never_binds_render_source() {
        let binding = static_binding().with_derived_artifact(MeshDerivedArtifact {
            key: [7; 32],
            kind: 3,
            schema: 1,
        });
        let (source, selection) = binding.select(MeshConsumer::Collision, true).unwrap();
        assert!(matches!(selection, MeshConsumerSelection::Derived { .. }));
        assert!(matches!(source, GpuMeshSource::DerivedArtifact { .. }));
        assert_ne!(source, &binding.current);
    }
}
