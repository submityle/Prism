//! The mesh asset: attributes in, a vertex layout and packed bytes out.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_asset::Asset;
use prism_render_driver::{PrimitiveTopology, VertexAttribute, VertexBufferLayout, VertexStepMode};

use crate::attribute::{MeshVertexAttribute, VertexAttributeValues};
use crate::bounds::MeshBounds;
use crate::error::MeshError;
use crate::indices::Indices;
use crate::limits::MeshLimits;
use crate::streams::StreamLayoutProfile;
use crate::submesh::{validate_submeshes, validate_submeshes_with_limits, SubMesh};
use crate::topology::PrimitiveTopologyExt;

/// The semantic name of the attribute mesh bounds are computed from.
///
/// A convention rather than a registry: bounds come from *positions*, and a
/// renderer that authors its positions under another name sets the bounds itself.
pub const POSITION_ATTRIBUTE: &str = "Vertex_Position";

/// One mesh: named vertex attributes, optional indices, and a topology.
///
/// The asset stores what the author wrote — typed values, grouped by semantic —
/// and derives what the GPU needs on demand: an interleaved
/// [`VertexBufferLayout`], the packed [`vertex buffer data`](Mesh::vertex_buffer_data),
/// and local-space [`bounds`](Mesh::bounds). Nothing here is GPU-specific, so a
/// mesh can be built, validated, and tested without a device.
///
/// Attributes are keyed by **shader location**, because that is what the pipeline
/// binds: the layout is emitted in location order, and two meshes with the same
/// attributes produce byte-identical layouts (so their pipelines are shared).
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    attributes: BTreeMap<u32, (MeshVertexAttribute, VertexAttributeValues)>,
    indices: Option<Indices>,
    primitive_topology: PrimitiveTopology,
    submeshes: Vec<SubMesh>,
}

impl Mesh {
    /// An empty mesh drawn as `primitive_topology`.
    #[must_use]
    pub fn new(primitive_topology: PrimitiveTopology) -> Self {
        Self {
            attributes: BTreeMap::new(),
            indices: None,
            primitive_topology,
            submeshes: Vec::new(),
        }
    }

    /// Inserts `values` under `attribute`.
    ///
    /// Shader locations and persisted semantic names are unique identities; a
    /// conflicting insertion is rejected rather than silently discarding authored
    /// data.
    ///
    /// Fails when the value's storage does not match the attribute's declared
    /// format, or when its element count disagrees with the mesh's other
    /// attributes. Either would produce a vertex buffer the GPU would read as
    /// garbage, so the mesh refuses to hold it.
    pub fn insert_attribute(
        &mut self,
        attribute: MeshVertexAttribute,
        values: VertexAttributeValues,
    ) -> Result<(), MeshError> {
        if self.attributes.contains_key(&attribute.shader_location) {
            return Err(MeshError::DuplicateShaderLocation {
                shader_location: attribute.shader_location,
            });
        }
        if self
            .attributes
            .iter()
            .any(|(_, (existing, _))| existing.name == attribute.name)
        {
            return Err(MeshError::DuplicateAttributeName {
                attribute: attribute.name,
            });
        }
        if !values.matches(attribute.format) {
            return Err(MeshError::FormatMismatch {
                attribute: attribute.name,
            });
        }
        if let Some((_, existing)) = self.attributes.get(&attribute.shader_location)
            && existing.len() != values.len()
        {
            return Err(MeshError::VertexCountMismatch {
                attribute: attribute.name,
                expected: existing.len(),
                found: values.len(),
            });
        }
        if let Some((_, other)) = self.attributes.values().next()
            && other.len() != values.len()
        {
            return Err(MeshError::VertexCountMismatch {
                attribute: attribute.name,
                expected: other.len(),
                found: values.len(),
            });
        }
        self.attributes
            .insert(attribute.shader_location, (attribute, values));
        Ok(())
    }

    /// Builder form of [`insert_attribute`](Self::insert_attribute).
    pub fn with_attribute(
        mut self,
        attribute: MeshVertexAttribute,
        values: VertexAttributeValues,
    ) -> Result<Self, MeshError> {
        self.insert_attribute(attribute, values)?;
        Ok(self)
    }

    /// The index list, if the mesh is indexed.
    pub fn with_indices(&mut self, indices: Indices) {
        self.indices = Some(indices);
    }

    /// Builder form of [`with_indices`](Self::with_indices).
    #[must_use]
    pub fn indexed(mut self, indices: Indices) -> Self {
        self.with_indices(indices);
        self
    }

    /// The index list, if any.
    #[must_use]
    pub fn indices(&self) -> Option<&Indices> {
        self.indices.as_ref()
    }

    /// The material-partitioned index ranges, in authored order.
    #[must_use]
    pub fn submeshes(&self) -> &[SubMesh] {
        &self.submeshes
    }

    /// Replaces the mesh's material-partitioned index ranges.
    pub fn with_submeshes(&mut self, submeshes: Vec<SubMesh>) {
        self.submeshes = submeshes;
    }

    /// Builder form of [`with_submeshes`](Self::with_submeshes).
    #[must_use]
    pub fn submeshed(mut self, submeshes: Vec<SubMesh>) -> Self {
        self.with_submeshes(submeshes);
        self
    }

    /// The topology this mesh is drawn as.
    #[must_use]
    pub const fn primitive_topology(&self) -> PrimitiveTopology {
        self.primitive_topology
    }

    /// The attribute at `shader_location`, if the mesh has one.
    #[must_use]
    pub fn attribute(
        &self,
        shader_location: u32,
    ) -> Option<&(MeshVertexAttribute, VertexAttributeValues)> {
        self.attributes.get(&shader_location)
    }

    /// The mesh's attributes, in shader-location order.
    pub fn attributes(
        &self,
    ) -> impl Iterator<Item = &(MeshVertexAttribute, VertexAttributeValues)> {
        self.attributes.values()
    }

    /// The number of attributes, i.e. the size of the vertex stage.
    #[must_use]
    pub fn attribute_count(&self) -> usize {
        self.attributes.len()
    }

    /// The number of vertices, taken from any attribute (they must agree).
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.attributes
            .values()
            .next()
            .map_or(0, |(_, values)| values.len())
    }

    /// The interleaved vertex layout the pipeline binds, in shader-location
    /// order, or `None` for a mesh with no attributes.
    ///
    /// One buffer, level 0: each vertex's attributes are packed back to back with
    /// the offsets the layout declares. A mesh that needed split streams would
    /// need a layout per stream, which no renderer in the tree asks for yet.
    #[must_use]
    pub fn vertex_layout(&self) -> Option<VertexBufferLayout> {
        if self.attributes.is_empty() {
            return None;
        }
        let mut offset = 0_u64;
        let mut attributes = Vec::with_capacity(self.attributes.len());
        for (attribute, values) in self.attributes.values() {
            debug_assert!(values.matches(attribute.format));
            attributes.push(VertexAttribute {
                format: attribute.format,
                offset,
                shader_location: attribute.shader_location,
            });
            offset += attribute.format.size();
        }
        Some(VertexBufferLayout {
            array_stride: offset,
            step_mode: VertexStepMode::Vertex,
            attributes,
        })
    }

    /// The vertex-buffer bytes: every vertex's attributes interleaved at the
    /// offsets [`vertex_layout`](Self::vertex_layout) declares.
    ///
    /// Returns an empty buffer for a mesh with no attributes, matching
    /// `vertex_layout`'s `None`.
    #[must_use]
    pub fn vertex_buffer_data(&self) -> Vec<u8> {
        let Some(layout) = self.vertex_layout() else {
            return Vec::new();
        };
        let vertex_count = self.vertex_count();
        let mut data = Vec::with_capacity(vertex_count * layout.array_stride as usize);
        for vertex in 0..vertex_count {
            for (attribute, values) in self.attributes.values() {
                debug_assert!(values.matches(attribute.format));
                values.write_vertex(vertex, &mut data);
            }
        }
        data
    }

    /// The local-space bounds of the mesh's positions, if it has a
    /// [`POSITION_ATTRIBUTE`] attribute of `Float32x3`.
    #[must_use]
    pub fn bounds(&self) -> Option<MeshBounds> {
        self.attributes
            .values()
            .find(|(attribute, _)| attribute.name == POSITION_ATTRIBUTE)
            .and_then(|(_, values)| values.as_positions())
            .and_then(MeshBounds::from_positions)
    }

    /// Checks everything a device would otherwise discover the hard way.
    ///
    /// The renderer runs this before allocating: an index past the vertex range
    /// would read another mesh's vertices on the GPU, and an attribute-less mesh
    /// has no vertex stage to bind at all.
    pub fn validate(&self) -> Result<(), MeshError> {
        if self.attributes.is_empty() {
            return Err(MeshError::NoAttributes);
        }
        let vertex_count = checked_mesh_count(self.vertex_count(), "vertex")?;
        let mut semantic_names = alloc::collections::BTreeSet::new();
        for (attribute, values) in self.attributes.values() {
            if !semantic_names.insert(attribute.name) {
                return Err(MeshError::DuplicateAttributeName {
                    attribute: attribute.name,
                });
            }
            if !values.matches(attribute.format) {
                return Err(MeshError::FormatMismatch {
                    attribute: attribute.name,
                });
            }
            if let Some((vertex, component)) = values.first_non_finite() {
                return Err(MeshError::NonFiniteAttributeValue {
                    attribute: attribute.name,
                    vertex,
                    component,
                });
            }
            if values.len() != self.vertex_count() {
                return Err(MeshError::VertexCountMismatch {
                    attribute: attribute.name,
                    expected: self.vertex_count(),
                    found: values.len(),
                });
            }
        }
        if let Some(max_index) = self.indices.as_ref().and_then(Indices::max_index) {
            if max_index >= vertex_count {
                return Err(MeshError::IndexOutOfRange {
                    max_index,
                    vertex_count,
                });
            }
        }
        let element_count = self
            .indices
            .as_ref()
            .map_or(self.vertex_count(), Indices::len);
        let element_count = checked_mesh_count(element_count, "element")?;
        self.primitive_topology
            .validate_element_count(element_count)?;
        if !self.submeshes.is_empty() {
            let Some(indices) = self.indices.as_ref() else {
                return Err(MeshError::SubMeshesRequireIndices);
            };
            validate_submeshes(
                &self.submeshes,
                checked_mesh_count(indices.len(), "index")?,
                self.primitive_topology,
            )?;
        }
        Ok(())
    }

    /// Validates the mesh against an explicit device/content profile.
    pub fn validate_with_limits(&self, limits: &MeshLimits) -> Result<(), MeshError> {
        self.validate()?;
        let stream_count = self.stream_layouts(&StreamLayoutProfile::default()).len();
        limits.validate_counts(self.vertex_count(), self.attribute_count(), stream_count)?;
        if !self.submeshes.is_empty() {
            let indices = self
                .indices
                .as_ref()
                .ok_or(MeshError::SubMeshesRequireIndices)?;
            validate_submeshes_with_limits(
                &self.submeshes,
                checked_mesh_count(indices.len(), "index")?,
                self.primitive_topology,
                limits,
            )?;
        }
        Ok(())
    }
}

fn checked_mesh_count(count: usize, kind: &'static str) -> Result<u32, MeshError> {
    u32::try_from(count).map_err(|_| MeshError::CountExceedsU32 { kind, found: count })
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn checked_mesh_count_accepts_u32_max_and_rejects_larger_counts() {
        assert_eq!(
            checked_mesh_count(u32::MAX as usize, "vertex"),
            Ok(u32::MAX)
        );
        if usize::BITS > u32::BITS {
            let too_many = (u32::MAX as usize).checked_add(1).unwrap();
            assert_eq!(
                checked_mesh_count(too_many, "vertex"),
                Err(MeshError::CountExceedsU32 {
                    kind: "vertex",
                    found: too_many,
                })
            );
        }
    }
}

impl Asset for Mesh {
    const TYPE_NAME: &'static str = "prism_mesh::Mesh";
}
