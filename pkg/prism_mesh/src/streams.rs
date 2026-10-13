//! Deterministic vertex-stream partitioning for standard and deforming meshes.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_render_driver::{VertexFormat, VertexStepMode};

use crate::{Mesh, MeshVertexAttribute, POSITION_ATTRIBUTE};

/// The semantic stream a vertex attribute belongs to.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum VertexStreamKind {
    /// Position and attributes needed by depth/visibility passes.
    Position,
    /// Normal, tangent, UV, and color data used by shading.
    Shading,
    /// Joint indices and weights used by skinning.
    Skinning,
    /// Morph or other per-vertex deformation data.
    Deformation,
    /// An explicitly numbered project-defined stream.
    Custom(u8),
}

/// One attribute within a split vertex stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamAttribute {
    /// The source mesh shader location.
    pub shader_location: u32,
    /// The RHI fetch format.
    pub format: VertexFormat,
    /// Byte offset within the stream element.
    pub offset: u64,
}

/// One tightly packed per-vertex stream layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VertexStreamLayout {
    /// The stream's semantic role.
    pub kind: VertexStreamKind,
    /// Bytes between consecutive vertices.
    pub array_stride: u64,
    /// Attributes in shader-location order.
    pub attributes: Vec<StreamAttribute>,
}

impl VertexStreamLayout {
    /// Builds a deterministic tightly packed stream from source attributes.
    #[must_use]
    pub fn new(kind: VertexStreamKind, mut attributes: Vec<MeshVertexAttribute>) -> Self {
        attributes.sort_by_key(|attribute| attribute.shader_location);
        let mut offset = 0;
        let mut layout = Vec::with_capacity(attributes.len());
        for attribute in attributes {
            layout.push(StreamAttribute {
                shader_location: attribute.shader_location,
                format: attribute.format,
                offset,
            });
            offset += attribute.format.size();
        }
        Self {
            kind,
            array_stride: offset,
            attributes: layout,
        }
    }

    /// The layout is valid for tightly packed vertex data.
    #[must_use]
    pub fn is_within_stride(&self) -> bool {
        self.attributes
            .iter()
            .all(|attribute| attribute.offset + attribute.format.size() <= self.array_stride)
    }
}

/// A named rule set for assigning mesh attributes to streams.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamLayoutProfile {
    custom: BTreeMap<&'static str, VertexStreamKind>,
}

impl Default for StreamLayoutProfile {
    fn default() -> Self {
        Self {
            custom: BTreeMap::new(),
        }
    }
}

impl StreamLayoutProfile {
    /// Overrides the default assignment for one semantic name.
    pub fn assign(&mut self, name: &'static str, kind: VertexStreamKind) {
        self.custom.insert(name, kind);
    }

    /// Resolves the stream for one attribute semantic.
    #[must_use]
    pub fn stream_for(&self, name: &str) -> VertexStreamKind {
        if let Some(kind) = self.custom.get(name) {
            return *kind;
        }
        if name == POSITION_ATTRIBUTE {
            return VertexStreamKind::Position;
        }
        if name.contains("Joint") || name.contains("Weight") {
            return VertexStreamKind::Skinning;
        }
        if name.contains("Morph") || name.contains("Deform") {
            return VertexStreamKind::Deformation;
        }
        VertexStreamKind::Shading
    }
}

impl Mesh {
    /// Builds the split layouts without changing the mesh's authored ordering.
    #[must_use]
    pub fn stream_layouts(&self, profile: &StreamLayoutProfile) -> Vec<VertexStreamLayout> {
        let mut grouped: BTreeMap<VertexStreamKind, Vec<MeshVertexAttribute>> = BTreeMap::new();
        for (attribute, _) in self.attributes() {
            grouped
                .entry(profile.stream_for(attribute.name))
                .or_default()
                .push(*attribute);
        }
        grouped
            .into_iter()
            .map(|(kind, attributes)| VertexStreamLayout::new(kind, attributes))
            .collect()
    }

    /// Packs one split stream in the layout's declared attribute order.
    ///
    /// `None` means the layout names an attribute that is not present in this
    /// mesh. The method still accepts an invalid mesh so callers can report the
    /// precise preparation error before allocating GPU buffers.
    #[must_use]
    pub fn stream_buffer_data(&self, layout: &VertexStreamLayout) -> Option<Vec<u8>> {
        let mut values = Vec::with_capacity(layout.attributes.len());
        for attribute in &layout.attributes {
            let (_, values_for_attribute) = self.attribute(attribute.shader_location)?;
            if values_for_attribute.format() != attribute.format {
                return None;
            }
            values.push(values_for_attribute);
        }
        let mut data = Vec::with_capacity(self.vertex_count() * layout.array_stride as usize);
        for vertex in 0..self.vertex_count() {
            for value in &values {
                value.write_vertex(vertex, &mut data);
            }
        }
        Some(data)
    }

    /// Returns the stream layout and bytes together for upload preparation.
    #[must_use]
    pub fn stream_buffers(
        &self,
        profile: &StreamLayoutProfile,
    ) -> Vec<(VertexStreamLayout, Vec<u8>)> {
        self.stream_layouts(profile)
            .into_iter()
            .filter_map(|layout| {
                let bytes = self.stream_buffer_data(&layout)?;
                Some((layout, bytes))
            })
            .collect()
    }
}

#[allow(dead_code)]
const _: Option<VertexStepMode> = Some(VertexStepMode::Vertex);

#[cfg(test)]
mod tests {
    use alloc::vec;
    use prism_render_driver::{PrimitiveTopology, VertexFormat};

    use super::*;
    use crate::VertexAttributeValues;

    #[test]
    fn default_profile_separates_position_and_skinning() {
        let mesh = Mesh::new(PrimitiveTopology::PointList)
            .with_attribute(
                MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
                VertexAttributeValues::Float32x3(vec![[1.0, 2.0, 3.0]]),
            )
            .expect("position fits")
            .with_attribute(
                MeshVertexAttribute::new("Vertex_JointIndex", VertexFormat::Uint32x4, 4),
                VertexAttributeValues::Uint32x4(vec![[0, 1, 0, 0]]),
            )
            .expect("joints fit");

        let layouts = mesh.stream_layouts(&StreamLayoutProfile::default());
        assert_eq!(layouts.len(), 2);
        assert_eq!(layouts[0].kind, VertexStreamKind::Position);
        assert_eq!(layouts[1].kind, VertexStreamKind::Skinning);
        assert_eq!(layouts[0].array_stride, 12);
        assert_eq!(layouts[1].array_stride, 16);
        assert_eq!(
            mesh.stream_buffers(&StreamLayoutProfile::default()).len(),
            2
        );
    }
}
