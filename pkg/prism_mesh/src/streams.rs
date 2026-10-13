//! Deterministic vertex-stream partitioning for standard and deforming meshes.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_render_driver::{VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};

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
        self.attributes.iter().all(|attribute| {
            attribute
                .offset
                .checked_add(attribute.format.size())
                .is_some_and(|end| end <= self.array_stride)
        })
    }

    /// Converts the device-independent stream description into the RHI layout
    /// consumed by a render pipeline.
    #[must_use]
    pub fn rhi_layout(&self) -> VertexBufferLayout {
        VertexBufferLayout {
            array_stride: self.array_stride,
            step_mode: VertexStepMode::Vertex,
            attributes: self
                .attributes
                .iter()
                .map(|attribute| VertexAttribute {
                    format: attribute.format,
                    offset: attribute.offset,
                    shader_location: attribute.shader_location,
                })
                .collect(),
        }
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
        if !layout.is_within_stride() {
            return None;
        }
        let stride = usize::try_from(layout.array_stride).ok()?;
        let total_len = self.vertex_count().checked_mul(stride)?;
        let mut data = vec![0_u8; total_len];
        for vertex in 0..self.vertex_count() {
            let vertex_base = vertex.checked_mul(stride)?;
            for (attribute, value) in layout.attributes.iter().zip(values.iter()) {
                let mut encoded = Vec::with_capacity(attribute.format.size() as usize);
                value.write_vertex(vertex, &mut encoded);
                let offset = usize::try_from(attribute.offset).ok()?;
                let start = vertex_base.checked_add(offset)?;
                let end = start.checked_add(encoded.len())?;
                let destination = data.get_mut(start..end)?;
                destination.copy_from_slice(&encoded);
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
    fn stream_buffer_data_honors_declared_padding_and_offsets() {
        let mesh = Mesh::new(PrimitiveTopology::PointList)
            .with_attribute(
                MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
                VertexAttributeValues::Float32x3(vec![[1.0, 2.0, 3.0]]),
            )
            .expect("position fits");
        let layout = VertexStreamLayout {
            kind: VertexStreamKind::Position,
            array_stride: 32,
            attributes: vec![StreamAttribute {
                shader_location: 0,
                format: VertexFormat::Float32x3,
                offset: 16,
            }],
        };
        let data = mesh.stream_buffer_data(&layout).expect("layout fits");
        assert_eq!(data.len(), 32);
        assert!(data[..16].iter().all(|byte| *byte == 0));
        assert_eq!(
            data[16..28],
            [1.0_f32, 2.0, 3.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        assert!(data[28..].iter().all(|byte| *byte == 0));
    }

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
