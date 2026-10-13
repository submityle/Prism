//! Deterministic content identity for authored mesh data.

use alloc::vec::Vec;

use prism_render_driver::{PrimitiveTopology, VertexFormat};

use crate::{Indices, Mesh};

/// Schema version for the canonical mesh hash encoding.
pub const MESH_HASH_SCHEMA: u32 = 1;

/// The stable digest of a mesh's authored data and build-relevant declarations.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MeshContentHash(pub [u8; 32]);

impl MeshContentHash {
    /// Returns the raw BLAKE3 digest bytes.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    /// Returns the lowercase hexadecimal digest.
    #[must_use]
    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut text = String::with_capacity(64);
        for byte in self.0 {
            text.push(HEX[(byte >> 4) as usize] as char);
            text.push(HEX[(byte & 0x0f) as usize] as char);
        }
        text
    }
}

/// Computes a stable hash from mesh data only.
///
/// The encoding is versioned, domain-separated, length-prefixed, and traverses
/// attributes in shader-location order. Source paths, timestamps, allocations,
/// and other host state are intentionally absent from the input.
#[must_use]
pub fn content_hash(mesh: &Mesh) -> MeshContentHash {
    let bytes = canonical_bytes(mesh);
    MeshContentHash(*blake3::hash(&bytes).as_bytes())
}

/// Returns the canonical bytes used as the hash input.
#[must_use]
pub fn canonical_bytes(mesh: &Mesh) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"PRISM-MESH\0");
    out.extend_from_slice(&MESH_HASH_SCHEMA.to_le_bytes());
    out.push(topology_tag(mesh.primitive_topology()));
    put_u64(&mut out, mesh.attribute_count());
    for (attribute, values) in mesh.attributes() {
        put_bytes(&mut out, attribute.name.as_bytes());
        put_u32(&mut out, attribute.shader_location);
        out.push(format_tag(attribute.format));
        put_u64(&mut out, values.len());
        for index in 0..values.len() {
            values.write_vertex(index, &mut out);
        }
    }

    match mesh.indices() {
        None => out.push(0),
        Some(indices) => {
            out.push(1);
            out.push(match indices {
                Indices::Uint16(_) => 16,
                Indices::Uint32(_) => 32,
            });
            put_u64(&mut out, indices.len());
            out.extend_from_slice(&indices.to_bytes());
        }
    }

    put_u64(&mut out, mesh.submeshes().len());
    for submesh in mesh.submeshes() {
        out.extend_from_slice(&submesh.material_slot.to_le_bytes());
        out.extend_from_slice(&submesh.index_range.start.to_le_bytes());
        out.extend_from_slice(&submesh.index_range.end.to_le_bytes());
        match &submesh.cluster_range {
            None => out.push(0),
            Some(range) => {
                out.push(1);
                out.extend_from_slice(&range.start.to_le_bytes());
                out.extend_from_slice(&range.end.to_le_bytes());
            }
        }
    }
    out
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_u64(out, bytes.len());
    out.extend_from_slice(bytes);
}

fn topology_tag(topology: PrimitiveTopology) -> u8 {
    match topology {
        PrimitiveTopology::PointList => 0,
        PrimitiveTopology::LineList => 1,
        PrimitiveTopology::LineStrip => 2,
        PrimitiveTopology::TriangleList => 3,
        PrimitiveTopology::TriangleStrip => 4,
    }
}

fn format_tag(format: VertexFormat) -> u8 {
    match format {
        VertexFormat::Float32 => 0,
        VertexFormat::Float32x2 => 1,
        VertexFormat::Float32x3 => 2,
        VertexFormat::Float32x4 => 3,
        VertexFormat::Float16x2 => 4,
        VertexFormat::Float16x4 => 5,
        VertexFormat::Uint32 => 6,
        VertexFormat::Uint32x2 => 7,
        VertexFormat::Uint32x4 => 8,
        VertexFormat::Sint32 => 9,
        VertexFormat::Unorm8x4 => 10,
        VertexFormat::Snorm8x4 => 11,
        _ => u8::MAX,
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use prism_render_driver::{PrimitiveTopology, VertexFormat};

    use super::*;
    use crate::{MeshVertexAttribute, VertexAttributeValues};

    fn mesh() -> Mesh {
        Mesh::new(PrimitiveTopology::TriangleList)
            .with_attribute(
                MeshVertexAttribute::new("Vertex_Position", VertexFormat::Float32x3, 0),
                VertexAttributeValues::Float32x3(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            )
            .expect("position fits")
    }

    #[test]
    fn hash_is_stable_and_changes_for_build_relevant_data() {
        let first = content_hash(&mesh());
        assert_eq!(first, content_hash(&mesh()));

        let mut changed = mesh();
        changed.with_indices(Indices::Uint16(vec![0, 1, 0]));
        assert_ne!(first, content_hash(&changed));

        let mut reordered = Mesh::new(PrimitiveTopology::TriangleList);
        reordered
            .insert_attribute(
                MeshVertexAttribute::new("Other", VertexFormat::Float32, 1),
                VertexAttributeValues::Float32(vec![0.0, 1.0]),
            )
            .expect("other fits");
        reordered
            .insert_attribute(
                MeshVertexAttribute::new("Vertex_Position", VertexFormat::Float32x3, 0),
                VertexAttributeValues::Float32x3(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            )
            .expect("position fits");
        assert_ne!(first, content_hash(&reordered));
        assert_eq!(first.to_hex().len(), 64);
    }
}
