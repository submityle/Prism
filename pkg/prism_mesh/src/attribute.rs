//! Named vertex attributes and the typed values they hold.

use alloc::vec::Vec;

use prism_render_driver::VertexFormat;

/// A vertex attribute's authoring identity.
///
/// The name is the semantic (`"Vertex_Position"`, `"Vertex_Normal"`), the format
/// is what the GPU reads, and the shader location is where the pipeline expects
/// it. Two meshes that carry the same attribute set produce the same vertex
/// layout, which is what lets their pipelines be shared.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MeshVertexAttribute {
    /// The attribute's semantic name (persisted identity, like a component name).
    pub name: &'static str,
    /// The format the GPU fetches this attribute as.
    pub format: VertexFormat,
    /// The shader location the pipeline binds it to.
    pub shader_location: u32,
}

impl MeshVertexAttribute {
    /// An attribute named `name`, fetched as `format`, at `shader_location`.
    #[must_use]
    pub const fn new(name: &'static str, format: VertexFormat, shader_location: u32) -> Self {
        Self {
            name,
            format,
            shader_location,
        }
    }
}

/// The values of one vertex attribute, one element per vertex.
///
/// The variants are the formats a mesh can be authored in. [`Mesh`](crate::Mesh)
/// interleaves them into the GPU layout, so the variant and the attribute's
/// [`VertexFormat`] must agree — [`VertexAttributeValues::matches`] is the check,
/// and a mismatch is reported rather than silently reinterpreted.
#[derive(Clone, Debug, PartialEq)]
pub enum VertexAttributeValues {
    /// One `f32` per vertex.
    Float32(Vec<f32>),
    /// Two `f32` per vertex.
    Float32x2(Vec<[f32; 2]>),
    /// Three `f32` per vertex.
    Float32x3(Vec<[f32; 3]>),
    /// Four `f32` per vertex.
    Float32x4(Vec<[f32; 4]>),
    /// Two `u32` per vertex.
    Uint32x2(Vec<[u32; 2]>),
    /// Four `u32` per vertex.
    Uint32x4(Vec<[u32; 4]>),
    /// Four `u8` normalised to `[0, 1]` per vertex.
    Unorm8x4(Vec<[u8; 4]>),
}

impl VertexAttributeValues {
    /// The number of vertices these values describe.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Float32(values) => values.len(),
            Self::Float32x2(values) => values.len(),
            Self::Float32x3(values) => values.len(),
            Self::Float32x4(values) => values.len(),
            Self::Uint32x2(values) => values.len(),
            Self::Uint32x4(values) => values.len(),
            Self::Unorm8x4(values) => values.len(),
        }
    }

    /// Whether there are no vertices.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The [`VertexFormat`] this value can be uploaded as.
    #[must_use]
    pub const fn format(&self) -> VertexFormat {
        match self {
            Self::Float32(_) => VertexFormat::Float32,
            Self::Float32x2(_) => VertexFormat::Float32x2,
            Self::Float32x3(_) => VertexFormat::Float32x3,
            Self::Float32x4(_) => VertexFormat::Float32x4,
            Self::Uint32x2(_) => VertexFormat::Uint32x2,
            Self::Uint32x4(_) => VertexFormat::Uint32x4,
            Self::Unorm8x4(_) => VertexFormat::Unorm8x4,
        }
    }

    /// Whether this value can be fetched as `format`.
    #[must_use]
    pub const fn matches(&self, format: VertexFormat) -> bool {
        // `const` comparison of two enums: no `PartialEq` in const context, so
        // the check is spelled as a match on both.
        matches!(
            (self, format),
            (Self::Float32(_), VertexFormat::Float32)
                | (Self::Float32x2(_), VertexFormat::Float32x2)
                | (Self::Float32x3(_), VertexFormat::Float32x3)
                | (Self::Float32x4(_), VertexFormat::Float32x4)
                | (Self::Uint32x2(_), VertexFormat::Uint32x2)
                | (Self::Uint32x4(_), VertexFormat::Uint32x4)
                | (Self::Unorm8x4(_), VertexFormat::Unorm8x4)
        )
    }

    /// Writes vertex `index`'s value into `out` as `format`'s bytes.
    ///
    /// The caller has already checked the format ([`matches`](Self::matches)),
    /// so this cannot fail; a wrong format is a programming error and panics
    /// rather than uploading reinterpreted bytes.
    pub(crate) fn write_vertex(&self, index: usize, out: &mut Vec<u8>) {
        match self {
            Self::Float32(values) => out.extend_from_slice(&values[index].to_le_bytes()),
            Self::Float32x2(values) => {
                for value in values[index] {
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
            Self::Float32x3(values) => {
                for value in values[index] {
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
            Self::Float32x4(values) => {
                for value in values[index] {
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
            Self::Uint32x2(values) => {
                for value in values[index] {
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
            Self::Uint32x4(values) => {
                for value in values[index] {
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
            Self::Unorm8x4(values) => out.extend_from_slice(&values[index]),
        }
    }

    /// The `[f32; 3]` positions, when this is a `Float32x3` attribute.
    ///
    /// Used to derive a mesh's bounds; a mesh with no such attribute has no
    /// computed bounds (a renderer may still supply them).
    #[must_use]
    pub fn as_positions(&self) -> Option<&[[f32; 3]]> {
        match self {
            Self::Float32x3(values) => Some(values),
            _ => None,
        }
    }
}

