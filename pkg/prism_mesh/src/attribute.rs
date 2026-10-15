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
    /// Two IEEE-754 binary16 values per vertex, stored as their raw bits.
    Float16x2(Vec<[u16; 2]>),
    /// Four IEEE-754 binary16 values per vertex, stored as their raw bits.
    Float16x4(Vec<[u16; 4]>),
    /// One `u32` per vertex.
    Uint32(Vec<u32>),
    /// Two `u32` per vertex.
    Uint32x2(Vec<[u32; 2]>),
    /// Four `u32` per vertex.
    Uint32x4(Vec<[u32; 4]>),
    /// One `i32` per vertex.
    Sint32(Vec<i32>),
    /// Four `u8` normalised to `[0, 1]` per vertex.
    Unorm8x4(Vec<[u8; 4]>),
    /// Four `i8` normalised to `[-1, 1]` per vertex.
    Snorm8x4(Vec<[i8; 4]>),
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
            Self::Float16x2(values) => values.len(),
            Self::Float16x4(values) => values.len(),
            Self::Uint32(values) => values.len(),
            Self::Uint32x2(values) => values.len(),
            Self::Uint32x4(values) => values.len(),
            Self::Sint32(values) => values.len(),
            Self::Unorm8x4(values) => values.len(),
            Self::Snorm8x4(values) => values.len(),
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
            Self::Float16x2(_) => VertexFormat::Float16x2,
            Self::Float16x4(_) => VertexFormat::Float16x4,
            Self::Uint32(_) => VertexFormat::Uint32,
            Self::Uint32x2(_) => VertexFormat::Uint32x2,
            Self::Uint32x4(_) => VertexFormat::Uint32x4,
            Self::Sint32(_) => VertexFormat::Sint32,
            Self::Unorm8x4(_) => VertexFormat::Unorm8x4,
            Self::Snorm8x4(_) => VertexFormat::Snorm8x4,
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
                | (Self::Float16x2(_), VertexFormat::Float16x2)
                | (Self::Float16x4(_), VertexFormat::Float16x4)
                | (Self::Uint32(_), VertexFormat::Uint32)
                | (Self::Uint32x2(_), VertexFormat::Uint32x2)
                | (Self::Uint32x4(_), VertexFormat::Uint32x4)
                | (Self::Sint32(_), VertexFormat::Sint32)
                | (Self::Unorm8x4(_), VertexFormat::Unorm8x4)
                | (Self::Snorm8x4(_), VertexFormat::Snorm8x4)
        )
    }

    /// Returns the first non-finite f32 component, if any, as (vertex, component).
    #[must_use]
    pub(crate) fn first_non_finite(&self) -> Option<(usize, usize)> {
        match self {
            Self::Float32(values) => values
                .iter()
                .enumerate()
                .find_map(|(vertex, value)| (!value.is_finite()).then_some((vertex, 0))),
            Self::Float32x2(values) => values.iter().enumerate().find_map(|(vertex, value)| {
                value
                    .iter()
                    .position(|component| !component.is_finite())
                    .map(|component| (vertex, component))
            }),
            Self::Float32x3(values) => values.iter().enumerate().find_map(|(vertex, value)| {
                value
                    .iter()
                    .position(|component| !component.is_finite())
                    .map(|component| (vertex, component))
            }),
            Self::Float32x4(values) => values.iter().enumerate().find_map(|(vertex, value)| {
                value
                    .iter()
                    .position(|component| !component.is_finite())
                    .map(|component| (vertex, component))
            }),
            Self::Float16x2(values) => values.iter().enumerate().find_map(|(vertex, value)| {
                value
                    .iter()
                    .position(|bits| (bits & 0x7c00) == 0x7c00)
                    .map(|component| (vertex, component))
            }),
            Self::Float16x4(values) => values.iter().enumerate().find_map(|(vertex, value)| {
                value
                    .iter()
                    .position(|bits| (bits & 0x7c00) == 0x7c00)
                    .map(|component| (vertex, component))
            }),
            _ => None,
        }
    }

    /// Gathers typed values without conversion, preserving raw half/integer
    /// bits, request order, and duplicates. Validates all indices before allocation.
    #[must_use]
    pub fn gather(&self, vertex_indices: &[u32]) -> Option<Self> {
        if vertex_indices
            .iter()
            .any(|index| *index as usize >= self.len())
        {
            return None;
        }
        vertex_indices
            .len()
            .checked_mul(self.format().size() as usize)?;
        macro_rules! gather {
            ($values:expr, $variant:ident) => {
                Self::$variant(
                    vertex_indices
                        .iter()
                        .map(|index| $values[*index as usize])
                        .collect(),
                )
            };
        }
        Some(match self {
            Self::Float32(v) => gather!(v, Float32),
            Self::Float32x2(v) => gather!(v, Float32x2),
            Self::Float32x3(v) => gather!(v, Float32x3),
            Self::Float32x4(v) => gather!(v, Float32x4),
            Self::Float16x2(v) => gather!(v, Float16x2),
            Self::Float16x4(v) => gather!(v, Float16x4),
            Self::Uint32(v) => gather!(v, Uint32),
            Self::Uint32x2(v) => gather!(v, Uint32x2),
            Self::Uint32x4(v) => gather!(v, Uint32x4),
            Self::Sint32(v) => gather!(v, Sint32),
            Self::Unorm8x4(v) => gather!(v, Unorm8x4),
            Self::Snorm8x4(v) => gather!(v, Snorm8x4),
        })
    }

    /// Gathers the requested vertices into tightly packed little-endian bytes.
    ///
    /// Request order and repeated indices are preserved. Returns `None` if an
    /// index is out of range or the byte count cannot be represented; all indices
    /// are checked before allocation. This does not change or convert the format.
    #[must_use]
    pub fn gather_bytes(&self, vertex_indices: &[u32]) -> Option<Vec<u8>> {
        if vertex_indices
            .iter()
            .any(|index| *index as usize >= self.len())
        {
            return None;
        }
        let capacity = vertex_indices
            .len()
            .checked_mul(self.format().size() as usize)?;
        let mut bytes = Vec::with_capacity(capacity);
        for index in vertex_indices {
            self.write_vertex(*index as usize, &mut bytes);
        }
        debug_assert_eq!(bytes.len(), capacity);
        Some(bytes)
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
            Self::Float16x2(values) => {
                for value in values[index] {
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
            Self::Float16x4(values) => {
                for value in values[index] {
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
            Self::Uint32(values) => out.extend_from_slice(&values[index].to_le_bytes()),
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
            Self::Sint32(values) => out.extend_from_slice(&values[index].to_le_bytes()),
            Self::Unorm8x4(values) => out.extend_from_slice(&values[index]),
            Self::Snorm8x4(values) => {
                for value in values[index] {
                    out.push(value as u8);
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_gather_preserves_bits_order_duplicates_and_empty_for_all_formats() {
        // Gathering is byte-preserving, even for data the later Mesh validator
        // will reject (NaN payloads). It must not numerically convert values.
        let nan = f32::from_bits(0x7fc0_1234);
        let values = [
            VertexAttributeValues::Float32(vec![-0.0, nan]),
            VertexAttributeValues::Float32x2(vec![[-0.0, nan], [f32::MIN, f32::MAX]]),
            VertexAttributeValues::Float32x3(vec![[-0.0, nan, 1.0], [2.0; 3]]),
            VertexAttributeValues::Float32x4(vec![[-0.0, nan, 1.0, -2.0], [2.0; 4]]),
            VertexAttributeValues::Float16x2(vec![[0x8000, 0x7e01], [0x0001, 0x7bff]]),
            VertexAttributeValues::Float16x4(vec![[0x8000, 0x7e01, 0x0001, 0x7bff], [0; 4]]),
            VertexAttributeValues::Uint32(vec![0, u32::MAX]),
            VertexAttributeValues::Uint32x2(vec![[0, u32::MAX], [7, 9]]),
            VertexAttributeValues::Uint32x4(vec![[0, u32::MAX, 1, 2], [7; 4]]),
            VertexAttributeValues::Sint32(vec![i32::MIN, i32::MAX]),
            VertexAttributeValues::Unorm8x4(vec![[0, 1, 128, 255], [7; 4]]),
            VertexAttributeValues::Snorm8x4(vec![[-128, -127, 0, 127], [7; 4]]),
        ];
        for values in values {
            let gathered = values.gather(&[1, 0, 1]).unwrap();
            assert_eq!(gathered.format(), values.format());
            assert_eq!(gathered.len(), 3);
            assert_eq!(
                gathered.gather_bytes(&[0, 1, 2]),
                values.gather_bytes(&[1, 0, 1])
            );
            assert!(values.gather(&[0, 2]).is_none());
            assert!(values.gather(&[u32::MAX]).is_none());
            let empty = values.gather(&[]).unwrap();
            assert!(empty.is_empty());
            assert_eq!(empty.format(), values.format());
            assert!(empty.gather(&[0]).is_none());
            assert!(empty.gather(&[]).unwrap().is_empty());
        }
    }
}
