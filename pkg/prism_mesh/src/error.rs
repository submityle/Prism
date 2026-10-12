//! What a mesh can be wrong about.

use core::fmt;

/// A mesh that cannot be uploaded as authored.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MeshError {
    /// An attribute has a different number of elements than the mesh's vertices.
    ///
    /// Every attribute is indexed by the same vertex, so a short attribute would
    /// read past its own data for the last vertices.
    VertexCountMismatch {
        /// The offending attribute's semantic name.
        attribute: &'static str,
        /// The mesh's vertex count.
        expected: usize,
        /// The attribute's element count.
        found: usize,
    },
    /// A value's storage does not match the attribute's declared format.
    FormatMismatch {
        /// The offending attribute's semantic name.
        attribute: &'static str,
    },
    /// An index points outside the mesh's vertices.
    IndexOutOfRange {
        /// The largest index in the list.
        max_index: u32,
        /// The mesh's vertex count.
        vertex_count: u32,
    },
    /// The mesh has no attributes, so it has no vertex stage to draw with.
    NoAttributes,
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VertexCountMismatch {
                attribute,
                expected,
                found,
            } => write!(
                f,
                "attribute {attribute} has {found} elements, the mesh has {expected} vertices"
            ),
            Self::FormatMismatch { attribute } => {
                write!(f, "attribute {attribute} does not match its declared format")
            }
            Self::IndexOutOfRange {
                max_index,
                vertex_count,
            } => write!(
                f,
                "index {max_index} is outside the mesh's {vertex_count} vertices"
            ),
            Self::NoAttributes => write!(f, "the mesh has no vertex attributes"),
        }
    }
}

