//! What a mesh can be wrong about.

use core::fmt;

use prism_render_driver::PrimitiveTopology;

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
    /// Two attributes use the same semantic name at different locations.
    DuplicateAttributeName {
        /// The duplicated attribute semantic.
        attribute: &'static str,
    },
    /// Two attributes claim the same shader location.
    DuplicateShaderLocation {
        /// The already-assigned shader location.
        shader_location: u32,
    },
    /// A value's storage does not match the attribute's declared format.
    FormatMismatch {
        /// The offending attribute's semantic name.
        attribute: &'static str,
    },
    /// A floating-point attribute contains NaN or infinity.
    NonFiniteAttributeValue {
        /// The offending attribute's semantic name.
        attribute: &'static str,
        /// The vertex containing the non-finite component.
        vertex: usize,
        /// The component within that vertex.
        component: usize,
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
    /// Submeshes were supplied for a non-indexed mesh.
    SubMeshesRequireIndices,
    /// A vertex or index count exceeds the range represented by the mesh ABI.
    CountExceedsU32 {
        /// The kind of count that overflowed (for example, `vertex` or `index`).
        kind: &'static str,
        /// The original count.
        found: usize,
    },
    /// A list topology was given a trailing partial primitive.
    TopologyElementCount {
        /// The RHI topology being validated.
        topology: PrimitiveTopology,
        /// The number of vertices or indices supplied.
        element_count: u32,
    },
    /// A list-topology submesh starts in the middle of a primitive.
    SubMeshIndexRangeMisaligned {
        /// Start of the half-open range.
        start: u32,
        /// The RHI topology being validated.
        topology: PrimitiveTopology,
        /// Number of indices per complete list primitive.
        elements_per_primitive: u32,
    },
    /// A submesh index range is outside its parent index buffer.
    SubMeshIndexRangeOutOfBounds {
        /// Start of the half-open range.
        start: u32,
        /// End of the half-open range.
        end: u32,
        /// Parent index-buffer length.
        index_count: u32,
    },
    /// A submesh has no indices.
    EmptySubMeshIndexRange,
    /// A submesh has indices but cannot assemble one primitive.
    SubMeshHasNoPrimitives {
        /// Material slot associated with the submesh.
        material_slot: u16,
        /// Number of indices in the range.
        index_count: u32,
        /// The RHI topology being validated.
        topology: PrimitiveTopology,
    },
    /// An optional cluster range is malformed.
    SubMeshClusterRangeOutOfBounds {
        /// Start of the half-open range.
        start: u32,
        /// End of the half-open range.
        end: u32,
        /// The largest representable cluster bound.
        cluster_count: u32,
    },
    /// An optional cluster range is empty.
    EmptySubMeshClusterRange,
    /// Two submesh index ranges overlap.
    SubMeshIndexRangesOverlap {
        /// First submesh position in the table.
        first: usize,
        /// Second submesh position in the table.
        second: usize,
    },
    /// The material slot is outside the configured slot table.
    MaterialSlotOutOfRange {
        /// The requested slot.
        slot: u16,
        /// Number of slots available.
        maximum: u32,
    },
    /// The vertex count exceeds the mesh profile.
    VertexLimitExceeded {
        /// Actual count.
        actual: u64,
        /// Maximum accepted count.
        maximum: u64,
    },
    /// The attribute count exceeds the mesh profile.
    AttributeLimitExceeded {
        /// Actual count.
        actual: u32,
        /// Maximum accepted count.
        maximum: u32,
    },
    /// The stream count exceeds the mesh profile.
    StreamLimitExceeded {
        /// Actual count.
        actual: u32,
        /// Maximum accepted count.
        maximum: u32,
    },
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
            Self::DuplicateAttributeName { attribute } => {
                write!(f, "attribute semantic {attribute} is declared more than once")
            }
            Self::DuplicateShaderLocation { shader_location } => {
                write!(f, "shader location {shader_location} is already assigned")
            }
            Self::FormatMismatch { attribute } => {
                write!(
                    f,
                    "attribute {attribute} does not match its declared format"
                )
            }
            Self::NonFiniteAttributeValue {
                attribute,
                vertex,
                component,
            } => write!(
                f,
                "attribute {attribute} vertex {vertex} component {component} is not finite"
            ),
            Self::IndexOutOfRange {
                max_index,
                vertex_count,
            } => write!(
                f,
                "index {max_index} is outside the mesh's {vertex_count} vertices"
            ),
            Self::NoAttributes => write!(f, "the mesh has no vertex attributes"),
            Self::SubMeshesRequireIndices => {
                write!(f, "submeshes require an indexed mesh")
            }
            Self::CountExceedsU32 { kind, found } => {
                write!(f, "mesh {kind} count {found} exceeds the u32 limit")
            }
            Self::TopologyElementCount {
                topology,
                element_count,
            } => write!(f, "{topology:?} cannot assemble {element_count} elements"),
            Self::SubMeshIndexRangeMisaligned {
                start,
                topology,
                elements_per_primitive,
            } => write!(
                f,
                "submesh index range starts at {start}, not aligned to {elements_per_primitive}-element {topology:?} primitives"
            ),
            Self::SubMeshIndexRangeOutOfBounds {
                start,
                end,
                index_count,
            } => write!(
                f,
                "submesh index range {start}..{end} is outside {index_count} indices"
            ),
            Self::EmptySubMeshIndexRange => write!(f, "submesh index range is empty"),
            Self::SubMeshHasNoPrimitives {
                material_slot,
                index_count,
                topology,
            } => write!(
                f,
                "submesh material slot {material_slot} has {index_count} indices but no {topology:?} primitive"
            ),
            Self::SubMeshClusterRangeOutOfBounds {
                start,
                end,
                cluster_count,
            } => write!(
                f,
                "submesh cluster range {start}..{end} is outside {cluster_count} clusters"
            ),
            Self::EmptySubMeshClusterRange => write!(f, "submesh cluster range is empty"),
            Self::SubMeshIndexRangesOverlap { first, second } => write!(
                f,
                "submesh index ranges at positions {first} and {second} overlap"
            ),
            Self::MaterialSlotOutOfRange { slot, maximum } => write!(
                f,
                "material slot {slot} is outside the {maximum}-slot mesh profile"
            ),
            Self::VertexLimitExceeded { actual, maximum } => {
                write!(f, "mesh has {actual} vertices, maximum is {maximum}")
            }
            Self::AttributeLimitExceeded { actual, maximum } => {
                write!(f, "mesh has {actual} attributes, maximum is {maximum}")
            }
            Self::StreamLimitExceeded { actual, maximum } => {
                write!(f, "mesh has {actual} streams, maximum is {maximum}")
            }
        }
    }
}
