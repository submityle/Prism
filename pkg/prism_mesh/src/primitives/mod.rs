//! Deterministic built-in mesh generators.

mod capsule;
mod common;
mod cube;
mod plane;
mod sphere;
mod terrain;

pub use capsule::capsule;
pub use cube::cube;
pub use plane::plane;
pub use sphere::uv_sphere;
pub use terrain::terrain_patch;

/// Mesh primitive generation failure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrimitiveError {
    /// A dimension or radius is not finite or is not strictly positive.
    InvalidDimension {
        /// Dimension label used in diagnostics.
        name: &'static str,
        /// Supplied value.
        value: f32,
    },
    /// A subdivision count is below the generator's minimum.
    InvalidSubdivision {
        /// Subdivision label used in diagnostics.
        name: &'static str,
        /// Supplied value.
        value: u32,
        /// Minimum accepted value.
        minimum: u32,
    },
    /// Generated vertex/index counts exceed the Mesh ABI's u32 range.
    CountOverflow {
        /// `vertex` or `index`.
        kind: &'static str,
    },
    /// The complete generated vertex/index payload exceeds the primitive
    /// generator's bounded working-set budget.
    WorkingSetExceeded {
        /// Requested payload bytes.
        requested: u64,
        /// Maximum accepted payload bytes.
        maximum: u64,
    },
    /// Output reservation failed before generation started.
    AllocationFailed {
        /// `vertex` or `index`.
        kind: &'static str,
        /// Requested element count.
        elements: usize,
    },
    /// The generated data could not satisfy the mesh contract.
    Mesh(crate::MeshError),
    /// Height sample count does not match the requested grid.
    HeightCountMismatch {
        /// Expected number of grid samples.
        expected: usize,
        /// Number of supplied samples.
        found: usize,
    },
    /// At least one height sample is NaN or infinite.
    NonFiniteHeight,
}

impl core::fmt::Display for PrimitiveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDimension { name, value } => {
                write!(
                    f,
                    "primitive dimension {name} must be finite and positive, got {value}"
                )
            }
            Self::InvalidSubdivision {
                name,
                value,
                minimum,
            } => write!(
                f,
                "primitive subdivision {name} must be >= {minimum}, got {value}"
            ),
            Self::CountOverflow { kind } => {
                write!(f, "generated mesh {kind} count exceeds the u32 mesh ABI")
            }
            Self::WorkingSetExceeded { requested, maximum } => write!(
                f,
                "generated primitive needs {requested} bytes, exceeding the {maximum}-byte working-set limit"
            ),
            Self::AllocationFailed { kind, elements } => {
                write!(f, "could not reserve {elements} generated mesh {kind} elements")
            }
            Self::Mesh(error) => write!(f, "generated primitive violates mesh contract: {error}"),
            Self::HeightCountMismatch { expected, found } => {
                write!(f, "terrain height count is {found}, expected {expected}")
            }
            Self::NonFiniteHeight => f.write_str("terrain contains a non-finite height"),
        }
    }
}

impl From<crate::MeshError> for PrimitiveError {
    fn from(value: crate::MeshError) -> Self {
        Self::Mesh(value)
    }
}

pub(crate) const POSITION: crate::MeshVertexAttribute = crate::MeshVertexAttribute::new(
    crate::POSITION_ATTRIBUTE,
    prism_render_driver::VertexFormat::Float32x3,
    0,
);
pub(crate) const NORMAL: crate::MeshVertexAttribute = crate::MeshVertexAttribute::new(
    "Vertex_Normal",
    prism_render_driver::VertexFormat::Float32x3,
    1,
);
pub(crate) const UV0: crate::MeshVertexAttribute = crate::MeshVertexAttribute::new(
    "Vertex_Uv0",
    prism_render_driver::VertexFormat::Float32x2,
    2,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generators_produce_valid_indexed_meshes() {
        let plane = plane([4.0, 2.0], [2, 1]).expect("plane");
        assert_eq!(plane.vertex_count(), 6);
        assert_eq!(plane.indices().expect("indices").len(), 12);
        assert_eq!(plane.validate(), Ok(()));

        let cube = cube([2.0, 4.0, 6.0]).expect("cube");
        assert_eq!(cube.vertex_count(), 24);
        assert_eq!(cube.indices().expect("indices").len(), 36);
        assert_eq!(cube.bounds().expect("bounds").min, [-1.0, -2.0, -3.0]);

        let sphere = uv_sphere(2.0, 8, 4).expect("sphere");
        assert_eq!(sphere.vertex_count(), 45);
        assert_eq!(sphere.indices().expect("indices").len(), 8 * 4 * 6);

        let capsule = capsule(0.5, 1.0, 8, 2).expect("capsule");
        assert_eq!(capsule.validate(), Ok(()));
        assert!(capsule.bounds().expect("bounds").max[1] > 1.4);

        let terrain = terrain_patch([2.0, 2.0], [1, 1], &[0.0, 1.0, 1.0, 0.0]).expect("terrain");
        assert_eq!(terrain.vertex_count(), 4);
        assert_eq!(terrain.validate(), Ok(()));
    }

    #[test]
    fn generators_reject_invalid_dimensions_and_resolution() {
        assert!(matches!(
            plane([0.0, 1.0], [1, 1]),
            Err(PrimitiveError::InvalidDimension { name: "width", .. })
        ));
        assert!(matches!(
            uv_sphere(1.0, 2, 4),
            Err(PrimitiveError::InvalidSubdivision {
                name: "sectors",
                minimum: 3,
                ..
            })
        ));
        assert!(matches!(
            capsule(1.0, f32::NAN, 8, 2),
            Err(PrimitiveError::InvalidDimension {
                name: "half_length",
                ..
            })
        ));
        assert!(matches!(
            terrain_patch([1.0, 1.0], [1, 1], &[0.0]),
            Err(PrimitiveError::HeightCountMismatch { .. })
        ));
    }

    #[test]
    fn generators_reject_unrepresentable_counts_before_allocating_or_iterating() {
        assert!(matches!(
            plane([1.0, 1.0], [u32::MAX, 1]),
            Err(PrimitiveError::CountOverflow { kind: "vertex" })
        ));
        assert!(matches!(
            terrain_patch([1.0, 1.0], [u32::MAX, 1], &[]),
            Err(PrimitiveError::CountOverflow { kind: "vertex" })
        ));
        assert!(matches!(
            uv_sphere(1.0, u32::MAX, 2),
            Err(PrimitiveError::CountOverflow { kind: "vertex" })
        ));
        assert!(matches!(
            capsule(1.0, 0.0, u32::MAX, 1),
            Err(PrimitiveError::CountOverflow { kind: "vertex" })
        ));

        // Counts can fit the vertex ABI while the six-index-per-cell expansion
        // exceeds the index ABI; reject it before attempting a multi-gigabyte
        // vertex allocation.
        assert!(matches!(
            plane([1.0, 1.0], [u32::MAX / 6 + 1, 1]),
            Err(PrimitiveError::CountOverflow { kind: "index" })
        ));
        assert!(matches!(
            plane([1.0, 1.0], [4096, 4096]),
            Err(PrimitiveError::WorkingSetExceeded { .. })
        ));
    }
}
