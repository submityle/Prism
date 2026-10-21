//! Renderer-independent consumer for versioned reconstructed surface meshes.
//!
//! The consumer owns the temporal pair needed by motion-vector producers, but
//! does not choose a vertex format, allocate a GPU resource, or depend on a
//! renderer backend. Accepted meshes are moved into the consumer so publishing
//! a large surface does not copy its vertex or index arrays.

use std::fmt;

use prism_physics_core::reconstruct::SurfaceMesh;

/// Version of the dynamic surface mesh consumer contract.
pub const DYNAMIC_SURFACE_MESH_CONTRACT_VERSION: u32 = 1;

/// Validation failures for a reconstructed surface mesh or its revision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DynamicSurfaceMeshError {
    /// Revision zero is reserved for an uninitialized consumer.
    InvalidRevision,
    /// The incoming revision is not newer than the accepted revision.
    NonMonotonicRevision { previous: u64, incoming: u64 },
    /// Every position needs one corresponding normal.
    PositionNormalCountMismatch { positions: usize, normals: usize },
    /// A position contains a NaN or infinity.
    NonFinitePosition { index: usize },
    /// A normal contains a NaN or infinity.
    NonFiniteNormal { index: usize },
    /// A normal with no direction cannot shade a surface.
    ZeroLengthNormal { index: usize },
    /// Triangle indices must be grouped in triples.
    IndexCountNotMultipleOfThree { count: usize },
    /// An index does not refer to a supplied vertex.
    IndexOutOfBounds { index: usize, vertex_count: usize },
}

impl fmt::Display for DynamicSurfaceMeshError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRevision => {
                formatter.write_str("dynamic surface revision must be non-zero")
            }
            Self::NonMonotonicRevision { previous, incoming } => write!(
                formatter,
                "dynamic surface revision {incoming} is not newer than {previous}"
            ),
            Self::PositionNormalCountMismatch { positions, normals } => write!(
                formatter,
                "surface mesh has {positions} positions but {normals} normals"
            ),
            Self::NonFinitePosition { index } => {
                write!(formatter, "surface mesh position {index} is not finite")
            }
            Self::NonFiniteNormal { index } => {
                write!(formatter, "surface mesh normal {index} is not finite")
            }
            Self::ZeroLengthNormal { index } => {
                write!(formatter, "surface mesh normal {index} has zero length")
            }
            Self::IndexCountNotMultipleOfThree { count } => write!(
                formatter,
                "surface mesh has {count} indices, which is not a multiple of three"
            ),
            Self::IndexOutOfBounds {
                index,
                vertex_count,
            } => write!(
                formatter,
                "surface mesh index {index} is outside {vertex_count} vertices"
            ),
        }
    }
}

impl std::error::Error for DynamicSurfaceMeshError {}

/// A read-only temporal view of one accepted dynamic surface mesh state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicSurfaceMeshView<'a> {
    /// The newest accepted revision, or `None` before the first publication.
    pub revision: Option<u64>,
    /// The current mesh. `None` means the latest publication cleared it.
    pub current: Option<&'a SurfaceMesh>,
    /// The mesh accepted immediately before the current publication, when one
    /// exists. It may remain available while `current` is cleared.
    pub previous: Option<&'a SurfaceMesh>,
    /// Whether both meshes form a continuous pair suitable for motion vectors.
    pub history_valid: bool,
}

/// Owns the current and previous reconstructed surface meshes for one object.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DynamicSurfaceMeshConsumer {
    revision: Option<u64>,
    previous: Option<SurfaceMesh>,
    current: Option<SurfaceMesh>,
    history_valid: bool,
}

impl DynamicSurfaceMeshConsumer {
    /// Creates an empty consumer with no accepted revision.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            revision: None,
            previous: None,
            current: None,
            history_valid: false,
        }
    }

    /// Publishes a validated mesh at a strictly newer revision.
    ///
    /// The input is moved into the consumer after validation. A publication
    /// with no triangles clears `current`; the prior current mesh is retained
    /// as `previous`, while `history_valid` becomes false because there is no
    /// current surface pair to use for motion vectors.
    pub fn publish(
        &mut self,
        revision: u64,
        mesh: SurfaceMesh,
    ) -> Result<(), DynamicSurfaceMeshError> {
        if revision == 0 {
            return Err(DynamicSurfaceMeshError::InvalidRevision);
        }
        if let Some(previous) = self.revision
            && revision <= previous
        {
            return Err(DynamicSurfaceMeshError::NonMonotonicRevision {
                previous,
                incoming: revision,
            });
        }
        validate_surface_mesh(&mesh)?;

        let old_current = self.current.take();
        let next_current = (!mesh.indices.is_empty()).then_some(mesh);
        self.previous = old_current;
        self.history_valid = next_current.is_some() && self.previous.is_some();
        self.current = next_current;
        self.revision = Some(revision);
        Ok(())
    }

    /// Returns the newest accepted revision.
    #[must_use]
    pub const fn revision(&self) -> Option<u64> {
        self.revision
    }

    /// Returns the current mesh, or `None` when the latest publication cleared it.
    #[must_use]
    pub fn current(&self) -> Option<&SurfaceMesh> {
        self.current.as_ref()
    }

    /// Returns the mesh from the immediately preceding accepted publication.
    #[must_use]
    pub fn previous(&self) -> Option<&SurfaceMesh> {
        self.previous.as_ref()
    }

    /// Returns whether current and previous form a continuous motion-vector pair.
    #[must_use]
    pub const fn history_valid(&self) -> bool {
        self.history_valid
    }

    /// Returns an immutable view without copying either mesh.
    #[must_use]
    pub fn view(&self) -> DynamicSurfaceMeshView<'_> {
        DynamicSurfaceMeshView {
            revision: self.revision,
            current: self.current(),
            previous: self.previous(),
            history_valid: self.history_valid,
        }
    }
}

fn validate_surface_mesh(mesh: &SurfaceMesh) -> Result<(), DynamicSurfaceMeshError> {
    if mesh.positions.len() != mesh.normals.len() {
        return Err(DynamicSurfaceMeshError::PositionNormalCountMismatch {
            positions: mesh.positions.len(),
            normals: mesh.normals.len(),
        });
    }
    for (index, position) in mesh.positions.iter().enumerate() {
        if !position.is_finite() {
            return Err(DynamicSurfaceMeshError::NonFinitePosition { index });
        }
    }
    for (index, normal) in mesh.normals.iter().enumerate() {
        if !normal.is_finite() {
            return Err(DynamicSurfaceMeshError::NonFiniteNormal { index });
        }
        if normal.length_squared() <= f32::EPSILON {
            return Err(DynamicSurfaceMeshError::ZeroLengthNormal { index });
        }
    }
    if !mesh.indices.len().is_multiple_of(3) {
        return Err(DynamicSurfaceMeshError::IndexCountNotMultipleOfThree {
            count: mesh.indices.len(),
        });
    }
    let vertex_count = mesh.positions.len();
    for &index in &mesh.indices {
        let index = index as usize;
        if index >= vertex_count {
            return Err(DynamicSurfaceMeshError::IndexOutOfBounds {
                index,
                vertex_count,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use glam::Vec3;
    use prism_physics_core::reconstruct::SurfaceMesh;

    use super::*;

    fn triangle(offset: f32) -> SurfaceMesh {
        SurfaceMesh {
            positions: vec![
                Vec3::new(offset, 0.0, 0.0),
                Vec3::new(offset + 1.0, 0.0, 0.0),
                Vec3::new(offset, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z; 3],
            indices: vec![0, 1, 2],
        }
    }

    #[test]
    fn publishes_current_and_previous_without_copying_api_state() {
        let mut consumer = DynamicSurfaceMeshConsumer::new();
        consumer.publish(1, triangle(0.0)).unwrap();
        assert_eq!(consumer.revision(), Some(1));
        assert!(consumer.previous().is_none());
        assert!(!consumer.history_valid());

        consumer.publish(2, triangle(2.0)).unwrap();
        assert_eq!(consumer.current().unwrap().positions[0].x, 2.0);
        assert_eq!(consumer.previous().unwrap().positions[0].x, 0.0);
        assert!(consumer.history_valid());
        assert_eq!(consumer.view().revision, Some(2));
    }

    #[test]
    fn empty_publication_clears_current_and_preserves_previous() {
        let mut consumer = DynamicSurfaceMeshConsumer::new();
        consumer.publish(1, triangle(0.0)).unwrap();
        consumer.publish(2, SurfaceMesh::new()).unwrap();

        assert!(consumer.current().is_none());
        assert_eq!(consumer.previous().unwrap().positions[0].x, 0.0);
        assert!(!consumer.history_valid());

        consumer.publish(3, triangle(3.0)).unwrap();
        assert_eq!(consumer.current().unwrap().positions[0].x, 3.0);
        assert!(consumer.previous().is_none());
        assert!(!consumer.history_valid());
    }

    #[test]
    fn rejects_invalid_mesh_data_without_mutating_state() {
        let mut consumer = DynamicSurfaceMeshConsumer::new();
        consumer.publish(1, triangle(0.0)).unwrap();
        let accepted = consumer.clone();

        let mut mismatch = triangle(1.0);
        mismatch.normals.pop();
        assert_eq!(
            consumer.publish(2, mismatch),
            Err(DynamicSurfaceMeshError::PositionNormalCountMismatch {
                positions: 3,
                normals: 2,
            })
        );

        let mut non_finite = triangle(1.0);
        non_finite.positions[1].x = f32::NAN;
        assert_eq!(
            consumer.publish(2, non_finite),
            Err(DynamicSurfaceMeshError::NonFinitePosition { index: 1 })
        );

        let mut bad_indices = triangle(1.0);
        bad_indices.indices = vec![0, 1, 7];
        assert_eq!(
            consumer.publish(2, bad_indices),
            Err(DynamicSurfaceMeshError::IndexOutOfBounds {
                index: 7,
                vertex_count: 3,
            })
        );
        assert_eq!(consumer, accepted);
    }

    #[test]
    fn rejects_non_monotonic_revisions_without_mutating_state() {
        let mut consumer = DynamicSurfaceMeshConsumer::new();
        consumer.publish(4, triangle(0.0)).unwrap();
        let accepted = consumer.clone();

        assert_eq!(
            consumer.publish(4, triangle(1.0)),
            Err(DynamicSurfaceMeshError::NonMonotonicRevision {
                previous: 4,
                incoming: 4,
            })
        );
        assert_eq!(
            consumer.publish(3, triangle(1.0)),
            Err(DynamicSurfaceMeshError::NonMonotonicRevision {
                previous: 4,
                incoming: 3,
            })
        );
        assert_eq!(consumer, accepted);
    }

    #[test]
    fn validates_normals_and_index_shape() {
        let mut consumer = DynamicSurfaceMeshConsumer::new();

        let mut zero_normal = triangle(0.0);
        zero_normal.normals[2] = Vec3::ZERO;
        assert_eq!(
            consumer.publish(1, zero_normal),
            Err(DynamicSurfaceMeshError::ZeroLengthNormal { index: 2 })
        );

        let mut non_finite_normal = triangle(0.0);
        non_finite_normal.normals[0].y = f32::INFINITY;
        assert_eq!(
            consumer.publish(1, non_finite_normal),
            Err(DynamicSurfaceMeshError::NonFiniteNormal { index: 0 })
        );

        let mut malformed_indices = triangle(0.0);
        malformed_indices.indices.pop();
        assert_eq!(
            consumer.publish(1, malformed_indices),
            Err(DynamicSurfaceMeshError::IndexCountNotMultipleOfThree { count: 2 })
        );
        assert!(consumer.current().is_none());
    }
}
