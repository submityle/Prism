//! Device-independent morph-target position deltas.
//!
//! A target can be stored densely for small or frequently sampled targets, or
//! sparsely for targets that touch only a small portion of the base mesh.  The
//! constructors validate the representation so a consumer can rely on the
//! vertex count and sparse lookup order without silently accepting malformed
//! asset data.

use core::fmt;

use alloc::string::String;
use alloc::vec::Vec;

/// Position-space delta contributed by one morph target at one vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MorphDelta {
    /// XYZ position displacement in the mesh's local coordinate space.
    pub position: [f32; 3],
}

impl MorphDelta {
    /// Creates a position delta.
    #[must_use]
    pub const fn position(position: [f32; 3]) -> Self {
        Self { position }
    }
}

/// Dense or sparse storage for a morph target.
#[derive(Clone, Debug, PartialEq)]
pub enum MorphTargetData {
    /// One delta for every vertex, in base-mesh vertex order.
    Dense(Vec<MorphDelta>),
    /// Only changed vertices, with strictly increasing base-mesh indices.
    Sparse {
        /// Base-mesh vertex indices for [`Self::Sparse::deltas`].
        indices: Vec<u32>,
        /// Deltas paired one-to-one with `indices`.
        deltas: Vec<MorphDelta>,
    },
}

/// A named and validated morph target.
#[derive(Clone, Debug, PartialEq)]
pub struct MorphTarget {
    name: String,
    vertex_count: u32,
    data: MorphTargetData,
}

impl MorphTarget {
    /// Creates a dense target.  Its vertex count is inferred from `deltas`.
    pub fn new_dense<N: Into<String>>(
        name: N,
        deltas: Vec<MorphDelta>,
    ) -> Result<Self, MorphError> {
        let vertex_count =
            u32::try_from(deltas.len()).map_err(|_| MorphError::VertexCountOverflow {
                found: deltas.len(),
            })?;
        Self::from_data(name, vertex_count, MorphTargetData::Dense(deltas))
    }

    /// Creates a sparse target for a mesh with `vertex_count` vertices.
    pub fn new_sparse<N: Into<String>>(
        name: N,
        vertex_count: u32,
        indices: Vec<u32>,
        deltas: Vec<MorphDelta>,
    ) -> Result<Self, MorphError> {
        Self::from_data(
            name,
            vertex_count,
            MorphTargetData::Sparse { indices, deltas },
        )
    }

    /// Creates a target from an explicit storage representation.
    pub fn from_data<N: Into<String>>(
        name: N,
        vertex_count: u32,
        data: MorphTargetData,
    ) -> Result<Self, MorphError> {
        let target = Self {
            name: name.into(),
            vertex_count,
            data,
        };
        target.validate()?;
        Ok(target)
    }

    /// Target name as authored by the asset source.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of vertices in the base mesh.
    #[must_use]
    pub const fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Read-only access to the validated representation.
    #[must_use]
    pub const fn data(&self) -> &MorphTargetData {
        &self.data
    }

    /// Returns the target's delta for a base-mesh vertex.
    #[must_use]
    pub fn delta_at(&self, vertex_index: u32) -> Option<MorphDelta> {
        if vertex_index >= self.vertex_count {
            return None;
        }
        match &self.data {
            MorphTargetData::Dense(deltas) => usize::try_from(vertex_index)
                .ok()
                .and_then(|index| deltas.get(index).copied()),
            MorphTargetData::Sparse { indices, deltas } => indices
                .binary_search(&vertex_index)
                .ok()
                .and_then(|index| deltas.get(index).copied()),
        }
    }

    /// Number of stored delta entries.  For dense data this equals the base
    /// mesh vertex count; for sparse data it is the changed-vertex count.
    #[must_use]
    pub const fn stored_delta_count(&self) -> usize {
        match &self.data {
            MorphTargetData::Dense(deltas) => deltas.len(),
            MorphTargetData::Sparse { deltas, .. } => deltas.len(),
        }
    }

    /// Validates the representation and all delta values.
    pub fn validate(&self) -> Result<(), MorphError> {
        match &self.data {
            MorphTargetData::Dense(deltas) => {
                let expected = usize::try_from(self.vertex_count).unwrap_or(usize::MAX);
                if deltas.len() != expected {
                    return Err(MorphError::DenseLengthMismatch {
                        expected: self.vertex_count,
                        found: deltas.len(),
                    });
                }
                for (delta_index, delta) in deltas.iter().enumerate() {
                    validate_delta(delta, u32::try_from(delta_index).unwrap_or(u32::MAX))?;
                }
            }
            MorphTargetData::Sparse { indices, deltas } => {
                if indices.len() != deltas.len() {
                    return Err(MorphError::SparseLengthMismatch {
                        indices: indices.len(),
                        deltas: deltas.len(),
                    });
                }
                let mut previous = None;
                for (entry, (&index, delta)) in indices.iter().zip(deltas).enumerate() {
                    if index >= self.vertex_count {
                        return Err(MorphError::SparseIndexOutOfRange {
                            entry,
                            index,
                            vertex_count: self.vertex_count,
                        });
                    }
                    if let Some(previous) = previous {
                        if index <= previous {
                            return Err(MorphError::SparseIndicesNotStrictlyIncreasing {
                                entry,
                                previous,
                                current: index,
                            });
                        }
                    }
                    validate_delta(delta, index)?;
                    previous = Some(index);
                }
            }
        }
        Ok(())
    }
}

fn validate_delta(delta: &MorphDelta, vertex_index: u32) -> Result<(), MorphError> {
    for (component, value) in delta.position.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(MorphError::NonFiniteDelta {
                vertex_index,
                component,
            });
        }
    }
    Ok(())
}

/// A violated morph-target contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MorphError {
    /// A dense target's `usize` length cannot be represented as a `u32` count.
    VertexCountOverflow {
        /// Number of authored entries.
        found: usize,
    },
    /// Dense storage does not contain one delta per base-mesh vertex.
    DenseLengthMismatch {
        /// Declared base-mesh vertex count.
        expected: u32,
        /// Number of dense deltas.
        found: usize,
    },
    /// Sparse indices and deltas are not paired one-to-one.
    SparseLengthMismatch {
        /// Number of sparse indices.
        indices: usize,
        /// Number of sparse deltas.
        deltas: usize,
    },
    /// A sparse index is outside the base mesh.
    SparseIndexOutOfRange {
        /// Entry position in the sparse arrays.
        entry: usize,
        /// Referenced vertex index.
        index: u32,
        /// Declared base-mesh vertex count.
        vertex_count: u32,
    },
    /// Sparse indices must be strictly increasing, so duplicate entries and
    /// unsorted input are both rejected.
    SparseIndicesNotStrictlyIncreasing {
        /// Entry position in the sparse arrays.
        entry: usize,
        /// Previous sparse index.
        previous: u32,
        /// Current sparse index.
        current: u32,
    },
    /// One component of a position delta is NaN or infinite.
    NonFiniteDelta {
        /// Base-mesh vertex index.
        vertex_index: u32,
        /// XYZ component index.
        component: usize,
    },
}

impl fmt::Display for MorphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VertexCountOverflow { found } => {
                write!(f, "dense morph target has {found} entries, exceeding u32 vertex count")
            }
            Self::DenseLengthMismatch { expected, found } => write!(
                f,
                "dense morph target has {found} deltas but the mesh has {expected} vertices"
            ),
            Self::SparseLengthMismatch { indices, deltas } => write!(
                f,
                "sparse morph target has {indices} indices but {deltas} deltas"
            ),
            Self::SparseIndexOutOfRange {
                entry,
                index,
                vertex_count,
            } => write!(
                f,
                "sparse morph entry {entry} targets vertex {index}, outside {vertex_count} vertices"
            ),
            Self::SparseIndicesNotStrictlyIncreasing {
                entry,
                previous,
                current,
            } => write!(
                f,
                "sparse morph entry {entry} has index {current} after {previous}; indices must increase strictly"
            ),
            Self::NonFiniteDelta {
                vertex_index,
                component,
            } => write!(
                f,
                "morph delta for vertex {vertex_index} has a non-finite component {component}"
            ),
        }
    }
}
