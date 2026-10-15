//! Device-independent skinning influence data.
//!
//! The asset-side representation keeps authored per-vertex influence lists
//! intact; render upload chooses a fixed-width GPU stream from this validated
//! contract. Construction and [`SkinningData::validate`] enforce the
//! invariants that make that conversion safe.

use core::fmt;

use alloc::vec::Vec;

/// One bone influence for one vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkinInfluence {
    /// Index into the skeleton's joint palette.
    pub joint_index: u32,
    /// The authored influence weight.  Weights need not already sum to one,
    /// but they must be finite and non-negative.
    pub weight: f32,
}

/// Validated, device-independent skinning data for a mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct SkinningData {
    vertex_count: u32,
    joint_count: u32,
    influences: Vec<Vec<SkinInfluence>>,
}

impl SkinningData {
    /// Creates skinning data and validates its fundamental invariants.
    ///
    /// `influences` contains one list per mesh vertex.  A vertex may have no
    /// influences; a skinning consumer must then use its bind-pose value.
    pub fn new(
        vertex_count: u32,
        joint_count: u32,
        influences: Vec<Vec<SkinInfluence>>,
    ) -> Result<Self, SkinError> {
        let data = Self {
            vertex_count,
            joint_count,
            influences,
        };
        data.validate()?;
        Ok(data)
    }

    /// Number of vertices covered by this data.
    #[must_use]
    pub const fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Number of joints in the referenced skeleton palette.
    #[must_use]
    pub const fn joint_count(&self) -> u32 {
        self.joint_count
    }

    /// Returns the influences for a vertex, or `None` when the vertex is out
    /// of range.
    #[must_use]
    pub fn influences(&self, vertex_index: u32) -> Option<&[SkinInfluence]> {
        usize::try_from(vertex_index)
            .ok()
            .and_then(|index| self.influences.get(index).map(Vec::as_slice))
    }

    /// Iterates over influence lists in vertex order.
    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = &[SkinInfluence]> {
        self.influences.iter().map(Vec::as_slice)
    }

    /// Returns the sum of one vertex's weights after validation.
    #[must_use]
    pub fn weight_sum(&self, vertex_index: u32) -> Option<f32> {
        self.influences(vertex_index)
            .map(|values| values.iter().map(|influence| influence.weight).sum())
    }

    /// Returns the largest authored influence-list length.
    #[must_use]
    pub fn max_influences(&self) -> usize {
        self.influences.iter().map(Vec::len).max().unwrap_or(0)
    }

    /// Validates all invariants that are independent of a renderer profile.
    pub fn validate(&self) -> Result<(), SkinError> {
        let found = self.influences.len();
        let expected = usize::try_from(self.vertex_count).unwrap_or(usize::MAX);
        if found != expected {
            return Err(SkinError::VertexCountMismatch {
                expected: self.vertex_count,
                found,
            });
        }

        for (vertex, values) in self.influences.iter().enumerate() {
            let vertex = u32::try_from(vertex).unwrap_or(u32::MAX);
            let mut weight_sum = 0.0_f32;
            for (influence_index, influence) in values.iter().enumerate() {
                if influence.joint_index >= self.joint_count {
                    return Err(SkinError::JointIndexOutOfRange {
                        vertex,
                        influence: influence_index,
                        joint_index: influence.joint_index,
                        joint_count: self.joint_count,
                    });
                }
                if !influence.weight.is_finite() {
                    return Err(SkinError::NonFiniteWeight {
                        vertex,
                        influence: influence_index,
                    });
                }
                if influence.weight < 0.0 {
                    return Err(SkinError::NegativeWeight {
                        vertex,
                        influence: influence_index,
                    });
                }
                if values[..influence_index]
                    .iter()
                    .any(|previous| previous.joint_index == influence.joint_index)
                {
                    return Err(SkinError::DuplicateJoint {
                        vertex,
                        joint_index: influence.joint_index,
                    });
                }
                weight_sum += influence.weight;
            }
            if !weight_sum.is_finite() {
                return Err(SkinError::NonFiniteWeightSum { vertex });
            }
        }
        Ok(())
    }

    /// Validates the data against a renderer or asset-profile influence cap.
    pub fn validate_with_max_influences(&self, max_influences: usize) -> Result<(), SkinError> {
        self.validate()?;
        for (vertex, values) in self.influences.iter().enumerate() {
            if values.len() > max_influences {
                return Err(SkinError::TooManyInfluences {
                    vertex: u32::try_from(vertex).unwrap_or(u32::MAX),
                    found: values.len(),
                    max: max_influences,
                });
            }
        }
        Ok(())
    }
}

/// A violated skinning-data contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkinError {
    /// The number of per-vertex lists does not match the declared vertex count.
    VertexCountMismatch {
        /// Declared vertex count.
        expected: u32,
        /// Number of authored influence lists.
        found: usize,
    },
    /// An influence references a joint outside the skeleton palette.
    JointIndexOutOfRange {
        /// Vertex carrying the influence.
        vertex: u32,
        /// Position in that vertex's influence list.
        influence: usize,
        /// Referenced joint index.
        joint_index: u32,
        /// Number of joints in the palette.
        joint_count: u32,
    },
    /// A weight is NaN or infinite.
    NonFiniteWeight {
        /// Vertex carrying the influence.
        vertex: u32,
        /// Position in that vertex's influence list.
        influence: usize,
    },
    /// Negative weights are not valid skinning influences.
    NegativeWeight {
        /// Vertex carrying the influence.
        vertex: u32,
        /// Position in that vertex's influence list.
        influence: usize,
    },
    /// A vertex lists the same joint more than once.
    DuplicateJoint {
        /// Vertex carrying the duplicate.
        vertex: u32,
        /// Repeated joint index.
        joint_index: u32,
    },
    /// Finite input weights overflowed while being summed.
    NonFiniteWeightSum {
        /// Vertex whose weight sum overflowed.
        vertex: u32,
    },
    /// A profile's fixed influence budget was exceeded.
    TooManyInfluences {
        /// Vertex exceeding the profile cap.
        vertex: u32,
        /// Number of authored influences.
        found: usize,
        /// Maximum allowed influences.
        max: usize,
    },
}

impl fmt::Display for SkinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VertexCountMismatch { expected, found } => {
                write!(f, "skinning data declares {expected} vertices but contains {found} lists")
            }
            Self::JointIndexOutOfRange {
                vertex,
                influence,
                joint_index,
                joint_count,
            } => write!(
                f,
                "vertex {vertex} influence {influence} references joint {joint_index}, outside {joint_count} joints"
            ),
            Self::NonFiniteWeight { vertex, influence } => {
                write!(f, "vertex {vertex} influence {influence} has a non-finite weight")
            }
            Self::NegativeWeight { vertex, influence } => {
                write!(f, "vertex {vertex} influence {influence} has a negative weight")
            }
            Self::DuplicateJoint {
                vertex,
                joint_index,
            } => write!(f, "vertex {vertex} repeats joint {joint_index}"),
            Self::NonFiniteWeightSum { vertex } => {
                write!(f, "vertex {vertex} has a non-finite weight sum")
            }
            Self::TooManyInfluences { vertex, found, max } => write!(
                f,
                "vertex {vertex} has {found} influences, exceeding the profile limit {max}"
            ),
        }
    }
}
