//! Deterministic, bounded Spatial/Tactical query contracts.
//!
//! The evaluator is intentionally host-neutral. It uses integer coordinates,
//! caller-supplied fixed samples, and an explicit candidate capacity so the
//! same input produces the same result on every supported target.

extern crate alloc;

use alloc::vec::Vec;
use core::cmp::Ordering;

/// Maximum number of candidates retained by one evaluator.
pub const MAX_QUERY_CANDIDATES: usize = 64;
/// Maximum number of results returned by one query.
pub const MAX_QUERY_RESULTS: usize = 16;
/// Maximum number of fixed samples in one query.
pub const MAX_QUERY_SAMPLES: usize = 8;
/// Scale used for non-distance score terms.
pub const QUERY_SCORE_SCALE: i64 = 1_000;

/// The query domain recorded in a result and replay trace.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum QueryKind {
    /// A query concerned primarily with spatial proximity.
    Spatial,
    /// A query concerned with spatial proximity plus caller-defined utility.
    Tactical,
}

/// A signed, integer coordinate vector.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FixedVec3 {
    /// X coordinate in the caller's fixed unit.
    pub x: i32,
    /// Y coordinate in the caller's fixed unit.
    pub y: i32,
    /// Z coordinate in the caller's fixed unit.
    pub z: i32,
}

impl FixedVec3 {
    /// Creates a coordinate vector without floating-point conversion.
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

/// One fixed sample offset and its relative importance.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct QuerySample {
    /// Offset from the query origin.
    pub offset: FixedVec3,
    /// Positive integer sample weight.
    pub weight: u32,
}

impl QuerySample {
    /// Creates a weighted fixed sample.
    pub const fn new(offset: FixedVec3, weight: u32) -> Self {
        Self { offset, weight }
    }
}

/// A validated, bounded fixed-sample set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampleSet {
    samples: Vec<QuerySample>,
}

impl SampleSet {
    /// Validates and copies a fixed sample set.
    pub fn new(samples: &[QuerySample]) -> Result<Self, QueryError> {
        if samples.is_empty() {
            return Err(QueryError::EmptySampleSet);
        }
        if samples.len() > MAX_QUERY_SAMPLES {
            return Err(QueryError::SampleCapacityExceeded {
                requested: samples.len(),
                capacity: MAX_QUERY_SAMPLES,
            });
        }
        if let Some((index, _)) = samples
            .iter()
            .enumerate()
            .find(|(_, sample)| sample.weight == 0)
        {
            return Err(QueryError::ZeroSampleWeight { index });
        }
        Ok(Self {
            samples: samples.to_vec(),
        })
    }

    /// Returns the samples in their stable declaration order.
    pub fn samples(&self) -> &[QuerySample] {
        &self.samples
    }
}

/// Integer weights used by the deterministic score function.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct QueryWeights {
    /// Weight for the best fixed-sample proximity term.
    pub distance: i32,
    /// Weight for the candidate's generic priority term.
    pub priority: i32,
    /// Weight for the candidate's generic utility term.
    pub utility: i32,
}

impl QueryWeights {
    /// Creates integer score weights.
    pub const fn new(distance: i32, priority: i32, utility: i32) -> Self {
        Self {
            distance,
            priority,
            utility,
        }
    }
}

/// A host-neutral candidate accepted by both Spatial and Tactical queries.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct QueryCandidate {
    /// Stable host-owned identifier. IDs must be unique within an evaluator.
    pub id: u32,
    /// Candidate position in the same fixed unit as the query origin.
    pub position: FixedVec3,
    /// Generic deterministic ranking input.
    pub priority: i32,
    /// Generic caller-defined utility input.
    pub utility: i32,
}

impl QueryCandidate {
    /// Creates a candidate with explicit deterministic ranking inputs.
    pub const fn new(id: u32, position: FixedVec3, priority: i32, utility: i32) -> Self {
        Self {
            id,
            position,
            priority,
            utility,
        }
    }
}

/// A validated query request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueryRequest {
    kind: QueryKind,
    origin: FixedVec3,
    samples: SampleSet,
    max_distance_squared: u64,
    weights: QueryWeights,
    max_results: usize,
}

impl QueryRequest {
    /// Creates a bounded Spatial or Tactical request.
    pub fn new(
        kind: QueryKind,
        origin: FixedVec3,
        samples: SampleSet,
        max_distance_squared: u64,
        weights: QueryWeights,
        max_results: usize,
    ) -> Result<Self, QueryError> {
        if max_results == 0 || max_results > MAX_QUERY_RESULTS {
            return Err(QueryError::InvalidResultLimit {
                requested: max_results,
                capacity: MAX_QUERY_RESULTS,
            });
        }
        Ok(Self {
            kind,
            origin,
            samples,
            max_distance_squared,
            weights,
            max_results,
        })
    }

    /// Returns the query domain.
    pub const fn kind(&self) -> QueryKind {
        self.kind
    }
}

/// One scored query result.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ScoredCandidate {
    /// Candidate identifier.
    pub candidate_id: u32,
    /// Deterministic integer score; larger scores rank first.
    pub score: i64,
    /// Declaration index of the fixed sample that produced the best distance.
    pub best_sample: u8,
}

/// The ordered output of one query evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueryResult {
    kind: QueryKind,
    candidates: Vec<ScoredCandidate>,
}

impl QueryResult {
    /// Returns the query domain recorded in this result.
    pub const fn kind(&self) -> QueryKind {
        self.kind
    }

    /// Returns candidates in deterministic rank order.
    pub fn candidates(&self) -> &[ScoredCandidate] {
        &self.candidates
    }
}

/// Errors produced while constructing or evaluating a bounded query.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum QueryError {
    /// The sample set contains no samples.
    EmptySampleSet,
    /// The sample set exceeds the fixed contract capacity.
    SampleCapacityExceeded {
        /// Number of supplied samples.
        requested: usize,
        /// Maximum accepted sample count.
        capacity: usize,
    },
    /// A sample has no positive contribution.
    ZeroSampleWeight {
        /// Declaration index of the invalid sample.
        index: usize,
    },
    /// The result limit is zero or exceeds the fixed contract capacity.
    InvalidResultLimit {
        /// Requested result count.
        requested: usize,
        /// Maximum accepted result count.
        capacity: usize,
    },
    /// The evaluator capacity is zero or exceeds the fixed contract capacity.
    InvalidCandidateCapacity {
        /// Requested candidate count.
        requested: usize,
        /// Maximum accepted candidate count.
        capacity: usize,
    },
    /// The evaluator already contains its maximum number of candidates.
    CandidateCapacityExceeded {
        /// Configured evaluator capacity.
        capacity: usize,
    },
    /// Candidate IDs must be unique for stable tie-breaking.
    DuplicateCandidateId {
        /// The repeated candidate ID.
        id: u32,
    },
    /// A checked integer operation would overflow the contract's score range.
    ArithmeticOverflow,
}

/// A bounded, read-only evaluator for Spatial and Tactical queries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueryEvaluator {
    capacity: usize,
    candidates: Vec<QueryCandidate>,
}

impl QueryEvaluator {
    /// Creates an evaluator with an explicit finite candidate capacity.
    pub fn new(capacity: usize) -> Result<Self, QueryError> {
        if capacity == 0 || capacity > MAX_QUERY_CANDIDATES {
            return Err(QueryError::InvalidCandidateCapacity {
                requested: capacity,
                capacity: MAX_QUERY_CANDIDATES,
            });
        }
        Ok(Self {
            capacity,
            candidates: Vec::with_capacity(capacity),
        })
    }

    /// Adds one candidate while preserving the unique-ID invariant.
    pub fn add_candidate(&mut self, candidate: QueryCandidate) -> Result<(), QueryError> {
        if self
            .candidates
            .iter()
            .any(|stored| stored.id == candidate.id)
        {
            return Err(QueryError::DuplicateCandidateId { id: candidate.id });
        }
        if self.candidates.len() == self.capacity {
            return Err(QueryError::CandidateCapacityExceeded {
                capacity: self.capacity,
            });
        }
        self.candidates.push(candidate);
        Ok(())
    }

    /// Returns candidates in insertion order; evaluation never depends on it.
    pub fn candidates(&self) -> &[QueryCandidate] {
        &self.candidates
    }

    /// Evaluates a request using fixed samples and a stable total ordering.
    pub fn evaluate(&self, request: &QueryRequest) -> Result<QueryResult, QueryError> {
        let mut scored = Vec::with_capacity(self.candidates.len());
        for candidate in &self.candidates {
            if let Some(result) = score_candidate(*candidate, request)? {
                scored.push(result);
            }
        }
        scored.sort_by(rank_order);
        scored.truncate(request.max_results);
        Ok(QueryResult {
            kind: request.kind,
            candidates: scored,
        })
    }
}

fn score_candidate(
    candidate: QueryCandidate,
    request: &QueryRequest,
) -> Result<Option<ScoredCandidate>, QueryError> {
    let mut best: Option<(u128, usize)> = None;
    for (index, sample) in request.samples.samples.iter().enumerate() {
        let distance = squared_distance(candidate.position, request.origin, sample.offset)?;
        if distance > u128::from(request.max_distance_squared) {
            continue;
        }
        let proximity = u128::from(request.max_distance_squared) - distance;
        let weighted = proximity
            .checked_mul(u128::from(sample.weight))
            .ok_or(QueryError::ArithmeticOverflow)?;
        if best.is_none_or(|(current, current_index)| {
            weighted > current || (weighted == current && index < current_index)
        }) {
            best = Some((weighted, index));
        }
    }

    let Some((proximity, best_sample)) = best else {
        return Ok(None);
    };
    let distance_term = i128::try_from(proximity)
        .map_err(|_| QueryError::ArithmeticOverflow)?
        .checked_mul(i128::from(request.weights.distance))
        .ok_or(QueryError::ArithmeticOverflow)?;
    let priority_term = i128::from(candidate.priority)
        .checked_mul(i128::from(request.weights.priority))
        .and_then(|value| value.checked_mul(i128::from(QUERY_SCORE_SCALE)))
        .ok_or(QueryError::ArithmeticOverflow)?;
    let utility_term = i128::from(candidate.utility)
        .checked_mul(i128::from(request.weights.utility))
        .and_then(|value| value.checked_mul(i128::from(QUERY_SCORE_SCALE)))
        .ok_or(QueryError::ArithmeticOverflow)?;
    let score = distance_term
        .checked_add(priority_term)
        .and_then(|value| value.checked_add(utility_term))
        .and_then(|value| i64::try_from(value).ok())
        .ok_or(QueryError::ArithmeticOverflow)?;

    Ok(Some(ScoredCandidate {
        candidate_id: candidate.id,
        score,
        best_sample: u8::try_from(best_sample).map_err(|_| QueryError::ArithmeticOverflow)?,
    }))
}

fn squared_distance(
    candidate: FixedVec3,
    origin: FixedVec3,
    sample_offset: FixedVec3,
) -> Result<u128, QueryError> {
    let dx = i128::from(candidate.x) - i128::from(origin.x) - i128::from(sample_offset.x);
    let dy = i128::from(candidate.y) - i128::from(origin.y) - i128::from(sample_offset.y);
    let dz = i128::from(candidate.z) - i128::from(origin.z) - i128::from(sample_offset.z);
    let sum = dx
        .checked_mul(dx)
        .and_then(|value| value.checked_add(dy.checked_mul(dy)?))
        .and_then(|value| value.checked_add(dz.checked_mul(dz)?))
        .ok_or(QueryError::ArithmeticOverflow)?;
    u128::try_from(sum).map_err(|_| QueryError::ArithmeticOverflow)
}

fn rank_order(left: &ScoredCandidate, right: &ScoredCandidate) -> Ordering {
    right
        .score
        .cmp(&left.score)
        .then_with(|| left.candidate_id.cmp(&right.candidate_id))
        .then_with(|| left.best_sample.cmp(&right.best_sample))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(kind: QueryKind) -> QueryRequest {
        let samples = SampleSet::new(&[
            QuerySample::new(FixedVec3::new(0, 0, 0), 2),
            QuerySample::new(FixedVec3::new(4, 0, 0), 1),
        ])
        .unwrap();
        QueryRequest::new(
            kind,
            FixedVec3::new(0, 0, 0),
            samples,
            100,
            QueryWeights::new(1, 3, 2),
            3,
        )
        .unwrap()
    }

    #[test]
    fn golden_spatial_result_is_order_and_input_order_independent() {
        let mut evaluator = QueryEvaluator::new(4).unwrap();
        for candidate in [
            QueryCandidate::new(42, FixedVec3::new(5, 0, 0), 0, 0),
            QueryCandidate::new(7, FixedVec3::new(0, 0, 0), 0, 0),
            QueryCandidate::new(11, FixedVec3::new(4, 0, 0), 0, 0),
        ] {
            evaluator.add_candidate(candidate).unwrap();
        }
        let result = evaluator.evaluate(&request(QueryKind::Spatial)).unwrap();
        assert_eq!(
            result.candidates(),
            &[
                ScoredCandidate {
                    candidate_id: 7,
                    score: 200,
                    best_sample: 0,
                },
                ScoredCandidate {
                    candidate_id: 11,
                    score: 168,
                    best_sample: 0,
                },
                ScoredCandidate {
                    candidate_id: 42,
                    score: 150,
                    best_sample: 0,
                },
            ]
        );

        let mut reversed = QueryEvaluator::new(4).unwrap();
        for candidate in evaluator.candidates().iter().rev().copied() {
            reversed.add_candidate(candidate).unwrap();
        }
        assert_eq!(
            reversed.evaluate(&request(QueryKind::Spatial)).unwrap(),
            result
        );
    }

    #[test]
    fn golden_tactical_result_uses_utility_and_stable_tie_break() {
        let mut evaluator = QueryEvaluator::new(3).unwrap();
        evaluator
            .add_candidate(QueryCandidate::new(20, FixedVec3::new(0, 0, 0), 0, 4))
            .unwrap();
        evaluator
            .add_candidate(QueryCandidate::new(10, FixedVec3::new(0, 0, 0), 0, 4))
            .unwrap();
        let result = evaluator.evaluate(&request(QueryKind::Tactical)).unwrap();
        assert_eq!(result.kind(), QueryKind::Tactical);
        assert_eq!(result.candidates()[0].candidate_id, 10);
        assert_eq!(result.candidates()[1].candidate_id, 20);
        assert_eq!(result.candidates()[0].score, 8_200);
    }

    #[test]
    fn golden_errors_are_bounded_and_stable() {
        assert_eq!(SampleSet::new(&[]), Err(QueryError::EmptySampleSet));
        assert_eq!(
            SampleSet::new(&[QuerySample::new(FixedVec3::new(0, 0, 0), 0)]),
            Err(QueryError::ZeroSampleWeight { index: 0 })
        );
        assert_eq!(
            QueryEvaluator::new(MAX_QUERY_CANDIDATES + 1),
            Err(QueryError::InvalidCandidateCapacity {
                requested: MAX_QUERY_CANDIDATES + 1,
                capacity: MAX_QUERY_CANDIDATES,
            })
        );

        let mut evaluator = QueryEvaluator::new(1).unwrap();
        evaluator
            .add_candidate(QueryCandidate::new(1, FixedVec3::new(0, 0, 0), 0, 0))
            .unwrap();
        assert_eq!(
            evaluator.add_candidate(QueryCandidate::new(1, FixedVec3::new(1, 0, 0), 0, 0)),
            Err(QueryError::DuplicateCandidateId { id: 1 })
        );
        assert_eq!(
            evaluator.add_candidate(QueryCandidate::new(2, FixedVec3::new(1, 0, 0), 0, 0)),
            Err(QueryError::CandidateCapacityExceeded { capacity: 1 })
        );
    }

    #[test]
    fn no_candidate_in_range_produces_empty_result() {
        let evaluator = QueryEvaluator::new(1).unwrap();
        let mut far = QueryEvaluator::new(1).unwrap();
        far.add_candidate(QueryCandidate::new(9, FixedVec3::new(1_000, 0, 0), 0, 0))
            .unwrap();
        assert_eq!(evaluator.candidates(), &[]);
        assert_eq!(
            far.evaluate(&request(QueryKind::Spatial))
                .unwrap()
                .candidates(),
            &[]
        );
    }
}
