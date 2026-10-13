//! Deterministic runtime chunk planning.
//!
//! The planner is intentionally independent from the on-disk writer: it turns
//! a validated artifact/asset closure into a reproducible layout plan, while
//! [`crate::ContainerWriter`] remains responsible for encoding bytes. Keeping
//! planning separate makes the layout explainable and lets a future cook
//! pipeline feed compressed, encrypted, or GPU-page payloads without changing
//! dependency validation.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::fmt;

use prism_asset::StableGuid;

extern crate alloc;

/// One runtime payload considered by the chunk planner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkInput {
    /// Stable logical identity of the payload.
    pub guid: StableGuid,
    /// Uncompressed payload size in bytes.
    pub size: u64,
    /// Required start alignment within a chunk.
    pub alignment: u64,
    /// Direct hard dependencies. Dependencies must be placed before this item.
    pub dependencies: Vec<StableGuid>,
    /// Higher values are scheduled earlier among currently-ready items.
    pub priority: u32,
}

/// Layout policy for one deterministic planning pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkPolicy {
    /// Maximum raw bytes in a chunk, including alignment padding.
    pub max_chunk_size: u64,
    /// Alignment required for every chunk start.
    pub chunk_alignment: u64,
    /// Maximum number of output chunks.
    pub max_chunks: usize,
}

impl Default for ChunkPolicy {
    fn default() -> Self {
        Self {
            max_chunk_size: 64 * 1024 * 1024,
            chunk_alignment: 4096,
            max_chunks: usize::MAX,
        }
    }
}

/// A placed payload within a planned chunk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkPlacement {
    /// Payload identity.
    pub guid: StableGuid,
    /// Offset relative to the beginning of the chunk.
    pub offset: u64,
    /// Raw payload length.
    pub size: u64,
}

/// One planned chunk, in deterministic output order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedChunk {
    /// Zero-based stable chunk ordinal.
    pub id: usize,
    /// Absolute offset once chunks are concatenated.
    pub offset: u64,
    /// Required alignment of the chunk's absolute start. This is at least
    /// the policy alignment and may be higher when the first payload has a
    /// stricter absolute alignment requirement.
    pub start_alignment: u64,
    /// Extent including internal alignment padding.
    pub raw_size: u64,
    /// Highest priority among members, useful for boot scheduling.
    pub priority: u32,
    /// Payload placements in canonical planning order.
    pub placements: Vec<ChunkPlacement>,
}

/// Explainable result of a planning pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkPlan {
    /// Chunks in deterministic file order.
    pub chunks: Vec<PlannedChunk>,
    /// Direct lookup from payload identity to its chunk.
    pub assignments: BTreeMap<StableGuid, usize>,
}

/// Why planning failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChunkPlanError {
    /// The policy contains an invalid size or alignment.
    InvalidPolicy(&'static str),
    /// An input has an invalid size or alignment.
    InvalidInput {
        /// Input identity.
        guid: StableGuid,
        /// Validation reason.
        reason: &'static str,
    },
    /// Two inputs use the same stable identity.
    DuplicateGuid(StableGuid),
    /// A dependency is absent from the input closure.
    MissingDependency {
        /// Input whose closure is incomplete.
        asset: StableGuid,
        /// Missing hard dependency.
        dependency: StableGuid,
    },
    /// The hard dependency graph contains a cycle.
    DependencyCycle {
        /// Deterministically ordered members left in the cycle region.
        path: Vec<StableGuid>,
    },
    /// A payload cannot fit in one chunk under the selected policy.
    ItemTooLarge {
        /// Payload identity.
        guid: StableGuid,
        /// Payload size.
        size: u64,
        /// Policy maximum.
        max: u64,
    },
    /// The policy's chunk count limit would be exceeded.
    TooManyChunks {
        /// Maximum number of chunks.
        max: usize,
    },
    /// Arithmetic overflow while computing layout offsets.
    IntegerOverflow,
}

impl fmt::Display for ChunkPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicy(reason) => write!(f, "invalid chunk policy: {reason}"),
            Self::InvalidInput { guid, reason } => write!(f, "invalid input {guid:?}: {reason}"),
            Self::DuplicateGuid(guid) => write!(f, "duplicate chunk input {guid:?}"),
            Self::MissingDependency { asset, dependency } => {
                write!(f, "asset {asset:?} depends on missing {dependency:?}")
            }
            Self::DependencyCycle { path } => {
                write!(
                    f,
                    "chunk dependency cycle involving {} asset(s)",
                    path.len()
                )
            }
            Self::ItemTooLarge { guid, size, max } => {
                write!(f, "asset {guid:?} is {size} bytes, maximum chunk is {max}")
            }
            Self::TooManyChunks { max } => write!(f, "chunk count exceeds policy maximum {max}"),
            Self::IntegerOverflow => f.write_str("chunk layout arithmetic overflow"),
        }
    }
}

impl std::error::Error for ChunkPlanError {}

/// Plans a dependency-valid, aligned, bounded chunk layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkPlanner {
    policy: ChunkPolicy,
}

impl ChunkPlanner {
    /// Creates a planner. Invalid policies are reported by [`Self::plan`].
    #[must_use]
    pub const fn new(policy: ChunkPolicy) -> Self {
        Self { policy }
    }

    /// Returns the policy used by this planner.
    #[must_use]
    pub const fn policy(self) -> ChunkPolicy {
        self.policy
    }

    /// Validates the hard-dependency closure and produces a deterministic plan.
    pub fn plan(
        &self,
        inputs: impl IntoIterator<Item = ChunkInput>,
    ) -> Result<ChunkPlan, ChunkPlanError> {
        let policy = self.policy;
        if policy.max_chunk_size == 0 {
            return Err(ChunkPlanError::InvalidPolicy(
                "max_chunk_size must be non-zero",
            ));
        }
        if !policy.chunk_alignment.is_power_of_two() {
            return Err(ChunkPlanError::InvalidPolicy(
                "chunk_alignment must be a power of two",
            ));
        }
        if policy.max_chunks == 0 {
            return Err(ChunkPlanError::InvalidPolicy("max_chunks must be non-zero"));
        }

        let mut by_guid = BTreeMap::new();
        for input in inputs {
            if input.alignment == 0 || !input.alignment.is_power_of_two() {
                return Err(ChunkPlanError::InvalidInput {
                    guid: input.guid,
                    reason: "alignment must be a non-zero power of two",
                });
            }
            if input.alignment > policy.max_chunk_size {
                return Err(ChunkPlanError::InvalidInput {
                    guid: input.guid,
                    reason: "alignment exceeds max_chunk_size",
                });
            }
            if input.size > policy.max_chunk_size {
                return Err(ChunkPlanError::ItemTooLarge {
                    guid: input.guid,
                    size: input.size,
                    max: policy.max_chunk_size,
                });
            }
            let guid = input.guid;
            if by_guid.insert(guid, input).is_some() {
                return Err(ChunkPlanError::DuplicateGuid(guid));
            }
        }

        for input in by_guid.values() {
            for dependency in &input.dependencies {
                if !by_guid.contains_key(dependency) {
                    return Err(ChunkPlanError::MissingDependency {
                        asset: input.guid,
                        dependency: *dependency,
                    });
                }
            }
        }

        let order = topological_order(&by_guid)?;
        let mut chunks = Vec::<PlannedChunk>::new();
        let mut assignments = BTreeMap::new();
        for guid in order {
            let input = &by_guid[&guid];
            let alignment = policy.chunk_alignment.max(input.alignment);
            // First-fit is intentionally deterministic, but it must not be
            // allowed to place a dependent into an earlier chunk merely
            // because that chunk still has room. Topological processing only
            // orders the planning pass; physical chunk order is the order in
            // which the chunks are emitted. Therefore a payload may start in
            // the latest chunk containing any direct dependency (or later).
            // Dependencies in the same chunk are already before this payload
            // because `order` is topological.
            let first_allowed_chunk = input
                .dependencies
                .iter()
                .map(|dependency| assignments[dependency])
                .max()
                .unwrap_or(0);
            let mut placed = false;
            for chunk in chunks.iter_mut().skip(first_allowed_chunk) {
                let offset = align_up(chunk.raw_size, alignment)?;
                let end = offset
                    .checked_add(input.size)
                    .ok_or(ChunkPlanError::IntegerOverflow)?;
                if end <= policy.max_chunk_size {
                    chunk.raw_size = end;
                    chunk.priority = chunk.priority.max(input.priority);
                    chunk.placements.push(ChunkPlacement {
                        guid,
                        offset,
                        size: input.size,
                    });
                    assignments.insert(guid, chunk.id);
                    placed = true;
                    break;
                }
            }
            if !placed {
                if chunks.len() >= policy.max_chunks {
                    return Err(ChunkPlanError::TooManyChunks {
                        max: policy.max_chunks,
                    });
                }
                let id = chunks.len();
                let offset = 0;
                let start_alignment = alignment;
                let raw_size = input.size;
                chunks.push(PlannedChunk {
                    id,
                    offset: 0,
                    start_alignment,
                    raw_size,
                    priority: input.priority,
                    placements: vec![ChunkPlacement {
                        guid,
                        offset,
                        size: input.size,
                    }],
                });
                assignments.insert(guid, id);
            }
        }

        let mut absolute = 0;
        for chunk in &mut chunks {
            absolute = align_up(absolute, chunk.start_alignment)?;
            chunk.offset = absolute;
            absolute = absolute
                .checked_add(chunk.raw_size)
                .ok_or(ChunkPlanError::IntegerOverflow)?;
        }
        Ok(ChunkPlan {
            chunks,
            assignments,
        })
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, ChunkPlanError> {
    let mask = alignment
        .checked_sub(1)
        .ok_or(ChunkPlanError::IntegerOverflow)?;
    value
        .checked_add(mask)
        .map(|v| v & !mask)
        .ok_or(ChunkPlanError::IntegerOverflow)
}

fn topological_order(
    inputs: &BTreeMap<StableGuid, ChunkInput>,
) -> Result<Vec<StableGuid>, ChunkPlanError> {
    // Kahn's algorithm keeps the graph iterative and lets the ready queue
    // honor the runtime priority contract. Ties remain stable by GUID.
    let mut indegree: BTreeMap<StableGuid, usize> = inputs
        .iter()
        .map(|(&guid, input)| (guid, input.dependencies.len()))
        .collect();
    let mut dependents: BTreeMap<StableGuid, Vec<StableGuid>> = BTreeMap::new();
    for (&guid, input) in inputs {
        for &dependency in &input.dependencies {
            dependents.entry(dependency).or_default().push(guid);
        }
    }

    let mut ready = BTreeSet::<(Reverse<u32>, StableGuid)>::new();
    for (&guid, &count) in &indegree {
        if count == 0 {
            ready.insert((Reverse(inputs[&guid].priority), guid));
        }
    }
    let mut output = Vec::with_capacity(inputs.len());
    while let Some((Reverse(_priority), guid)) = ready.pop_first() {
        output.push(guid);
        if let Some(children) = dependents.get(&guid) {
            for &child in children {
                let count = indegree
                    .get_mut(&child)
                    .expect("dependency graph contains every child");
                *count -= 1;
                if *count == 0 {
                    ready.insert((Reverse(inputs[&child].priority), child));
                }
            }
        }
    }
    if output.len() != inputs.len() {
        let path = indegree
            .into_iter()
            .filter_map(|(guid, count)| (count != 0).then_some(guid))
            .collect();
        return Err(ChunkPlanError::DependencyCycle { path });
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guid(value: u128) -> StableGuid {
        StableGuid::from_u128(value)
    }

    #[test]
    fn places_dependencies_before_dependents_deterministically() {
        let plan = ChunkPlanner::new(ChunkPolicy {
            max_chunk_size: 16,
            chunk_alignment: 4,
            max_chunks: 4,
        })
        .plan([
            ChunkInput {
                guid: guid(2),
                size: 4,
                alignment: 4,
                dependencies: vec![guid(1)],
                priority: 100,
            },
            ChunkInput {
                guid: guid(1),
                size: 4,
                alignment: 4,
                dependencies: vec![],
                priority: 0,
            },
        ])
        .unwrap();
        assert_eq!(plan.chunks.len(), 1);
        assert_eq!(plan.chunks[0].placements[0].guid, guid(1));
        assert_eq!(plan.chunks[0].placements[1].guid, guid(2));
        assert_eq!(plan.assignments[&guid(1)], 0);
        assert_eq!(plan.assignments[&guid(2)], 0);
    }

    #[test]
    fn prioritizes_ready_roots_without_violating_dependencies() {
        let plan = ChunkPlanner::new(ChunkPolicy {
            max_chunk_size: 4,
            chunk_alignment: 1,
            max_chunks: 4,
        })
        .plan([
            ChunkInput {
                guid: guid(1),
                size: 1,
                alignment: 1,
                dependencies: vec![],
                priority: 1,
            },
            ChunkInput {
                guid: guid(2),
                size: 1,
                alignment: 1,
                dependencies: vec![],
                priority: 9,
            },
        ])
        .unwrap();
        assert_eq!(
            plan.chunks[0]
                .placements
                .iter()
                .map(|placement| placement.guid)
                .collect::<Vec<_>>(),
            vec![guid(2), guid(1)]
        );
    }

    #[test]
    fn splits_on_capacity_and_aligns_chunk_starts() {
        let plan = ChunkPlanner::new(ChunkPolicy {
            max_chunk_size: 8,
            chunk_alignment: 8,
            max_chunks: 4,
        })
        .plan([
            ChunkInput {
                guid: guid(1),
                size: 5,
                alignment: 1,
                dependencies: vec![],
                priority: 0,
            },
            ChunkInput {
                guid: guid(2),
                size: 4,
                alignment: 4,
                dependencies: vec![],
                priority: 0,
            },
        ])
        .unwrap();
        assert_eq!(plan.chunks.len(), 2);
        assert_eq!(plan.chunks[1].offset % 8, 0);
    }

    #[test]
    fn rejects_missing_dependency_and_cycles() {
        let planner = ChunkPlanner::new(ChunkPolicy {
            max_chunk_size: 8,
            chunk_alignment: 4,
            max_chunks: 4,
        });
        assert!(matches!(
            planner.plan([ChunkInput {
                guid: guid(1),
                size: 1,
                alignment: 1,
                dependencies: vec![guid(2)],
                priority: 0
            }]),
            Err(ChunkPlanError::MissingDependency { .. })
        ));
        assert!(matches!(
            planner.plan([
                ChunkInput {
                    guid: guid(1),
                    size: 1,
                    alignment: 1,
                    dependencies: vec![guid(2)],
                    priority: 0
                },
                ChunkInput {
                    guid: guid(2),
                    size: 1,
                    alignment: 1,
                    dependencies: vec![guid(1)],
                    priority: 0
                },
            ]),
            Err(ChunkPlanError::DependencyCycle { .. })
        ));
    }
}
