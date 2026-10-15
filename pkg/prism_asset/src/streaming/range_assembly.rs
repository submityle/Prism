//! Bounded range-backed byte assembly.
//!
//! A container manifest or a streaming decoder can describe a payload as a
//! finite list of physical ranges. This module reads those ranges without
//! falling back to a whole-file read, preserves declaration order, verifies
//! every returned length, and bounds both total bytes and concurrent reads.
//! The async executor uses a separate caller-owned I/O pool; cancellation is
//! cooperative at range boundaries because a blocking [`AssetReader`] call
//! cannot be interrupted safely by the asset layer.

use crate::{AssetPath, AssetSources, ReadError};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::{Context, Poll};
use prism_tasks::{Task, TaskPool};

/// One physical range contributing bytes to an assembled payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RangePart {
    offset: u64,
    len: u64,
}

impl RangePart {
    /// Creates a range. Bounds are checked when the range is placed in a plan.
    #[must_use]
    pub const fn new(offset: u64, len: u64) -> Self {
        Self { offset, len }
    }

    /// Start offset in the source asset.
    #[must_use]
    pub const fn offset(self) -> u64 {
        self.offset
    }

    /// Number of bytes to read.
    #[must_use]
    pub const fn len(self) -> u64 {
        self.len
    }
}

/// Bounds applied before any source I/O is started.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeAssemblyLimits {
    /// Maximum number of physical ranges in one plan.
    pub max_parts: usize,
    /// Maximum sum of requested bytes.
    pub max_bytes: u64,
    /// Maximum reads issued concurrently by the async executor.
    pub max_parallel: usize,
}

impl Default for RangeAssemblyLimits {
    fn default() -> Self {
        Self {
            max_parts: 256,
            max_bytes: 64 * 1024 * 1024,
            max_parallel: 8,
        }
    }
}

/// A validated, ordered range assembly plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeAssemblyPlan {
    path: AssetPath,
    source_len: u64,
    parts: Vec<RangePart>,
    total_bytes: u64,
    limits: RangeAssemblyLimits,
}

impl RangeAssemblyPlan {
    /// Validates an ordered range list against the source length and limits.
    pub fn new(
        path: impl Into<AssetPath>,
        source_len: u64,
        parts: impl IntoIterator<Item = RangePart>,
        limits: RangeAssemblyLimits,
    ) -> Result<Self, RangeAssemblyPlanError> {
        if limits.max_parts == 0 {
            return Err(RangeAssemblyPlanError::ZeroPartLimit);
        }
        if limits.max_parallel == 0 {
            return Err(RangeAssemblyPlanError::ZeroParallelLimit);
        }
        let parts: Vec<_> = parts.into_iter().collect();
        if parts.len() > limits.max_parts {
            return Err(RangeAssemblyPlanError::TooManyParts {
                actual: parts.len(),
                limit: limits.max_parts,
            });
        }
        let mut total_bytes = 0u64;
        for part in &parts {
            let end =
                part.offset
                    .checked_add(part.len)
                    .ok_or(RangeAssemblyPlanError::RangeOverflow {
                        offset: part.offset,
                        len: part.len,
                    })?;
            if end > source_len {
                return Err(RangeAssemblyPlanError::OutOfBounds {
                    offset: part.offset,
                    len: part.len,
                    source_len,
                });
            }
            total_bytes = total_bytes
                .checked_add(part.len)
                .ok_or(RangeAssemblyPlanError::ByteBudgetOverflow)?;
            if total_bytes > limits.max_bytes {
                return Err(RangeAssemblyPlanError::ByteBudgetExceeded {
                    requested: total_bytes,
                    limit: limits.max_bytes,
                });
            }
        }
        Ok(Self {
            path: path.into().without_label(),
            source_len,
            parts,
            total_bytes,
            limits,
        })
    }

    /// Creates a single-range plan for a complete source payload.
    pub fn whole(
        path: impl Into<AssetPath>,
        source_len: u64,
        limits: RangeAssemblyLimits,
    ) -> Result<Self, RangeAssemblyPlanError> {
        Self::new(path, source_len, [RangePart::new(0, source_len)], limits)
    }

    /// Source path without a label.
    #[must_use]
    pub fn path(&self) -> &AssetPath {
        &self.path
    }

    /// Declared source length used for validation.
    #[must_use]
    pub const fn source_len(&self) -> u64 {
        self.source_len
    }

    /// Ranges in the exact order used for the assembled output.
    #[must_use]
    pub fn parts(&self) -> &[RangePart] {
        &self.parts
    }

    /// Total bytes returned by the assembly.
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Limits captured by this plan.
    #[must_use]
    pub const fn limits(&self) -> RangeAssemblyLimits {
        self.limits
    }
}

/// Why a range assembly plan was rejected before I/O.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeAssemblyPlanError {
    ZeroPartLimit,
    ZeroParallelLimit,
    TooManyParts {
        actual: usize,
        limit: usize,
    },
    RangeOverflow {
        offset: u64,
        len: u64,
    },
    OutOfBounds {
        offset: u64,
        len: u64,
        source_len: u64,
    },
    ByteBudgetOverflow,
    ByteBudgetExceeded {
        requested: u64,
        limit: u64,
    },
}

impl fmt::Display for RangeAssemblyPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPartLimit => f.write_str("range assembly max_parts must be non-zero"),
            Self::ZeroParallelLimit => f.write_str("range assembly max_parallel must be non-zero"),
            Self::TooManyParts { actual, limit } => {
                write!(f, "range assembly has {actual} parts, limit is {limit}")
            }
            Self::RangeOverflow { offset, len } => {
                write!(f, "range {offset}+{len} overflows u64")
            }
            Self::OutOfBounds {
                offset,
                len,
                source_len,
            } => write!(
                f,
                "range {offset}..{} exceeds source length {source_len}",
                offset + len
            ),
            Self::ByteBudgetOverflow => f.write_str("range assembly byte budget overflows u64"),
            Self::ByteBudgetExceeded { requested, limit } => {
                write!(
                    f,
                    "range assembly requests {requested} bytes, limit is {limit}"
                )
            }
        }
    }
}

impl std::error::Error for RangeAssemblyPlanError {}

/// Why an otherwise valid range assembly failed during I/O.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RangeAssemblyError {
    Read {
        part: usize,
        error: ReadError,
    },
    ShortRead {
        part: usize,
        expected: u64,
        actual: u64,
    },
    SourcePanicked {
        part: usize,
    },
    Cancelled,
}

impl fmt::Display for RangeAssemblyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { part, error } => write!(f, "range part {part} failed: {error}"),
            Self::ShortRead {
                part,
                expected,
                actual,
            } => write!(
                f,
                "range part {part} short read: expected {expected}, got {actual}"
            ),
            Self::SourcePanicked { part } => write!(f, "range part {part} source panicked"),
            Self::Cancelled => f.write_str("range assembly cancelled"),
        }
    }
}

impl std::error::Error for RangeAssemblyError {}

/// A cooperative cancellation token for range assembly.
#[derive(Clone, Default)]
pub struct RangeAssemblyCancellation(Arc<AtomicBool>);

impl RangeAssemblyCancellation {
    /// Creates a live token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation. In-flight blocking reads finish, but their
    /// result is discarded and no assembled payload is published.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Reads every requested range sequentially and concatenates the bytes in plan
/// order. This is the deterministic baseline used by tests and small payloads.
pub fn assemble_ranges(
    sources: &AssetSources,
    plan: &RangeAssemblyPlan,
) -> Result<Vec<u8>, RangeAssemblyError> {
    let mut output = Vec::with_capacity(plan.total_bytes as usize);
    for (index, part) in plan.parts.iter().copied().enumerate() {
        let bytes = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sources.read_range(plan.path(), part.offset, part.len)
        }))
        .map_err(|_| RangeAssemblyError::SourcePanicked { part: index })?
        .map_err(|error| RangeAssemblyError::Read { part: index, error })?;
        if bytes.len() as u64 != part.len {
            return Err(RangeAssemblyError::ShortRead {
                part: index,
                expected: part.len,
                actual: bytes.len() as u64,
            });
        }
        output.extend_from_slice(&bytes);
    }
    Ok(output)
}

/// A cancellable asynchronous range assembly operation.
pub struct RangeAssemblyTicket {
    task: Task<Result<Vec<u8>, RangeAssemblyError>>,
    cancellation: RangeAssemblyCancellation,
}

impl RangeAssemblyTicket {
    /// Requests cooperative cancellation of this operation.
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    /// Whether this operation has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
}

impl Future for RangeAssemblyTicket {
    type Output = Result<Vec<u8>, RangeAssemblyError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.cancellation.is_cancelled() {
            return Poll::Ready(Err(RangeAssemblyError::Cancelled));
        }
        Pin::new(&mut self.task).poll(cx)
    }
}

/// Starts bounded parallel range reads. The caller should use an I/O pool
/// distinct from the CPU pool that awaits the returned ticket.
pub fn assemble_ranges_async(
    sources: AssetSources,
    plan: RangeAssemblyPlan,
    pool: TaskPool,
) -> RangeAssemblyTicket {
    let cancellation = RangeAssemblyCancellation::new();
    let task_cancellation = cancellation.clone();
    let child_pool = pool.clone();
    let task = pool.spawn_async(async move {
        let mut output = Vec::with_capacity(plan.total_bytes as usize);
        let parallel = plan.limits.max_parallel;
        for (chunk_index, chunk) in plan.parts.chunks(parallel).enumerate() {
            if task_cancellation.is_cancelled() {
                return Err(RangeAssemblyError::Cancelled);
            }
            let mut reads = Vec::with_capacity(chunk.len());
            for (relative, part) in chunk.iter().copied().enumerate() {
                let sources = sources.clone();
                let path = plan.path.clone();
                let cancellation = task_cancellation.clone();
                let index = chunk_index * parallel;
                reads.push(child_pool.spawn_async(async move {
                    if cancellation.is_cancelled() {
                        return Err(RangeAssemblyError::Cancelled);
                    }
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        sources.read_range(&path, part.offset, part.len)
                    }));
                    let bytes = match result {
                        Ok(Ok(bytes)) => bytes,
                        Ok(Err(error)) => {
                            return Err(RangeAssemblyError::Read {
                                part: index + relative,
                                error,
                            })
                        }
                        Err(_) => {
                            return Err(RangeAssemblyError::SourcePanicked {
                                part: index + relative,
                            })
                        }
                    };
                    if bytes.len() as u64 != part.len {
                        return Err(RangeAssemblyError::ShortRead {
                            part: index + relative,
                            expected: part.len,
                            actual: bytes.len() as u64,
                        });
                    }
                    Ok(bytes)
                }));
            }
            for read in reads {
                let bytes = read.await?;
                if task_cancellation.is_cancelled() {
                    return Err(RangeAssemblyError::Cancelled);
                }
                output.extend_from_slice(&bytes);
            }
        }
        Ok(output)
    });
    RangeAssemblyTicket { task, cancellation }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssetReader, MemSource};
    use alloc::sync::Arc;
    use prism_tasks::TaskPool;

    fn source() -> AssetSources {
        let mut source = MemSource::new();
        source.insert("payload.bin", Arc::<[u8]>::from(&b"0123456789"[..]));
        AssetSources::from_reader("memory", source)
    }

    fn limits() -> RangeAssemblyLimits {
        RangeAssemblyLimits {
            max_parts: 8,
            max_bytes: 32,
            max_parallel: 2,
        }
    }

    #[test]
    fn assembles_disjoint_ranges_in_declared_order() {
        let sources = source();
        let plan = RangeAssemblyPlan::new(
            "payload.bin",
            10,
            [RangePart::new(6, 2), RangePart::new(0, 3)],
            limits(),
        )
        .unwrap();
        assert_eq!(assemble_ranges(&sources, &plan).unwrap(), b"67012");
    }

    #[test]
    fn rejects_overflow_and_budget_before_io() {
        assert!(matches!(
            RangeAssemblyPlan::new("payload.bin", 10, [RangePart::new(u64::MAX, 1)], limits(),),
            Err(RangeAssemblyPlanError::RangeOverflow { .. })
        ));
        assert!(matches!(
            RangeAssemblyPlan::new(
                "payload.bin",
                10,
                [
                    RangePart::new(0, 10),
                    RangePart::new(0, 10),
                    RangePart::new(0, 10),
                    RangePart::new(0, 10)
                ],
                RangeAssemblyLimits {
                    max_bytes: 16,
                    ..limits()
                },
            ),
            Err(RangeAssemblyPlanError::ByteBudgetExceeded { .. })
        ));
    }

    #[test]
    fn async_executor_matches_sequential_assembly() {
        let sources = source();
        let plan = RangeAssemblyPlan::new(
            "payload.bin",
            10,
            [RangePart::new(1, 3), RangePart::new(7, 3)],
            limits(),
        )
        .unwrap();
        let expected = assemble_ranges(&sources, &plan).unwrap();
        let pool = TaskPool::with_threads(2);
        let actual = pool
            .block_on(assemble_ranges_async(sources, plan, pool.clone()))
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn cancellation_discards_assembled_result() {
        let sources = source();
        let plan = RangeAssemblyPlan::new(
            "payload.bin",
            10,
            [RangePart::new(0, 4), RangePart::new(4, 4)],
            limits(),
        )
        .unwrap();
        let pool = TaskPool::with_threads(2);
        let ticket = assemble_ranges_async(sources, plan, pool.clone());
        ticket.cancel();
        assert_eq!(pool.block_on(ticket), Err(RangeAssemblyError::Cancelled));
    }
}
