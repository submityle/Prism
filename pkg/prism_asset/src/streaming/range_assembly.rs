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

    /// Whether this range contributes no bytes.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
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
        let parts = collect_range_parts(parts, limits.max_parts)?;
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
            if total_bytes > isize::MAX as u64 {
                return Err(RangeAssemblyPlanError::HostAllocationOverflow { bytes: total_bytes });
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

/// Stops after the first excess item, including for infinite iterators. Never
/// trusts the iterator's size hint as an allocation request.
pub(crate) fn collect_range_parts(
    parts: impl IntoIterator<Item = RangePart>,
    limit: usize,
) -> Result<Vec<RangePart>, RangeAssemblyPlanError> {
    let mut result = Vec::new();
    for part in parts {
        if result.len() == limit {
            return Err(RangeAssemblyPlanError::TooManyParts {
                actual: limit.saturating_add(1),
                limit,
            });
        }
        result.push(part);
    }
    Ok(result)
}

/// Why a range assembly plan was rejected before I/O.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeAssemblyPlanError {
    /// The plan allowed no physical ranges.
    ZeroPartLimit,
    /// The async plan allowed no concurrent reads.
    ZeroParallelLimit,
    /// The number of physical ranges exceeded the configured limit.
    TooManyParts {
        /// Number observed before rejecting; iteration stops at the first excess item.
        actual: usize,
        /// Maximum allowed ranges.
        limit: usize,
    },
    /// A range end overflowed `u64`.
    RangeOverflow {
        /// Range start offset.
        offset: u64,
        /// Range length.
        len: u64,
    },
    /// A range extended beyond the declared source length.
    OutOfBounds {
        /// Range start offset.
        offset: u64,
        /// Range length.
        len: u64,
        /// Declared source length.
        source_len: u64,
    },
    /// The total byte count overflowed `u64`.
    ByteBudgetOverflow,
    /// The total byte count exceeded the configured limit.
    ByteBudgetExceeded {
        /// Total requested bytes.
        requested: u64,
        /// Maximum allowed bytes.
        limit: u64,
    },
    /// The assembled result cannot be represented by the host allocation API.
    HostAllocationOverflow {
        /// Requested output size.
        bytes: u64,
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
            Self::HostAllocationOverflow { bytes } => {
                write!(f, "range assembly output size {bytes} is not representable")
            }
        }
    }
}

impl std::error::Error for RangeAssemblyPlanError {}

/// Why an otherwise valid range assembly failed during I/O.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RangeAssemblyError {
    /// The output buffer could not be reserved before starting source I/O.
    AllocationFailed {
        /// Requested output size.
        bytes: u64,
    },
    /// A source read failed.
    Read {
        /// Zero-based range index in the plan.
        part: usize,
        /// Source error.
        error: ReadError,
    },
    /// A source returned fewer bytes than requested.
    ShortRead {
        /// Zero-based range index in the plan.
        part: usize,
        /// Requested byte count.
        expected: u64,
        /// Returned byte count.
        actual: u64,
    },
    /// The source panicked during a read.
    SourcePanicked {
        /// Zero-based range index in the plan.
        part: usize,
    },
    /// Cancellation was requested before the payload was published.
    Cancelled,
}

impl fmt::Display for RangeAssemblyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllocationFailed { bytes } => write!(f, "cannot reserve {bytes} assembly bytes"),
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
    let resolved = resolve_source(sources, plan)?;
    assemble_ranges_resolved(&resolved, plan)
}

fn resolve_source(
    sources: &AssetSources,
    plan: &RangeAssemblyPlan,
) -> Result<crate::source::ResolvedAssetRead, RangeAssemblyError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sources.resolve_read(plan.path())
    }))
    .map_err(|_| RangeAssemblyError::SourcePanicked { part: 0 })?
    .map_err(|error| RangeAssemblyError::Read { part: 0, error })
}

pub(crate) fn assemble_ranges_resolved(
    source: &crate::source::ResolvedAssetRead,
    plan: &RangeAssemblyPlan,
) -> Result<Vec<u8>, RangeAssemblyError> {
    let mut output = reserve_output(plan.total_bytes)?;
    for (index, part) in plan.parts.iter().copied().enumerate() {
        let bytes = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            source
                .reader
                .read_range(&source.path, part.offset, part.len)
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

fn reserve_output(bytes: u64) -> Result<Vec<u8>, RangeAssemblyError> {
    let capacity =
        usize::try_from(bytes).map_err(|_| RangeAssemblyError::AllocationFailed { bytes })?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| RangeAssemblyError::AllocationFailed { bytes })?;
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
        if task_cancellation.is_cancelled() {
            return Err(RangeAssemblyError::Cancelled);
        }
        let source = resolve_source(&sources, &plan)?;
        let mut output = reserve_output(plan.total_bytes)?;
        let parallel = plan.limits.max_parallel;
        for (chunk_index, chunk) in plan.parts.chunks(parallel).enumerate() {
            if task_cancellation.is_cancelled() {
                return Err(RangeAssemblyError::Cancelled);
            }
            let mut reads = Vec::with_capacity(chunk.len());
            for (relative, part) in chunk.iter().copied().enumerate() {
                let reader = Arc::clone(&source.reader);
                let path = source.path.clone();
                let cancellation = task_cancellation.clone();
                let index = chunk_index * parallel;
                reads.push(child_pool.spawn_async(async move {
                    if cancellation.is_cancelled() {
                        return Err(RangeAssemblyError::Cancelled);
                    }
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        reader.read_range(&path, part.offset, part.len)
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
    use crate::{AssetSources, MemSource};
    use alloc::sync::Arc;
    use prism_tasks::TaskPool;

    fn source() -> AssetSources {
        let mut source = MemSource::new();
        source.insert("payload.bin", Arc::<[u8]>::from(&b"0123456789"[..]));
        let mut sources = AssetSources::new();
        sources.mount("memory", Arc::new(source), 0);
        sources
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
    fn rejects_host_allocation_overflow_before_io() {
        let bytes = isize::MAX as u64 + 1;
        assert_eq!(
            RangeAssemblyPlan::whole(
                "huge.bin",
                bytes,
                RangeAssemblyLimits {
                    max_bytes: u64::MAX,
                    ..limits()
                }
            ),
            Err(RangeAssemblyPlanError::HostAllocationOverflow { bytes })
        );
    }

    #[test]
    fn finite_part_budget_stops_an_infinite_iterator() {
        let mut observed = 0;
        let parts = core::iter::repeat_with(|| {
            observed += 1;
            RangePart::new(0, 0)
        });
        assert_eq!(
            RangeAssemblyPlan::new("payload.bin", 10, parts, limits()),
            Err(RangeAssemblyPlanError::TooManyParts {
                actual: 9,
                limit: 8
            })
        );
        assert_eq!(observed, 9);
        assert!(matches!(
            crate::AssetReadPlan::ranges(core::iter::repeat(RangePart::new(0, 0))),
            Err(RangeAssemblyPlanError::TooManyParts {
                actual: 257,
                limit: 256
            })
        ));
    }

    struct PartialOverlay {
        short_read: bool,
    }

    impl crate::AssetReader for PartialOverlay {
        fn read(&self, _: &str) -> Result<Vec<u8>, ReadError> {
            Err(ReadError::Unsupported)
        }

        fn metadata(&self, _: &str) -> Result<crate::AssetMeta, ReadError> {
            Ok(crate::AssetMeta {
                size: 10,
                modified_ms: None,
                is_dir: false,
            })
        }

        fn read_range(&self, _: &str, offset: u64, len: u64) -> Result<Vec<u8>, ReadError> {
            if offset == 0 {
                Ok(alloc::vec![b'X'; len as usize])
            } else if self.short_read {
                Ok(Vec::new())
            } else {
                Err(ReadError::NotFound)
            }
        }
    }

    #[test]
    fn sequential_and_parallel_parts_never_mix_overlay_sources() {
        let mut sources = source();
        sources.mount("patch", Arc::new(PartialOverlay { short_read: false }), 10);
        let plan = RangeAssemblyPlan::new(
            "payload.bin",
            10,
            [RangePart::new(0, 2), RangePart::new(4, 2)],
            limits(),
        )
        .unwrap();
        let expected = Err(RangeAssemblyError::Read {
            part: 1,
            error: ReadError::NotFound,
        });
        assert_eq!(assemble_ranges(&sources, &plan), expected);
        let pool = TaskPool::with_threads(2);
        assert_eq!(
            pool.block_on(assemble_ranges_async(sources, plan, pool.clone())),
            expected
        );
    }

    #[test]
    fn sequential_and_parallel_assembly_discard_partial_short_reads() {
        let mut sources = source();
        sources.mount("patch", Arc::new(PartialOverlay { short_read: true }), 10);
        let plan = RangeAssemblyPlan::new(
            "payload.bin",
            10,
            [RangePart::new(0, 2), RangePart::new(4, 2)],
            limits(),
        )
        .unwrap();
        let expected = Err(RangeAssemblyError::ShortRead {
            part: 1,
            expected: 2,
            actual: 0,
        });
        assert_eq!(assemble_ranges(&sources, &plan), expected);
        let pool = TaskPool::with_threads(2);
        assert_eq!(
            pool.block_on(assemble_ranges_async(sources, plan, pool.clone())),
            expected
        );
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
