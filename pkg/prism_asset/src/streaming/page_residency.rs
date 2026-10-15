//! CPU-side fixed-page range residency.
//!
//! This module owns page identity, CPU bytes, admission, and deterministic
//! reclamation. It performs real `AssetSources::read_range` calls but creates no
//! GPU resources, performs no GPU upload/decode, and depends on no renderer.

use super::super::{AssetPath, AssetSources, ReadError};
use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;
use core::fmt;
use prism_tasks::Priority;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// A validated fixed page size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageSize(u64);

impl PageSize {
    /// Creates a non-zero power-of-two page size.
    pub fn new(bytes: u64) -> Result<Self, PageSizeError> {
        if bytes == 0 {
            return Err(PageSizeError::Zero);
        }
        if !bytes.is_power_of_two() {
            return Err(PageSizeError::NotPowerOfTwo(bytes));
        }
        Ok(Self(bytes))
    }

    /// Returns the size in bytes.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.0
    }
}

/// Invalid fixed page size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageSizeError {
    /// The requested page size was zero.
    Zero,
    /// The requested page size was not a power of two.
    NotPowerOfTwo(u64),
}

impl fmt::Display for PageSizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => f.write_str("page size must be non-zero"),
            Self::NotPowerOfTwo(bytes) => write!(f, "page size {bytes} is not a power of two"),
        }
    }
}

impl std::error::Error for PageSizeError {}

/// CPU byte, in-flight, and terminal-diagnostic limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageBudget {
    max_bytes: u64,
    max_in_flight: usize,
    max_terminal_records: usize,
}

impl PageBudget {
    /// Creates a budget with at least one admission slot.
    pub fn new(max_bytes: u64, max_in_flight: usize) -> Result<Self, PageBudgetError> {
        if max_in_flight == 0 {
            return Err(PageBudgetError::ZeroInFlight);
        }
        Ok(Self {
            max_bytes,
            max_in_flight,
            max_terminal_records: max_in_flight.saturating_mul(2).max(1),
        })
    }

    /// Returns the maximum CPU bytes admitted by this budget.
    #[must_use]
    pub const fn max_bytes(self) -> u64 {
        self.max_bytes
    }

    /// Returns the maximum number of simultaneous loads.
    #[must_use]
    pub const fn max_in_flight(self) -> usize {
        self.max_in_flight
    }

    /// Returns the bounded number of failed/cancelled records retained for
    /// diagnostics. A value of zero drops terminal metadata immediately.
    #[must_use]
    pub const fn max_terminal_records(self) -> usize {
        self.max_terminal_records
    }

    /// Overrides terminal diagnostic retention without changing byte or
    /// in-flight admission limits.
    #[must_use]
    pub const fn with_max_terminal_records(mut self, max_terminal_records: usize) -> Self {
        self.max_terminal_records = max_terminal_records;
        self
    }
}

/// Invalid page budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageBudgetError {
    /// The in-flight limit was zero.
    ZeroInFlight,
    /// The byte budget is smaller than one page.
    SmallerThanPage {
        /// Configured byte budget.
        budget: u64,
        /// Fixed page size.
        page_size: u64,
    },
    /// The byte budget is not an integer number of pages.
    NotPageAligned {
        /// Configured byte budget.
        budget: u64,
        /// Fixed page size.
        page_size: u64,
    },
}

impl fmt::Display for PageBudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroInFlight => f.write_str("page budget max_in_flight must be non-zero"),
            Self::SmallerThanPage { budget, page_size } => {
                write!(
                    f,
                    "page budget {budget} is smaller than page size {page_size}"
                )
            }
            Self::NotPageAligned { budget, page_size } => {
                write!(
                    f,
                    "page budget {budget} is not aligned to page size {page_size}"
                )
            }
        }
    }
}

impl std::error::Error for PageBudgetError {}

/// Source path, revision, and zero-based fixed page index.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageKey {
    path: AssetPath,
    page_index: u64,
    source_revision: u64,
}

impl PageKey {
    /// Creates a key tied to one source-table/content revision.
    #[must_use]
    pub fn new(path: impl Into<AssetPath>, page_index: u64, source_revision: u64) -> Self {
        Self {
            path: path.into(),
            page_index,
            source_revision,
        }
    }

    /// Creates a key using the current mounted-source revision.
    #[must_use]
    pub fn from_sources(
        sources: &AssetSources,
        path: impl Into<AssetPath>,
        page_index: u64,
    ) -> Self {
        Self::new(path, page_index, sources.revision())
    }

    /// Returns the revision that produced this page key.
    #[must_use]
    pub const fn source_revision(&self) -> u64 {
        self.source_revision
    }

    /// Returns the source path associated with this page.
    #[must_use]
    pub fn path(&self) -> &AssetPath {
        &self.path
    }

    /// Returns the zero-based page index.
    #[must_use]
    pub const fn page_index(&self) -> u64 {
        self.page_index
    }

    /// Computes the fixed range, rejecting offset and end overflow.
    pub fn range(&self, page_size: PageSize) -> Result<(u64, u64), PageRangeError> {
        let offset = self.page_index.checked_mul(page_size.bytes()).ok_or(
            PageRangeError::OffsetOverflow {
                page_index: self.page_index,
                page_size: page_size.bytes(),
            },
        )?;
        offset
            .checked_add(page_size.bytes())
            .ok_or(PageRangeError::EndOverflow {
                offset,
                page_size: page_size.bytes(),
            })?;
        Ok((offset, page_size.bytes()))
    }
}

/// A page offset that cannot be represented in u64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageRangeError {
    /// Multiplying the page index by the page size overflowed.
    OffsetOverflow {
        /// Page index that overflowed when multiplied.
        page_index: u64,
        /// Page size used for multiplication.
        page_size: u64,
    },
    /// Adding the page size to the offset overflowed.
    EndOverflow {
        /// Computed page offset.
        offset: u64,
        /// Page size added to the offset.
        page_size: u64,
    },
}

impl fmt::Display for PageRangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OffsetOverflow {
                page_index,
                page_size,
            } => write!(
                f,
                "page range offset overflows: page_index={page_index}, page_size={page_size}"
            ),
            Self::EndOverflow { offset, page_size } => write!(
                f,
                "page range end overflows: offset={offset}, page_size={page_size}"
            ),
        }
    }
}

impl std::error::Error for PageRangeError {}

/// Stable identity of one page request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageId(u64);

impl PageId {
    /// Returns the numeric page identity.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Page lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageState {
    /// Requested but not yet admitted.
    Pending,
    /// Admitted and being loaded.
    Loading,
    /// Loaded and retaining page bytes.
    Resident,
    /// Load failed and retained a diagnostic.
    Failed,
    /// Cancelled before becoming resident.
    Cancelled,
}

/// Failure retained by a terminal page record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PageFailure {
    /// The source read returned an error.
    Read(ReadError),
    /// The source returned fewer bytes than the fixed page size.
    ShortRead {
        /// Required page byte count.
        expected: u64,
        /// Bytes returned by the source.
        actual: u64,
    },
    /// The source panicked during the read.
    SourcePanicked,
    /// An external subsystem reported its numeric failure code.
    External(u32),
}

/// Error while requesting a page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PageRequestError {
    /// The fixed page range could not be represented.
    Range(PageRangeError),
    /// No more page ids can be allocated.
    IdExhausted,
}

impl fmt::Display for PageRequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Range(error) => error.fmt(f),
            Self::IdExhausted => f.write_str("page id space exhausted"),
        }
    }
}

impl std::error::Error for PageRequestError {}

/// Error while reserving a pending page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PageAdmissionError {
    /// The page id is not known.
    Unknown(PageId),
    /// The page is not pending.
    InvalidState {
        /// Page whose state was rejected.
        id: PageId,
        /// State observed at rejection.
        state: PageState,
    },
    /// The CPU byte budget would be exceeded.
    BudgetExceeded {
        /// Aggregate bytes after admission.
        requested: u64,
        /// Configured byte limit.
        limit: u64,
    },
    /// The in-flight page limit would be exceeded.
    Backpressure {
        /// Current in-flight page count.
        in_flight: usize,
        /// Configured in-flight limit.
        max_in_flight: usize,
    },
    /// Page accounting overflowed.
    BudgetOverflow,
}

impl fmt::Display for PageAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(id) => write!(f, "unknown page {}", id.get()),
            Self::InvalidState { id, state } => {
                write!(f, "page {} is not pending; state is {state:?}", id.get())
            }
            Self::BudgetExceeded { requested, limit } => {
                write!(
                    f,
                    "page CPU budget exceeded: requested {requested}, limit {limit}"
                )
            }
            Self::Backpressure {
                in_flight,
                max_in_flight,
            } => write!(
                f,
                "page load backpressure: {in_flight} in flight, limit {max_in_flight}"
            ),
            Self::BudgetOverflow => f.write_str("page CPU budget arithmetic overflowed"),
        }
    }
}

impl std::error::Error for PageAdmissionError {}

/// Error for a non-admission state transition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PageTransitionError {
    /// The page id is not known.
    Unknown(PageId),
    /// The page is not valid for the requested transition.
    InvalidState {
        /// Page whose transition was rejected.
        id: PageId,
        /// State observed at rejection.
        state: PageState,
    },
}

impl fmt::Display for PageTransitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(id) => write!(f, "unknown page {}", id.get()),
            Self::InvalidState { id, state } => {
                write!(f, "invalid transition for page {} in {state:?}", id.get())
            }
        }
    }
}

impl std::error::Error for PageTransitionError {}

/// Error from a real synchronous page read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PageLoadError {
    /// Admission failed before reading.
    Admission(PageAdmissionError),
    /// The source read failed.
    Read {
        /// Page whose read failed.
        id: PageId,
        /// Source error.
        error: ReadError,
    },
    /// The source returned fewer bytes than one page.
    ShortRead {
        /// Page whose read was short.
        id: PageId,
        /// Required page size.
        expected: u64,
        /// Bytes actually returned.
        actual: u64,
    },
    /// The source panicked during the read.
    SourcePanicked {
        /// Page whose source panicked.
        id: PageId,
    },
    /// The terminal transition could not be committed.
    Transition(PageTransitionError),
}

impl fmt::Display for PageLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => error.fmt(f),
            Self::Read { id, error } => write!(f, "page {} read failed: {error}", id.get()),
            Self::ShortRead {
                id,
                expected,
                actual,
            } => write!(
                f,
                "page {} short read: expected {expected}, received {actual}",
                id.get()
            ),
            Self::SourcePanicked { id } => write!(f, "page {} source read panicked", id.get()),
            Self::Transition(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for PageLoadError {}

/// CPU budget usage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageUsage {
    /// Bytes held by resident pages.
    pub resident_bytes: u64,
    /// Bytes reserved by loading pages.
    pub loading_bytes: u64,
    /// Number of pages currently loading.
    pub in_flight_pages: usize,
}

#[derive(Debug)]
struct PageRecord {
    key: PageKey,
    state: PageState,
    priority: Priority,
    importance: u32,
    sequence: u64,
    failure: Option<PageFailure>,
    bytes: Option<Vec<u8>>,
}

/// Deterministic CPU fixed-page residency authority.
///
/// A key is deduplicated while pending, loading, or resident. Callers can
/// split `begin_load` and `complete_load` around asynchronous I/O, or use load for
/// a real synchronous `AssetSources::read_range` call.
pub struct PageResidency {
    page_size: PageSize,
    budget: PageBudget,
    usage: PageUsage,
    next_id: u64,
    next_sequence: u64,
    active: BTreeMap<PageKey, PageId>,
    records: BTreeMap<PageId, PageRecord>,
    terminal_order: VecDeque<PageId>,
}

impl PageResidency {
    /// Creates a manager after validating budget/page alignment.
    pub fn new(page_size: PageSize, budget: PageBudget) -> Result<Self, PageBudgetError> {
        if budget.max_bytes < page_size.bytes() {
            return Err(PageBudgetError::SmallerThanPage {
                budget: budget.max_bytes,
                page_size: page_size.bytes(),
            });
        }
        if budget.max_bytes % page_size.bytes() != 0 {
            return Err(PageBudgetError::NotPageAligned {
                budget: budget.max_bytes,
                page_size: page_size.bytes(),
            });
        }
        Ok(Self {
            page_size,
            budget,
            usage: PageUsage::default(),
            next_id: 0,
            next_sequence: 0,
            active: BTreeMap::new(),
            records: BTreeMap::new(),
            terminal_order: VecDeque::new(),
        })
    }

    /// Returns the configured fixed page size.
    #[must_use]
    pub const fn page_size(&self) -> PageSize {
        self.page_size
    }

    /// Returns the configured page budget.
    #[must_use]
    pub const fn budget(&self) -> PageBudget {
        self.budget
    }

    /// Returns current resident/loading byte usage.
    #[must_use]
    pub const fn usage(&self) -> PageUsage {
        self.usage
    }

    /// Requests one page, returning the existing id for an active duplicate.
    /// Repeated requests can raise priority and importance.
    pub fn request(
        &mut self,
        key: PageKey,
        priority: Priority,
        importance: u32,
    ) -> Result<PageId, PageRequestError> {
        key.range(self.page_size).map_err(PageRequestError::Range)?;
        if let Some(id) = self.active.get(&key).copied() {
            if let Some(record) = self.records.get_mut(&id) {
                if matches!(
                    record.state,
                    PageState::Pending | PageState::Loading | PageState::Resident
                ) {
                    record.priority = record.priority.max(priority);
                    record.importance = record.importance.max(importance);
                    return Ok(id);
                }
            }
            self.active.remove(&key);
        }
        let id = PageId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(PageRequestError::IdExhausted)?;
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(PageRequestError::IdExhausted)?;
        self.records.insert(
            id,
            PageRecord {
                key: key.clone(),
                state: PageState::Pending,
                priority,
                importance,
                sequence,
                failure: None,
                bytes: None,
            },
        );
        self.active.insert(key, id);
        Ok(id)
    }

    /// Returns the highest-priority pending page, FIFO within a priority tier.
    #[must_use]
    pub fn next_pending(&self) -> Option<(PageId, &PageKey)> {
        self.records
            .iter()
            .filter(|(_, record)| record.state == PageState::Pending)
            .max_by_key(|(id, record)| {
                (
                    record.priority,
                    record.importance,
                    core::cmp::Reverse(record.sequence),
                    core::cmp::Reverse(id.get()),
                )
            })
            .map(|(id, record)| (*id, &record.key))
    }

    /// Returns resident pages in deterministic eviction order.
    #[must_use]
    pub fn eviction_candidates(&self) -> Vec<PageId> {
        let mut candidates = self
            .records
            .iter()
            .filter(|(_, record)| record.state == PageState::Resident)
            .map(|(id, record)| {
                (
                    record.priority,
                    record.importance,
                    record.sequence,
                    id.get(),
                    *id,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|candidate| (candidate.0, candidate.1, candidate.2, candidate.3));
        candidates
            .into_iter()
            .map(|candidate| candidate.4)
            .collect()
    }

    /// Reserves one page of CPU budget and enters Loading.
    pub fn begin_load(&mut self, id: PageId) -> Result<(), PageAdmissionError> {
        let state = self
            .records
            .get(&id)
            .map(|record| record.state)
            .ok_or(PageAdmissionError::Unknown(id))?;
        if state != PageState::Pending {
            return Err(PageAdmissionError::InvalidState { id, state });
        }
        if self.usage.in_flight_pages >= self.budget.max_in_flight {
            return Err(PageAdmissionError::Backpressure {
                in_flight: self.usage.in_flight_pages,
                max_in_flight: self.budget.max_in_flight,
            });
        }
        let requested = self
            .usage
            .resident_bytes
            .checked_add(self.usage.loading_bytes)
            .and_then(|used| used.checked_add(self.page_size.bytes()))
            .ok_or(PageAdmissionError::BudgetOverflow)?;
        if requested > self.budget.max_bytes {
            return Err(PageAdmissionError::BudgetExceeded {
                requested,
                limit: self.budget.max_bytes,
            });
        }
        self.usage.loading_bytes = self
            .usage
            .loading_bytes
            .checked_add(self.page_size.bytes())
            .ok_or(PageAdmissionError::BudgetOverflow)?;
        self.usage.in_flight_pages = self
            .usage
            .in_flight_pages
            .checked_add(1)
            .ok_or(PageAdmissionError::BudgetOverflow)?;
        self.records
            .get_mut(&id)
            .expect("page record checked above")
            .state = PageState::Loading;
        Ok(())
    }

    /// Commits one complete fixed page and takes ownership of its CPU bytes.
    pub fn complete_load(&mut self, id: PageId, bytes: Vec<u8>) -> Result<(), PageLoadError> {
        let state = self
            .records
            .get(&id)
            .map(|record| record.state)
            .ok_or(PageTransitionError::Unknown(id))
            .map_err(PageLoadError::Transition)?;
        if state != PageState::Loading {
            return Err(PageLoadError::Transition(
                PageTransitionError::InvalidState { id, state },
            ));
        }
        let expected = self.page_size.bytes();
        let actual = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if actual != expected {
            self.fail(id, PageFailure::ShortRead { expected, actual })
                .map_err(PageLoadError::Transition)?;
            return Err(PageLoadError::ShortRead {
                id,
                expected,
                actual,
            });
        }
        self.usage.loading_bytes = self.usage.loading_bytes.saturating_sub(expected);
        self.usage.resident_bytes = self.usage.resident_bytes.saturating_add(expected);
        self.usage.in_flight_pages = self.usage.in_flight_pages.saturating_sub(1);
        let record = self
            .records
            .get_mut(&id)
            .expect("page record checked above");
        record.state = PageState::Resident;
        record.failure = None;
        record.bytes = Some(bytes);
        Ok(())
    }

    /// Marks a pending or loading page failed and releases loading budget.
    pub fn fail(&mut self, id: PageId, failure: PageFailure) -> Result<(), PageTransitionError> {
        let (state, key) = self
            .records
            .get(&id)
            .map(|record| (record.state, record.key.clone()))
            .ok_or(PageTransitionError::Unknown(id))?;
        match state {
            PageState::Pending => {}
            PageState::Loading => self.release_loading(),
            state => return Err(PageTransitionError::InvalidState { id, state }),
        }
        self.deactivate(&key, id);
        let record = self
            .records
            .get_mut(&id)
            .expect("page record checked above");
        record.state = PageState::Failed;
        record.failure = Some(failure);
        record.bytes = None;
        self.retain_terminal(id);
        Ok(())
    }

    /// Cancels a pending or loading page and releases loading budget.
    pub fn cancel(&mut self, id: PageId) -> Result<(), PageTransitionError> {
        let (state, key) = self
            .records
            .get(&id)
            .map(|record| (record.state, record.key.clone()))
            .ok_or(PageTransitionError::Unknown(id))?;
        match state {
            PageState::Pending => {}
            PageState::Loading => self.release_loading(),
            state => return Err(PageTransitionError::InvalidState { id, state }),
        }
        self.deactivate(&key, id);
        let record = self
            .records
            .get_mut(&id)
            .expect("page record checked above");
        record.state = PageState::Cancelled;
        record.failure = None;
        record.bytes = None;
        self.retain_terminal(id);
        Ok(())
    }

    /// Evicts a resident page, dropping its bytes and record.
    pub fn evict(&mut self, id: PageId) -> Result<(), PageTransitionError> {
        let (state, key) = self
            .records
            .get(&id)
            .map(|record| (record.state, record.key.clone()))
            .ok_or(PageTransitionError::Unknown(id))?;
        if state != PageState::Resident {
            return Err(PageTransitionError::InvalidState { id, state });
        }
        self.usage.resident_bytes = self
            .usage
            .resident_bytes
            .saturating_sub(self.page_size.bytes());
        self.deactivate(&key, id);
        self.records.remove(&id);
        Ok(())
    }

    /// Drops a failed or cancelled historical record.
    pub fn forget_terminal(&mut self, id: PageId) -> Result<(), PageTransitionError> {
        let state = self
            .records
            .get(&id)
            .map(|record| record.state)
            .ok_or(PageTransitionError::Unknown(id))?;
        match state {
            PageState::Failed | PageState::Cancelled => {
                self.records.remove(&id);
                self.terminal_order.retain(|candidate| *candidate != id);
                Ok(())
            }
            state => Err(PageTransitionError::InvalidState { id, state }),
        }
    }

    /// Performs one real synchronous `AssetSources::read_range` call.
    pub fn load(&mut self, sources: &AssetSources, id: PageId) -> Result<(), PageLoadError> {
        self.begin_load(id).map_err(PageLoadError::Admission)?;
        let (key, offset) = {
            let record = self.records.get(&id).expect("page record checked above");
            let range = record
                .key
                .range(self.page_size)
                .expect("page range was validated at request");
            (record.key.clone(), range.0)
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            sources.read_range(key.path(), offset, self.page_size.bytes())
        }));
        match result {
            Ok(Ok(bytes)) => self.complete_load(id, bytes),
            Ok(Err(error)) => {
                self.fail(id, PageFailure::Read(error.clone()))
                    .map_err(PageLoadError::Transition)?;
                Err(PageLoadError::Read { id, error })
            }
            Err(_) => {
                self.fail(id, PageFailure::SourcePanicked)
                    .map_err(PageLoadError::Transition)?;
                Err(PageLoadError::SourcePanicked { id })
            }
        }
    }

    /// Loads the highest-priority pending page, if any.
    pub fn load_next(&mut self, sources: &AssetSources) -> Result<Option<PageId>, PageLoadError> {
        let id = self.next_pending().map(|(id, _)| id);
        id.map(|id| self.load(sources, id).map(|()| id)).transpose()
    }

    /// Returns the current state of a page, if known.
    #[must_use]
    pub fn state(&self, id: PageId) -> Option<PageState> {
        self.records.get(&id).map(|record| record.state)
    }

    /// Returns the retained failure for a page, if any.
    #[must_use]
    pub fn failure(&self, id: PageId) -> Option<&PageFailure> {
        self.records
            .get(&id)
            .and_then(|record| record.failure.as_ref())
    }

    /// Returns the key associated with a page, if known.
    #[must_use]
    pub fn key(&self, id: PageId) -> Option<&PageKey> {
        self.records.get(&id).map(|record| &record.key)
    }

    /// Returns resident bytes for a page, if it is resident.
    #[must_use]
    pub fn resident_bytes(&self, id: PageId) -> Option<&[u8]> {
        self.records
            .get(&id)
            .filter(|record| record.state == PageState::Resident)
            .and_then(|record| record.bytes.as_deref())
    }

    /// Returns the effective scheduling priority for a page.
    #[must_use]
    pub fn priority(&self, id: PageId) -> Option<Priority> {
        self.records.get(&id).map(|record| record.priority)
    }

    fn deactivate(&mut self, key: &PageKey, id: PageId) {
        if self.active.get(key).copied() == Some(id) {
            self.active.remove(key);
        }
    }

    fn retain_terminal(&mut self, id: PageId) {
        self.terminal_order.push_back(id);
        while self.terminal_order.len() > self.budget.max_terminal_records {
            let Some(oldest) = self.terminal_order.pop_front() else {
                break;
            };
            if matches!(
                self.records.get(&oldest).map(|record| record.state),
                Some(PageState::Failed | PageState::Cancelled)
            ) {
                self.records.remove(&oldest);
            }
        }
    }

    fn release_loading(&mut self) {
        self.usage.loading_bytes = self
            .usage
            .loading_bytes
            .saturating_sub(self.page_size.bytes());
        self.usage.in_flight_pages = self.usage.in_flight_pages.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemSource;
    use alloc::sync::Arc;

    fn size() -> PageSize {
        PageSize::new(4).unwrap()
    }

    fn manager(bytes: u64, in_flight: usize) -> PageResidency {
        PageResidency::new(size(), PageBudget::new(bytes, in_flight).unwrap()).unwrap()
    }

    fn sources() -> AssetSources {
        let mut sources = AssetSources::new();
        let mut memory = MemSource::new();
        memory.insert("payload.bin", b"abcdefghijkl".to_vec());
        sources.mount("memory", Arc::new(memory), 0);
        sources
    }

    #[test]
    fn validates_page_size_and_budget_alignment() {
        assert_eq!(PageSize::new(0), Err(PageSizeError::Zero));
        assert_eq!(PageSize::new(6), Err(PageSizeError::NotPowerOfTwo(6)));
        assert_eq!(PageBudget::new(16, 0), Err(PageBudgetError::ZeroInFlight));
        assert!(matches!(
            PageResidency::new(size(), PageBudget::new(2, 1).unwrap()),
            Err(PageBudgetError::SmallerThanPage {
                budget: 2,
                page_size: 4
            })
        ));
        assert!(matches!(
            PageResidency::new(PageSize::new(8).unwrap(), PageBudget::new(10, 1).unwrap()),
            Err(PageBudgetError::NotPageAligned {
                budget: 10,
                page_size: 8
            })
        ));
    }

    #[test]
    fn computes_fixed_ranges_and_rejects_overflow() {
        assert_eq!(
            PageKey::new("payload.bin", 2, 0).range(size()).unwrap(),
            (8, 4)
        );
        assert_eq!(
            PageKey::new("payload.bin", u64::MAX, 0).range(size()),
            Err(PageRangeError::OffsetOverflow {
                page_index: u64::MAX,
                page_size: 4
            })
        );
        assert_eq!(
            PageKey::new("payload.bin", u64::MAX / 4, 0).range(size()),
            Err(PageRangeError::EndOverflow {
                offset: u64::MAX - 3,
                page_size: 4
            })
        );
    }

    #[test]
    fn deduplicates_active_request_and_upgrades_priority() {
        let mut residency = manager(8, 1);
        let key = PageKey::new("payload.bin", 1, 0);
        let first = residency.request(key.clone(), Priority::Low, 1).unwrap();
        let duplicate = residency.request(key, Priority::High, 9).unwrap();
        assert_eq!(duplicate, first);
        assert_eq!(residency.priority(first), Some(Priority::High));
        assert_eq!(residency.next_pending().unwrap().0, first);
    }

    #[test]
    fn source_revision_prevents_stale_page_deduplication() {
        let mut residency = manager(8, 2);
        let old = residency
            .request(PageKey::new("payload.bin", 0, 1), Priority::Normal, 0)
            .unwrap();
        let new = residency
            .request(PageKey::new("payload.bin", 0, 2), Priority::Normal, 0)
            .unwrap();
        assert_ne!(old, new);
        assert_eq!(residency.key(old).unwrap().source_revision(), 1);
        assert_eq!(residency.key(new).unwrap().source_revision(), 2);
    }

    #[test]
    fn terminal_records_are_bounded_and_oldest_is_reclaimed() {
        let mut residency = PageResidency::new(
            size(),
            PageBudget::new(8, 1).unwrap().with_max_terminal_records(1),
        )
        .unwrap();
        let first = residency
            .request(PageKey::new("first.bin", 0, 0), Priority::Low, 0)
            .unwrap();
        residency.cancel(first).unwrap();
        assert_eq!(residency.state(first), Some(PageState::Cancelled));

        let second = residency
            .request(PageKey::new("second.bin", 0, 0), Priority::Low, 0)
            .unwrap();
        residency.cancel(second).unwrap();
        assert_eq!(residency.state(first), None);
        assert_eq!(residency.state(second), Some(PageState::Cancelled));
    }

    #[test]
    fn real_range_read_resides_exact_cpu_page() {
        let sources = sources();
        let mut residency = manager(8, 1);
        let id = residency
            .request(
                PageKey::new("memory://payload.bin", 1, sources.revision()),
                Priority::Normal,
                0,
            )
            .unwrap();
        residency.load(&sources, id).unwrap();
        assert_eq!(residency.state(id), Some(PageState::Resident));
        assert_eq!(residency.resident_bytes(id), Some(&b"efgh"[..]));
        assert_eq!(
            residency.usage(),
            PageUsage {
                resident_bytes: 4,
                loading_bytes: 0,
                in_flight_pages: 0
            }
        );
    }

    #[test]
    fn backpressure_failure_and_cancel_release_loading_budget() {
        let mut residency = manager(8, 1);
        let first = residency
            .request(PageKey::new("payload.bin", 0, 0), Priority::Normal, 0)
            .unwrap();
        let second = residency
            .request(PageKey::new("payload.bin", 1, 0), Priority::Normal, 0)
            .unwrap();
        residency.begin_load(first).unwrap();
        assert_eq!(
            residency.begin_load(second),
            Err(PageAdmissionError::Backpressure {
                in_flight: 1,
                max_in_flight: 1
            })
        );
        residency.fail(first, PageFailure::External(7)).unwrap();
        assert_eq!(residency.usage(), PageUsage::default());
        residency.begin_load(second).unwrap();
        residency.cancel(second).unwrap();
        assert_eq!(residency.usage(), PageUsage::default());
        assert_eq!(residency.state(first), Some(PageState::Failed));
        assert_eq!(residency.state(second), Some(PageState::Cancelled));
    }

    #[test]
    fn read_failure_releases_budget_and_retry_is_not_deduplicated() {
        let sources = sources();
        let mut residency = manager(4, 1);
        let first = residency
            .request(
                PageKey::new("memory://missing.bin", 0, sources.revision()),
                Priority::Normal,
                0,
            )
            .unwrap();
        assert!(matches!(
            residency.load(&sources, first),
            Err(PageLoadError::Read {
                id,
                error: ReadError::NotFound
            }) if id == first
        ));
        assert_eq!(residency.usage(), PageUsage::default());
        let retry = residency
            .request(
                PageKey::new("memory://missing.bin", 0, sources.revision()),
                Priority::Normal,
                0,
            )
            .unwrap();
        assert_ne!(retry, first);
    }

    #[test]
    fn deterministic_eviction_orders_low_priority_old_pages_first() {
        let sources = sources();
        let mut residency = manager(12, 3);
        let old_low = residency
            .request(
                PageKey::new("memory://payload.bin", 0, sources.revision()),
                Priority::Low,
                0,
            )
            .unwrap();
        let normal = residency
            .request(
                PageKey::new("memory://payload.bin", 1, sources.revision()),
                Priority::Normal,
                0,
            )
            .unwrap();
        let new_low = residency
            .request(
                PageKey::new("memory://payload.bin", 2, sources.revision()),
                Priority::Low,
                0,
            )
            .unwrap();
        residency.load(&sources, old_low).unwrap();
        residency.load(&sources, normal).unwrap();
        residency.load(&sources, new_low).unwrap();
        assert_eq!(
            residency.eviction_candidates(),
            vec![old_low, new_low, normal]
        );
        residency.evict(old_low).unwrap();
        assert_eq!(residency.usage().resident_bytes, 8);
        assert_eq!(residency.state(old_low), None);
    }
}
