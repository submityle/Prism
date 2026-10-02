//! Stale-while-revalidate (SWR) caching for async values.
//!
//! A classic latency trick: when a cached value exists but may be out of date,
//! serve the *stale* value immediately so the UI stays responsive, and kick off
//! a background refresh at the same time. Once the refresh lands, the cache is
//! updated and future reads are fresh again. Loom models this without threads
//! or futures — the cache is an inert state machine that the owner drives:
//! [`Swr::read`] reports what to show and whether a revalidation is now owed,
//! and [`Swr::commit`]/[`Swr::fail_revalidation`] record the refresh outcome.
//!
//! The lifecycle moves through [`SwrPhase`]:
//!
//! * `Empty` — never populated; a read is a miss that owes a first fetch.
//! * `Fresh` — the cached value is trusted; reads are hits with no refresh.
//! * `Stale` — the value is served but a refresh is owed on the next read.
//! * `Revalidating` — a refresh is in flight; reads keep serving the old value.

use crate::state::AsyncState;

/// The freshness phase of an [`Swr`] cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SwrPhase {
    /// No value has ever been cached.
    Empty,
    /// The cached value is current and trusted.
    Fresh,
    /// A value is cached but considered out of date.
    Stale,
    /// A refresh is in flight; the previous value (if any) is still served.
    Revalidating,
}

impl SwrPhase {
    /// Returns `true` when no value has ever been cached.
    #[must_use]
    pub fn is_empty(self) -> bool {
        matches!(self, SwrPhase::Empty)
    }

    /// Returns `true` when the cached value is current.
    #[must_use]
    pub fn is_fresh(self) -> bool {
        matches!(self, SwrPhase::Fresh)
    }

    /// Returns `true` when the cached value is out of date.
    #[must_use]
    pub fn is_stale(self) -> bool {
        matches!(self, SwrPhase::Stale)
    }

    /// Returns `true` when a refresh is currently in flight.
    #[must_use]
    pub fn is_revalidating(self) -> bool {
        matches!(self, SwrPhase::Revalidating)
    }
}

/// The outcome of an [`Swr::read`], describing what to display and whether a
/// background revalidation is now owed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SwrRead<T> {
    /// Nothing is cached; the caller should fetch for the first time.
    Miss,
    /// A trusted value; no refresh is needed.
    Fresh(T),
    /// A stale value served immediately; the caller should revalidate.
    Stale(T),
    /// A previous value served while a refresh is already in flight.
    Revalidating(T),
}

impl<T> SwrRead<T> {
    /// Returns `true` when there was no value to serve.
    #[must_use]
    pub fn is_miss(&self) -> bool {
        matches!(self, SwrRead::Miss)
    }

    /// Borrows the served value, if any.
    #[must_use]
    pub fn value(&self) -> Option<&T> {
        match self {
            SwrRead::Miss => None,
            SwrRead::Fresh(v) | SwrRead::Stale(v) | SwrRead::Revalidating(v) => Some(v),
        }
    }

    /// Returns `true` when the caller should start a revalidation as a result
    /// of this read (a first-time miss or a stale hit).
    #[must_use]
    pub fn should_revalidate(&self) -> bool {
        matches!(self, SwrRead::Miss | SwrRead::Stale(_))
    }
}

/// A stale-while-revalidate cache cell holding at most one value.
///
/// The cell is driven explicitly: [`Swr::read`] reports the current serve
/// decision and advances the phase to `Revalidating` when a refresh becomes
/// owed, while [`Swr::commit`] and [`Swr::fail_revalidation`] record how a
/// refresh ended.
///
/// # Example
///
/// ```
/// use prism_ui_async::{Swr, SwrRead};
///
/// // Seed with a value, then let it go stale.
/// let mut cache = Swr::with_value("v1");
/// cache.mark_stale();
///
/// // A stale read serves the old value *and* owes a revalidation.
/// let read = cache.read();
/// assert_eq!(read, SwrRead::Stale("v1"));
/// assert!(read.should_revalidate());
/// assert!(cache.is_revalidating());
///
/// // The background refresh lands and the cache is fresh again.
/// cache.commit("v2");
/// assert_eq!(cache.read(), SwrRead::Fresh("v2"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Swr<T> {
    value: Option<T>,
    phase: SwrPhase,
}

impl<T> Default for Swr<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Swr<T> {
    /// Creates an empty cache that owes a first fetch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            value: None,
            phase: SwrPhase::Empty,
        }
    }

    /// Creates a cache pre-seeded with a fresh value.
    #[must_use]
    pub fn with_value(value: T) -> Self {
        Self {
            value: Some(value),
            phase: SwrPhase::Fresh,
        }
    }

    /// The current freshness phase.
    #[must_use]
    pub fn phase(&self) -> SwrPhase {
        self.phase
    }

    /// Borrows the cached value without changing the phase.
    #[must_use]
    pub fn peek(&self) -> Option<&T> {
        self.value.as_ref()
    }

    /// Returns `true` when no value has ever been cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.phase.is_empty()
    }

    /// Returns `true` when the cached value is current.
    #[must_use]
    pub fn is_fresh(&self) -> bool {
        self.phase.is_fresh()
    }

    /// Returns `true` when the cached value is out of date.
    #[must_use]
    pub fn is_stale(&self) -> bool {
        self.phase.is_stale()
    }

    /// Returns `true` when a refresh is currently in flight.
    #[must_use]
    pub fn is_revalidating(&self) -> bool {
        self.phase.is_revalidating()
    }

    /// Marks a currently-fresh value as stale so the next read triggers a
    /// background revalidation. Has no effect in any other phase.
    pub fn mark_stale(&mut self) {
        if self.phase.is_fresh() {
            self.phase = SwrPhase::Stale;
        }
    }

    /// Records a successful refresh: stores the new value and marks it fresh.
    pub fn commit(&mut self, value: T) {
        self.value = Some(value);
        self.phase = SwrPhase::Fresh;
    }

    /// Records a failed refresh. A previously served value is retained but
    /// demoted to stale (so a later read can retry); an empty cache stays empty.
    pub fn fail_revalidation(&mut self) {
        if self.value.is_some() {
            self.phase = SwrPhase::Stale;
        } else {
            self.phase = SwrPhase::Empty;
        }
    }

    /// Projects the current cache into an [`AsyncState`]: `Pending` while a
    /// first value is still missing, otherwise `Ready` with a reference to the
    /// served value. Never fails because the cache models success-or-absent.
    #[must_use]
    pub fn as_state<E>(&self) -> AsyncState<&T, E> {
        match &self.value {
            Some(v) => AsyncState::Ready(v),
            None => AsyncState::Pending,
        }
    }
}

impl<T: Clone> Swr<T> {
    /// Reads the cache, deciding what to serve and advancing the phase when a
    /// revalidation becomes owed.
    ///
    /// * `Empty` → [`SwrRead::Miss`], phase becomes `Revalidating` (a first
    ///   fetch is now in flight).
    /// * `Fresh` → [`SwrRead::Fresh`] clone, phase unchanged.
    /// * `Stale` → [`SwrRead::Stale`] clone, phase becomes `Revalidating`
    ///   (serve stale now, refresh in the background).
    /// * `Revalidating` → [`SwrRead::Revalidating`] clone of the retained value
    ///   (or [`SwrRead::Miss`] if none), phase unchanged.
    pub fn read(&mut self) -> SwrRead<T> {
        match self.phase {
            SwrPhase::Empty => {
                self.phase = SwrPhase::Revalidating;
                SwrRead::Miss
            }
            SwrPhase::Fresh => match &self.value {
                Some(v) => SwrRead::Fresh(v.clone()),
                None => {
                    self.phase = SwrPhase::Revalidating;
                    SwrRead::Miss
                }
            },
            SwrPhase::Stale => match &self.value {
                Some(v) => {
                    let served = v.clone();
                    self.phase = SwrPhase::Revalidating;
                    SwrRead::Stale(served)
                }
                None => {
                    self.phase = SwrPhase::Revalidating;
                    SwrRead::Miss
                }
            },
            SwrPhase::Revalidating => match &self.value {
                Some(v) => SwrRead::Revalidating(v.clone()),
                None => SwrRead::Miss,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Swr, SwrPhase, SwrRead};
    use crate::state::AsyncState;

    #[test]
    fn empty_cache_misses_and_owes_fetch() {
        let mut cache: Swr<i32> = Swr::new();
        assert!(cache.is_empty());
        assert_eq!(cache.peek(), None);
        let read = cache.read();
        assert_eq!(read, SwrRead::Miss);
        assert!(read.should_revalidate());
        assert_eq!(cache.phase(), SwrPhase::Revalidating);
    }

    #[test]
    fn with_value_starts_fresh() {
        let mut cache = Swr::with_value(10);
        assert!(cache.is_fresh());
        assert_eq!(cache.peek(), Some(&10));
        let read = cache.read();
        assert_eq!(read, SwrRead::Fresh(10));
        assert!(!read.should_revalidate());
        // A fresh read does not change the phase.
        assert!(cache.is_fresh());
    }

    #[test]
    fn stale_hit_serves_value_and_triggers_revalidate() {
        let mut cache = Swr::with_value(1);
        cache.mark_stale();
        assert!(cache.is_stale());

        let read = cache.read();
        assert_eq!(read, SwrRead::Stale(1));
        assert!(read.should_revalidate());
        // The read moved the cache into the revalidating phase.
        assert_eq!(cache.phase(), SwrPhase::Revalidating);
        // The old value is still served while revalidating.
        assert_eq!(cache.read(), SwrRead::Revalidating(1));
    }

    #[test]
    fn commit_after_revalidation_restores_fresh() {
        let mut cache = Swr::with_value(1);
        cache.mark_stale();
        let _ = cache.read();
        assert!(cache.is_revalidating());

        cache.commit(2);
        assert!(cache.is_fresh());
        assert_eq!(cache.peek(), Some(&2));
        assert_eq!(cache.read(), SwrRead::Fresh(2));
    }

    #[test]
    fn commit_into_empty_cache_populates_fresh() {
        let mut cache: Swr<i32> = Swr::new();
        let _ = cache.read();
        assert!(cache.is_revalidating());
        cache.commit(99);
        assert!(cache.is_fresh());
        assert_eq!(cache.read(), SwrRead::Fresh(99));
    }

    #[test]
    fn failed_revalidation_demotes_value_to_stale() {
        let mut cache = Swr::with_value(5);
        cache.mark_stale();
        let _ = cache.read();
        assert!(cache.is_revalidating());

        cache.fail_revalidation();
        // Value retained, demoted to stale so a later read retries.
        assert!(cache.is_stale());
        assert_eq!(cache.peek(), Some(&5));
        assert_eq!(cache.read(), SwrRead::Stale(5));
    }

    #[test]
    fn failed_first_fetch_returns_to_empty() {
        let mut cache: Swr<i32> = Swr::new();
        let _ = cache.read();
        assert!(cache.is_revalidating());
        cache.fail_revalidation();
        assert!(cache.is_empty());
        assert_eq!(cache.peek(), None);
    }

    #[test]
    fn mark_stale_only_affects_fresh() {
        let mut empty: Swr<i32> = Swr::new();
        empty.mark_stale();
        assert!(empty.is_empty());

        let mut cache = Swr::with_value(1);
        let _ = cache.read();
        // read on fresh keeps it fresh, so mark_stale still works
        cache.mark_stale();
        assert!(cache.is_stale());
        // Entering revalidating then marking stale is a no-op.
        let _ = cache.read();
        assert!(cache.is_revalidating());
        cache.mark_stale();
        assert!(cache.is_revalidating());
    }

    #[test]
    fn revalidating_without_value_misses() {
        let mut cache: Swr<i32> = Swr::new();
        // First read moves to revalidating with no value.
        assert_eq!(cache.read(), SwrRead::Miss);
        // Subsequent reads during the in-flight first fetch still miss.
        assert_eq!(cache.read(), SwrRead::Miss);
    }

    #[test]
    fn as_state_projects_pending_then_ready() {
        let empty: Swr<i32> = Swr::new();
        let state: AsyncState<&i32, ()> = empty.as_state();
        assert_eq!(state, AsyncState::Pending);

        let cache = Swr::with_value(3);
        let state: AsyncState<&i32, ()> = cache.as_state();
        assert_eq!(state, AsyncState::Ready(&3));
    }

    #[test]
    fn swr_read_value_accessor() {
        assert_eq!(SwrRead::<i32>::Miss.value(), None);
        assert_eq!(SwrRead::Fresh(1).value(), Some(&1));
        assert_eq!(SwrRead::Stale(2).value(), Some(&2));
        assert_eq!(SwrRead::Revalidating(3).value(), Some(&3));
        assert!(SwrRead::<i32>::Miss.is_miss());
    }

    #[test]
    fn phase_predicates() {
        assert!(SwrPhase::Empty.is_empty());
        assert!(SwrPhase::Fresh.is_fresh());
        assert!(SwrPhase::Stale.is_stale());
        assert!(SwrPhase::Revalidating.is_revalidating());
    }
}
