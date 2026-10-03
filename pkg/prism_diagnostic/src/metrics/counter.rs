//! Metric instruments and a name-keyed registry.
//!
//! This module provides three thread-safe, allocation-light instruments plus a
//! [`MetricRegistry`] that hands out shared handles so a subsystem can register
//! a metric once and update it cheaply from any thread:
//!
//! - [`Gauge`]: a last-value `f64` (set/get), stored in a single atomic.
//! - [`Counter`] (aliased [`Sum`]): a monotonically increasing `u64`.
//! - [`Histogram`]: configurable upper-bound buckets with mean/min/max and
//!   percentile estimation from an immutable, lock-free snapshot.
//!
//! All instruments use `core` atomics only; there is no `unsafe` code. The
//! `f64`-valued instruments store the IEEE-754 bit pattern in an [`AtomicU64`]
//! and accumulate via compare-exchange loops.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

/// A last-value gauge holding a single `f64`.
///
/// Writers call [`Gauge::set`]; readers call [`Gauge::get`]. The value is held
/// as the IEEE-754 bit pattern inside one [`AtomicU64`], so updates are
/// wait-free and visible across threads. The default value is `0.0`.
#[derive(Debug, Default)]
pub struct Gauge {
    bits: AtomicU64,
}

impl Gauge {
    /// Create a gauge initialized to `0.0`.
    ///
    /// The bit pattern of `0.0f64` is `0`, so the zeroed atomic is already the
    /// correct initial value.
    pub const fn new() -> Self {
        Self {
            bits: AtomicU64::new(0),
        }
    }

    /// Overwrite the stored value.
    pub fn set(&self, value: f64) {
        self.bits.store(value.to_bits(), Ordering::Relaxed);
    }

    /// Read the current value.
    pub fn get(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::Relaxed))
    }
}

/// A monotonically increasing unsigned counter (a "sum" instrument).
///
/// Adds are performed with a single atomic fetch-add, so concurrent writers
/// never lose increments. Use [`Counter::add`] for arbitrary deltas or
/// [`Counter::incr`] for the common `+1` case.
#[derive(Debug, Default)]
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    /// Create a counter initialized to `0`.
    pub const fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Add `delta`, returning the previous value.
    pub fn add(&self, delta: u64) -> u64 {
        self.value.fetch_add(delta, Ordering::Relaxed)
    }

    /// Increment by one, returning the previous value.
    pub fn incr(&self) -> u64 {
        self.add(1)
    }

    /// Read the current total.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
}

/// Alias for [`Counter`]: a monotonic "sum" instrument.
pub type Sum = Counter;

/// Compare-exchange `atom` down to `value` if `value` is smaller (interpreted
/// as `f64` bits).
fn fetch_min_f64(atom: &AtomicU64, value: f64) {
    let mut cur = atom.load(Ordering::Relaxed);
    loop {
        if value >= f64::from_bits(cur) {
            break;
        }
        match atom.compare_exchange_weak(
            cur,
            value.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}

/// Compare-exchange `atom` up to `value` if `value` is larger (interpreted as
/// `f64` bits).
fn fetch_max_f64(atom: &AtomicU64, value: f64) {
    let mut cur = atom.load(Ordering::Relaxed);
    loop {
        if value <= f64::from_bits(cur) {
            break;
        }
        match atom.compare_exchange_weak(
            cur,
            value.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}

/// Compare-exchange add `value` into `atom` (interpreted as `f64` bits).
fn fetch_add_f64(atom: &AtomicU64, value: f64) {
    let mut cur = atom.load(Ordering::Relaxed);
    loop {
        let next = f64::from_bits(cur) + value;
        match atom.compare_exchange_weak(
            cur,
            next.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}

/// A histogram with caller-supplied upper-bound buckets.
///
/// Construct with [`Histogram::new`], passing the inclusive upper bounds of the
/// finite buckets (for example `[10.0, 20.0, 50.0]`). An implicit overflow
/// bucket catches everything greater than the last bound. Each [`record`] value
/// lands in the first bucket whose upper bound is `>=` the value, and updates
/// the running count, sum, min, and max.
///
/// [`record`]: Histogram::record
#[derive(Debug)]
pub struct Histogram {
    bounds: Vec<f64>,
    counts: Vec<AtomicU64>,
    count: AtomicU64,
    sum_bits: AtomicU64,
    min_bits: AtomicU64,
    max_bits: AtomicU64,
}

impl Histogram {
    /// Create a histogram from a slice of finite upper bounds.
    ///
    /// The bounds are sorted and de-duplicated. Non-finite bounds are dropped.
    /// A trailing overflow bucket is always present, so the internal bucket
    /// count is `bounds.len() + 1`.
    pub fn new(bounds: &[f64]) -> Self {
        let mut bounds: Vec<f64> = bounds.iter().copied().filter(|b| b.is_finite()).collect();
        bounds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        bounds.dedup();
        let counts = (0..bounds.len() + 1).map(|_| AtomicU64::new(0)).collect();
        Self {
            bounds,
            counts,
            count: AtomicU64::new(0),
            sum_bits: AtomicU64::new(0.0f64.to_bits()),
            min_bits: AtomicU64::new(f64::INFINITY.to_bits()),
            max_bits: AtomicU64::new(f64::NEG_INFINITY.to_bits()),
        }
    }

    /// Index of the bucket a `value` falls into.
    fn bucket_index(&self, value: f64) -> usize {
        self.bounds.partition_point(|&b| b < value)
    }

    /// Record one observation.
    pub fn record(&self, value: f64) {
        let idx = self.bucket_index(value);
        self.counts[idx].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        fetch_add_f64(&self.sum_bits, value);
        fetch_min_f64(&self.min_bits, value);
        fetch_max_f64(&self.max_bits, value);
    }

    /// Take an immutable snapshot of the current state.
    pub fn snapshot(&self) -> HistogramSnapshot {
        let count = self.count.load(Ordering::Relaxed);
        let bucket_counts = self
            .counts
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        let (min, max) = if count == 0 {
            (0.0, 0.0)
        } else {
            (
                f64::from_bits(self.min_bits.load(Ordering::Relaxed)),
                f64::from_bits(self.max_bits.load(Ordering::Relaxed)),
            )
        };
        HistogramSnapshot {
            count,
            sum: f64::from_bits(self.sum_bits.load(Ordering::Relaxed)),
            min,
            max,
            bounds: self.bounds.clone(),
            bucket_counts,
        }
    }
}

/// An immutable snapshot of a [`Histogram`].
///
/// `bucket_counts` has length `bounds.len() + 1`; the final entry is the
/// overflow bucket for observations greater than the last bound.
#[derive(Clone, Debug, PartialEq)]
pub struct HistogramSnapshot {
    /// Total number of recorded observations.
    pub count: u64,
    /// Sum of all recorded observations.
    pub sum: f64,
    /// Smallest observation (`0.0` when empty).
    pub min: f64,
    /// Largest observation (`0.0` when empty).
    pub max: f64,
    /// The finite bucket upper bounds, sorted ascending.
    pub bounds: Vec<f64>,
    /// Per-bucket counts, including the trailing overflow bucket.
    pub bucket_counts: Vec<u64>,
}

impl HistogramSnapshot {
    /// Arithmetic mean of all observations (`0.0` when empty).
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// Estimate the value at percentile `p` (where `p` is in `0.0..=1.0`).
    ///
    /// The estimate interpolates linearly within the bucket that contains the
    /// target rank. Finite buckets interpolate between their lower and upper
    /// bound; the overflow bucket returns the observed [`max`]. Returns `0.0`
    /// for an empty histogram.
    ///
    /// [`max`]: HistogramSnapshot::max
    pub fn percentile(&self, p: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let p = p.clamp(0.0, 1.0);
        let target = p * self.count as f64;
        let mut cum: u64 = 0;
        for (idx, &bucket) in self.bucket_counts.iter().enumerate() {
            let cum_before = cum;
            cum += bucket;
            if cum as f64 >= target {
                if idx >= self.bounds.len() {
                    return self.max;
                }
                if bucket == 0 {
                    return if idx == 0 { self.min } else { self.bounds[idx - 1] };
                }
                let lower = if idx == 0 { self.min } else { self.bounds[idx - 1] };
                let upper = self.bounds[idx];
                let pos = ((target - cum_before as f64) / bucket as f64).clamp(0.0, 1.0);
                return lower + (upper - lower) * pos;
            }
        }
        self.max
    }
}

/// An immutable snapshot of every metric held by a [`MetricRegistry`].
///
/// Entries are sorted by name within each category, giving deterministic output
/// for HUDs and tests.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RegistrySnapshot {
    /// Gauge name/value pairs.
    pub gauges: Vec<(String, f64)>,
    /// Counter name/value pairs.
    pub counters: Vec<(String, u64)>,
    /// Histogram name/snapshot pairs.
    pub histograms: Vec<(String, HistogramSnapshot)>,
}

/// A name-keyed registry of shared metric handles.
///
/// Call [`gauge`](MetricRegistry::gauge), [`counter`](MetricRegistry::counter),
/// or [`histogram`](MetricRegistry::histogram) to register-or-fetch a metric by
/// name. Each returns an [`Arc`] handle, so a subsystem registers once and then
/// updates the metric through the cheap atomic operations above. Lookups take a
/// read lock; first-time registration takes a brief write lock.
#[derive(Debug, Default)]
pub struct MetricRegistry {
    gauges: RwLock<BTreeMap<String, Arc<Gauge>>>,
    counters: RwLock<BTreeMap<String, Arc<Counter>>>,
    histograms: RwLock<BTreeMap<String, Arc<Histogram>>>,
}

impl MetricRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register-or-fetch a [`Gauge`] by name.
    pub fn gauge(&self, name: &str) -> Arc<Gauge> {
        if let Some(existing) = self
            .gauges
            .read()
            .expect("gauge registry lock poisoned")
            .get(name)
        {
            return Arc::clone(existing);
        }
        let mut map = self.gauges.write().expect("gauge registry lock poisoned");
        Arc::clone(
            map.entry(name.into())
                .or_insert_with(|| Arc::new(Gauge::new())),
        )
    }

    /// Register-or-fetch a [`Counter`] by name.
    pub fn counter(&self, name: &str) -> Arc<Counter> {
        if let Some(existing) = self
            .counters
            .read()
            .expect("counter registry lock poisoned")
            .get(name)
        {
            return Arc::clone(existing);
        }
        let mut map = self
            .counters
            .write()
            .expect("counter registry lock poisoned");
        Arc::clone(
            map.entry(name.into())
                .or_insert_with(|| Arc::new(Counter::new())),
        )
    }

    /// Register-or-fetch a [`Histogram`] by name.
    ///
    /// `bounds` is used only the first time a given name is registered;
    /// subsequent calls return the existing handle and ignore `bounds`.
    pub fn histogram(&self, name: &str, bounds: &[f64]) -> Arc<Histogram> {
        if let Some(existing) = self
            .histograms
            .read()
            .expect("histogram registry lock poisoned")
            .get(name)
        {
            return Arc::clone(existing);
        }
        let mut map = self
            .histograms
            .write()
            .expect("histogram registry lock poisoned");
        Arc::clone(
            map.entry(name.into())
                .or_insert_with(|| Arc::new(Histogram::new(bounds))),
        )
    }

    /// Take an immutable snapshot of every registered metric.
    pub fn snapshot(&self) -> RegistrySnapshot {
        let gauges = self
            .gauges
            .read()
            .expect("gauge registry lock poisoned")
            .iter()
            .map(|(name, g)| (name.clone(), g.get()))
            .collect();
        let counters = self
            .counters
            .read()
            .expect("counter registry lock poisoned")
            .iter()
            .map(|(name, c)| (name.clone(), c.get()))
            .collect();
        let histograms = self
            .histograms
            .read()
            .expect("histogram registry lock poisoned")
            .iter()
            .map(|(name, h)| (name.clone(), h.snapshot()))
            .collect();
        RegistrySnapshot {
            gauges,
            counters,
            histograms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn gauge_set_and_get() {
        let g = Gauge::new();
        assert_eq!(g.get(), 0.0);
        g.set(42.5);
        assert_eq!(g.get(), 42.5);
        g.set(-7.0);
        assert_eq!(g.get(), -7.0);
    }

    #[test]
    fn counter_monotonic_add_across_threads() {
        let counter = Arc::new(Counter::new());
        let threads = 8;
        let per_thread = 10_000u64;
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let c = Arc::clone(&counter);
                thread::spawn(move || {
                    for _ in 0..per_thread {
                        c.incr();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread panicked");
        }
        assert_eq!(counter.get(), threads as u64 * per_thread);
    }

    #[test]
    fn histogram_bucket_counts_and_percentiles() {
        let hist = Histogram::new(&[25.0, 50.0, 75.0, 100.0]);
        for v in 1..=100 {
            hist.record(v as f64);
        }
        let snap = hist.snapshot();
        assert_eq!(snap.count, 100);
        assert_eq!(snap.bucket_counts, vec![25, 25, 25, 25, 0]);
        assert_eq!(snap.min, 1.0);
        assert_eq!(snap.max, 100.0);
        assert!((snap.mean() - 50.5).abs() < 1e-9);
        assert!((snap.percentile(0.25) - 25.0).abs() < 1e-9);
        assert!((snap.percentile(0.5) - 50.0).abs() < 1e-9);
        assert!((snap.percentile(0.9) - 90.0).abs() < 1e-9);
    }

    #[test]
    fn histogram_overflow_bucket() {
        let hist = Histogram::new(&[10.0]);
        hist.record(5.0);
        hist.record(100.0);
        let snap = hist.snapshot();
        assert_eq!(snap.bucket_counts, vec![1, 1]);
        assert_eq!(snap.max, 100.0);
        assert_eq!(snap.percentile(1.0), 100.0);
    }

    #[test]
    fn registry_returns_shared_handles() {
        let reg = MetricRegistry::new();
        let a = reg.counter("frames");
        let b = reg.counter("frames");
        a.add(3);
        b.add(4);
        assert_eq!(reg.counter("frames").get(), 7);
        reg.gauge("mem_mb").set(512.0);
        let snap = reg.snapshot();
        assert_eq!(snap.counters, vec![("frames".to_string(), 7)]);
        assert_eq!(snap.gauges, vec![("mem_mb".to_string(), 512.0)]);
    }
}
