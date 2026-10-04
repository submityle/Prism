//! Instrumented-span + sampling fusion overlay (§24.5).
//!
//! Instrumented spans are *precise* for the code that was instrumented;
//! sampling is *statistical* but covers everything. Fusing them onto one
//! surface gives the best of both: the exact cost of key spans plus the
//! sampled global distribution, in a single flame-graph-style table.
//!
//! [`fuse`] joins a sampled [`FlatProfile`] (resolved to names through the
//! [`SymbolTable`]) with a set of [`InstrumentedSpan`]s keyed by the same
//! names. Each resulting [`FusedEntry`] carries both the sampled estimate and
//! the instrumented measurement when available, tags its [`FusionSource`], and
//! for frames present in both reports computes an agreement ratio
//! (sampled ÷ instrumented) that flags statistical disagreement — a sampled
//! estimate wildly off its instrumented truth means too few samples or a
//! mis-attributed stack. Pure `core`/`alloc`, no `unsafe`.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use super::fold::FlatProfile;
use super::symbol::SymbolTable;

/// A precise instrumented span measurement to overlay onto the sampled profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstrumentedSpan {
    /// Span name; must match the sampled frame name to fuse onto one entry.
    pub name: String,
    /// Measured inclusive (wall) time across the window, in nanoseconds.
    pub inclusive_nanos: u64,
    /// Measured exclusive (self) time across the window, in nanoseconds.
    pub self_nanos: u64,
    /// Number of times the span was entered in the window.
    pub call_count: u64,
}

impl InstrumentedSpan {
    /// Construct an instrumented span measurement.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        inclusive_nanos: u64,
        self_nanos: u64,
        call_count: u64,
    ) -> Self {
        Self {
            name: name.into(),
            inclusive_nanos,
            self_nanos,
            call_count,
        }
    }
}

/// Where a [`FusedEntry`]'s data came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionSource {
    /// Only an instrumented span (sampling never caught this frame).
    InstrumentedOnly,
    /// Only sampling (the frame was not instrumented).
    SampledOnly,
    /// Both an instrumented span and sampled hits — the overlap.
    Both,
}

/// One frame on the fused flame-graph surface.
#[derive(Clone, Debug, PartialEq)]
pub struct FusedEntry {
    /// The frame name.
    pub name: String,
    /// Sampled inclusive estimate (nanoseconds); `0` if sampling missed it.
    pub sampled_inclusive_nanos: u64,
    /// Sampled self estimate (nanoseconds); `0` if sampling missed it.
    pub sampled_self_nanos: u64,
    /// Instrumented inclusive measurement, if the frame was instrumented.
    pub instrumented_inclusive_nanos: Option<u64>,
    /// Instrumented self measurement, if the frame was instrumented.
    pub instrumented_self_nanos: Option<u64>,
    /// Instrumented call count, if the frame was instrumented.
    pub call_count: Option<u64>,
    /// Where this entry's data came from.
    pub source: FusionSource,
    /// For [`FusionSource::Both`], `sampled_inclusive ÷ instrumented_inclusive`
    /// (`1.0` is perfect agreement); `None` otherwise or when the instrumented
    /// value is `0`.
    pub agreement_ratio: Option<f64>,
}

impl FusedEntry {
    /// The best inclusive estimate: the instrumented measurement when present,
    /// otherwise the sampled estimate. Used as the ranking key.
    #[inline]
    #[must_use]
    pub fn effective_inclusive_nanos(&self) -> u64 {
        self.instrumented_inclusive_nanos
            .unwrap_or(self.sampled_inclusive_nanos)
    }
}

/// A fused flame-graph surface: instrumented + sampled frames joined by name,
/// sorted by effective inclusive time descending (ties by name ascending).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FusedProfile {
    /// Per-frame fused entries, hottest effective inclusive time first.
    entries: Vec<FusedEntry>,
}

impl FusedProfile {
    /// Borrow the fused entries (hottest effective inclusive time first).
    #[inline]
    #[must_use]
    pub fn entries(&self) -> &[FusedEntry] {
        &self.entries
    }

    /// Number of fused entries.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the fused surface is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a fused entry by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&FusedEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Entries whose sampled/instrumented inclusive estimates disagree by more
    /// than `tolerance` (e.g. `0.25` for ±25%). Only [`FusionSource::Both`]
    /// entries with a computed ratio are considered.
    #[must_use]
    pub fn disagreements(&self, tolerance: f64) -> Vec<&FusedEntry> {
        self.entries
            .iter()
            .filter(|e| {
                e.agreement_ratio
                    .is_some_and(|ratio| (ratio - 1.0).abs() > tolerance)
            })
            .collect()
    }
}

/// Fuse a sampled [`FlatProfile`] (resolved via `symbols`) with instrumented
/// spans into a single [`FusedProfile`].
///
/// Frames are joined by name: a sampled frame and an instrumented span with the
/// same name merge into one [`FusionSource::Both`] entry; unmatched frames
/// become [`FusionSource::SampledOnly`] or [`FusionSource::InstrumentedOnly`].
#[must_use]
pub fn fuse(
    flat: &FlatProfile,
    symbols: &SymbolTable,
    spans: &[InstrumentedSpan],
) -> FusedProfile {
    let interval = flat.interval_nanos();
    // Name-keyed accumulator so sampled + instrumented sides merge cleanly, in
    // deterministic (lexicographic) order before the final sort.
    let mut by_name: BTreeMap<String, FusedEntry> = BTreeMap::new();

    for row in flat.rows() {
        let Some(name) = symbols.resolve(row.frame) else {
            continue;
        };
        by_name.insert(
            String::from(name),
            FusedEntry {
                name: String::from(name),
                sampled_inclusive_nanos: row.inclusive_samples.saturating_mul(interval),
                sampled_self_nanos: row.self_samples.saturating_mul(interval),
                instrumented_inclusive_nanos: None,
                instrumented_self_nanos: None,
                call_count: None,
                source: FusionSource::SampledOnly,
                agreement_ratio: None,
            },
        );
    }

    for span in spans {
        match by_name.get_mut(&span.name) {
            Some(entry) => {
                entry.instrumented_inclusive_nanos = Some(span.inclusive_nanos);
                entry.instrumented_self_nanos = Some(span.self_nanos);
                entry.call_count = Some(span.call_count);
                entry.source = FusionSource::Both;
                entry.agreement_ratio = if span.inclusive_nanos == 0 {
                    None
                } else {
                    Some(entry.sampled_inclusive_nanos as f64 / span.inclusive_nanos as f64)
                };
            }
            None => {
                by_name.insert(
                    span.name.clone(),
                    FusedEntry {
                        name: span.name.clone(),
                        sampled_inclusive_nanos: 0,
                        sampled_self_nanos: 0,
                        instrumented_inclusive_nanos: Some(span.inclusive_nanos),
                        instrumented_self_nanos: Some(span.self_nanos),
                        call_count: Some(span.call_count),
                        source: FusionSource::InstrumentedOnly,
                        agreement_ratio: None,
                    },
                );
            }
        }
    }

    let mut entries: Vec<FusedEntry> = by_name.into_values().collect();
    entries.sort_by(|a, b| {
        b.effective_inclusive_nanos()
            .cmp(&a.effective_inclusive_nanos())
            .then_with(|| a.name.cmp(&b.name))
    });

    FusedProfile { entries }
}
