//! Deterministic telemetry sampling (§24.7).
//!
//! Release telemetry has a bandwidth budget: not every event can be uploaded.
//! The sampling decision here is *deterministic* — never a real `RNG` — so it
//! is reproducible, testable, and consistent across a session. A `keep/out_of`
//! ratio ([`SampleRatio`]) is applied to a bucket derived by stably hashing the
//! event name together with a stable key (a session id, event digest, ...), so
//! the same key under the same event name always lands in the same bucket and
//! the keep/drop choice is fixed.
//!
//! Salting the bucket with the event name means one session being sampled-in
//! for event `A` tells you nothing about whether it is sampled-in for event
//! `B`: each event stream samples independently. [`TelemetrySampler`] carries a
//! default ratio plus per-event overrides so release builds can keep crashes at
//! full resolution while thinning high-frequency frame stats.
//!
//! Pure `core`/`alloc` integer arithmetic: deterministic, `no_std` + `alloc`,
//! no `unsafe`, no clock, no `RNG`.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};

use crate::determinism::hash::StateHasher;
use crate::telemetry::event::RedactedEvent;

/// A `keep/out_of` sampling ratio, e.g. keep 1 of every 100 events.
///
/// Invariants (enforced by the constructors): `out_of >= 1` and
/// `keep <= out_of`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleRatio {
    /// How many of each `out_of` window are kept.
    keep: u32,
    /// Window size.
    out_of: u32,
}

impl SampleRatio {
    /// Keep everything (`1/1`).
    #[must_use]
    pub const fn always() -> Self {
        Self { keep: 1, out_of: 1 }
    }

    /// Drop everything (`0/1`).
    #[must_use]
    pub const fn never() -> Self {
        Self { keep: 0, out_of: 1 }
    }

    /// Keep `1` of every `n` (clamped so `n >= 1`).
    #[must_use]
    pub fn one_in(n: u32) -> Self {
        Self {
            keep: 1,
            out_of: n.max(1),
        }
    }

    /// Keep `keep` of every `out_of` (clamped: `out_of >= 1`, `keep <= out_of`).
    #[must_use]
    pub fn ratio(keep: u32, out_of: u32) -> Self {
        let out_of = out_of.max(1);
        Self {
            keep: keep.min(out_of),
            out_of,
        }
    }

    /// The numerator (events kept per window).
    #[must_use]
    pub const fn keep(&self) -> u32 {
        self.keep
    }

    /// The denominator (window size).
    #[must_use]
    pub const fn out_of(&self) -> u32 {
        self.out_of
    }

    /// The keep fraction in `[0, 1]`.
    #[must_use]
    pub fn fraction(&self) -> f64 {
        f64::from(self.keep) / f64::from(self.out_of)
    }

    /// Whether a given bucket is admitted (kept).
    ///
    /// Admits exactly `keep` of the `out_of` residues, so over uniformly
    /// distributed buckets the keep rate converges to [`fraction`](Self::fraction).
    #[must_use]
    pub fn admits(&self, bucket: u64) -> bool {
        (bucket % u64::from(self.out_of)) < u64::from(self.keep)
    }

    /// Expected number of kept events out of `seen` (integer floor).
    #[must_use]
    pub fn expected_keep(&self, seen: u64) -> u64 {
        seen.saturating_mul(u64::from(self.keep)) / u64::from(self.out_of)
    }
}

/// The outcome of a sampling decision, retained for observability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleOutcome {
    /// Whether the event is kept.
    pub kept: bool,
    /// The derived bucket (name-salted key hash reduced to `out_of`).
    pub bucket: u64,
    /// The ratio that was applied.
    pub ratio: SampleRatio,
}

/// A deterministic sampler with a default ratio and per-event overrides.
#[derive(Clone, Debug)]
pub struct TelemetrySampler {
    /// Ratio applied to events without a specific override.
    default_rate: SampleRatio,
    /// Per-event-name ratio overrides.
    per_event: BTreeMap<String, SampleRatio>,
}

impl TelemetrySampler {
    /// A sampler that applies `default_rate` to every event.
    #[must_use]
    pub fn new(default_rate: SampleRatio) -> Self {
        Self {
            default_rate,
            per_event: BTreeMap::new(),
        }
    }

    /// Override the ratio for a specific event name.
    #[must_use]
    pub fn with_event_rate(mut self, name: &str, rate: SampleRatio) -> Self {
        self.per_event.insert(name.to_string(), rate);
        self
    }

    /// The ratio that applies to `name` (its override, else the default).
    #[must_use]
    pub fn rate_for(&self, name: &str) -> SampleRatio {
        self.per_event
            .get(name)
            .copied()
            .unwrap_or(self.default_rate)
    }

    /// Derive the deterministic bucket for `(name, key)`.
    ///
    /// Folds the event name, a separator byte, then the key through the stable
    /// `FNV`-1a [`StateHasher`]. Salting by name keeps event streams
    /// independent; the key (session id / event digest / ...) selects the
    /// residue within a stream.
    #[must_use]
    pub fn bucket(&self, name: &str, key: &str) -> u64 {
        let mut hasher = StateHasher::new();
        hasher.write_bytes(name.as_bytes());
        hasher.write_u8(0);
        hasher.write_bytes(key.as_bytes());
        hasher.finish()
    }

    /// Whether `(name, key)` is sampled in under the applicable ratio.
    #[must_use]
    pub fn should_sample(&self, name: &str, key: &str) -> bool {
        let ratio = self.rate_for(name);
        ratio.admits(self.bucket(name, key))
    }

    /// A full [`SampleOutcome`] for `(name, key)`.
    #[must_use]
    pub fn decide(&self, name: &str, key: &str) -> SampleOutcome {
        let ratio = self.rate_for(name);
        let bucket = self.bucket(name, key);
        SampleOutcome {
            kept: ratio.admits(bucket),
            bucket,
            ratio,
        }
    }

    /// Convenience: sample a built [`RedactedEvent`] by `key`.
    #[must_use]
    pub fn sample_event(&self, event: &RedactedEvent, key: &str) -> bool {
        self.should_sample(&event.name, key)
    }
}
