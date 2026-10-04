//! Telemetry event model: field tiering, redacted events, canonical
//! serialization, and aggregation counting (§24.7).
//!
//! A telemetry event is a named bag of key/value fields (crash reason, frame
//! `p99`, memory peak, build id, ...). Before it leaves the process it is run
//! through two orthogonal gates:
//!
//! 1. **Field tiering** ([`EventSchema`] / [`FieldTier`]): each field is
//!    declared *required* (must be present), *optional* (collected if present),
//!    or *forbidden* (never collected). Unknown fields default to forbidden, so
//!    the schema is a strict collection whitelist.
//! 2. **Value redaction** ([`crate::telemetry::RedactionPolicy`]): surviving
//!    values are path-stripped, truncated, hashed, or dropped.
//!
//! The product is a [`RedactedEvent`] whose fields are sorted by key so the
//! [`canonical`](RedactedEvent::canonical) serialization — and therefore its
//! [`digest`](RedactedEvent::digest) — is identical for identical inputs on
//! every run and platform. [`EventAggregator`] folds a stream of redacted
//! events into per-digest counts for low-bandwidth release reporting.
//!
//! Pure `core`/`alloc` integer arithmetic: deterministic, `no_std` + `alloc`,
//! no `unsafe`, no clock, no `RNG`.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::determinism::hash::fnv1a_64;
use crate::telemetry::redact::RedactionPolicy;

/// Collection tier of a telemetry field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldTier {
    /// Must be present; a missing required field is reported, not fabricated.
    Required,
    /// Collected when supplied, ignored when absent.
    Optional,
    /// Never collected; a supplied forbidden field is dropped and audited.
    Forbidden,
}

/// A per-event-name schema mapping field names to their [`FieldTier`].
///
/// Unknown fields (not declared on the schema) default to
/// [`FieldTier::Forbidden`]: the schema is a strict collection whitelist. Call
/// [`allow_unknown`](Self::allow_unknown) to instead treat undeclared fields as
/// optional.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventSchema {
    /// The event name this schema governs.
    name: String,
    /// Declared field tiers.
    tiers: BTreeMap<String, FieldTier>,
    /// Tier applied to fields absent from `tiers`.
    unknown_tier: FieldTier,
}

impl EventSchema {
    /// A schema for `name` with no declared fields (everything undeclared is
    /// forbidden until declared otherwise).
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            tiers: BTreeMap::new(),
            unknown_tier: FieldTier::Forbidden,
        }
    }

    /// Declare `key` as [`FieldTier::Required`].
    #[must_use]
    pub fn require(mut self, key: &str) -> Self {
        self.tiers.insert(key.to_string(), FieldTier::Required);
        self
    }

    /// Declare `key` as [`FieldTier::Optional`].
    #[must_use]
    pub fn optional(mut self, key: &str) -> Self {
        self.tiers.insert(key.to_string(), FieldTier::Optional);
        self
    }

    /// Declare `key` as [`FieldTier::Forbidden`].
    #[must_use]
    pub fn forbid(mut self, key: &str) -> Self {
        self.tiers.insert(key.to_string(), FieldTier::Forbidden);
        self
    }

    /// Treat undeclared fields as [`FieldTier::Optional`] instead of forbidden.
    #[must_use]
    pub fn allow_unknown(mut self) -> Self {
        self.unknown_tier = FieldTier::Optional;
        self
    }

    /// The event name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The tier of `key` (the undeclared default when not declared).
    #[must_use]
    pub fn tier_of(&self, key: &str) -> FieldTier {
        self.tiers.get(key).copied().unwrap_or(self.unknown_tier)
    }

    /// The declared required field names, in sorted order.
    #[must_use]
    pub fn required_fields(&self) -> Vec<String> {
        self.tiers
            .iter()
            .filter(|&(_, &tier)| tier == FieldTier::Required)
            .map(|(k, _)| k.clone())
            .collect()
    }
}

/// A telemetry event after tiering + redaction, ready to serialize/aggregate.
///
/// `fields` are sorted by key; `dropped` and `missing_required` are sorted and
/// de-duplicated. The source `PII` is guaranteed absent from `fields` — it was
/// either hashed, path-stripped, truncated, or dropped upstream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RedactedEvent {
    /// The event name.
    pub name: String,
    /// Surviving fields, sorted by key.
    pub fields: Vec<(String, String)>,
    /// Field names that were present but removed (forbidden or policy-dropped).
    pub dropped: Vec<String>,
    /// Required field names that were not supplied.
    pub missing_required: Vec<String>,
}

impl RedactedEvent {
    /// Whether every required field was present (no missing requireds).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.missing_required.is_empty()
    }

    /// Look up a surviving field's redacted value.
    #[must_use]
    pub fn field(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// A deterministic, reversible canonical string.
    ///
    /// Form: `name` then, for each sorted field, `|key=value`. The name, keys,
    /// and values are escaped so `\`, `|`, and `=` are unambiguous
    /// (`\` -> `\\`, `|` -> `\p`, `=` -> `\e`). Identical events always produce
    /// an identical canonical string.
    #[must_use]
    pub fn canonical(&self) -> String {
        let mut out = String::new();
        escape_into(&mut out, &self.name);
        for (k, v) in &self.fields {
            out.push('|');
            escape_into(&mut out, k);
            out.push('=');
            escape_into(&mut out, v);
        }
        out
    }

    /// A stable 64-bit `FNV`-1a digest of [`canonical`](Self::canonical), used
    /// as the aggregation / de-duplication key.
    #[must_use]
    pub fn digest(&self) -> u64 {
        fnv1a_64(self.canonical().as_bytes())
    }
}

/// Escape `s` into `out`, making `\`, `|`, and `=` unambiguous.
fn escape_into(out: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '|' => out.push_str("\\p"),
            '=' => out.push_str("\\e"),
            _ => out.push(ch),
        }
    }
}

/// Build a [`RedactedEvent`] from raw fields via a schema + redaction policy.
///
/// For each raw `(key, value)`: a [`FieldTier::Forbidden`] field is dropped;
/// otherwise the [`RedactionPolicy`] decides to drop, hash, or keep (with
/// path-strip + truncation). Required fields that never appear are collected
/// into [`RedactedEvent::missing_required`] rather than invented. The output's
/// `fields`/`dropped`/`missing_required` are sorted for determinism.
#[must_use]
pub fn build_event(
    schema: &EventSchema,
    policy: &RedactionPolicy,
    raw: &[(&str, &str)],
) -> RedactedEvent {
    let mut fields: Vec<(String, String)> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();
    let mut present: BTreeMap<String, ()> = BTreeMap::new();

    for &(key, value) in raw {
        if schema.tier_of(key) == FieldTier::Forbidden {
            dropped.push(key.to_string());
            continue;
        }
        present.insert(key.to_string(), ());
        match policy.redact_field(key, value) {
            Some(redacted) => fields.push((key.to_string(), redacted)),
            None => dropped.push(key.to_string()),
        }
    }

    let mut missing_required: Vec<String> = schema
        .required_fields()
        .into_iter()
        .filter(|k| !present.contains_key(k))
        .collect();

    fields.sort_by(|a, b| a.0.cmp(&b.0));
    dropped.sort();
    dropped.dedup();
    missing_required.sort();
    missing_required.dedup();

    RedactedEvent {
        name: schema.name().to_string(),
        fields,
        dropped,
        missing_required,
    }
}

/// One aggregated bucket: a distinct redacted event and how many times it
/// occurred.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregatedEvent {
    /// The [`RedactedEvent::digest`] identifying this bucket.
    pub digest: u64,
    /// The event name.
    pub name: String,
    /// The canonical string of the first event folded into this bucket.
    pub canonical: String,
    /// Number of events folded into this bucket.
    pub count: u64,
}

/// Folds a stream of [`RedactedEvent`]s into per-digest counts.
///
/// Identical events (same canonical string) share one bucket, so a release
/// build can ship a compact "this crash signature happened N times" summary
/// instead of N copies. Iteration order of the output is deterministic.
#[derive(Clone, Debug, Default)]
pub struct EventAggregator {
    /// Buckets keyed by digest.
    buckets: BTreeMap<u64, AggregatedEvent>,
    /// Total events recorded across all buckets.
    total: u64,
}

impl EventAggregator {
    /// An empty aggregator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buckets: BTreeMap::new(),
            total: 0,
        }
    }

    /// Fold one event in, incrementing its bucket (creating it on first sight).
    pub fn record(&mut self, event: &RedactedEvent) {
        let digest = event.digest();
        self.total += 1;
        self.buckets
            .entry(digest)
            .and_modify(|b| b.count += 1)
            .or_insert_with(|| AggregatedEvent {
                digest,
                name: event.name.clone(),
                canonical: event.canonical(),
                count: 1,
            });
    }

    /// Total number of events recorded.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Number of distinct buckets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buckets.len()
    }

    /// Whether no events have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    /// The bucket count for a specific event, if present.
    #[must_use]
    pub fn count_of(&self, event: &RedactedEvent) -> u64 {
        self.buckets.get(&event.digest()).map_or(0, |b| b.count)
    }

    /// Buckets ordered by descending count, then ascending digest (a stable,
    /// deterministic "top signatures" ordering).
    #[must_use]
    pub fn ranked(&self) -> Vec<AggregatedEvent> {
        let mut out: Vec<AggregatedEvent> = self.buckets.values().cloned().collect();
        out.sort_by(|a, b| b.count.cmp(&a.count).then(a.digest.cmp(&b.digest)));
        out
    }
}
