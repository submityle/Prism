//! §24.7 release telemetry & privacy redaction — deterministic core.
//!
//! A shipped build still needs to phone home the few signals that keep a live
//! game healthy — crash rate, frame-time `p99`, memory peak — but it must do so
//! without ever carrying personally identifiable information (`PII`) off the
//! player's machine. This module owns the deterministic, pure data model behind
//! that: how a telemetry event is tiered, scrubbed, canonicalized, counted, and
//! sampled. The actual network upload (and user-consent gating at the OS layer)
//! is upper-layer wiring; this layer makes every redaction and sampling result
//! a pure, testable function of its input.
//!
//! It delivers the §24.7 pieces as pure `core`/`alloc` integer arithmetic
//! (deterministic, `no_std` + `alloc`, no `unsafe`, no clock, no `RNG`), always
//! compiled regardless of crate features:
//!
//! 1. **Privacy redaction** ([`redact`]): [`redact_user_path`] strips the
//!    user-name segment out of absolute home paths, [`hash_identifier`] turns an
//!    id into a stable one-way token (reusing the crate's `FNV`-1a primitive),
//!    [`truncate_str`] bounds string length `UTF-8`-safely, and
//!    [`RedactionPolicy`] combines allow/deny field filtering with those
//!    per-value transforms.
//! 2. **Event model** ([`event`]): [`EventSchema`] tiers each field
//!    required/optional/forbidden (unknown fields forbidden by default), and
//!    [`build_event`] produces a [`RedactedEvent`] with key-sorted fields,
//!    a deterministic [`canonical`](RedactedEvent::canonical) form + 64-bit
//!    [`digest`](RedactedEvent::digest), and an [`EventAggregator`] that folds
//!    identical events into per-signature counts.
//! 3. **Deterministic sampling** ([`sampling`]): a [`SampleRatio`] applied to a
//!    name-salted key hash so the keep/drop choice is reproducible, with a
//!    [`TelemetrySampler`] carrying a default rate plus per-event overrides.
//!
//! Determinism + privacy are the twin contracts: identical input events always
//! redact, serialize, and sample identically, and the sensitive source text is
//! provably absent from the output (asserted by the tests).
//!
//! Honest boundary: real network upload / backend ingestion and user-consent
//! acquisition are upper-layer (`prism_platform` §16 produces the raw dump and
//! owns the consent prompt); offline crash-stack symbolication happens in the
//! backend. This layer owns only the deterministic redaction + event-building +
//! sampling data model, all oracle-checked offline.

pub mod event;
pub mod redact;
pub mod sampling;

pub use event::{
    build_event, AggregatedEvent, EventAggregator, EventSchema, FieldTier, RedactedEvent,
};
pub use redact::{
    hash_identifier, redact_user_path, truncate_str, FieldDisposition, RedactionPolicy,
    DEFAULT_MAX_STRING_CHARS, REDACTED_SEGMENT, TRUNCATION_MARKER,
};
pub use sampling::{SampleOutcome, SampleRatio, TelemetrySampler};
