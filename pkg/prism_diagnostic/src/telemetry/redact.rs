//! Privacy redaction primitives for release telemetry (§24.7).
//!
//! Release telemetry must never carry personally identifiable information
//! (`PII`) off the player's machine. This module owns the deterministic,
//! allocation-only transforms that scrub a field *value* before it is placed in
//! a telemetry event: absolute-path user-name stripping, stable non-crypto
//! identifier hashing, string truncation, and an allow-/deny-list field filter.
//!
//! Every transform is a pure function of its input (`no_std` + `alloc`, no
//! `unsafe`, no clock, no `RNG`): the same raw value always redacts to the same
//! output, and the sensitive source text is provably absent from the result
//! (the tests assert the original user name / identifier never survives). The
//! identifier hash reuses the crate's stable `FNV`-1a primitive
//! ([`crate::determinism::hash::fnv1a_64`]) so a hashed id is stable across
//! runs and platforms but reveals nothing about the source.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::string::{String, ToString};

use crate::determinism::hash::fnv1a_64;

/// The replacement token substituted for a stripped user-name path segment.
pub const REDACTED_SEGMENT: &str = "redacted";

/// The marker appended to a value that was truncated by [`truncate_str`].
pub const TRUNCATION_MARKER: char = '\u{2026}'; // horizontal ellipsis

/// Default maximum string length (in `char`s) before truncation kicks in.
pub const DEFAULT_MAX_STRING_CHARS: usize = 256;

/// Strip user-name segments out of absolute home-directory paths.
///
/// Scans `input` for the well-known home roots and replaces the single
/// path segment that follows each with [`REDACTED_SEGMENT`], leaving the rest
/// of the path intact so the structure is still useful for debugging:
///
/// - `macOS`: `/Users/<name>/...`   -> `/Users/redacted/...`
/// - `Linux`: `/home/<name>/...`    -> `/home/redacted/...`
/// - `Windows`: `\Users\<name>\...` -> `\Users\redacted\...`
///
/// The search runs anywhere in the string (a path embedded in a longer message
/// such as a stack frame is still scrubbed), replaces every occurrence, and is
/// `UTF-8`-safe (user names may be non-`ASCII`; only `ASCII` separators and
/// markers are matched on raw bytes). A marker with no following segment (the
/// path ends exactly at `/Users/`) is left untouched.
#[must_use]
pub fn redact_user_path(input: &str) -> String {
    // (marker, path separator that terminates the user-name segment)
    const MARKERS: [(&str, u8); 3] =
        [("/Users/", b'/'), ("/home/", b'/'), ("\\Users\\", b'\\')];

    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < bytes.len() {
        let mut matched = false;
        for (marker, sep) in MARKERS {
            let m = marker.as_bytes();
            if bytes[i..].starts_with(m) {
                let seg_start = i + m.len();
                // Only redact when a non-empty user-name segment follows.
                if seg_start < bytes.len() && bytes[seg_start] != sep {
                    out.push_str(marker);
                    out.push_str(REDACTED_SEGMENT);
                    // Advance past the user-name segment, leaving the trailing
                    // separator (or end-of-string) for the next iteration.
                    let mut j = seg_start;
                    while j < bytes.len() && bytes[j] != sep {
                        j += 1;
                    }
                    i = j;
                    matched = true;
                    break;
                }
            }
        }
        if !matched {
            // Copy exactly one `UTF-8` scalar so byte indices stay on a
            // character boundary.
            if let Some(ch) = input[i..].chars().next() {
                out.push(ch);
                i += ch.len_utf8();
            } else {
                break;
            }
        }
    }
    out
}

/// Hash an identifier into a stable, non-reversible token.
///
/// Returns `h:` followed by the 16-digit lowercase hex of the 64-bit `FNV`-1a
/// digest of `raw`. Deterministic across runs and platforms and non-crypto: it
/// is a one-way stand-in that lets the backend correlate events from the same
/// id without ever seeing the id itself. The same input always yields the same
/// token; distinct inputs almost always differ.
#[must_use]
pub fn hash_identifier(raw: &str) -> String {
    use core::fmt::Write as _;

    let mut out = String::with_capacity(2 + 16);
    // Writing to a `String` is infallible; the digest is a fixed 16 hex digits.
    let _ = write!(out, "h:{:016x}", fnv1a_64(raw.as_bytes()));
    out
}

/// Truncate `input` to at most `max_chars` characters.
///
/// Counting is by `UTF-8` scalar (not byte), so the result is always valid and
/// never splits a multi-byte character. When truncation occurs a
/// [`TRUNCATION_MARKER`] is appended so the backend can tell a value was cut.
/// A value already within the limit is returned unchanged (no marker). A
/// `max_chars` of `0` yields just the marker for any non-empty input.
#[must_use]
pub fn truncate_str(input: &str, max_chars: usize) -> String {
    let mut chars = input.chars();
    let mut kept: String = chars.by_ref().take(max_chars).collect();
    // If any character remains, the value was cut: flag it with the marker.
    if chars.next().is_some() {
        kept.push(TRUNCATION_MARKER);
    }
    kept
}

/// What a [`RedactionPolicy`] decides to do with a named field's value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldDisposition {
    /// Keep the value, applying path stripping + truncation per policy.
    Keep,
    /// Drop the field entirely (deny-listed, or not on a non-empty allow-list).
    Drop,
    /// Replace the value with its [`hash_identifier`] token.
    Hash,
}

/// A value-level redaction policy: field allow/deny filtering plus the
/// per-value transforms (path stripping, truncation, identifier hashing).
///
/// The field name decides the disposition ([`FieldDisposition`]); the value is
/// then transformed accordingly. Precedence is deny > allow > hash > keep: a
/// deny-listed key is always dropped, then (if the allow-list is non-empty)
/// only allow-listed keys survive, then hash-listed keys are hashed, and
/// everything else is kept with path+truncation applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RedactionPolicy {
    /// Keys that are always dropped.
    deny: BTreeSet<String>,
    /// When non-empty, only these keys are kept (strict allow-list).
    allow: BTreeSet<String>,
    /// Keys whose value is replaced by its stable hash token.
    hash: BTreeSet<String>,
    /// Maximum kept-value length in `char`s before truncation.
    max_string_chars: usize,
    /// Whether to run [`redact_user_path`] on kept string values.
    strip_paths: bool,
}

impl RedactionPolicy {
    /// A policy with no filters, path stripping on, and the default length cap.
    #[must_use]
    pub fn new() -> Self {
        Self {
            deny: BTreeSet::new(),
            allow: BTreeSet::new(),
            hash: BTreeSet::new(),
            max_string_chars: DEFAULT_MAX_STRING_CHARS,
            strip_paths: true,
        }
    }

    /// Add a key to the deny-list (always dropped).
    #[must_use]
    pub fn deny_field(mut self, key: &str) -> Self {
        self.deny.insert(key.to_string());
        self
    }

    /// Add a key to the allow-list. Once any key is allow-listed, every key not
    /// on the list is dropped (strict whitelist).
    #[must_use]
    pub fn allow_field(mut self, key: &str) -> Self {
        self.allow.insert(key.to_string());
        self
    }

    /// Add a key whose value must be hashed via [`hash_identifier`].
    #[must_use]
    pub fn hash_field(mut self, key: &str) -> Self {
        self.hash.insert(key.to_string());
        self
    }

    /// Set the maximum kept-value length (in `char`s) before truncation.
    #[must_use]
    pub fn with_max_string_chars(mut self, max_chars: usize) -> Self {
        self.max_string_chars = max_chars;
        self
    }

    /// Enable or disable user-path stripping on kept string values.
    #[must_use]
    pub fn with_strip_paths(mut self, strip: bool) -> Self {
        self.strip_paths = strip;
        self
    }

    /// The configured maximum kept-value length, in `char`s.
    #[must_use]
    pub const fn max_string_chars(&self) -> usize {
        self.max_string_chars
    }

    /// Whether user-path stripping is enabled.
    #[must_use]
    pub const fn strips_paths(&self) -> bool {
        self.strip_paths
    }

    /// Decide what to do with the named field (see precedence on the type doc).
    #[must_use]
    pub fn classify(&self, key: &str) -> FieldDisposition {
        if self.deny.contains(key) {
            return FieldDisposition::Drop;
        }
        if !self.allow.is_empty() && !self.allow.contains(key) {
            return FieldDisposition::Drop;
        }
        if self.hash.contains(key) {
            return FieldDisposition::Hash;
        }
        FieldDisposition::Keep
    }

    /// Apply the policy to one field, returning the redacted value or `None`
    /// when the field is dropped.
    #[must_use]
    pub fn redact_field(&self, key: &str, value: &str) -> Option<String> {
        match self.classify(key) {
            FieldDisposition::Drop => None,
            FieldDisposition::Hash => Some(hash_identifier(value)),
            FieldDisposition::Keep => Some(self.redact_text(value)),
        }
    }

    /// Apply the kept-value transforms (path stripping then truncation) to a
    /// raw string, independent of any field name.
    #[must_use]
    pub fn redact_text(&self, value: &str) -> String {
        let stripped = if self.strip_paths {
            redact_user_path(value)
        } else {
            value.to_string()
        };
        truncate_str(&stripped, self.max_string_chars)
    }
}

impl Default for RedactionPolicy {
    fn default() -> Self {
        Self::new()
    }
}
