//! §24.7 release telemetry & privacy redaction tests: user-path stripping,
//! stable identifier hashing, `UTF-8`-safe truncation, allow/deny field
//! policy, field tiering + redacted-event building, canonical serialization +
//! digest, aggregation counting, and the deterministic sampler. Every
//! expectation is a hand-computed oracle over fixed inputs (pinned `FNV`-1a
//! digests + structural oracles), and the sensitive source text is asserted
//! absent from every redacted output.

use crate::telemetry::event::{
    build_event, EventAggregator, EventSchema, FieldTier, RedactedEvent,
};
use crate::telemetry::redact::{
    hash_identifier, redact_user_path, truncate_str, FieldDisposition, RedactionPolicy,
    TRUNCATION_MARKER,
};
use crate::telemetry::sampling::{SampleRatio, TelemetrySampler};

// ---- redact: user-path stripping --------------------------------------------

#[test]
fn redact_user_path_macos() {
    assert_eq!(
        redact_user_path("/Users/wangkailong/Documents/a.rs"),
        "/Users/redacted/Documents/a.rs"
    );
    // The original user name must not survive anywhere in the output.
    assert!(!redact_user_path("/Users/wangkailong/x").contains("wangkailong"));
}

#[test]
fn redact_user_path_linux() {
    assert_eq!(
        redact_user_path("/home/alice/.config/game/prefs"),
        "/home/redacted/.config/game/prefs"
    );
}

#[test]
fn redact_user_path_windows() {
    // Literal backslashes: `\Users\bob\AppData` -> `\Users\redacted\AppData`.
    assert_eq!(
        redact_user_path("\\Users\\bob\\AppData\\Local"),
        "\\Users\\redacted\\AppData\\Local"
    );
}

#[test]
fn redact_user_path_embedded_and_multiple() {
    // Embedded in a longer message, and more than one occurrence.
    assert_eq!(
        redact_user_path("at /Users/carol/proj/main.rs:10"),
        "at /Users/redacted/proj/main.rs:10"
    );
    assert_eq!(
        redact_user_path("/Users/a/x and /home/b/y"),
        "/Users/redacted/x and /home/redacted/y"
    );
}

#[test]
fn redact_user_path_no_segment_untouched() {
    // A marker with no following user-name segment is left as-is.
    assert_eq!(redact_user_path("/Users/"), "/Users/");
    // No trailing slash means no marker match at all.
    assert_eq!(redact_user_path("/Users"), "/Users");
    // Unrelated paths pass through unchanged.
    assert_eq!(redact_user_path("/opt/game/data"), "/opt/game/data");
}

#[test]
fn redact_user_path_non_ascii_username() {
    // A non-`ASCII` user name is replaced wholesale, and the result is valid.
    let out = redact_user_path("/Users/\u{7528}\u{6237}/doc.txt");
    assert_eq!(out, "/Users/redacted/doc.txt");
    assert!(!out.contains('\u{7528}'));
}

// ---- redact: identifier hashing ---------------------------------------------

#[test]
fn hash_identifier_pinned_vector() {
    // Pinned against an independent FNV-1a computation of "player-1234".
    assert_eq!(hash_identifier("player-1234"), "h:ff7321f61f1b6047");
}

#[test]
fn hash_identifier_deterministic_and_distinct() {
    assert_eq!(
        hash_identifier("session-abc"),
        hash_identifier("session-abc")
    );
    assert_ne!(
        hash_identifier("session-abc"),
        hash_identifier("session-abd")
    );
    // Always "h:" + 16 lowercase hex digits.
    let h = hash_identifier("anything");
    assert_eq!(h.len(), 18);
    assert!(h.starts_with("h:"));
    assert!(h[2..]
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
}

#[test]
fn hash_identifier_hides_source() {
    // The source identifier text must not leak into the token.
    let h = hash_identifier("player-1234");
    assert!(!h.contains("player"));
    assert!(!h.contains("1234"));
}

// ---- redact: truncation -----------------------------------------------------

#[test]
fn truncate_str_within_limit() {
    // Within the limit: unchanged, no marker.
    assert_eq!(truncate_str("hello", 10), "hello");
    assert_eq!(truncate_str("hello", 5), "hello");
}

#[test]
fn truncate_str_cuts_with_marker() {
    let mut expected = String::from("hel");
    expected.push(TRUNCATION_MARKER);
    assert_eq!(truncate_str("hello", 3), expected);
}

#[test]
fn truncate_str_multibyte_safe() {
    // "h", "é" are two chars; cutting at 2 keeps both and appends the marker.
    let mut expected = String::from("h\u{e9}");
    expected.push(TRUNCATION_MARKER);
    let out = truncate_str("h\u{e9}llo", 2);
    assert_eq!(out, expected);
    // Result is valid UTF-8 (a String always is) and did not split a char.
    assert_eq!(out.chars().count(), 3);
}

#[test]
fn truncate_str_zero_and_empty() {
    let marker: String = core::iter::once(TRUNCATION_MARKER).collect();
    assert_eq!(truncate_str("abc", 0), marker);
    // Empty input is empty regardless of limit (nothing to truncate).
    assert_eq!(truncate_str("", 5), "");
    assert_eq!(truncate_str("", 0), "");
}

// ---- redact: policy ---------------------------------------------------------

#[test]
fn policy_defaults() {
    let p = RedactionPolicy::new();
    assert_eq!(p.max_string_chars(), 256);
    assert!(p.strips_paths());
    assert_eq!(RedactionPolicy::default(), p);
}

#[test]
fn policy_classify_precedence() {
    // deny beats allow beats hash; everything else keeps.
    let p = RedactionPolicy::new()
        .deny_field("secret")
        .hash_field("user_id");
    assert_eq!(p.classify("secret"), FieldDisposition::Drop);
    assert_eq!(p.classify("user_id"), FieldDisposition::Hash);
    assert_eq!(p.classify("fps"), FieldDisposition::Keep);

    // deny wins even when the same key is also allow-listed.
    let p2 = RedactionPolicy::new().allow_field("x").deny_field("x");
    assert_eq!(p2.classify("x"), FieldDisposition::Drop);
}

#[test]
fn policy_allow_list_strict() {
    // A non-empty allow-list drops everything not on it.
    let p = RedactionPolicy::new().allow_field("fps").allow_field("ram");
    assert_eq!(p.classify("fps"), FieldDisposition::Keep);
    assert_eq!(p.classify("ram"), FieldDisposition::Keep);
    assert_eq!(p.classify("email"), FieldDisposition::Drop);
}

#[test]
fn policy_redact_field_keep_hash_drop() {
    let p = RedactionPolicy::new()
        .hash_field("user_id")
        .deny_field("email");
    // Keep + path strip.
    assert_eq!(
        p.redact_field("path", "/Users/alice/x").as_deref(),
        Some("/Users/redacted/x")
    );
    // Hash.
    assert_eq!(
        p.redact_field("user_id", "player-1234").as_deref(),
        Some("h:ff7321f61f1b6047")
    );
    // Drop.
    assert_eq!(p.redact_field("email", "a@b.com"), None);
}

#[test]
fn policy_redact_text_strip_toggle_and_truncate() {
    // Stripping on by default.
    assert_eq!(
        RedactionPolicy::new().redact_text("/home/bob/x"),
        "/home/redacted/x"
    );
    // Stripping off leaves the path (still truncated by the length cap).
    assert_eq!(
        RedactionPolicy::new()
            .with_strip_paths(false)
            .redact_text("/home/bob/x"),
        "/home/bob/x"
    );
    // Truncation cap applies to kept text.
    let mut expected = String::from("hel");
    expected.push(TRUNCATION_MARKER);
    assert_eq!(
        RedactionPolicy::new()
            .with_max_string_chars(3)
            .redact_text("hello"),
        expected
    );
}

// ---- event: schema tiering --------------------------------------------------

#[test]
fn schema_tiers_and_defaults() {
    let s = EventSchema::new("crash")
        .require("reason")
        .optional("build")
        .forbid("user_path");
    assert_eq!(s.name(), "crash");
    assert_eq!(s.tier_of("reason"), FieldTier::Required);
    assert_eq!(s.tier_of("build"), FieldTier::Optional);
    assert_eq!(s.tier_of("user_path"), FieldTier::Forbidden);
    // Undeclared fields are forbidden by default (strict whitelist).
    assert_eq!(s.tier_of("mystery"), FieldTier::Forbidden);
    assert_eq!(s.required_fields(), alloc::vec!["reason".to_string()]);
}

#[test]
fn schema_allow_unknown() {
    let s = EventSchema::new("e").allow_unknown();
    assert_eq!(s.tier_of("whatever"), FieldTier::Optional);
}

// ---- event: building --------------------------------------------------------

#[test]
fn build_event_basic_sorts_and_drops_forbidden() {
    let schema = EventSchema::new("crash")
        .require("reason")
        .optional("build");
    let policy = RedactionPolicy::new();
    // `user_path` is undeclared -> forbidden -> dropped.
    let ev = build_event(
        &schema,
        &policy,
        &[
            ("reason", "segv"),
            ("build", "1.2.3"),
            ("user_path", "/Users/alice/x"),
        ],
    );
    assert_eq!(
        ev.fields,
        alloc::vec![
            ("build".to_string(), "1.2.3".to_string()),
            ("reason".to_string(), "segv".to_string()),
        ]
    );
    assert_eq!(ev.dropped, alloc::vec!["user_path".to_string()]);
    assert!(ev.missing_required.is_empty());
    assert!(ev.is_complete());
    assert_eq!(ev.field("reason"), Some("segv"));
    assert_eq!(ev.field("absent"), None);
}

#[test]
fn build_event_canonical_and_digest_pinned() {
    let schema = EventSchema::new("crash")
        .require("reason")
        .optional("build");
    let ev = build_event(
        &schema,
        &RedactionPolicy::new(),
        &[("reason", "segv"), ("build", "1.2.3")],
    );
    assert_eq!(ev.canonical(), "crash|build=1.2.3|reason=segv");
    // Pinned FNV-1a digest of the canonical string.
    assert_eq!(ev.digest(), 0x3712_8671_7c15_120b);
}

#[test]
fn build_event_missing_required() {
    let schema = EventSchema::new("crash")
        .require("reason")
        .optional("build");
    let ev = build_event(&schema, &RedactionPolicy::new(), &[("build", "1.2.3")]);
    assert_eq!(ev.missing_required, alloc::vec!["reason".to_string()]);
    assert!(!ev.is_complete());
}

#[test]
fn build_event_policy_drop_and_hash() {
    let schema = EventSchema::new("crash")
        .require("reason")
        .optional("build")
        .optional("user_id");
    let policy = RedactionPolicy::new()
        .deny_field("build")
        .hash_field("user_id");
    let ev = build_event(
        &schema,
        &policy,
        &[
            ("reason", "segv"),
            ("build", "1.2.3"),
            ("user_id", "player-1234"),
        ],
    );
    // `build` present but policy-dropped; `user_id` hashed.
    assert!(ev.dropped.contains(&"build".to_string()));
    assert_eq!(ev.field("build"), None);
    assert_eq!(ev.field("user_id"), Some("h:ff7321f61f1b6047"));
    assert_eq!(ev.field("reason"), Some("segv"));
}

#[test]
fn build_event_strips_path_and_hides_pii() {
    let schema = EventSchema::new("open").optional("path").forbid("home");
    let ev = build_event(
        &schema,
        &RedactionPolicy::new(),
        &[("path", "/Users/alice/save.dat"), ("home", "/Users/alice")],
    );
    assert_eq!(ev.field("path"), Some("/Users/redacted/save.dat"));
    // The forbidden field and the user name are both absent from the output.
    assert_eq!(ev.field("home"), None);
    assert!(!ev.canonical().contains("alice"));
}

#[test]
fn canonical_escapes_delimiters() {
    let schema = EventSchema::new("e").allow_unknown();
    let ev = build_event(&schema, &RedactionPolicy::new(), &[("q", "a=b|c")]);
    // `=` -> `\e`, `|` -> `\p`.
    assert_eq!(ev.canonical(), "e|q=a\\eb\\pc");
    // Round-trip determinism: rebuilding yields the same digest.
    let ev2 = build_event(&schema, &RedactionPolicy::new(), &[("q", "a=b|c")]);
    assert_eq!(ev.digest(), ev2.digest());
}

// ---- event: aggregation -----------------------------------------------------

fn crash_event(reason: &str) -> RedactedEvent {
    let schema = EventSchema::new("crash").require("reason");
    build_event(&schema, &RedactionPolicy::new(), &[("reason", reason)])
}

#[test]
fn aggregator_counts_and_ranks() {
    let a = crash_event("segv");
    let b = crash_event("abrt");
    let mut agg = EventAggregator::new();
    agg.record(&a);
    agg.record(&a);
    agg.record(&b);

    assert_eq!(agg.total(), 3);
    assert_eq!(agg.len(), 2);
    assert_eq!(agg.count_of(&a), 2);
    assert_eq!(agg.count_of(&b), 1);

    let ranked = agg.ranked();
    // Descending count: the twice-seen "segv" signature first.
    assert_eq!(ranked[0].count, 2);
    assert_eq!(ranked[0].canonical, "crash|reason=segv");
    assert_eq!(ranked[0].digest, 0x3bca_2518_64c7_4c66);
    assert_eq!(ranked[1].count, 1);
    assert_eq!(ranked[1].canonical, "crash|reason=abrt");
}

#[test]
fn aggregator_empty() {
    let agg = EventAggregator::new();
    assert!(agg.is_empty());
    assert_eq!(agg.total(), 0);
    assert_eq!(agg.len(), 0);
}

// ---- sampling ---------------------------------------------------------------

#[test]
fn ratio_always_never() {
    let a = SampleRatio::always();
    assert!(a.admits(0));
    assert!(a.admits(5));
    assert!(a.admits(100));
    assert_eq!(a.fraction(), 1.0);

    let n = SampleRatio::never();
    assert!(!n.admits(0));
    assert!(!n.admits(7));
    assert_eq!(n.fraction(), 0.0);
}

#[test]
fn ratio_one_in_and_clamp() {
    let r = SampleRatio::one_in(4);
    assert_eq!((r.keep(), r.out_of()), (1, 4));
    assert!(r.admits(0));
    assert!(r.admits(4));
    assert!(r.admits(8));
    assert!(!r.admits(1));
    assert!(!r.admits(3));
    // `one_in(0)` clamps to a 1/1 keep-all.
    assert_eq!(SampleRatio::one_in(0), SampleRatio::always());
}

#[test]
fn ratio_ratio_clamp_and_fraction() {
    // out_of clamps to >= 1 and keep clamps to <= out_of.
    let clamped = SampleRatio::ratio(5, 3);
    assert_eq!((clamped.keep(), clamped.out_of()), (3, 3));
    assert_eq!(clamped.fraction(), 1.0);
    assert!(clamped.admits(0) && clamped.admits(2));
    // out_of == 0 clamps to 1, and keep then clamps to 1 -> keep-all 1/1.
    assert_eq!(SampleRatio::ratio(7, 0), SampleRatio::always());
    assert_eq!(SampleRatio::ratio(1, 4).fraction(), 0.25);
}

#[test]
fn ratio_admits_boundary() {
    // keep=73/100 admits residues 0..=72; keep=72/100 admits 0..=71.
    assert!(SampleRatio::ratio(73, 100).admits(72));
    assert!(!SampleRatio::ratio(72, 100).admits(72));
}

#[test]
fn ratio_expected_keep() {
    assert_eq!(SampleRatio::ratio(1, 4).expected_keep(100), 25);
    assert_eq!(SampleRatio::ratio(73, 100).expected_keep(1000), 730);
    assert_eq!(SampleRatio::always().expected_keep(42), 42);
    assert_eq!(SampleRatio::never().expected_keep(42), 0);
}

#[test]
fn sampler_bucket_pinned_and_salted() {
    let s = TelemetrySampler::new(SampleRatio::always());
    // Pinned bucket for ("crash", "sess-1").
    assert_eq!(s.bucket("crash", "sess-1"), 0x2075_b8dc_d282_adec);
    // Name-salting: the same key under a different event name differs.
    assert_ne!(
        s.bucket("crash", "sess-1"),
        s.bucket("frame_stats", "sess-1")
    );
}

#[test]
fn sampler_default_and_override_rates() {
    let s = TelemetrySampler::new(SampleRatio::one_in(10))
        .with_event_rate("crash", SampleRatio::always());
    assert_eq!(s.rate_for("crash"), SampleRatio::always());
    assert_eq!(s.rate_for("frame_stats"), SampleRatio::one_in(10));
}

#[test]
fn sampler_should_sample_deterministic() {
    let s = TelemetrySampler::new(SampleRatio::one_in(10))
        .with_event_rate("crash", SampleRatio::always());
    // crash is always kept.
    assert!(s.should_sample("crash", "sess-1"));
    // frame_stats under 1/10: bucket%10 == 0 keeps sess-1, == 9 drops sess-2.
    assert!(s.should_sample("frame_stats", "sess-1"));
    assert!(!s.should_sample("frame_stats", "sess-2"));
    // Repeat calls are stable.
    assert_eq!(
        s.should_sample("frame_stats", "sess-2"),
        s.should_sample("frame_stats", "sess-2")
    );
}

#[test]
fn sampler_decide_outcome() {
    let s = TelemetrySampler::new(SampleRatio::one_in(10));
    let out = s.decide("frame_stats", "sess-1");
    assert_eq!(out.bucket, 0x4439_0139_5af1_c2a2);
    assert_eq!(out.ratio, SampleRatio::one_in(10));
    assert!(out.kept); // bucket % 10 == 0 < 1
}

#[test]
fn sampler_sample_event() {
    let schema = EventSchema::new("frame_stats").optional("p99");
    let ev = build_event(&schema, &RedactionPolicy::new(), &[("p99", "16")]);
    let s = TelemetrySampler::new(SampleRatio::one_in(10));
    // sample_event routes through the event name.
    assert_eq!(
        s.sample_event(&ev, "sess-1"),
        s.should_sample("frame_stats", "sess-1")
    );
    assert!(s.sample_event(&ev, "sess-1"));
}
