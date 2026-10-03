//! M6 tests: [`JobTrace`](crate::JobTrace) recording, steal/occupancy counters,
//! and (behind the `trace` feature) the chrome-tracing JSON export.
//!
//! Anti-vacuous contract: the trace records exactly one span per instrumented
//! job, the counters are internally consistent (`migrated <= total`,
//! `steal_rate` in `0.0..=1.0`, per-bucket jobs sum to the total), and the
//! chrome JSON actually parses into an array of `ph:"X"` complete events whose
//! names match the recorded spans.

use crate::TaskPool;
#[cfg(feature = "trace")]
use alloc::string::String;
#[cfg(feature = "trace")]
use alloc::vec::Vec;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn trace_records_one_span_per_instrumented_job() {
    let pool = TaskPool::with_threads(4);
    let trace = pool.new_job_trace();
    let n = 200usize;
    let ran = AtomicUsize::new(0);
    pool.scope(|s| {
        for i in 0..n {
            let job = trace.instrument("job", {
                let ran = &ran;
                move || {
                    ran.fetch_add(i & 1, Ordering::Relaxed);
                }
            });
            s.spawn(job);
        }
    });
    assert_eq!(trace.span_count(), n, "one span per instrumented job");
    assert_eq!(trace.total_jobs(), n as u64);
}

#[test]
fn trace_counters_are_consistent() {
    let pool = TaskPool::with_threads(4);
    let trace = pool.new_job_trace();
    let n = 500usize;
    pool.scope(|s| {
        for _ in 0..n {
            let job = trace.instrument("spin", || {
                // A little real work so durations are non-trivial.
                let mut acc = 0u64;
                for k in 0..256u64 {
                    acc = acc.wrapping_add(k);
                }
                core::hint::black_box(acc);
            });
            s.spawn(job);
        }
    });

    let total = trace.total_jobs();
    assert_eq!(total, n as u64);

    // Per-bucket executed counts must sum to the total.
    let per_bucket: u64 = (0..trace.bucket_count()).map(|b| trace.jobs_on(b)).sum();
    assert_eq!(per_bucket, total);

    // Migration count cannot exceed the total, and the steal rate is a fraction.
    assert!(trace.migrated_jobs() <= total, "migrated must not exceed total");
    let rate = trace.steal_rate();
    assert!((0.0..=1.0).contains(&rate), "steal_rate out of range: {rate}");

    // Occupancy of every bucket is a clamped fraction.
    for b in 0..trace.bucket_count() {
        let occ = trace.occupancy(b);
        assert!((0.0..=1.0).contains(&occ), "occupancy out of range: {occ}");
    }
}

#[test]
fn trace_captures_parent_for_nested_spans() {
    let pool = TaskPool::with_threads(2);
    let trace = pool.new_job_trace();
    // Record an outer span that, inside, records an inner span. The inner span
    // must point at the outer span as its parent.
    trace.record("outer", || {
        trace.record("inner", || {});
    });
    let spans = trace.spans();
    assert_eq!(spans.len(), 2);
    let outer = spans.iter().find(|s| s.name == "outer").unwrap();
    let inner = spans.iter().find(|s| s.name == "inner").unwrap();
    assert_eq!(inner.parent, Some(outer.id), "inner span parent is outer");
    assert_eq!(outer.parent, None, "outer span has no parent");
}

#[test]
fn trace_empty_has_zero_counters() {
    let pool = TaskPool::with_threads(2);
    let trace = pool.new_job_trace();
    assert_eq!(trace.span_count(), 0);
    assert_eq!(trace.total_jobs(), 0);
    assert_eq!(trace.migrated_jobs(), 0);
    assert_eq!(trace.steal_rate(), 0.0);
}

// --- chrome-tracing JSON export (feature-gated) ---------------------------

#[cfg(feature = "trace")]
mod chrome {
    use super::{String, TaskPool, Vec};

    /// A minimal JSON value, enough to validate the chrome-tracing export.
    #[derive(Debug, PartialEq)]
    enum Json {
        Null,
        Bool(bool),
        Num(f64),
        Str(String),
        Arr(Vec<Json>),
        Obj(Vec<(String, Json)>),
    }

    /// A tiny recursive-descent JSON parser. Returns `None` on any malformed
    /// input, so a successful parse genuinely proves the export is valid JSON.
    struct Parser<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl<'a> Parser<'a> {
        fn new(s: &'a str) -> Self {
            Self { bytes: s.as_bytes(), pos: 0 }
        }

        fn skip_ws(&mut self) {
            while let Some(&b) = self.bytes.get(self.pos) {
                if b == b' ' || b == b'\n' || b == b'\r' || b == b'\t' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }

        fn peek(&self) -> Option<u8> {
            self.bytes.get(self.pos).copied()
        }

        fn eat(&mut self, b: u8) -> Option<()> {
            if self.peek() == Some(b) {
                self.pos += 1;
                Some(())
            } else {
                None
            }
        }

        fn parse_value(&mut self) -> Option<Json> {
            self.skip_ws();
            match self.peek()? {
                b'{' => self.parse_obj(),
                b'[' => self.parse_arr(),
                b'"' => Some(Json::Str(self.parse_string()?)),
                b't' => self.parse_lit("true", Json::Bool(true)),
                b'f' => self.parse_lit("false", Json::Bool(false)),
                b'n' => self.parse_lit("null", Json::Null),
                _ => self.parse_number(),
            }
        }

        fn parse_lit(&mut self, lit: &str, val: Json) -> Option<Json> {
            let end = self.pos + lit.len();
            if self.bytes.get(self.pos..end)? == lit.as_bytes() {
                self.pos = end;
                Some(val)
            } else {
                None
            }
        }

        fn parse_number(&mut self) -> Option<Json> {
            let start = self.pos;
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() || b == b'-' || b == b'+' || b == b'.' || b == b'e' || b == b'E' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.pos == start {
                return None;
            }
            let text = core::str::from_utf8(&self.bytes[start..self.pos]).ok()?;
            text.parse::<f64>().ok().map(Json::Num)
        }

        fn parse_string(&mut self) -> Option<String> {
            self.eat(b'"')?;
            let mut out = String::new();
            loop {
                let b = self.peek()?;
                self.pos += 1;
                match b {
                    b'"' => return Some(out),
                    b'\\' => {
                        let esc = self.peek()?;
                        self.pos += 1;
                        match esc {
                            b'"' => out.push('"'),
                            b'\\' => out.push('\\'),
                            b'/' => out.push('/'),
                            b'n' => out.push('\n'),
                            b'r' => out.push('\r'),
                            b't' => out.push('\t'),
                            b'u' => {
                                let hex = self.bytes.get(self.pos..self.pos + 4)?;
                                let hs = core::str::from_utf8(hex).ok()?;
                                let code = u32::from_str_radix(hs, 16).ok()?;
                                self.pos += 4;
                                out.push(char::from_u32(code)?);
                            }
                            _ => return None,
                        }
                    }
                    _ => {
                        // Collect the raw UTF-8 byte(s); push as a char via the
                        // surrounding string once validated below.
                        out.push(b as char);
                    }
                }
            }
        }

        fn parse_arr(&mut self) -> Option<Json> {
            self.eat(b'[')?;
            let mut items = Vec::new();
            self.skip_ws();
            if self.peek() == Some(b']') {
                self.pos += 1;
                return Some(Json::Arr(items));
            }
            loop {
                items.push(self.parse_value()?);
                self.skip_ws();
                match self.peek()? {
                    b',' => {
                        self.pos += 1;
                    }
                    b']' => {
                        self.pos += 1;
                        return Some(Json::Arr(items));
                    }
                    _ => return None,
                }
            }
        }

        fn parse_obj(&mut self) -> Option<Json> {
            self.eat(b'{')?;
            let mut entries = Vec::new();
            self.skip_ws();
            if self.peek() == Some(b'}') {
                self.pos += 1;
                return Some(Json::Obj(entries));
            }
            loop {
                self.skip_ws();
                let key = self.parse_string()?;
                self.skip_ws();
                self.eat(b':')?;
                let value = self.parse_value()?;
                entries.push((key, value));
                self.skip_ws();
                match self.peek()? {
                    b',' => {
                        self.pos += 1;
                    }
                    b'}' => {
                        self.pos += 1;
                        return Some(Json::Obj(entries));
                    }
                    _ => return None,
                }
            }
        }

        fn parse_document(mut self) -> Option<Json> {
            let value = self.parse_value()?;
            self.skip_ws();
            if self.pos == self.bytes.len() {
                Some(value)
            } else {
                None
            }
        }
    }

    fn get<'j>(obj: &'j Json, key: &str) -> Option<&'j Json> {
        if let Json::Obj(entries) = obj {
            entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        } else {
            None
        }
    }

    #[test]
    fn chrome_json_parses_and_matches_spans() {
        let pool = TaskPool::with_threads(4);
        let trace = pool.new_job_trace();
        let n = 64usize;
        pool.scope(|s| {
            for i in 0..n {
                let job = trace.instrument("ecs_system", move || {
                    core::hint::black_box(i);
                });
                s.spawn(job);
            }
        });

        let json = trace.to_chrome_json();
        let parsed = Parser::new(&json)
            .parse_document()
            .expect("chrome JSON must parse");
        let events = match parsed {
            Json::Arr(items) => items,
            other => panic!("expected a JSON array, got {other:?}"),
        };
        assert_eq!(events.len(), n, "one event per span");

        for ev in &events {
            assert_eq!(get(ev, "ph"), Some(&Json::Str(String::from("X"))), "complete event");
            assert_eq!(get(ev, "name"), Some(&Json::Str(String::from("ecs_system"))));
            assert_eq!(get(ev, "pid"), Some(&Json::Num(1.0)));
            assert!(matches!(get(ev, "tid"), Some(Json::Num(_))), "tid present");
            assert!(matches!(get(ev, "ts"), Some(Json::Num(_))), "ts present");
            assert!(matches!(get(ev, "dur"), Some(Json::Num(_))), "dur present");
            // args.id must be present and parseable.
            let args = get(ev, "args").expect("args object");
            assert!(matches!(get(args, "id"), Some(Json::Num(_))), "args.id present");
        }
    }

    #[test]
    fn chrome_json_escapes_special_characters() {
        let pool = TaskPool::with_threads(2);
        let trace = pool.new_job_trace();
        trace.record("quote\"and\\slash\nnewline", || {});
        let json = trace.to_chrome_json();
        let parsed = Parser::new(&json).parse_document().expect("parses");
        let events = match parsed {
            Json::Arr(items) => items,
            other => panic!("expected array, got {other:?}"),
        };
        assert_eq!(events.len(), 1);
        assert_eq!(
            get(&events[0], "name"),
            Some(&Json::Str(String::from("quote\"and\\slash\nnewline")))
        );
    }

    #[test]
    fn chrome_json_empty_trace_is_empty_array() {
        let pool = TaskPool::with_threads(2);
        let trace = pool.new_job_trace();
        let json = trace.to_chrome_json();
        assert_eq!(json, "[]");
        let parsed = Parser::new(&json).parse_document().expect("parses");
        assert_eq!(parsed, Json::Arr(Vec::new()));
    }
}
