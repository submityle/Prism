use crate::prelude::*;
use crate::trace::{current_thread_id, prune_finished_threads, registered_threads};
use alloc::sync::Arc;

extern crate alloc;

fn install_capture() -> CaptureSink {
    let sink = CaptureSink::new();
    set_sink(Arc::new(sink.clone()));
    sink
}

#[test]
fn runtime_filter_drops_below_threshold() {
    let sink = install_capture();
    set_max_level(Level::Warn);
    crate::info!("this is filtered {}", 1);
    crate::warn!("this passes {}", 2);
    crate::error!("this also passes");
    let events = sink.events();

    // The compile-time floor (`max_level_*` features) can further suppress
    // events before the runtime threshold even sees them, so compute the
    // expected survivors from that floor rather than assuming `std`-only
    // defaults. With the runtime threshold at `Warn`, `info` is always dropped;
    // `warn`/`error` survive only if the compile-time floor also admits them.
    let warn_in = crate::filter::enabled(Level::Warn);
    let error_in = crate::filter::enabled(Level::Error);
    let expected: Vec<(Level, &str)> = [
        warn_in.then_some((Level::Warn, "this passes 2")),
        error_in.then_some((Level::Error, "this also passes")),
    ]
    .into_iter()
    .flatten()
    .collect();

    assert_eq!(events.len(), expected.len());
    for (event, (level, message)) in events.iter().zip(expected) {
        assert_eq!(event.level, level);
        assert_eq!(event.message, message);
    }
    crate::clear_sink();
}

#[test]
fn fields_render_in_formatted_line() {
    let event = Event::new(Level::Info, "subsys", "spawned")
        .with_field("count", FieldValue::U64(3))
        .with_field("ok", FieldValue::Bool(true));
    let line = crate::fmt::format_line(&event);
    assert!(line.contains("[INFO ]"));
    assert!(line.contains("subsys: spawned"));
    assert!(line.contains("count=3"));
    assert!(line.contains("ok=true"));
}

#[test]
fn level_ordering_is_correct() {
    assert!(Level::Trace < Level::Info);
    assert!(Level::Error > Level::Warn);
}

/// Autoref-specialization probe: the inherent `is_not_send` (returning `false`)
/// applies only when `T: Send`; otherwise the trait fallback (`true`) is used —
/// a stable-Rust negative `Send` assertion.
struct SendProbe<T>(core::marker::PhantomData<T>);
trait NotSendFallback {
    fn is_not_send(&self) -> bool {
        true
    }
}
impl<T> NotSendFallback for SendProbe<T> {}
#[expect(dead_code, reason = "the Send arm is never instantiated by design")]
impl<T: Send> SendProbe<T> {
    fn is_not_send(&self) -> bool {
        false
    }
}

#[test]
fn timing_guards_are_not_send() {
    // `Scope` and `ProfiledZone` capture the constructing thread's depth/ring
    // and write back on `Drop`; moving one to another thread would corrupt the
    // original thread's nesting (CR-found). Assert `!Send`.
    assert!(SendProbe::<Scope>(core::marker::PhantomData).is_not_send());
    assert!(SendProbe::<ProfiledZone>(core::marker::PhantomData).is_not_send());
}

#[test]
fn finished_threads_are_reclaimable() {
    // A finished thread's track stays readable for a later export, but must be
    // reclaimable so a process churning short-lived threads does not leak a
    // ThreadTrace + ring per thread forever (CR-found).
    let dead_id = std::thread::spawn(|| {
        let _scope = Scope::new("probe");
        current_thread_id()
    })
    .join()
    .unwrap();
    let has = |id: u64| registered_threads().iter().any(|t| t.thread_id == id);
    assert!(
        has(dead_id),
        "a finished thread is retained until reclaimed"
    );
    assert!(prune_finished_threads() >= 1);
    assert!(!has(dead_id), "after pruning the finished track is gone");
}
