use crate::prelude::*;
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
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].level, Level::Warn);
    assert_eq!(events[0].message, "this passes 2");
    assert_eq!(events[1].level, Level::Error);
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
