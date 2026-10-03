//! Level filtering: a compile-time floor plus a runtime threshold.

use core::sync::atomic::{AtomicU8, Ordering};

use crate::model::Level;

/// Compile-time maximum verbosity. Events more verbose than this are removed by
/// the macros at compile time (zero runtime cost). Selected via the
/// `max_level_*` features; defaults to `Trace` (everything compiled in).
pub const STATIC_MAX_LEVEL: Option<Level> = static_max_level();

const fn static_max_level() -> Option<Level> {
    if cfg!(feature = "max_level_off") {
        None
    } else if cfg!(feature = "max_level_error") {
        Some(Level::Error)
    } else if cfg!(feature = "max_level_warn") {
        Some(Level::Warn)
    } else if cfg!(feature = "max_level_info") {
        Some(Level::Info)
    } else if cfg!(feature = "max_level_debug") {
        Some(Level::Debug)
    } else {
        Some(Level::Trace)
    }
}

static RUNTIME_LEVEL: AtomicU8 = AtomicU8::new(Level::Info as u8);

/// Set the runtime minimum level; events below it are dropped.
pub fn set_max_level(level: Level) {
    RUNTIME_LEVEL.store(level as u8, Ordering::Release);
}

/// Current runtime minimum level.
pub fn max_level() -> Level {
    match RUNTIME_LEVEL.load(Ordering::Acquire) {
        0 => Level::Trace,
        1 => Level::Debug,
        2 => Level::Info,
        3 => Level::Warn,
        _ => Level::Error,
    }
}

/// Whether an event at `level` passes both the compile-time floor and the
/// current runtime threshold.
#[inline]
pub fn enabled(level: Level) -> bool {
    match STATIC_MAX_LEVEL {
        None => false,
        Some(floor) => level >= floor && level >= max_level(),
    }
}
