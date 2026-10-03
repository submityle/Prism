//! Logging macros with compile-time level trimming.

/// Log an event at an explicit [`Level`](crate::Level) with `format!` syntax.
///
/// The event is only built and dispatched when its level passes both the
/// compile-time floor ([`STATIC_MAX_LEVEL`](crate::filter::STATIC_MAX_LEVEL))
/// and the runtime threshold.
#[macro_export]
macro_rules! event {
    ($level:expr, $($arg:tt)+) => {{
        let level = $level;
        if $crate::filter::enabled(level) {
            $crate::__dispatch_message(level, module_path!(), ::core::format_args!($($arg)+));
        }
    }};
}

/// Log at [`Level::Trace`](crate::Level::Trace).
#[macro_export]
macro_rules! trace {
    ($($arg:tt)+) => { $crate::event!($crate::Level::Trace, $($arg)+) };
}

/// Log at [`Level::Debug`](crate::Level::Debug).
#[macro_export]
macro_rules! debug {
    ($($arg:tt)+) => { $crate::event!($crate::Level::Debug, $($arg)+) };
}

/// Log at [`Level::Info`](crate::Level::Info).
#[macro_export]
macro_rules! info {
    ($($arg:tt)+) => { $crate::event!($crate::Level::Info, $($arg)+) };
}

/// Log at [`Level::Warn`](crate::Level::Warn).
#[macro_export]
macro_rules! warn {
    ($($arg:tt)+) => { $crate::event!($crate::Level::Warn, $($arg)+) };
}

/// Log at [`Level::Error`](crate::Level::Error).
#[macro_export]
macro_rules! error {
    ($($arg:tt)+) => { $crate::event!($crate::Level::Error, $($arg)+) };
}
