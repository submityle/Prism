//! Structured trace event stream + Chrome/Perfetto timeline export
//! (`trace` feature, design §16.6 "系统火焰图" 的时间线对偶 / §19 feature 清单).
//!
//! # Why this exists alongside [`profiler`](crate::diagnostics::profiler)
//!
//! The profiler folds the executor's nested `begin`/`end` calls into a
//! self-time *tree* and exports a flame graph — it answers *where the time
//! went*. This module keeps the complementary artefact every AAA engine also
//! ships: the raw *ordered event stream* with explicit timestamps and parallel
//! tracks — it answers *what ran when, on which track, next to what*. The two
//! share the same [`SystemInstrument`](crate::diagnostics::profiler::SystemInstrument)
//! capture hook, so enabling tracing adds a timeline view without changing the
//! executor or paying anything on the default (uninstrumented) path.
//!
//! # Layers
//!
//! * [`event`] — the data model: [`TrackId`], [`EventPhase`], typed
//!   [`TraceArg`]/[`TraceArgValue`], and a single [`TraceEvent`]. `no_std`.
//! * [`buffer`] — [`TraceBuffer`], the ordered, optionally bounded sink that
//!   stamps a stable sequence number on every event. `no_std`.
//! * [`chrome`] — [`to_chrome_json`], a dependency-free Chrome Trace Event
//!   Format exporter (consumed by `chrome://tracing`, Perfetto, catapult).
//!   `no_std + alloc`.
//! * [`recorder`] — [`TraceRecorder`], a `std` bridge implementing
//!   [`SystemInstrument`](crate::diagnostics::profiler::SystemInstrument) that
//!   timestamps executor spans from [`std::time::Instant`]. `std`-gated.
//!
//! # Example (core, caller-supplied timestamps)
//!
//! ```
//! use prism_ecs::diagnostics::trace::{TraceBuffer, TrackId, to_chrome_json};
//!
//! let mut buf = TraceBuffer::with_capacity(1024);
//! buf.complete_counted(TrackId::MAIN, 1_000, 8_000, "system", "physics", "dirty_chunks", 4);
//! buf.instant(TrackId::MAIN, 9_000, "frame_end");
//! assert_eq!(buf.len(), 2);
//!
//! let json = to_chrome_json(&buf);
//! assert!(json.starts_with("{\"displayTimeUnit\":\"ns\",\"traceEvents\":["));
//! ```

pub mod buffer;
pub mod chrome;
pub mod event;
#[cfg(feature = "std")]
pub mod recorder;

pub use buffer::TraceBuffer;
pub use chrome::to_chrome_json;
pub use event::{EventPhase, TraceArg, TraceArgValue, TraceEvent, TrackId};
#[cfg(feature = "std")]
pub use recorder::TraceRecorder;

#[cfg(test)]
mod tests;
