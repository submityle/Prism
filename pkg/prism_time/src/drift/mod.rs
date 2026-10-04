//! **§24.5 — long-session precision and drift correction.** The pieces a 24/7
//! server or a multi-hour single-player session needs so time neither loses
//! precision nor diverges from an external reference.
//!
//! Two cooperating pieces, each in its own submodule:
//!
//! - [`MonotonicBaseline`] (in [`monotonic`]) — turns a fixed-width hardware
//!   counter (whose raw value wraps when the platform register overflows) into
//!   a monotonically increasing elapsed measure. It keeps the authoritative
//!   elapsed time as an **integer tick count** (`u128`), so there is no `f32`
//!   precision loss over a long run (the design doc's "`elapsed` in f64/integer
//!   ticks" requirement), and it detects and absorbs counter wrap-around.
//! - [`DriftCorrector`] (in [`corrector`]) — slowly reconciles a local
//!   monotonic clock toward an occasionally-sampled external reference (wall
//!   clock / NTP / authoritative server time) by **rate slew**, not by
//!   stepping. Each correction is bounded to a few hundred parts-per-million of
//!   real time, so corrected time stays strictly monotonic and never jumps —
//!   preserving the fixed-step determinism the simulation relies on. A separate
//!   [`resync`](DriftCorrector::resync) is offered for true session boundaries
//!   (first sync, resume-from-suspend) where a step is unavoidable.
//!
//! Everything here is deterministic integer arithmetic (`u128` / `i128`
//! nanoseconds); no floating point enters the correction path, so a given
//! sequence of deltas and reference samples produces a bit-identical corrected
//! timeline across runs. `no_std + alloc`, no `unsafe`.
//!
//! ## Honest boundary
//! Reading the real platform monotonic counter, obtaining NTP / server
//! reference samples, and the actual register width live in `prism_platform` /
//! the net layer — not here. This module is the deterministic *algorithm* those
//! callers drive with explicit raw counter values and reference samples; it
//! does not itself touch any hardware or wall clock (see design doc §24.9).

mod corrector;
mod monotonic;

pub use corrector::DriftCorrector;
pub use monotonic::MonotonicBaseline;
