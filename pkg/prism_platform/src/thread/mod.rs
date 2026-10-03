//! Threads, thread-local storage, affinity, lightweight sync, and park/unpark.
//!
//! This module is the M2 threading layer for the three desktop platforms
//! (Linux / macOS / Windows). It is a thin, testable surface over
//! [`std::thread`] plus a few primitives that `std` does not expose directly
//! (a spin lock, a backoff helper, a registry-based thread-local, and a
//! token-based parker).
//!
//! The whole module requires the `std` feature; OS threads are unavailable in
//! a `no_std` build. Under `no_std` the module is absent and the crate still
//! compiles (the OS probe layer in M0 stays `core`-only), mirroring [`crate::fs`].
//!
//! ## Layout
//! - [`spawn`]: a [`spawn::Builder`] (name + optional stack size) and a
//!   [`spawn::JoinHandle`], plus [`spawn::yield_now`], [`spawn::sleep`],
//!   [`spawn::current_id`], and [`spawn::hardware_concurrency`].
//! - [`tls`]: a registry-based [`tls::ThreadLocal`] giving per-thread slots.
//! - [`affinity`]: best-effort CPU-core pinning for the current thread, with an
//!   honest [`affinity::AffinityError::Unsupported`] where an OS cannot pin.
//! - [`sync`]: a [`sync::SpinLock`], a [`sync::Backoff`] helper, and a
//!   [`sync::Once`] one-time initializer.
//! - [`park`]: a token-based [`park::Parker`]/[`park::Unparker`] pair.

pub mod affinity;
pub mod park;
pub mod spawn;
pub mod sync;
pub mod tls;

pub use affinity::{affinity_supported, set_current_thread_affinity, set_current_thread_affinity_mask, AffinityError};
pub use park::{Parker, Unparker};
pub use spawn::{current_id, hardware_concurrency, sleep, spawn, yield_now, Builder, JoinHandle, ThreadId};
pub use sync::{Backoff, Once, SpinLock, SpinLockGuard};
pub use tls::ThreadLocal;

#[cfg(test)]
mod tests;
