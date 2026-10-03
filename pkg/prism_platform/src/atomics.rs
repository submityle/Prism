//! Atomic/ordering/pause facade (`no_std`-friendly).
//!
//! Re-exports the `core::sync::atomic` primitives so downstream crates can
//! depend on a single platform surface, plus a CPU `pause`/spin hint and a
//! full memory fence helper.

pub use core::sync::atomic::{
    AtomicBool, AtomicI32, AtomicI64, AtomicIsize, AtomicU32, AtomicU64, AtomicUsize, Ordering,
};

/// Emit a spin-loop/`pause` hint to the CPU while busy-waiting.
#[inline]
pub fn spin_hint() {
    core::hint::spin_loop();
}

/// Full sequentially-consistent memory fence.
#[inline]
pub fn full_fence() {
    core::sync::atomic::fence(Ordering::SeqCst);
}
