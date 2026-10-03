//! Machine context: a saved stack pointer plus the arch-specific `switch` and
//! initial-frame construction.
//!
//! The whole fiber mechanism rests on one primitive: [`switch`], which saves
//! the current callee-saved registers and stack pointer into one [`Context`]
//! and restores them from another, transferring control (a cooperative,
//! stackful coroutine switch). A [`Context`] is deliberately tiny — just the
//! saved stack pointer — because every callee-saved register is pushed onto the
//! *owning* stack by `switch` itself; only the stack pointer needs to live in
//! the struct.
//!
//! Two architectures are implemented, selected at compile time:
//! - `x86_64` (System V): [`mod@super::context_x86_64`]
//! - `aarch64` (AAPCS64): [`mod@super::context_aarch64`]
//!
//! Any other target fails to compile with a clear message, rather than silently
//! shipping a broken switch.
#![expect(
    unsafe_code,
    reason = "fiber context facade: Send over a raw SP and forwarding to arch switch/init"
)]

#[cfg(target_arch = "x86_64")]
use super::context_x86_64 as imp;

#[cfg(target_arch = "aarch64")]
use super::context_aarch64 as imp;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!(
    "prism_tasks `fibers` feature requires x86_64 or aarch64 context switching; \
     this target is unsupported. Build without the `fibers` feature to use the \
     help-on-wait fallback."
);

use super::FiberInner;

/// A saved machine context: the stack pointer at the point of a [`switch`].
///
/// All other callee-saved registers are spilled to (and reloaded from) the
/// owning stack by the switch routine, so this struct only has to remember
/// where that stack is. A freshly-[`Context::zeroed`] value is only valid as a
/// *destination to save into* (the `from` side of the first [`switch`]); it
/// must never be used as a resume target until something has been saved there.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct Context {
    /// Saved stack pointer. `switch` reads/writes exactly this field (offset 0,
    /// relied on by the assembly).
    pub(crate) sp: *mut u8,
}

// SAFETY: a `Context` is a plain saved stack pointer. It is only ever accessed
// by one thread at a time under the fiber scheduling protocol (the thread
// currently executing the associated stack), so sharing the pointer value
// across threads as the fiber migrates is sound.
unsafe impl Send for Context {}

impl Context {
    /// A zeroed context, valid only as the first save destination.
    pub(crate) fn zeroed() -> Self {
        Self {
            sp: core::ptr::null_mut(),
        }
    }
}

/// Save the current context into `from` and restore the one in `to`.
///
/// # Safety
/// - `from` must be a valid, writable `*mut Context` and `to` a valid
///   `*const Context` whose `sp` points at a stack previously prepared by
///   [`init_stack`] or saved by a prior `switch`.
/// - The stack referenced by `to` must not be concurrently in use by any other
///   thread. The scheduler guarantees this: a suspended fiber is owned by
///   exactly one queue slot and resumed by exactly one worker.
#[inline]
pub(crate) unsafe fn switch(from: *mut Context, to: *const Context) {
    // SAFETY: forwarded to the arch implementation under the same contract.
    unsafe {
        imp::switch(from, to);
    }
}

/// Prepare `stack` so the first [`switch`] into the returned context begins
/// executing the fiber trampoline, which calls [`super::fiber_enter`] with
/// `fiber` as its argument.
///
/// # Safety
/// `stack_top` must be the one-past-the-end address of a writable region of at
/// least a few hundred bytes below it (the initial frame), and `fiber` must
/// remain valid until the fiber finishes.
pub(crate) unsafe fn init_stack(stack_top: *mut u8, fiber: *mut FiberInner) -> Context {
    // SAFETY: forwarded to the arch implementation under the same contract.
    unsafe { imp::init_stack(stack_top, fiber) }
}
