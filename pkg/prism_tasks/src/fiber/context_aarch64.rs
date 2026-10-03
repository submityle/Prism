//! aarch64 (AAPCS64) context switch and initial-frame construction.
//!
//! A [`Context`](super::context::Context) stores only the stack pointer. The
//! callee-saved state the AAPCS64 ABI requires us to preserve across a call is
//! spilled to (and reloaded from) the owning stack by [`switch`]:
//! - integer callee-saved registers `x19`–`x28`,
//! - the frame pointer `x29` and link register `x30`,
//! - the low 64 bits of the SIMD/FP callee-saved registers `d8`–`d15`.
//!
//! That is a fixed 160-byte frame (12 general-purpose + 8 FP, each 8 bytes),
//! kept 16-byte aligned as the ABI requires for `sp` at all times.
#![expect(
    unsafe_code,
    reason = "arch-specific fiber context switch: naked register save/restore and raw stack init"
)]

use core::arch::naked_asm;

use super::FiberInner;
use super::context::Context;

/// Save the running context into `*from` and restore `*to`.
///
/// `from`/`to` arrive in `x0`/`x1` per AAPCS64. The routine reserves a
/// 160-byte frame, stores the callee-saved registers, writes the resulting
/// `sp` to `from->sp` (offset 0), loads `to->sp`, reloads the callee-saved
/// registers, releases the frame, and `ret`s to the restored `x30`.
///
/// # Safety
/// See [`super::context::switch`].
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn switch(from: *mut Context, to: *const Context) {
    naked_asm!(
        "sub sp, sp, #160",
        "stp x19, x20, [sp, #0]",
        "stp x21, x22, [sp, #16]",
        "stp x23, x24, [sp, #32]",
        "stp x25, x26, [sp, #48]",
        "stp x27, x28, [sp, #64]",
        "stp x29, x30, [sp, #80]",
        "stp d8, d9, [sp, #96]",
        "stp d10, d11, [sp, #112]",
        "stp d12, d13, [sp, #128]",
        "stp d14, d15, [sp, #144]",
        "mov x2, sp",
        "str x2, [x0]", // from->sp = sp
        "ldr x2, [x1]", // sp = to->sp
        "mov sp, x2",
        "ldp x19, x20, [sp, #0]",
        "ldp x21, x22, [sp, #16]",
        "ldp x23, x24, [sp, #32]",
        "ldp x25, x26, [sp, #48]",
        "ldp x27, x28, [sp, #64]",
        "ldp x29, x30, [sp, #80]",
        "ldp d8, d9, [sp, #96]",
        "ldp d10, d11, [sp, #112]",
        "ldp d12, d13, [sp, #128]",
        "ldp d14, d15, [sp, #144]",
        "add sp, sp, #160",
        "ret",
    )
}

/// First-run trampoline. Reached via the `ret` in [`switch`] on the very first
/// resume of a freshly [`init_stack`]-prepared fiber. The fiber pointer was
/// planted in the `x19` save slot, so on entry `x19` holds it; we move it into
/// `x0` (first AAPCS64 argument) and branch into the Rust entry, which never
/// returns.
///
/// # Safety
/// Only ever entered via the prepared initial frame; not a normal callable.
#[unsafe(naked)]
unsafe extern "C" fn trampoline() {
    naked_asm!(
        "mov x0, x19",  // first argument = fiber pointer
        "bl {enter}",   // enter the Rust fiber body; never returns
        "brk #1",       // trap if it ever did
        enter = sym super::fiber_enter,
    )
}

/// Build the initial frame on `stack_top` so the first switch lands in
/// [`trampoline`] with `x19 == fiber` and `x30 == trampoline`.
///
/// # Safety
/// See [`super::context::init_stack`].
pub(crate) unsafe fn init_stack(stack_top: *mut u8, fiber: *mut FiberInner) -> Context {
    // 12 GP registers (96 bytes) + 8 FP registers (64 bytes) = 160-byte frame.
    const FRAME: usize = 160;
    const SLOTS: usize = FRAME / 8;
    // Align the usable top down to 16, then reserve the frame.
    let top = (stack_top as usize) & !0xf;
    let sp = (top - FRAME) as *mut u8;

    // SAFETY: `sp` lies inside the caller-provided stack region with FRAME
    // bytes of headroom; we initialize every slot the switch/trampoline reads.
    unsafe {
        let slots = sp as *mut u64;
        for i in 0..SLOTS {
            slots.add(i).write(0);
        }
        // x19 save slot (offset 0) carries the fiber pointer argument.
        slots.add(0).write(fiber as u64);
        // x30 (link register) save slot (offset 88 == slot 11) -> trampoline.
        slots.add(11).write(trampoline as *const () as usize as u64);
    }

    Context { sp }
}
