//! x86_64 System V context switch and initial-frame construction.
//!
//! A [`Context`](super::context::Context) stores only the stack pointer; the
//! six callee-saved integer registers (`rbx`, `rbp`, `r12`–`r15`) are pushed
//! onto the owning stack by [`switch`] and popped back on resume. This is the
//! classic "swapcontext" shape used by stackful-coroutine libraries, reduced to
//! the minimum the System V ABI requires.
#![expect(
    unsafe_code,
    reason = "arch-specific fiber context switch: naked register save/restore and raw stack init"
)]

use core::arch::naked_asm;

use super::FiberInner;
use super::context::Context;

/// Save the running context into `*from` and restore `*to`.
///
/// Layout pushed onto the current stack (high → low as pushed): `rbp`, `rbx`,
/// `r12`, `r13`, `r14`, `r15`. The resulting `rsp` is written to `from->sp`
/// (offset 0). The target `to->sp` is loaded, the same six registers are popped
/// in reverse, and `ret` transfers control to the saved return address.
///
/// # Safety
/// See [`super::context::switch`]. `from`/`to` are `*mut`/`*const Context`
/// passed in `rdi`/`rsi` per System V.
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn switch(from: *mut Context, to: *const Context) {
    naked_asm!(
        "push rbp",
        "push rbx",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        "mov [rdi], rsp", // from->sp = rsp
        "mov rsp, [rsi]", // rsp = to->sp
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbx",
        "pop rbp",
        "ret",
    )
}

/// First-run trampoline. Reached via the `ret` in [`switch`] on the very first
/// resume of a freshly [`init_stack`]-prepared fiber. The fiber pointer was
/// planted in the `r12` save slot, so on entry `r12` holds it; we move it into
/// `rdi` (first System V argument), align the stack, and call the Rust entry,
/// which never returns.
///
/// # Safety
/// Only ever entered via the prepared initial frame; not a normal callable.
#[unsafe(naked)]
unsafe extern "C" fn trampoline() {
    naked_asm!(
        "mov rdi, r12",   // first argument = fiber pointer
        "and rsp, -16",   // realign to 16 bytes for the call
        "call {enter}",   // enter the Rust fiber body; never returns
        "ud2",            // trap if it ever did
        enter = sym super::fiber_enter,
    )
}

/// Build the initial frame on `stack_top` so the first switch lands in
/// [`trampoline`] with `r12 == fiber`.
///
/// # Safety
/// See [`super::context::init_stack`].
pub(crate) unsafe fn init_stack(stack_top: *mut u8, fiber: *mut FiberInner) -> Context {
    // 6 saved registers (48 bytes) + return address (8) + 8 pad = 64-byte frame.
    const FRAME: usize = 64;
    // Align the usable top down to 16, then reserve the frame.
    let top = (stack_top as usize) & !0xf;
    let sp = (top - FRAME) as *mut u8;

    // SAFETY: `sp` lies inside the caller-provided stack region with FRAME
    // bytes of headroom; we initialize every slot the switch/trampoline reads.
    unsafe {
        let slots = sp as *mut u64;
        // r15, r14, r13 at offsets 0, 8, 16 -> zero.
        slots.add(0).write(0);
        slots.add(1).write(0);
        slots.add(2).write(0);
        // r12 slot (offset 24) carries the fiber pointer argument.
        slots.add(3).write(fiber as u64);
        // rbx, rbp at offsets 32, 40 -> zero.
        slots.add(4).write(0);
        slots.add(5).write(0);
        // Return address slot (offset 48) -> trampoline.
        slots.add(6).write(trampoline as *const () as usize as u64);
    }

    Context { sp }
}
