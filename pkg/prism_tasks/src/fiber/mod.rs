//! Stackful fibers: the machinery that lets a job `wait` on a [`Counter`]
//! without blocking its OS worker thread.
//!
//! A *fiber* is a job plus its own machine stack and a saved register context.
//! When the `fibers` feature is on, every job a worker picks up runs on a
//! fiber. If that job calls [`TaskPool::wait`](crate::TaskPool::wait) on a
//! counter that has not yet reached zero, the fiber saves its context and
//! switches back to the worker's scheduler loop ([`suspend_current`]); the
//! worker is now free to run other jobs or resume other fibers. When the
//! counter later reaches zero the fiber is moved to a resume queue and picked up
//! by some worker, continuing exactly where it left off.
//!
//! ## Module layout
//! - [`context`]: the architecture-neutral [`switch`](context::switch) /
//!   [`init_stack`](context::init_stack) facade.
//! - `context_x86_64` / `context_aarch64`: the two implemented backends.
//! - [`stack`]: the reusable fiber stack pool.
//! - [`wait_set`]: suspended-fiber bookkeeping and the resume queue.
//!
//! ## Ownership and threading
//! A [`FiberInner`] is heap-allocated and owned by exactly one place at a time:
//! the worker currently running it, or a queue slot (the pool's job→fiber path,
//! the wait-set's parked list, or its resume queue). Ownership is passed around
//! as a raw pointer wrapped in [`FiberPtr`]; the single-owner protocol is what
//! makes that sound. A fiber may migrate between workers across a suspend, so
//! [`FiberInner::return_context`] is refreshed on every resume to point at the
//! *current* worker's on-stack scheduler context.

pub(crate) mod context;
pub(crate) mod stack;
pub(crate) mod wait_set;

#[cfg(target_arch = "x86_64")]
mod context_x86_64;

#[cfg(target_arch = "aarch64")]
mod context_aarch64;

use std::any::Any;
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use context::Context;

use crate::Counter;
use crate::job::Job;
use crate::scheduler::Shared;

thread_local! {
    /// The fiber currently executing on this worker thread, or null when the
    /// thread is running its scheduler loop rather than a fiber. [`on_fiber`]
    /// reads it so `wait` knows whether it can suspend.
    static CURRENT: Cell<*mut FiberInner> = const { Cell::new(std::ptr::null_mut()) };
}

/// Lifecycle state of a fiber, observed by the worker after a switch-back.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FiberState {
    /// Currently executing (set just before switching into the fiber).
    Running,
    /// Switched out in `wait`, waiting on [`FiberInner::wait_on`].
    Suspended,
    /// The job finished (normally or via a captured panic); stack reclaimable.
    Done,
}

/// A job together with its own stack and saved context: one stackful coroutine.
pub(crate) struct FiberInner {
    /// This fiber's saved machine context (valid while it is suspended/ready).
    context: Context,
    /// Where to switch back to: the running worker's on-stack scheduler
    /// context. Refreshed on every resume so a migrated fiber returns to
    /// whichever worker is driving it now.
    return_context: *mut Context,
    /// The owned stack this fiber runs on; taken and released to the pool when
    /// the fiber finishes.
    stack: Option<stack::Stack>,
    /// The job body, taken by [`fiber_enter`] on first run.
    job: Option<Job>,
    /// Lifecycle state (see [`FiberState`]).
    state: FiberState,
    /// The counter this fiber is suspended on, set by [`suspend_current`].
    wait_on: Option<Counter>,
    /// A panic captured from the job, re-raised by the worker after cleanup so
    /// it never crosses the assembly switch boundary.
    panic: Option<Box<dyn Any + Send + 'static>>,
}

/// A `Send` raw handle to a [`FiberInner`], moved through the scheduler queues.
///
/// Sending the pointer across threads is sound under the single-owner protocol:
/// a fiber is only ever enqueued once, dequeued once, and run by one worker at a
/// time, and its context is fully saved before it becomes eligible for pickup.
#[derive(Clone, Copy)]
pub(crate) struct FiberPtr(pub(crate) *mut FiberInner);

// SAFETY: see the type docs — exactly one owner touches the pointee at a time.
#[expect(unsafe_code, reason = "single-owner fiber handoff across worker threads")]
unsafe impl Send for FiberPtr {}

/// Whether the calling thread is currently executing inside a fiber.
pub(crate) fn on_fiber() -> bool {
    CURRENT.with(|c| !c.get().is_null())
}

/// Allocate a fiber for `job`, drawing a stack from `shared`'s pool and priming
/// its initial context so the first switch begins executing the job.
pub(crate) fn spawn_fiber(shared: &Shared, job: Job) -> *mut FiberInner {
    let stack = shared.stack_pool().acquire();
    let top = stack.top();
    let inner = Box::new(FiberInner {
        context: Context::zeroed(),
        return_context: std::ptr::null_mut(),
        stack: Some(stack),
        job: Some(job),
        state: FiberState::Suspended,
        wait_on: None,
        panic: None,
    });
    let raw = Box::into_raw(inner);
    // SAFETY: `raw` is a fresh unique allocation; `top` is the one-past-end of
    // the stack we just acquired for it. The fiber lives until it reports
    // `Done`, well past this initialization.
    #[expect(unsafe_code, reason = "prime the fiber's initial register image")]
    unsafe {
        (*raw).context = context::init_stack(top, raw);
    }
    raw
}

/// Run or resume `fiber` on the current worker until it next suspends or
/// finishes, then dispose of it: a finished fiber's stack is returned to the
/// pool (and any captured panic re-raised here, on the worker, where it is
/// safe), while a suspended fiber is parked on its counter.
///
/// This is the only place that switches *into* a fiber, and it keeps the
/// worker's scheduler context (`sched`) as a stack local — a stable address for
/// the duration of the switch, so the fiber can switch back to it.
pub(crate) fn run_fiber_switch(shared: &Shared, fiber: *mut FiberInner) {
    let prev = CURRENT.with(|c| c.replace(fiber));
    let mut sched = Context::zeroed();

    // SAFETY: `fiber` is exclusively owned here. We write its return path and
    // state through the raw pointer (holding no long-lived reference across the
    // switch, so the fiber's own `&mut` inside `fiber_enter`/`suspend_current`
    // never aliases ours), then switch in. `context` was prepared by
    // `init_stack` (fresh fiber) or saved by a prior `suspend_current`.
    #[expect(unsafe_code, reason = "switch into the fiber's saved/initial context")]
    unsafe {
        (*fiber).return_context = &raw mut sched;
        (*fiber).state = FiberState::Running;
        let target: *const Context = &raw const (*fiber).context;
        context::switch(&raw mut sched, target);
    }

    // Back on the scheduler stack: restore the previous fiber (if we were
    // nested under a resume driven from `help_until`) and dispose of this one.
    CURRENT.with(|c| c.set(prev));

    // SAFETY: the fiber switched back to us, so its stack is idle and we are its
    // sole owner; reading `state` through the raw pointer is sound.
    #[expect(unsafe_code, reason = "inspect the fiber's post-switch state")]
    let state = unsafe { (*fiber).state };

    match state {
        FiberState::Done => {
            // SAFETY: reclaim the uniquely-owned box exactly once.
            #[expect(unsafe_code, reason = "reclaim the finished fiber allocation")]
            let inner = unsafe { Box::from_raw(fiber) };
            let FiberInner { stack, panic, .. } = *inner;
            if let Some(stack) = stack {
                shared.stack_pool().release(stack);
            }
            if let Some(payload) = panic {
                resume_unwind(payload);
            }
        }
        FiberState::Suspended => {
            // SAFETY: still the sole owner; take the wait target it recorded.
            #[expect(unsafe_code, reason = "read the suspended fiber's wait target")]
            let counter = unsafe { (*fiber).wait_on.take() }
                .expect("suspended fiber recorded no wait target");
            shared.wait_set().park(FiberPtr(fiber), counter);
            // Nudge an idle worker in case `park` routed us straight to resume.
            shared.wake_one();
        }
        FiberState::Running => {
            // A fiber can only yield control by finishing or suspending; any
            // other state here means the switch protocol was violated.
            std::process::abort();
        }
    }
}

/// Suspend the currently running fiber on `counter`, switching back to the
/// worker's scheduler loop. Returns only once the fiber is resumed, which the
/// wait-set guarantees happens only after `counter` has reached zero.
pub(crate) fn suspend_current(counter: &Counter) {
    let fiber = CURRENT.with(|c| c.get());
    debug_assert!(!fiber.is_null(), "suspend_current called off a fiber");

    // SAFETY: `fiber` is the fiber running on this stack and is exclusively
    // owned by it. We record the wait target and save our context; the worker's
    // `run_fiber_switch` reads `state == Suspended` after this switch and parks
    // us only once we are fully switched out.
    #[expect(unsafe_code, reason = "save the running fiber and yield to the worker")]
    unsafe {
        (*fiber).state = FiberState::Suspended;
        (*fiber).wait_on = Some(counter.clone());
        let ret = (*fiber).return_context;
        let ctx = &raw mut (*fiber).context;
        context::switch(ctx, ret);
    }
    // Resumed: `counter` is complete (park/flush only resume on completion).
}

/// The Rust entry point every fiber begins at, reached from the arch
/// trampoline on first switch-in. Runs the job (catching any panic so it never
/// unwinds across the assembly boundary), marks the fiber `Done`, and switches
/// back to the worker. Never returns.
///
/// Declared `extern "C"` because the trampoline calls it with the C ABI and the
/// fiber pointer in the first-argument register.
pub(crate) extern "C" fn fiber_enter(fiber: *mut FiberInner) -> ! {
    // SAFETY: the fiber is exclusively owned while it runs; `job` was planted by
    // `spawn_fiber` and is taken exactly once.
    #[expect(unsafe_code, reason = "take the job out of the running fiber")]
    let job = unsafe { (*fiber).job.take() }.expect("fiber started without a job");

    let result = catch_unwind(AssertUnwindSafe(job));

    // SAFETY: still exclusively owned. Record any panic, mark done, then switch
    // back to the worker's scheduler context, which is live and awaiting us.
    #[expect(unsafe_code, reason = "finalize the fiber and switch back to the worker")]
    unsafe {
        if let Err(payload) = result {
            (*fiber).panic = Some(payload);
        }
        (*fiber).state = FiberState::Done;
        let ret = (*fiber).return_context;
        let ctx = &raw mut (*fiber).context;
        context::switch(ctx, ret);
    }

    // The switch above transfers control to the worker and never comes back to
    // this (now-finished) stack. Reaching here would mean the context was
    // corrupt, so fail loudly rather than execute into garbage.
    std::process::abort();
}
