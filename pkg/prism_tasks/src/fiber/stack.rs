//! The fiber stack pool.
//!
//! Each fiber executes on its own contiguous, heap-allocated stack. Allocating
//! a stack per job would be wasteful, so finished stacks are returned to a pool
//! and reused. Acquire and release are O(1) (a free-list pop/push under one
//! mutex).
//!
//! ## Capacity and deadlock avoidance
//! The design notes (§23 risk 6) warn that a *fixed* pool can deadlock: if every
//! worker's fiber is suspended in `wait` and no free stack remains to resume
//! anyone, the system wedges. We avoid this class of deadlock entirely by
//! growing the pool on demand — [`StackPool::acquire`] never blocks and never
//! fails while memory is available, so the live-stack count simply tracks the
//! peak number of concurrently live fibers. The free list is what is bounded
//! (via `max_retained`); surplus stacks are freed on release rather than being
//! hoarded.

use std::alloc::{self, Layout};
use std::sync::Mutex;

/// Default fiber stack size: 128 KiB. Large enough for the recursive fork-join
/// and structured-parallel workloads this crate drives, while small enough that
/// hundreds of concurrently suspended fibers stay well within process limits.
pub(crate) const DEFAULT_STACK_SIZE: usize = 128 * 1024;

/// Alignment of a fiber stack. A page (4 KiB) is comfortably stricter than the
/// 16-byte ABI minimum and keeps each stack from straddling a page boundary at
/// its base, which simplifies a future guard-page upgrade.
const STACK_ALIGN: usize = 4096;

/// Default cap on the number of idle stacks kept on the free list. Beyond this,
/// released stacks are deallocated instead of retained, bounding steady-state
/// memory without ever blocking an acquire.
const DEFAULT_MAX_RETAINED: usize = 256;

/// One owned fiber stack: a raw heap region plus its layout for deallocation.
///
/// The usable region is `[ptr, ptr + size)`; the machine stack grows downward
/// from [`Stack::top`] (the one-past-the-end address).
pub(crate) struct Stack {
    ptr: *mut u8,
    layout: Layout,
}

// SAFETY: a `Stack` owns its heap region exclusively; the raw pointer is never
// aliased. Moving that ownership between threads (worker A acquires, the fiber
// migrates, worker B releases) is sound because only one thread holds the
// `Stack` at a time under the scheduler's single-owner protocol.
#[expect(unsafe_code, reason = "Stack uniquely owns its heap region; move is sound")]
unsafe impl Send for Stack {}

impl Stack {
    /// Allocate a fresh stack of `size` bytes.
    fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size, STACK_ALIGN)
            .expect("invalid fiber stack layout");
        // SAFETY: `layout` has a non-zero size and a valid power-of-two
        // alignment, so this call satisfies `alloc`'s contract.
        #[expect(unsafe_code, reason = "raw stack allocation for a fiber")]
        let ptr = unsafe { alloc::alloc(layout) };
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        Self { ptr, layout }
    }

    /// The one-past-the-end address of the stack region: the initial machine
    /// stack pointer (the stack grows downward from here).
    pub(crate) fn top(&self) -> *mut u8 {
        // SAFETY: `ptr + size` is the one-past-the-end address of the owned
        // allocation, which is a valid pointer to form (never dereferenced).
        #[expect(unsafe_code, reason = "compute one-past-end of the owned region")]
        unsafe {
            self.ptr.add(self.layout.size())
        }
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`layout` came from the matching `alloc::alloc` in
        // `Stack::new` and this is the sole owner, dropped exactly once.
        #[expect(unsafe_code, reason = "free the stack's heap region")]
        unsafe {
            alloc::dealloc(self.ptr, self.layout);
        }
    }
}

/// A pool of reusable fiber stacks.
pub(crate) struct StackPool {
    free: Mutex<Vec<Stack>>,
    stack_size: usize,
    max_retained: usize,
}

impl StackPool {
    /// Create a pool handing out stacks of [`DEFAULT_STACK_SIZE`].
    pub(crate) fn new() -> Self {
        Self::with_size(DEFAULT_STACK_SIZE)
    }

    /// Create a pool handing out stacks of `stack_size` bytes.
    pub(crate) fn with_size(stack_size: usize) -> Self {
        assert!(stack_size >= 4096, "fiber stack must be at least one page");
        Self {
            free: Mutex::new(Vec::new()),
            stack_size,
            max_retained: DEFAULT_MAX_RETAINED,
        }
    }

    /// Obtain a stack, reusing an idle one when available or allocating a fresh
    /// one otherwise. Never blocks and never fails while memory is available.
    pub(crate) fn acquire(&self) -> Stack {
        if let Some(stack) = self.free.lock().unwrap().pop() {
            return stack;
        }
        Stack::new(self.stack_size)
    }

    /// Return a stack for reuse. Kept on the free list up to `max_retained`
    /// entries; surplus stacks are deallocated (by dropping) to bound memory.
    pub(crate) fn release(&self, stack: Stack) {
        let mut free = self.free.lock().unwrap();
        if free.len() < self.max_retained {
            free.push(stack);
        }
        // else: `stack` drops here, freeing its region.
    }
}
