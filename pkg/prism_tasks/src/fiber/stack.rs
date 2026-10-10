//! The fiber stack pool, with large/small size classes (design §8, §12, §16).
//!
//! Each fiber executes on its own contiguous, heap-allocated stack. Allocating
//! a stack per job would be wasteful, so finished stacks are returned to a pool
//! and reused. Acquire and release are O(1) (a free-list pop/push under one
//! mutex per class).
//!
//! ## Size classes (design §8: "分大/小两档")
//! Fibers fall into two size classes so shallow jobs do not reserve a big
//! stack:
//! - [`StackClass::Large`] — [`LARGE_STACK_SIZE`], for deep recursive
//!   fork-join / structured-parallel bodies (the default).
//! - [`StackClass::Small`] — [`SMALL_STACK_SIZE`], for shallow leaf jobs.
//!
//! Each class has its own free list and may be [pre-warmed](StackPool::prewarm)
//! so the first frame does not pay allocation latency.
//!
//! ## Water marks (design §16: "fiber 栈占用峰值")
//! The pool tracks, per class, the peak number of concurrently live stacks
//! ([`StackPool::high_water`]) so capacity can be tuned and the §23-risk-6
//! "ran out of stacks" condition can be alarmed on.
//!
//! ## NUMA placement (design §14)
//! A stack can be tagged with the NUMA node of the worker that will run it so
//! that, together with the per-worker arena, a fiber's memory stays node-local.
//! Without OS NUMA support the tag is advisory (honest degradation) — the bytes
//! are ordinary heap — but the association is recorded on [`Stack::node`].
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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::numa::NumaNodeId;

/// Large fiber stack size: 128 KiB. Large enough for the recursive fork-join
/// and structured-parallel workloads this crate drives, while small enough that
/// hundreds of concurrently suspended fibers stay well within process limits.
pub(crate) const LARGE_STACK_SIZE: usize = 128 * 1024;

/// Small fiber stack size: 32 KiB, for shallow leaf jobs that never recurse
/// deeply. Four small stacks fit in the footprint of one large stack.
pub(crate) const SMALL_STACK_SIZE: usize = 32 * 1024;

/// Alignment of a fiber stack. A page (4 KiB) is comfortably stricter than the
/// 16-byte ABI minimum and keeps each stack from straddling a page boundary at
/// its base, which simplifies a future guard-page upgrade.
const STACK_ALIGN: usize = 4096;

/// Default cap on the number of idle stacks kept on each class's free list.
/// Beyond this, released stacks are deallocated instead of retained, bounding
/// steady-state memory without ever blocking an acquire.
const DEFAULT_MAX_RETAINED: usize = 256;

/// Fiber stack size class (design §8).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum StackClass {
    /// Small stack ([`SMALL_STACK_SIZE`]) for shallow jobs.
    Small,
    /// Large stack ([`LARGE_STACK_SIZE`]) for deep fork-join (default).
    Large,
}

/// One owned fiber stack: a raw heap region plus its layout for deallocation.
///
/// The usable region is `[ptr, ptr + size)`; the machine stack grows downward
/// from [`Stack::top`] (the one-past-the-end address).
pub(crate) struct Stack {
    ptr: *mut u8,
    layout: Layout,
    class: StackClass,
    node: NumaNodeId,
}

// SAFETY: a `Stack` owns its heap region exclusively; the raw pointer is never
// aliased. Moving that ownership between threads (worker A acquires, the fiber
// migrates, worker B releases) is sound because only one thread holds the
// `Stack` at a time under the scheduler's single-owner protocol.
#[expect(
    unsafe_code,
    reason = "Stack uniquely owns its heap region; move is sound"
)]
unsafe impl Send for Stack {}

impl Stack {
    /// Allocate a fresh stack for `class` tagged with NUMA `node`.
    fn new(class: StackClass, node: NumaNodeId) -> Self {
        let size = match class {
            StackClass::Small => SMALL_STACK_SIZE,
            StackClass::Large => LARGE_STACK_SIZE,
        };
        let layout =
            Layout::from_size_align(size, STACK_ALIGN).expect("invalid fiber stack layout");
        // SAFETY: `layout` has a non-zero size and a valid power-of-two
        // alignment, so this call satisfies `alloc`'s contract.
        #[expect(unsafe_code, reason = "raw stack allocation for a fiber")]
        let ptr = unsafe { alloc::alloc(layout) };
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        Self {
            ptr,
            layout,
            class,
            node,
        }
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

    /// This stack's size class.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "diagnostic accessor, used in tests")
    )]
    pub(crate) fn class(&self) -> StackClass {
        self.class
    }

    /// The NUMA node this stack is associated with.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "diagnostic accessor, used in tests")
    )]
    pub(crate) fn node(&self) -> NumaNodeId {
        self.node
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

/// Per-class free list plus live/peak counters.
struct ClassPool {
    free: Mutex<Vec<Stack>>,
    live: AtomicUsize,
    high_water: AtomicUsize,
}

impl ClassPool {
    fn new() -> Self {
        Self {
            free: Mutex::new(Vec::new()),
            live: AtomicUsize::new(0),
            high_water: AtomicUsize::new(0),
        }
    }
}

/// A pool of reusable fiber stacks, split into large and small size classes.
pub(crate) struct StackPool {
    small: ClassPool,
    large: ClassPool,
    max_retained: usize,
}

impl StackPool {
    /// Create a pool with empty free lists for both classes.
    pub(crate) fn new() -> Self {
        Self {
            small: ClassPool::new(),
            large: ClassPool::new(),
            max_retained: DEFAULT_MAX_RETAINED,
        }
    }

    fn class_pool(&self, class: StackClass) -> &ClassPool {
        match class {
            StackClass::Small => &self.small,
            StackClass::Large => &self.large,
        }
    }

    /// Pre-allocate `count` idle stacks of `class` so the first fibers of a
    /// frame reuse rather than allocate (design §8: 预分配复用).
    pub(crate) fn prewarm(&self, class: StackClass, count: usize) {
        let mut free = self.class_pool(class).free.lock().unwrap();
        for _ in 0..count {
            if free.len() >= self.max_retained {
                break;
            }
            free.push(Stack::new(class, NumaNodeId::ZERO));
        }
    }

    /// Obtain a large stack (the default class). Back-compat entry point.
    pub(crate) fn acquire(&self) -> Stack {
        self.acquire_class(StackClass::Large, NumaNodeId::ZERO)
    }

    /// Obtain a stack of `class`, tagged with NUMA `node`, reusing an idle one
    /// when available or allocating a fresh one otherwise. Never blocks and
    /// never fails while memory is available.
    pub(crate) fn acquire_class(&self, class: StackClass, node: NumaNodeId) -> Stack {
        let pool = self.class_pool(class);
        let live = pool.live.fetch_add(1, Ordering::AcqRel) + 1;
        pool.high_water.fetch_max(live, Ordering::AcqRel);
        if let Some(mut stack) = pool.free.lock().unwrap().pop() {
            stack.node = node;
            return stack;
        }
        Stack::new(class, node)
    }

    /// Return a stack for reuse. Kept on its class's free list up to
    /// `max_retained` entries; surplus stacks are deallocated (by dropping) to
    /// bound memory.
    pub(crate) fn release(&self, stack: Stack) {
        let pool = self.class_pool(stack.class);
        pool.live.fetch_sub(1, Ordering::AcqRel);
        let mut free = pool.free.lock().unwrap();
        if free.len() < self.max_retained {
            free.push(stack);
        }
        // else: `stack` drops here, freeing its region.
    }

    /// Peak number of concurrently live stacks of `class` since creation
    /// (design §16 stack water mark).
    pub(crate) fn high_water(&self, class: StackClass) -> usize {
        self.class_pool(class).high_water.load(Ordering::Acquire)
    }

    /// Current number of live (acquired, not yet released) stacks of `class`.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "diagnostic accessor, used in tests")
    )]
    pub(crate) fn live(&self, class: StackClass) -> usize {
        self.class_pool(class).live.load(Ordering::Acquire)
    }

    /// Number of idle stacks currently retained on `class`'s free list.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "diagnostic accessor, used in tests")
    )]
    pub(crate) fn free_len(&self, class: StackClass) -> usize {
        self.class_pool(class).free.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_classes_have_distinct_sizes() {
        let pool = StackPool::new();
        let small = pool.acquire_class(StackClass::Small, NumaNodeId::ZERO);
        let large = pool.acquire_class(StackClass::Large, NumaNodeId::ZERO);
        assert_eq!(small.class(), StackClass::Small);
        assert_eq!(large.class(), StackClass::Large);
        const { assert!(SMALL_STACK_SIZE < LARGE_STACK_SIZE) };
        pool.release(small);
        pool.release(large);
    }

    #[test]
    fn released_stacks_are_reused_per_class() {
        let pool = StackPool::new();
        let s = pool.acquire_class(StackClass::Small, NumaNodeId::ZERO);
        let top = s.top();
        pool.release(s);
        assert_eq!(pool.free_len(StackClass::Small), 1);
        let s2 = pool.acquire_class(StackClass::Small, NumaNodeId::ZERO);
        // Reused the same backing region.
        assert_eq!(s2.top(), top);
        assert_eq!(pool.free_len(StackClass::Small), 0);
        pool.release(s2);
    }

    #[test]
    fn high_water_tracks_peak_live() {
        let pool = StackPool::new();
        let a = pool.acquire_class(StackClass::Large, NumaNodeId::ZERO);
        let b = pool.acquire_class(StackClass::Large, NumaNodeId::ZERO);
        let c = pool.acquire_class(StackClass::Large, NumaNodeId::ZERO);
        assert_eq!(pool.live(StackClass::Large), 3);
        assert_eq!(pool.high_water(StackClass::Large), 3);
        pool.release(a);
        pool.release(b);
        // Peak stays at 3 even though only one is live now.
        assert_eq!(pool.live(StackClass::Large), 1);
        assert_eq!(pool.high_water(StackClass::Large), 3);
        pool.release(c);
        // Small class untouched.
        assert_eq!(pool.high_water(StackClass::Small), 0);
    }

    #[test]
    fn prewarm_fills_the_free_list() {
        let pool = StackPool::new();
        pool.prewarm(StackClass::Small, 8);
        assert_eq!(pool.free_len(StackClass::Small), 8);
        // Acquiring now reuses prewarmed stacks without raising live beyond 1.
        let s = pool.acquire_class(StackClass::Small, NumaNodeId::ZERO);
        assert_eq!(pool.free_len(StackClass::Small), 7);
        pool.release(s);
    }

    #[test]
    fn node_tag_is_recorded() {
        let pool = StackPool::new();
        let s = pool.acquire_class(StackClass::Large, NumaNodeId::new(2));
        assert_eq!(s.node(), NumaNodeId::new(2));
        pool.release(s);
        // Reacquiring re-tags the reused stack with the new node.
        let s2 = pool.acquire_class(StackClass::Large, NumaNodeId::new(5));
        assert_eq!(s2.node(), NumaNodeId::new(5));
        pool.release(s2);
    }

    #[test]
    fn default_acquire_is_large() {
        let pool = StackPool::new();
        let s = pool.acquire();
        assert_eq!(s.class(), StackClass::Large);
        pool.release(s);
    }
}
