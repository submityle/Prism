//! A hand-rolled [`RawWaker`] vtable over an `Arc<W>`.
//!
//! The standard library offers [`std::task::Wake`], but M3 implements the
//! vtable by hand (the design calls for a "real `RawWaker`/`Waker` vtable") so
//! the ownership protocol is explicit and auditable. The data pointer of each
//! [`RawWaker`] is a *thin* `Arc<W>` (recovered with [`Arc::into_raw`] /
//! [`Arc::from_raw`]); every vtable entry is monomorphized for the concrete
//! `W`, so the pointee type is always known at the call site.
//!
//! Any type that can be re-scheduled from a bare `&Arc<Self>` implements
//! [`WakeTask`]; [`waker_of`] turns an `Arc<W>` into a [`Waker`]. Two callers
//! use it: the async task harness (re-enqueues itself) and `block_on`'s
//! thread-notify (unparks the blocked thread).

use alloc::sync::Arc;
use std::task::{RawWaker, RawWakerVTable, Waker};

/// A target that a [`Waker`] can re-schedule by reference to its `Arc`.
///
/// Implementors are reference-counted; cloning the waker clones the `Arc` and
/// dropping the waker drops one reference, so the target lives exactly as long
/// as the outstanding wakers plus its owner.
pub(crate) trait WakeTask: Send + Sync + 'static {
    /// Schedule the task for progress without consuming the `Arc`.
    fn wake_by_ref(self: &Arc<Self>);

    /// Schedule the task, consuming this reference. Defaults to
    /// [`WakeTask::wake_by_ref`] followed by dropping the `Arc`.
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
}

/// Build a [`Waker`] that re-schedules `arc`'s target when woken.
pub(crate) fn waker_of<W: WakeTask>(arc: Arc<W>) -> Waker {
    // SAFETY: `raw_waker` returns a `RawWaker` whose data pointer is a live
    // `Arc<W>` (one owned reference transferred in) paired with the matching
    // monomorphized vtable, so the clone/wake/drop contract of `Waker` holds.
    #[expect(unsafe_code, reason = "construct a Waker from our audited RawWaker")]
    unsafe {
        Waker::from_raw(raw_waker(arc))
    }
}

/// Turn one owned `Arc<W>` reference into a [`RawWaker`] (thin data pointer plus
/// the monomorphized vtable). Consumes the reference; the vtable's `drop`/`wake`
/// entries are responsible for releasing it.
fn raw_waker<W: WakeTask>(arc: Arc<W>) -> RawWaker {
    let data = Arc::into_raw(arc).cast::<()>();
    RawWaker::new(data, vtable::<W>())
}

/// The `'static` vtable for `W`. Each `&RawWakerVTable::new(..)` of `const fn`
/// arguments is promoted to a `'static` by rvalue static promotion.
fn vtable<W: WakeTask>() -> &'static RawWakerVTable {
    &RawWakerVTable::new(
        clone_raw::<W>,
        wake_raw::<W>,
        wake_by_ref_raw::<W>,
        drop_raw::<W>,
    )
}

/// Clone entry: bump the refcount and hand back a second `RawWaker` without
/// disturbing the reference `data` already represents.
///
/// # Safety
/// `data` must be a pointer produced by `Arc::into_raw::<W>` that is still
/// owned by the waker being cloned (the standard [`RawWaker`] contract).
#[expect(
    unsafe_code,
    reason = "RawWaker clone: recover the Arc, clone it, re-leak the original"
)]
unsafe fn clone_raw<W: WakeTask>(data: *const ()) -> RawWaker {
    // SAFETY: `data` is a pointer previously produced by `Arc::into_raw::<W>`
    // and still owned by the waker being cloned.
    let arc = unsafe { Arc::from_raw(data.cast::<W>()) };
    let cloned = Arc::clone(&arc);
    // Keep the original reference alive (it still backs the source waker).
    let _ = Arc::into_raw(arc);
    raw_waker(cloned)
}

/// Wake entry: consume this waker's reference and schedule the target.
///
/// # Safety
/// `data` must be a pointer produced by `Arc::into_raw::<W>` whose owned
/// reference is transferred into this call (the standard [`RawWaker`] contract).
#[expect(unsafe_code, reason = "RawWaker wake: consume the owned Arc reference")]
unsafe fn wake_raw<W: WakeTask>(data: *const ()) {
    // SAFETY: `data` came from `Arc::into_raw::<W>` and this call owns exactly
    // one reference, which we take ownership of here and consume via `wake`.
    let arc = unsafe { Arc::from_raw(data.cast::<W>()) };
    WakeTask::wake(arc);
}

/// Wake-by-ref entry: schedule the target without consuming the reference.
///
/// # Safety
/// `data` must be a pointer produced by `Arc::into_raw::<W>` that remains owned
/// by the live waker across the call (the standard [`RawWaker`] contract).
#[expect(
    unsafe_code,
    reason = "RawWaker wake_by_ref: borrow the Arc, then re-leak it"
)]
unsafe fn wake_by_ref_raw<W: WakeTask>(data: *const ()) {
    // SAFETY: `data` came from `Arc::into_raw::<W>`; we temporarily reconstruct
    // the `Arc` to borrow it, then re-leak it so the refcount is unchanged.
    let arc = unsafe { Arc::from_raw(data.cast::<W>()) };
    arc.wake_by_ref();
    let _ = Arc::into_raw(arc);
}

/// Drop entry: release this waker's reference.
///
/// # Safety
/// `data` must be a pointer produced by `Arc::into_raw::<W>` whose owned
/// reference is released exactly once here (the standard [`RawWaker`] contract).
#[expect(unsafe_code, reason = "RawWaker drop: release the owned Arc reference")]
unsafe fn drop_raw<W: WakeTask>(data: *const ()) {
    // SAFETY: `data` came from `Arc::into_raw::<W>` and this waker owns the
    // reference it represents; dropping the reconstructed `Arc` frees it once.
    drop(unsafe { Arc::from_raw(data.cast::<W>()) });
}
