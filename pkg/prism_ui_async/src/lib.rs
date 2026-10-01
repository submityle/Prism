//! `prism_ui_async` — Loom's async-UI layer: resource state machines,
//! suspense fallbacks and error boundaries.
//!
//! Game UIs constantly await things — assets streaming in, a save file
//! loading, an RPC resolving — yet a UI crate cannot drag in a `std` futures
//! runtime: this crate is `no_std`. So instead of polling futures, Loom models
//! async as an **explicitly-driven state machine**:
//!
//! * [`AsyncState`] is the inert three-state value — `Pending`, `Ready(T)`,
//!   `Failed(E)` — with ergonomic accessors and `map`/`map_err`.
//! * [`Resource`] stores an `AsyncState` inside a reactive
//!   [`Signal`](prism_ui_reactive::Signal), so transitions
//!   ([`resolve`](Resource::resolve), [`fail`](Resource::fail),
//!   [`reload`](Resource::reload)) notify every dependent view automatically.
//! * [`suspense`]/[`suspense_all`] render a fallback while awaiting; a value is
//!   produced only once the awaited state(s) are ready.
//! * [`error_boundary`] routes failures to a recovery view, and [`guarded`]
//!   fuses suspense + error handling into one exhaustive call.
//! * [`all`], [`pending_count`], [`failed_count`] and [`ready_count`] aggregate
//!   a batch of states into a single verdict or tally.
//!
//! Nothing here spawns tasks or blocks; the owner advances a [`Resource`] when
//! real work completes and the reactive graph does the rest.
//!
//! # Example
//!
//! ```
//! use prism_ui::Element;
//! use prism_ui_reactive::Runtime;
//! use prism_ui_async::{guarded, Resource};
//!
//! let rt = Runtime::new();
//! let user: Resource<&str, &str> = Resource::pending(&rt);
//!
//! // A view that always reflects the resource's current state.
//! let render = {
//!     let user = user.clone();
//!     move || {
//!         user.with_state(|state| {
//!             guarded(
//!                 state,
//!                 Element::text("loading…"),
//!                 |err| Element::text(*err),
//!                 |name| Element::text(*name),
//!             )
//!         })
//!     }
//! };
//!
//! assert_eq!(render().text_content(), Some("loading…"));
//!
//! user.resolve("Ada");
//! assert_eq!(render().text_content(), Some("Ada"));
//!
//! user.fail("offline");
//! assert_eq!(render().text_content(), Some("offline"));
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod aggregate;
mod boundary;
mod resource;
mod state;
mod suspense;

pub use aggregate::{all, failed_count, pending_count, ready_count};
pub use boundary::{error_boundary, guarded};
pub use resource::Resource;
pub use state::AsyncState;
pub use suspense::{ready_values, suspense, suspense_all};

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::cell::RefCell;

    use prism_ui::Element;
    use prism_ui_reactive::Runtime;

    use super::*;

    #[test]
    fn async_state_predicates_and_accessors() {
        let pending: AsyncState<i32, &str> = AsyncState::Pending;
        let ready: AsyncState<i32, &str> = AsyncState::Ready(7);
        let failed: AsyncState<i32, &str> = AsyncState::Failed("nope");

        assert!(pending.is_pending() && !pending.is_ready() && !pending.is_failed());
        assert!(ready.is_ready() && !ready.is_pending() && !ready.is_failed());
        assert!(failed.is_failed() && !failed.is_pending() && !failed.is_ready());

        assert_eq!(ready.ready(), Some(&7));
        assert_eq!(ready.failed(), None);
        assert_eq!(failed.failed(), Some(&"nope"));
        assert_eq!(failed.ready(), None);
        assert_eq!(pending.ready(), None);
    }

    #[test]
    fn async_state_map_and_map_err() {
        let ready: AsyncState<i32, &str> = AsyncState::Ready(3);
        assert_eq!(ready.map(|n| n + 1), AsyncState::Ready(4));

        let failed: AsyncState<i32, i32> = AsyncState::Failed(2);
        assert_eq!(failed.map_err(|e| e * 10), AsyncState::Failed(20));

        let pending: AsyncState<i32, &str> = AsyncState::Pending;
        assert_eq!(pending.map(|n| n + 1), AsyncState::Pending);
    }

    #[test]
    fn resource_transitions_pending_ready_failed_reload() {
        let rt = Runtime::new();
        let res: Resource<i32, &str> = Resource::pending(&rt);
        assert_eq!(res.state(), AsyncState::Pending);

        res.resolve(42);
        assert_eq!(res.state(), AsyncState::Ready(42));
        assert!(res.is_ready());

        res.reload();
        assert_eq!(res.state(), AsyncState::Pending);

        res.fail("boom");
        assert_eq!(res.state(), AsyncState::Failed("boom"));
        assert!(res.is_failed());
    }

    #[test]
    fn resource_notifies_reactive_observers_on_transition() {
        let rt = Runtime::new();
        let res: Resource<i32, &str> = Resource::pending(&rt);

        // A memo that classifies the state into a small, comparable tag.
        let tag = rt.memo({
            let res = res.clone();
            move || match res.state() {
                AsyncState::Pending => 0u8,
                AsyncState::Ready(_) => 1,
                AsyncState::Failed(_) => 2,
            }
        });

        let log = Rc::new(RefCell::new(Vec::new()));
        let _effect = rt.effect({
            let tag = tag.clone();
            let log = log.clone();
            move || log.borrow_mut().push(tag.get())
        });

        assert_eq!(*log.borrow(), vec![0]);
        res.resolve(1);
        assert_eq!(*log.borrow(), vec![0, 1]);
        res.fail("x");
        assert_eq!(*log.borrow(), vec![0, 1, 2]);
        res.reload();
        assert_eq!(*log.borrow(), vec![0, 1, 2, 0]);
    }

    #[test]
    fn suspense_picks_fallback_then_ready() {
        let pending: AsyncState<&str, ()> = AsyncState::Pending;
        let view = suspense(&pending, Element::text("loading"), |v| Element::text(*v));
        assert_eq!(view.text_content(), Some("loading"));

        let ready: AsyncState<&str, ()> = AsyncState::Ready("data");
        let view = suspense(&ready, Element::text("loading"), |v| Element::text(*v));
        assert_eq!(view.text_content(), Some("data"));
    }

    #[test]
    fn suspense_all_waits_for_every_pending() {
        let mixed: [AsyncState<i32, ()>; 2] = [AsyncState::Ready(1), AsyncState::Pending];
        let view = suspense_all(&mixed, Element::text("loading"), || Element::text("done"));
        assert_eq!(view.text_content(), Some("loading"));

        let settled: [AsyncState<i32, ()>; 2] = [AsyncState::Ready(1), AsyncState::Ready(2)];
        let view = suspense_all(&settled, Element::text("loading"), || {
            let values = ready_values(&settled).expect("all ready");
            Element::text(alloc::format!("{}", values[0] + values[1]))
        });
        assert_eq!(view.text_content(), Some("3"));
    }

    #[test]
    fn error_boundary_routes_failed_and_ready() {
        let failed: AsyncState<i32, &str> = AsyncState::Failed("err");
        let view = error_boundary(&failed, |e| Element::text(*e), |_| Element::text("ok"));
        assert_eq!(view.text_content(), Some("err"));

        let ready: AsyncState<i32, &str> = AsyncState::Ready(9);
        let view = error_boundary(&ready, |e| Element::text(*e), |_| Element::text("ok"));
        assert_eq!(view.text_content(), Some("ok"));
    }

    #[test]
    fn guarded_covers_all_three_branches() {
        let make = |state: &AsyncState<&str, &str>| {
            guarded(
                state,
                Element::text("loading"),
                |e| Element::text(*e),
                |v| Element::text(*v),
            )
        };

        assert_eq!(make(&AsyncState::Pending).text_content(), Some("loading"));
        assert_eq!(make(&AsyncState::Failed("e")).text_content(), Some("e"));
        assert_eq!(make(&AsyncState::Ready("v")).text_content(), Some("v"));
    }

    #[test]
    fn all_aggregation_and_counts() {
        let pending_mix: [AsyncState<i32, &str>; 3] = [
            AsyncState::Ready(1),
            AsyncState::Pending,
            AsyncState::Failed("e"),
        ];
        // Pending wins even though a failure is present.
        assert_eq!(all(&pending_mix), AsyncState::Pending);
        assert_eq!(pending_count(&pending_mix), 1);
        assert_eq!(failed_count(&pending_mix), 1);
        assert_eq!(ready_count(&pending_mix), 1);

        let failed_mix: [AsyncState<i32, &str>; 3] = [
            AsyncState::Ready(1),
            AsyncState::Failed("first"),
            AsyncState::Failed("second"),
        ];
        assert_eq!(all(&failed_mix), AsyncState::Failed("first"));

        let ready_all: [AsyncState<i32, &str>; 2] = [AsyncState::Ready(1), AsyncState::Ready(2)];
        assert_eq!(all(&ready_all), AsyncState::Ready(()));

        let empty: [AsyncState<i32, &str>; 0] = [];
        assert_eq!(all(&empty), AsyncState::Ready(()));
    }
}
