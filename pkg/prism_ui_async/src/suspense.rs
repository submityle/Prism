//! Suspense: render a fallback until the awaited value(s) are ready.
//!
//! Suspense is the "loading state" primitive. It renders a `fallback`
//! [`Element`] while the awaited value is not yet [`Ready`](AsyncState::Ready),
//! and the real view once it is. It does not itself distinguish errors — pair
//! it with an [`error_boundary`](crate::error_boundary) or use
//! [`guarded`](crate::guarded) when you need all three branches.

use alloc::vec::Vec;

use prism_ui::Element;

use crate::state::AsyncState;

/// Renders `ready(value)` when `state` is [`Ready`](AsyncState::Ready),
/// otherwise `fallback`.
///
/// Both the [`Pending`](AsyncState::Pending) and
/// [`Failed`](AsyncState::Failed) states show `fallback`, so a bare `suspense`
/// treats "not ready yet" and "errored" alike. Use
/// [`guarded`](crate::guarded) to split out the error branch.
///
/// # Examples
///
/// ```
/// use prism_ui::Element;
/// use prism_ui_async::{suspense, AsyncState};
///
/// let state: AsyncState<&str, ()> = AsyncState::Ready("hi");
/// let view = suspense(&state, Element::text("loading"), |v| Element::text(*v));
/// assert_eq!(view.text_content(), Some("hi"));
/// ```
pub fn suspense<T, E>(
    state: &AsyncState<T, E>,
    fallback: Element,
    ready: impl FnOnce(&T) -> Element,
) -> Element {
    match state.ready() {
        Some(value) => ready(value),
        None => fallback,
    }
}

/// Renders `fallback` while **any** state in `states` is still
/// [`Pending`](AsyncState::Pending); otherwise renders `ready_builder()`.
///
/// This is the "wait for all" combinator: a view that depends on several
/// resources shows a single loading state until none of them are pending. Note
/// that a [`Failed`](AsyncState::Failed) input does *not* keep the fallback up
/// — only pending does — so an error can surface to a downstream boundary.
///
/// # Examples
///
/// ```
/// use prism_ui::Element;
/// use prism_ui_async::{suspense_all, AsyncState};
///
/// let states: [AsyncState<i32, ()>; 2] = [AsyncState::Ready(1), AsyncState::Ready(2)];
/// let view = suspense_all(&states, Element::text("loading"), || Element::text("done"));
/// assert_eq!(view.text_content(), Some("done"));
/// ```
pub fn suspense_all<T, E>(
    states: &[AsyncState<T, E>],
    fallback: Element,
    ready_builder: impl FnOnce() -> Element,
) -> Element {
    if states.iter().any(AsyncState::is_pending) {
        fallback
    } else {
        ready_builder()
    }
}

/// Collects borrowed success values from `states`, or returns `None` if any
/// state is not [`Ready`](AsyncState::Ready).
///
/// Useful inside a [`suspense_all`] `ready_builder` to pull out every value
/// once they are all known to be present.
#[must_use]
pub fn ready_values<'a, T, E>(states: &'a [AsyncState<T, E>]) -> Option<Vec<&'a T>> {
    let mut values = Vec::with_capacity(states.len());
    for state in states {
        values.push(state.ready()?);
    }
    Some(values)
}
