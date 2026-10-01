//! Aggregating many [`AsyncState`]s into one summary state and tallies.
//!
//! When a view awaits a batch of resources it usually wants a single verdict:
//! are we still loading, did something fail, or is everything ready?
//! [`all`] answers that, and [`pending_count`]/[`failed_count`] expose the raw
//! tallies for progress indicators.

use crate::state::AsyncState;

/// Combines a slice of states into one summary over `()`.
///
/// Precedence is **pending first**, then failure:
///
/// * if any state is [`Pending`](AsyncState::Pending), the result is
///   [`Pending`](AsyncState::Pending);
/// * otherwise, if any state is [`Failed`](AsyncState::Failed), the result is
///   [`Failed`](AsyncState::Failed) carrying a clone of the **first** error in
///   slice order;
/// * otherwise every state is [`Ready`](AsyncState::Ready) and the result is
///   `Ready(())`.
///
/// An empty slice is vacuously `Ready(())`.
///
/// # Examples
///
/// ```
/// use prism_ui_async::{all, AsyncState};
///
/// let states: [AsyncState<i32, &str>; 2] =
///     [AsyncState::Ready(1), AsyncState::Failed("bad")];
/// assert_eq!(all(&states), AsyncState::Failed("bad"));
/// ```
#[must_use]
pub fn all<T, E: Clone>(states: &[AsyncState<T, E>]) -> AsyncState<(), E> {
    if states.iter().any(AsyncState::is_pending) {
        return AsyncState::Pending;
    }
    for state in states {
        if let AsyncState::Failed(error) = state {
            return AsyncState::Failed(error.clone());
        }
    }
    AsyncState::Ready(())
}

/// Counts how many states are still [`Pending`](AsyncState::Pending).
#[must_use]
pub fn pending_count<T, E>(states: &[AsyncState<T, E>]) -> usize {
    states.iter().filter(|s| s.is_pending()).count()
}

/// Counts how many states have [`Failed`](AsyncState::Failed).
#[must_use]
pub fn failed_count<T, E>(states: &[AsyncState<T, E>]) -> usize {
    states.iter().filter(|s| s.is_failed()).count()
}

/// Counts how many states are [`Ready`](AsyncState::Ready).
#[must_use]
pub fn ready_count<T, E>(states: &[AsyncState<T, E>]) -> usize {
    states.iter().filter(|s| s.is_ready()).count()
}
