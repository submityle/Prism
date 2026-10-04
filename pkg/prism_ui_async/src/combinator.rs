//! Combinators for [`AsyncState`]: projection, folding, defaulting, chaining and
//! `Option` conversion.
//!
//! [`AsyncState`] ships with `map`/`map_err` plus the `is_*`/`ready`/`failed`
//! accessors in its defining module. This module completes the surface with the
//! combinators callers reach for when composing async UI state, mirroring the
//! conventions of [`Result`] and [`Option`] so the behaviour is unsurprising:
//!
//! * **Projection** — [`as_ref`](AsyncState::as_ref) and
//!   [`as_mut`](AsyncState::as_mut) borrow the payload without consuming the
//!   state, so a value can be inspected or mutated in place.
//! * **Folding / defaulting** — [`fold`](AsyncState::fold),
//!   [`map_or`](AsyncState::map_or), [`unwrap_or`](AsyncState::unwrap_or),
//!   [`unwrap_or_else`](AsyncState::unwrap_or_else) and
//!   [`unwrap_or_default`](AsyncState::unwrap_or_default) collapse the three
//!   states down to a single value.
//! * **Chaining** — [`and_then`](AsyncState::and_then) sequences another async
//!   step off a `Ready` value, and [`or_else`](AsyncState::or_else) recovers
//!   from a `Failed` one.
//! * **`Option` conversion** — [`ok`](AsyncState::ok) and
//!   [`err`](AsyncState::err) extract the success or error payload.
//!
//! Every combinator is a pure match with no allocation, so the crate stays
//! `no_std` and these helpers are available everywhere [`AsyncState`] is.

use crate::state::AsyncState;

impl<T, E> AsyncState<T, E> {
    /// Borrows the payload, converting from `&AsyncState<T, E>` to an
    /// `AsyncState<&T, &E>` without consuming the original.
    ///
    /// This is the non-consuming entry point for the other combinators: for
    /// example `state.as_ref().map(f)` transforms a `Ready` value by reference.
    #[must_use]
    pub fn as_ref(&self) -> AsyncState<&T, &E> {
        match self {
            AsyncState::Pending => AsyncState::Pending,
            AsyncState::Ready(value) => AsyncState::Ready(value),
            AsyncState::Failed(error) => AsyncState::Failed(error),
        }
    }

    /// Mutably borrows the payload, converting to an
    /// `AsyncState<&mut T, &mut E>` without consuming the original.
    #[must_use]
    pub fn as_mut(&mut self) -> AsyncState<&mut T, &mut E> {
        match self {
            AsyncState::Pending => AsyncState::Pending,
            AsyncState::Ready(value) => AsyncState::Ready(value),
            AsyncState::Failed(error) => AsyncState::Failed(error),
        }
    }

    /// Collapses the three states into a single value by applying exactly one of
    /// the three handlers.
    ///
    /// This is the exhaustive consumer the other defaulting helpers are built
    /// on; it guarantees every state is handled.
    pub fn fold<U>(
        self,
        on_pending: impl FnOnce() -> U,
        on_ready: impl FnOnce(T) -> U,
        on_failed: impl FnOnce(E) -> U,
    ) -> U {
        match self {
            AsyncState::Pending => on_pending(),
            AsyncState::Ready(value) => on_ready(value),
            AsyncState::Failed(error) => on_failed(error),
        }
    }

    /// Maps a `Ready` value with `f`, or returns `default` for `Pending`/
    /// `Failed`.
    pub fn map_or<U>(self, default: U, f: impl FnOnce(T) -> U) -> U {
        match self {
            AsyncState::Ready(value) => f(value),
            AsyncState::Pending | AsyncState::Failed(_) => default,
        }
    }

    /// Returns the `Ready` value, or `default` for `Pending`/`Failed`.
    #[must_use]
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            AsyncState::Ready(value) => value,
            AsyncState::Pending | AsyncState::Failed(_) => default,
        }
    }

    /// Returns the `Ready` value, or computes a fallback with `f` for
    /// `Pending`/`Failed`.
    pub fn unwrap_or_else(self, f: impl FnOnce() -> T) -> T {
        match self {
            AsyncState::Ready(value) => value,
            AsyncState::Pending | AsyncState::Failed(_) => f(),
        }
    }

    /// Returns the `Ready` value, or [`Default::default`] for `Pending`/
    /// `Failed`.
    #[must_use]
    pub fn unwrap_or_default(self) -> T
    where
        T: Default,
    {
        match self {
            AsyncState::Ready(value) => value,
            AsyncState::Pending | AsyncState::Failed(_) => T::default(),
        }
    }

    /// Chains another async step off a `Ready` value.
    ///
    /// `Ready(v)` becomes `f(v)` (which may itself be `Pending`, `Ready` or
    /// `Failed`); `Pending` and `Failed` pass through unchanged. This is the
    /// monadic bind that lets dependent fetches compose.
    #[must_use]
    pub fn and_then<U>(self, f: impl FnOnce(T) -> AsyncState<U, E>) -> AsyncState<U, E> {
        match self {
            AsyncState::Pending => AsyncState::Pending,
            AsyncState::Ready(value) => f(value),
            AsyncState::Failed(error) => AsyncState::Failed(error),
        }
    }

    /// Recovers from a `Failed` state.
    ///
    /// `Failed(e)` becomes `f(e)` (which may resolve to any state); `Pending`
    /// and `Ready` pass through unchanged.
    #[must_use]
    pub fn or_else<F>(self, f: impl FnOnce(E) -> AsyncState<T, F>) -> AsyncState<T, F> {
        match self {
            AsyncState::Pending => AsyncState::Pending,
            AsyncState::Ready(value) => AsyncState::Ready(value),
            AsyncState::Failed(error) => f(error),
        }
    }

    /// Converts to [`Option`], keeping the success value: `Ready(v)` becomes
    /// `Some(v)`, everything else `None`.
    #[must_use]
    pub fn ok(self) -> Option<T> {
        match self {
            AsyncState::Ready(value) => Some(value),
            AsyncState::Pending | AsyncState::Failed(_) => None,
        }
    }

    /// Converts to [`Option`], keeping the error value: `Failed(e)` becomes
    /// `Some(e)`, everything else `None`.
    #[must_use]
    pub fn err(self) -> Option<E> {
        match self {
            AsyncState::Failed(error) => Some(error),
            AsyncState::Pending | AsyncState::Ready(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::state::AsyncState;

    // A deterministic, representative sample of states used as a correctness
    // oracle by the algebraic-law tests below. Enumeration (not RNG) keeps the
    // tests `no_std`-friendly and fully reproducible.
    fn samples() -> [AsyncState<i32, i32>; 4] {
        [
            AsyncState::Pending,
            AsyncState::Ready(7),
            AsyncState::Ready(-3),
            AsyncState::Failed(5),
        ]
    }

    #[test]
    fn as_ref_projects_without_consuming() {
        let ready = AsyncState::<i32, i32>::Ready(9);
        assert_eq!(ready.as_ref(), AsyncState::Ready(&9));
        assert_eq!(ready, AsyncState::Ready(9)); // original still owned

        let failed = AsyncState::<i32, i32>::Failed(2);
        assert_eq!(failed.as_ref(), AsyncState::Failed(&2));

        let pending = AsyncState::<i32, i32>::Pending;
        assert_eq!(pending.as_ref(), AsyncState::Pending);
    }

    #[test]
    fn as_mut_allows_in_place_edit() {
        let mut state = AsyncState::<i32, i32>::Ready(1);
        if let AsyncState::Ready(v) = state.as_mut() {
            *v += 40;
        }
        assert_eq!(state, AsyncState::Ready(41));
    }

    #[test]
    fn fold_dispatches_each_arm() {
        let f = |s: AsyncState<i32, i32>| s.fold(|| 0, |v| v * 2, |e| -e);
        assert_eq!(f(AsyncState::Pending), 0);
        assert_eq!(f(AsyncState::Ready(10)), 20);
        assert_eq!(f(AsyncState::Failed(4)), -4);
    }

    #[test]
    fn defaulting_helpers_known_values() {
        assert_eq!(AsyncState::<i32, i32>::Ready(5).map_or(0, |v| v + 1), 6);
        assert_eq!(AsyncState::<i32, i32>::Failed(9).map_or(0, |v| v + 1), 0);
        assert_eq!(AsyncState::<i32, i32>::Pending.map_or(0, |v| v + 1), 0);

        assert_eq!(AsyncState::<i32, i32>::Ready(5).unwrap_or(99), 5);
        assert_eq!(AsyncState::<i32, i32>::Pending.unwrap_or(99), 99);
        assert_eq!(AsyncState::<i32, i32>::Failed(1).unwrap_or_else(|| 42), 42);
        assert_eq!(AsyncState::<i32, i32>::Pending.unwrap_or_default(), 0);
        assert_eq!(AsyncState::<i32, i32>::Ready(8).unwrap_or_default(), 8);
    }

    #[test]
    fn and_then_or_else_known_values() {
        let step = |v: i32| {
            if v > 0 {
                AsyncState::<i32, i32>::Ready(v * 10)
            } else {
                AsyncState::Failed(v)
            }
        };
        assert_eq!(AsyncState::<i32, i32>::Ready(3).and_then(step), AsyncState::Ready(30));
        assert_eq!(AsyncState::<i32, i32>::Ready(-1).and_then(step), AsyncState::Failed(-1));
        assert_eq!(AsyncState::<i32, i32>::Pending.and_then(step), AsyncState::Pending);
        assert_eq!(AsyncState::<i32, i32>::Failed(2).and_then(step), AsyncState::Failed(2));

        let recover = |e: i32| AsyncState::<i32, i32>::Ready(e + 100);
        assert_eq!(AsyncState::<i32, i32>::Failed(1).or_else(recover), AsyncState::Ready(101));
        assert_eq!(AsyncState::<i32, i32>::Ready(1).or_else(recover), AsyncState::Ready(1));
        assert_eq!(AsyncState::<i32, i32>::Pending.or_else(recover), AsyncState::Pending);
    }

    #[test]
    fn ok_and_err_conversions() {
        assert_eq!(AsyncState::<i32, i32>::Ready(4).ok(), Some(4));
        assert_eq!(AsyncState::<i32, i32>::Failed(4).ok(), None);
        assert_eq!(AsyncState::<i32, i32>::Pending.ok(), None);
        assert_eq!(AsyncState::<i32, i32>::Failed(4).err(), Some(4));
        assert_eq!(AsyncState::<i32, i32>::Ready(4).err(), None);
        assert_eq!(AsyncState::<i32, i32>::Pending.err(), None);
    }

    // --- Algebraic laws (strong oracle over the enumerated sample set) ---

    #[test]
    fn functor_composition_law() {
        // map(g) . map(f) == map(g . f)
        let f = |x: i32| x + 1;
        let g = |x: i32| x * 3;
        for s in samples() {
            let lhs = s.map(f).map(g);
            let rhs = s.map(|x| g(f(x)));
            assert_eq!(lhs, rhs);
        }
    }

    #[test]
    fn monad_left_identity_law() {
        // Ready(x).and_then(k) == k(x)
        let k = |x: i32| {
            if x >= 0 {
                AsyncState::<i32, i32>::Ready(x + 1)
            } else {
                AsyncState::Failed(x)
            }
        };
        for x in [0, 7, -3, 42] {
            assert_eq!(AsyncState::<i32, i32>::Ready(x).and_then(k), k(x));
        }
    }

    #[test]
    fn monad_right_identity_law() {
        // s.and_then(Ready) == s
        for s in samples() {
            assert_eq!(s.and_then(AsyncState::Ready), s);
        }
    }

    #[test]
    fn or_else_right_identity_law() {
        // s.or_else(Failed) == s
        for s in samples() {
            assert_eq!(s.or_else(AsyncState::Failed), s);
        }
    }

    #[test]
    fn unwrap_or_matches_ok_unwrap_or() {
        // unwrap_or(d) is defined by the ok() projection.
        for s in samples() {
            assert_eq!(s.unwrap_or(1234), s.ok().unwrap_or(1234));
        }
    }

    #[test]
    fn as_ref_ok_matches_owned_ok() {
        // as_ref().ok().copied() == clone().ok()
        for s in samples() {
            assert_eq!(s.as_ref().ok().copied(), s.ok());
        }
    }
}
