//! The three-state async value at the heart of this crate.
//!
//! [`AsyncState`] models a value that is produced out-of-band — a network
//! fetch, an asset load, a background computation — without pulling in a
//! futures runtime. It is a plain, cloneable enum that a caller advances
//! explicitly; the reactive wiring lives in [`Resource`](crate::Resource).

/// The lifecycle of an asynchronously produced value.
///
/// A value starts [`Pending`](AsyncState::Pending), then transitions exactly
/// once to either [`Ready`](AsyncState::Ready) on success or
/// [`Failed`](AsyncState::Failed) on error. Reloading returns it to
/// `Pending`. The type is deliberately inert: it holds no runtime and performs
/// no polling, so it is cheap to clone and store in a reactive signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AsyncState<T, E> {
    /// The value is still being produced; no result is available yet.
    Pending,
    /// The value was produced successfully.
    Ready(T),
    /// Production failed with the carried error.
    Failed(E),
}

impl<T, E> AsyncState<T, E> {
    /// Returns `true` while the value is still [`Pending`](AsyncState::Pending).
    #[must_use]
    pub fn is_pending(&self) -> bool {
        matches!(self, AsyncState::Pending)
    }

    /// Returns `true` once the value is [`Ready`](AsyncState::Ready).
    #[must_use]
    pub fn is_ready(&self) -> bool {
        matches!(self, AsyncState::Ready(_))
    }

    /// Returns `true` once production has [`Failed`](AsyncState::Failed).
    #[must_use]
    pub fn is_failed(&self) -> bool {
        matches!(self, AsyncState::Failed(_))
    }

    /// Borrows the success value, or `None` unless [`Ready`](AsyncState::Ready).
    #[must_use]
    pub fn ready(&self) -> Option<&T> {
        match self {
            AsyncState::Ready(value) => Some(value),
            _ => None,
        }
    }

    /// Borrows the error, or `None` unless [`Failed`](AsyncState::Failed).
    #[must_use]
    pub fn failed(&self) -> Option<&E> {
        match self {
            AsyncState::Failed(error) => Some(error),
            _ => None,
        }
    }

    /// Transforms the success value in place, leaving the other states intact.
    #[must_use]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> AsyncState<U, E> {
        match self {
            AsyncState::Pending => AsyncState::Pending,
            AsyncState::Ready(value) => AsyncState::Ready(f(value)),
            AsyncState::Failed(error) => AsyncState::Failed(error),
        }
    }

    /// Transforms the error in place, leaving the other states intact.
    #[must_use]
    pub fn map_err<F>(self, f: impl FnOnce(E) -> F) -> AsyncState<T, F> {
        match self {
            AsyncState::Pending => AsyncState::Pending,
            AsyncState::Ready(value) => AsyncState::Ready(value),
            AsyncState::Failed(error) => AsyncState::Failed(f(error)),
        }
    }
}
