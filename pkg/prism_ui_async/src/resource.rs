//! A reactive handle around an [`AsyncState`].
//!
//! A [`Resource`] stores its state in a [`Signal`], so reading it inside a memo
//! or effect records a dependency and every transition (`pending → ready`,
//! `pending → failed`, `reload → pending`) notifies observers automatically.

use prism_ui_reactive::{Runtime, Signal};

use crate::state::AsyncState;

/// A reactive, explicitly-driven async value.
///
/// `Resource` owns a [`Signal<AsyncState<T, E>>`]. It never polls on its own:
/// the owner drives it by calling [`resolve`](Resource::resolve),
/// [`fail`](Resource::fail) or [`reload`](Resource::reload). Because the state
/// lives in a signal, any reactive computation that reads it via
/// [`state`](Resource::state) or [`with_state`](Resource::with_state) is
/// re-run on the next transition.
///
/// `Resource` is a cheap, clonable handle; clones share one underlying signal.
pub struct Resource<T: Clone + 'static, E: Clone + 'static> {
    state: Signal<AsyncState<T, E>>,
}

impl<T: Clone + 'static, E: Clone + 'static> Clone for Resource<T, E> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<T: Clone + 'static, E: Clone + 'static> Resource<T, E> {
    /// Creates a resource that starts in the [`Pending`](AsyncState::Pending)
    /// state on the given runtime.
    #[must_use]
    pub fn pending(rt: &Runtime) -> Self {
        Self {
            state: rt.signal(AsyncState::Pending),
        }
    }

    /// Creates a resource already resolved to `value`.
    #[must_use]
    pub fn ready(rt: &Runtime, value: T) -> Self {
        Self {
            state: rt.signal(AsyncState::Ready(value)),
        }
    }

    /// Creates a resource already failed with `error`.
    #[must_use]
    pub fn failed(rt: &Runtime, error: E) -> Self {
        Self {
            state: rt.signal(AsyncState::Failed(error)),
        }
    }

    /// The runtime this resource belongs to.
    #[must_use]
    pub fn runtime(&self) -> &Runtime {
        self.state.runtime()
    }

    /// The underlying signal, for wiring into memos or effects directly.
    #[must_use]
    pub fn signal(&self) -> &Signal<AsyncState<T, E>> {
        &self.state
    }

    /// Transitions to [`Ready`](AsyncState::Ready), notifying observers.
    pub fn resolve(&self, value: T) {
        self.state.set(AsyncState::Ready(value));
    }

    /// Transitions to [`Failed`](AsyncState::Failed), notifying observers.
    pub fn fail(&self, error: E) {
        self.state.set(AsyncState::Failed(error));
    }

    /// Transitions back to [`Pending`](AsyncState::Pending), notifying
    /// observers — e.g. to re-trigger a fetch.
    pub fn reload(&self) {
        self.state.set(AsyncState::Pending);
    }

    /// Reads a clone of the current state, recording a reactive dependency.
    #[must_use]
    pub fn state(&self) -> AsyncState<T, E> {
        self.state.get()
    }

    /// Reads the current state through `f`, recording a reactive dependency,
    /// without cloning it.
    pub fn with_state<R>(&self, f: impl FnOnce(&AsyncState<T, E>) -> R) -> R {
        self.state.with(f)
    }

    /// Returns `true` while the resource is [`Pending`](AsyncState::Pending).
    /// Records a reactive dependency.
    #[must_use]
    pub fn is_pending(&self) -> bool {
        self.with_state(AsyncState::is_pending)
    }

    /// Returns `true` once the resource is [`Ready`](AsyncState::Ready).
    /// Records a reactive dependency.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.with_state(AsyncState::is_ready)
    }

    /// Returns `true` once the resource has [`Failed`](AsyncState::Failed).
    /// Records a reactive dependency.
    #[must_use]
    pub fn is_failed(&self) -> bool {
        self.with_state(AsyncState::is_failed)
    }
}
