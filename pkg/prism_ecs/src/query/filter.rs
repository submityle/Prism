//! [`QueryFilter`]: archetype-level predicates that narrow which rows a query
//! visits without yielding any data.
//!
//! Filters are evaluated per archetype (not per row), so they are free: an
//! archetype that fails the filter is skipped wholesale. The supported filters
//! mirror the design doc (§7):
//!
//! - `()` — the empty filter, matches everything.
//! - [`With<T>`] — the archetype must have component `T`.
//! - [`Without<T>`] — the archetype must not have component `T`.
//! - [`Or<(F0, F1, ...)>`] — at least one inner filter matches.
//! - tuples of filters — every element matches (logical AND).
//!
//! Change-detection filters (`Added<T>` / `Changed<T>`) are intentionally **not**
//! part of M0: they require the double-layer tick + chunk-version machinery
//! that lands in M2 (design §10). Shipping them now as always-matching stubs
//! would be a fake implementation, so they are deferred until they can be real.

use core::marker::PhantomData;

use crate::archetype::Archetype;
use crate::component::{Component, ComponentId, Components};

/// An archetype-level predicate that narrows a query's matched archetypes.
///
/// # Safety
/// This trait is `unsafe` to implement because [`QueryFilter::matches`] gates
/// which archetypes a query touches; an incorrect implementation cannot cause
/// memory unsafety on its own (filters access no data), but it is grouped with
/// the other query traits for a consistent, auditable surface.
pub unsafe trait QueryFilter {
    /// World-static resolved state (component ids), computed once.
    type State: Send + Sync;

    /// Resolve this filter's [`State`](Self::State), registering any named
    /// component types.
    fn init_state(components: &mut Components) -> Self::State;

    /// Whether `archetype` passes this filter.
    fn matches(state: &Self::State, archetype: &Archetype) -> bool;
}

/// Matches archetypes that **have** component `T` (without reading it).
pub struct With<T>(PhantomData<fn() -> T>);

/// Matches archetypes that **lack** component `T`.
pub struct Without<T>(PhantomData<fn() -> T>);

/// Matches archetypes where **at least one** inner filter matches (logical OR).
///
/// `F` is a tuple of filters, e.g. `Or<(With<A>, Without<B>)>`.
pub struct Or<F>(PhantomData<fn() -> F>);

// SAFETY: `With<T>` reads no data; `matches` is a pure archetype membership test.
unsafe impl<T: Component> QueryFilter for With<T> {
    type State = ComponentId;

    fn init_state(components: &mut Components) -> Self::State {
        components.register::<T>()
    }

    fn matches(state: &Self::State, archetype: &Archetype) -> bool {
        archetype.contains(*state)
    }
}

// SAFETY: `Without<T>` reads no data; `matches` is a pure archetype test.
unsafe impl<T: Component> QueryFilter for Without<T> {
    type State = ComponentId;

    fn init_state(components: &mut Components) -> Self::State {
        components.register::<T>()
    }

    fn matches(state: &Self::State, archetype: &Archetype) -> bool {
        !archetype.contains(*state)
    }
}

// SAFETY: the empty filter matches everything and touches nothing.
unsafe impl QueryFilter for () {
    type State = ();

    fn init_state(_components: &mut Components) -> Self::State {}

    fn matches(_state: &Self::State, _archetype: &Archetype) -> bool {
        true
    }
}

macro_rules! impl_filter_tuple {
    ($($F:ident),+) => {
        // SAFETY: each element is a `QueryFilter`; the AND combinator only
        // narrows the matched set and reads no data.
        #[allow(non_snake_case)]
        unsafe impl<$($F: QueryFilter),+> QueryFilter for ($($F,)+) {
            type State = ($($F::State,)+);

            fn init_state(components: &mut Components) -> Self::State {
                ($($F::init_state(components),)+)
            }

            fn matches(state: &Self::State, archetype: &Archetype) -> bool {
                let ($($F,)+) = state;
                $($F::matches($F, archetype))&&+
            }
        }

        // SAFETY: `Or` over the same element set; only the combinator differs
        // (OR instead of AND). Still reads no data.
        #[allow(non_snake_case)]
        unsafe impl<$($F: QueryFilter),+> QueryFilter for Or<($($F,)+)> {
            type State = ($($F::State,)+);

            fn init_state(components: &mut Components) -> Self::State {
                ($($F::init_state(components),)+)
            }

            fn matches(state: &Self::State, archetype: &Archetype) -> bool {
                let ($($F,)+) = state;
                $($F::matches($F, archetype))||+
            }
        }
    };
}

impl_filter_tuple!(A);
impl_filter_tuple!(A, B);
impl_filter_tuple!(A, B, C);
impl_filter_tuple!(A, B, C, D);
impl_filter_tuple!(A, B, C, D, E);
impl_filter_tuple!(A, B, C, D, E, F);
impl_filter_tuple!(A, B, C, D, E, F, G);
impl_filter_tuple!(A, B, C, D, E, F, G, H);
