//! Error boundaries: route a [`Failed`](AsyncState::Failed) state to a
//! recovery view instead of letting it reach the happy path.
//!
//! [`error_boundary`] handles the success/error split; [`guarded`] layers a
//! suspense fallback on top so all three async states are covered by one call.

use prism_ui::Element;

use crate::state::AsyncState;

/// Routes a [`Failed`](AsyncState::Failed) state to `on_error` and a
/// [`Ready`](AsyncState::Ready) state to `on_ok`.
///
/// While [`Pending`](AsyncState::Pending) there is nothing to show and no error
/// to recover from, so this renders an empty [`Element::box_`] placeholder;
/// wrap the resource in [`suspense`](crate::suspense) or use [`guarded`] to
/// supply a loading view for that case.
///
/// # Examples
///
/// ```
/// use prism_ui::Element;
/// use prism_ui_async::{error_boundary, AsyncState};
///
/// let state: AsyncState<(), &str> = AsyncState::Failed("boom");
/// let view = error_boundary(
///     &state,
///     |e| Element::text(*e),
///     |()| Element::text("ok"),
/// );
/// assert_eq!(view.text_content(), Some("boom"));
/// ```
pub fn error_boundary<T, E>(
    state: &AsyncState<T, E>,
    on_error: impl FnOnce(&E) -> Element,
    on_ok: impl FnOnce(&T) -> Element,
) -> Element {
    match state {
        AsyncState::Ready(value) => on_ok(value),
        AsyncState::Failed(error) => on_error(error),
        AsyncState::Pending => Element::box_(),
    }
}

/// The all-in-one combinator: covers every [`AsyncState`] branch.
///
/// * [`Pending`](AsyncState::Pending) renders `fallback`.
/// * [`Failed`](AsyncState::Failed) renders `on_error(error)`.
/// * [`Ready`](AsyncState::Ready) renders `on_ok(value)`.
///
/// This is [`suspense`](crate::suspense) and [`error_boundary`] fused into a
/// single, exhaustive match — the recommended way to render a resource.
///
/// # Examples
///
/// ```
/// use prism_ui::Element;
/// use prism_ui_async::{guarded, AsyncState};
///
/// let state: AsyncState<i32, &str> = AsyncState::Pending;
/// let view = guarded(
///     &state,
///     Element::text("loading"),
///     |e| Element::text(*e),
///     |_n| Element::text("ready"),
/// );
/// assert_eq!(view.text_content(), Some("loading"));
/// ```
pub fn guarded<T, E>(
    state: &AsyncState<T, E>,
    fallback: Element,
    on_error: impl FnOnce(&E) -> Element,
    on_ok: impl FnOnce(&T) -> Element,
) -> Element {
    match state {
        AsyncState::Pending => fallback,
        AsyncState::Failed(error) => on_error(error),
        AsyncState::Ready(value) => on_ok(value),
    }
}
