//! Integration tests exercising the public `prism_ui_async` API end to end.

extern crate alloc;

use alloc::rc::Rc;
use core::cell::RefCell;

use prism_ui::Element;
use prism_ui_async::{
    all, error_boundary, failed_count, guarded, pending_count, ready_count, ready_values, suspense,
    suspense_all, AsyncState, Resource,
};
use prism_ui_reactive::Runtime;

#[test]
fn resource_drives_a_reactive_view_through_its_lifecycle() {
    let rt = Runtime::new();
    let res: Resource<&str, &str> = Resource::pending(&rt);

    // The rendered text, recomputed reactively from the resource state.
    let text = rt.memo({
        let res = res.clone();
        move || {
            res.with_state(|state| {
                guarded(
                    state,
                    Element::text("loading"),
                    |e| Element::text(*e),
                    |v| Element::text(*v),
                )
            })
            .text_content()
            .map(str::to_string)
            .unwrap_or_default()
        }
    });

    let log = Rc::new(RefCell::new(Vec::new()));
    let _effect = rt.effect({
        let text = text.clone();
        let log = log.clone();
        move || log.borrow_mut().push(text.get())
    });

    assert_eq!(*log.borrow(), vec!["loading".to_string()]);

    res.resolve("payload");
    assert_eq!(
        *log.borrow(),
        vec!["loading".to_string(), "payload".to_string()]
    );

    res.reload();
    res.fail("offline");
    assert_eq!(
        *log.borrow(),
        vec![
            "loading".to_string(),
            "payload".to_string(),
            "loading".to_string(),
            "offline".to_string(),
        ]
    );
}

#[test]
fn suspense_and_suspense_all_pick_the_right_branch() {
    let pending: AsyncState<i32, ()> = AsyncState::Pending;
    assert_eq!(
        suspense(&pending, Element::text("wait"), |n| Element::text(
            n.to_string()
        ))
        .text_content(),
        Some("wait")
    );

    let ready: AsyncState<i32, ()> = AsyncState::Ready(5);
    assert_eq!(
        suspense(&ready, Element::text("wait"), |n| Element::text(
            n.to_string()
        ))
        .text_content(),
        Some("5")
    );

    let mixed: [AsyncState<i32, ()>; 3] = [
        AsyncState::Ready(1),
        AsyncState::Pending,
        AsyncState::Ready(3),
    ];
    assert_eq!(
        suspense_all(&mixed, Element::text("wait"), || Element::text("ready")).text_content(),
        Some("wait")
    );

    let settled: [AsyncState<i32, ()>; 3] = [
        AsyncState::Ready(1),
        AsyncState::Ready(2),
        AsyncState::Ready(3),
    ];
    let view = suspense_all(&settled, Element::text("wait"), || {
        let values = ready_values(&settled).expect("all values present");
        let sum: i32 = values.into_iter().sum();
        Element::text(sum.to_string())
    });
    assert_eq!(view.text_content(), Some("6"));
}

#[test]
fn error_boundary_and_guarded_cover_every_state() {
    let failed: AsyncState<i32, &str> = AsyncState::Failed("bad");
    assert_eq!(
        error_boundary(&failed, |e| Element::text(*e), |_| Element::text("ok")).text_content(),
        Some("bad")
    );

    let ready: AsyncState<i32, &str> = AsyncState::Ready(1);
    assert_eq!(
        error_boundary(&ready, |e| Element::text(*e), |_| Element::text("ok")).text_content(),
        Some("ok")
    );

    // error_boundary renders an empty box while pending.
    let pending: AsyncState<i32, &str> = AsyncState::Pending;
    let placeholder = error_boundary(&pending, |e| Element::text(*e), |_| Element::text("ok"));
    assert_eq!(placeholder.text_content(), None);

    let branch = |state: &AsyncState<&str, &str>| {
        guarded(
            state,
            Element::text("loading"),
            |e| Element::text(*e),
            |v| Element::text(*v),
        )
        .text_content()
        .map(str::to_string)
    };
    assert_eq!(branch(&AsyncState::Pending).as_deref(), Some("loading"));
    assert_eq!(branch(&AsyncState::Failed("e")).as_deref(), Some("e"));
    assert_eq!(branch(&AsyncState::Ready("v")).as_deref(), Some("v"));
}

#[test]
fn all_and_counts_summarise_a_batch() {
    let states: [AsyncState<i32, &str>; 4] = [
        AsyncState::Ready(1),
        AsyncState::Pending,
        AsyncState::Failed("x"),
        AsyncState::Ready(2),
    ];
    // Pending has priority over the failure.
    assert_eq!(all(&states), AsyncState::Pending);
    assert_eq!(pending_count(&states), 1);
    assert_eq!(failed_count(&states), 1);
    assert_eq!(ready_count(&states), 2);

    let failures: [AsyncState<i32, &str>; 2] =
        [AsyncState::Failed("first"), AsyncState::Failed("second")];
    assert_eq!(all(&failures), AsyncState::Failed("first"));

    let done: [AsyncState<i32, &str>; 2] = [AsyncState::Ready(1), AsyncState::Ready(2)];
    assert_eq!(all(&done), AsyncState::Ready(()));
}
