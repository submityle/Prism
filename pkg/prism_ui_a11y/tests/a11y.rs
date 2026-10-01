//! Integration tests for the `prism_ui_a11y` accessibility layer.

use prism_ui::Key;
use prism_ui_a11y::{
    screen_reader_text, A11yNode, A11yTree, AriaState, Arrow, FocusMove, FocusOrder, KeyInput,
    KeyboardNav, Label, LiveRegion, NavAction, Politeness, Role,
};

fn k(id: i64) -> Key {
    Key::Int(id)
}

/// Builds a representative dialog: title heading, two inputs (one disabled, one
/// hidden), a labelled checkbox and OK/Cancel buttons with explicit tab order.
fn dialog_tree() -> A11yTree {
    let mut tree = A11yTree::new();
    tree.insert(
        A11yNode::builder(k(1), Role::Dialog)
            .label(Label::text("Preferences"))
            .focusable(false)
            .build(),
    );
    tree.insert(
        A11yNode::builder(k(2), Role::Heading { level: 1 })
            .label(Label::text("Preferences"))
            .build(),
    );
    tree.insert(
        A11yNode::builder(k(3), Role::Textbox)
            .label(Label::text("Name"))
            .tab_index(0)
            .build(),
    );
    tree.insert(
        A11yNode::builder(k(4), Role::Textbox)
            .label(Label::text("Legacy"))
            .state(AriaState::new().disabled(true))
            .build(),
    );
    tree.insert(
        A11yNode::builder(k(5), Role::Textbox)
            .label(Label::text("Secret"))
            .state(AriaState::new().hidden(true))
            .build(),
    );
    tree.insert(
        A11yNode::builder(k(6), Role::Checkbox)
            .label(Label::labelled_by(k(7)))
            .state(AriaState::new().checked(true).required(true))
            .build(),
    );
    tree.insert(
        A11yNode::builder(k(7), Role::Presentation)
            .label(Label::text("Remember me"))
            .build(),
    );
    // OK has an explicit priority, Cancel is in-order.
    tree.insert(
        A11yNode::builder(k(8), Role::Button)
            .label(Label::text("OK"))
            .tab_index(1)
            .build(),
    );
    tree.insert(
        A11yNode::builder(k(9), Role::Button)
            .label(Label::text("Cancel"))
            .build(),
    );
    tree
}

#[test]
fn focus_order_mixes_tab_index_and_skips_inert() {
    let tree = dialog_tree();
    let focus = FocusOrder::compute(&tree);

    // Positive tab_index (OK=1) first, then zero-index stops in insertion
    // order: Name(3), checkbox(6), Cancel(9). Disabled(4), hidden(5),
    // non-focusable dialog/heading/presentation are skipped.
    assert_eq!(focus.keys(), [k(8), k(3), k(6), k(9)]);
    assert_eq!(focus.first(), Some(&k(8)));
    assert_eq!(focus.last(), Some(&k(9)));
}

#[test]
fn focus_wraps_both_directions() {
    let tree = dialog_tree();
    let focus = FocusOrder::compute(&tree);

    assert_eq!(focus.next(&k(8)), Some(&k(3)));
    assert_eq!(focus.next(&k(9)), Some(&k(8))); // wrap to first
    assert_eq!(focus.prev(&k(8)), Some(&k(9))); // wrap to last
    assert_eq!(focus.prev(&k(3)), Some(&k(8)));
}

#[test]
fn keyboard_nav_is_role_aware() {
    let tree = dialog_tree();
    let nav = KeyboardNav::new();

    // Button: Enter and Space activate.
    assert_eq!(
        nav.resolve(&tree, &k(8), KeyInput::Enter),
        NavAction::Activate
    );
    assert_eq!(
        nav.resolve(&tree, &k(8), KeyInput::Space),
        NavAction::Activate
    );

    // Checkbox: Space toggles, Enter is inert.
    assert_eq!(
        nav.resolve(&tree, &k(6), KeyInput::Space),
        NavAction::Toggle
    );
    assert_eq!(nav.resolve(&tree, &k(6), KeyInput::Enter), NavAction::None);

    // Dialog: Escape dismisses.
    assert_eq!(
        nav.resolve(&tree, &k(1), KeyInput::Escape),
        NavAction::Dismiss
    );

    // Tab/Shift+Tab are role-independent focus moves.
    assert_eq!(
        nav.resolve(&tree, &k(3), KeyInput::Tab),
        NavAction::MoveFocus(FocusMove::Next)
    );
    assert_eq!(
        nav.resolve(&tree, &k(3), KeyInput::ShiftTab),
        NavAction::MoveFocus(FocusMove::Prev)
    );
}

#[test]
fn list_arrow_navigation() {
    let mut tree = A11yTree::new();
    for id in 10..13 {
        tree.insert(
            A11yNode::builder(k(id), Role::ListItem)
                .label(Label::text("item"))
                .focusable(true)
                .build(),
        );
    }
    let nav = KeyboardNav::new();
    assert_eq!(
        nav.resolve(&tree, &k(10), KeyInput::Arrow(Arrow::Down)),
        NavAction::MoveFocus(FocusMove::Next)
    );
    assert_eq!(
        nav.resolve(&tree, &k(11), KeyInput::Arrow(Arrow::Up)),
        NavAction::MoveFocus(FocusMove::Prev)
    );

    // Driving focus with the computed order yields real movement + wrap.
    let focus = FocusOrder::compute(&tree);
    assert_eq!(focus.next(&k(12)), Some(&k(10)));
}

#[test]
fn labelled_by_feeds_screen_reader_text() {
    let tree = dialog_tree();
    assert_eq!(
        screen_reader_text(&tree, &k(6)),
        "checkbox, Remember me, checked, required"
    );
    assert_eq!(
        screen_reader_text(&tree, &k(2)),
        "heading level 1, Preferences"
    );
    assert_eq!(
        screen_reader_text(&tree, &k(4)),
        "text field, Legacy, disabled"
    );
    // Hidden node is spoken as nothing.
    assert_eq!(screen_reader_text(&tree, &k(5)), "");
}

#[test]
fn live_region_queue_ordering_and_drain() {
    let mut polite = LiveRegion::polite();
    assert_eq!(polite.politeness(), Politeness::Polite);
    polite.announce("Saved");
    polite.announce("2 items updated");
    assert_eq!(polite.pending(), 2);
    assert_eq!(polite.drain(), ["Saved", "2 items updated"]);
    assert!(polite.is_empty());

    let mut assertive = LiveRegion::assertive();
    assert_eq!(assertive.politeness(), Politeness::Assertive);
    assert!(assertive.politeness().interrupts());
    assertive.announce("Error: network lost");
    assert_eq!(assertive.drain(), ["Error: network lost"]);
}
