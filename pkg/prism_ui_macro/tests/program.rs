//! Consumer tests for the `loom_program!` macro (§9.1 dual-mode / §9.2 hoist).
//!
//! `loom_program!` expands to a `&'static str` whose first line is the
//! static/dynamic hoisting classification and whose remaining lines are the
//! stable semantic-program signature shared by the interpret and freeze paths.
//! These tests pin that textual contract from a real downstream crate.

use prism_ui_macro::loom_program;

#[test]
fn fully_literal_tree_is_static() {
    let program: &'static str = loom_program! {
        box {
            class: "card";
            text("hi");
        }
    };

    let mut lines = program.lines();
    assert_eq!(lines.next(), Some("static"));
    assert_eq!(lines.next(), Some("construct box@"));
    assert_eq!(lines.next(), Some("class card"));
    assert_eq!(lines.next(), Some("enter"));
    assert_eq!(lines.next(), Some("construct text@0"));
    assert_eq!(lines.next(), Some("leave"));
    assert_eq!(lines.next(), None);
}

#[test]
fn reactive_read_marks_tree_dynamic() {
    let program: &'static str = loom_program! {
        text($label)
    };

    assert_eq!(program, "dynamic\nconstruct text@\n");
}

#[test]
fn for_each_marks_tree_dynamic() {
    let program: &'static str = loom_program! {
        box {
            for_each(items);
        }
    };

    let mut lines = program.lines();
    assert_eq!(lines.next(), Some("dynamic"));
    assert_eq!(lines.next(), Some("construct box@"));
    assert_eq!(lines.next(), Some("for_each"));
    assert_eq!(lines.next(), None);
}

#[test]
fn attribute_source_order_is_normalised() {
    // Two trees that differ only in the source order of their attribute lines
    // must freeze to the identical program signature (reproducible build).
    let a: &'static str = loom_program! {
        box {
            class: "c";
            key: 3;
            style: { width: px(1.0); };
        }
    };
    let b: &'static str = loom_program! {
        box {
            style: { width: px(1.0); };
            key: 3;
            class: "c";
        }
    };

    assert_eq!(a, b);
}
