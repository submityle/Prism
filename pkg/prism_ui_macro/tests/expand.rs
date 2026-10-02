//! Integration tests that exercise the `loom!` macro end to end.
//!
//! Each test builds a real [`prism_ui::Element`] tree with the macro and
//! asserts the resulting structure through the public `Element` accessors.

use prism_ui::style::{Color, Keyword, Length, StyleProp, StyleValue};
use prism_ui::{Element, ElementKind, Key, StableId};
use prism_ui_macro::loom;

extern crate alloc;

#[test]
fn box_with_classes_key_and_styles() {
    let el = loom! {
        box {
            class: "card", "elevated";
            key: 42;
            style: {
                width: px(300.0);
                background_color: token("color.bg");
            };
        }
    };

    assert_eq!(el.kind(), &ElementKind::Box);
    assert_eq!(
        el.class_names(),
        &["card".to_string(), "elevated".to_string()]
    );
    assert_eq!(el.explicit_key(), Some(&Key::Int(42)));

    let pairs = el.inline_pairs();
    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0], (StyleProp::Width, StyleValue::px(300.0)));
    assert_eq!(
        pairs[1],
        (StyleProp::BackgroundColor, StyleValue::token("color.bg"))
    );
}

#[test]
fn nested_children_preserve_order() {
    let el = loom! {
        box {
            text("first");
            box { class: "row"; }
            text("third");
        }
    };

    let children = el.child_elements();
    assert_eq!(children.len(), 3);
    assert_eq!(children[0].text_content(), Some("first"));
    assert_eq!(children[1].kind(), &ElementKind::Box);
    assert_eq!(children[1].class_names(), &["row".to_string()]);
    assert_eq!(children[2].text_content(), Some("third"));
}

#[test]
fn text_node_carries_content() {
    let el = loom! { text("hello world") };

    assert_eq!(el.kind(), &ElementKind::Text);
    assert_eq!(el.text_content(), Some("hello world"));
    assert!(el.child_elements().is_empty());
}

#[test]
fn custom_node_with_string_key() {
    let el = loom! {
        custom("my_widget") {
            key: "slot-a";
            text("inner");
        }
    };

    assert_eq!(el.kind(), &ElementKind::Custom("my_widget".to_string()));
    assert_eq!(el.explicit_key(), Some(&Key::Str("slot-a".to_string())));
    assert_eq!(el.child_elements().len(), 1);
    assert_eq!(el.child_elements()[0].text_content(), Some("inner"));
}

#[test]
fn keyword_style_values_map_to_pascal_case() {
    let el = loom! {
        box {
            style: {
                flex_direction: column;
                justify_content: space_between;
                align_items: center;
            };
        }
    };

    let pairs = el.inline_pairs();
    assert_eq!(
        pairs[0],
        (
            StyleProp::FlexDirection,
            StyleValue::Keyword(Keyword::Column)
        )
    );
    assert_eq!(
        pairs[1],
        (
            StyleProp::JustifyContent,
            StyleValue::Keyword(Keyword::SpaceBetween)
        )
    );
    assert_eq!(
        pairs[2],
        (StyleProp::AlignItems, StyleValue::Keyword(Keyword::Center))
    );
}

#[test]
fn style_value_constructors_and_numbers() {
    let el = loom! {
        box {
            style: {
                width: px(120.0);
                color: rgba8(10, 20, 30, 40);
                background_color: token("color.surface");
                opacity: 0.5;
                flex_grow: 2;
            };
        }
    };

    let pairs = el.inline_pairs();
    assert_eq!(pairs[0].1, StyleValue::Length(Length::Px(120.0)));
    assert_eq!(pairs[1].1, StyleValue::Color(Color::rgba8(10, 20, 30, 40)));
    assert_eq!(
        pairs[2].1,
        StyleValue::TokenRef("color.surface".to_string())
    );
    assert_eq!(pairs[3].1, StyleValue::Number(0.5));
    assert_eq!(pairs[4].1, StyleValue::Number(2.0));
}

#[test]
fn for_each_splices_dynamic_children() {
    let labels = ["a", "b", "c"];
    let el = loom! {
        box {
            text("header");
            for_each(labels.iter().map(|label| loom! { text(*label) }));
        }
    };

    let children = el.child_elements();
    assert_eq!(children.len(), 4);
    assert_eq!(children[0].text_content(), Some("header"));
    assert_eq!(children[1].text_content(), Some("a"));
    assert_eq!(children[2].text_content(), Some("b"));
    assert_eq!(children[3].text_content(), Some("c"));
}

#[test]
fn for_each_accepts_a_vec_of_elements() {
    let el = loom! {
        box {
            for_each(vec![Element::text("x"), Element::text("y")]);
        }
    };

    let children = el.child_elements();
    assert_eq!(children.len(), 2);
    assert_eq!(children[0].text_content(), Some("x"));
    assert_eq!(children[1].text_content(), Some("y"));
}

/// Collects every node's stable-id string in depth-first preorder.
fn stable_ids(root: &Element) -> Vec<Option<String>> {
    let mut out = Vec::new();
    collect_stable_ids(root, &mut out);
    out
}

/// Pushes `element`'s stable id, then recurses into its children in order.
fn collect_stable_ids(element: &Element, out: &mut Vec<Option<String>>) {
    out.push(element.stable_id().map(StableId::as_str).map(String::from));
    for child in element.child_elements() {
        collect_stable_ids(child, out);
    }
}

#[test]
fn root_node_has_empty_stable_id() {
    let el = loom! { box {} };
    assert_eq!(el.stable_id().map(StableId::as_str), Some(""));

    let text = loom! { text("hi") };
    assert_eq!(text.stable_id().map(StableId::as_str), Some(""));
}

#[test]
fn nested_children_follow_position_paths() {
    let el = loom! {
        box {
            text("first");
            box {
                text("deep");
            }
        }
    };

    assert_eq!(el.stable_id().map(StableId::as_str), Some(""));
    let children = el.child_elements();
    assert_eq!(children[0].stable_id().map(StableId::as_str), Some("0"));
    assert_eq!(children[1].stable_id().map(StableId::as_str), Some("1"));
    let grandchild = &children[1].child_elements()[0];
    assert_eq!(grandchild.stable_id().map(StableId::as_str), Some("1/0"));
}

#[test]
fn multi_child_paths_match_sibling_indices() {
    let el = loom! {
        box {
            text("a");
            text("b");
            text("c");
        }
    };

    let ids = stable_ids(&el);
    assert_eq!(
        ids,
        vec![
            Some(String::new()),
            Some("0".to_string()),
            Some("1".to_string()),
            Some("2".to_string()),
        ]
    );
}

#[test]
fn two_expansions_produce_identical_ids() {
    let first = loom! {
        box {
            text("a");
            box {
                text("b");
                custom("c") {}
            }
        }
    };
    let second = loom! {
        box {
            text("a");
            box {
                text("b");
                custom("c") {}
            }
        }
    };
    assert_eq!(stable_ids(&first), stable_ids(&second));
    // Spot-check the deepest deterministic path.
    assert_eq!(
        first.child_elements()[1].child_elements()[1]
            .stable_id()
            .map(StableId::as_str),
        Some("1/1")
    );
}

#[test]
fn for_each_items_carry_no_stable_id() {
    // Plain (non-macro) elements spliced via `for_each` receive no path id from
    // the enclosing node: dynamic lists align by runtime key, not static path.
    let el = loom! {
        box {
            text("header");
            for_each(vec![Element::text("x"), Element::text("y")]);
        }
    };

    assert_eq!(el.stable_id().map(StableId::as_str), Some(""));
    let children = el.child_elements();
    assert_eq!(children[0].text_content(), Some("header"));
    assert_eq!(children[0].stable_id().map(StableId::as_str), Some("0"));
    // Spliced items were built with plain builders, so they carry no stable id.
    assert_eq!(children[1].text_content(), Some("x"));
    assert!(children[1].stable_id().is_none());
    assert_eq!(children[2].text_content(), Some("y"));
    assert!(children[2].stable_id().is_none());
}

#[test]
fn dollar_sigil_reads_signal_content() {
    // `$sig` in a `text(..)` slot lowers to `sig.get()`: the macro reads the
    // signal's current value when the tree is built.
    use prism_ui::reactive::Runtime;

    let rt = Runtime::new();
    let label = rt.signal(String::from("hello"));

    let el = loom! { text($label) };
    assert_eq!(el.kind(), &ElementKind::Text);
    assert_eq!(el.text_content(), Some("hello"));
}

#[test]
fn dollar_sigil_tracks_dependency_in_effect() {
    // Because `$sig` lowers to a tracked `get()`, building the tree inside an
    // effect subscribes that effect to the signal, so a later `set` re-runs it.
    use core::cell::RefCell;
    use prism_ui::reactive::Runtime;
    use alloc::rc::Rc;

    let rt = Runtime::new();
    let label = rt.signal(String::from("first"));

    let seen = Rc::new(RefCell::new(Vec::<String>::new()));
    let effect = {
        let label = label.clone();
        let seen = Rc::clone(&seen);
        rt.effect(move || {
            let el = loom! { box { text($label); } };
            let text = el.child_elements()[0]
                .text_content()
                .unwrap()
                .to_string();
            seen.borrow_mut().push(text);
        })
    };

    label.set(String::from("second"));
    effect.dispose();

    assert_eq!(
        *seen.borrow(),
        vec![String::from("first"), String::from("second")]
    );
}

#[test]
fn dollar_sigil_on_custom_name() {
    // The sigil also works for `custom(..)` names.
    use prism_ui::reactive::Runtime;

    let rt = Runtime::new();
    let name = rt.signal(String::from("my_widget"));

    let el = loom! { custom($name) {} };
    assert_eq!(el.kind(), &ElementKind::Custom("my_widget".to_string()));
}

#[test]
fn bare_expression_is_spliced_without_get() {
    // Without `$`, the expression is spliced verbatim (no `.get()`), so a plain
    // owned value still works as content.
    let text = String::from("plain");
    let el = loom! { text(text.clone()) };
    assert_eq!(el.text_content(), Some("plain"));
}
