//! Integration tests that exercise the `loom!` macro end to end.
//!
//! Each test builds a real [`prism_ui::Element`] tree with the macro and
//! asserts the resulting structure through the public `Element` accessors.

use prism_ui::style::{Color, Keyword, Length, StyleProp, StyleValue};
use prism_ui::{Element, ElementKind, Key};
use prism_ui_macro::loom;

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
