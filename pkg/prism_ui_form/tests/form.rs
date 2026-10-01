//! Integration tests exercising the public form + validation surface.

use prism_ui_form::{
    custom, int_range, max_len, min_len, pattern, required, Form, ValidationError,
};
use prism_ui_reactive::Runtime;

fn form() -> Form {
    Form::new(Runtime::new())
}

#[test]
fn binding_get_set_round_trip() {
    let form = form();
    form.register("name", "ada", vec![]);
    assert_eq!(form.value("name"), "ada");

    form.set("name", "grace");
    assert_eq!(form.value("name"), "grace");

    // Two-way binding: writing the raw signal is reflected by the form read.
    let binding = form.binding("name").expect("field registered");
    binding.set(String::from("hopper"));
    assert_eq!(form.value("name"), "hopper");

    // Unknown fields read as empty and expose no binding.
    assert_eq!(form.value("missing"), "");
    assert!(form.binding("missing").is_none());
}

#[test]
fn required_validator() {
    let form = form();
    form.register("name", "", vec![required()]);
    assert_eq!(form.errors_for("name").len(), 1);

    form.set("name", "   ");
    assert_eq!(
        form.errors_for("name").len(),
        1,
        "whitespace is still empty"
    );

    form.set("name", "ada");
    assert!(form.errors_for("name").is_empty());
}

#[test]
fn min_and_max_len_validators() {
    let form = form();
    form.register("code", "ab", vec![min_len(3), max_len(5)]);
    assert_eq!(form.errors_for("code").len(), 1);

    form.set("code", "abcd");
    assert!(form.errors_for("code").is_empty());

    form.set("code", "abcdef");
    assert_eq!(form.errors_for("code").len(), 1);
}

#[test]
fn int_range_validator() {
    let form = form();
    form.register("age", "", vec![int_range(1, 10)]);
    assert_eq!(
        form.first_error("age").unwrap().message,
        "Must be a whole number."
    );

    form.set("age", "0");
    assert_eq!(
        form.first_error("age").unwrap().message,
        "Must be between 1 and 10."
    );

    form.set("age", "5");
    assert!(form.errors_for("age").is_empty());
    assert_eq!(form.value_as_i64("age"), Some(5));
}

#[test]
fn pattern_validator() {
    let form = form();
    form.register(
        "slug",
        "Hello",
        vec![pattern(
            |value| value.chars().all(|c| c.is_ascii_lowercase()),
            "must be lowercase",
        )],
    );
    assert_eq!(
        form.first_error("slug").unwrap().message,
        "must be lowercase"
    );

    form.set("slug", "hello");
    assert!(form.errors_for("slug").is_empty());
}

#[test]
fn custom_validator() {
    let form = form();
    form.register(
        "even",
        "abc",
        vec![custom(|value| {
            if value.len() % 2 == 0 {
                Ok(())
            } else {
                Err(String::from("length must be even"))
            }
        })],
    );
    assert_eq!(
        form.first_error("even").unwrap().message,
        "length must be even"
    );

    form.set("even", "abcd");
    assert!(form.errors_for("even").is_empty());
}

#[test]
fn multi_validator_first_versus_all_ordering() {
    let form = form();
    form.register("name", "", vec![required(), min_len(3)]);

    // All errors, in declaration order.
    let all = form.all_errors("name");
    let messages: Vec<&str> = all.iter().map(|e| e.message.as_str()).collect();
    assert_eq!(
        messages,
        vec!["This field is required.", "Must be at least 3 characters."]
    );

    // First error only.
    assert_eq!(
        form.first_error("name").unwrap().message,
        "This field is required."
    );

    // When required passes, the next rule becomes the first error.
    form.set("name", "ab");
    assert_eq!(
        form.first_error("name").unwrap().message,
        "Must be at least 3 characters."
    );
}

#[test]
fn touched_and_dirty_transitions() {
    let form = form();
    form.register("email", "", vec![]);
    assert!(!form.is_touched("email"));
    assert!(!form.is_dirty("email"));

    form.touch("email");
    assert!(form.is_touched("email"));
    assert!(!form.is_dirty("email"));

    form.set("email", "a@b.c");
    assert!(form.is_dirty("email"));
}

#[test]
fn aggregated_validate_tags_fields() {
    let form = form();
    form.register("name", "", vec![required()]);
    form.register("age", "x", vec![int_range(0, 120)]);

    let mut errors: Vec<ValidationError> = form.validate();
    errors.sort_by(|a, b| a.field.as_str().cmp(b.field.as_str()));
    assert_eq!(errors.len(), 2);
    assert_eq!(errors[0].field.as_str(), "age");
    assert_eq!(errors[1].field.as_str(), "name");
    assert!(!form.is_valid());

    form.set("name", "ada");
    form.set("age", "40");
    assert!(form.validate().is_empty());
    assert!(form.is_valid());
}

#[test]
fn errors_memo_recomputes_after_set() {
    let form = form();
    form.register("name", "", vec![required()]);

    let errors = form.errors_memo();
    assert_eq!(errors.get().len(), 1);

    form.set("name", "ada");
    assert!(errors.get().is_empty());

    // The memo is cached: repeated calls return the same reactive node.
    let same = form.errors_memo();
    form.set("name", "");
    assert_eq!(same.get().len(), 1);
    assert_eq!(errors.get().len(), 1);
}
