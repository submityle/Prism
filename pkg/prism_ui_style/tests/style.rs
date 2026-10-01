//! Integration tests for the `prism_ui_style` cascade engine.
#![allow(
    clippy::std_instead_of_alloc,
    reason = "integration tests run under std"
)]

use prism_ui_style::{
    Breakpoint, Cascade, Class, ComputedStyle, InteractionState, MatchContext, StyleError,
    StyleProp, StyleSheet, StyleValue, Theme, TokenStore,
};

fn computed_for(
    sheet: &StyleSheet,
    tokens: &TokenStore,
    classes: &[&str],
    ctx: &MatchContext,
) -> ComputedStyle {
    Cascade::new(sheet, tokens)
        .resolve(classes, ctx)
        .expect("cascade should resolve")
}

#[test]
fn token_resolution_direct() {
    let mut tokens = TokenStore::new();
    tokens.insert("space.md", StyleValue::px(16.0));

    assert_eq!(tokens.resolve("space.md"), Ok(StyleValue::px(16.0)));
}

#[test]
fn token_resolution_reference_chain() {
    // alias -> mid -> base literal.
    let mut tokens = TokenStore::new();
    tokens.insert("base", StyleValue::px(16.0));
    tokens.insert("mid", StyleValue::token("base"));
    tokens.insert("alias", StyleValue::token("mid"));

    assert_eq!(tokens.resolve("alias"), Ok(StyleValue::px(16.0)));
}

#[test]
fn token_resolution_unknown_is_error() {
    let tokens = TokenStore::new();
    assert_eq!(
        tokens.resolve("missing"),
        Err(StyleError::UnknownToken("missing".into())),
    );
}

#[test]
fn token_resolution_cycle_is_error_not_panic() {
    // a -> b -> a forms a cycle.
    let mut tokens = TokenStore::new();
    tokens.insert("a", StyleValue::token("b"));
    tokens.insert("b", StyleValue::token("a"));

    match tokens.resolve("a") {
        Err(StyleError::CycleDetected(_)) => {}
        other => panic!("expected CycleDetected, got {other:?}"),
    }
}

#[test]
fn token_resolution_self_cycle_is_error() {
    let mut tokens = TokenStore::new();
    tokens.insert("loop", StyleValue::token("loop"));

    assert_eq!(
        tokens.resolve("loop"),
        Err(StyleError::CycleDetected("loop".into())),
    );
}

#[test]
fn cascade_order_last_class_wins() {
    let sheet = StyleSheet::new()
        .with_class(Class::new("a").with(StyleProp::Width, StyleValue::px(100.0)))
        .with_class(Class::new("b").with(StyleProp::Width, StyleValue::px(200.0)));
    let tokens = TokenStore::new();
    let ctx = MatchContext::new(0.0);

    // b listed last wins.
    let computed = computed_for(&sheet, &tokens, &["a", "b"], &ctx);
    assert_eq!(computed.get(StyleProp::Width), Some(&StyleValue::px(200.0)));

    // reversed order: a wins.
    let computed = computed_for(&sheet, &tokens, &["b", "a"], &ctx);
    assert_eq!(computed.get(StyleProp::Width), Some(&StyleValue::px(100.0)));
}

#[test]
fn cascade_unknown_class_is_skipped() {
    let sheet = StyleSheet::new()
        .with_class(Class::new("real").with(StyleProp::Height, StyleValue::px(10.0)));
    let tokens = TokenStore::new();
    let ctx = MatchContext::new(0.0);

    let computed = computed_for(&sheet, &tokens, &["ghost", "real", "phantom"], &ctx);
    assert_eq!(computed.get(StyleProp::Height), Some(&StyleValue::px(10.0)));
    assert_eq!(computed.len(), 1);
}

#[test]
fn state_override_applies_only_when_active() {
    let sheet = StyleSheet::new().with_class(
        Class::new("btn")
            .with(StyleProp::Color, StyleValue::rgba8(0, 0, 0, 255))
            .with_state(
                InteractionState::Hover,
                StyleProp::Color,
                StyleValue::rgba8(255, 0, 0, 255),
            )
            .with_state(
                InteractionState::Pressed,
                StyleProp::Color,
                StyleValue::rgba8(0, 0, 255, 255),
            ),
    );
    let tokens = TokenStore::new();

    // No active states -> base color.
    let normal = computed_for(&sheet, &tokens, &["btn"], &MatchContext::new(0.0));
    assert_eq!(
        normal.get(StyleProp::Color),
        Some(&StyleValue::rgba8(0, 0, 0, 255)),
    );

    // Hover active -> hover color.
    let hover = computed_for(
        &sheet,
        &tokens,
        &["btn"],
        &MatchContext::new(0.0).with_state(InteractionState::Hover),
    );
    assert_eq!(
        hover.get(StyleProp::Color),
        Some(&StyleValue::rgba8(255, 0, 0, 255)),
    );

    // Hover + Pressed active -> Pressed wins (higher priority).
    let pressed = computed_for(
        &sheet,
        &tokens,
        &["btn"],
        &MatchContext::new(0.0)
            .with_state(InteractionState::Hover)
            .with_state(InteractionState::Pressed),
    );
    assert_eq!(
        pressed.get(StyleProp::Color),
        Some(&StyleValue::rgba8(0, 0, 255, 255)),
    );
}

#[test]
fn breakpoint_selection_by_viewport_width() {
    let sheet = StyleSheet::new().with_class(
        Class::new("col")
            .with(StyleProp::Width, StyleValue::percent(100.0))
            .with_breakpoint(Breakpoint::Md, StyleProp::Width, StyleValue::percent(50.0))
            .with_breakpoint(Breakpoint::Lg, StyleProp::Width, StyleValue::percent(25.0)),
    );
    let tokens = TokenStore::new();

    // Narrow: base only.
    let narrow = computed_for(&sheet, &tokens, &["col"], &MatchContext::new(500.0));
    assert_eq!(
        narrow.get(StyleProp::Width),
        Some(&StyleValue::percent(100.0)),
    );

    // >= Md but < Lg: md override.
    let medium = computed_for(&sheet, &tokens, &["col"], &MatchContext::new(800.0));
    assert_eq!(
        medium.get(StyleProp::Width),
        Some(&StyleValue::percent(50.0)),
    );

    // >= Lg: lg override wins (applied after md).
    let large = computed_for(&sheet, &tokens, &["col"], &MatchContext::new(1400.0));
    assert_eq!(
        large.get(StyleProp::Width),
        Some(&StyleValue::percent(25.0)),
    );
}

#[test]
fn breakpoint_thresholds_and_for_width() {
    assert_eq!(Breakpoint::Base.min_width(), 0.0);
    assert_eq!(Breakpoint::for_width(0.0), Breakpoint::Base);
    assert_eq!(Breakpoint::for_width(640.0), Breakpoint::Sm);
    assert_eq!(Breakpoint::for_width(767.0), Breakpoint::Sm);
    assert_eq!(Breakpoint::for_width(768.0), Breakpoint::Md);
    assert_eq!(Breakpoint::for_width(1024.0), Breakpoint::Lg);
    assert_eq!(Breakpoint::for_width(5000.0), Breakpoint::Xl);
}

#[test]
fn token_refs_resolved_in_computed_style() {
    let theme = Theme::with_default_palette();
    let sheet = StyleSheet::new().with_class(
        Class::new("card")
            .with(StyleProp::BackgroundColor, StyleValue::token("color.bg"))
            // color.action -> color.primary (a reference chain through tokens).
            .with(StyleProp::Color, StyleValue::token("color.action"))
            .with(StyleProp::BorderRadius, StyleValue::token("radius.md"))
            .with_padding_x(StyleValue::token("space.md")),
    );
    let ctx = MatchContext::new(0.0);

    let computed = computed_for(&sheet, &theme.tokens, &["card"], &ctx);

    // No value in the computed style is left as an unresolved reference.
    for (_, value) in computed.iter() {
        assert!(
            !value.is_token_ref(),
            "computed value should be fully resolved: {value:?}",
        );
    }

    assert_eq!(
        computed.get(StyleProp::BackgroundColor),
        Some(&StyleValue::rgba8(255, 255, 255, 255)),
    );
    // Chained reference resolves to primary blue.
    assert_eq!(
        computed.get(StyleProp::Color),
        Some(&StyleValue::rgba8(59, 130, 246, 255)),
    );
    assert_eq!(
        computed.get(StyleProp::BorderRadius),
        Some(&StyleValue::px(8.0)),
    );
    assert_eq!(
        computed.get(StyleProp::PaddingLeft),
        Some(&StyleValue::px(16.0)),
    );
    assert_eq!(
        computed.get(StyleProp::PaddingRight),
        Some(&StyleValue::px(16.0)),
    );
}

#[test]
fn cascade_errors_on_unknown_token_reference() {
    let sheet = StyleSheet::new()
        .with_class(Class::new("bad").with(StyleProp::Color, StyleValue::token("color.nope")));
    let tokens = TokenStore::new();
    let ctx = MatchContext::new(0.0);

    let result = Cascade::new(&sheet, &tokens).resolve(&["bad"], &ctx);
    assert_eq!(result, Err(StyleError::UnknownToken("color.nope".into())));
}

#[test]
fn default_theme_matches_palette_constructor() {
    assert_eq!(Theme::default(), Theme::with_default_palette());
    assert!(!Theme::default().tokens.is_empty());
}
