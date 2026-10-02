//! Integration tests exercising the public surface end to end:
//! signal-driven theme switching, token-cycle detection and `RTL` mirroring.

extern crate alloc;

use prism_ui_reactive::Runtime;
use prism_ui_style::{Color, StyleError, StyleProp, StyleValue};
use prism_ui_theme::{
    compile_theme, mirror_prop_map, Direction, LogicalEdge, LogicalSide, LogicalSpacing, Palette,
    ReactiveTheme, SemanticMap, ThemeDefinition, ThemeMode,
};

#[test]
fn signal_switch_drives_compiled_theme() {
    let rt = Runtime::new();
    let theme = ReactiveTheme::new(&rt, ThemeDefinition::studio(), ThemeMode::Light);

    let surface = theme.color("color.surface");
    let text = theme.color("color.text");

    assert_eq!(
        surface.get().unwrap(),
        Some(Color::rgba8(249, 250, 251, 255))
    );
    assert_eq!(text.get().unwrap(), Some(Color::rgba8(17, 24, 39, 255)));

    // One write flips the whole theme.
    theme.set_mode(ThemeMode::Dark);
    assert_eq!(surface.get().unwrap(), Some(Color::rgba8(17, 24, 39, 255)));
    assert_eq!(text.get().unwrap(), Some(Color::rgba8(249, 250, 251, 255)));

    theme.set_mode(ThemeMode::HighContrast);
    assert_eq!(surface.get().unwrap(), Some(Color::rgba8(0, 0, 0, 255)));
    assert_eq!(text.get().unwrap(), Some(Color::rgba8(255, 255, 255, 255)));
}

#[test]
fn switch_only_recomputes_changed_tokens() {
    use alloc::rc::Rc;
    use core::cell::RefCell;

    let rt = Runtime::new();
    let theme = ReactiveTheme::new(&rt, ThemeDefinition::studio(), ThemeMode::Light);

    let surface = theme.color("color.surface");
    let primary = theme.color("color.primary");

    let surface_runs = Rc::new(RefCell::new(0usize));
    let primary_runs = Rc::new(RefCell::new(0usize));

    let _surface_effect = rt.effect({
        let surface = surface.clone();
        let runs = surface_runs.clone();
        move || {
            let _ = surface.get();
            *runs.borrow_mut() += 1;
        }
    });
    let _primary_effect = rt.effect({
        let primary = primary.clone();
        let runs = primary_runs.clone();
        move || {
            let _ = primary.get();
            *runs.borrow_mut() += 1;
        }
    });

    assert_eq!(*surface_runs.borrow(), 1);
    assert_eq!(*primary_runs.borrow(), 1);

    // Light -> Dark: surface changes, the constant brand primary does not.
    theme.set_mode(ThemeMode::Dark);
    assert_eq!(*surface_runs.borrow(), 2);
    assert_eq!(*primary_runs.borrow(), 1);
}

#[test]
fn direct_cycle_is_detected_not_overflowed() {
    let palette = Palette::new();
    let semantics = SemanticMap::new()
        .with_default("a", StyleValue::token("b"))
        .with_default("b", StyleValue::token("a"));
    let theme = ThemeDefinition::new(palette, semantics);

    let err = compile_theme(&theme, &ThemeMode::Light).unwrap_err();
    assert!(matches!(err, StyleError::CycleDetected(_)));
}

#[test]
fn cross_layer_cycle_is_detected() {
    // semantic -> palette -> semantic back-reference forms a cycle.
    let palette = Palette::new().with("p", StyleValue::token("s"));
    let semantics = SemanticMap::new().with_default("s", StyleValue::token("p"));
    let theme = ThemeDefinition::new(palette, semantics);

    let err = compile_theme(&theme, &ThemeMode::Light).unwrap_err();
    assert!(matches!(err, StyleError::CycleDetected(_)));
}

#[test]
fn missing_reference_is_reported() {
    let palette = Palette::new();
    let semantics = SemanticMap::new().with_default("x", StyleValue::token("nope"));
    let theme = ThemeDefinition::new(palette, semantics);

    let err = compile_theme(&theme, &ThemeMode::Light).unwrap_err();
    assert_eq!(err, StyleError::UnknownToken("nope".into()));
}

#[test]
fn reactive_theme_propagates_cycle_error() {
    let rt = Runtime::new();
    let palette = Palette::new();
    let semantics = SemanticMap::new()
        .with_default("a", StyleValue::token("b"))
        .with_default("b", StyleValue::token("a"));
    let broken = ThemeDefinition::new(palette, semantics);

    let theme = ReactiveTheme::new(&rt, broken, ThemeMode::Light);
    let token = theme.token("a");
    assert!(matches!(token.get(), Err(StyleError::CycleDetected(_))));
}

#[test]
fn rtl_mirrors_logical_spacing() {
    let spacing = LogicalSpacing::new()
        .pad(LogicalSide::InlineStart, StyleValue::px(12.0))
        .margin(LogicalSide::InlineEnd, StyleValue::px(6.0))
        .pad(LogicalSide::BlockStart, StyleValue::px(4.0));

    let ltr = spacing.resolve(Direction::Ltr);
    assert_eq!(
        ltr.get(&StyleProp::PaddingLeft),
        Some(&StyleValue::px(12.0))
    );
    assert_eq!(ltr.get(&StyleProp::MarginRight), Some(&StyleValue::px(6.0)));

    let rtl = spacing.resolve(Direction::Rtl);
    assert_eq!(
        rtl.get(&StyleProp::PaddingRight),
        Some(&StyleValue::px(12.0))
    );
    assert_eq!(rtl.get(&StyleProp::MarginLeft), Some(&StyleValue::px(6.0)));
    // Block side never mirrors.
    assert_eq!(rtl.get(&StyleProp::PaddingTop), Some(&StyleValue::px(4.0)));
}

#[test]
fn rtl_mirror_prop_map_round_trips() {
    let base = LogicalSpacing::new()
        .pad(LogicalSide::InlineStart, StyleValue::px(8.0))
        .resolve(Direction::Ltr);

    // Mirroring twice returns to the original.
    let once = mirror_prop_map(&base, Direction::Rtl);
    let twice = mirror_prop_map(&once, Direction::Rtl);
    assert_eq!(twice, base);
}

#[test]
fn direction_driven_spacing_follows_signal() {
    let rt = Runtime::new();
    let theme = ReactiveTheme::new(&rt, ThemeDefinition::studio(), ThemeMode::Light);

    let edge = LogicalEdge::padding(LogicalSide::InlineStart);
    let dir_memo = {
        let theme = theme.clone();
        rt.memo(move || edge.to_physical(theme.direction()))
    };

    assert_eq!(dir_memo.get(), StyleProp::PaddingLeft);
    theme.set_direction(Direction::Rtl);
    assert_eq!(dir_memo.get(), StyleProp::PaddingRight);
}
