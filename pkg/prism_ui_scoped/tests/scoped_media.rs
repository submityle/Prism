//! End-to-end coverage: scope a component's sheet and view, then resolve the
//! scoped classes responsively — exercising both features together the way a
//! Loom component host would.

use prism_ui::{Element, ElementKind};
use prism_ui_scoped::{MediaResolver, Scope, ScopeId};
use prism_ui_style::{Breakpoint, Class, StyleProp, StyleSheet, StyleValue};

fn card_sheet() -> StyleSheet {
    StyleSheet::new()
        .with_class(
            Class::new("title")
                .with(StyleProp::FontSize, StyleValue::px(16.0))
                .with_breakpoint(Breakpoint::Md, StyleProp::FontSize, StyleValue::px(28.0)),
        )
        .with_class(
            Class::new("body")
                .with(StyleProp::Opacity, StyleValue::number(0.8))
                .with_breakpoint(Breakpoint::Lg, StyleProp::Opacity, StyleValue::number(1.0)),
        )
}

#[test]
fn two_components_do_not_collide() {
    let sheet = card_sheet();
    let hero = Scope::from_name("Hero").with_local("title").scope(&sheet);
    let aside = Scope::from_name("Aside").with_local("title").scope(&sheet);

    let hero_name = hero.scoped_name("title").unwrap();
    let aside_name = aside.scoped_name("title").unwrap();
    assert_ne!(hero_name, aside_name);
    // Each scoped sheet only knows its own rewrite.
    assert!(hero.sheet().get(hero_name).is_some());
    assert!(hero.sheet().get(aside_name).is_none());
}

#[test]
fn view_and_sheet_rewrites_stay_consistent() {
    let sheet = card_sheet();
    let scope = Scope::from_name("Card")
        .with_local("title")
        .with_local("body");
    let scoped = scope.scope(&sheet);

    let view = Element::box_()
        .class("card") // global, untouched
        .child(Element::text("Heading").class("title"))
        .child(Element::box_().class("body").class("title"));
    let out = scope.apply(&view);

    let title = scoped.scoped_name("title").unwrap();
    let body = scoped.scoped_name("body").unwrap();

    assert_eq!(out.kind(), &ElementKind::Box);
    assert_eq!(out.class_names(), &["card".to_string()]);
    assert_eq!(out.child_elements()[0].class_names(), &[title.to_string()]);
    assert_eq!(
        out.child_elements()[1].class_names(),
        &[body.to_string(), title.to_string()]
    );

    // Every class referenced by the rewritten view resolves in the scoped sheet
    // (except the global `card`, which the scope never claimed).
    for child in out.child_elements() {
        for name in child.class_names() {
            assert!(scoped.sheet().get(name).is_some());
        }
    }
}

#[test]
fn responsive_resolution_over_scoped_sheet() {
    let sheet = card_sheet();
    let scoped = Scope::new(ScopeId::from_name("Card"))
        .with_local("title")
        .with_local("body")
        .scope(&sheet);
    let title = scoped.scoped_name("title").unwrap().to_string();

    // Narrow viewport: base sizes.
    let narrow = MediaResolver::new(500.0).resolve_sheet(&scoped);
    assert_eq!(
        narrow.get(&title).unwrap().get(&StyleProp::FontSize),
        Some(&StyleValue::px(16.0))
    );

    // Wide viewport: md override on title, lg override on body.
    let wide = MediaResolver::new(1024.0).resolve_sheet(&scoped);
    assert_eq!(
        wide.get(&title).unwrap().get(&StyleProp::FontSize),
        Some(&StyleValue::px(28.0))
    );
    let body = scoped.scoped_name("body").unwrap();
    assert_eq!(
        wide.get(body).unwrap().get(&StyleProp::Opacity),
        Some(&StyleValue::number(1.0))
    );
}
