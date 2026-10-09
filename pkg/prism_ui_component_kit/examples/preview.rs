//! Inline, no-HTML preview: prints each control's element tree + the kit
//! classes it attaches, straight to stdout. This reflects the real
//! `Component::render` output and the real theme-backed class names; it is a
//! structural preview, not a pixel render (the kit is data-only).

#![allow(
    clippy::print_stdout,
    clippy::std_instead_of_alloc,
    clippy::uninlined_format_args,
    reason = "dev-only CLI example: prints a structural preview to stdout and may use std"
)]

use prism_ui::{Element, ElementKind};
use prism_ui_component::Component;
use prism_ui_component_kit::basics::{Button, ButtonProps, Heading, HeadingLevel, HeadingProps};
use prism_ui_component_kit::kit::{ButtonVariant, ControlSize};
use prism_ui_component_kit::stylesheet;

fn print_tree(el: &Element, depth: usize) {
    let indent = "  ".repeat(depth);
    let classes = el.class_names();
    let class_str = if classes.is_empty() {
        String::new()
    } else {
        format!("  .{}", classes.join(" ."))
    };
    match el.kind() {
        ElementKind::Box => println!("{indent}<box>{class_str}"),
        ElementKind::Custom(name) => println!("{indent}<{name}>{class_str}"),
        ElementKind::Text => {
            let t = el.text_content().unwrap_or("");
            println!("{indent}\"{t}\"{class_str}");
        }
    }
    for child in el.child_elements() {
        print_tree(child, depth + 1);
    }
}

fn show(title: &str, el: &Element) {
    println!("\n=== {title} ===");
    print_tree(el, 0);
}

fn main() {
    show(
        "Button / filled / md",
        &Button.render(&ButtonProps::new("Save").variant(ButtonVariant::Filled)),
    );
    show(
        "Button / glass / lg / disabled",
        &Button.render(
            &ButtonProps::new("Frosted")
                .variant(ButtonVariant::Glass)
                .size(ControlSize::Large)
                .disabled(true),
        ),
    );
    show(
        "Heading / L2",
        &Heading.render(&HeadingProps::new("Section title").level(HeadingLevel::L2)),
    );

    let sheet = stylesheet();
    println!("\n=== Registered class families ({} classes) ===", sheet.len());
    let mut families: std::collections::BTreeMap<String, usize> = Default::default();
    for (name, _class) in sheet.iter() {
        let base = name.split("--").next().unwrap_or(name);
        let base = base.split("__").next().unwrap_or(base);
        *families.entry(base.to_string()).or_default() += 1;
    }
    for (base, variants) in &families {
        println!("  {base}  ({variants} rules)");
    }
    println!("\n{} base control classes total.", families.len());
}
