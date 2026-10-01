//! Integration tests for the windowed `Element` output of `prism_ui_virtual`.

use prism_ui::style::{Length, StyleProp, StyleValue};
use prism_ui::{Element, Key};
use prism_ui_virtual::{
    virtualize_fixed, virtualize_variable, FixedList, VariableList, Viewport, CONTAINER_CLASS,
    ITEM_CLASS, SPACER_CLASS,
};

/// Reads the pixel value of a style property from an element's inline styles.
fn px_prop(el: &Element, prop: StyleProp) -> Option<f32> {
    el.inline_pairs().iter().find_map(|(p, v)| {
        if *p == prop {
            match v {
                StyleValue::Length(Length::Px(px)) => Some(*px),
                _ => None,
            }
        } else {
            None
        }
    })
}

fn height_of(el: &Element) -> f32 {
    px_prop(el, StyleProp::Height).expect("element should have an explicit height")
}

#[test]
fn fixed_produces_spacer_items_spacer() {
    let list = FixedList::new(1000, 20.0, 0.0);
    let viewport = Viewport::new(100.0, 80.0);
    let (range, view) = virtualize_fixed(&list, &viewport, |i| Element::text(format!("row {i}")));

    assert_eq!(range, 5..10);
    assert_eq!(view.class_names(), &[CONTAINER_CLASS.to_string()]);

    let children = view.child_elements();
    // leading spacer + 5 items + trailing spacer.
    assert_eq!(children.len(), 7);

    let leading = &children[0];
    let trailing = &children[children.len() - 1];
    assert_eq!(leading.class_names(), &[SPACER_CLASS.to_string()]);
    assert_eq!(trailing.class_names(), &[SPACER_CLASS.to_string()]);

    // Leading spacer == offset of first visible item.
    assert_eq!(height_of(leading), list.offset_of(5));

    // Items are keyed by absolute index and carry the fixed height.
    for (offset, index) in (5..10).enumerate() {
        let item = &children[1 + offset];
        assert_eq!(item.class_names(), &[ITEM_CLASS.to_string()]);
        assert_eq!(item.explicit_key(), Some(&Key::Int(index as i64)));
        assert_eq!(height_of(item), 20.0);
        // The built content lives inside the slot.
        assert_eq!(
            item.child_elements()[0].text_content(),
            Some(format!("row {index}").as_str())
        );
    }
}

#[test]
fn fixed_spacers_and_items_sum_to_total() {
    let list = FixedList::new(50, 18.0, 4.0);
    let viewport = Viewport::new(120.0, 90.0).with_overscan(2);
    let (range, view) = virtualize_fixed(&list, &viewport, |_| Element::box_());

    let children = view.child_elements();
    let leading = height_of(&children[0]);
    let trailing = height_of(&children[children.len() - 1]);

    // Sum item heights plus their trailing margins (the inter-item gaps).
    let mut content = 0.0_f32;
    for item in &children[1..children.len() - 1] {
        content += height_of(item);
        content += px_prop(item, StyleProp::MarginBottom).unwrap_or(0.0);
    }

    let sum = leading + content + trailing;
    assert!(
        (sum - list.total_size()).abs() < 1e-3,
        "sum {sum} != {}",
        list.total_size()
    );
    assert_eq!(children.len(), 2 + range.len());
}

#[test]
fn fixed_internal_gaps_become_item_margins() {
    let list = FixedList::new(10, 20.0, 5.0);
    // Show items 2..=4.
    let viewport = Viewport::new(50.0, 60.0);
    let (range, view) = virtualize_fixed(&list, &viewport, |_| Element::box_());
    assert_eq!(range, 2..5);

    let children = view.child_elements();
    // Each visible item is not the last in the whole list, so each has a gap.
    for item in &children[1..children.len() - 1] {
        assert_eq!(px_prop(item, StyleProp::MarginBottom), Some(5.0));
    }
}

#[test]
fn fixed_last_item_has_no_trailing_gap() {
    let list = FixedList::new(5, 20.0, 5.0);
    // Scroll to the very bottom so the final item is visible.
    let viewport = Viewport::new(200.0, 80.0);
    let (range, view) = virtualize_fixed(&list, &viewport, |_| Element::box_());
    assert_eq!(range.end, 5);

    let children = view.child_elements();
    let last_item = &children[children.len() - 2];
    // The globally-last item carries no trailing-gap margin.
    assert_eq!(px_prop(last_item, StyleProp::MarginBottom), None);
}

#[test]
fn variable_produces_correct_structure_and_spacers() {
    // offsets: 0, 30, 40, 90, 100, 160
    let list = VariableList::from_sizes([30.0, 10.0, 50.0, 10.0, 60.0]);
    assert_eq!(list.total_size(), 160.0);

    // Window [35, 95): index_at(35)=1, index_at(95)=3 => 1..4.
    let viewport = Viewport::new(35.0, 60.0);
    let (range, view) = virtualize_variable(&list, &viewport, |i| Element::text(format!("{i}")));
    assert_eq!(range, 1..4);

    let children = view.child_elements();
    assert_eq!(children.len(), 2 + range.len());

    let leading = height_of(&children[0]);
    let trailing = height_of(&children[children.len() - 1]);
    assert_eq!(leading, list.offset_of(1)); // 30
    assert_eq!(trailing, list.total_size() - list.offset_of(4)); // 160 - 100 = 60

    // Items keyed by index and sized from the per-item sizes.
    for (offset, index) in (1..4).enumerate() {
        let item = &children[1 + offset];
        assert_eq!(item.explicit_key(), Some(&Key::Int(index as i64)));
        assert_eq!(height_of(item), list.size_of(index));
    }

    // Spacers plus item heights reconstruct the full extent.
    let content: f32 = children[1..children.len() - 1].iter().map(height_of).sum();
    assert!((leading + content + trailing - list.total_size()).abs() < 1e-3);
}

#[test]
fn variable_overscan_widens_window() {
    let list = VariableList::from_sizes([30.0, 10.0, 50.0, 10.0, 60.0]);
    let viewport = Viewport::new(35.0, 20.0).with_overscan(1);
    // Base window [35,55): index_at(35)=1, index_at(55)=2 => 1..3, overscan => 0..4.
    let (range, _view) = virtualize_variable(&list, &viewport, |_| Element::box_());
    assert_eq!(range, 0..4);
}

#[test]
fn empty_list_still_has_two_spacers() {
    let list = FixedList::new(0, 20.0, 0.0);
    let (range, view) = virtualize_fixed(&list, &Viewport::new(0.0, 50.0), |_| Element::box_());
    assert!(range.is_empty());
    let children = view.child_elements();
    assert_eq!(children.len(), 2);
    assert_eq!(height_of(&children[0]), 0.0);
    assert_eq!(height_of(&children[1]), 0.0);
}
