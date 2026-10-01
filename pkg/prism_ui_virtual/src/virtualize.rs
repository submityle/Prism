//! Turning list metrics plus a viewport into a windowed [`Element`] tree.

use alloc::vec::Vec;
use core::ops::Range;

use prism_ui::style::{Keyword, StyleProp, StyleValue};
use prism_ui::Element;

use crate::fixed::FixedList;
use crate::variable::VariableList;
use crate::viewport::Viewport;

/// Class applied to the virtualization container box.
pub const CONTAINER_CLASS: &str = "virtual-list";
/// Class applied to the leading and trailing spacer boxes.
pub const SPACER_CLASS: &str = "virtual-spacer";
/// Class applied to each materialized item slot box.
pub const ITEM_CLASS: &str = "virtual-item";

/// Builds a spacer box of the given main-axis height (clamped to be
/// non-negative) that reserves scroll extent without materializing items.
fn spacer(height: f32) -> Element {
    Element::box_()
        .class(SPACER_CLASS)
        .style(StyleProp::Height, StyleValue::px(height.max(0.0)))
}

/// Wraps a built item in a keyed, fixed-size slot box.
fn slot(index: usize, height: f32, margin_bottom: f32, content: Element) -> Element {
    let mut el = Element::box_()
        .class(ITEM_CLASS)
        .key_int(index as i64)
        .style(StyleProp::Height, StyleValue::px(height))
        .child(content);
    if margin_bottom > 0.0 {
        el = el.style(StyleProp::MarginBottom, StyleValue::px(margin_bottom));
    }
    el
}

/// Assembles the container box from the leading spacer, the slot children and
/// the trailing spacer.
fn container(leading: f32, slots: Vec<Element>, trailing: f32) -> Element {
    let mut children = Vec::with_capacity(slots.len() + 2);
    children.push(spacer(leading));
    children.extend(slots);
    children.push(spacer(trailing));
    Element::box_()
        .class(CONTAINER_CLASS)
        .style(StyleProp::Display, StyleValue::keyword(Keyword::Flex))
        .style(
            StyleProp::FlexDirection,
            StyleValue::keyword(Keyword::Column),
        )
        .children(children)
}

/// Virtualizes a fixed-height list against `viewport`.
///
/// Returns the visible index range and a container [`Element`] holding a
/// leading spacer sized to the offset of the first visible item, one keyed slot
/// per visible item (each built by `build`), and a trailing spacer filling the
/// remaining extent. The spacer sizes keep the scrollbar geometry identical to
/// a fully materialized list even though only the visible slice exists.
pub fn virtualize_fixed<F>(
    list: &FixedList,
    viewport: &Viewport,
    build: F,
) -> (Range<usize>, Element)
where
    F: Fn(usize) -> Element,
{
    let range = list.visible_range(viewport);
    let total = list.total_size();
    let leading = list.offset_of(range.start);

    let mut slots = Vec::with_capacity(range.len());
    let mut content_extent = 0.0_f32;
    for index in range.clone() {
        // Every item but the last in the whole list carries a trailing gap.
        let margin = if index + 1 < list.item_count {
            list.gap
        } else {
            0.0
        };
        content_extent += list.item_height + margin;
        slots.push(slot(index, list.item_height, margin, build(index)));
    }

    let trailing = (total - leading - content_extent).max(0.0);
    (range.clone(), container(leading, slots, trailing))
}

/// Virtualizes a variable-height list against `viewport`.
///
/// Behaves like [`virtualize_fixed`] but sizes each slot from the list's
/// per-item sizes. The leading spacer is the offset of the first visible item
/// and the trailing spacer is the extent below the last visible item, so the
/// three regions always sum to [`VariableList::total_size`].
pub fn virtualize_variable<F>(
    list: &VariableList,
    viewport: &Viewport,
    build: F,
) -> (Range<usize>, Element)
where
    F: Fn(usize) -> Element,
{
    let range = list.visible_range(viewport);
    let total = list.total_size();
    let leading = list.offset_of(range.start);
    let trailing = (total - list.offset_of(range.end)).max(0.0);

    let mut slots = Vec::with_capacity(range.len());
    for index in range.clone() {
        slots.push(slot(index, list.size_of(index), 0.0, build(index)));
    }

    (range.clone(), container(leading, slots, trailing))
}
