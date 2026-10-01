//! The computed output of the layout algorithm for a single box.

use crate::geometry::{Point, Size};

/// The resolved geometry of a box after layout.
///
/// `location` is the position of the box's top-left (border-box) corner
/// relative to the top-left corner of its parent box. `size` is the box's
/// border-box size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layout {
    /// Paint order of the box among its siblings. Lower values are painted
    /// first; this matches the order the children were declared.
    pub order: u32,
    /// Position of the box relative to its parent's top-left corner.
    pub location: Point<f32>,
    /// Border-box size of the box.
    pub size: Size<f32>,
}

impl Layout {
    /// A zeroed layout located at the origin with no size.
    pub const ZERO: Self = Self {
        order: 0,
        location: Point::ZERO,
        size: Size::ZERO,
    };

    /// Creates a layout from its parts.
    pub const fn new(order: u32, location: Point<f32>, size: Size<f32>) -> Self {
        Self {
            order,
            location,
            size,
        }
    }
}

impl Default for Layout {
    fn default() -> Self {
        Layout::ZERO
    }
}
