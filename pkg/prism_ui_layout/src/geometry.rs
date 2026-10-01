//! Core geometric primitives used throughout the layout solver.
//!
//! All quantities are expressed in abstract layout units (conventionally
//! logical pixels). The types here are deliberately engine-agnostic: they
//! carry no rendering or scene-graph semantics, only pure geometry.

use core::ops::Add;

/// A point in 2D space.
///
/// The coordinate system has its origin in the top-left corner, with `x`
/// increasing to the right and `y` increasing downward, matching typical UI
/// conventions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point<T> {
    /// Horizontal coordinate.
    pub x: T,
    /// Vertical coordinate.
    pub y: T,
}

impl<T> Point<T> {
    /// Creates a new point from its `x` and `y` components.
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl Point<f32> {
    /// A point at the origin `(0, 0)`.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
}

/// A 2D size with a `width` and a `height`.
///
/// `Size` is generic so it can describe concrete pixel extents
/// (`Size<f32>`), style inputs (`Size<Dimension>`), optional known
/// dimensions (`Size<Option<f32>>`), or available space constraints
/// (`Size<AvailableSpace>`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Size<T> {
    /// Extent along the horizontal axis.
    pub width: T,
    /// Extent along the vertical axis.
    pub height: T,
}

impl<T> Size<T> {
    /// Creates a new size from `width` and `height`.
    pub const fn new(width: T, height: T) -> Self {
        Self { width, height }
    }
}

impl<T: Copy> Size<T> {
    /// Returns the component lying on the main axis.
    ///
    /// When `is_row` is `true` the main axis is horizontal, so the `width`
    /// is returned; otherwise the `height` is returned.
    pub fn main(self, is_row: bool) -> T {
        if is_row {
            self.width
        } else {
            self.height
        }
    }

    /// Returns the component lying on the cross axis.
    ///
    /// This is the opposite axis to [`Size::main`].
    pub fn cross(self, is_row: bool) -> T {
        if is_row {
            self.height
        } else {
            self.width
        }
    }

    /// Returns a copy of this size with the main-axis component replaced.
    pub fn with_main(self, is_row: bool, value: T) -> Self {
        if is_row {
            Self {
                width: value,
                height: self.height,
            }
        } else {
            Self {
                width: self.width,
                height: value,
            }
        }
    }

    /// Returns a copy of this size with the cross-axis component replaced.
    pub fn with_cross(self, is_row: bool, value: T) -> Self {
        if is_row {
            Self {
                width: self.width,
                height: value,
            }
        } else {
            Self {
                width: value,
                height: self.height,
            }
        }
    }
}

impl Size<f32> {
    /// A size with both dimensions equal to zero.
    pub const ZERO: Self = Self {
        width: 0.0,
        height: 0.0,
    };
}

/// An axis-aligned rectangle defined by its top-left `location` and `size`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    /// Top-left corner of the rectangle.
    pub location: Point<f32>,
    /// Extent of the rectangle.
    pub size: Size<f32>,
}

impl Rect {
    /// Creates a rectangle from a `location` and a `size`.
    pub const fn new(location: Point<f32>, size: Size<f32>) -> Self {
        Self { location, size }
    }

    /// Returns the `x` coordinate of the left edge.
    pub fn left(&self) -> f32 {
        self.location.x
    }

    /// Returns the `x` coordinate of the right edge.
    pub fn right(&self) -> f32 {
        self.location.x + self.size.width
    }

    /// Returns the `y` coordinate of the top edge.
    pub fn top(&self) -> f32 {
        self.location.y
    }

    /// Returns the `y` coordinate of the bottom edge.
    pub fn bottom(&self) -> f32 {
        self.location.y + self.size.height
    }
}

/// Values associated with the four edges of a box.
///
/// Used to describe `margin`, `padding`, `border`, and the `inset` used by
/// absolutely positioned boxes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edges<T> {
    /// Value on the left edge.
    pub left: T,
    /// Value on the right edge.
    pub right: T,
    /// Value on the top edge.
    pub top: T,
    /// Value on the bottom edge.
    pub bottom: T,
}

impl<T: Copy> Edges<T> {
    /// Creates edges from the four individual values.
    pub const fn new(left: T, right: T, top: T, bottom: T) -> Self {
        Self {
            left,
            right,
            top,
            bottom,
        }
    }

    /// Creates edges with the same value on every side.
    pub const fn splat(value: T) -> Self {
        Self {
            left: value,
            right: value,
            top: value,
            bottom: value,
        }
    }
}

impl<T: Copy + Add<Output = T>> Edges<T> {
    /// Returns the sum of the left and right edges.
    pub fn horizontal(self) -> T {
        self.left + self.right
    }

    /// Returns the sum of the top and bottom edges.
    pub fn vertical(self) -> T {
        self.top + self.bottom
    }

    /// Returns the sum along the main axis.
    ///
    /// For a row axis (`is_row == true`) this is the horizontal sum; for a
    /// column axis it is the vertical sum.
    pub fn main_axis(self, is_row: bool) -> T {
        if is_row {
            self.horizontal()
        } else {
            self.vertical()
        }
    }

    /// Returns the sum along the cross axis.
    pub fn cross_axis(self, is_row: bool) -> T {
        if is_row {
            self.vertical()
        } else {
            self.horizontal()
        }
    }
}

impl Edges<f32> {
    /// Edges with every side equal to zero.
    pub const ZERO: Self = Self {
        left: 0.0,
        right: 0.0,
        top: 0.0,
        bottom: 0.0,
    };

    /// Returns the leading edge on the main axis (left for rows, top for
    /// columns).
    pub fn main_start(&self, is_row: bool) -> f32 {
        if is_row {
            self.left
        } else {
            self.top
        }
    }

    /// Returns the leading edge on the cross axis (top for rows, left for
    /// columns).
    pub fn cross_start(&self, is_row: bool) -> f32 {
        if is_row {
            self.top
        } else {
            self.left
        }
    }
}

/// A length used by style inputs, resolved later against a reference length.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Dimension {
    /// The length is determined automatically by the layout algorithm.
    #[default]
    Auto,
    /// An absolute length in layout units.
    Points(f32),
    /// A fraction of a reference length, where `1.0` means 100%.
    Percent(f32),
}

impl Dimension {
    /// Resolves this dimension against an optional reference length.
    ///
    /// [`Dimension::Auto`] always resolves to `None`. [`Dimension::Points`]
    /// resolves to its absolute value. [`Dimension::Percent`] resolves only
    /// when a `basis` is provided.
    pub fn resolve(self, basis: Option<f32>) -> Option<f32> {
        match self {
            Dimension::Auto => None,
            Dimension::Points(value) => Some(value),
            Dimension::Percent(fraction) => basis.map(|b| b * fraction),
        }
    }

    /// Resolves this dimension, falling back to zero when it is unresolved.
    pub fn resolve_or_zero(self, basis: Option<f32>) -> f32 {
        self.resolve(basis).unwrap_or(0.0)
    }

    /// Returns `true` when this dimension is [`Dimension::Auto`].
    pub fn is_auto(self) -> bool {
        matches!(self, Dimension::Auto)
    }
}

impl Edges<Dimension> {
    /// Resolves every edge against an optional reference length, treating
    /// unresolved edges as zero.
    pub fn resolve_or_zero(self, basis: Option<f32>) -> Edges<f32> {
        Edges {
            left: self.left.resolve_or_zero(basis),
            right: self.right.resolve_or_zero(basis),
            top: self.top.resolve_or_zero(basis),
            bottom: self.bottom.resolve_or_zero(basis),
        }
    }
}

impl Size<Dimension> {
    /// Resolves each dimension against the matching reference length.
    pub fn resolve(self, basis: Size<Option<f32>>) -> Size<Option<f32>> {
        Size {
            width: self.width.resolve(basis.width),
            height: self.height.resolve(basis.height),
        }
    }
}

/// A constraint describing how much space is available to a box along one
/// axis while it is being sized.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AvailableSpace {
    /// A concrete amount of space is available.
    Definite(f32),
    /// The box should assume the smallest space, producing its min-content
    /// size.
    MinContent,
    /// The box should assume unbounded space, producing its max-content
    /// size.
    MaxContent,
}

impl AvailableSpace {
    /// Returns the definite value when this is [`AvailableSpace::Definite`],
    /// and `None` otherwise.
    pub fn into_option(self) -> Option<f32> {
        match self {
            AvailableSpace::Definite(value) => Some(value),
            AvailableSpace::MinContent | AvailableSpace::MaxContent => None,
        }
    }

    /// Returns `true` for [`AvailableSpace::MaxContent`].
    pub fn is_max_content(self) -> bool {
        matches!(self, AvailableSpace::MaxContent)
    }
}
