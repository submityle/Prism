//! Leaf measurement, used to size content such as text or images that the
//! layout algorithm cannot derive from style alone.

use crate::geometry::{AvailableSpace, Size};

/// Measures the intrinsic size of a leaf box's content.
///
/// Implementors receive the dimensions already fixed by style (`known`) and
/// the space available on each axis (`available`), and return the content's
/// preferred size. A `known` component of `Some(value)` means that axis is
/// already determined and the returned value on that axis is ignored by the
/// solver.
pub trait Measure {
    /// Returns the content size given the known dimensions and available
    /// space.
    fn measure(&self, known: Size<Option<f32>>, available: Size<AvailableSpace>) -> Size<f32>;
}

impl<F> Measure for F
where
    F: Fn(Size<Option<f32>>, Size<AvailableSpace>) -> Size<f32>,
{
    fn measure(&self, known: Size<Option<f32>>, available: Size<AvailableSpace>) -> Size<f32> {
        self(known, available)
    }
}
