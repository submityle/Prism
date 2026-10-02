//! A fixed-column grid layout.
//!
//! [`GridProtocol`] arranges its children row-major into a fixed number of
//! columns. Each column is sized to the widest child it contains and each row
//! to the tallest child it contains, so the result is deterministic and
//! depends only on the child sizes and the gap configuration.

use alloc::vec::Vec;

use crate::geometry::{Point, Rect, Size};
use crate::protocol::{Constraints, LayoutProtocol};

/// A grid layout with a fixed column count and axis gaps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridProtocol {
    /// Number of columns. Values below one are treated as a single column.
    pub columns: usize,
    /// Horizontal gap between adjacent columns.
    pub column_gap: f32,
    /// Vertical gap between adjacent rows.
    pub row_gap: f32,
}

impl Default for GridProtocol {
    fn default() -> Self {
        Self {
            columns: 1,
            column_gap: 0.0,
            row_gap: 0.0,
        }
    }
}

impl GridProtocol {
    /// Creates a grid with `columns` columns and no gaps.
    pub fn new(columns: usize) -> Self {
        Self {
            columns: columns.max(1),
            column_gap: 0.0,
            row_gap: 0.0,
        }
    }

    /// Returns the effective (at least one) column count.
    fn cols(&self) -> usize {
        self.columns.max(1)
    }

    fn row_count(&self, child_count: usize) -> usize {
        let cols = self.cols();
        child_count.div_ceil(cols)
    }

    /// Computes the per-column widths and per-row heights for `children`.
    fn tracks(&self, children: &[Size<f32>]) -> (Vec<f32>, Vec<f32>) {
        let cols = self.cols();
        let rows = self.row_count(children.len());
        let mut col_widths = alloc::vec![0.0_f32; cols];
        let mut row_heights = alloc::vec![0.0_f32; rows];
        for (index, child) in children.iter().enumerate() {
            let col = index % cols;
            let row = index / cols;
            col_widths[col] = col_widths[col].max(child.width);
            row_heights[row] = row_heights[row].max(child.height);
        }
        (col_widths, row_heights)
    }
}

impl LayoutProtocol for GridProtocol {
    fn measure(&self, children: &[Size<f32>], constraints: Constraints) -> Size<f32> {
        if children.is_empty() {
            return constraints.constrain(Size::ZERO);
        }
        let (col_widths, row_heights) = self.tracks(children);
        let width = sum(&col_widths) + self.column_gap * gaps(col_widths.len());
        let height = sum(&row_heights) + self.row_gap * gaps(row_heights.len());
        constraints.constrain(Size::new(width, height))
    }

    fn place(&self, children: &[Size<f32>], bounds: Rect) -> Vec<Rect> {
        let cols = self.cols();
        let (col_widths, row_heights) = self.tracks(children);

        // Precompute the leading edge of every column and row.
        let col_offsets = offsets(&col_widths, self.column_gap);
        let row_offsets = offsets(&row_heights, self.row_gap);

        let mut rects = Vec::with_capacity(children.len());
        for (index, child) in children.iter().enumerate() {
            let col = index % cols;
            let row = index / cols;
            let x = bounds.location.x + col_offsets[col];
            let y = bounds.location.y + row_offsets[row];
            rects.push(Rect::new(Point::new(x, y), *child));
        }
        rects
    }
}

/// Returns the running leading offset of each track given its size and gap.
fn offsets(sizes: &[f32], gap: f32) -> Vec<f32> {
    let mut result = Vec::with_capacity(sizes.len());
    let mut cursor = 0.0;
    for &size in sizes {
        result.push(cursor);
        cursor += size + gap;
    }
    result
}

fn sum(values: &[f32]) -> f32 {
    let mut total = 0.0;
    for &v in values {
        total += v;
    }
    total
}

fn gaps(count: usize) -> f32 {
    if count <= 1 {
        0.0
    } else {
        (count - 1) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_column_grid_places_row_major() {
        let grid = GridProtocol {
            columns: 2,
            column_gap: 10.0,
            row_gap: 5.0,
        };
        let children = [
            Size::new(20.0, 20.0),
            Size::new(30.0, 20.0),
            Size::new(20.0, 40.0),
        ];
        let bounds = Rect::new(Point::ZERO, Size::new(200.0, 200.0));
        let rects = grid.place(&children, bounds);
        // Column widths: [max(20,20)=20, 30]; row heights: [20, 40].
        assert_eq!(rects[0].location, Point::new(0.0, 0.0));
        assert_eq!(rects[1].location, Point::new(20.0 + 10.0, 0.0));
        assert_eq!(rects[2].location, Point::new(0.0, 20.0 + 5.0));
    }

    #[test]
    fn grid_measure_sums_tracks_and_gaps() {
        let grid = GridProtocol {
            columns: 2,
            column_gap: 10.0,
            row_gap: 5.0,
        };
        let children = [
            Size::new(20.0, 20.0),
            Size::new(30.0, 20.0),
            Size::new(20.0, 40.0),
        ];
        let size = grid.measure(&children, Constraints::loose(Size::new(500.0, 500.0)));
        // width = 20 + 30 + 10 gap = 60; height = 20 + 40 + 5 gap = 65.
        assert_eq!(size, Size::new(60.0, 65.0));
    }

    #[test]
    fn zero_columns_falls_back_to_single_column() {
        let grid = GridProtocol::new(0);
        assert_eq!(grid.cols(), 1);
        let children = [Size::new(10.0, 10.0), Size::new(10.0, 10.0)];
        let rects = grid.place(&children, Rect::new(Point::ZERO, Size::new(50.0, 50.0)));
        assert_eq!(rects[0].location, Point::new(0.0, 0.0));
        assert_eq!(rects[1].location, Point::new(0.0, 10.0));
    }

    #[test]
    fn empty_grid_measures_to_zero() {
        let grid = GridProtocol::new(3);
        let size = grid.measure(&[], Constraints::loose(Size::new(100.0, 100.0)));
        assert_eq!(size, Size::ZERO);
    }
}
