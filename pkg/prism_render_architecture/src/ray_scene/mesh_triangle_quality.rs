//! Per-triangle quality and area statistics for the `CPU` golden path.
//!
//! Meshing, tessellation, and simulation all degrade on *sliver* triangles —
//! long thin faces whose tiny area relative to their edge lengths wrecks
//! interpolation, shadow bias, and numerical conditioning. The canonical
//! scale-invariant score is the **normalized shape quality** (the mean-ratio
//! metric):
//!
//! ```text
//! q = 4 * sqrt(3) * A / (l0^2 + l1^2 + l2^2)
//! ```
//!
//! where `A` is the triangle area and `l0..l2` its edge lengths. It equals
//! `1` for an equilateral triangle and falls toward `0` as the triangle
//! degenerates, so a single threshold flags slivers regardless of absolute
//! size. [`triangle_quality`] computes this score and the area for every face,
//! plus aggregate total area and min / average quality, in one pass.
//!
//! Area uses one `sqrt` of the cross-product magnitude and the `sqrt(3)`
//! numerator is itself a `sqrt`; everything else is squared edge lengths and
//! dot/cross products, so no `f32` transcendental function is used.

use super::triangle_mesh::TriangleMesh;

/// Per-triangle area and normalized shape-quality scores for a
/// [`TriangleMesh`].
#[derive(Clone, Debug)]
pub struct TriangleQuality {
    /// Area of each triangle.
    areas: Vec<f32>,
    /// Normalized shape quality of each triangle in `[0, 1]` (`1` =
    /// equilateral, `0` = degenerate).
    qualities: Vec<f32>,
}

impl TriangleQuality {
    /// Returns the number of triangles described.
    pub fn len(&self) -> usize {
        self.areas.len()
    }

    /// Returns whether the mesh had no triangles.
    pub fn is_empty(&self) -> bool {
        self.areas.is_empty()
    }

    /// Returns the area of triangle `index`, or `None` when out of range.
    pub fn area(&self, index: usize) -> Option<f32> {
        self.areas.get(index).copied()
    }

    /// Returns the normalized shape quality of triangle `index` in `[0, 1]`,
    /// or `None` when out of range.
    pub fn quality(&self, index: usize) -> Option<f32> {
        self.qualities.get(index).copied()
    }

    /// Returns the per-triangle areas.
    pub fn areas(&self) -> &[f32] {
        &self.areas
    }

    /// Returns the per-triangle normalized shape qualities.
    pub fn qualities(&self) -> &[f32] {
        &self.qualities
    }

    /// Returns the total surface area of the mesh.
    pub fn total_area(&self) -> f32 {
        self.areas.iter().copied().sum()
    }

    /// Returns the minimum shape quality over all triangles, or `None` for an
    /// empty mesh.
    pub fn min_quality(&self) -> Option<f32> {
        self.qualities.iter().copied().reduce(f32::min)
    }

    /// Returns the mean shape quality over all triangles, or `None` for an
    /// empty mesh.
    pub fn average_quality(&self) -> Option<f32> {
        if self.qualities.is_empty() {
            return None;
        }
        let sum: f32 = self.qualities.iter().copied().sum();
        Some(sum / self.qualities.len() as f32)
    }

    /// Returns the number of triangles whose quality is strictly below
    /// `threshold` — the sliver (and degenerate) faces.
    pub fn sliver_count(&self, threshold: f32) -> usize {
        self.qualities.iter().filter(|&&q| q < threshold).count()
    }

    /// Returns the number of fully degenerate (zero-area) triangles.
    pub fn degenerate_count(&self) -> usize {
        self.areas.iter().filter(|&&a| a <= 0.0).count()
    }
}

/// Computes the area and normalized shape quality of every triangle in `mesh`.
pub fn triangle_quality(mesh: &TriangleMesh) -> TriangleQuality {
    // sqrt(3) numerator of the normalized shape-quality metric.
    let sqrt3 = 3.0_f32.sqrt();
    let positions = mesh.positions();

    let mut areas = Vec::with_capacity(mesh.triangle_count());
    let mut qualities = Vec::with_capacity(mesh.triangle_count());

    for tri in mesh.indices() {
        let p0 = positions[tri[0] as usize];
        let p1 = positions[tri[1] as usize];
        let p2 = positions[tri[2] as usize];

        let e0 = sub(p1, p0);
        let e1 = sub(p2, p1);
        let e2 = sub(p0, p2);

        let cross = [
            e0[1] * (-e2[2]) - e0[2] * (-e2[1]),
            e0[2] * (-e2[0]) - e0[0] * (-e2[2]),
            e0[0] * (-e2[1]) - e0[1] * (-e2[0]),
        ];
        let cross_mag = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        let area = 0.5 * cross_mag;

        let sum_sq = dot(e0, e0) + dot(e1, e1) + dot(e2, e2);
        let quality = if sum_sq > 0.0 {
            (4.0 * sqrt3 * area / sum_sq).clamp(0.0, 1.0)
        } else {
            0.0
        };

        areas.push(area);
        qualities.push(quality);
    }

    TriangleQuality { areas, qualities }
}

/// Returns `a - b` componentwise.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Returns the dot product of two vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single equilateral triangle of unit edge length.
    fn equilateral() -> TriangleMesh {
        let h = 3.0_f32.sqrt() / 2.0;
        TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, h, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .unwrap()
    }

    /// A unit right-isosceles triangle (legs of length 1).
    fn right_isosceles() -> TriangleMesh {
        TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .unwrap()
    }

    /// A near-degenerate sliver: three almost-collinear points.
    fn sliver() -> TriangleMesh {
        TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.001, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .unwrap()
    }

    #[test]
    fn equilateral_quality_is_one() {
        let q = triangle_quality(&equilateral());
        assert_eq!(q.len(), 1);
        let quality = q.quality(0).unwrap();
        assert!((quality - 1.0).abs() < 1e-5, "quality = {quality}");
    }

    #[test]
    fn equilateral_area_matches_formula() {
        let q = triangle_quality(&equilateral());
        // Area of a unit equilateral triangle is sqrt(3) / 4.
        let expected = 3.0_f32.sqrt() / 4.0;
        assert!((q.area(0).unwrap() - expected).abs() < 1e-6);
    }

    #[test]
    fn right_isosceles_quality_is_known() {
        let q = triangle_quality(&right_isosceles());
        // area 0.5, sum of squared edges 1 + 2 + 1 = 4 -> q = sqrt(3) / 2.
        let expected = 3.0_f32.sqrt() / 2.0;
        assert!((q.quality(0).unwrap() - expected).abs() < 1e-6);
        assert!((q.area(0).unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sliver_has_low_quality() {
        let q = triangle_quality(&sliver());
        assert!(q.quality(0).unwrap() < 0.05);
        assert_eq!(q.sliver_count(0.1), 1);
        assert_eq!(q.sliver_count(0.001), 0);
    }

    #[test]
    fn degenerate_triangle_is_zero_quality() {
        // Three collinear points -> zero area, zero quality.
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .unwrap();
        let q = triangle_quality(&mesh);
        assert_eq!(q.area(0), Some(0.0));
        assert_eq!(q.quality(0), Some(0.0));
        assert_eq!(q.degenerate_count(), 1);
        assert_eq!(q.sliver_count(0.5), 1);
    }

    #[test]
    fn total_area_sums_triangles() {
        // Two unit right triangles forming a unit square -> total area 1.
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [2, 1, 3]],
        )
        .unwrap();
        let q = triangle_quality(&mesh);
        assert!((q.total_area() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn average_and_min_quality() {
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 3.0_f32.sqrt() / 2.0, 0.0],
                [0.5, 0.001, 0.0],
            ],
            Vec::new(),
            Vec::new(),
            // One equilateral (q ~= 1) and one sliver (q ~= 0).
            vec![[0, 1, 2], [0, 1, 3]],
        )
        .unwrap();
        let q = triangle_quality(&mesh);
        let min = q.min_quality().unwrap();
        let avg = q.average_quality().unwrap();
        assert!(min < 0.05);
        assert!(avg > 0.4 && avg < 0.6);
    }

    #[test]
    fn out_of_range_returns_none() {
        let q = triangle_quality(&equilateral());
        assert_eq!(q.area(5), None);
        assert_eq!(q.quality(5), None);
    }

    #[test]
    fn empty_mesh_has_no_triangles() {
        let mesh = TriangleMesh::new(Vec::new(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let q = triangle_quality(&mesh);
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        assert_eq!(q.total_area(), 0.0);
        assert_eq!(q.min_quality(), None);
        assert_eq!(q.average_quality(), None);
        assert_eq!(q.degenerate_count(), 0);
    }

    #[test]
    fn quality_is_scale_invariant() {
        // Scaling an equilateral triangle 100x leaves quality at 1.
        let h = 3.0_f32.sqrt() / 2.0;
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [50.0, 100.0 * h, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .unwrap();
        let q = triangle_quality(&mesh);
        assert!((q.quality(0).unwrap() - 1.0).abs() < 1e-5);
    }
}
