//! Sphere-versus-heightfield contact geometry, shared bit-for-bit with the
//! `WGSL` kernel.
//!
//! A [`Heightfield`] is a regular `rows x cols` grid of height samples spaced
//! `cell_size` apart on the world `XZ` plane, with `Y` up: the sample at grid
//! index `(r, c)` lives at the world point `origin + (c * cell_size,
//! height[r][c], r * cell_size)`. Four neighbouring samples bound one quad
//! *cell*; a `rows x cols` sample grid therefore has `(rows - 1) x (cols - 1)`
//! cells. Each cell is split into two triangles along the `(r, c)`-to-`(r+1,
//! c+1)` diagonal, wound so the geometric face normal of a flat field points
//! `+Y` (up).
//!
//! [`cpu_sphere_heightfield_narrowphase`] is the `CPU` golden twin: for each
//! candidate `(sphere, heightfield)` pair it finds the grid cells overlapping
//! the sphere's `XZ` bounding box, rebuilds each candidate cell's two triangles,
//! runs the shared sphere-versus-triangle narrow phase on all of them (reusing
//! [`closest_point_on_triangle`](super::sphere_triangle) via the private
//! `sphere_triangle_contact`, so the Voronoi-region logic lives in exactly one
//! place), and reduces to the single deepest contact for that sphere. The
//! device kernel (`shaders/narrowphase_sphere_heightfield.wgsl`) runs the exact
//! same arithmetic in the same order, so the real-device parity test matches it
//! slot for slot.
//!
//! # Geometry
//!
//! A heightfield collider is how a sphere (or any bounding-sphere particle)
//! rests on terrain: the broad phase surfaces a `(sphere, heightfield)` pair,
//! and this slice turns it into the deepest sphere-versus-triangle contact over
//! the handful of cells the sphere's footprint can touch. The candidate cells
//! come from [`Heightfield::xz_cell_range`], which maps the sphere's `XZ` box to
//! an inclusive grid-index rectangle (or reports no overlap when the box misses
//! the grid entirely); [`Heightfield::cell_triangles`] builds a cell's two
//! triangles.
//!
//! # Normal convention
//!
//! The reported [`Contact`] stores the sphere index in `a` and the heightfield
//! index in `b`; its unit normal points **from the heightfield toward the
//! sphere**, the direction that pushes a dynamic sphere off static terrain,
//! matching the sphere-versus-triangle slice it is built on.
//!
//! # Deepest-contact reduction
//!
//! A sphere straddling a cell edge or vertex can touch several triangles at
//! once; the solver wants one manifold, so the reduction keeps the contact with
//! the greatest penetration `depth`. Ties are broken by iteration order (row
//! outer, column inner, first triangle before second), and a candidate replaces
//! the running best only when its depth is *strictly* greater, so the first
//! deepest contact wins. The device kernel iterates in the identical order with
//! the identical strict comparison, so both paths pick the same winner; the
//! parity scenes avoid symmetric ties that would make the choice ambiguous.
//!
//! Provenance: closest-point-on-triangle is the Voronoi-region method from
//! Christer Ericson, *Real-Time Collision Detection* (2004), section 5.1.5; the
//! heightfield cell triangulation is textbook. No Unreal Engine source or
//! derived code.

use glam::Vec3;

use super::contact::Contact;
use super::sphere_triangle::{sphere_triangle_contact, Triangle};

/// A regular grid of height samples on the world `XZ` plane with `Y` up.
///
/// The grid has `rows x cols` samples spaced `cell_size` apart; the sample at
/// index `(r, c)` is the world point `origin + (c * cell_size, height[r][c],
/// r * cell_size)`. Heights are stored row-major, so `height(r, c)` reads
/// `heights[r * cols + c]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Heightfield {
    /// Number of sample rows (along `+Z`).
    rows: u32,
    /// Number of sample columns (along `+X`).
    cols: u32,
    /// Spacing between adjacent samples on both the `X` and `Z` axes.
    cell_size: f32,
    /// World point of the `(0, 0)` sample's `XZ` corner and the `Y` datum the
    /// stored heights are added to.
    origin: Vec3,
    /// Row-major height samples; `heights[r * cols + c]` is the `(r, c)` height.
    heights: Vec<f32>,
}

/// An axis-aligned bounding rectangle on the world `XZ` plane, the sphere
/// footprint [`Heightfield::xz_cell_range`] maps to candidate cells.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct XzAabb {
    /// Minimum world `X`.
    pub min_x: f32,
    /// Maximum world `X`.
    pub max_x: f32,
    /// Minimum world `Z`.
    pub min_z: f32,
    /// Maximum world `Z`.
    pub max_z: f32,
}

/// An inclusive rectangle of grid *cell* indices, the candidate cells a sphere
/// footprint overlaps. Cell `(r, c)` is bounded by the samples `(r, c)`,
/// `(r, c+1)`, `(r+1, c)`, and `(r+1, c+1)`, so a valid cell row is in
/// `0..=rows-2` and a valid cell column in `0..=cols-2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellRange {
    /// First cell row (inclusive).
    pub min_row: u32,
    /// Last cell row (inclusive).
    pub max_row: u32,
    /// First cell column (inclusive).
    pub min_col: u32,
    /// Last cell column (inclusive).
    pub max_col: u32,
}

impl Heightfield {
    /// Builds a heightfield from its grid dimensions, spacing, origin, and
    /// row-major height samples.
    ///
    /// # Panics
    ///
    /// Panics if `heights.len()` is not exactly `rows * cols`, which would make
    /// the sample indexing ill-defined.
    #[must_use]
    pub fn new(rows: u32, cols: u32, cell_size: f32, origin: Vec3, heights: Vec<f32>) -> Heightfield {
        assert_eq!(
            heights.len(),
            rows as usize * cols as usize,
            "heightfield needs exactly rows * cols samples"
        );
        Heightfield {
            rows,
            cols,
            cell_size,
            origin,
            heights,
        }
    }

    /// Number of sample rows (along `+Z`).
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Number of sample columns (along `+X`).
    #[must_use]
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Spacing between adjacent samples on both axes.
    #[must_use]
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// World origin: the `(0, 0)` sample's `XZ` corner and the `Y` datum.
    #[must_use]
    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    /// The row-major height samples.
    #[must_use]
    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    /// The height sample at grid index `(r, c)`.
    ///
    /// # Panics
    ///
    /// Panics if `r >= rows` or `c >= cols`.
    #[must_use]
    pub fn height(&self, r: u32, c: u32) -> f32 {
        assert!(r < self.rows && c < self.cols, "sample index out of range");
        self.heights[r as usize * self.cols as usize + c as usize]
    }

    /// World-space position of the sample at grid index `(r, c)`.
    ///
    /// # Panics
    ///
    /// Panics if `r >= rows` or `c >= cols`.
    #[must_use]
    pub fn vertex(&self, r: u32, c: u32) -> Vec3 {
        Vec3::new(
            self.origin.x + c as f32 * self.cell_size,
            self.origin.y + self.height(r, c),
            self.origin.z + r as f32 * self.cell_size,
        )
    }

    /// The two triangles of cell `(r, c)`, split along the `(r, c)`-to-`(r+1,
    /// c+1)` diagonal and wound so a flat field's face normal points `+Y`.
    ///
    /// The cell's four corner samples are `v00 = (r, c)`, `v01 = (r, c+1)`,
    /// `v10 = (r+1, c)`, and `v11 = (r+1, c+1)`. The returned triangles are
    /// `[v00, v10, v11]` and `[v00, v11, v01]`.
    ///
    /// # Panics
    ///
    /// Panics if `r + 1 >= rows` or `c + 1 >= cols`, i.e. `(r, c)` is not a
    /// valid cell.
    #[must_use]
    pub fn cell_triangles(&self, r: u32, c: u32) -> [Triangle; 2] {
        assert!(
            r + 1 < self.rows && c + 1 < self.cols,
            "cell index out of range"
        );
        let v00 = self.vertex(r, c);
        let v01 = self.vertex(r, c + 1);
        let v10 = self.vertex(r + 1, c);
        let v11 = self.vertex(r + 1, c + 1);
        [Triangle::new(v00, v10, v11), Triangle::new(v00, v11, v01)]
    }

    /// Maps a sphere's `XZ` bounding box to the inclusive rectangle of grid
    /// cells it overlaps, or [`None`] when the box misses the grid (or the grid
    /// has no cells because `rows < 2` or `cols < 2`).
    ///
    /// Cell column `c` covers `x` in `[origin.x + c * cell, origin.x + (c+1) *
    /// cell]`, so the overlapping columns run from `floor((min_x - origin.x) /
    /// cell)` to `floor((max_x - origin.x) / cell)`, clamped to the valid cell
    /// range `0..=cols-2`; rows follow the same mapping on `Z`. A box lying
    /// entirely before the first sample or beyond the last reports no overlap.
    #[must_use]
    pub fn xz_cell_range(&self, aabb: XzAabb) -> Option<CellRange> {
        if self.rows < 2 || self.cols < 2 {
            return None;
        }
        let last_col = self.cols - 2;
        let last_row = self.rows - 2;
        let span_x = (self.cols - 1) as f32 * self.cell_size;
        let span_z = (self.rows - 1) as f32 * self.cell_size;

        // Reject a footprint lying wholly off either side of the grid before it
        // can clamp onto a spurious border cell.
        if aabb.max_x < self.origin.x
            || aabb.min_x > self.origin.x + span_x
            || aabb.max_z < self.origin.z
            || aabb.min_z > self.origin.z + span_z
        {
            return None;
        }

        let inv = 1.0 / self.cell_size;
        let min_col = clamp_index((aabb.min_x - self.origin.x) * inv, last_col);
        let max_col = clamp_index((aabb.max_x - self.origin.x) * inv, last_col);
        let min_row = clamp_index((aabb.min_z - self.origin.z) * inv, last_row);
        let max_row = clamp_index((aabb.max_z - self.origin.z) * inv, last_row);

        Some(CellRange {
            min_row,
            max_row,
            min_col,
            max_col,
        })
    }
}

/// Floors `value` to a cell index and clamps it to `0..=last`.
///
/// Mirrors the `WGSL` `clamp(i32(floor(value)), 0, last)` so both paths pick the
/// same bounding cells.
fn clamp_index(value: f32, last: u32) -> u32 {
    let floored = value.floor();
    if floored <= 0.0 {
        0
    } else if floored >= last as f32 {
        last
    } else {
        floored as u32
    }
}

/// A candidate sphere-versus-heightfield pair: an index into the sphere slice
/// and an index into the heightfield slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeightfieldSpherePair {
    /// Index of the sphere (bounding-sphere particle) in the sphere slice.
    pub sphere: u32,
    /// Index of the heightfield in the heightfield slice.
    pub heightfield: u32,
}

impl HeightfieldSpherePair {
    /// Creates a sphere-versus-heightfield candidate pair.
    #[must_use]
    pub fn new(sphere: u32, heightfield: u32) -> HeightfieldSpherePair {
        HeightfieldSpherePair {
            sphere,
            heightfield,
        }
    }
}

/// Finds the deepest sphere-versus-heightfield contact for one pair, or [`None`]
/// when the sphere touches no cell triangle.
///
/// The sphere's `XZ` footprint maps to candidate cells; each cell's two
/// triangles run through the shared sphere-versus-triangle test, and the deepest
/// penetrating contact wins (first deepest on an exact tie). The arithmetic
/// mirrors `narrowphase_sphere_heightfield.wgsl` operation for operation.
#[must_use]
fn sphere_heightfield_contact(
    sphere_id: u32,
    field_id: u32,
    p: Vec3,
    r: f32,
    field: &Heightfield,
) -> Option<Contact> {
    let aabb = XzAabb {
        min_x: p.x - r,
        max_x: p.x + r,
        min_z: p.z - r,
        max_z: p.z + r,
    };
    let range = field.xz_cell_range(aabb)?;

    let mut best: Option<Contact> = None;
    for row in range.min_row..=range.max_row {
        for col in range.min_col..=range.max_col {
            for tri in &field.cell_triangles(row, col) {
                if let Some(contact) = sphere_triangle_contact(sphere_id, field_id, p, r, tri) {
                    // Strictly-greater replace keeps the first deepest contact,
                    // matching the kernel's reduction order exactly.
                    let deeper = best.is_none_or(|b| contact.depth > b.depth);
                    if deeper {
                        best = Some(contact);
                    }
                }
            }
        }
    }
    best
}

/// `CPU` golden twin of the sphere-versus-heightfield narrow phase.
///
/// Turns a set of candidate `pairs` into one contact slot each, in input order:
/// [`Some`] carrying the deepest manifold when the sphere penetrates the
/// terrain, or [`None`] when it is clear of every candidate cell. Emitting a
/// slot per pair (rather than compacting) keeps the contact index aligned with
/// the pair index, which the real-device parity test relies on.
///
/// # Panics
///
/// Panics if a pair references a sphere or heightfield index outside the
/// corresponding slice, which is never valid output from a broad phase over the
/// same sets.
#[must_use]
pub fn cpu_sphere_heightfield_narrowphase(
    spheres: &[crate::broadphase::Particle],
    fields: &[Heightfield],
    pairs: &[HeightfieldSpherePair],
) -> Vec<Option<Contact>> {
    pairs
        .iter()
        .map(|pair| {
            let sphere = &spheres[pair.sphere as usize];
            let field = &fields[pair.heightfield as usize];
            sphere_heightfield_contact(
                pair.sphere,
                pair.heightfield,
                sphere.position,
                sphere.radius,
                field,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broadphase::Particle;

    /// A flat 3x3 (so 2x2 cells) field at `y = 0` with unit spacing, origin at
    /// the world origin. Samples span `x,z` in `[0, 2]`.
    fn flat_field() -> Heightfield {
        Heightfield::new(3, 3, 1.0, Vec3::ZERO, vec![0.0; 9])
    }

    #[test]
    fn flat_cell_triangles_face_up() {
        // Both triangles of a flat cell have an upward (+Y) geometric normal, so
        // the degenerate centre-on-face fallback pushes a sphere up.
        let field = flat_field();
        let [t0, t1] = field.cell_triangles(0, 0);
        assert!(t0.raw_normal().normalize().dot(Vec3::Y) > 0.999);
        assert!(t1.raw_normal().normalize().dot(Vec3::Y) > 0.999);
    }

    #[test]
    fn hit_over_cell_interior() {
        // Sphere hovering over the interior of cell (0,0) at (0.3, 0.4, 0.3):
        // nearest point is straight below on a face, so the normal is +Y and the
        // depth is r - height above = 0.5 - 0.4 = 0.1.
        let field = flat_field();
        let spheres = [Particle::new(Vec3::new(0.3, 0.4, 0.3), 0.5)];
        let pairs = [HeightfieldSpherePair::new(0, 0)];
        let contacts = cpu_sphere_heightfield_narrowphase(&spheres, &field_slice(&field), &pairs);
        let c = contacts[0].expect("sphere over the interior must contact");
        assert_eq!(c.a, 0);
        assert_eq!(c.b, 0);
        assert!((c.normal - Vec3::Y).length() < 1.0e-6);
        assert!((c.depth - 0.1).abs() < 1.0e-6, "depth {}", c.depth);
        assert!((c.point - Vec3::new(0.3, 0.0, 0.3)).length() < 1.0e-6);
    }

    #[test]
    fn hit_over_cell_edge() {
        // Sphere centred over a shared cell edge (x = 1, the seam between the
        // two columns) at (1.0, 0.3, 0.4): still a flat +Y face contact, depth
        // 0.6 - 0.3 = 0.3. Both bordering cells report it; the reduction keeps
        // one deepest contact.
        let field = flat_field();
        let spheres = [Particle::new(Vec3::new(1.0, 0.3, 0.4), 0.6)];
        let pairs = [HeightfieldSpherePair::new(0, 0)];
        let contacts = cpu_sphere_heightfield_narrowphase(&spheres, &field_slice(&field), &pairs);
        let c = contacts[0].expect("sphere over a cell edge must contact");
        assert!((c.normal - Vec3::Y).length() < 1.0e-6);
        assert!((c.depth - 0.3).abs() < 1.0e-6, "depth {}", c.depth);
        assert!((c.point - Vec3::new(1.0, 0.0, 0.4)).length() < 1.0e-6);
    }

    #[test]
    fn hit_over_interior_vertex() {
        // Sphere centred over the shared interior vertex (1,1) at (1.0, 0.2,
        // 1.0): a flat +Y face contact, depth 0.5 - 0.2 = 0.3. All four cells
        // touch this vertex; the deepest-contact reduction yields a single slot.
        let field = flat_field();
        let spheres = [Particle::new(Vec3::new(1.0, 0.2, 1.0), 0.5)];
        let pairs = [HeightfieldSpherePair::new(0, 0)];
        let contacts = cpu_sphere_heightfield_narrowphase(&spheres, &field_slice(&field), &pairs);
        let c = contacts[0].expect("sphere over the interior vertex must contact");
        assert!((c.normal - Vec3::Y).length() < 1.0e-6);
        assert!((c.depth - 0.3).abs() < 1.0e-6, "depth {}", c.depth);
        assert!((c.point - Vec3::new(1.0, 0.0, 1.0)).length() < 1.0e-6);
    }

    #[test]
    fn hit_on_slope() {
        // A ramp climbing +1 per unit along +X over one 1x1 cell: samples are
        // (0,0,0)->0, (1,0,0)->1, (0,0,1)->0, (1,0,1)->1. A sphere above the
        // middle contacts the inclined face with a tilted (not +Y) normal.
        let field = Heightfield::new(2, 2, 1.0, Vec3::ZERO, vec![0.0, 1.0, 0.0, 1.0]);
        // Surface foot (0.3, 0.3, 0.7) on the plane y = x, pushed 0.3 along the
        // upward normal (-1, 1, 0)/sqrt(2) to the sphere centre; radius 0.4
        // leaves depth 0.4 - 0.3 = 0.1. The foot sits off the diagonal (z > x),
        // so the upper triangle [v00, v10, v11] owns the contact unambiguously.
        let n = Vec3::new(-1.0, 1.0, 0.0).normalize();
        let foot = Vec3::new(0.3, 0.3, 0.7);
        let centre = foot + n * 0.3;
        let spheres = [Particle::new(centre, 0.4)];
        let pairs = [HeightfieldSpherePair::new(0, 0)];
        let contacts = cpu_sphere_heightfield_narrowphase(&spheres, &field_slice(&field), &pairs);
        let c = contacts[0].expect("sphere above the slope must contact");
        assert!((c.normal - n).length() < 1.0e-5, "normal {:?}", c.normal);
        assert!((c.depth - 0.1).abs() < 1.0e-5, "depth {}", c.depth);
        assert!((c.point - foot).length() < 1.0e-5, "point {:?}", c.point);
    }

    #[test]
    fn clear_miss_above() {
        // Sphere well above the flat field: no cell triangle is within reach.
        let field = flat_field();
        let spheres = [Particle::new(Vec3::new(1.0, 5.0, 1.0), 0.5)];
        let pairs = [HeightfieldSpherePair::new(0, 0)];
        let contacts = cpu_sphere_heightfield_narrowphase(&spheres, &field_slice(&field), &pairs);
        assert!(contacts[0].is_none());
    }

    #[test]
    fn sphere_off_the_grid_misses() {
        // Sphere whose XZ footprint lies entirely to the -X side of the grid:
        // the cell range is empty, so there is no contact even at y = 0.
        let field = flat_field();
        let spheres = [Particle::new(Vec3::new(-3.0, 0.0, 1.0), 0.5)];
        let pairs = [HeightfieldSpherePair::new(0, 0)];
        let contacts = cpu_sphere_heightfield_narrowphase(&spheres, &field_slice(&field), &pairs);
        assert!(contacts[0].is_none());
        // The range helper agrees the footprint is off the grid.
        assert_eq!(
            field.xz_cell_range(XzAabb {
                min_x: -3.5,
                max_x: -2.5,
                min_z: 0.5,
                max_z: 1.5,
            }),
            None
        );
    }

    #[test]
    fn empty_pairs_returns_empty() {
        let field = flat_field();
        let spheres = [Particle::new(Vec3::ZERO, 1.0)];
        let contacts = cpu_sphere_heightfield_narrowphase(&spheres, &field_slice(&field), &[]);
        assert!(contacts.is_empty());
    }

    #[test]
    fn batch_preserves_order_and_indices() {
        // Three spheres against one field: a clear interior hit, a clear miss
        // above, and a hit over the interior vertex, checked in input order with
        // their pair indices intact.
        let field = flat_field();
        let spheres = [
            Particle::new(Vec3::new(0.3, 0.4, 0.3), 0.5), // hit cell (0,0)
            Particle::new(Vec3::new(1.0, 5.0, 1.0), 0.5), // clear gap above
            Particle::new(Vec3::new(1.0, 0.2, 1.0), 0.5), // hit the shared vertex
        ];
        let pairs = [
            HeightfieldSpherePair::new(0, 0),
            HeightfieldSpherePair::new(1, 0),
            HeightfieldSpherePair::new(2, 0),
        ];
        let contacts = cpu_sphere_heightfield_narrowphase(&spheres, &field_slice(&field), &pairs);
        assert_eq!(contacts.len(), 3);
        assert_eq!(contacts[0].expect("first hits").a, 0);
        assert!(contacts[1].is_none());
        assert_eq!(contacts[2].expect("third hits").a, 2);
    }

    #[test]
    fn second_field_index_is_reported() {
        // A pair against the second field in the slice must carry heightfield
        // index 1 in the contact's `b`, proving the field index is preserved.
        let flat = flat_field();
        let raised = Heightfield::new(3, 3, 1.0, Vec3::new(0.0, 1.0, 0.0), vec![0.0; 9]);
        let fields = [flat, raised];
        let spheres = [Particle::new(Vec3::new(1.0, 1.3, 1.0), 0.5)];
        let pairs = [HeightfieldSpherePair::new(0, 1)];
        let contacts = cpu_sphere_heightfield_narrowphase(&spheres, &fields, &pairs);
        let c = contacts[0].expect("sphere over the raised field must contact");
        assert_eq!(c.b, 1);
        assert!((c.normal - Vec3::Y).length() < 1.0e-6);
        assert!((c.depth - 0.2).abs() < 1.0e-6, "depth {}", c.depth);
    }

    #[test]
    fn degenerate_field_without_cells_misses() {
        // A single row of samples has no complete cell (rows < 2), so the range
        // helper reports no overlap and a sphere sitting on it finds no contact.
        let field = Heightfield::new(1, 4, 1.0, Vec3::ZERO, vec![0.0; 4]);
        let spheres = [Particle::new(Vec3::new(1.0, 0.0, 0.0), 1.0)];
        let pairs = [HeightfieldSpherePair::new(0, 0)];
        let contacts = cpu_sphere_heightfield_narrowphase(&spheres, &field_slice(&field), &pairs);
        assert!(contacts[0].is_none());
        assert_eq!(
            field.xz_cell_range(XzAabb {
                min_x: 0.0,
                max_x: 2.0,
                min_z: -1.0,
                max_z: 1.0,
            }),
            None
        );
    }

    #[test]
    fn cell_range_covers_full_footprint() {
        // A footprint spanning the whole 3x3-sample grid clamps to every cell.
        let field = flat_field();
        let range = field
            .xz_cell_range(XzAabb {
                min_x: -1.0,
                max_x: 3.0,
                min_z: -1.0,
                max_z: 3.0,
            })
            .expect("a footprint over the grid must map to cells");
        assert_eq!(
            range,
            CellRange {
                min_row: 0,
                max_row: 1,
                min_col: 0,
                max_col: 1,
            }
        );
    }

    /// Wraps a single field in a one-element slice for the batch entry point.
    fn field_slice(field: &Heightfield) -> [Heightfield; 1] {
        [field.clone()]
    }
}
