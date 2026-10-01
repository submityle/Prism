//! Displacement heightfield: a regular grid of height samples ray-traced as an
//! implicit triangulated terrain surface.
//!
//! Terrain, displaced water sheets, and sculpted detail are authored as a 2-D
//! array of heights rather than an explicit vertex/index mesh. [`Heightfield`]
//! stores just those `width * height` samples plus a planar domain (`origin`
//! and the `extent` of the grid in the `X`/`Z` plane) and *implicitly* generates
//! the surface: grid vertex `(ix, iz)` sits at world position
//! `origin + (ix/(width-1) * extent.x, heights[iz*width+ix], iz/(height-1) *
//! extent.y)`, and each grid cell is split into two triangles. This is the
//! compact representation a `GPU` kernel wants — a height texture, not a fat
//! `BLAS` — while still giving exact, watertight triangle hits.
//!
//! Ray intersection reuses the proven Möller–Trumbore test per cell triangle
//! (same arithmetic as [`super::triangle_mesh`]) and reports the geometric
//! normal oriented against the ray, the interpolated domain `UV`, and the cell /
//! triangle that was struck. [`HeightfieldBvh`] accelerates this with a binned
//! `SAH` `BVH` over the per-cell bounds (each cell's four displaced corners), so
//! a leaf tests only its two triangles — mirroring every other `ray_scene`
//! primitive. The on-device variant adds only the packed height samples and the
//! domain header to the shared `BVH` node layout, so it lives in a sibling
//! `heightfield_gpu_layout` module.

use super::bvh::{Aabb, BvhBuildConfig, LinearBvhNode, build_linear_bvh};
use super::traversal::Ray;

/// Why [`Heightfield::new`] rejected its inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeightfieldError {
    /// A grid dimension was below two, so no cell could be formed.
    DegenerateGrid {
        /// Requested column count.
        width: usize,
        /// Requested row count.
        height: usize,
    },
    /// The sample count did not equal `width * height`.
    SampleCountMismatch {
        /// Number of samples supplied.
        samples: usize,
        /// Number of samples required (`width * height`).
        expected: usize,
    },
}

impl core::fmt::Display for HeightfieldError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DegenerateGrid { width, height } => write!(
                f,
                "heightfield grid must be at least 2 x 2 but was {width} x {height}"
            ),
            Self::SampleCountMismatch { samples, expected } => write!(
                f,
                "heightfield has {samples} samples but {expected} were expected"
            ),
        }
    }
}

impl std::error::Error for HeightfieldError {}

/// A single ray/heightfield intersection record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightfieldHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Barycentric `u` within the struck triangle (weight of its 2nd vertex).
    pub u: f32,
    /// Barycentric `v` within the struck triangle (weight of its 3rd vertex).
    pub v: f32,
    /// Flattened cell index (`iz * (width - 1) + ix`) that was struck.
    pub cell: u32,
    /// Which of the cell's two triangles was struck (`0` or `1`).
    pub triangle: u8,
    /// World-space hit position reconstructed from the barycentric weights.
    pub position: [f32; 3],
    /// Unit geometric normal oriented into the ray's hemisphere.
    pub normal: [f32; 3],
    /// Interpolated domain `UV` in `[0, 1]^2` (grid parameterization).
    pub uv: [f32; 2],
    /// `true` when the ray struck the front (counter-clockwise) face.
    pub front_face: bool,
}

/// A regular grid of height samples displaced into an implicit terrain surface.
///
/// Samples are stored row-major (`heights[iz * width + ix]`). The grid spans the
/// axis-aligned rectangle `[origin.xz, origin.xz + extent]` in the `X`/`Z` plane
/// and displaces along `Y`; `origin.y` is the base height added to every sample.
/// All stored fields are kept verbatim so a flat `GPU` layout decodes a
/// bit-identical primitive.
#[derive(Clone, Debug, PartialEq)]
pub struct Heightfield {
    /// Grid columns (vertices along `X`); always at least two.
    width: usize,
    /// Grid rows (vertices along `Z`); always at least two.
    height: usize,
    /// Row-major height samples, `width * height` of them.
    heights: Vec<f32>,
    /// World-space corner the grid's `(0, 0)` vertex maps to.
    origin: [f32; 3],
    /// Grid extent along `X` and `Z` in world units.
    extent: [f32; 2],
}

impl Heightfield {
    /// Builds a heightfield from a `width * height` row-major sample grid.
    ///
    /// # Errors
    ///
    /// Returns [`HeightfieldError`] when a dimension is below two or the sample
    /// count does not equal `width * height`.
    pub fn new(
        width: usize,
        height: usize,
        heights: Vec<f32>,
        origin: [f32; 3],
        extent: [f32; 2],
    ) -> Result<Self, HeightfieldError> {
        if width < 2 || height < 2 {
            return Err(HeightfieldError::DegenerateGrid { width, height });
        }
        let expected = width * height;
        if heights.len() != expected {
            return Err(HeightfieldError::SampleCountMismatch {
                samples: heights.len(),
                expected,
            });
        }
        Ok(Self {
            width,
            height,
            heights,
            origin,
            extent,
        })
    }

    /// Grid columns (vertices along `X`).
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Grid rows (vertices along `Z`).
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Row-major height samples.
    #[must_use]
    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    /// World-space corner the grid's `(0, 0)` vertex maps to.
    #[must_use]
    pub fn origin(&self) -> [f32; 3] {
        self.origin
    }

    /// Grid extent along `X` and `Z` in world units.
    #[must_use]
    pub fn extent(&self) -> [f32; 2] {
        self.extent
    }

    /// Number of cells along `X` (`width - 1`).
    #[must_use]
    pub fn cells_x(&self) -> usize {
        self.width - 1
    }

    /// Number of cells along `Z` (`height - 1`).
    #[must_use]
    pub fn cells_z(&self) -> usize {
        self.height - 1
    }

    /// Total number of grid cells (`cells_x * cells_z`).
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cells_x() * self.cells_z()
    }

    /// Raw height sample at grid coordinate `(ix, iz)`.
    #[must_use]
    pub fn sample(&self, ix: usize, iz: usize) -> f32 {
        self.heights[iz * self.width + ix]
    }

    /// World-space position of grid vertex `(ix, iz)`.
    #[must_use]
    pub fn vertex(&self, ix: usize, iz: usize) -> [f32; 3] {
        let (fx, fz) = self.grid_param(ix, iz);
        [
            self.origin[0] + fx * self.extent[0],
            self.origin[1] + self.sample(ix, iz),
            self.origin[2] + fz * self.extent[1],
        ]
    }

    /// Domain `UV` of grid vertex `(ix, iz)` in `[0, 1]^2`.
    fn grid_param(&self, ix: usize, iz: usize) -> (f32, f32) {
        (
            ix as f32 / (self.width - 1) as f32,
            iz as f32 / (self.height - 1) as f32,
        )
    }

    /// The three corner positions and domain `UV`s of `cell`'s `tri`-th
    /// triangle (`tri` is `0` or `1`), in winding order.
    ///
    /// Triangle `0` is `(ix,iz) → (ix+1,iz) → (ix+1,iz+1)` and triangle `1` is
    /// `(ix,iz) → (ix+1,iz+1) → (ix,iz+1)`; both wind counter-clockwise when
    /// viewed from `+Y`.
    fn triangle_corners(&self, cell: usize, tri: u8) -> [([f32; 3], [f32; 2]); 3] {
        let cells_x = self.cells_x();
        let ix = cell % cells_x;
        let iz = cell / cells_x;
        let corner = |cx: usize, cz: usize| {
            let (u, v) = self.grid_param(cx, cz);
            (self.vertex(cx, cz), [u, v])
        };
        if tri == 0 {
            [
                corner(ix, iz),
                corner(ix + 1, iz),
                corner(ix + 1, iz + 1),
            ]
        } else {
            [
                corner(ix, iz),
                corner(ix + 1, iz + 1),
                corner(ix, iz + 1),
            ]
        }
    }

    /// Axis-aligned bounds of `cell` (the box over its four displaced corners).
    #[must_use]
    pub fn cell_aabb(&self, cell: usize) -> Aabb {
        let cells_x = self.cells_x();
        let ix = cell % cells_x;
        let iz = cell / cells_x;
        let mut aabb = Aabb::empty();
        for (dz, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            aabb = aabb.union(&point_aabb(self.vertex(ix + dx, iz + dz)));
        }
        aabb
    }

    /// Bounds of the entire displaced surface.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let mut aabb = Aabb::empty();
        for iz in 0..self.height {
            for ix in 0..self.width {
                aabb = aabb.union(&point_aabb(self.vertex(ix, iz)));
            }
        }
        aabb
    }

    /// Intersects `ray` with `cell`'s `tri`-th triangle via Möller–Trumbore.
    ///
    /// The arithmetic matches [`super::triangle_mesh::TriangleMesh::intersect_triangle`]
    /// so the `GPU` layout — which decodes this same primitive and calls this
    /// same method — reproduces every hit bit-for-bit.
    #[must_use]
    pub fn intersect_cell_triangle(
        &self,
        cell: usize,
        tri: u8,
        ray: &Ray,
    ) -> Option<HeightfieldHit> {
        const EPS: f32 = 1e-8;
        let [(p0, t0), (p1, t1), (p2, t2)] = self.triangle_corners(cell, tri);
        let e1 = sub(p1, p0);
        let e2 = sub(p2, p0);
        let dir = ray.direction();
        let p = cross(dir, e2);
        let det = dot(e1, p);
        if det.abs() < EPS {
            return None;
        }
        let inv_det = 1.0 / det;
        let tvec = sub(ray.origin(), p0);
        let u = dot(tvec, p) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = cross(tvec, e1);
        let v = dot(dir, q) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = dot(e2, q) * inv_det;
        if t < ray.t_min() || t > ray.t_max() {
            return None;
        }

        let w0 = 1.0 - u - v;
        let position = [
            w0 * p0[0] + u * p1[0] + v * p2[0],
            w0 * p0[1] + u * p1[1] + v * p2[1],
            w0 * p0[2] + u * p1[2] + v * p2[2],
        ];

        let ng_raw = cross(e1, e2);
        let front_face = dot(dir, ng_raw) < 0.0;
        let geo = if front_face { ng_raw } else { negate(ng_raw) };
        let normal = normalize_or(geo, geo);

        let uv = [
            w0 * t0[0] + u * t1[0] + v * t2[0],
            w0 * t0[1] + u * t1[1] + v * t2[1],
        ];

        Some(HeightfieldHit {
            t,
            u,
            v,
            cell: cell as u32,
            triangle: tri,
            position,
            normal,
            uv,
            front_face,
        })
    }

    /// Nearest intersection along `ray` by brute force over every cell triangle.
    ///
    /// This is the un-accelerated reference the [`HeightfieldBvh`] is validated
    /// against; both shrink the ray interval on each closer hit so the result is
    /// the globally nearest surface point.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<HeightfieldHit> {
        let mut ray = *ray;
        let mut best: Option<HeightfieldHit> = None;
        for cell in 0..self.cell_count() {
            for tri in 0..2u8 {
                if let Some(hit) = self.intersect_cell_triangle(cell, tri, &ray) {
                    ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                    best = Some(hit);
                }
            }
        }
        best
    }

    /// True when any cell triangle intersects `ray`.
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        for cell in 0..self.cell_count() {
            for tri in 0..2u8 {
                if self.intersect_cell_triangle(cell, tri, ray).is_some() {
                    return true;
                }
            }
        }
        false
    }
}

/// A binned-`SAH` `BVH` over a [`Heightfield`]'s cells.
///
/// The hierarchy is identical in spirit to [`super::triangle_mesh::TriangleMeshBvh`]:
/// each `BVH` primitive is one grid *cell*, whose bounds enclose its four
/// displaced corners, and a leaf tests both of the cell's triangles. The owned
/// heightfield is kept intact so a flat `GPU` layout decodes bit-identically.
#[derive(Clone, Debug)]
pub struct HeightfieldBvh {
    /// The owned heightfield, kept verbatim.
    field: Heightfield,
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Maps each `BVH` primitive slot back to its original cell index.
    order: Vec<u32>,
}

impl HeightfieldBvh {
    /// Builds the `BVH` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(field: Heightfield) -> Self {
        Self::build_with(field, BvhBuildConfig::default())
    }

    /// Builds the `BVH` over the field's cells with the given binned-`SAH`
    /// `config`.
    #[must_use]
    pub fn build_with(field: Heightfield, config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = (0..field.cell_count())
            .map(|cell| field.cell_aabb(cell))
            .collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        Self {
            field,
            nodes,
            order,
        }
    }

    /// The owned heightfield.
    #[must_use]
    pub fn field(&self) -> &Heightfield {
        &self.field
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of pooled cells.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.order.len()
    }

    /// True when the hierarchy holds no cells.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Root bounds, or the empty box when the hierarchy is empty.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map_or_else(Aabb::empty, |n| n.bounds)
    }

    /// The flattened node array.
    #[must_use]
    pub fn nodes(&self) -> &[LinearBvhNode] {
        &self.nodes
    }

    /// Maps each `BVH` primitive slot back to its original cell index.
    #[must_use]
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// Nearest intersection along `ray`; mirrors
    /// [`super::triangle_mesh::TriangleMeshBvh::closest_hit`] with a per-cell
    /// two-triangle leaf test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<HeightfieldHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<HeightfieldHit> = None;

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let node = &self.nodes[node_index as usize];
            if ray
                .aabb_interval(&node.bounds, ray.t_min(), ray.t_max())
                .is_some()
            {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for &slot in &self.order[start..end] {
                        for tri in 0..2u8 {
                            if let Some(hit) =
                                self.field.intersect_cell_triangle(slot as usize, tri, &ray)
                            {
                                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                                best = Some(hit);
                            }
                        }
                    }
                    match stack_pop(&mut stack, &mut sp) {
                        Some(n) => node_index = n,
                        None => break,
                    }
                } else {
                    let first_child = node_index + 1;
                    let second_child = node.second_child;
                    let neg = ray.direction()[node.axis as usize] < 0.0;
                    let (near, far) = if neg {
                        (second_child, first_child)
                    } else {
                        (first_child, second_child)
                    };
                    if sp < stack.len() {
                        stack[sp] = far;
                        sp += 1;
                    }
                    node_index = near;
                }
            } else {
                match stack_pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        best
    }

    /// True when any cell triangle intersects `ray`; mirrors
    /// [`super::triangle_mesh::TriangleMeshBvh::any_hit`].
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let node = &self.nodes[node_index as usize];
            if ray
                .aabb_interval(&node.bounds, ray.t_min(), ray.t_max())
                .is_some()
            {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for &slot in &self.order[start..end] {
                        for tri in 0..2u8 {
                            if self
                                .field
                                .intersect_cell_triangle(slot as usize, tri, ray)
                                .is_some()
                            {
                                return true;
                            }
                        }
                    }
                    match stack_pop(&mut stack, &mut sp) {
                        Some(n) => node_index = n,
                        None => break,
                    }
                } else {
                    let first_child = node_index + 1;
                    if sp < stack.len() {
                        stack[sp] = node.second_child;
                        sp += 1;
                    }
                    node_index = first_child;
                }
            } else {
                match stack_pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        false
    }
}

/// A degenerate [`Aabb`] enclosing a single point.
fn point_aabb(p: [f32; 3]) -> Aabb {
    Aabb::new(p, p)
}

/// Component-wise `a - b`.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Negates each component.
fn negate(v: [f32; 3]) -> [f32; 3] {
    [-v[0], -v[1], -v[2]]
}

/// Right-handed cross product `a × b`.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product `a · b`.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Normalizes `v`, falling back to `fallback` when `v` is near zero length.
fn normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len_sq = dot(v, v);
    if len_sq < 1e-24 {
        return fallback;
    }
    let inv = 1.0 / len_sq.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Pops the top node index off the traversal stack, or `None` when empty.
fn stack_pop(stack: &mut [u32; 64], sp: &mut usize) -> Option<u32> {
    if *sp == 0 {
        None
    } else {
        *sp -= 1;
        Some(stack[*sp])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic xorshift `RNG`, matching the other `ray_scene`
    /// suites so tests never depend on an external crate.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / u32::MAX as f32
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
        fn point(&mut self, lo: f32, hi: f32) -> [f32; 3] {
            [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
        }
    }

    /// Independent (intersect-free) vector helpers for the oracles.
    fn o_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    fn o_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    fn o_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    fn o_norm(v: [f32; 3]) -> [f32; 3] {
        let inv = 1.0 / o_dot(v, v).sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }

    /// Builds a random `width * height` heightfield with bounded heights and a
    /// random planar domain.
    fn random_field(rng: &mut Rng, width: usize, height: usize) -> Heightfield {
        let heights: Vec<f32> = (0..width * height).map(|_| rng.range(-1.5, 1.5)).collect();
        let origin = rng.point(-5.0, 5.0);
        let extent = [rng.range(2.0, 6.0), rng.range(2.0, 6.0)];
        Heightfield::new(width, height, heights, origin, extent).expect("valid grid")
    }

    /// Reconstructs a cell triangle's three world corners purely from the public
    /// [`Heightfield::vertex`] accessor, serving as an independent oracle for the
    /// private corner lookup.
    fn oracle_corners(field: &Heightfield, cell: usize, tri: u8) -> [[f32; 3]; 3] {
        let cells_x = field.cells_x();
        let ix = cell % cells_x;
        let iz = cell / cells_x;
        if tri == 0 {
            [
                field.vertex(ix, iz),
                field.vertex(ix + 1, iz),
                field.vertex(ix + 1, iz + 1),
            ]
        } else {
            [
                field.vertex(ix, iz),
                field.vertex(ix + 1, iz + 1),
                field.vertex(ix, iz + 1),
            ]
        }
    }

    #[test]
    fn new_rejects_degenerate_and_mismatch() {
        assert_eq!(
            Heightfield::new(1, 4, vec![0.0; 4], [0.0; 3], [1.0, 1.0]).unwrap_err(),
            HeightfieldError::DegenerateGrid {
                width: 1,
                height: 4,
            }
        );
        assert_eq!(
            Heightfield::new(3, 3, vec![0.0; 8], [0.0; 3], [1.0, 1.0]).unwrap_err(),
            HeightfieldError::SampleCountMismatch {
                samples: 8,
                expected: 9,
            }
        );
        let field = Heightfield::new(3, 2, vec![0.0; 6], [0.0; 3], [4.0, 2.0]).unwrap();
        assert_eq!(field.cells_x(), 2);
        assert_eq!(field.cells_z(), 1);
        assert_eq!(field.cell_count(), 2);
    }

    #[test]
    fn vertex_matches_oracle() {
        let mut rng = Rng::new(0x1234_5678);
        for _ in 0..200 {
            let width = 2 + (rng.next_u32() % 5) as usize;
            let height = 2 + (rng.next_u32() % 5) as usize;
            let field = random_field(&mut rng, width, height);
            for iz in 0..height {
                for ix in 0..width {
                    let fx = ix as f32 / (width - 1) as f32;
                    let fz = iz as f32 / (height - 1) as f32;
                    let expected = [
                        field.origin()[0] + fx * field.extent()[0],
                        field.origin()[1] + field.sample(ix, iz),
                        field.origin()[2] + fz * field.extent()[1],
                    ];
                    let got = field.vertex(ix, iz);
                    for k in 0..3 {
                        assert!((got[k] - expected[k]).abs() < 1e-5);
                    }
                }
            }
        }
    }

    #[test]
    fn hits_lie_on_their_triangle_plane() {
        let mut rng = Rng::new(0xABCD_1234);
        let mut hits = 0usize;
        for _ in 0..6000 {
            let width = 2 + (rng.next_u32() % 5) as usize;
            let height = 2 + (rng.next_u32() % 5) as usize;
            let field = random_field(&mut rng, width, height);
            let cell = (rng.next_u32() as usize) % field.cell_count();
            let tri = (rng.next_u32() % 2) as u8;
            let [p0, p1, p2] = oracle_corners(&field, cell, tri);
            let mut a = rng.unit();
            let mut b = rng.unit();
            if a + b > 1.0 {
                a = 1.0 - a;
                b = 1.0 - b;
            }
            let w0 = 1.0 - a - b;
            let target = [
                w0 * p0[0] + a * p1[0] + b * p2[0],
                w0 * p0[1] + a * p1[1] + b * p2[1],
                w0 * p0[2] + a * p1[2] + b * p2[2],
            ];
            let origin = o_sub(target, rng.point(-4.0, 4.0));
            let d = o_sub(target, origin);
            if o_dot(d, d).sqrt() < 0.5 {
                continue;
            }
            let dir = o_norm(d);
            let ng = o_norm(o_cross(o_sub(p1, p0), o_sub(p2, p0)));
            if o_dot(dir, ng).abs() < 0.2 {
                continue; // grazing: reject for numerical stability
            }
            let ray = Ray::new(origin, dir, 1e-3, 1e6);
            if let Some(hit) = field.intersect(&ray) {
                // The nearest hit may be a different triangle than the one we
                // aimed at, so validate against the triangle actually struck.
                let [h0, h1, h2] = oracle_corners(&field, hit.cell as usize, hit.triangle);
                let hng = o_norm(o_cross(o_sub(h1, h0), o_sub(h2, h0)));
                assert!(o_dot(o_sub(hit.position, h0), hng).abs() < 2e-3);
                assert!(o_dot(dir, hit.normal) <= 1e-4);
                // Barycentric reconstruction agrees with the stored position.
                let rw0 = 1.0 - hit.u - hit.v;
                let recon = [
                    rw0 * h0[0] + hit.u * h1[0] + hit.v * h2[0],
                    rw0 * h0[1] + hit.u * h1[1] + hit.v * h2[1],
                    rw0 * h0[2] + hit.u * h1[2] + hit.v * h2[2],
                ];
                for (r, hp) in recon.iter().zip(hit.position) {
                    assert!((r - hp).abs() < 2e-3);
                }
                hits += 1;
            }
        }
        assert!(hits > 2000, "expected many hits, got {hits}");
    }

    #[test]
    fn bvh_matches_brute_force() {
        for seed in [0x5EED_F00D_u64, 0x0BAD_C0DE, 0xDEAD_BEEF] {
            let mut rng = Rng::new(seed);
            for _ in 0..40 {
                let width = 2 + (rng.next_u32() % 7) as usize;
                let height = 2 + (rng.next_u32() % 7) as usize;
                let field = random_field(&mut rng, width, height);
                let bvh = HeightfieldBvh::build(field.clone());
                assert!(!bvh.is_empty());
                assert_eq!(bvh.primitive_count(), field.cell_count());
                for _ in 0..60 {
                    let origin = rng.point(-8.0, 8.0);
                    let dir = o_norm(rng.point(-1.0, 1.0));
                    let ray = Ray::new(origin, dir, 1e-3, 1e6);
                    let brute = field.intersect(&ray);
                    let fast = bvh.closest_hit(&ray);
                    match (brute, fast) {
                        (None, None) => {}
                        (Some(a), Some(b)) => {
                            assert_eq!(a.t.to_bits(), b.t.to_bits());
                            assert_eq!(a.u.to_bits(), b.u.to_bits());
                            assert_eq!(a.v.to_bits(), b.v.to_bits());
                            assert_eq!(a.cell, b.cell);
                            assert_eq!(a.triangle, b.triangle);
                            assert_eq!(a.front_face, b.front_face);
                            for k in 0..3 {
                                assert_eq!(a.position[k].to_bits(), b.position[k].to_bits());
                                assert_eq!(a.normal[k].to_bits(), b.normal[k].to_bits());
                            }
                            for k in 0..2 {
                                assert_eq!(a.uv[k].to_bits(), b.uv[k].to_bits());
                            }
                        }
                        (a, b) => panic!("brute/bvh disagree: {a:?} vs {b:?}"),
                    }
                    assert_eq!(field.any_hit(&ray), bvh.any_hit(&ray));
                }
            }
        }
    }

    #[test]
    fn ray_that_misses_reports_nothing() {
        let field = Heightfield::new(
            2,
            2,
            vec![0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0],
        )
        .unwrap();
        let bvh = HeightfieldBvh::build(field.clone());
        assert_eq!(bvh.primitive_count(), 1);
        // Ray parallel to the flat sheet, well above it, never meets the plane.
        let ray = Ray::new([0.5, 5.0, 0.5], [1.0, 0.0, 0.0], 1e-3, 1e6);
        assert!(field.intersect(&ray).is_none());
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
        // Ray straight down through the centre does hit the sheet.
        let down = Ray::new([0.5, 5.0, 0.5], [0.0, -1.0, 0.0], 1e-3, 1e6);
        let hit = bvh.closest_hit(&down).expect("down ray hits sheet");
        assert!((hit.position[1]).abs() < 1e-4);
        assert!(hit.normal[1] > 0.0);
    }
}
