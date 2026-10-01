//! Froxel (frustum-voxel) grid with exponential depth slicing — CPU golden.
//!
//! A *froxel* grid is a view-frustum-aligned 3-D voxel volume used by
//! clustered / volumetric renderers (Frostbite's "Physically Based and Unified
//! Volumetric Lighting", Hillaire 2015).  The grid tiles the screen into
//! `dims.x * dims.y` columns and slices depth into `dims.z` layers.  Unlike a
//! uniform depth split, the slices are spaced *exponentially* so that near-field
//! froxels (where parallax and scattering detail matter) stay small while
//! far-field froxels grow, matching the `1/z` perspective density:
//!
//! ```text
//! slice_depth(s) = near * (far / near)^(s / N)      for s in [0, N]
//! ```
//!
//! This module is the backend-neutral numerical reference for the froxel
//! addressing the WESL/GPU twin must reproduce bit-for-bit.
//!
//! # Conventions
//! * View space is right-handed with the camera at the origin looking down the
//!   **+Z** axis; `+Z` is positive forward depth.  A froxel's view-space extent
//!   at depth `z` is `[-z * tan_half_fov, +z * tan_half_fov]` on each lateral
//!   axis, i.e. a standard symmetric perspective frustum.
//! * Froxel coordinates are *fractional*: `f.x in [0, dims.x]`,
//!   `f.y in [0, dims.y]`, `f.z in [0, dims.z]`, with integer values on cell
//!   boundaries and `+0.5` offsets landing on cell centres.  Lateral axes map
//!   linearly to normalised device coordinates; the depth axis maps through the
//!   exponential slicing above.
//! * World↔froxel mapping is parameterised by a `view_from_world` matrix (and
//!   its inverse `world_from_view`); the grid itself only knows the frustum.
//! * Every parameter is clamped on construction (`near > 0`, `far > near`,
//!   positive half-tangents, at least one cell per axis) and every mapping
//!   clamps its result into range, so the helpers never divide by zero, never
//!   index out of bounds, and never return `NaN`.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   and no `unsafe`.  Transcendental maths goes through [`bevy_math::ops`].

use bevy_math::{ops, Mat4, UVec3, Vec3};

/// Smallest positive scalar used to keep frustum maths well-conditioned.
const EPS: f32 = 1.0e-6;

/// A view-frustum-aligned voxel grid with exponential depth slicing.
///
/// Construct through [`FroxelGrid::new`] so all parameters are validated and
/// clamped; the public fields are then guaranteed to satisfy the invariants
/// documented on each field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FroxelGrid {
    /// Grid resolution in froxels along `(x, y, z)`; every component is `>= 1`.
    pub dims: UVec3,
    /// Near plane view-space depth; strictly positive.
    pub near: f32,
    /// Far plane view-space depth; strictly greater than [`near`](Self::near).
    pub far: f32,
    /// Tangent of the half horizontal field of view; strictly positive.
    pub tan_half_fov_x: f32,
    /// Tangent of the half vertical field of view; strictly positive.
    pub tan_half_fov_y: f32,
}

impl FroxelGrid {
    /// Builds a froxel grid, clamping every parameter into a valid range.
    ///
    /// `dims` components are forced to at least `1`, `near` to a small positive
    /// value, `far` to strictly above `near`, and both half-tangents to a small
    /// positive value.  Non-finite inputs collapse to the clamp bounds, so the
    /// returned grid always satisfies the module invariants.
    #[inline]
    pub fn new(
        dims: UVec3,
        near: f32,
        far: f32,
        tan_half_fov_x: f32,
        tan_half_fov_y: f32,
    ) -> Self {
        let dims = UVec3::new(dims.x.max(1), dims.y.max(1), dims.z.max(1));
        let near = if near.is_finite() { near.max(EPS) } else { EPS };
        let far = if far.is_finite() {
            far.max(near * (1.0 + EPS))
        } else {
            near * 2.0
        };
        let tan_half_fov_x = if tan_half_fov_x.is_finite() {
            tan_half_fov_x.max(EPS)
        } else {
            EPS
        };
        let tan_half_fov_y = if tan_half_fov_y.is_finite() {
            tan_half_fov_y.max(EPS)
        } else {
            EPS
        };
        Self {
            dims,
            near,
            far,
            tan_half_fov_x,
            tan_half_fov_y,
        }
    }

    /// Number of depth slices `N` as a float (`dims.z`).
    #[inline]
    pub fn depth_slices(&self) -> f32 {
        self.dims.z as f32
    }

    /// View-space depth of the exponential slice boundary at fractional slice
    /// `s`: `near * (far / near)^(s / N)`.
    ///
    /// `s` is clamped to `[0, N]`, so the result is always within
    /// `[near, far]`.  `s = 0` returns [`near`](Self::near) and `s = N` returns
    /// [`far`](Self::far).
    #[inline]
    pub fn slice_depth(&self, s: f32) -> f32 {
        let n = self.depth_slices();
        let s = if s.is_finite() { s.clamp(0.0, n) } else { 0.0 };
        let ratio = self.far / self.near;
        // ratio > 1 by construction, so ln(ratio) > 0 and the exp is finite.
        let depth = self.near * ops::exp(ops::ln(ratio) * (s / n));
        depth.clamp(self.near, self.far)
    }

    /// Inverse of [`slice_depth`](Self::slice_depth): the fractional slice index
    /// of a view-space depth `z`, `N * ln(z / near) / ln(far / near)`.
    ///
    /// `z` is clamped to `[near, far]`, so the result is always within
    /// `[0, N]`.
    #[inline]
    pub fn depth_to_slice(&self, z: f32) -> f32 {
        let n = self.depth_slices();
        let z = if z.is_finite() {
            z.clamp(self.near, self.far)
        } else {
            self.near
        };
        let ratio = self.far / self.near;
        let s = n * ops::ln(z / self.near) / ops::ln(ratio);
        s.clamp(0.0, n)
    }

    /// Half the view-space lateral extent of the frustum at view-space depth
    /// `z`, returned as `(half_width, half_height)`.
    #[inline]
    fn half_extent(&self, z: f32) -> (f32, f32) {
        let z = z.max(0.0);
        (z * self.tan_half_fov_x, z * self.tan_half_fov_y)
    }

    /// Maps a *fractional* froxel coordinate to the view-space position of that
    /// point in the frustum.
    ///
    /// `f.z` selects the depth through the exponential slicing (so `f.z + 0.5`
    /// is the geometric slice centre), and `f.x` / `f.y` map linearly to the
    /// `[-1, 1]` normalised device range scaled by the frustum half-extent at
    /// that depth.  Inputs are clamped into range so the output is always
    /// finite and inside the frustum.
    #[inline]
    pub fn froxel_to_view(&self, f: Vec3) -> Vec3 {
        let fx = clamp_axis(f.x, self.dims.x);
        let fy = clamp_axis(f.y, self.dims.y);
        let fz = if f.z.is_finite() {
            f.z.clamp(0.0, self.depth_slices())
        } else {
            0.0
        };
        let z = self.slice_depth(fz);
        let (half_w, half_h) = self.half_extent(z);
        let ndc_x = fx / (self.dims.x as f32) * 2.0 - 1.0;
        let ndc_y = fy / (self.dims.y as f32) * 2.0 - 1.0;
        Vec3::new(ndc_x * half_w, ndc_y * half_h, z)
    }

    /// Maps a view-space position to *fractional* froxel coordinates, inverse
    /// of [`froxel_to_view`](Self::froxel_to_view).
    ///
    /// The depth component uses the exponential slicing; the lateral components
    /// undo the perspective scaling at that depth.  All three outputs are
    /// clamped to `[0, dims]` so the point is projected onto the nearest valid
    /// froxel location even when it lies outside the frustum.
    #[inline]
    pub fn view_to_froxel(&self, p_view: Vec3) -> Vec3 {
        let z = if p_view.z.is_finite() {
            p_view.z.clamp(self.near, self.far)
        } else {
            self.near
        };
        let fz = self.depth_to_slice(z);
        let (half_w, half_h) = self.half_extent(z);
        let ndc_x = safe_div(p_view.x, half_w).clamp(-1.0, 1.0);
        let ndc_y = safe_div(p_view.y, half_h).clamp(-1.0, 1.0);
        let fx = (ndc_x * 0.5 + 0.5) * self.dims.x as f32;
        let fy = (ndc_y * 0.5 + 0.5) * self.dims.y as f32;
        Vec3::new(
            fx.clamp(0.0, self.dims.x as f32),
            fy.clamp(0.0, self.dims.y as f32),
            fz,
        )
    }

    /// Integer froxel index containing the view-space position `p_view`.
    ///
    /// Each component is floored and clamped to `[0, dims - 1]`, so the index is
    /// always a valid cell address.
    #[inline]
    pub fn froxel_index(&self, p_view: Vec3) -> UVec3 {
        let f = self.view_to_froxel(p_view);
        UVec3::new(
            index_axis(f.x, self.dims.x),
            index_axis(f.y, self.dims.y),
            index_axis(f.z, self.dims.z),
        )
    }

    /// View-space position of the centre of integer froxel `(ix, iy, iz)`.
    ///
    /// Out-of-range indices are clamped to the last valid cell first, so the
    /// result is always a point inside the frustum.
    #[inline]
    pub fn froxel_center_view(&self, index: UVec3) -> Vec3 {
        let ix = index.x.min(self.dims.x - 1) as f32 + 0.5;
        let iy = index.y.min(self.dims.y - 1) as f32 + 0.5;
        let iz = index.z.min(self.dims.z - 1) as f32 + 0.5;
        self.froxel_to_view(Vec3::new(ix, iy, iz))
    }

    /// Maps a world-space position to fractional froxel coordinates using the
    /// supplied `view_from_world` transform.
    ///
    /// The point is first transformed into view space and then run through
    /// [`view_to_froxel`](Self::view_to_froxel), inheriting all of its clamping
    /// guarantees.
    #[inline]
    pub fn world_to_froxel(&self, view_from_world: &Mat4, p_world: Vec3) -> Vec3 {
        let p_view = view_from_world.transform_point3(p_world);
        self.view_to_froxel(p_view)
    }

    /// Maps fractional froxel coordinates to a world-space position using the
    /// supplied `world_from_view` transform (the inverse of `view_from_world`).
    ///
    /// This is the inverse of [`world_to_froxel`](Self::world_to_froxel) up to
    /// clamping and the view↔world transform pair.
    #[inline]
    pub fn froxel_to_world(&self, world_from_view: &Mat4, f: Vec3) -> Vec3 {
        let p_view = self.froxel_to_view(f);
        world_from_view.transform_point3(p_view)
    }
}

/// Clamps a fractional lateral froxel coordinate to `[0, dim]`.
#[inline]
fn clamp_axis(value: f32, dim: u32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, dim as f32)
    } else {
        0.0
    }
}

/// Floors a fractional froxel coordinate to a valid integer cell in `[0, dim)`.
#[inline]
fn index_axis(value: f32, dim: u32) -> u32 {
    let max_index = dim - 1;
    if value.is_finite() {
        let floored = ops::floor(value.clamp(0.0, dim as f32));
        (floored as u32).min(max_index)
    } else {
        0
    }
}

/// Division that returns `0` for a zero / non-finite denominator instead of a
/// `NaN` or infinity.
#[inline]
fn safe_div(num: f32, den: f32) -> f32 {
    if den.abs() > EPS && den.is_finite() {
        let q = num / den;
        if q.is_finite() {
            q
        } else {
            0.0
        }
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> FroxelGrid {
        FroxelGrid::new(UVec3::new(16, 8, 32), 0.1, 100.0, 1.0, 0.5)
    }

    #[test]
    fn new_clamps_degenerate_parameters() {
        let g = FroxelGrid::new(UVec3::ZERO, -1.0, -2.0, 0.0, f32::NAN);
        assert!(g.dims.x >= 1 && g.dims.y >= 1 && g.dims.z >= 1);
        assert!(g.near > 0.0);
        assert!(g.far > g.near);
        assert!(g.tan_half_fov_x > 0.0 && g.tan_half_fov_y > 0.0);
    }

    #[test]
    fn slice_depth_hits_near_and_far_endpoints() {
        let g = grid();
        assert!((g.slice_depth(0.0) - g.near).abs() < 1e-4);
        assert!((g.slice_depth(g.depth_slices()) - g.far).abs() < 1e-3);
    }

    #[test]
    fn slice_depth_is_monotonic_and_exponential() {
        let g = grid();
        let n = g.depth_slices();
        let mut prev = g.slice_depth(0.0);
        for s in 1..=g.dims.z {
            let d = g.slice_depth(s as f32);
            assert!(d > prev, "slice {s} not increasing: {d} <= {prev}");
            prev = d;
        }
        // Equal slice steps multiply depth by a constant ratio (geometric).
        let ratio_a = g.slice_depth(2.0) / g.slice_depth(1.0);
        let ratio_b = g.slice_depth(n - 1.0) / g.slice_depth(n - 2.0);
        assert!((ratio_a - ratio_b).abs() < 1e-3, "a={ratio_a} b={ratio_b}");
    }

    #[test]
    fn slice_depth_and_depth_to_slice_round_trip() {
        let g = grid();
        for s in [0.0f32, 3.5, 10.0, 31.0] {
            let z = g.slice_depth(s);
            let back = g.depth_to_slice(z);
            assert!((back - s).abs() < 1e-3, "s={s} back={back}");
        }
    }

    #[test]
    fn view_froxel_round_trip() {
        let g = grid();
        let f = Vec3::new(5.5, 2.5, 7.5);
        let p = g.froxel_to_view(f);
        let back = g.view_to_froxel(p);
        assert!((back - f).abs().max_element() < 1e-2, "f={f} back={back}");
        // Depth component matches the slice centre's depth.
        assert!((p.z - g.slice_depth(7.5)).abs() < 1e-3);
    }

    #[test]
    fn out_of_range_coordinates_are_clamped() {
        let g = grid();
        // Points behind the near plane and beyond the lateral frustum clamp in.
        let f = g.view_to_froxel(Vec3::new(1.0e6, -1.0e6, -5.0));
        assert!(f.x >= 0.0 && f.x <= g.dims.x as f32);
        assert!(f.y >= 0.0 && f.y <= g.dims.y as f32);
        assert!(f.z >= 0.0 && f.z <= g.dims.z as f32);
        let idx = g.froxel_index(Vec3::new(1.0e6, -1.0e6, 1.0e6));
        assert!(idx.x < g.dims.x && idx.y < g.dims.y && idx.z < g.dims.z);
    }

    #[test]
    fn froxel_center_lies_inside_its_cell() {
        let g = grid();
        let idx = UVec3::new(4, 3, 9);
        let center = g.froxel_center_view(idx);
        assert_eq!(g.froxel_index(center), idx);
    }

    #[test]
    fn world_round_trip_with_identity_view() {
        let g = grid();
        let eye = Mat4::IDENTITY;
        let f = Vec3::new(6.5, 4.5, 12.5);
        let world = g.froxel_to_world(&eye, f);
        let back = g.world_to_froxel(&eye, world);
        assert!((back - f).abs().max_element() < 1e-2, "f={f} back={back}");
    }

    #[test]
    fn world_round_trip_with_translated_view() {
        let g = grid();
        // Camera translated: view_from_world moves world by -t, inverse by +t.
        let t = Vec3::new(3.0, -2.0, 5.0);
        let view_from_world = Mat4::from_translation(-t);
        let world_from_view = Mat4::from_translation(t);
        let f = Vec3::new(8.5, 1.5, 20.5);
        let world = g.froxel_to_world(&world_from_view, f);
        let back = g.world_to_froxel(&view_from_world, world);
        assert!((back - f).abs().max_element() < 1e-2, "f={f} back={back}");
    }

    #[test]
    fn results_are_deterministic() {
        let g = grid();
        let p = Vec3::new(0.3, -0.2, 12.0);
        assert_eq!(g.view_to_froxel(p), g.view_to_froxel(p));
        assert_eq!(g.froxel_index(p), g.froxel_index(p));
    }

    #[test]
    fn no_nan_on_degenerate_inputs() {
        let g = grid();
        let p = g.view_to_froxel(Vec3::new(f32::NAN, f32::INFINITY, f32::NAN));
        assert!(p.is_finite());
        let v = g.froxel_to_view(Vec3::new(f32::NAN, f32::NAN, f32::NAN));
        assert!(v.is_finite());
    }
}
