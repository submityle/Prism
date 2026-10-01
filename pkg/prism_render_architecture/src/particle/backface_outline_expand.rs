//! Inverted-hull backface outline expansion for mesh particles (design §18).
//!
//! Design §18 ("`NPR` 粒子与风格化着色") lists two outline techniques for mesh
//! particles: *背面外扩描边* (backface expand) and *法线-深度边缘检测*
//! (normal/depth edge detection). This module owns the first one — the classic
//! inverted-hull stroke used by `Guilty Gear Xrd`, `Genshin Impact`, and most
//! toon-shaded `VFX` — and is deliberately **distinct** from the screen-space
//! Sobel/Roberts edge pass in [`super::edge_detect`], which implements the
//! second technique. Nothing here samples a framebuffer; this is a pure
//! object-space geometry contract.
//!
//! # The technique
//!
//! An outline shell is produced by displacing every vertex of the mesh outward
//! along its (unit) shading normal by a distance `d`:
//!
//! ```text
//! p' = p + normalize(n) * d
//! ```
//!
//! The shell is then drawn with **front-face culling** so only its back faces
//! survive. Those back faces peek out from behind the original mesh exactly
//! around the silhouette, forming a solid stroke of thickness `d`. The original
//! mesh is drawn afterward with ordinary back-face culling and covers the shell
//! everywhere except the outline band. This module supplies both halves of the
//! contract: the vertex displacement maths ([`displace_along_normal`],
//! [`OutlineExpandParams`]) and the canonical shell render state
//! ([`OutlineRenderState::inverted_hull`]).
//!
//! # Expansion distance: world vs screen space
//!
//! The outward distance `d` is resolved in one of two spaces (design §18
//! "轮廓干净…分辨率无关"):
//!
//! * [`ExpandSpace::World`] — `d` is a fixed object/world-space thickness,
//!   independent of camera distance. The stroke shrinks on screen as the
//!   particle recedes.
//! * [`ExpandSpace::Screen`] — `d` is chosen so the stroke subtends a constant
//!   number of pixels regardless of depth. A world segment of length `L` at
//!   view-space depth `z` projects to roughly `L * focal / z` pixels, where
//!   `focal` is the vertical focal length in pixels
//!   (`viewport_half_height / tan(fov/2)`). Inverting that for a target pixel
//!   width `w` gives `d = w * z / focal`. The caller supplies `focal` (it owns
//!   the projection; this module never evaluates `tan`), so the only operation
//!   here is a guarded division — no transcendental is used.
//!
//! # Determinism
//!
//! The only transcendental permitted by the workspace contract is `sqrt`, used
//! transitively through [`Vec3::normalize_or_zero`] / [`Vec3::length`]. Every
//! other step is `+ - * /` guarded against a zero denominator, so a future
//! `GPU` draw kernel reproduces this `CPU` reference bit for bit. Degenerate
//! (zero-length) normals normalize to the zero vector and therefore leave the
//! vertex untouched instead of emitting a `NaN`. `GPU` packing follows the
//! shared `std430` `vec4` alignment from [`super::gpu_layout`].

extern crate alloc;

use alloc::vec::Vec;

use super::gpu_layout::{storage_bytes, VEC4_STRIDE};
use super::mesh_emission::MeshVertex;
use super::Vec3;

/// Absolute tolerance for the `f32` comparisons used by the tests; direct `==`
/// / `!=` on floating point is intentionally avoided across the workspace.
#[cfg(test)]
const CMP_EPS: f32 = 1e-6;

/// Minimum focal length (in pixels) used to guard the screen-space division.
///
/// A focal length at or below this magnitude is treated as degenerate and the
/// resolved distance collapses to zero rather than dividing by (near) zero and
/// producing a `NaN` or a runaway stroke.
const MIN_FOCAL: f32 = 1e-6;

/// Byte stride of one [`OutlineExpandParams`] record in a `std430` storage
/// buffer.
///
/// The four scalars pack into a single `vec4` slot
/// `vec4(thickness, focal_scale, max_distance, space_tag)`.
pub const BACKFACE_OUTLINE_PARAMS_STRIDE: usize = VEC4_STRIDE;

/// `std430` tag written for [`ExpandSpace::World`].
const SPACE_TAG_WORLD: u32 = 0;

/// `std430` tag written for [`ExpandSpace::Screen`].
const SPACE_TAG_SCREEN: u32 = 1;

/// Which triangle faces a raster pass discards.
///
/// The inverted-hull outline shell is drawn with [`CullMode::Front`] so only
/// its back faces survive and form the stroke around the silhouette.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CullMode {
    /// Draw both faces (no culling).
    None,
    /// Discard front faces, keep back faces — the outline-shell setting.
    Front,
    /// Discard back faces, keep front faces — the ordinary opaque setting.
    Back,
}

impl CullMode {
    /// `std430` discriminant used when packing a render state for a `GPU`
    /// kernel: `0` = [`CullMode::None`], `1` = [`CullMode::Front`],
    /// `2` = [`CullMode::Back`].
    #[must_use]
    pub const fn tag(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Front => 1,
            Self::Back => 2,
        }
    }
}

/// Triangle winding convention that counts as the *front* face.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FrontFace {
    /// Counter-clockwise winding is the front face (the common default).
    Ccw,
    /// Clockwise winding is the front face.
    Cw,
}

/// Raster state for one draw of the outline shell.
///
/// This is the geometry-side policy the inverted-hull technique requires; it
/// says nothing about the stroke color, which the shading model owns.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct OutlineRenderState {
    /// Which faces are discarded. The shell uses [`CullMode::Front`].
    pub cull_mode: CullMode,
    /// Which winding is treated as the front face. Expansion never reorders
    /// vertices, so the mesh's authored winding is preserved.
    pub front_face: FrontFace,
    /// Whether the shell writes depth. The stroke writes depth so it occludes
    /// correctly against other opaque geometry.
    pub depth_write: bool,
    /// Whether the shell depth-tests with a `less` comparison against the depth
    /// buffer (standard opaque ordering).
    pub depth_test_less: bool,
}

impl OutlineRenderState {
    /// The canonical inverted-hull outline-shell state: cull front faces so the
    /// back faces form the stroke, keep the authored (`CCW`) winding, and
    /// participate in the opaque depth buffer.
    #[must_use]
    pub const fn inverted_hull() -> Self {
        Self {
            cull_mode: CullMode::Front,
            front_face: FrontFace::Ccw,
            depth_write: true,
            depth_test_less: true,
        }
    }

    /// Whether this state renders back faces (i.e. culls the front faces),
    /// which is the defining property of the outline shell pass.
    #[must_use]
    pub fn renders_back_faces(self) -> bool {
        matches!(self.cull_mode, CullMode::Front)
    }
}

/// The space in which an outline thickness is interpreted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExpandSpace {
    /// `thickness` is a fixed world/object-space distance, independent of the
    /// camera.
    World,
    /// `thickness` is a pixel width; the world distance is scaled by the
    /// view-space depth and the supplied vertical focal length (in pixels) so
    /// the stroke stays a constant number of pixels wide at any depth.
    Screen {
        /// Vertical focal length in pixels (`viewport_half_height / tan(fov/2)`),
        /// supplied by the caller that owns the projection.
        focal_scale: f32,
    },
}

/// World-space outward distance for a fixed [`ExpandSpace::World`] thickness.
///
/// This is the identity map — world thickness is depth-independent — exposed as
/// a named function so both spaces share one call shape.
#[must_use]
pub fn expand_distance_world(thickness: f32) -> f32 {
    thickness
}

/// World-space outward distance that keeps a `pixel_width`-wide stroke constant
/// on screen at `view_depth` (view-space `+z` distance from the camera).
///
/// `d = pixel_width * view_depth / focal_scale`. A degenerate `focal_scale`
/// (at or below [`MIN_FOCAL`] in magnitude) yields `0.0` rather than dividing by
/// (near) zero.
#[must_use]
pub fn expand_distance_screen(pixel_width: f32, view_depth: f32, focal_scale: f32) -> f32 {
    if focal_scale.abs() <= MIN_FOCAL {
        return 0.0;
    }
    pixel_width * view_depth / focal_scale
}

/// Displaces `position` outward along `normal` by `distance`.
///
/// The normal is normalized first (it need not be unit length). A degenerate
/// (zero-length) normal normalizes to [`Vec3::ZERO`], so the vertex is returned
/// unchanged — the outline simply contributes nothing there instead of emitting
/// a `NaN`.
#[must_use]
pub fn displace_along_normal(position: Vec3, normal: Vec3, distance: f32) -> Vec3 {
    let unit = normal.normalize_or_zero();
    position.add(unit.scale(distance))
}

/// Displaces `position` outward along `normal`, falling back to `fallback`
/// (typically the triangle's geometric normal) when `normal` is degenerate.
///
/// This mirrors the authoring rule in [`super::mesh_emission`]: a mesh with no
/// authored shading normal still gets a well-defined outward direction from its
/// geometric normal. If both are degenerate the vertex is left unchanged.
#[must_use]
pub fn displace_with_fallback(
    position: Vec3,
    normal: Vec3,
    fallback: Vec3,
    distance: f32,
) -> Vec3 {
    let unit = normal.normalize_or_zero();
    let dir = if unit.length_squared() > 0.0 {
        unit
    } else {
        fallback.normalize_or_zero()
    };
    position.add(dir.scale(distance))
}

/// The signed outward offset actually applied to a vertex: the projection of
/// `displaced - original` onto the unit `normal`.
///
/// For a well-formed expansion this equals the requested distance; it is
/// exposed so a caller (or a test) can confirm the shell grew outward and did
/// not fold inward.
#[must_use]
pub fn outward_offset(original: Vec3, displaced: Vec3, normal: Vec3) -> f32 {
    displaced.sub(original).dot(normal.normalize_or_zero())
}

/// Policy describing how thick the outline stroke is and in which space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutlineExpandParams {
    /// Stroke thickness. Interpreted as world units for [`ExpandSpace::World`]
    /// and as pixels for [`ExpandSpace::Screen`]. Negative values are clamped
    /// to `0.0` (an inward "stroke" would invert the hull and is disallowed).
    pub thickness: f32,
    /// The space in which `thickness` is interpreted.
    pub space: ExpandSpace,
    /// Hard upper bound on the resolved world distance, so a near-camera
    /// screen-space stroke cannot balloon without limit. A non-positive value
    /// means "unbounded".
    pub max_distance: f32,
}

impl OutlineExpandParams {
    /// Builds a world-space outline of the given thickness with no distance
    /// clamp.
    #[must_use]
    pub const fn world(thickness: f32) -> Self {
        Self {
            thickness,
            space: ExpandSpace::World,
            max_distance: 0.0,
        }
    }

    /// Builds a screen-space outline `pixel_width` pixels wide for a camera
    /// whose vertical focal length is `focal_scale` pixels, with no distance
    /// clamp.
    #[must_use]
    pub const fn screen(pixel_width: f32, focal_scale: f32) -> Self {
        Self {
            thickness: pixel_width,
            space: ExpandSpace::Screen { focal_scale },
            max_distance: 0.0,
        }
    }

    /// Returns a copy with the world-distance clamp set to `max_distance`.
    #[must_use]
    pub const fn with_max_distance(mut self, max_distance: f32) -> Self {
        self.max_distance = max_distance;
        self
    }

    /// Resolves the outward world distance `d` at the given view-space depth.
    ///
    /// The thickness is floored at `0.0` (no inward stroke), resolved per the
    /// active space, and finally clamped to [`OutlineExpandParams::max_distance`]
    /// when that bound is positive.
    #[must_use]
    pub fn expand_distance(&self, view_depth: f32) -> f32 {
        let thickness = self.thickness.max(0.0);
        let raw = match self.space {
            ExpandSpace::World => expand_distance_world(thickness),
            ExpandSpace::Screen { focal_scale } => {
                expand_distance_screen(thickness, view_depth, focal_scale)
            }
        };
        if self.max_distance > 0.0 {
            raw.min(self.max_distance)
        } else {
            raw
        }
    }

    /// Displaces one vertex position for this policy at the given view depth.
    #[must_use]
    pub fn displace_vertex(&self, position: Vec3, normal: Vec3, view_depth: f32) -> Vec3 {
        displace_along_normal(position, normal, self.expand_distance(view_depth))
    }

    /// Packs the policy into a `std430` `vec4` word
    /// `vec4(thickness, focal_scale, max_distance, space_tag)`.
    ///
    /// The `space_tag` is a `u32` bit pattern (`0` world, `1` screen); world
    /// space stores a zero `focal_scale`.
    #[must_use]
    pub fn to_std430(&self) -> [u32; 4] {
        let (focal, tag) = match self.space {
            ExpandSpace::World => (0.0_f32, SPACE_TAG_WORLD),
            ExpandSpace::Screen { focal_scale } => (focal_scale, SPACE_TAG_SCREEN),
        };
        [
            self.thickness.to_bits(),
            focal.to_bits(),
            self.max_distance.to_bits(),
            tag,
        ]
    }
}

/// Expands a whole vertex list by a precomputed world `distance`, returning the
/// outline-shell vertices.
///
/// Each output vertex keeps its original normal and `UV`; only the position
/// moves outward along the normal. Topology (the index buffer) is unchanged, so
/// the caller reuses the mesh's existing indices for the shell draw.
#[must_use]
pub fn expand_shell(vertices: &[MeshVertex], distance: f32) -> Vec<MeshVertex> {
    let mut out = Vec::with_capacity(vertices.len());
    for v in vertices {
        let moved = displace_along_normal(v.position, v.normal, distance);
        out.push(MeshVertex::new(moved, v.normal, v.uv));
    }
    out
}

/// Expands a whole vertex list for a screen-space policy at a single
/// object-level `view_depth` (the common per-instance approximation).
///
/// Using one depth per object keeps the stroke uniform across the silhouette;
/// per-vertex depth variation within a particle mesh is negligible for an
/// outline and would otherwise make the stroke waver.
#[must_use]
pub fn expand_shell_screen(
    vertices: &[MeshVertex],
    params: &OutlineExpandParams,
    view_depth: f32,
) -> Vec<MeshVertex> {
    expand_shell(vertices, params.expand_distance(view_depth))
}

/// The `std430` byte size of a storage buffer holding `count` records of
/// [`OutlineExpandParams`], clamped up to a single element (see
/// [`super::gpu_layout::storage_bytes`]).
#[must_use]
pub fn backface_outline_buffer_bytes(count: usize) -> usize {
    storage_bytes(BACKFACE_OUTLINE_PARAMS_STRIDE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute-tolerance comparison, avoiding a forbidden `f32` `==`.
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= CMP_EPS
    }

    /// Vector equality within tolerance.
    fn vec_approx(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    /// The six axis vertices of a unit octahedron, each with an outward
    /// (radial) unit normal — a convex test shape whose radial expansion has a
    /// closed-form answer.
    fn unit_octahedron() -> [MeshVertex; 6] {
        [
            MeshVertex::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), [0.0, 0.0]),
            MeshVertex::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), [0.0, 0.0]),
            MeshVertex::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), [0.0, 0.0]),
            MeshVertex::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, -1.0, 0.0), [0.0, 0.0]),
            MeshVertex::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0), [0.0, 0.0]),
            MeshVertex::new(Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, -1.0), [0.0, 0.0]),
        ]
    }

    #[test]
    fn displace_matches_analytic_solution() {
        // Non-unit normal (0,0,2) normalizes to +z; distance 0.5 -> +0.5 z.
        let p = Vec3::new(1.0, 2.0, 3.0);
        let n = Vec3::new(0.0, 0.0, 2.0);
        let got = displace_along_normal(p, n, 0.5);
        assert!(vec_approx(got, Vec3::new(1.0, 2.0, 3.5)));
    }

    #[test]
    fn nonunit_normal_is_normalized_before_displacement() {
        // Normal length 2 along x; distance 3 must move exactly 3, not 6.
        let p = Vec3::ZERO;
        let n = Vec3::new(2.0, 0.0, 0.0);
        let got = displace_along_normal(p, n, 3.0);
        assert!(vec_approx(got, Vec3::new(3.0, 0.0, 0.0)));
        assert!(approx(outward_offset(p, got, n), 3.0));
    }

    #[test]
    fn degenerate_zero_normal_leaves_vertex_unchanged() {
        let p = Vec3::new(5.0, -4.0, 2.0);
        let got = displace_along_normal(p, Vec3::ZERO, 10.0);
        assert!(vec_approx(got, p));
        assert!(got.x.is_finite() && got.y.is_finite() && got.z.is_finite());
    }

    #[test]
    fn fallback_normal_used_when_shading_normal_degenerate() {
        let p = Vec3::ZERO;
        let fallback = Vec3::new(0.0, 4.0, 0.0); // non-unit geometric normal
        let got = displace_with_fallback(p, Vec3::ZERO, fallback, 2.0);
        assert!(vec_approx(got, Vec3::new(0.0, 2.0, 0.0)));
    }

    #[test]
    fn screen_distance_is_proportional_to_depth() {
        // Constant-pixel semantics: doubling depth doubles the world distance.
        let near = expand_distance_screen(4.0, 10.0, 800.0);
        let far = expand_distance_screen(4.0, 20.0, 800.0);
        assert!(approx(near, 4.0 * 10.0 / 800.0));
        assert!(approx(far, 2.0 * near));
    }

    #[test]
    fn world_distance_is_depth_independent() {
        let params = OutlineExpandParams::world(0.25);
        assert!(approx(params.expand_distance(1.0), 0.25));
        assert!(approx(params.expand_distance(1000.0), 0.25));
    }

    #[test]
    fn screen_distance_guards_degenerate_focal() {
        assert!(approx(expand_distance_screen(4.0, 10.0, 0.0), 0.0));
    }

    #[test]
    fn negative_thickness_is_clamped_to_zero() {
        let params = OutlineExpandParams::world(-1.0);
        assert!(approx(params.expand_distance(1.0), 0.0));
    }

    #[test]
    fn distance_is_clamped_to_max() {
        let params = OutlineExpandParams::screen(10.0, 100.0).with_max_distance(0.5);
        // Raw = 10 * 20 / 100 = 2.0, clamped to 0.5.
        assert!(approx(params.expand_distance(20.0), 0.5));
        // Raw = 10 * 2 / 100 = 0.2, below the clamp and untouched.
        assert!(approx(params.expand_distance(2.0), 0.2));
    }

    #[test]
    fn convex_shell_contains_original_mesh() {
        // Radial expansion of a unit octahedron by d yields vertices at
        // (1 + d) on the axes. The expanded L1 ball of radius (1 + d) strictly
        // contains the original unit L1 ball, so every original vertex is
        // inside the shell and the hull did not fold inward.
        let d = 0.3_f32;
        let shell = expand_shell(&unit_octahedron(), d);
        let expanded_radius = 1.0 + d;
        for (orig, moved) in unit_octahedron().iter().zip(shell.iter()) {
            // Each shell vertex sits exactly d further out along its normal.
            assert!(approx(outward_offset(orig.position, moved.position, orig.normal), d));
            // Shell vertex is strictly outside the original surface (support
            // value grew), proving outward growth / containment.
            let orig_support = orig.position.length();
            let shell_support = moved.position.length();
            assert!(shell_support > orig_support);
            assert!(approx(shell_support, expanded_radius));
        }
    }

    #[test]
    fn shell_face_normal_does_not_flip() {
        // Build one octahedron face (+x, +y, +z corners) and verify its
        // geometric normal keeps the same orientation after expansion: a
        // flipped (self-intersecting) hull would reverse the normal sign.
        let verts = unit_octahedron();
        let shell = expand_shell(&verts, 0.5);
        let face_normal = |a: Vec3, b: Vec3, c: Vec3| b.sub(a).cross(c.sub(a));
        let before = face_normal(verts[0].position, verts[2].position, verts[4].position);
        let after = face_normal(shell[0].position, shell[2].position, shell[4].position);
        assert!(before.dot(after) > 0.0);
    }

    #[test]
    fn shell_preserves_normals_and_uv_and_topology() {
        let verts = unit_octahedron();
        let shell = expand_shell(&verts, 0.1);
        assert_eq!(shell.len(), verts.len());
        for (orig, moved) in verts.iter().zip(shell.iter()) {
            assert!(vec_approx(orig.normal, moved.normal));
            assert!(approx(orig.uv[0], moved.uv[0]));
            assert!(approx(orig.uv[1], moved.uv[1]));
        }
    }

    #[test]
    fn inverted_hull_render_state_culls_front_faces() {
        let state = OutlineRenderState::inverted_hull();
        assert_eq!(state.cull_mode, CullMode::Front);
        assert!(state.renders_back_faces());
        assert_eq!(state.front_face, FrontFace::Ccw);
        assert!(state.depth_write);
        assert!(state.depth_test_less);
        assert_eq!(state.cull_mode.tag(), 1);
    }

    #[test]
    fn expansion_is_bitwise_deterministic() {
        let verts = unit_octahedron();
        let params = OutlineExpandParams::screen(3.5, 640.0).with_max_distance(1.0);
        let depth = 12.5_f32;
        let a = expand_shell_screen(&verts, &params, depth);
        let b = expand_shell_screen(&verts, &params, depth);
        assert_eq!(a.len(), b.len());
        for (va, vb) in a.iter().zip(b.iter()) {
            assert_eq!(va.position.x.to_bits(), vb.position.x.to_bits());
            assert_eq!(va.position.y.to_bits(), vb.position.y.to_bits());
            assert_eq!(va.position.z.to_bits(), vb.position.z.to_bits());
        }
    }

    #[test]
    fn std430_packing_tags_space_and_bits() {
        let world = OutlineExpandParams::world(0.2).to_std430();
        assert_eq!(world[0], 0.2_f32.to_bits());
        assert_eq!(world[1], 0.0_f32.to_bits());
        assert_eq!(world[3], SPACE_TAG_WORLD);

        let screen = OutlineExpandParams::screen(4.0, 800.0)
            .with_max_distance(0.75)
            .to_std430();
        assert_eq!(screen[0], 4.0_f32.to_bits());
        assert_eq!(screen[1], 800.0_f32.to_bits());
        assert_eq!(screen[2], 0.75_f32.to_bits());
        assert_eq!(screen[3], SPACE_TAG_SCREEN);
    }

    #[test]
    fn buffer_bytes_follow_std430_and_clamp_empty() {
        assert_eq!(backface_outline_buffer_bytes(4), 4 * VEC4_STRIDE);
        assert_eq!(backface_outline_buffer_bytes(0), VEC4_STRIDE);
    }
}
