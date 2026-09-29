//! Camera query `DataInterface` (`DI`) for the particle module (design §8.3,
//! "Scene" category), the `CPU` reference matching Unreal `Niagara`'s "Camera
//! Query" and Unity `VFX Graph`'s camera sampling nodes.
//!
//! This module is deliberately *orthogonal* to two neighbours:
//!
//! * [`super::renderers`] owns the **render-orientation** camera: its
//!   [`CameraFrame`] (particle-to-camera direction plus world up) and
//!   `billboard_basis` build the quad basis a sprite *faces* the camera with.
//!   That is an output-geometry concern. This module never rebuilds that basis;
//!   it *reuses* [`CameraFrame`] through [`CameraProjection::billboard_frame_for`]
//!   so a query-side camera can hand a render-side billboard its facing frame.
//! * [`super::lod`] is a *consumer* of screen coverage: it maps a screen
//!   fraction to a particle `LOD` tier. This module *produces* the coverage
//!   inputs (approximate `NDC`/pixel radius of a bounding sphere) that a `LOD`
//!   or size module reads; it does not classify tiers itself.
//!
//! The role here is the **input-query** side: Set/Force modules ask the camera
//! for projection-relative quantities — world->view->clip->`NDC`->screen
//! transforms, camera-relative distance/direction, linearized depth, and
//! screen-space size of a world radius — to drive effects such as
//! distance-scaled size, camera-facing offsets, or screen-fraction `LOD`
//! selection.
//!
//! Determinism contract, matching the rest of the crate: pure `f32` algebra,
//! **no transcendental functions**. Angle-derived quantities (the `FOV`) are
//! received already reduced to `tan(fov/2)` numbers supplied by the caller, so
//! this reference never calls a trig function. Only `sqrt` (through [`Vec3`])
//! and `f32::abs`/`floor`/`ceil` are used, keeping the `CPU` path bit-
//! reproducible against a future `GPU` kernel. Every perspective divide and
//! reciprocal is guarded by [`EPS`] so a `w` at or behind the camera plane, a
//! collapsed near/far range, or a zeroed `FOV` degrades to a safe fallback
//! rather than producing `NaN` or panicking.

use super::renderers::CameraFrame;
use super::Vec3;

/// Absolute tolerance for the guarded reciprocals and clip-space tests in this
/// module. A denominator whose magnitude is at or below this value (a `w` at
/// the camera plane, a zero near/far range, a zeroed `FOV` tangent) is treated
/// as degenerate and routed to a safe fallback.
pub const EPS: f32 = 1e-6;

/// Guarded scalar reciprocal: returns `1.0 / value` when `|value|` clears
/// [`EPS`], and `0.0` otherwise so a degenerate divisor collapses the mapped
/// coordinate to the origin instead of yielding `NaN` or infinity.
#[must_use]
fn safe_recip(value: f32) -> f32 {
    if value.abs() > EPS {
        1.0 / value
    } else {
        0.0
    }
}

/// Returns any unit vector perpendicular to `v` (deterministic), or `+X` when
/// `v` is (numerically) zero. Used only as a basis-repair fallback so a
/// collapsed cross product never leaves a non-orthonormal frame.
#[must_use]
fn any_perpendicular(v: Vec3) -> Vec3 {
    let ax = v.x * v.x;
    let ay = v.y * v.y;
    let az = v.z * v.z;
    let reference = if ax <= ay && ax <= az {
        Vec3::new(1.0, 0.0, 0.0)
    } else if ay <= az {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let perp = v.cross(reference).normalize_or_zero();
    if perp.length_squared() > EPS {
        perp
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    }
}

/// The pixel dimensions of the render target a screen mapping targets.
///
/// Supplied by the caller (the query `DI` does not own the swapchain), so a
/// module can convert an `NDC` position into pixel coordinates for pixel-space
/// effects and pixel-fraction `LOD`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// Target width in pixels.
    pub width: f32,
    /// Target height in pixels.
    pub height: f32,
}

impl Viewport {
    /// Builds a viewport from pixel dimensions.
    #[must_use]
    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }
}

/// The camera's world-space orthonormal basis, supplied by the caller.
///
/// The three axes are the *inputs* to every world->view transform: `right`
/// (view `+X`), `up` (view `+Y`), and `forward` (view `+Z`, the look direction
/// pointing into the scene). A point in front of the camera therefore has a
/// positive `forward`-projected (view-space `z`) coordinate. No trigonometry is
/// involved: the caller hands the already-computed basis, mirroring how the
/// design routes angle-derived quantities in as numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraBasis {
    /// View-space `+X` (screen right).
    pub right: Vec3,
    /// View-space `+Y` (screen up).
    pub up: Vec3,
    /// View-space `+Z` (the look direction, into the scene).
    pub forward: Vec3,
}

impl CameraBasis {
    /// Builds a basis directly from three axes the caller already computed.
    ///
    /// The axes are assumed orthonormal; use [`CameraBasis::from_forward_up`]
    /// to re-orthonormalize an approximate look/up pair.
    #[must_use]
    pub const fn new(right: Vec3, up: Vec3, forward: Vec3) -> Self {
        Self { right, up, forward }
    }

    /// Builds an orthonormal basis from a look direction and an approximate up,
    /// using only cross products and `normalize_or_zero` (so only `sqrt`, no
    /// transcendentals).
    ///
    /// Degenerate inputs are repaired deterministically: a zero `forward` falls
    /// back to `+Z`, a zero or parallel `up` falls back to a computed
    /// perpendicular, so the result is always orthonormal and never `NaN`.
    #[must_use]
    pub fn from_forward_up(forward: Vec3, up: Vec3) -> Self {
        let mut f = forward.normalize_or_zero();
        if f.length_squared() <= EPS {
            f = Vec3::new(0.0, 0.0, 1.0);
        }
        let mut right = up.cross(f).normalize_or_zero();
        if right.length_squared() <= EPS {
            right = any_perpendicular(f);
        }
        let up = f.cross(right);
        Self {
            right,
            up,
            forward: f,
        }
    }
}

/// Which projection the camera applies at the view->clip stage, carrying the
/// caller-provided (already trig-free) shape parameters.
///
/// Holding `f32` payloads, this only derives [`PartialEq`]; it is compared with
/// [`EPS`] tolerances in tests rather than for exact structural equality.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProjectionKind {
    /// A perspective frustum described by the half-`FOV` tangents. `NDC` `z`
    /// runs `0` at the near plane to `1` at the far plane.
    Perspective {
        /// `tan(horizontal_fov / 2)`, computed by the caller.
        tan_half_fov_x: f32,
        /// `tan(vertical_fov / 2)`, computed by the caller.
        tan_half_fov_y: f32,
    },
    /// An orthographic box described by its half-extents at the view plane.
    Orthographic {
        /// Half the view-plane width in world units.
        half_width: f32,
        /// Half the view-plane height in world units.
        half_height: f32,
    },
}

/// The result of testing a world point against the frustum, used by modules
/// that only need a coarse visibility class rather than a screen position.
///
/// A field-less enum, so it derives [`Eq`]/[`Hash`] for use as a map key or
/// match discriminant.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ClipVisibility {
    /// Inside the frustum (and, for perspective, in front of the camera).
    Visible,
    /// Behind the camera plane (perspective only); never projected.
    BehindCamera,
    /// In front of the camera but outside the clip cube on some axis.
    OutsideFrustum,
    /// The perspective divide collapsed (a `w` at the camera plane).
    Degenerate,
}

/// A camera query `DataInterface`: world position, orthonormal basis, clip
/// range, projection shape, and target viewport, with the transforms and
/// camera-relative queries a Set/Force module reads (design §8.3).
///
/// All angle-derived data arrives pre-reduced (see [`ProjectionKind`]); this
/// type performs only algebra plus `sqrt`. Every method is degenerate-safe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraProjection {
    /// Camera position in world space.
    pub world_position: Vec3,
    /// Camera orthonormal basis (view axes).
    pub basis: CameraBasis,
    /// Near clip distance along `forward` (world units, positive).
    pub near: f32,
    /// Far clip distance along `forward` (world units, positive, `> near`).
    pub far: f32,
    /// Projection shape and its trig-free parameters.
    pub projection: ProjectionKind,
    /// Target viewport for `NDC`->screen mapping.
    pub viewport: Viewport,
}

impl CameraProjection {
    /// Builds a perspective camera from half-`FOV` tangents supplied by the
    /// caller (no trig performed here).
    #[must_use]
    pub const fn perspective(
        world_position: Vec3,
        basis: CameraBasis,
        near: f32,
        far: f32,
        tan_half_fov_x: f32,
        tan_half_fov_y: f32,
        viewport: Viewport,
    ) -> Self {
        Self {
            world_position,
            basis,
            near,
            far,
            projection: ProjectionKind::Perspective {
                tan_half_fov_x,
                tan_half_fov_y,
            },
            viewport,
        }
    }

    /// Builds an orthographic camera from the view-plane half-extents.
    #[must_use]
    pub const fn orthographic(
        world_position: Vec3,
        basis: CameraBasis,
        near: f32,
        far: f32,
        half_width: f32,
        half_height: f32,
        viewport: Viewport,
    ) -> Self {
        Self {
            world_position,
            basis,
            near,
            far,
            projection: ProjectionKind::Orthographic {
                half_width,
                half_height,
            },
            viewport,
        }
    }

    /// Whether this camera uses a perspective projection.
    #[must_use]
    pub fn is_perspective(&self) -> bool {
        matches!(self.projection, ProjectionKind::Perspective { .. })
    }

    /// Transforms a world position into view space (camera-relative axes).
    ///
    /// The result's `z` component is the signed depth along `forward`: positive
    /// in front of the camera, negative behind it.
    #[must_use]
    pub fn world_to_view(&self, world_pos: Vec3) -> Vec3 {
        let rel = world_pos.sub(self.world_position);
        Vec3::new(
            rel.dot(self.basis.right),
            rel.dot(self.basis.up),
            rel.dot(self.basis.forward),
        )
    }

    /// Alias for [`CameraProjection::world_to_view`], named for the module
    /// query it backs (a module asking "where is this in view space?").
    #[must_use]
    pub fn view_space_position(&self, world_pos: Vec3) -> Vec3 {
        self.world_to_view(world_pos)
    }

    /// Projects a view-space position into homogeneous clip space
    /// `[x, y, z, w]` (pre perspective divide).
    ///
    /// Perspective clip `w` is the view-space depth; orthographic `w` is `1`.
    /// The near/far range and the `FOV` tangents are guarded by [`EPS`], so a
    /// collapsed range or zeroed tangent yields a zeroed coordinate rather than
    /// `NaN`.
    #[must_use]
    pub fn view_to_clip(&self, view: Vec3) -> [f32; 4] {
        match self.projection {
            ProjectionKind::Perspective {
                tan_half_fov_x,
                tan_half_fov_y,
            } => {
                let inv_tx = safe_recip(tan_half_fov_x);
                let inv_ty = safe_recip(tan_half_fov_y);
                let inv_range = safe_recip(self.far - self.near);
                let cx = view.x * inv_tx;
                let cy = view.y * inv_ty;
                let cz = self.far * (view.z - self.near) * inv_range;
                [cx, cy, cz, view.z]
            }
            ProjectionKind::Orthographic {
                half_width,
                half_height,
            } => {
                let inv_hw = safe_recip(half_width);
                let inv_hh = safe_recip(half_height);
                let inv_range = safe_recip(self.far - self.near);
                let cx = view.x * inv_hw;
                let cy = view.y * inv_hh;
                let cz = (view.z - self.near) * inv_range;
                [cx, cy, cz, 1.0]
            }
        }
    }

    /// Performs the perspective divide, returning `NDC` `[x, y, z]` or [`None`]
    /// when `|w|` is at or below [`EPS`] (a point on the camera plane).
    #[must_use]
    pub fn clip_to_ndc(clip: [f32; 4]) -> Option<Vec3> {
        let w = clip[3];
        if w.abs() <= EPS {
            return None;
        }
        let inv_w = 1.0 / w;
        Some(Vec3::new(clip[0] * inv_w, clip[1] * inv_w, clip[2] * inv_w))
    }

    /// Transforms a world position all the way to `NDC`, returning [`None`]
    /// when the point is behind the camera (perspective) or the perspective
    /// divide is degenerate.
    ///
    /// `NDC` `x`/`y` span `-1..=1` inside the frustum and `z` spans `0..=1`
    /// from the near to the far plane.
    #[must_use]
    pub fn world_to_ndc(&self, world_pos: Vec3) -> Option<Vec3> {
        let view = self.world_to_view(world_pos);
        if self.is_perspective() && view.z <= EPS {
            return None;
        }
        Self::clip_to_ndc(self.view_to_clip(view))
    }

    /// Maps an `NDC` position to pixel coordinates in the target viewport.
    ///
    /// `x` increases rightward from the left edge; `y` increases downward from
    /// the top edge (the usual screen convention), so `NDC` `+Y` (up) maps to a
    /// smaller pixel `y`.
    #[must_use]
    pub fn ndc_to_screen(&self, ndc: Vec3) -> [f32; 2] {
        let sx = (ndc.x * 0.5 + 0.5) * self.viewport.width;
        let sy = (0.5 - ndc.y * 0.5) * self.viewport.height;
        [sx, sy]
    }

    /// Projects a world position to screen pixels, returning [`None`] when the
    /// point is behind the camera, the perspective divide is degenerate, or the
    /// point lies outside the clip cube (frustum-culled).
    #[must_use]
    pub fn world_to_screen(&self, world_pos: Vec3) -> Option<[f32; 2]> {
        let ndc = self.world_to_ndc(world_pos)?;
        if ndc.x < -1.0 - EPS || ndc.x > 1.0 + EPS {
            return None;
        }
        if ndc.y < -1.0 - EPS || ndc.y > 1.0 + EPS {
            return None;
        }
        if ndc.z < -EPS || ndc.z > 1.0 + EPS {
            return None;
        }
        Some(self.ndc_to_screen(ndc))
    }

    /// Classifies a world point against the frustum without producing a screen
    /// position, for modules that only need a coarse visibility bucket.
    #[must_use]
    pub fn classify_visibility(&self, world_pos: Vec3) -> ClipVisibility {
        let view = self.world_to_view(world_pos);
        if self.is_perspective() && view.z <= EPS {
            return ClipVisibility::BehindCamera;
        }
        let Some(ndc) = Self::clip_to_ndc(self.view_to_clip(view)) else {
            return ClipVisibility::Degenerate;
        };
        let outside = ndc.x < -1.0 - EPS
            || ndc.x > 1.0 + EPS
            || ndc.y < -1.0 - EPS
            || ndc.y > 1.0 + EPS
            || ndc.z < -EPS
            || ndc.z > 1.0 + EPS;
        if outside {
            ClipVisibility::OutsideFrustum
        } else {
            ClipVisibility::Visible
        }
    }

    /// Converts an `NDC` depth (`0` at near, `1` at far) back to a positive,
    /// view-space linear distance along `forward`.
    ///
    /// Perspective inverts the projective depth with pure algebra; orthographic
    /// depth is already linear. The perspective denominator is guarded by
    /// [`EPS`], falling back to `far` rather than dividing by (near) zero.
    #[must_use]
    pub fn linearize_depth(&self, ndc_z: f32) -> f32 {
        match self.projection {
            ProjectionKind::Perspective { .. } => {
                let range = self.far - self.near;
                let denom = self.far - range * ndc_z;
                if denom.abs() <= EPS {
                    self.far
                } else {
                    self.far * self.near / denom
                }
            }
            ProjectionKind::Orthographic { .. } => self.near + (self.far - self.near) * ndc_z,
        }
    }

    /// Straight-line distance from a world position to the camera.
    #[must_use]
    pub fn distance_to_camera(&self, world_pos: Vec3) -> f32 {
        world_pos.distance(self.world_position)
    }

    /// Squared distance to the camera, cheaper when only a comparison is
    /// needed (avoids the `sqrt`).
    #[must_use]
    pub fn distance_squared_to_camera(&self, world_pos: Vec3) -> f32 {
        world_pos.distance_squared(self.world_position)
    }

    /// Unit direction from a world position toward the camera, or
    /// [`Vec3::ZERO`] when the point coincides with the camera.
    #[must_use]
    pub fn direction_to_camera(&self, world_pos: Vec3) -> Vec3 {
        self.world_position.sub(world_pos).normalize_or_zero()
    }

    /// Approximate `NDC` half-size (a fraction of the half-height) that a world
    /// bounding sphere of `world_radius` subtends at `world_center`.
    ///
    /// This is the coverage *input* the size/`LOD` modules read. Perspective
    /// coverage shrinks with view depth (`radius / (depth * tan_half_fov_y)`);
    /// orthographic coverage is depth-invariant (`radius / half_height`). A
    /// point at or behind the camera yields `0`. All divides are [`EPS`]-
    /// guarded and the result is clamped non-negative.
    #[must_use]
    pub fn screen_coverage_ndc(&self, world_center: Vec3, world_radius: f32) -> f32 {
        let view = self.world_to_view(world_center);
        match self.projection {
            ProjectionKind::Perspective { tan_half_fov_y, .. } => {
                if view.z <= EPS {
                    return 0.0;
                }
                let inv = safe_recip(view.z * tan_half_fov_y);
                (world_radius * inv).abs()
            }
            ProjectionKind::Orthographic { half_height, .. } => {
                let inv = safe_recip(half_height);
                (world_radius * inv).abs()
            }
        }
    }

    /// Approximate screen radius, in pixels, that a world bounding sphere of
    /// `world_radius` subtends at `world_center`.
    ///
    /// Scales [`CameraProjection::screen_coverage_ndc`] by half the viewport
    /// height (the `NDC` half-height maps to `viewport.height / 2` pixels).
    #[must_use]
    pub fn screen_coverage_pixels(&self, world_center: Vec3, world_radius: f32) -> f32 {
        self.screen_coverage_ndc(world_center, world_radius) * 0.5 * self.viewport.height
    }

    /// Builds the render-side billboard [`CameraFrame`] for a particle at
    /// `particle_world_pos`, reusing [`super::renderers`]'s facing frame rather
    /// than redefining it.
    ///
    /// This is the bridge between the query `DI` (this module) and the
    /// orientation `DI` (`renderers`): a module that has a query camera can
    /// hand a sprite its camera-facing frame without recomputing a basis.
    #[must_use]
    pub fn billboard_frame_for(&self, particle_world_pos: Vec3) -> CameraFrame {
        CameraFrame {
            to_camera: self.direction_to_camera(particle_world_pos),
            up: self.basis.up,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Axis-aligned reference camera: at the origin looking down `+Z`, a 90°
    /// symmetric `FOV` (`tan(45°) = 1`), near `1`, far `100`, 800x600 target.
    fn reference_camera() -> CameraProjection {
        CameraProjection::perspective(
            Vec3::ZERO,
            CameraBasis::new(
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ),
            1.0,
            100.0,
            1.0,
            1.0,
            Viewport::new(800.0, 600.0),
        )
    }

    fn ortho_camera() -> CameraProjection {
        CameraProjection::orthographic(
            Vec3::ZERO,
            CameraBasis::new(
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ),
            1.0,
            100.0,
            10.0,
            10.0,
            Viewport::new(800.0, 600.0),
        )
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-3
    }

    #[test]
    fn from_forward_up_builds_orthonormal_basis() {
        let basis =
            CameraBasis::from_forward_up(Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 3.0, 0.0));
        assert!(approx(basis.right.x, 1.0));
        assert!(approx(basis.up.y, 1.0));
        assert!(approx(basis.forward.z, 1.0));
        // Orthonormal: axes are unit length and mutually perpendicular.
        assert!(approx(basis.right.length(), 1.0));
        assert!(approx(basis.up.length(), 1.0));
        assert!(approx(basis.forward.length(), 1.0));
        assert!(basis.right.dot(basis.up).abs() <= EPS);
        assert!(basis.right.dot(basis.forward).abs() <= EPS);
    }

    #[test]
    fn from_forward_up_is_safe_for_degenerate_inputs() {
        let basis = CameraBasis::from_forward_up(Vec3::ZERO, Vec3::ZERO);
        assert!(approx(basis.right.length(), 1.0));
        assert!(approx(basis.up.length(), 1.0));
        assert!(approx(basis.forward.length(), 1.0));
        assert!(basis.right.dot(basis.forward).abs() <= EPS);
    }

    #[test]
    fn on_axis_point_projects_to_screen_center() {
        let cam = reference_camera();
        let screen = cam.world_to_screen(Vec3::new(0.0, 0.0, 10.0));
        let px = screen.expect("point in front should project");
        assert!(approx(px[0], 400.0));
        assert!(approx(px[1], 300.0));
    }

    #[test]
    fn world_to_screen_round_trips_offset_point() {
        let cam = reference_camera();
        // At depth 10 with tan_half_fov = 1, view.x = 2 maps to NDC.x = 0.2.
        let world = Vec3::new(2.0, 0.0, 10.0);
        let ndc = cam.world_to_ndc(world).expect("visible");
        assert!(approx(ndc.x, 0.2));
        assert!(approx(ndc.y, 0.0));
        let px = cam.world_to_screen(world).expect("visible");
        assert!(approx(px[0], (0.2 * 0.5 + 0.5) * 800.0));
        assert!(approx(px[1], 300.0));
    }

    #[test]
    fn point_behind_camera_returns_none() {
        let cam = reference_camera();
        assert!(cam.world_to_screen(Vec3::new(0.0, 0.0, -5.0)).is_none());
        assert!(cam.world_to_ndc(Vec3::new(0.0, 0.0, -5.0)).is_none());
        assert_eq!(
            cam.classify_visibility(Vec3::new(0.0, 0.0, -5.0)),
            ClipVisibility::BehindCamera
        );
    }

    #[test]
    fn point_outside_frustum_is_culled() {
        let cam = reference_camera();
        // Far off-axis at shallow depth: NDC.x well beyond 1.
        let world = Vec3::new(100.0, 0.0, 5.0);
        assert!(cam.world_to_screen(world).is_none());
        assert_eq!(
            cam.classify_visibility(world),
            ClipVisibility::OutsideFrustum
        );
    }

    #[test]
    fn depth_linearizes_at_near_and_far_endpoints() {
        let cam = reference_camera();
        assert!(approx(cam.linearize_depth(0.0), cam.near));
        assert!(approx(cam.linearize_depth(1.0), cam.far));
    }

    #[test]
    fn depth_round_trips_through_projection() {
        let cam = reference_camera();
        // Project a known depth to NDC z, then linearize back to view depth.
        let view = cam.world_to_view(Vec3::new(0.0, 0.0, 10.0));
        let ndc = cam
            .world_to_ndc(Vec3::new(0.0, 0.0, 10.0))
            .expect("visible");
        assert!(approx(cam.linearize_depth(ndc.z), view.z));
        assert!(approx(view.z, 10.0));
    }

    #[test]
    fn ortho_depth_is_linear() {
        let cam = ortho_camera();
        assert!(approx(cam.linearize_depth(0.0), cam.near));
        assert!(approx(
            cam.linearize_depth(0.5),
            cam.near + 0.5 * (cam.far - cam.near)
        ));
        assert!(approx(cam.linearize_depth(1.0), cam.far));
    }

    #[test]
    fn distance_and_direction_to_camera() {
        let cam = reference_camera();
        let world = Vec3::new(3.0, 4.0, 0.0);
        assert!(approx(cam.distance_to_camera(world), 5.0));
        assert!(approx(cam.distance_squared_to_camera(world), 25.0));
        let dir = cam.direction_to_camera(world);
        assert!(approx(dir.x, -0.6));
        assert!(approx(dir.y, -0.8));
        assert!(approx(dir.z, 0.0));
        assert!(approx(dir.length(), 1.0));
    }

    #[test]
    fn direction_to_camera_is_zero_at_camera() {
        let cam = reference_camera();
        let dir = cam.direction_to_camera(cam.world_position);
        assert!(dir.length_squared() <= EPS);
    }

    #[test]
    fn view_space_position_matches_world_to_view() {
        let cam = reference_camera();
        let world = Vec3::new(1.0, 2.0, 7.0);
        let a = cam.view_space_position(world);
        let b = cam.world_to_view(world);
        assert!(approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z));
        assert!(approx(a.z, 7.0));
    }

    #[test]
    fn perspective_coverage_shrinks_with_distance() {
        let cam = reference_camera();
        let near = cam.screen_coverage_ndc(Vec3::new(0.0, 0.0, 10.0), 1.0);
        let far = cam.screen_coverage_ndc(Vec3::new(0.0, 0.0, 20.0), 1.0);
        assert!(near > far);
        // radius / (depth * tan) = 1 / (10 * 1) = 0.1.
        assert!(approx(near, 0.1));
        assert!(approx(far, 0.05));
    }

    #[test]
    fn coverage_grows_with_radius() {
        let cam = reference_camera();
        let small = cam.screen_coverage_ndc(Vec3::new(0.0, 0.0, 10.0), 1.0);
        let big = cam.screen_coverage_ndc(Vec3::new(0.0, 0.0, 10.0), 2.0);
        assert!(big > small);
        assert!(approx(big, 2.0 * small));
    }

    #[test]
    fn coverage_pixels_scale_from_ndc() {
        let cam = reference_camera();
        let ndc = cam.screen_coverage_ndc(Vec3::new(0.0, 0.0, 10.0), 1.0);
        let px = cam.screen_coverage_pixels(Vec3::new(0.0, 0.0, 10.0), 1.0);
        assert!(approx(px, ndc * 0.5 * 600.0));
    }

    #[test]
    fn coverage_behind_camera_is_zero() {
        let cam = reference_camera();
        let cov = cam.screen_coverage_ndc(Vec3::new(0.0, 0.0, -5.0), 1.0);
        assert!(cov.abs() <= EPS);
    }

    #[test]
    fn ortho_coverage_is_depth_invariant() {
        let cam = ortho_camera();
        let near = cam.screen_coverage_ndc(Vec3::new(0.0, 0.0, 10.0), 2.0);
        let far = cam.screen_coverage_ndc(Vec3::new(0.0, 0.0, 50.0), 2.0);
        assert!(approx(near, far));
        // radius / half_height = 2 / 10 = 0.2.
        assert!(approx(near, 0.2));
    }

    #[test]
    fn degenerate_fov_does_not_nan() {
        let cam = CameraProjection::perspective(
            Vec3::ZERO,
            CameraBasis::new(
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ),
            1.0,
            100.0,
            0.0,
            0.0,
            Viewport::new(800.0, 600.0),
        );
        let ndc = cam
            .world_to_ndc(Vec3::new(2.0, 3.0, 10.0))
            .expect("w is depth");
        // Zeroed tangents collapse x/y to the optical axis, never NaN.
        assert!(!ndc.x.is_nan() && !ndc.y.is_nan() && !ndc.z.is_nan());
        assert!(ndc.x.abs() <= EPS && ndc.y.abs() <= EPS);
    }

    #[test]
    fn collapsed_near_far_range_is_safe() {
        let cam = CameraProjection::perspective(
            Vec3::ZERO,
            CameraBasis::new(
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ),
            5.0,
            5.0,
            1.0,
            1.0,
            Viewport::new(800.0, 600.0),
        );
        let depth = cam.linearize_depth(0.5);
        assert!(!depth.is_nan());
        assert!(approx(depth, cam.far));
    }

    #[test]
    fn clip_to_ndc_rejects_zero_w() {
        assert!(CameraProjection::clip_to_ndc([1.0, 2.0, 3.0, 0.0]).is_none());
        let ndc = CameraProjection::clip_to_ndc([1.0, 2.0, 3.0, 2.0]).expect("finite w");
        assert!(approx(ndc.x, 0.5) && approx(ndc.y, 1.0) && approx(ndc.z, 1.5));
    }

    #[test]
    fn billboard_frame_reuses_renderer_camera_frame() {
        let cam = reference_camera();
        let frame: CameraFrame = cam.billboard_frame_for(Vec3::new(0.0, 0.0, 10.0));
        // Particle in front looks back along -Z toward the camera at the origin.
        assert!(approx(frame.to_camera.z, -1.0));
        assert!(approx(frame.up.y, 1.0));
    }
}
