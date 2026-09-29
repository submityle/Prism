//! Per-particle `billboard` orientation: choosing a facing mode and building an
//! orthonormal `right`/`up` basis for the quad (design §16).
//!
//! A particle is rendered as a camera-facing (or axis-constrained) quad. Before
//! the `GPU` expands a point into two triangles it needs a local 2D frame — a
//! `right` vector and an `up` vector — that is orthonormal so the sprite is
//! neither skewed nor scaled by the frame itself. This module owns the
//! `CPU`-verifiable contract that turns a facing mode plus the particle's world
//! state into that frame.
//!
//! # Strict scope
//! This module *only* selects a [`FacingMode`] and constructs the orthonormal
//! [`OrientationBasis`]. It deliberately does **not** perform camera projection
//! or the view transform (that is the camera module's `CameraBasis`), and it
//! does **not** apply velocity-driven stretch scaling (that is the sprite
//! stretch module). It neither imports nor reconstructs those types.
//!
//! # No transcendental math
//! Orientation is built entirely from vector `cross`/`normalize` products, so
//! the only floating-point primitive used is `sqrt` (inside `normalize`). There
//! is no `sin`/`cos`/`atan`/quaternion path. Every basis is orthonormal by
//! construction: two of the axes come from a normalized cross product of the
//! third with a reference direction, which is exact orthogonality up to
//! floating-point rounding. Degenerate inputs (a camera sitting on the
//! particle, a zero velocity, a reference axis parallel to the facing
//! direction) can never produce a `NaN`: each cross product that collapses to
//! near-zero length falls back to a stable perpendicular or a world basis.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Length below which a vector is treated as degenerate (zero) and its
/// normalization falls back instead of dividing by a near-zero magnitude.
const MIN_LENGTH: f32 = 1.0e-6;

/// World-space X axis, used as a stable fallback `right` vector.
const WORLD_X: [f32; 3] = [1.0, 0.0, 0.0];
/// World-space Y axis, used as a stable fallback `up` vector.
const WORLD_Y: [f32; 3] = [0.0, 1.0, 0.0];
/// World-space Z axis, completing the fallback world basis.
const WORLD_Z: [f32; 3] = [0.0, 0.0, 1.0];

/// How a particle quad orients itself toward the camera or a constraint axis
/// (design §16).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FacingMode {
    /// Spherical `billboard`: the quad's normal points straight at the camera,
    /// so it always faces the viewer fully.
    Billboard,
    /// Cylindrical `billboard` about the world `up` axis: the `up` vector is
    /// locked to world `up` and the quad only yaws to face the camera.
    HorizontalBillboard,
    /// Ground-aligned `billboard`: the quad lies flat with its normal along
    /// world `up`, and its in-plane axes turn to face the camera.
    VerticalBillboard,
    /// `velocity-aligned`: the `up` vector follows the particle velocity so the
    /// sprite trails along its motion.
    VelocityAligned,
    /// Axis-constrained: the `up` vector is locked to a caller-supplied fixed
    /// axis and the quad rotates about it to face the camera.
    FixedAxis,
}

/// An orthonormal 2D frame for a particle quad: the local `right` and `up`
/// axes, each a unit vector and mutually perpendicular.
///
/// The pair is padded to two `vec4` slots when uploaded, matching the shared
/// `std430` `vec3`-to-`vec4` rule (see [`Self::STORAGE_STRIDE`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrientationBasis {
    /// The quad's local `right` axis (unit length).
    pub right: [f32; 3],
    /// The quad's local `up` axis (unit length, perpendicular to `right`).
    pub up: [f32; 3],
}

impl OrientationBasis {
    /// `std430` byte stride of one basis: two `vec3` axes each padded up to a
    /// `vec4`, matching the layout a `GPU` kernel binds.
    pub const STORAGE_STRIDE: usize = VEC4_STRIDE * 2;

    /// Total `std430` byte size of a storage buffer holding `count` bases.
    ///
    /// Clamps up to a single element so an empty pool still yields a valid,
    /// non-empty `GPU` binding.
    #[must_use]
    pub fn gpu_storage_bytes(count: usize) -> usize {
        storage_bytes(Self::STORAGE_STRIDE, count)
    }
}

/// Returns `a - b` component-wise.
#[must_use]
pub fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Returns the dot product of `a` and `b`.
#[must_use]
pub fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Returns the right-handed cross product `a × b`.
#[must_use]
pub fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Returns the Euclidean length of `v` (uses `sqrt`).
#[must_use]
pub fn length3(v: [f32; 3]) -> f32 {
    dot3(v, v).sqrt()
}

/// Normalizes `v` to unit length, or `None` when `v` is shorter than
/// [`MIN_LENGTH`] and therefore too degenerate to give a stable direction.
#[must_use]
pub fn normalize3(v: [f32; 3]) -> Option<[f32; 3]> {
    let len = length3(v);
    if len < MIN_LENGTH {
        None
    } else {
        let inv = 1.0 / len;
        Some([v[0] * inv, v[1] * inv, v[2] * inv])
    }
}

/// Returns a stable unit vector perpendicular to the unit vector `axis`.
///
/// Crosses `axis` with the world axis it is least aligned with, which
/// guarantees a well-conditioned (far-from-parallel) cross product and hence a
/// non-degenerate result.
#[must_use]
fn any_perpendicular(axis: [f32; 3]) -> [f32; 3] {
    let ax = axis[0].abs();
    let ay = axis[1].abs();
    let az = axis[2].abs();
    let reference = if ax <= ay && ax <= az {
        WORLD_X
    } else if ay <= az {
        WORLD_Y
    } else {
        WORLD_Z
    };
    normalize3(cross3(axis, reference)).unwrap_or(WORLD_X)
}

/// Builds a unit `right` vector perpendicular to the unit `up` vector, aiming
/// it toward the camera when a valid view direction is available.
///
/// Falls back to a stable perpendicular of `up` when the camera direction is
/// missing (camera on the particle) or parallel to `up`.
#[must_use]
fn right_perpendicular_to(up_unit: [f32; 3], to_cam: Option<[f32; 3]>) -> [f32; 3] {
    if let Some(view) = to_cam
        && let Some(right) = normalize3(cross3(up_unit, view))
    {
        return right;
    }
    any_perpendicular(up_unit)
}

/// The stable fallback basis used when no facing direction can be derived.
#[must_use]
fn fallback_basis() -> OrientationBasis {
    OrientationBasis {
        right: WORLD_X,
        up: WORLD_Y,
    }
}

/// Computes the orthonormal [`OrientationBasis`] for a particle under `mode`.
///
/// `pos` is the particle position, `cam_pos` the camera position, `vel` the
/// particle velocity, `world_up` the scene's world `up` axis, and `fixed_axis`
/// the constraint axis consulted by [`FacingMode::FixedAxis`]. Inputs that a
/// mode does not need are ignored. The returned `right` and `up` are always
/// unit length and mutually perpendicular; degenerate inputs fall back to a
/// stable frame rather than producing a `NaN`.
#[must_use]
pub fn compute_basis(
    mode: FacingMode,
    pos: [f32; 3],
    cam_pos: [f32; 3],
    vel: [f32; 3],
    world_up: [f32; 3],
    fixed_axis: [f32; 3],
) -> OrientationBasis {
    let to_cam = normalize3(sub3(cam_pos, pos));
    match mode {
        FacingMode::Billboard => {
            let Some(view) = to_cam else {
                return fallback_basis();
            };
            let right = match normalize3(cross3(world_up, view)) {
                Some(r) => r,
                None => any_perpendicular(view),
            };
            let up = cross3(view, right);
            OrientationBasis { right, up }
        }
        FacingMode::HorizontalBillboard => {
            let up = normalize3(world_up).unwrap_or(WORLD_Y);
            let right = right_perpendicular_to(up, to_cam);
            OrientationBasis { right, up }
        }
        FacingMode::VerticalBillboard => {
            let normal = normalize3(world_up).unwrap_or(WORLD_Y);
            let right = right_perpendicular_to(normal, to_cam);
            // Ground-aligned: the plane spans the two horizontal axes while the
            // normal stays along world `up`.
            let up = cross3(normal, right);
            OrientationBasis { right, up }
        }
        FacingMode::VelocityAligned => {
            let up = normalize3(vel).unwrap_or_else(|| normalize3(world_up).unwrap_or(WORLD_Y));
            let right = right_perpendicular_to(up, to_cam);
            OrientationBasis { right, up }
        }
        FacingMode::FixedAxis => {
            let up =
                normalize3(fixed_axis).unwrap_or_else(|| normalize3(world_up).unwrap_or(WORLD_Y));
            let right = right_perpendicular_to(up, to_cam);
            OrientationBasis { right, up }
        }
    }
}

/// Computes an [`OrientationBasis`] for a batch of particles that share the
/// same camera, world `up`, and constraint axis.
///
/// `particles` is a slice of `(position, velocity)` pairs; the returned vector
/// has one basis per input, in order. This mirrors the per-particle `GPU`
/// kernel that fills an orientation buffer in a single dispatch.
#[must_use]
pub fn compute_bases(
    mode: FacingMode,
    cam_pos: [f32; 3],
    world_up: [f32; 3],
    fixed_axis: [f32; 3],
    particles: &[([f32; 3], [f32; 3])],
) -> Vec<OrientationBasis> {
    particles
        .iter()
        .map(|&(pos, vel)| compute_basis(mode, pos, cam_pos, vel, world_up, fixed_axis))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for floating-point comparisons in tests.
    const CMP_EPS: f32 = 1.0e-6;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn is_unit(v: [f32; 3]) -> bool {
        close(length3(v), 1.0)
    }

    fn is_finite3(v: [f32; 3]) -> bool {
        v.iter().all(|c| c.is_finite())
    }

    fn assert_orthonormal(basis: OrientationBasis) {
        assert!(
            is_finite3(basis.right),
            "right not finite: {:?}",
            basis.right
        );
        assert!(is_finite3(basis.up), "up not finite: {:?}", basis.up);
        assert!(is_unit(basis.right), "right not unit: {:?}", basis.right);
        assert!(is_unit(basis.up), "up not unit: {:?}", basis.up);
        assert!(
            close(dot3(basis.right, basis.up), 0.0),
            "right not perpendicular to up: dot = {}",
            dot3(basis.right, basis.up)
        );
    }

    const ALL_MODES: [FacingMode; 5] = [
        FacingMode::Billboard,
        FacingMode::HorizontalBillboard,
        FacingMode::VerticalBillboard,
        FacingMode::VelocityAligned,
        FacingMode::FixedAxis,
    ];

    #[test]
    fn every_mode_is_orthonormal_for_generic_input() {
        let pos = [1.0, 2.0, 3.0];
        let cam_pos = [10.0, 5.0, -4.0];
        let vel = [0.5, -2.0, 1.5];
        let world_up = WORLD_Y;
        let fixed_axis = [0.3, 0.4, 0.8];
        for mode in ALL_MODES {
            let basis = compute_basis(mode, pos, cam_pos, vel, world_up, fixed_axis);
            assert_orthonormal(basis);
        }
    }

    #[test]
    fn billboard_right_is_perpendicular_to_view_direction() {
        let pos = [0.0, 0.0, 0.0];
        let cam_pos = [3.0, 1.0, 5.0];
        let basis = compute_basis(
            FacingMode::Billboard,
            pos,
            cam_pos,
            [0.0, 0.0, 0.0],
            WORLD_Y,
            WORLD_X,
        );
        let to_cam = normalize3(sub3(cam_pos, pos)).unwrap();
        assert!(close(dot3(basis.right, to_cam), 0.0));
        assert!(close(dot3(basis.up, to_cam), 0.0));
        assert_orthonormal(basis);
    }

    #[test]
    fn horizontal_billboard_locks_up_to_world_up() {
        let basis = compute_basis(
            FacingMode::HorizontalBillboard,
            [0.0, 0.0, 0.0],
            [4.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            WORLD_Y,
            WORLD_X,
        );
        assert!(close(basis.up[0], 0.0));
        assert!(close(basis.up[1], 1.0));
        assert!(close(basis.up[2], 0.0));
        assert_orthonormal(basis);
    }

    #[test]
    fn vertical_billboard_normal_is_world_up() {
        let basis = compute_basis(
            FacingMode::VerticalBillboard,
            [0.0, 0.0, 0.0],
            [4.0, 9.0, 2.0],
            [0.0, 0.0, 0.0],
            WORLD_Y,
            WORLD_X,
        );
        // The ground-aligned plane's normal is right × up and must be world up.
        let normal = cross3(basis.right, basis.up);
        assert!(close(normal[0], 0.0));
        assert!(close(normal[1], 1.0));
        assert!(close(normal[2], 0.0));
        assert_orthonormal(basis);
    }

    #[test]
    fn velocity_aligned_up_follows_velocity() {
        let vel = [0.0, 0.0, 4.0];
        let basis = compute_basis(
            FacingMode::VelocityAligned,
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            vel,
            WORLD_Y,
            WORLD_X,
        );
        let dir = normalize3(vel).unwrap();
        assert!(close(basis.up[0], dir[0]));
        assert!(close(basis.up[1], dir[1]));
        assert!(close(basis.up[2], dir[2]));
        assert_orthonormal(basis);
    }

    #[test]
    fn velocity_aligned_falls_back_when_velocity_is_zero() {
        let basis = compute_basis(
            FacingMode::VelocityAligned,
            [0.0, 0.0, 0.0],
            [5.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
            WORLD_Y,
            WORLD_X,
        );
        assert_orthonormal(basis);
    }

    #[test]
    fn fixed_axis_locks_up_to_axis() {
        let axis = [0.0, 0.0, 1.0];
        let basis = compute_basis(
            FacingMode::FixedAxis,
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
            WORLD_Y,
            axis,
        );
        assert!(close(basis.up[0], 0.0));
        assert!(close(basis.up[1], 0.0));
        assert!(close(basis.up[2], 1.0));
        assert_orthonormal(basis);
    }

    #[test]
    fn camera_on_particle_never_produces_nan() {
        let same = [2.0, 2.0, 2.0];
        for mode in ALL_MODES {
            let basis = compute_basis(mode, same, same, [0.0, 0.0, 0.0], WORLD_Y, WORLD_Y);
            assert_orthonormal(basis);
        }
    }

    #[test]
    fn world_up_parallel_to_view_never_produces_nan() {
        // Camera directly above the particle: to_cam is parallel to world up,
        // so the cross(world_up, to_cam) collapses and must fall back.
        let pos = [0.0, 0.0, 0.0];
        let cam_pos = [0.0, 7.0, 0.0];
        for mode in ALL_MODES {
            let basis = compute_basis(mode, pos, cam_pos, [0.0, 0.0, 0.0], WORLD_Y, WORLD_Y);
            assert_orthonormal(basis);
        }
    }

    #[test]
    fn fixed_axis_parallel_to_view_never_produces_nan() {
        let pos = [0.0, 0.0, 0.0];
        let cam_pos = [0.0, 0.0, 6.0];
        let axis = [0.0, 0.0, 1.0]; // parallel to to_cam
        let basis = compute_basis(
            FacingMode::FixedAxis,
            pos,
            cam_pos,
            [0.0, 0.0, 0.0],
            WORLD_Y,
            axis,
        );
        assert_orthonormal(basis);
    }

    #[test]
    fn zero_world_up_falls_back_without_nan() {
        for mode in ALL_MODES {
            let basis = compute_basis(
                mode,
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            );
            assert_orthonormal(basis);
        }
    }

    #[test]
    fn compute_basis_is_deterministic() {
        let pos = [1.0, -2.0, 0.5];
        let cam_pos = [4.0, 3.0, 2.0];
        let vel = [0.1, 0.2, -0.3];
        for mode in ALL_MODES {
            let a = compute_basis(mode, pos, cam_pos, vel, WORLD_Y, WORLD_Z);
            let b = compute_basis(mode, pos, cam_pos, vel, WORLD_Y, WORLD_Z);
            for k in 0..3 {
                assert!(close(a.right[k], b.right[k]));
                assert!(close(a.up[k], b.up[k]));
            }
        }
    }

    #[test]
    fn helpers_are_exact_on_known_vectors() {
        assert!(close(dot3(WORLD_X, WORLD_Y), 0.0));
        assert!(close(dot3([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0));
        let c = cross3(WORLD_X, WORLD_Y);
        assert!(close(c[0], 0.0) && close(c[1], 0.0) && close(c[2], 1.0));
        assert!(close(length3([3.0, 4.0, 0.0]), 5.0));
        assert!(normalize3([0.0, 0.0, 0.0]).is_none());
        let n = normalize3([0.0, 5.0, 0.0]).unwrap();
        assert!(close(n[1], 1.0));
    }

    #[test]
    fn compute_bases_matches_per_particle_and_preserves_order() {
        let particles = [
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ([1.0, 1.0, 1.0], [0.0, 2.0, 0.0]),
            ([-3.0, 2.0, 5.0], [0.0, 0.0, -1.0]),
        ];
        let cam_pos = [4.0, 3.0, 2.0];
        let bases = compute_bases(
            FacingMode::VelocityAligned,
            cam_pos,
            WORLD_Y,
            WORLD_X,
            &particles,
        );
        assert_eq!(bases.len(), particles.len());
        for (i, &(pos, vel)) in particles.iter().enumerate() {
            let expected = compute_basis(
                FacingMode::VelocityAligned,
                pos,
                cam_pos,
                vel,
                WORLD_Y,
                WORLD_X,
            );
            for k in 0..3 {
                assert!(close(bases[i].right[k], expected.right[k]));
                assert!(close(bases[i].up[k], expected.up[k]));
            }
            assert_orthonormal(bases[i]);
        }
        assert!(compute_bases(FacingMode::Billboard, cam_pos, WORLD_Y, WORLD_X, &[]).is_empty());
    }

    #[test]
    fn std430_storage_stride_and_bytes() {
        assert_eq!(OrientationBasis::STORAGE_STRIDE, 32);
        // Empty pool still reserves one element (non-zero binding).
        assert_eq!(OrientationBasis::gpu_storage_bytes(0), 32);
        assert_eq!(OrientationBasis::gpu_storage_bytes(1), 32);
        assert_eq!(OrientationBasis::gpu_storage_bytes(10), 320);
    }
}
