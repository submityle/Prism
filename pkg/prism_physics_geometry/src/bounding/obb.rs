//! Oriented bounding box (OBB) and the separating-axis overlap test.

use glam::{Mat3, Quat, Vec3};

use crate::bounding::Aabb;

/// An oriented bounding box: a box centered at `center`, sized by
/// `half_extents` along its local axes, and rotated by `orientation`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Obb {
    /// World-space center of the box.
    pub center: Vec3,
    /// Half-sizes along the box's local x, y, z axes (all non-negative).
    pub half_extents: Vec3,
    /// Rotation from local axes into world space.
    pub orientation: Quat,
}

impl Obb {
    /// Creates an oriented box from its center, half-extents, and orientation.
    pub fn new(center: Vec3, half_extents: Vec3, orientation: Quat) -> Obb {
        Obb {
            center,
            half_extents,
            orientation,
        }
    }

    /// Returns the three world-space unit axes of the box.
    pub fn axes(&self) -> [Vec3; 3] {
        let m = Mat3::from_quat(self.orientation);
        [m.x_axis, m.y_axis, m.z_axis]
    }

    /// Returns the point on or inside the box closest to `point`.
    pub fn closest_point(&self, point: Vec3) -> Vec3 {
        let axes = self.axes();
        let e = [self.half_extents.x, self.half_extents.y, self.half_extents.z];
        let d = point - self.center;
        let mut result = self.center;
        for i in 0..3 {
            let dist = d.dot(axes[i]).clamp(-e[i], e[i]);
            result += axes[i] * dist;
        }
        result
    }

    /// Returns `true` when `point` lies on or inside the box.
    pub fn contains_point(&self, point: Vec3) -> bool {
        let axes = self.axes();
        let e = [self.half_extents.x, self.half_extents.y, self.half_extents.z];
        let d = point - self.center;
        for i in 0..3 {
            if d.dot(axes[i]).abs() > e[i] {
                return false;
            }
        }
        true
    }

    /// Returns a world-space axis-aligned bounding box enclosing the OBB.
    pub fn aabb(&self) -> Aabb {
        let axes = self.axes();
        let e = self.half_extents;
        let extent = (axes[0] * e.x).abs() + (axes[1] * e.y).abs() + (axes[2] * e.z).abs();
        Aabb::new(self.center - extent, self.center + extent)
    }

    /// Returns `true` when this box overlaps `other`, using the 15-axis
    /// separating-axis test (three face normals per box plus nine edge-edge
    /// cross products).
    pub fn intersects_obb(&self, other: &Obb) -> bool {
        // Small epsilon added to the absolute rotation terms so that nearly
        // parallel edge cross products (whose true axis is ill-defined) are
        // treated conservatively rather than reporting a false separation.
        const EPSILON: f32 = 1.0e-6;

        let a = self.axes();
        let b = other.axes();
        let ea = [self.half_extents.x, self.half_extents.y, self.half_extents.z];
        let eb = [other.half_extents.x, other.half_extents.y, other.half_extents.z];

        // r[i][j] projects b's j-th axis onto a's i-th axis.
        let mut r = [[0.0f32; 3]; 3];
        let mut abs_r = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = a[i].dot(b[j]);
                abs_r[i][j] = r[i][j].abs() + EPSILON;
            }
        }

        // Translation from self to other, expressed in self's frame.
        let t_world = other.center - self.center;
        let t = [t_world.dot(a[0]), t_world.dot(a[1]), t_world.dot(a[2])];

        // Axes L = A0, A1, A2 (self's face normals).
        for i in 0..3 {
            let ra = ea[i];
            let rb = eb[0] * abs_r[i][0] + eb[1] * abs_r[i][1] + eb[2] * abs_r[i][2];
            if t[i].abs() > ra + rb {
                return false;
            }
        }

        // Axes L = B0, B1, B2 (other's face normals).
        for j in 0..3 {
            let ra = ea[0] * abs_r[0][j] + ea[1] * abs_r[1][j] + ea[2] * abs_r[2][j];
            let rb = eb[j];
            let tj = t[0] * r[0][j] + t[1] * r[1][j] + t[2] * r[2][j];
            if tj.abs() > ra + rb {
                return false;
            }
        }

        // Axis L = A0 x B0.
        if (t[2] * r[1][0] - t[1] * r[2][0]).abs()
            > ea[1] * abs_r[2][0] + ea[2] * abs_r[1][0] + eb[1] * abs_r[0][2] + eb[2] * abs_r[0][1]
        {
            return false;
        }
        // Axis L = A0 x B1.
        if (t[2] * r[1][1] - t[1] * r[2][1]).abs()
            > ea[1] * abs_r[2][1] + ea[2] * abs_r[1][1] + eb[0] * abs_r[0][2] + eb[2] * abs_r[0][0]
        {
            return false;
        }
        // Axis L = A0 x B2.
        if (t[2] * r[1][2] - t[1] * r[2][2]).abs()
            > ea[1] * abs_r[2][2] + ea[2] * abs_r[1][2] + eb[0] * abs_r[0][1] + eb[1] * abs_r[0][0]
        {
            return false;
        }
        // Axis L = A1 x B0.
        if (t[0] * r[2][0] - t[2] * r[0][0]).abs()
            > ea[0] * abs_r[2][0] + ea[2] * abs_r[0][0] + eb[1] * abs_r[1][2] + eb[2] * abs_r[1][1]
        {
            return false;
        }
        // Axis L = A1 x B1.
        if (t[0] * r[2][1] - t[2] * r[0][1]).abs()
            > ea[0] * abs_r[2][1] + ea[2] * abs_r[0][1] + eb[0] * abs_r[1][2] + eb[2] * abs_r[1][0]
        {
            return false;
        }
        // Axis L = A1 x B2.
        if (t[0] * r[2][2] - t[2] * r[0][2]).abs()
            > ea[0] * abs_r[2][2] + ea[2] * abs_r[0][2] + eb[0] * abs_r[1][1] + eb[1] * abs_r[1][0]
        {
            return false;
        }
        // Axis L = A2 x B0.
        if (t[1] * r[0][0] - t[0] * r[1][0]).abs()
            > ea[0] * abs_r[1][0] + ea[1] * abs_r[0][0] + eb[1] * abs_r[2][2] + eb[2] * abs_r[2][1]
        {
            return false;
        }
        // Axis L = A2 x B1.
        if (t[1] * r[0][1] - t[0] * r[1][1]).abs()
            > ea[0] * abs_r[1][1] + ea[1] * abs_r[0][1] + eb[0] * abs_r[2][2] + eb[2] * abs_r[2][0]
        {
            return false;
        }
        // Axis L = A2 x B2.
        if (t[1] * r[0][2] - t[0] * r[1][2]).abs()
            > ea[0] * abs_r[1][2] + ea[1] * abs_r[0][2] + eb[0] * abs_r[2][1] + eb[1] * abs_r[2][0]
        {
            return false;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::Obb;
    use approx::assert_relative_eq;
    use core::f32::consts::FRAC_PI_4;
    use glam::{Quat, Vec3};

    fn unit_box(center: Vec3, orientation: Quat) -> Obb {
        Obb::new(center, Vec3::splat(0.5), orientation)
    }

    #[test]
    fn axis_aligned_overlap_and_separation() {
        let a = unit_box(Vec3::ZERO, Quat::IDENTITY);
        let overlapping = unit_box(Vec3::new(0.5, 0.0, 0.0), Quat::IDENTITY);
        assert!(a.intersects_obb(&overlapping));
        let separated = unit_box(Vec3::new(1.5, 0.0, 0.0), Quat::IDENTITY);
        assert!(!a.intersects_obb(&separated));
    }

    #[test]
    fn rotation_closes_a_gap() {
        // Two unit boxes 1.1 apart on x are separated when axis-aligned
        // (half-widths 0.5 + 0.5 = 1.0 < 1.1), but spinning the second by 45°
        // widens its x-footprint to 0.5*sqrt(2) ~= 0.707, bridging the gap.
        let a = unit_box(Vec3::ZERO, Quat::IDENTITY);
        let aligned = unit_box(Vec3::new(1.1, 0.0, 0.0), Quat::IDENTITY);
        assert!(!a.intersects_obb(&aligned));
        let spun = unit_box(Vec3::new(1.1, 0.0, 0.0), Quat::from_rotation_z(FRAC_PI_4));
        assert!(a.intersects_obb(&spun));
    }

    #[test]
    fn closest_point_and_containment() {
        let b = Obb::new(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY);
        assert!(b.contains_point(Vec3::new(0.5, -1.0, 2.0)));
        assert!(!b.contains_point(Vec3::new(1.5, 0.0, 0.0)));
        let cp = b.closest_point(Vec3::new(5.0, 0.0, 0.0));
        assert!(cp.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn world_aabb_of_rotated_box() {
        let b = Obb::new(Vec3::ZERO, Vec3::new(0.5, 0.5, 0.5), Quat::from_rotation_z(FRAC_PI_4));
        let aabb = b.aabb();
        let half = 0.5 * core::f32::consts::SQRT_2;
        assert_relative_eq!(aabb.max.x, half, epsilon = 1e-6);
        assert_relative_eq!(aabb.max.y, half, epsilon = 1e-6);
        assert_relative_eq!(aabb.max.z, 0.5, epsilon = 1e-6);
    }
}
