use prism_math::{Mat4, Vec3, Vec4};

/// A normalized plane `normal · point + d >= 0` means inside.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HalfSpace {
    pub normal: Vec3,
    pub d: f32,
}

impl HalfSpace {
    pub fn signed_distance(self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.d
    }
}

/// Fixed-capacity frustum; infinite projections contain five planes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    pub planes: [HalfSpace; 6],
    pub count: u8,
}

impl Default for Frustum {
    fn default() -> Self {
        Self {
            planes: [HalfSpace::default(); 6],
            count: 0,
        }
    }
}

impl Frustum {
    pub fn contains_point(self, point: Vec3) -> bool {
        self.planes[..self.count as usize]
            .iter()
            .all(|plane| plane.signed_distance(point) >= 0.0)
    }

    pub fn intersects_sphere(self, center: Vec3, radius: f32) -> bool {
        self.planes[..self.count as usize]
            .iter()
            .all(|plane| plane.signed_distance(center) >= -radius)
    }
}

pub fn extract_frustum(clip_from_world: &Mat4, reverse_z: bool, finite_far: bool) -> Frustum {
    let rows = [
        Vec4::new(
            clip_from_world.x_axis.x,
            clip_from_world.y_axis.x,
            clip_from_world.z_axis.x,
            clip_from_world.w_axis.x,
        ),
        Vec4::new(
            clip_from_world.x_axis.y,
            clip_from_world.y_axis.y,
            clip_from_world.z_axis.y,
            clip_from_world.w_axis.y,
        ),
        Vec4::new(
            clip_from_world.x_axis.z,
            clip_from_world.y_axis.z,
            clip_from_world.z_axis.z,
            clip_from_world.w_axis.z,
        ),
        Vec4::new(
            clip_from_world.x_axis.w,
            clip_from_world.y_axis.w,
            clip_from_world.z_axis.w,
            clip_from_world.w_axis.w,
        ),
    ];
    let mut out = Frustum::default();
    let candidates = [
        rows[3] + rows[0],
        rows[3] - rows[0],
        rows[3] + rows[1],
        rows[3] - rows[1],
        if reverse_z {
            rows[3] - rows[2]
        } else {
            rows[2]
        },
    ];
    for plane in candidates {
        let n = Vec3::new(plane.x, plane.y, plane.z);
        let len = n.length();
        if len.is_finite() && len > 1.0e-8 {
            out.planes[out.count as usize] = HalfSpace {
                normal: n / len,
                d: plane.w / len,
            };
            out.count += 1;
        }
    }
    if finite_far {
        let plane = if reverse_z {
            rows[2]
        } else {
            rows[3] - rows[2]
        };
        let n = Vec3::new(plane.x, plane.y, plane.z);
        let len = n.length();
        if len.is_finite() && len > 1.0e-8 {
            out.planes[out.count as usize] = HalfSpace {
                normal: n / len,
                d: plane.w / len,
            };
            out.count += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_math::perspective_infinite_reverse_z_rh;

    #[test]
    fn infinite_projection_has_five_planes() {
        let projection = perspective_infinite_reverse_z_rh(core::f32::consts::FRAC_PI_2, 1.0, 0.1);
        let frustum = extract_frustum(&projection, true, false);
        assert_eq!(frustum.count, 5);
        assert!(frustum.contains_point(Vec3::new(0.0, 0.0, -1.0)));
        assert!(!frustum.contains_point(Vec3::new(100.0, 0.0, -1.0)));
    }
}
