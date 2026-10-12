//! View-frustum planes and the AABB classification test.
//!
//! The GPU shader and the CPU reference cull share this exact formulation —
//! the conservative "positive vertex" test — so the two classify instances
//! with identical arithmetic. [`aabb_inside`] is the reference implementation;
//! the WGSL in `shaders/cull.wgsl` is its mirror.

/// A half-space `dot(normal, point) + distance >= 0`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Plane {
    /// The plane normal, pointing into the kept half-space.
    pub normal: [f32; 3],
    /// The signed distance from the origin to the plane.
    pub distance: f32,
}

impl Plane {
    /// A plane from its normal and signed distance.
    #[must_use]
    pub const fn new(normal: [f32; 3], distance: f32) -> Self {
        Self { normal, distance }
    }
}

/// An axis-aligned bounding box.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Aabb {
    /// The minimum corner.
    pub min: [f32; 3],
    /// The maximum corner.
    pub max: [f32; 3],
}

impl Aabb {
    /// The box centred on `center` with the given half-extents.
    #[must_use]
    pub fn from_center(center: [f32; 3], half_extents: [f32; 3]) -> Self {
        Self {
            min: [
                center[0] - half_extents[0],
                center[1] - half_extents[1],
                center[2] - half_extents[2],
            ],
            max: [
                center[0] + half_extents[0],
                center[1] + half_extents[1],
                center[2] + half_extents[2],
            ],
        }
    }

    /// The box centre.
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }
}

/// A convex frustum described by six half-space planes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Frustum {
    planes: [Plane; 6],
}

impl Frustum {
    /// A frustum from exactly six planes.
    #[must_use]
    pub const fn from_planes(planes: [Plane; 6]) -> Self {
        Self { planes }
    }

    /// The axis-aligned box `[min, max]` expressed as a six-plane frustum.
    ///
    /// Each plane's positive vertex is an axis extreme, so
    /// [`aabb_inside`] is exact for this frustum rather than merely
    /// conservative: a box straddling an edge is kept, but a box fully outside
    /// is always rejected.
    #[must_use]
    pub fn from_box(min: [f32; 3], max: [f32; 3]) -> Self {
        Self {
            planes: [
                Plane::new([1.0, 0.0, 0.0], -min[0]),
                Plane::new([-1.0, 0.0, 0.0], max[0]),
                Plane::new([0.0, 1.0, 0.0], -min[1]),
                Plane::new([0.0, -1.0, 0.0], max[1]),
                Plane::new([0.0, 0.0, 1.0], -min[2]),
                Plane::new([0.0, 0.0, -1.0], max[2]),
            ],
        }
    }

    /// The six planes, in the order the GPU uniform uploads them.
    #[must_use]
    pub const fn planes(&self) -> &[Plane; 6] {
        &self.planes
    }
}

/// The smallest positive-vertex signed distance from `aabb` to any frustum
/// plane.
///
/// For each plane the distance is measured at the box corner furthest along the
/// plane normal, so it is non-negative exactly when the box is inside that
/// half-space and negative when the whole box is behind it. The minimum over
/// the six planes is therefore the margin by which the box survives: `>= 0`
/// inside, `< 0` culled. This is the conservative (positive-vertex) test, and
/// the GPU shader computes it verbatim; see `shaders/cull.wgsl`.
#[must_use]
pub fn signed_distance(frustum: &Frustum, aabb: &Aabb) -> f32 {
    frustum
        .planes()
        .iter()
        .map(|plane| {
            // The AABB corner furthest along the plane normal: if even that
            // corner is behind the plane, the whole box is.
            let extreme = [
                if plane.normal[0] >= 0.0 {
                    aabb.max[0]
                } else {
                    aabb.min[0]
                },
                if plane.normal[1] >= 0.0 {
                    aabb.max[1]
                } else {
                    aabb.min[1]
                },
                if plane.normal[2] >= 0.0 {
                    aabb.max[2]
                } else {
                    aabb.min[2]
                },
            ];
            plane.normal[0] * extreme[0]
                + plane.normal[1] * extreme[1]
                + plane.normal[2] * extreme[2]
                + plane.distance
        })
        .fold(f32::INFINITY, f32::min)
}

/// Whether `aabb` survives `frustum`: true unless it lies fully behind some
/// plane. A box that straddles a plane is always kept.
///
/// The classification runs on the GPU verbatim; see `shaders/cull.wgsl`.
#[must_use]
pub fn aabb_inside(frustum: &Frustum, aabb: &Aabb) -> bool {
    signed_distance(frustum, aabb) >= 0.0
}
