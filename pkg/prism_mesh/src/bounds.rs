//! A mesh's local-space bounds.

/// The axis-aligned box a mesh's vertices occupy, in its own space.
///
/// A mesh has no transform of its own — where it sits is the instance's business
/// — so these are *model-space* bounds: the renderer transforms them by the
/// instance transform when it needs world-space culling boxes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MeshBounds {
    /// Component-wise minimum corner.
    pub min: [f32; 3],
    /// Component-wise maximum corner.
    pub max: [f32; 3],
}

impl MeshBounds {
    /// The bounds of a vertex cloud, or `None` when there are no vertices.
    ///
    /// Non-finite positions are skipped rather than poisoning the box: a
    /// degenerate vertex should not make a whole mesh uncullable, and a bounds
    /// computation is not the place to report bad content.
    #[must_use]
    pub fn from_positions(positions: &[[f32; 3]]) -> Option<Self> {
        let mut bounds: Option<Self> = None;
        for position in positions {
            if !position.iter().all(|value| value.is_finite()) {
                continue;
            }
            match &mut bounds {
                None => bounds = Some(Self::from_point(*position)),
                Some(bounds) => bounds.merge_point(*position),
            }
        }
        bounds
    }

    /// Degenerate bounds containing exactly `point`.
    #[must_use]
    pub const fn from_point(point: [f32; 3]) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    /// Grows the box to contain `point`.
    pub fn merge_point(&mut self, point: [f32; 3]) {
        for (axis, value) in point.into_iter().enumerate() {
            self.min[axis] = self.min[axis].min(value);
            self.max[axis] = self.max[axis].max(value);
        }
    }

    /// Grows the box to contain `other`.
    pub fn merge(&mut self, other: Self) {
        self.merge_point(other.min);
        self.merge_point(other.max);
    }

    /// The box's center. Finite `f32` endpoints are averaged in `f64` to avoid
    /// overflowing their sum.
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        core::array::from_fn(|axis| {
            ((f64::from(self.min[axis]) + f64::from(self.max[axis])) * 0.5) as f32
        })
    }

    /// The box's extents (its full size along each axis). A span larger than
    /// `f32::MAX` saturates to `f32::MAX`; callers that need to distinguish that
    /// representational limit should check their input range.
    #[must_use]
    pub fn extents(&self) -> [f32; 3] {
        core::array::from_fn(|axis| {
            saturating_f64_to_f32(f64::from(self.max[axis]) - f64::from(self.min[axis]))
        })
    }

    /// The radius of the sphere around [`center`](Self::center) that contains the
    /// box, which is the cheap bound a renderer uses for a culling sphere. The
    /// calculation uses `f64` intermediates and saturates if the result exceeds
    /// the finite `f32` range.
    #[must_use]
    pub fn radius(&self) -> f32 {
        let half_extents = core::array::from_fn::<_, 3, _>(|axis| {
            (f64::from(self.max[axis]) - f64::from(self.min[axis])) * 0.5
        });
        let radius = (half_extents[0] * half_extents[0]
            + half_extents[1] * half_extents[1]
            + half_extents[2] * half_extents[2])
            .sqrt();
        saturating_f64_to_f32(radius)
    }
}

fn saturating_f64_to_f32(value: f64) -> f32 {
    if value > f64::from(f32::MAX) {
        f32::MAX
    } else if value < f64::from(f32::MIN) {
        f32::MIN
    } else {
        value as f32
    }
}
