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

    /// The box's center.
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// The box's extents (its full size along each axis).
    #[must_use]
    pub fn extents(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// The radius of the sphere around [`center`](Self::center) that contains the
    /// box, which is the cheap bound a renderer uses for a culling sphere.
    #[must_use]
    pub fn radius(&self) -> f32 {
        let extents = self.extents();
        (extents[0] * extents[0] + extents[1] * extents[1] + extents[2] * extents[2]).sqrt() * 0.5
    }
}
