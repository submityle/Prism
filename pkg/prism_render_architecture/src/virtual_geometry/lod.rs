//! Screen-space-error LOD selection for paged cluster geometry.
//!
//! A cluster carries a chain of discrete levels: level `0` is the finest and
//! has the smallest object-space [`geometric_error`](LodLevel::geometric_error),
//! and higher levels are coarser with larger error. The selector projects each
//! level's error to screen pixels at the current view distance and displays the
//! *coarsest* level whose projected error still fits the policy budget, which
//! minimizes triangle load without visibly degrading the silhouette.
//!
//! Two refinements keep the result stable and streaming-friendly:
//!
//! * **Hysteresis** — a level must cross the budget by more than
//!   [`GeometryLodPolicy::hysteresis_pixels`] before the selector switches away
//!   from the previously displayed level, which removes the shimmer of a LOD
//!   oscillating between two neighbours near the threshold.
//! * **Prefetch** — the prefetch query re-runs the selection at a reduced
//!   distance derived from the closing speed, so the page a fast-approaching
//!   camera will need next is requested before it is displayed.

use super::GeometryLodPolicy;

/// One discrete level of a cluster/mesh LOD chain.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LodLevel {
    /// Level index; `0` is the finest, larger is coarser.
    pub level: u32,
    /// Object-space geometric deviation bound of this level, in world units.
    pub geometric_error: f32,
}

/// Perspective factor mapping an object-space error to screen pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodProjection {
    /// Half the viewport height divided by `tan(0.5 * vertical_fov)`. A
    /// world-space size times this over the view-space distance is the size in
    /// pixels.
    pub focal_length_pixels: f32,
}

impl LodProjection {
    /// Builds a projection factor directly from a focal length in pixels.
    #[must_use]
    pub fn from_focal_length_pixels(focal_length_pixels: f32) -> Self {
        Self {
            focal_length_pixels,
        }
    }

    /// Builds a projection factor from a viewport height and the tangent of the
    /// half vertical field of view.
    ///
    /// The tangent is supplied by the caller rather than computed here so this
    /// contracts crate stays dependency-free and leaves libm-deterministic
    /// trig (e.g. `bevy_math::ops::tan`) to the render layer that owns it.
    #[must_use]
    pub fn from_half_fov_tan(viewport_height_pixels: f32, half_vertical_fov_tan: f32) -> Self {
        let half_tan = half_vertical_fov_tan.max(f32::EPSILON);
        Self {
            focal_length_pixels: 0.5 * viewport_height_pixels / half_tan,
        }
    }

    /// Projects an object-space `geometric_error` at `view_distance` to pixels.
    #[must_use]
    pub fn projected_error_pixels(&self, geometric_error: f32, view_distance: f32) -> f32 {
        let distance = view_distance.max(f32::EPSILON);
        geometric_error.max(0.0) * self.focal_length_pixels / distance
    }
}

/// Result of a LOD selection query.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LodSelection {
    /// Level chosen for display at the current distance.
    pub level: u32,
    /// Level that should be made resident to cover imminent motion. Never
    /// coarser than [`Self::level`].
    pub prefetch_level: u32,
}

/// Selects the display and prefetch LODs for a cluster.
///
/// `levels` need not be sorted and must be non-empty for a selection to exist.
/// `view_distance` is the view-space distance to the cluster; `closing_speed`
/// is the component of relative velocity that reduces that distance (positive
/// when approaching), in world units per selection step. `previous` is the
/// display level chosen on the prior step, used for hysteresis.
#[must_use]
pub fn select_lod(
    levels: &[LodLevel],
    projection: LodProjection,
    view_distance: f32,
    closing_speed: f32,
    policy: GeometryLodPolicy,
    previous: Option<u32>,
) -> Option<LodSelection> {
    if levels.is_empty() {
        return None;
    }
    let display = pick_level(levels, projection, view_distance, policy, previous);
    let prefetch_distance =
        (view_distance - closing_speed.max(0.0) * policy.prefetch_velocity_scale).max(0.0);
    let prefetch = pick_level(levels, projection, prefetch_distance, policy, Some(display));
    Some(LodSelection {
        level: display,
        // A closer prefetch distance can only ask for an equal or finer level.
        prefetch_level: prefetch.min(display),
    })
}

/// Coarsest level whose projected error fits `budget`, or the finest level when
/// none fit (the camera is closer than the finest level can satisfy).
fn coarsest_within(
    levels: &[LodLevel],
    projection: LodProjection,
    view_distance: f32,
    budget: f32,
) -> u32 {
    let finest = levels.iter().min_by_key(|l| l.level).map_or(0, |l| l.level);
    levels
        .iter()
        .filter(|l| projection.projected_error_pixels(l.geometric_error, view_distance) <= budget)
        .map(|l| l.level)
        .max()
        .unwrap_or(finest)
}

fn error_of(
    levels: &[LodLevel],
    projection: LodProjection,
    view_distance: f32,
    level: u32,
) -> Option<f32> {
    levels
        .iter()
        .find(|l| l.level == level)
        .map(|l| projection.projected_error_pixels(l.geometric_error, view_distance))
}

fn pick_level(
    levels: &[LodLevel],
    projection: LodProjection,
    view_distance: f32,
    policy: GeometryLodPolicy,
    previous: Option<u32>,
) -> u32 {
    let target = policy.target_error_pixels.max(0.0);
    let hysteresis = policy.hysteresis_pixels.max(0.0);
    let desired = coarsest_within(levels, projection, view_distance, target);

    let Some(previous) = previous else {
        return desired;
    };
    // Previous level no longer present in the chain: fall back to the fresh pick.
    let Some(previous_error) = error_of(levels, projection, view_distance, previous) else {
        return desired;
    };

    if desired == previous {
        return previous;
    }
    if desired < previous {
        // The fresh pick is finer, i.e. `previous` is now too coarse. Only
        // refine once it clearly overshoots the relaxed budget.
        if previous_error > target + hysteresis {
            desired
        } else {
            previous
        }
    } else {
        // The fresh pick is coarser. Only coarsen once the coarser level sits
        // comfortably under the tightened budget.
        let desired_error =
            error_of(levels, projection, view_distance, desired).unwrap_or(f32::INFINITY);
        if desired_error <= (target - hysteresis).max(0.0) {
            desired
        } else {
            previous
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain() -> [LodLevel; 4] {
        [
            LodLevel {
                level: 0,
                geometric_error: 0.01,
            },
            LodLevel {
                level: 1,
                geometric_error: 0.04,
            },
            LodLevel {
                level: 2,
                geometric_error: 0.16,
            },
            LodLevel {
                level: 3,
                geometric_error: 0.64,
            },
        ]
    }

    fn projection() -> LodProjection {
        // 1000px tall viewport, 90 deg vertical fov (tan(45 deg) = 1) => focal 500px.
        LodProjection::from_half_fov_tan(1000.0, 1.0)
    }

    #[test]
    fn projected_error_scales_inversely_with_distance() {
        let p = projection();
        let near = p.projected_error_pixels(0.1, 1.0);
        let far = p.projected_error_pixels(0.1, 4.0);
        assert!(near > far);
        assert!((near / far - 4.0).abs() < 1e-3);
    }

    #[test]
    fn picks_coarsest_level_within_budget() {
        let levels = chain();
        let p = projection();
        let policy = GeometryLodPolicy {
            target_error_pixels: 2.0,
            ..Default::default()
        };
        // Far away every level is cheap: coarsest (3) is chosen.
        let far = select_lod(&levels, p, 400.0, 0.0, policy, None).unwrap();
        assert_eq!(far.level, 3);
        // Up close only the finest fits the budget.
        let near = select_lod(&levels, p, 4.0, 0.0, policy, None).unwrap();
        assert_eq!(near.level, 0);
    }

    #[test]
    fn hysteresis_holds_previous_level_in_deadband() {
        let levels = chain();
        let p = projection();
        let policy = GeometryLodPolicy {
            target_error_pixels: 10.0,
            hysteresis_pixels: 6.0,
            ..Default::default()
        };
        // At this distance the fresh selection refines to level 1, but the
        // level-2 error (~13.3px) stays inside the relaxed budget (16px), so a
        // held level 2 is retained instead of popping finer.
        let held = select_lod(&levels, p, 6.0, 0.0, policy, Some(2)).unwrap();
        let fresh = select_lod(&levels, p, 6.0, 0.0, policy, None).unwrap();
        assert_eq!(fresh.level, 1);
        assert_eq!(held.level, 2);
    }

    #[test]
    fn prefetch_is_never_coarser_than_display() {
        let levels = chain();
        let p = projection();
        let policy = GeometryLodPolicy {
            target_error_pixels: 2.0,
            prefetch_velocity_scale: 4.0,
            ..Default::default()
        };
        let sel = select_lod(&levels, p, 40.0, 5.0, policy, None).unwrap();
        assert!(sel.prefetch_level <= sel.level);
    }

    #[test]
    fn empty_chain_has_no_selection() {
        let p = projection();
        assert!(select_lod(&[], p, 1.0, 0.0, GeometryLodPolicy::default(), None).is_none());
    }
}
