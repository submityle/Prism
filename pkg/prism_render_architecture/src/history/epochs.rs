//! Monotonic history epochs and category-scoped invalidation diffs.
//!
//! Not every history invalidation is a hard reset. When only the lighting
//! changed, a denoiser can keep its geometric history and refresh just the
//! irradiance term. To express that, each independent subsystem publishes a
//! monotonically increasing *epoch* counter, and a history consumer caches the
//! epochs it last integrated. Comparing the cached epochs against the current
//! ones yields exactly which categories moved, expressed as an
//! [`InvalidationMask`].
//!
//! The counters are plain `u64`s bumped with wrapping addition, so the type is
//! `Eq`/`Hash` and the diff is a deterministic integer comparison.

use crate::history::invalidation::InvalidationMask;

/// One independently versioned history category.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EpochCategory {
    /// Scene topology and instance transforms.
    Scene,
    /// The lighting environment.
    Lighting,
    /// Material parameters.
    Material,
    /// The world origin (large-world camera-relative rebasing).
    Origin,
}

impl EpochCategory {
    /// Every category, in field order.
    pub const ALL: [EpochCategory; 4] = [
        EpochCategory::Scene,
        EpochCategory::Lighting,
        EpochCategory::Material,
        EpochCategory::Origin,
    ];

    /// The [`InvalidationMask`] a change in this category contributes.
    ///
    /// An origin rebase shifts every world-space position, which breaks
    /// reprojection just like a camera cut, so it maps to
    /// [`InvalidationMask::CAMERA_CUT`].
    #[must_use]
    pub const fn invalidation(self) -> InvalidationMask {
        match self {
            EpochCategory::Scene => InvalidationMask::SCENE,
            EpochCategory::Lighting => InvalidationMask::LIGHTING,
            EpochCategory::Material => InvalidationMask::MATERIAL,
            EpochCategory::Origin => InvalidationMask::CAMERA_CUT,
        }
    }
}

/// A snapshot of every history category's monotonic epoch counter.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct HistoryEpochs {
    /// Scene-topology epoch.
    pub scene: u64,
    /// Lighting-environment epoch.
    pub lighting: u64,
    /// Material-parameter epoch.
    pub material: u64,
    /// World-origin (rebasing) epoch.
    pub origin: u64,
}

impl HistoryEpochs {
    /// The all-zero epoch snapshot (a freshly created history).
    pub const ZERO: Self = Self {
        scene: 0,
        lighting: 0,
        material: 0,
        origin: 0,
    };

    /// Reads the counter for `category`.
    #[must_use]
    pub const fn get(self, category: EpochCategory) -> u64 {
        match category {
            EpochCategory::Scene => self.scene,
            EpochCategory::Lighting => self.lighting,
            EpochCategory::Material => self.material,
            EpochCategory::Origin => self.origin,
        }
    }

    /// Advances `category` by one (wrapping), in place.
    pub const fn bump(&mut self, category: EpochCategory) {
        match category {
            EpochCategory::Scene => self.scene = self.scene.wrapping_add(1),
            EpochCategory::Lighting => self.lighting = self.lighting.wrapping_add(1),
            EpochCategory::Material => self.material = self.material.wrapping_add(1),
            EpochCategory::Origin => self.origin = self.origin.wrapping_add(1),
        }
    }

    /// Returns a copy with `category` advanced by one (wrapping).
    #[must_use]
    pub const fn bumped(mut self, category: EpochCategory) -> Self {
        self.bump(category);
        self
    }

    /// Computes which categories differ between `self` (the cached epochs) and
    /// `current`, as an [`InvalidationMask`].
    ///
    /// The comparison is symmetric: any counter that differs in either
    /// direction contributes its category's mask. This tolerates the wrapping
    /// counters, since equality — not ordering — decides staleness.
    #[must_use]
    pub fn diff(self, current: Self) -> InvalidationMask {
        let mut mask = InvalidationMask::EMPTY;
        for category in EpochCategory::ALL {
            if self.get(category) != current.get(category) {
                mask.insert(category.invalidation());
            }
        }
        mask
    }

    /// Returns `true` when no category differs from `current`.
    #[must_use]
    pub fn is_current_with(self, current: Self) -> bool {
        self == current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bump_advances_only_the_named_category() {
        let base = HistoryEpochs::ZERO;
        let bumped = base.bumped(EpochCategory::Lighting);
        assert_eq!(bumped.lighting, 1);
        assert_eq!(bumped.scene, 0);
        assert_eq!(bumped.material, 0);
        assert_eq!(bumped.origin, 0);
    }

    #[test]
    fn bump_in_place_matches_bumped_copy() {
        let mut a = HistoryEpochs::ZERO;
        a.bump(EpochCategory::Material);
        let b = HistoryEpochs::ZERO.bumped(EpochCategory::Material);
        assert_eq!(a, b);
        assert_eq!(a.get(EpochCategory::Material), 1);
    }

    #[test]
    fn identical_epochs_diff_to_empty() {
        let a = HistoryEpochs {
            scene: 3,
            lighting: 7,
            material: 1,
            origin: 9,
        };
        assert!(a.is_current_with(a));
        assert!(a.diff(a).is_empty());
    }

    #[test]
    fn diff_reports_each_changed_category() {
        let cached = HistoryEpochs::ZERO;
        let current = HistoryEpochs {
            scene: 1,
            lighting: 0,
            material: 4,
            origin: 0,
        };
        let mask = cached.diff(current);
        assert!(mask.contains(InvalidationMask::SCENE));
        assert!(mask.contains(InvalidationMask::MATERIAL));
        assert!(!mask.contains(InvalidationMask::LIGHTING));
        assert_eq!(mask.count(), 2);
    }

    #[test]
    fn origin_change_maps_to_camera_cut() {
        let mask = HistoryEpochs::ZERO.diff(HistoryEpochs::ZERO.bumped(EpochCategory::Origin));
        assert_eq!(mask, InvalidationMask::CAMERA_CUT);
        assert!(mask.forces_full_reset());
    }

    #[test]
    fn diff_is_symmetric() {
        let a = HistoryEpochs::ZERO;
        let b = HistoryEpochs::ZERO.bumped(EpochCategory::Scene);
        assert_eq!(a.diff(b), b.diff(a));
    }

    #[test]
    fn wrapping_bump_does_not_panic_at_max() {
        let mut e = HistoryEpochs {
            scene: u64::MAX,
            ..HistoryEpochs::ZERO
        };
        e.bump(EpochCategory::Scene);
        assert_eq!(e.scene, 0);
    }

    #[test]
    fn category_invalidation_is_stable() {
        assert_eq!(EpochCategory::Scene.invalidation(), InvalidationMask::SCENE);
        assert_eq!(
            EpochCategory::Lighting.invalidation(),
            InvalidationMask::LIGHTING
        );
        assert_eq!(
            EpochCategory::Material.invalidation(),
            InvalidationMask::MATERIAL
        );
        assert_eq!(
            EpochCategory::Origin.invalidation(),
            InvalidationMask::CAMERA_CUT
        );
    }
}
