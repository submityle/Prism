use core::ops::{BitOr, BitOrAssign};
use prism_render_architecture::gpu_scene::{GeometryHandle, SceneHandle, SceneMaterialHandle};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RenderPassMask(pub u32);
impl RenderPassMask {
    pub const OPAQUE: Self = Self(1 << 0);
    pub const MASKED: Self = Self(1 << 1);
    pub const TRANSPARENT: Self = Self(1 << 2);
    pub const SHADOW: Self = Self(1 << 3);
    pub const GI: Self = Self(1 << 4);
    pub const RAY_SCENE: Self = Self(1 << 5);
    pub const PICKING: Self = Self(1 << 6);
    pub const OFFLINE: Self = Self(1 << 7);
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VisibilityStageMask(pub u32);

impl VisibilityStageMask {
    /// Passed previous-frame HZB (or was conservatively retained).
    pub const EARLY: Self = Self(1 << 0);
    /// Must be retested against current-frame HZB after the early depth pass.
    pub const LATE_RETEST: Self = Self(1 << 1);
    /// Passed current-frame HZB or could not be rejected safely.
    pub const LATE_VISIBLE: Self = Self(1 << 2);

    pub const fn contains(self, flag: Self) -> bool {
        self.0 & flag.0 == flag.0
    }
}

impl BitOr for VisibilityStageMask {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for VisibilityStageMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
impl BitOr for RenderPassMask {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl BitOrAssign for RenderPassMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct WorkSortKey(pub u64);

impl WorkSortKey {
    pub fn new(
        pass_class: u8,
        pipeline_class: u8,
        material_class: u8,
        geometry_page: u16,
        depth_bucket: u16,
    ) -> Self {
        Self(
            ((pass_class as u64) << 56)
                | ((pipeline_class as u64) << 48)
                | ((material_class as u64) << 40)
                | ((geometry_page as u64) << 16)
                | depth_bucket as u64,
        )
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuRenderWorkItem {
    pub scene: SceneHandle,
    pub geometry: GeometryHandle,
    pub material: SceneMaterialHandle,
    pub lod_or_cluster: u32,
    pub pass_mask: RenderPassMask,
    pub visibility_stages: VisibilityStageMask,
    pub sort_key: WorkSortKey,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_pass_mask_union_and_membership() {
        let mut mask = RenderPassMask::OPAQUE | RenderPassMask::SHADOW;
        mask |= RenderPassMask::GI;
        assert_ne!(mask.0 & RenderPassMask::OPAQUE.0, 0);
        assert_ne!(mask.0 & RenderPassMask::SHADOW.0, 0);
        assert_ne!(mask.0 & RenderPassMask::GI.0, 0);
        // A pass that was never added is absent from the union.
        assert_eq!(mask.0 & RenderPassMask::OFFLINE.0, 0);
        // Every pass constant occupies a distinct single bit.
        assert_eq!(RenderPassMask::OPAQUE.0 & RenderPassMask::MASKED.0, 0);
        assert_eq!(RenderPassMask::default().0, 0);
    }

    #[test]
    fn visibility_stage_mask_contains_respects_multi_bit_queries() {
        let stages = VisibilityStageMask::EARLY | VisibilityStageMask::LATE_VISIBLE;
        assert!(stages.contains(VisibilityStageMask::EARLY));
        assert!(stages.contains(VisibilityStageMask::LATE_VISIBLE));
        // `contains` requires all queried bits to be present.
        assert!(!stages.contains(VisibilityStageMask::EARLY | VisibilityStageMask::LATE_RETEST));
        // The full set trivially contains any of its subsets.
        assert!(stages.contains(stages));
        assert!(stages.contains(VisibilityStageMask::default()));
    }

    #[test]
    fn sort_key_orders_primarily_by_pass_class() {
        // A higher pass class must sort after a lower one regardless of the
        // lower-priority fields, which encode into less significant bits.
        let opaque = WorkSortKey::new(0, 255, 255, u16::MAX, u16::MAX);
        let transparent = WorkSortKey::new(2, 0, 0, 0, 0);
        assert!(opaque < transparent);

        // Within one pass class, the pipeline class dominates the material.
        let a = WorkSortKey::new(1, 1, 255, 255, 255);
        let b = WorkSortKey::new(1, 2, 0, 0, 0);
        assert!(a < b);
    }

    #[test]
    fn sort_key_depth_bucket_breaks_ties_and_is_deterministic() {
        // With every higher field held equal, the depth bucket is the tie
        // breaker and preserves near-to-far ordering.
        let near = WorkSortKey::new(0, 0, 0, 7, 10);
        let far = WorkSortKey::new(0, 0, 0, 7, 900);
        assert!(near < far);

        // The encoding is a pure function of its inputs: same inputs, same key.
        assert_eq!(
            WorkSortKey::new(1, 2, 3, 4, 5),
            WorkSortKey::new(1, 2, 3, 4, 5)
        );
        // Distinct geometry pages produce distinct keys when all else is equal.
        assert_ne!(
            WorkSortKey::new(0, 0, 0, 1, 0),
            WorkSortKey::new(0, 0, 0, 2, 0)
        );
    }
}
