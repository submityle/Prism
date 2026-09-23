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
