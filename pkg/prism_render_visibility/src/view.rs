use core::ops::{BitOr, BitOrAssign};
use prism_render_architecture::abi::GenerationalHandle;

pub type ViewHandle = GenerationalHandle;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ViewFlags(pub u32);

impl ViewFlags {
    pub const CAMERA_CUT: Self = Self(1 << 0);
    pub const REVERSE_Z: Self = Self(1 << 1);
    pub const SHADOW: Self = Self(1 << 2);
    pub const REFLECTION: Self = Self(1 << 3);
    pub const OFFLINE: Self = Self(1 << 4);
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}
impl BitOr for ViewFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl BitOrAssign for ViewFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HistoryPolicy {
    Reuse,
    Reset,
    Conservative,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuViewRecord {
    pub handle: ViewHandle,
    pub clip_from_world: [[f32; 4]; 4],
    pub previous_clip_from_world: [[f32; 4]; 4],
    pub world_position: [f32; 3],
    pub lod_scale: f32,
    pub viewport: [u32; 4],
    pub frustum_planes: [[f32; 4]; 6],
    pub layer_mask: u32,
    pub flags: ViewFlags,
    pub history_epoch: u64,
}

impl GpuViewRecord {
    pub fn history_policy(&self, previous_epoch: Option<u64>) -> HistoryPolicy {
        if self.flags.contains(ViewFlags::CAMERA_CUT) || previous_epoch != Some(self.history_epoch)
        {
            HistoryPolicy::Reset
        } else {
            HistoryPolicy::Reuse
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::abi::GenerationalHandle;

    fn record() -> GpuViewRecord {
        GpuViewRecord {
            handle: GenerationalHandle {
                index: 3,
                generation: 7,
            },
            clip_from_world: [[0.0; 4]; 4],
            previous_clip_from_world: [[0.0; 4]; 4],
            world_position: [0.0, 0.0, 0.0],
            lod_scale: 1.0,
            viewport: [0, 0, 1280, 720],
            frustum_planes: [[0.0, 0.0, 0.0, 1.0]; 6],
            layer_mask: 1,
            flags: ViewFlags::default(),
            history_epoch: 42,
        }
    }

    #[test]
    fn view_flags_union_is_superset_of_each_operand() {
        let combined = ViewFlags::SHADOW | ViewFlags::OFFLINE;
        assert!(combined.contains(ViewFlags::SHADOW));
        assert!(combined.contains(ViewFlags::OFFLINE));
        // A flag that was never set is not reported as contained.
        assert!(!combined.contains(ViewFlags::REVERSE_Z));
        // Empty flags are contained in everything (subset of the bitset).
        assert!(combined.contains(ViewFlags::default()));
        // The raw bits equal the OR of the two distinct single-bit flags.
        assert_eq!(combined.0, ViewFlags::SHADOW.0 | ViewFlags::OFFLINE.0);
    }

    #[test]
    fn view_flags_assign_accumulates_without_dropping_bits() {
        let mut flags = ViewFlags::CAMERA_CUT;
        flags |= ViewFlags::REFLECTION;
        flags |= ViewFlags::REFLECTION;
        assert!(flags.contains(ViewFlags::CAMERA_CUT));
        assert!(flags.contains(ViewFlags::REFLECTION));
        // Re-applying an already-set flag is idempotent.
        assert_eq!(flags.0, ViewFlags::CAMERA_CUT.0 | ViewFlags::REFLECTION.0);
        // Distinct flag constants occupy distinct bits.
        assert_eq!(ViewFlags::CAMERA_CUT.0 & ViewFlags::REFLECTION.0, 0);
    }

    #[test]
    fn history_policy_reuses_only_on_matching_epoch_without_camera_cut() {
        let view = record();
        // Matching epoch and no camera cut allows the previous frame reuse.
        assert_eq!(view.history_policy(Some(42)), HistoryPolicy::Reuse);
        // A mismatched epoch forces a reset.
        assert_eq!(view.history_policy(Some(41)), HistoryPolicy::Reset);
        // A missing previous epoch (first frame) forces a reset.
        assert_eq!(view.history_policy(None), HistoryPolicy::Reset);
    }

    #[test]
    fn camera_cut_flag_forces_history_reset_even_on_epoch_match() {
        let mut view = record();
        view.flags |= ViewFlags::CAMERA_CUT;
        assert_eq!(view.history_policy(Some(42)), HistoryPolicy::Reset);
    }
}
