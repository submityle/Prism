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
