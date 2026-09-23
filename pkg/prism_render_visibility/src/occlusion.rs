use crate::ViewFlags;

/// Which temporal depth hierarchy is being queried.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HzbPhase {
    Previous,
    Current,
}

/// Conservative reverse-Z HZB test input. `nearest_depth` and
/// `occluder_depth` are NDC depths where larger values are nearer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HzbTest {
    pub nearest_depth: f32,
    pub occluder_depth: f32,
    pub depth_bias: f32,
    pub projected_velocity: f32,
    pub mip: u32,
    pub sampled_mip_count: u32,
    pub history_epoch: u64,
    pub expected_history_epoch: u64,
    pub view_flags: ViewFlags,
}

impl HzbTest {
    /// Returns true only when rejection is proven safe. Invalid history,
    /// camera cuts, missing mips, non-finite inputs, and fast motion all keep
    /// the candidate visible.
    pub fn is_occluded(self, phase: HzbPhase) -> bool {
        if self.view_flags.contains(ViewFlags::CAMERA_CUT)
            || !self.view_flags.contains(ViewFlags::REVERSE_Z)
            || self.sampled_mip_count == 0
            || self.mip >= self.sampled_mip_count
            || !self.nearest_depth.is_finite()
            || !self.occluder_depth.is_finite()
            || !self.depth_bias.is_finite()
            || !self.projected_velocity.is_finite()
        {
            return false;
        }
        if phase == HzbPhase::Previous && self.history_epoch != self.expected_history_epoch {
            return false;
        }
        let motion_bias = if phase == HzbPhase::Previous {
            self.projected_velocity.abs()
        } else {
            self.projected_velocity.abs() * 0.25
        };
        self.nearest_depth + self.depth_bias.abs() + motion_bias < self.occluder_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test() -> HzbTest {
        HzbTest {
            nearest_depth: 0.25,
            occluder_depth: 0.75,
            depth_bias: 0.01,
            projected_velocity: 0.0,
            mip: 2,
            sampled_mip_count: 8,
            history_epoch: 3,
            expected_history_epoch: 3,
            view_flags: ViewFlags::REVERSE_Z,
        }
    }

    #[test]
    fn reverse_z_rejects_only_strictly_behind_bounds() {
        assert!(test().is_occluded(HzbPhase::Previous));
        assert!(!HzbTest { nearest_depth: 0.8, ..test() }.is_occluded(HzbPhase::Current));
    }

    #[test]
    fn camera_cut_invalid_history_and_fast_motion_are_conservative() {
        assert!(!HzbTest {
            view_flags: ViewFlags::REVERSE_Z | ViewFlags::CAMERA_CUT,
            ..test()
        }
        .is_occluded(HzbPhase::Previous));
        assert!(!HzbTest { history_epoch: 2, ..test() }.is_occluded(HzbPhase::Previous));
        assert!(!HzbTest { projected_velocity: 0.6, ..test() }.is_occluded(HzbPhase::Previous));
        assert!(HzbTest { history_epoch: 2, ..test() }.is_occluded(HzbPhase::Current));
    }
}
