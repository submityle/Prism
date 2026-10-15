use alloc::vec::Vec;
use core::fmt;

use crate::pcg::pcg_hash_u32;

/// One HLOD representation and the distance at which it becomes eligible.
///
/// Levels are ordered from most detailed to most distant. `enter_distance` is
/// the boundary for this level, so level zero must start at distance zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HlodLevel {
    /// Stable derived asset handle supplied by the rendering/asset layer.
    pub asset_key: u64,
    /// Distance at which this level becomes eligible while moving away.
    pub enter_distance: u32,
    /// Half-width of the cross-fade band around `enter_distance`.
    pub transition_width: u32,
}

/// Invalid HLOD declaration data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HlodError {
    /// A cluster must contain at least one level.
    EmptyLevels,
    /// The first level must begin at zero distance.
    FirstLevelDistance,
    /// Level distances must increase strictly.
    NonIncreasingDistance,
    /// Cross-fade bands must be non-zero.
    ZeroTransitionWidth,
    /// A requested level index is outside the declaration.
    LevelIndex,
}

impl fmt::Display for HlodError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLevels => f.write_str("HLOD cluster has no levels"),
            Self::FirstLevelDistance => f.write_str("HLOD level zero must start at distance zero"),
            Self::NonIncreasingDistance => f.write_str("HLOD level distances are not increasing"),
            Self::ZeroTransitionWidth => f.write_str("HLOD transition width must be non-zero"),
            Self::LevelIndex => f.write_str("HLOD level index is out of range"),
        }
    }
}

/// A deterministic cross-fade result between two HLOD levels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodTransition {
    /// Source representation.
    pub from_level: usize,
    /// Destination representation.
    pub to_level: usize,
    /// Q16 alpha for the destination representation.
    pub alpha_q16: u16,
    /// Stable per-cluster transition seed for renderer dither.
    pub dither_seed: u32,
}

/// Cross-object HLOD declaration and transition policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HlodCluster {
    id: u64,
    members: Vec<u64>,
    levels: Vec<HlodLevel>,
}

impl HlodCluster {
    /// Validates and creates a cluster. Members are retained in caller order;
    /// the cluster id and level declaration are the deterministic seed inputs.
    pub fn new(id: u64, members: Vec<u64>, levels: Vec<HlodLevel>) -> Result<Self, HlodError> {
        if levels.is_empty() {
            return Err(HlodError::EmptyLevels);
        }
        if levels[0].enter_distance != 0 {
            return Err(HlodError::FirstLevelDistance);
        }
        for level in &levels {
            if level.transition_width == 0 {
                return Err(HlodError::ZeroTransitionWidth);
            }
        }
        for pair in levels.windows(2) {
            if pair[0].enter_distance >= pair[1].enter_distance {
                return Err(HlodError::NonIncreasingDistance);
            }
        }
        Ok(Self {
            id,
            members,
            levels,
        })
    }

    /// Returns the stable cluster identity.
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Returns the declared members in their stable declaration order.
    pub fn members(&self) -> &[u64] {
        &self.members
    }

    /// Returns the HLOD level declarations.
    pub fn levels(&self) -> &[HlodLevel] {
        &self.levels
    }

    /// Selects the farthest level whose boundary has been crossed.
    pub fn level_for_distance(&self, distance: u32) -> usize {
        let mut selected = 0;
        for (index, level) in self.levels.iter().enumerate() {
            if distance >= level.enter_distance {
                selected = index;
            } else {
                break;
            }
        }
        selected
    }

    /// Applies transition hysteresis to a previously active level.
    pub fn transition_target(
        &self,
        distance: u32,
        current_level: usize,
    ) -> Result<usize, HlodError> {
        if current_level >= self.levels.len() {
            return Err(HlodError::LevelIndex);
        }
        let candidate = self.level_for_distance(distance);
        if candidate > current_level {
            let boundary = self.levels[candidate].enter_distance;
            let width = self.levels[candidate].transition_width;
            if distance >= boundary.saturating_add(width) {
                return Ok(candidate);
            }
            Ok(current_level)
        } else if candidate < current_level {
            let boundary = self.levels[current_level].enter_distance;
            let width = self.levels[current_level].transition_width;
            if distance.saturating_add(width) < boundary {
                return Ok(candidate);
            }
            Ok(current_level)
        } else {
            Ok(current_level)
        }
    }

    /// Computes a continuous Q16 blend alpha for an adjacent-level transition.
    ///
    /// The returned alpha is zero before the band and one after it. For a
    /// nearward transition (`from_level > to_level`) the alpha rises as the
    /// distance decreases, so both directions use the same blend contract.
    pub fn transition_alpha(
        &self,
        distance: u32,
        from_level: usize,
        to_level: usize,
    ) -> Result<u16, HlodError> {
        if from_level >= self.levels.len() || to_level >= self.levels.len() {
            return Err(HlodError::LevelIndex);
        }
        if from_level == to_level {
            return Ok(0);
        }
        let boundary_level = from_level.max(to_level);
        let boundary = self.levels[boundary_level].enter_distance;
        let width = self.levels[boundary_level].transition_width;
        let low = boundary.saturating_sub(width);
        let high = boundary.saturating_add(width);
        let span = high.saturating_sub(low).max(1);
        let position = distance.saturating_sub(low).min(span);
        let rising = ((u64::from(position) * 65_535) / u64::from(span)) as u16;
        if from_level < to_level {
            Ok(rising)
        } else {
            Ok(65_535 - rising)
        }
    }

    /// Returns a deterministic dither seed for a level transition.
    pub const fn dither_seed(&self, from_level: usize, to_level: usize) -> Result<u32, HlodError> {
        if from_level >= self.levels.len() || to_level >= self.levels.len() {
            return Err(HlodError::LevelIndex);
        }
        let pair = ((from_level as u64) << 32) | to_level as u64;
        Ok(pcg_hash_u32(self.id ^ pair.rotate_left(11)))
    }

    /// Builds a complete renderer-facing transition record.
    pub fn transition(
        &self,
        distance: u32,
        from_level: usize,
        to_level: usize,
    ) -> Result<LodTransition, HlodError> {
        Ok(LodTransition {
            from_level,
            to_level,
            alpha_q16: self.transition_alpha(distance, from_level, to_level)?,
            dither_seed: self.dither_seed(from_level, to_level)?,
        })
    }
}
