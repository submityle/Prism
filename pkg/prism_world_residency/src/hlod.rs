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

/// Engine-neutral declaration for one baked HLOD representation.
///
/// This is metadata only. It does not contain mesh payload, a renderer
/// resource handle, or an asset-streaming request. The `id` is the stable key
/// that an external producer may use when materializing the representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HlodRepresentation {
    id: u64,
    cluster_id: u64,
    level: u32,
    source_hash: u64,
    member_count: u32,
    estimated_declaration_bytes: u64,
}

impl HlodRepresentation {
    pub(crate) const fn new(
        id: u64,
        cluster_id: u64,
        level: u32,
        source_hash: u64,
        member_count: u32,
        estimated_declaration_bytes: u64,
    ) -> Self {
        Self {
            id,
            cluster_id,
            level,
            source_hash,
            member_count,
            estimated_declaration_bytes,
        }
    }

    /// Returns the stable representation identity.
    pub const fn id(self) -> u64 {
        self.id
    }

    /// Returns the owning cluster identity.
    pub const fn cluster_id(self) -> u64 {
        self.cluster_id
    }

    /// Returns the zero-based HLOD level.
    pub const fn level(self) -> u32 {
        self.level
    }

    /// Returns the stable hash of the source proxy group.
    pub const fn source_hash(self) -> u64 {
        self.source_hash
    }

    /// Returns the number of source proxies represented by this declaration.
    pub const fn member_count(self) -> u32 {
        self.member_count
    }

    /// Returns the estimated bytes occupied by this metadata declaration.
    pub const fn estimated_declaration_bytes(self) -> u64 {
        self.estimated_declaration_bytes
    }
}

/// Axis-aligned bounds of a baked proxy artifact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HlodBounds {
    /// Minimum corner of the bounds in source world units.
    pub min: [f32; 3],
    /// Maximum corner of the bounds in source world units.
    pub max: [f32; 3],
    /// Center of the bounds in source world units.
    pub center: [f32; 3],
    /// Conservative scalar radius of the bounds.
    pub radius: f32,
}

impl HlodBounds {
    pub(crate) const fn new(min: [f32; 3], max: [f32; 3], center: [f32; 3], radius: f32) -> Self {
        Self {
            min,
            max,
            center,
            radius,
        }
    }
}

/// One engine-neutral vertex in a baked proxy mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HlodProxyVertex {
    /// Position in source world units.
    pub position: [f32; 3],
    /// Unit or corner normal used by a downstream material adapter.
    pub normal: [f32; 3],
    /// Deterministic normalized UV coordinates.
    pub uv: [f32; 2],
}

/// Concrete, engine-neutral proxy geometry for one representation.
#[derive(Clone, Debug, PartialEq)]
pub struct HlodProxyGeometry {
    bounds: HlodBounds,
    vertices: Vec<HlodProxyVertex>,
    indices: Vec<u32>,
    hash: u64,
}

impl HlodProxyGeometry {
    pub(crate) fn new(
        bounds: HlodBounds,
        vertices: Vec<HlodProxyVertex>,
        indices: Vec<u32>,
        hash: u64,
    ) -> Self {
        Self {
            bounds,
            vertices,
            indices,
            hash,
        }
    }

    /// Returns the conservative source and mesh bounds.
    pub const fn bounds(&self) -> &HlodBounds {
        &self.bounds
    }

    /// Returns vertices in stable corner order.
    pub fn vertices(&self) -> &[HlodProxyVertex] {
        &self.vertices
    }

    /// Returns triangle-list indices in stable winding order.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Returns the stable hash of bounds, vertices, and indices.
    pub const fn hash(&self) -> u64 {
        self.hash
    }
}

/// Engine-neutral material parameters for a baked proxy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HlodMaterialParams {
    /// Linear base color and opacity.
    pub base_color: [f32; 4],
    /// Scalar roughness in the inclusive range `[0, 1]`.
    pub roughness: f32,
    /// Scalar metallic weight in the inclusive range `[0, 1]`.
    pub metallic: f32,
}

/// Concrete, engine-neutral material payload for one representation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HlodProxyMaterial {
    material_key: u64,
    params: HlodMaterialParams,
    hash: u64,
}

impl HlodProxyMaterial {
    pub(crate) const fn new(material_key: u64, params: HlodMaterialParams, hash: u64) -> Self {
        Self {
            material_key,
            params,
            hash,
        }
    }

    /// Returns the stable material key for downstream material lookup.
    pub const fn material_key(self) -> u64 {
        self.material_key
    }

    /// Returns the material parameters without an engine-specific handle.
    pub const fn params(self) -> HlodMaterialParams {
        self.params
    }

    /// Returns the stable hash of the material key and parameters.
    pub const fn hash(self) -> u64 {
        self.hash
    }
}

/// Complete geometry/material artifact for one baked HLOD representation.
#[derive(Clone, Debug, PartialEq)]
pub struct HlodProxyArtifact {
    representation_id: u64,
    geometry: HlodProxyGeometry,
    material: HlodProxyMaterial,
    hash: u64,
}

impl HlodProxyArtifact {
    pub(crate) const fn new(
        representation_id: u64,
        geometry: HlodProxyGeometry,
        material: HlodProxyMaterial,
        hash: u64,
    ) -> Self {
        Self {
            representation_id,
            geometry,
            material,
            hash,
        }
    }

    /// Returns the representation identity associated with this artifact.
    pub const fn representation_id(&self) -> u64 {
        self.representation_id
    }

    /// Returns the concrete proxy geometry.
    pub const fn geometry(&self) -> &HlodProxyGeometry {
        &self.geometry
    }

    /// Returns the concrete proxy material.
    pub const fn material(&self) -> HlodProxyMaterial {
        self.material
    }

    /// Returns the stable hash of representation, geometry, and material.
    pub const fn hash(&self) -> u64 {
        self.hash
    }
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
