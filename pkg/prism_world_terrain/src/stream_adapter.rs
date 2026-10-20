//! Deterministic projection of terrain tiles into world-stream manifests.

use alloc::vec::Vec;

use prism_world_stream::{AabbI64, CellManifest, DataLayerMask, GlobalPos, SectorId};

use crate::{face_uv_to_direction, FaceUv, TileKey};

/// Configuration shared by every manifest emitted by a terrain adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TerrainStreamConfig {
    radius_mm: i64,
    scene_key: u128,
    content_version: u64,
    derived_hash_seed: u128,
    io_bytes: u64,
}

impl TerrainStreamConfig {
    /// Creates a configuration for a spherical terrain surface.
    pub const fn new(
        radius_mm: i64,
        scene_key: u128,
        content_version: u64,
        derived_hash_seed: u128,
        io_bytes: u64,
    ) -> Result<Self, TerrainStreamError> {
        if radius_mm <= 0 || radius_mm > i64::MAX / 2 {
            return Err(TerrainStreamError::RadiusOutOfRange);
        }
        if io_bytes == 0 {
            return Err(TerrainStreamError::ZeroIoBytes);
        }
        Ok(Self {
            radius_mm,
            scene_key,
            content_version,
            derived_hash_seed,
            io_bytes,
        })
    }

    /// Returns the planet radius in millimetres.
    pub const fn radius_mm(self) -> i64 {
        self.radius_mm
    }

    /// Returns the stable scene/template key.
    pub const fn scene_key(self) -> u128 {
        self.scene_key
    }

    /// Returns the scene schema/content version.
    pub const fn content_version(self) -> u64 {
        self.content_version
    }

    /// Returns the base used for derived-payload hashing.
    pub const fn derived_hash_seed(self) -> u128 {
        self.derived_hash_seed
    }

    /// Returns the estimated bytes required by one tile representation.
    pub const fn io_bytes(self) -> u64 {
        self.io_bytes
    }
}

/// Errors returned while projecting terrain into world-stream contracts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerrainStreamError {
    /// The configured radius is zero, negative, or too close to `i64::MAX`.
    RadiusOutOfRange,
    /// A zero-sized stream payload would make admission accounting ambiguous.
    ZeroIoBytes,
    /// Projected bounds could not be represented as signed millimetres.
    BoundsOverflow,
}

/// A stateless adapter from validated cube-sphere tiles to stream manifests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TerrainStreamAdapter {
    config: TerrainStreamConfig,
}

impl TerrainStreamAdapter {
    /// Creates an adapter with validated shared manifest configuration.
    pub const fn new(config: TerrainStreamConfig) -> Self {
        Self { config }
    }

    /// Returns the adapter configuration.
    pub const fn config(self) -> TerrainStreamConfig {
        self.config
    }

    /// Encodes a tile identity into a unique stream sector at the tile level.
    ///
    /// The stream coordinate convention is `(face, tile_x, tile_y)`. It is an
    /// identity atlas, not a claim that cube-face adjacency is Euclidean; the
    /// authoritative planetary spatial extent remains in `CellManifest::bounds`.
    pub const fn sector_for(self, tile: TileKey) -> SectorId {
        SectorId::new(
            tile.level(),
            tile.face().index() as i64,
            tile.x() as i64,
            tile.y() as i64,
        )
    }

    /// Computes a conservative signed-millimetre AABB for a terrain tile.
    pub fn bounds_for(self, tile: TileKey) -> Result<AabbI64, TerrainStreamError> {
        let (min_uv, max_uv) = tile.bounds();
        let us = axis_candidates(min_uv.u(), max_uv.u());
        let vs = axis_candidates(min_uv.v(), max_uv.v());
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for &u in &us {
            for &v in &vs {
                let direction = face_uv_to_direction(
                    FaceUv::new(tile.face(), u, v).expect("tile bounds are valid"),
                )
                .components();
                for axis in 0..3 {
                    let coordinate = f64::from(direction[axis]) * self.config.radius_mm as f64;
                    min[axis] = min[axis].min(coordinate);
                    max[axis] = max[axis].max(coordinate);
                }
            }
        }

        // `face_uv_to_direction` uses f32 normalization. Expand by a bound
        // proportional to the radius before integer enclosure so the adapter
        // never clips a surface point because of the f32 conversion.
        let margin = (self.config.radius_mm / 1_000_000).saturating_add(1) as f64;
        let min = [
            checked_floor(min[0] - margin)?,
            checked_floor(min[1] - margin)?,
            checked_floor(min[2] - margin)?,
        ];
        let max = [
            checked_ceil(max[0] + margin)?,
            checked_ceil(max[1] + margin)?,
            checked_ceil(max[2] + margin)?,
        ];
        AabbI64::new(
            GlobalPos::new(min[0], min[1], min[2]),
            GlobalPos::new(max[0], max[1], max[2]),
        )
        .ok_or(TerrainStreamError::BoundsOverflow)
    }

    /// Emits one stream manifest for a validated terrain tile.
    pub fn manifest_for(self, tile: TileKey) -> Result<CellManifest, TerrainStreamError> {
        Ok(CellManifest {
            sector: self.sector_for(tile),
            bounds: self.bounds_for(tile)?,
            scene_key: self.config.scene_key,
            content_version: self.config.content_version,
            derived_hash: derived_hash(self.config.derived_hash_seed, tile),
            io_bytes: self.config.io_bytes,
            data_layers: DataLayerMask::EMPTY,
        })
    }

    /// Emits manifests in input order, stopping at the first projection error.
    pub fn manifests_for(self, tiles: &[TileKey]) -> Result<Vec<CellManifest>, TerrainStreamError> {
        let mut manifests = Vec::with_capacity(tiles.len());
        for &tile in tiles {
            manifests.push(self.manifest_for(tile)?);
        }
        Ok(manifests)
    }
}

fn axis_candidates(min: f32, max: f32) -> [f32; 3] {
    [min, max, if min <= 0.0 && 0.0 <= max { 0.0 } else { min }]
}

fn checked_floor(value: f64) -> Result<i64, TerrainStreamError> {
    if !value.is_finite() || value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err(TerrainStreamError::BoundsOverflow);
    }
    let truncated = value as i64;
    if value < 0.0 && truncated as f64 > value {
        truncated
            .checked_sub(1)
            .ok_or(TerrainStreamError::BoundsOverflow)
    } else {
        Ok(truncated)
    }
}

fn checked_ceil(value: f64) -> Result<i64, TerrainStreamError> {
    if !value.is_finite() || value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err(TerrainStreamError::BoundsOverflow);
    }
    let truncated = value as i64;
    if value > 0.0 && (truncated as f64) < value {
        truncated
            .checked_add(1)
            .ok_or(TerrainStreamError::BoundsOverflow)
    } else {
        Ok(truncated)
    }
}

fn derived_hash(seed: u128, tile: TileKey) -> u128 {
    let identity = (u128::from(tile.face().index()) << 120)
        | (u128::from(tile.level()) << 112)
        | (u128::from(tile.x()) << 56)
        | u128::from(tile.y());
    let low = mix64((seed as u64) ^ identity as u64);
    let high = mix64((seed >> 64) as u64 ^ (identity >> 64) as u64 ^ low.rotate_left(17));
    (u128::from(high) << 64) | u128::from(low)
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CubeFace, TileKey, MAX_TILE_LEVEL};
    use prism_world_stream::{DemandSource, StreamAction, StreamBudget, StreamScheduler};

    fn adapter() -> TerrainStreamAdapter {
        TerrainStreamAdapter::new(
            TerrainStreamConfig::new(6_371_000_000, 0xabc, 7, 0x1234, 4096).expect("valid config"),
        )
    }

    #[test]
    fn tile_identity_maps_to_unique_stream_sector() {
        let a = adapter();
        let west = TileKey::new(CubeFace::PositiveZ, 4, 3, 8).unwrap();
        let east = TileKey::new(CubeFace::NegativeZ, 4, 3, 8).unwrap();
        assert_ne!(a.sector_for(west), a.sector_for(east));
        assert_eq!(a.sector_for(west).level, 4);
        assert_eq!(a.sector_for(west).x, CubeFace::PositiveZ.index() as i64);
    }

    #[test]
    fn bounds_are_finite_and_contain_tile_surface_samples() {
        let a = adapter();
        let tile = TileKey::new(CubeFace::PositiveZ, 5, 7, 19).unwrap();
        let bounds = a.bounds_for(tile).unwrap();
        let (min_uv, max_uv) = tile.bounds();
        for uv in [min_uv, max_uv] {
            let direction = face_uv_to_direction(uv).components();
            let point = [
                f64::from(direction[0]) * a.config().radius_mm() as f64,
                f64::from(direction[1]) * a.config().radius_mm() as f64,
                f64::from(direction[2]) * a.config().radius_mm() as f64,
            ];
            assert!(point[0] >= bounds.min.x as f64 && point[0] <= bounds.max.x as f64);
            assert!(point[1] >= bounds.min.y as f64 && point[1] <= bounds.max.y as f64);
            assert!(point[2] >= bounds.min.z as f64 && point[2] <= bounds.max.z as f64);
        }
    }

    #[test]
    fn manifest_is_accepted_by_stream_scheduler() {
        let a = adapter();
        let tile = TileKey::root(CubeFace::PositiveX);
        let manifest = a.manifest_for(tile).unwrap();
        let demand = DemandSource::new(manifest.sector, 0, 0, 10).unwrap();
        let mut scheduler = StreamScheduler::new();
        let plan = scheduler.plan(
            &[demand],
            &[manifest],
            StreamBudget {
                io_bytes: 4096,
                max_actions: 1,
            },
        );
        assert_eq!(plan.admitted_bytes, 4096);
        assert!(
            matches!(plan.actions.as_slice(), [StreamAction::Load(request)] if request.sector == manifest.sector)
        );
        assert!(plan.trace.replay().is_ok());
    }

    #[test]
    fn invalid_configuration_and_level_boundary_are_rejected_or_handled() {
        assert_eq!(
            TerrainStreamConfig::new(0, 1, 1, 1, 1),
            Err(TerrainStreamError::RadiusOutOfRange)
        );
        assert_eq!(
            TerrainStreamConfig::new(1, 1, 1, 1, 0),
            Err(TerrainStreamError::ZeroIoBytes)
        );
        let manifest = adapter()
            .manifest_for(TileKey::new(CubeFace::NegativeY, MAX_TILE_LEVEL, 0, 0).unwrap())
            .unwrap();
        assert_eq!(manifest.sector.level, MAX_TILE_LEVEL);
    }

    #[test]
    fn batch_preserves_order_and_hashes_are_stable() {
        let a = adapter();
        let tiles = [
            TileKey::root(CubeFace::PositiveZ),
            TileKey::new(CubeFace::PositiveZ, 1, 1, 0).unwrap(),
        ];
        let first = a.manifests_for(&tiles).unwrap();
        let second = a.manifests_for(&tiles).unwrap();
        assert_eq!(first, second);
        assert_eq!(first[0].sector, a.sector_for(tiles[0]));
        assert_ne!(first[0].derived_hash, first[1].derived_hash);
    }
}
