//! Atomic cell-lifecycle adapter for terrain-derived physics heightfields.
//!
//! The world runtime owns cell streaming and the scene host owns authoritative
//! content. This crate adds an optional physics consumer at that boundary:
//! payloads are decoded and fully staged before the downstream [`SceneHost`]
//! is called, and a failed host operation leaves the previous physics snapshot
//! unchanged. Physics data remains disposable derived state and can be evicted
//! independently of the authoritative cell payload.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::collections::BTreeMap;
use std::fmt;
use std::vec::Vec;

use prism_world_physics_bridge::{
    HeightfieldBridgeBudget, HeightfieldBridgeError, HeightfieldResidencyBudget,
    PhysicsHeightfieldSnapshot, WorldHeightfieldPhysicsResidency,
};
use prism_world_runtime::SceneHost;
use prism_world_stream::{CellManifest, SectorId};
use prism_world_terrain::{HeightfieldTile, PlanarHeightfieldEmbedding, TileKey};

/// One decoded terrain heightfield and its world embedding.
#[derive(Clone, Debug)]
pub struct DecodedTerrainHeightfield {
    tile: HeightfieldTile,
    embedding: PlanarHeightfieldEmbedding,
}

impl DecodedTerrainHeightfield {
    /// Creates a decoded heightfield input for a cell snapshot.
    #[must_use]
    pub const fn new(tile: HeightfieldTile, embedding: PlanarHeightfieldEmbedding) -> Self {
        Self { tile, embedding }
    }

    /// Returns the terrain tile identity.
    #[must_use]
    pub const fn tile(&self) -> TileKey {
        self.tile.tile()
    }

    /// Returns the baked terrain tile.
    #[must_use]
    pub const fn heightfield(&self) -> &HeightfieldTile {
        &self.tile
    }

    /// Returns the planar physics embedding.
    #[must_use]
    pub const fn embedding(&self) -> PlanarHeightfieldEmbedding {
        self.embedding
    }
}

/// A decoder-owned, validated terrain snapshot for one world cell.
#[derive(Clone, Debug)]
pub struct TerrainCellSnapshot {
    sector: SectorId,
    heightfields: Vec<DecodedTerrainHeightfield>,
}

impl TerrainCellSnapshot {
    /// Creates a snapshot and rejects duplicate tile identities.
    pub fn new(
        sector: SectorId,
        mut heightfields: Vec<DecodedTerrainHeightfield>,
    ) -> Result<Self, TerrainCellSnapshotError> {
        heightfields.sort_unstable_by_key(DecodedTerrainHeightfield::tile);
        for pair in heightfields.windows(2) {
            if pair[0].tile() == pair[1].tile() {
                return Err(TerrainCellSnapshotError::DuplicateTile {
                    tile: pair[0].tile(),
                });
            }
        }
        Ok(Self {
            sector,
            heightfields,
        })
    }

    /// Returns the owning cell identity.
    #[must_use]
    pub const fn sector(&self) -> SectorId {
        self.sector
    }

    /// Returns decoded heightfields in stable tile order.
    #[must_use]
    pub fn heightfields(&self) -> &[DecodedTerrainHeightfield] {
        &self.heightfields
    }
}

/// Error raised while validating a decoded terrain cell snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerrainCellSnapshotError {
    /// Two decoded heightfields claim the same tile identity.
    DuplicateTile {
        /// The duplicated tile.
        tile: TileKey,
    },
}

impl fmt::Display for TerrainCellSnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateTile { tile } => {
                write!(f, "decoded terrain cell duplicates tile {tile:?}")
            }
        }
    }
}

impl std::error::Error for TerrainCellSnapshotError {}

/// Decodes a cell payload into terrain-derived physics inputs.
pub trait TerrainCellDecoder {
    /// Decoder-specific failure type.
    type Error;

    /// Decodes and validates a complete cell snapshot without mutating a host.
    fn decode(
        &mut self,
        manifest: &CellManifest,
        payload: &[u8],
    ) -> Result<TerrainCellSnapshot, Self::Error>;
}

/// Failure raised by [`PhysicsCellHost`] before or during a cell transition.
#[derive(Debug)]
pub enum PhysicsCellHostError<D, H> {
    /// The decoder rejected the payload.
    Decoder(D),
    /// The decoder returned a snapshot for another sector.
    ManifestSectorMismatch {
        /// Sector requested by the manifest.
        manifest: SectorId,
        /// Sector returned by the decoder.
        decoded: SectorId,
    },
    /// The same tile is already owned by another resident cell.
    TileOwnedByOtherCell {
        /// Conflicting tile identity.
        tile: TileKey,
        /// Existing owning sector.
        owner: SectorId,
    },
    /// A staged transition found an inconsistent internal tile index.
    ResidencyInvariant {
        /// Cell whose index was inconsistent.
        sector: SectorId,
        /// Tile missing from the residency set.
        tile: TileKey,
    },
    /// The physics bridge rejected the staged snapshot.
    Physics(HeightfieldBridgeError),
    /// The adapter could not reserve the bounded cell tile index.
    AllocationFailed,
    /// The downstream scene host rejected the transition.
    Host(H),
}

impl<D: fmt::Display, H: fmt::Display> fmt::Display for PhysicsCellHostError<D, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decoder(error) => write!(f, "terrain cell decode failed: {error}"),
            Self::ManifestSectorMismatch { manifest, decoded } => {
                write!(
                    f,
                    "terrain cell sector mismatch: manifest {manifest:?}, decoded {decoded:?}"
                )
            }
            Self::TileOwnedByOtherCell { tile, owner } => {
                write!(f, "terrain tile {tile:?} already belongs to cell {owner:?}")
            }
            Self::ResidencyInvariant { sector, tile } => {
                write!(
                    f,
                    "physics residency lost tile {tile:?} for cell {sector:?}"
                )
            }
            Self::Physics(error) => write!(f, "physics heightfield staging failed: {error}"),
            Self::AllocationFailed => f.write_str("physics cell staging allocation failed"),
            Self::Host(error) => write!(f, "scene host transition failed: {error}"),
        }
    }
}

impl<D: std::error::Error + 'static, H: std::error::Error + 'static> std::error::Error
    for PhysicsCellHostError<D, H>
{
}

/// Scene-host decorator that atomically stages terrain physics state.
pub struct PhysicsCellHost<'a, H: SceneHost, D: TerrainCellDecoder> {
    host: &'a mut H,
    decoder: D,
    residency: WorldHeightfieldPhysicsResidency,
    cell_tiles: BTreeMap<SectorId, Vec<TileKey>>,
    tile_owners: BTreeMap<TileKey, SectorId>,
}

impl<'a, H: SceneHost, D: TerrainCellDecoder> PhysicsCellHost<'a, H, D> {
    /// Creates an empty adapter with bounded per-tile and aggregate budgets.
    #[must_use]
    pub fn new(
        host: &'a mut H,
        decoder: D,
        per_tile_budget: HeightfieldBridgeBudget,
        residency_budget: HeightfieldResidencyBudget,
    ) -> Self {
        Self {
            host,
            decoder,
            residency: WorldHeightfieldPhysicsResidency::new(per_tile_budget, residency_budget),
            cell_tiles: BTreeMap::new(),
            tile_owners: BTreeMap::new(),
        }
    }

    /// Returns the currently staged physics residency.
    #[must_use]
    pub const fn residency(&self) -> &WorldHeightfieldPhysicsResidency {
        &self.residency
    }

    /// Returns the decoder for configuration or statistics.
    #[must_use]
    pub const fn decoder(&self) -> &D {
        &self.decoder
    }

    /// Returns mutable decoder access.
    pub fn decoder_mut(&mut self) -> &mut D {
        &mut self.decoder
    }

    fn stage_cell(
        &self,
        manifest: &CellManifest,
        decoded: &TerrainCellSnapshot,
    ) -> Result<
        (
            WorldHeightfieldPhysicsResidency,
            BTreeMap<SectorId, Vec<TileKey>>,
            BTreeMap<TileKey, SectorId>,
        ),
        PhysicsCellHostError<D::Error, H::Error>,
    > {
        if decoded.sector() != manifest.sector {
            return Err(PhysicsCellHostError::ManifestSectorMismatch {
                manifest: manifest.sector,
                decoded: decoded.sector(),
            });
        }
        let mut residency = self.residency.clone();
        let mut cell_tiles = self.cell_tiles.clone();
        let mut tile_owners = self.tile_owners.clone();
        if let Some(old_tiles) = cell_tiles.remove(&manifest.sector) {
            for tile in old_tiles {
                if residency.evict(tile).is_none() {
                    return Err(PhysicsCellHostError::ResidencyInvariant {
                        sector: manifest.sector,
                        tile,
                    });
                }
                if tile_owners.remove(&tile) != Some(manifest.sector) {
                    return Err(PhysicsCellHostError::ResidencyInvariant {
                        sector: manifest.sector,
                        tile,
                    });
                }
            }
        }
        let mut next_tiles = Vec::new();
        next_tiles
            .try_reserve(decoded.heightfields().len())
            .map_err(|_| PhysicsCellHostError::AllocationFailed)?;
        for input in decoded.heightfields() {
            if let Some(&owner) = tile_owners.get(&input.tile()) {
                return Err(PhysicsCellHostError::TileOwnedByOtherCell {
                    tile: input.tile(),
                    owner,
                });
            }
            residency
                .publish(input.heightfield(), input.embedding())
                .map_err(PhysicsCellHostError::Physics)?;
            next_tiles.push(input.tile());
            tile_owners.insert(input.tile(), manifest.sector);
        }
        cell_tiles.insert(manifest.sector, next_tiles);
        Ok((residency, cell_tiles, tile_owners))
    }
}

impl<H: SceneHost, D: TerrainCellDecoder> SceneHost for PhysicsCellHost<'_, H, D> {
    type Error = PhysicsCellHostError<D::Error, H::Error>;

    fn load_cell(&mut self, manifest: &CellManifest, payload: &[u8]) -> Result<(), Self::Error> {
        let decoded = self
            .decoder
            .decode(manifest, payload)
            .map_err(PhysicsCellHostError::Decoder)?;
        let (residency, cell_tiles, tile_owners) = self.stage_cell(manifest, &decoded)?;
        self.host
            .load_cell(manifest, payload)
            .map_err(PhysicsCellHostError::Host)?;
        self.residency = residency;
        self.cell_tiles = cell_tiles;
        self.tile_owners = tile_owners;
        Ok(())
    }

    fn evict_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        let mut residency = self.residency.clone();
        let mut cell_tiles = self.cell_tiles.clone();
        let mut tile_owners = self.tile_owners.clone();
        if let Some(tiles) = cell_tiles.remove(&sector) {
            for tile in tiles {
                if residency.evict(tile).is_none() {
                    return Err(PhysicsCellHostError::ResidencyInvariant { sector, tile });
                }
                if tile_owners.remove(&tile) != Some(sector) {
                    return Err(PhysicsCellHostError::ResidencyInvariant { sector, tile });
                }
            }
        }
        self.host
            .evict_cell(sector)
            .map_err(PhysicsCellHostError::Host)?;
        self.residency = residency;
        self.cell_tiles = cell_tiles;
        self.tile_owners = tile_owners;
        Ok(())
    }
}

/// Returns a staged snapshot for a tile, if it is resident.
#[must_use]
pub fn resident_tile<'a>(
    residency: &'a WorldHeightfieldPhysicsResidency,
    tile: TileKey,
) -> Option<&'a PhysicsHeightfieldSnapshot> {
    residency.snapshot(tile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_world_stream::{AabbI64, DataLayerMask, GlobalPos};
    use prism_world_terrain::{BakeLimits, CubeFace, DeterministicHeightSampler, HeightfieldSpec};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct TestError(&'static str);

    impl fmt::Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }

    impl std::error::Error for TestError {}

    struct Host {
        reject_load: bool,
        reject_evict: bool,
        loads: usize,
        evictions: usize,
    }

    impl SceneHost for Host {
        type Error = TestError;

        fn load_cell(
            &mut self,
            _manifest: &CellManifest,
            _payload: &[u8],
        ) -> Result<(), Self::Error> {
            if self.reject_load {
                return Err(TestError("load rejected"));
            }
            self.loads += 1;
            Ok(())
        }

        fn evict_cell(&mut self, _sector: SectorId) -> Result<(), Self::Error> {
            if self.reject_evict {
                return Err(TestError("eviction rejected"));
            }
            self.evictions += 1;
            Ok(())
        }
    }

    struct Decoder {
        snapshot: TerrainCellSnapshot,
        reject: bool,
    }

    impl TerrainCellDecoder for Decoder {
        type Error = TestError;

        fn decode(
            &mut self,
            _manifest: &CellManifest,
            _payload: &[u8],
        ) -> Result<TerrainCellSnapshot, Self::Error> {
            if self.reject {
                Err(TestError("decode rejected"))
            } else {
                Ok(self.snapshot.clone())
            }
        }
    }

    fn manifest(x: i64) -> CellManifest {
        CellManifest {
            sector: SectorId::new(0, x, 0, 0),
            bounds: AabbI64::new(GlobalPos::new(x, 0, 0), GlobalPos::new(x + 1, 1, 1))
                .expect("valid bounds"),
            scene_key: 1,
            content_version: 1,
            derived_hash: 1,
            io_bytes: 1,
            data_layers: DataLayerMask::EMPTY,
        }
    }

    fn terrain(tile: TileKey) -> DecodedTerrainHeightfield {
        let spec = HeightfieldSpec::new(3, 10).expect("valid heightfield spec");
        let field = HeightfieldTile::bake(
            tile,
            spec,
            &DeterministicHeightSampler::new(7, 10),
            BakeLimits {
                max_samples: 9,
                max_bytes: 18,
            },
        )
        .expect("valid baked field");
        DecodedTerrainHeightfield::new(
            field,
            PlanarHeightfieldEmbedding {
                sample_origin_mm: [0, 0, 0],
                sample_spacing_mm: 500,
            },
        )
    }

    fn adapter(host: &mut Host, decoder: Decoder) -> PhysicsCellHost<'_, Host, Decoder> {
        PhysicsCellHost::new(
            host,
            decoder,
            HeightfieldBridgeBudget::new(9, 68).expect("valid per-tile budget"),
            HeightfieldResidencyBudget::new(2, 18, 136).expect("valid residency budget"),
        )
    }

    #[test]
    fn load_stages_physics_only_after_host_success_and_reload_failure_keeps_old_state() {
        let sector = SectorId::new(0, 0, 0, 0);
        let first = terrain(TileKey::root(CubeFace::PositiveY));
        let second = terrain(TileKey::root(CubeFace::NegativeY));
        let decoder = Decoder {
            snapshot: TerrainCellSnapshot::new(sector, vec![first]).unwrap(),
            reject: false,
        };
        let mut host = Host {
            reject_load: false,
            reject_evict: false,
            loads: 0,
            evictions: 0,
        };
        let mut adapter = adapter(&mut host, decoder);
        adapter.load_cell(&manifest(0), b"first").unwrap();
        assert!(adapter
            .residency()
            .snapshot(TileKey::root(CubeFace::PositiveY))
            .is_some());

        adapter.decoder_mut().snapshot = TerrainCellSnapshot::new(sector, vec![second]).unwrap();
        adapter.host.reject_load = true;
        assert!(matches!(
            adapter.load_cell(&manifest(0), b"reload"),
            Err(PhysicsCellHostError::Host(TestError("load rejected")))
        ));
        assert!(adapter
            .residency()
            .snapshot(TileKey::root(CubeFace::PositiveY))
            .is_some());
        assert!(adapter
            .residency()
            .snapshot(TileKey::root(CubeFace::NegativeY))
            .is_none());
    }

    #[test]
    fn decoder_and_manifest_failures_do_not_call_host_or_mutate_residency() {
        let sector = SectorId::new(0, 0, 0, 0);
        let decoder = Decoder {
            snapshot: TerrainCellSnapshot::new(
                SectorId::new(0, 9, 0, 0),
                vec![terrain(TileKey::root(CubeFace::PositiveY))],
            )
            .unwrap(),
            reject: false,
        };
        let mut host = Host {
            reject_load: false,
            reject_evict: false,
            loads: 0,
            evictions: 0,
        };
        let mut adapter = adapter(&mut host, decoder);
        assert!(matches!(
            adapter.load_cell(&manifest(0), b"bad-sector"),
            Err(PhysicsCellHostError::ManifestSectorMismatch { manifest: actual, decoded })
                if actual == sector && decoded == SectorId::new(0, 9, 0, 0)
        ));
        assert_eq!(adapter.host.loads, 0);
        assert!(adapter.residency().snapshots().is_empty());
    }

    #[test]
    fn eviction_failure_keeps_physics_snapshot_and_success_removes_it() {
        let tile = TileKey::root(CubeFace::PositiveY);
        let sector = SectorId::new(0, 0, 0, 0);
        let decoder = Decoder {
            snapshot: TerrainCellSnapshot::new(sector, vec![terrain(tile)]).unwrap(),
            reject: false,
        };
        let mut host = Host {
            reject_load: false,
            reject_evict: false,
            loads: 0,
            evictions: 0,
        };
        let mut adapter = adapter(&mut host, decoder);
        adapter.load_cell(&manifest(0), b"payload").unwrap();
        adapter.host.reject_evict = true;
        assert!(matches!(
            adapter.evict_cell(sector),
            Err(PhysicsCellHostError::Host(TestError("eviction rejected")))
        ));
        assert!(adapter.residency().snapshot(tile).is_some());
        adapter.host.reject_evict = false;
        adapter.evict_cell(sector).unwrap();
        assert!(adapter.residency().snapshot(tile).is_none());
        assert_eq!(host.evictions, 1);
    }
}
