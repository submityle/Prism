//! Atomic cell-lifecycle adapter for terrain-derived physics heightfields.
//!
//! The world runtime owns cell streaming and the scene host owns authoritative
//! content. This crate adds an optional physics consumer at that boundary:
//! payloads are decoded and fully staged before the downstream [`SceneHost`]
//! is called, and a failed host operation leaves the previous physics snapshot
//! unchanged. Physics data remains disposable derived state and can be evicted
//! independently of the authoritative cell payload.
//!
//! Sparse SDF cells use [`SdfPhysicsCellHost`] with [`SdfCellCodec`]. The optional
//! `tasks` feature adds `SdfCellRebakeRuntime`: it runs bounded local rebakes on
//! a caller-owned pool and publishes only on explicit owner collection. Queries
//! retain the previous revision while work is pending; successful reloads and
//! evictions invalidate that cell's old work without waiting for its evaluator.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

mod codec;
mod sdf;
mod sdf_codec;

#[cfg(feature = "tasks")]
pub use sdf::{
    SdfCellRebakeBudget, SdfCellRebakeError, SdfCellRebakeId, SdfCellRebakeResource,
    SdfCellRebakeRuntime,
};

pub use codec::{
    TerrainCellCodec, TerrainCellCodecError, TerrainCellCodecLimits, TERRAIN_CELL_CODEC_VERSION,
};
pub use sdf::{
    SdfCellDecoder, SdfCellSnapshot, SdfPhysicsCellHost, SdfPhysicsCellHostError,
    SdfResidencyBudget,
};
pub use sdf_codec::{
    SdfCellCodec, SdfCellCodecError, SdfCellCodecLimits, SdfCellCodecResource,
    SDF_CELL_CODEC_VERSION,
};

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

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
        (WorldHeightfieldPhysicsResidency, Vec<TileKey>),
        PhysicsCellHostError<D::Error, H::Error>,
    > {
        if decoded.sector() != manifest.sector {
            return Err(PhysicsCellHostError::ManifestSectorMismatch {
                manifest: manifest.sector,
                decoded: decoded.sector(),
            });
        }
        let old_tiles = self
            .cell_tiles
            .get(&manifest.sector)
            .map(Vec::as_slice)
            .unwrap_or_default();

        // Validate against committed residency before evicting old tiles from
        // the staged copy. Otherwise a same-cell reload could erase the only
        // revision floor before checking that its replacement is newer.
        for &tile in old_tiles {
            if self.residency.snapshot(tile).is_none()
                || self.tile_owners.get(&tile).copied() != Some(manifest.sector)
            {
                return Err(PhysicsCellHostError::ResidencyInvariant {
                    sector: manifest.sector,
                    tile,
                });
            }
        }
        for input in decoded.heightfields() {
            if let Some(&owner) = self.tile_owners.get(&input.tile()) {
                if owner != manifest.sector {
                    return Err(PhysicsCellHostError::TileOwnedByOtherCell {
                        tile: input.tile(),
                        owner,
                    });
                }
                if old_tiles.binary_search(&input.tile()).is_err() {
                    return Err(PhysicsCellHostError::ResidencyInvariant {
                        sector: manifest.sector,
                        tile: input.tile(),
                    });
                }
                let current = self.residency.snapshot(input.tile()).ok_or(
                    PhysicsCellHostError::ResidencyInvariant {
                        sector: manifest.sector,
                        tile: input.tile(),
                    },
                )?;
                let incoming = input.heightfield().revision();
                if incoming <= current.revision() {
                    return Err(PhysicsCellHostError::Physics(
                        HeightfieldBridgeError::StaleRevision {
                            current: current.revision(),
                            incoming,
                        },
                    ));
                }
            }
        }

        let mut next_tiles = Vec::new();
        next_tiles
            .try_reserve(decoded.heightfields().len())
            .map_err(|_| PhysicsCellHostError::AllocationFailed)?;

        // The height samples are the only deep clone. Its cost is capped by the
        // explicit aggregate residency budget; the two global ownership indexes
        // are validated in place and committed as a per-cell delta after the
        // downstream host succeeds.
        let mut residency = self.residency.clone();
        for &tile in old_tiles {
            if residency.evict(tile).is_none() {
                return Err(PhysicsCellHostError::ResidencyInvariant {
                    sector: manifest.sector,
                    tile,
                });
            }
        }

        for input in decoded.heightfields() {
            residency
                .publish(input.heightfield(), input.embedding())
                .map_err(PhysicsCellHostError::Physics)?;
            next_tiles.push(input.tile());
        }
        Ok((residency, next_tiles))
    }

    fn commit_cell(
        &mut self,
        sector: SectorId,
        residency: WorldHeightfieldPhysicsResidency,
        next_tiles: Vec<TileKey>,
    ) {
        if let Some(old_tiles) = self.cell_tiles.remove(&sector) {
            for tile in old_tiles {
                let previous_owner = self.tile_owners.remove(&tile);
                debug_assert_eq!(previous_owner, Some(sector));
            }
        }
        for &tile in &next_tiles {
            let previous_owner = self.tile_owners.insert(tile, sector);
            debug_assert!(previous_owner.is_none());
        }
        self.cell_tiles.insert(sector, next_tiles);
        self.residency = residency;
    }
}

impl<H: SceneHost, D: TerrainCellDecoder> SceneHost for PhysicsCellHost<'_, H, D> {
    type Error = PhysicsCellHostError<D::Error, H::Error>;

    fn load_cell(&mut self, manifest: &CellManifest, payload: &[u8]) -> Result<(), Self::Error> {
        let decoded = self
            .decoder
            .decode(manifest, payload)
            .map_err(PhysicsCellHostError::Decoder)?;
        let (residency, next_tiles) = self.stage_cell(manifest, &decoded)?;
        self.host
            .load_cell(manifest, payload)
            .map_err(PhysicsCellHostError::Host)?;
        self.commit_cell(manifest.sector, residency, next_tiles);
        Ok(())
    }

    fn evict_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        let mut residency = self.residency.clone();
        if let Some(tiles) = self.cell_tiles.get(&sector) {
            for &tile in tiles {
                if residency.evict(tile).is_none()
                    || self.tile_owners.get(&tile).copied() != Some(sector)
                {
                    return Err(PhysicsCellHostError::ResidencyInvariant { sector, tile });
                }
            }
        }
        self.host
            .evict_cell(sector)
            .map_err(PhysicsCellHostError::Host)?;
        if let Some(tiles) = self.cell_tiles.remove(&sector) {
            for tile in tiles {
                let previous_owner = self.tile_owners.remove(&tile);
                debug_assert_eq!(previous_owner, Some(sector));
            }
        }
        self.residency = residency;
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

    #[derive(Default)]
    struct Host {
        reject_load: bool,
        reject_evict: bool,
        panic_load: bool,
        panic_evict: bool,
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
            assert!(!self.panic_load, "load panicked");
            if self.reject_load {
                return Err(TestError("load rejected"));
            }
            self.loads += 1;
            Ok(())
        }

        fn evict_cell(&mut self, _sector: SectorId) -> Result<(), Self::Error> {
            assert!(!self.panic_evict, "eviction panicked");
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

    fn terrain_revision(tile: TileKey, revision: u64) -> DecodedTerrainHeightfield {
        let field = HeightfieldTile::from_encoded_samples(
            tile,
            HeightfieldSpec::new(3, 10).expect("valid heightfield spec"),
            revision,
            [0_i16; 9].into_iter(),
            BakeLimits {
                max_samples: 9,
                max_bytes: 18,
            },
        )
        .expect("valid encoded heightfield");
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
        let mut host = Host::default();
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
    fn same_tile_reload_rejects_equal_or_older_revision_before_host_call() {
        let sector = SectorId::new(0, 0, 0, 0);
        let tile = TileKey::root(CubeFace::PositiveY);
        let decoder = Decoder {
            snapshot: TerrainCellSnapshot::new(sector, vec![terrain_revision(tile, 10)]).unwrap(),
            reject: false,
        };
        let mut host = Host::default();
        let mut adapter = adapter(&mut host, decoder);
        adapter.load_cell(&manifest(0), b"revision-10").unwrap();
        assert_eq!(adapter.residency().snapshot(tile).unwrap().revision(), 10);
        let old_snapshot = adapter.residency().snapshot(tile).unwrap().clone();

        adapter.decoder_mut().snapshot =
            TerrainCellSnapshot::new(sector, vec![terrain_revision(tile, 9)]).unwrap();
        assert!(matches!(
            adapter.load_cell(&manifest(0), b"revision-9"),
            Err(PhysicsCellHostError::Physics(
                HeightfieldBridgeError::StaleRevision {
                    current: 10,
                    incoming: 9,
                }
            ))
        ));
        assert_eq!(adapter.host.loads, 1);
        assert_eq!(adapter.residency().snapshot(tile), Some(&old_snapshot));

        adapter.decoder_mut().snapshot =
            TerrainCellSnapshot::new(sector, vec![terrain_revision(tile, 10)]).unwrap();
        assert!(matches!(
            adapter.load_cell(&manifest(0), b"revision-10-equal"),
            Err(PhysicsCellHostError::Physics(
                HeightfieldBridgeError::StaleRevision {
                    current: 10,
                    incoming: 10,
                }
            ))
        ));
        assert_eq!(adapter.host.loads, 1);
        assert_eq!(adapter.residency().snapshot(tile), Some(&old_snapshot));
    }

    #[test]
    fn decoder_and_manifest_failures_do_not_call_host_or_mutate_residency() {
        let sector = SectorId::new(0, 0, 0, 0);
        let tile = TileKey::root(CubeFace::PositiveY);
        let decoder = Decoder {
            snapshot: TerrainCellSnapshot::new(sector, vec![terrain(tile)]).unwrap(),
            reject: false,
        };
        let mut host = Host::default();
        let mut adapter = adapter(&mut host, decoder);
        adapter.load_cell(&manifest(0), b"good").unwrap();
        adapter.decoder_mut().reject = true;
        assert!(matches!(
            adapter.load_cell(&manifest(0), b"decode-error"),
            Err(PhysicsCellHostError::Decoder(TestError("decode rejected")))
        ));
        assert_eq!(adapter.host.loads, 1);
        assert!(adapter.residency().snapshot(tile).is_some());

        adapter.decoder_mut().reject = false;
        adapter.decoder_mut().snapshot = TerrainCellSnapshot::new(
            SectorId::new(0, 9, 0, 0),
            vec![terrain(TileKey::root(CubeFace::NegativeY))],
        )
        .unwrap();
        assert!(matches!(
            adapter.load_cell(&manifest(0), b"bad-sector"),
            Err(PhysicsCellHostError::ManifestSectorMismatch { manifest: actual, decoded })
                if actual == sector && decoded == SectorId::new(0, 9, 0, 0)
        ));
        assert_eq!(adapter.host.loads, 1);
        assert_eq!(adapter.residency().snapshots().len(), 1);
        assert!(adapter.residency().snapshot(tile).is_some());
    }

    #[test]
    fn eviction_failure_keeps_physics_snapshot_and_success_removes_it() {
        let tile = TileKey::root(CubeFace::PositiveY);
        let sector = SectorId::new(0, 0, 0, 0);
        let decoder = Decoder {
            snapshot: TerrainCellSnapshot::new(sector, vec![terrain(tile)]).unwrap(),
            reject: false,
        };
        let mut host = Host::default();
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

    #[test]
    fn snapshot_rejects_duplicate_tiles_and_orders_distinct_tiles() {
        let sector = SectorId::new(0, 0, 0, 0);
        let positive = TileKey::root(CubeFace::PositiveY);
        let negative = TileKey::root(CubeFace::NegativeY);
        assert!(matches!(
            TerrainCellSnapshot::new(sector, vec![terrain(positive), terrain(positive)]),
            Err(TerrainCellSnapshotError::DuplicateTile { tile }) if tile == positive
        ));

        let snapshot =
            TerrainCellSnapshot::new(sector, vec![terrain(positive), terrain(negative)]).unwrap();
        let tiles = snapshot
            .heightfields()
            .iter()
            .map(DecodedTerrainHeightfield::tile)
            .collect::<Vec<_>>();
        let mut expected = vec![positive, negative];
        expected.sort_unstable();
        assert_eq!(tiles, expected);
    }

    #[test]
    fn successful_reload_replaces_only_the_cells_previous_tiles() {
        let sector = SectorId::new(0, 0, 0, 0);
        let first = TileKey::root(CubeFace::PositiveY);
        let second = TileKey::root(CubeFace::NegativeY);
        let decoder = Decoder {
            snapshot: TerrainCellSnapshot::new(sector, vec![terrain(first)]).unwrap(),
            reject: false,
        };
        let mut host = Host::default();
        let mut adapter = adapter(&mut host, decoder);
        adapter.load_cell(&manifest(0), b"first").unwrap();
        adapter.decoder_mut().snapshot =
            TerrainCellSnapshot::new(sector, vec![terrain(second)]).unwrap();
        adapter.load_cell(&manifest(0), b"second").unwrap();

        assert!(adapter.residency().snapshot(first).is_none());
        assert!(adapter.residency().snapshot(second).is_some());
        assert_eq!(adapter.host.loads, 2);
        assert_eq!(adapter.cell_tiles.get(&sector), Some(&vec![second]));
        assert_eq!(adapter.tile_owners.get(&second), Some(&sector));
    }

    #[test]
    fn cross_cell_ownership_conflict_preserves_residency_and_host_state() {
        let first_sector = SectorId::new(0, 0, 0, 0);
        let second_sector = SectorId::new(0, 1, 0, 0);
        let tile = TileKey::root(CubeFace::PositiveY);
        let decoder = Decoder {
            snapshot: TerrainCellSnapshot::new(first_sector, vec![terrain(tile)]).unwrap(),
            reject: false,
        };
        let mut host = Host::default();
        let mut adapter = adapter(&mut host, decoder);
        adapter.load_cell(&manifest(0), b"first").unwrap();
        adapter.decoder_mut().snapshot =
            TerrainCellSnapshot::new(second_sector, vec![terrain(tile)]).unwrap();

        assert!(matches!(
            adapter.load_cell(&manifest(1), b"conflict"),
            Err(PhysicsCellHostError::TileOwnedByOtherCell { tile: actual, owner })
                if actual == tile && owner == first_sector
        ));
        assert_eq!(adapter.host.loads, 1);
        assert!(adapter.residency().snapshot(tile).is_some());
        assert_eq!(adapter.tile_owners.get(&tile), Some(&first_sector));
        assert!(!adapter.cell_tiles.contains_key(&second_sector));
    }

    #[test]
    fn aggregate_budget_failure_preserves_previous_cell_snapshot() {
        let sector = SectorId::new(0, 0, 0, 0);
        let old = TileKey::root(CubeFace::PositiveY);
        let decoder = Decoder {
            snapshot: TerrainCellSnapshot::new(sector, vec![terrain(old)]).unwrap(),
            reject: false,
        };
        let mut host = Host::default();
        let mut adapter = adapter(&mut host, decoder);
        adapter.load_cell(&manifest(0), b"old").unwrap();
        adapter.decoder_mut().snapshot = TerrainCellSnapshot::new(
            sector,
            vec![
                terrain(TileKey::root(CubeFace::NegativeY)),
                terrain(TileKey::root(CubeFace::PositiveX)),
                terrain(TileKey::root(CubeFace::NegativeX)),
            ],
        )
        .unwrap();

        assert!(matches!(
            adapter.load_cell(&manifest(0), b"too-large"),
            Err(PhysicsCellHostError::Physics(
                HeightfieldBridgeError::ResidentTileCapacityExceeded {
                    required: 3,
                    capacity: 2
                }
            ))
        ));
        assert_eq!(adapter.host.loads, 1);
        assert_eq!(adapter.residency().snapshots().len(), 1);
        assert!(adapter.residency().snapshot(old).is_some());
    }

    #[test]
    fn multiple_cells_keep_stable_order_and_exact_aggregate_accounting() {
        let first_sector = SectorId::new(0, 0, 0, 0);
        let second_sector = SectorId::new(0, 1, 0, 0);
        let first = TileKey::root(CubeFace::PositiveY);
        let second = TileKey::root(CubeFace::NegativeY);
        let decoder = Decoder {
            snapshot: TerrainCellSnapshot::new(first_sector, vec![terrain(first)]).unwrap(),
            reject: false,
        };
        let mut host = Host::default();
        let mut adapter = adapter(&mut host, decoder);
        adapter.load_cell(&manifest(0), b"first").unwrap();
        adapter.decoder_mut().snapshot =
            TerrainCellSnapshot::new(second_sector, vec![terrain(second)]).unwrap();
        adapter.load_cell(&manifest(1), b"second").unwrap();

        let tiles = adapter
            .residency()
            .snapshots()
            .iter()
            .map(PhysicsHeightfieldSnapshot::tile)
            .collect::<Vec<_>>();
        let mut expected = vec![first, second];
        expected.sort_unstable();
        assert_eq!(tiles, expected);
        assert_eq!(adapter.residency().resident_samples(), 18);
        assert_eq!(adapter.residency().resident_upload_bytes(), 136);
    }

    #[test]
    fn empty_cell_and_unknown_eviction_have_explicit_host_semantics() {
        let sector = SectorId::new(0, 0, 0, 0);
        let unknown = SectorId::new(0, 9, 0, 0);
        let decoder = Decoder {
            snapshot: TerrainCellSnapshot::new(sector, Vec::new()).unwrap(),
            reject: false,
        };
        let mut host = Host::default();
        let mut adapter = adapter(&mut host, decoder);
        adapter.load_cell(&manifest(0), b"empty").unwrap();
        assert!(adapter.residency().snapshots().is_empty());
        assert_eq!(adapter.cell_tiles.get(&sector), Some(&Vec::new()));

        adapter.evict_cell(unknown).unwrap();
        assert_eq!(adapter.host.evictions, 1);
        assert!(adapter.cell_tiles.contains_key(&sector));
        adapter.evict_cell(sector).unwrap();
        assert_eq!(adapter.host.evictions, 2);
        assert!(!adapter.cell_tiles.contains_key(&sector));
    }

    #[test]
    fn downstream_panics_leave_committed_physics_state_unchanged() {
        let sector = SectorId::new(0, 0, 0, 0);
        let first = TileKey::root(CubeFace::PositiveY);
        let second = TileKey::root(CubeFace::NegativeY);
        let decoder = Decoder {
            snapshot: TerrainCellSnapshot::new(sector, vec![terrain(first)]).unwrap(),
            reject: false,
        };
        let mut host = Host::default();
        let mut adapter = adapter(&mut host, decoder);
        adapter.load_cell(&manifest(0), b"first").unwrap();
        adapter.decoder_mut().snapshot =
            TerrainCellSnapshot::new(sector, vec![terrain(second)]).unwrap();
        adapter.host.panic_load = true;
        let load_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = adapter.load_cell(&manifest(0), b"panic");
        }));
        assert!(load_panic.is_err());
        assert!(adapter.residency().snapshot(first).is_some());
        assert!(adapter.residency().snapshot(second).is_none());

        adapter.host.panic_load = false;
        adapter.host.panic_evict = true;
        let evict_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = adapter.evict_cell(sector);
        }));
        assert!(evict_panic.is_err());
        assert!(adapter.residency().snapshot(first).is_some());
        assert_eq!(adapter.tile_owners.get(&first), Some(&sector));
    }
}
