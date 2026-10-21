//! Atomic world-cell lifecycle for sparse-SDF physics queries.

use core::fmt;

use prism_world_physics_bridge::{
    SdfBounds, SdfBridgeError, SdfDistanceGradient, SdfPhysicsBridgeBudget, SdfPhysicsSnapshot,
    SdfQueryResult, WorldSdfPhysicsBridge, WorldSdfSnapshot,
};
use prism_world_residency::{SdfError, SdfQueryConsumer};
use prism_world_runtime::SceneHost;
use prism_world_stream::{CellManifest, SectorId};

#[cfg(feature = "tasks")]
mod rebake;
#[cfg(feature = "tasks")]
pub use rebake::{
    SdfCellRebakeBudget, SdfCellRebakeError, SdfCellRebakeId, SdfCellRebakeResource,
    SdfCellRebakeRuntime,
};

/// A decoder-owned sparse-SDF snapshot associated with one world cell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SdfCellSnapshot {
    sector: SectorId,
    snapshot: WorldSdfSnapshot,
}

impl SdfCellSnapshot {
    /// Creates a cell snapshot after associating it with a sector.
    #[must_use]
    pub const fn new(sector: SectorId, snapshot: WorldSdfSnapshot) -> Self {
        Self { sector, snapshot }
    }

    /// Returns the owning sector.
    #[must_use]
    pub const fn sector(&self) -> SectorId {
        self.sector
    }

    /// Returns the source revision.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.snapshot.revision()
    }

    /// Returns the owned world SDF snapshot.
    #[must_use]
    pub const fn snapshot(&self) -> &WorldSdfSnapshot {
        &self.snapshot
    }

    fn into_snapshot(self) -> WorldSdfSnapshot {
        self.snapshot
    }
}

/// Decoder for one complete sparse-SDF cell payload.
pub trait SdfCellDecoder {
    /// Decoder-specific failure type.
    type Error;

    /// Decodes a complete cell snapshot without mutating a resident host.
    fn decode(
        &mut self,
        manifest: &CellManifest,
        payload: &[u8],
    ) -> Result<SdfCellSnapshot, Self::Error>;
}

/// Aggregate limits for resident cell SDF snapshots.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfResidencyBudget {
    /// Maximum resident cells.
    pub max_cells: usize,
    /// Maximum resident allocated bricks.
    pub max_bricks: usize,
    /// Maximum resident quantized samples.
    pub max_samples: usize,
    /// Maximum resident logical bytes.
    pub max_bytes: usize,
}

impl SdfResidencyBudget {
    /// Creates aggregate limits. Zero limits admit only an empty residency.
    #[must_use]
    pub const fn new(
        max_cells: usize,
        max_bricks: usize,
        max_samples: usize,
        max_bytes: usize,
    ) -> Self {
        Self {
            max_cells,
            max_bricks,
            max_samples,
            max_bytes,
        }
    }
}

/// Failure raised before or during a sparse-SDF cell transition.
#[derive(Debug)]
pub enum SdfPhysicsCellHostError<D, H> {
    /// The decoder rejected the payload.
    Decoder(D),
    /// The decoded snapshot belongs to another sector.
    ManifestSectorMismatch {
        /// Sector declared by the manifest.
        manifest: SectorId,
        /// Sector returned by the decoder.
        decoded: SectorId,
    },
    /// A same-cell replacement was not newer than the resident snapshot.
    StaleRevision {
        /// Resident revision.
        current: u64,
        /// Candidate revision.
        incoming: u64,
    },
    /// Per-cell bridge validation rejected the candidate.
    Physics(SdfBridgeError),
    /// Resident cell count would exceed the aggregate bound.
    ResidentCellCapacityExceeded {
        /// Resident cells requested after replacement.
        required: usize,
        /// Maximum resident cells.
        capacity: usize,
    },
    /// Resident brick count would exceed the aggregate bound.
    ResidentBrickBudgetExceeded {
        /// Resident bricks requested after replacement.
        required: usize,
        /// Maximum resident bricks.
        limit: usize,
    },
    /// Resident sample count would exceed the aggregate bound.
    ResidentSampleBudgetExceeded {
        /// Resident samples requested after replacement.
        required: usize,
        /// Maximum resident samples.
        limit: usize,
    },
    /// Resident logical bytes would exceed the aggregate bound.
    ResidentByteBudgetExceeded {
        /// Logical bytes requested after replacement.
        required: usize,
        /// Maximum resident logical bytes.
        limit: usize,
    },
    /// Checked aggregate accounting overflowed.
    AccountingOverflow,
    /// The adapter could not reserve a resident-cell index slot.
    AllocationFailed,
    /// The downstream scene host rejected the cell transition.
    Host(H),
}

impl<D: fmt::Display, H: fmt::Display> fmt::Display for SdfPhysicsCellHostError<D, H> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decoder(error) => write!(formatter, "SDF cell decode failed: {error}"),
            Self::ManifestSectorMismatch { manifest, decoded } => write!(
                formatter,
                "SDF cell sector mismatch: manifest {manifest:?}, decoded {decoded:?}"
            ),
            Self::StaleRevision { current, incoming } => write!(
                formatter,
                "SDF cell revision {incoming} is not newer than {current}"
            ),
            Self::Physics(error) => write!(formatter, "SDF physics staging failed: {error}"),
            Self::ResidentCellCapacityExceeded { required, capacity } => write!(
                formatter,
                "SDF residency needs {required} cells; capacity is {capacity}"
            ),
            Self::ResidentBrickBudgetExceeded { required, limit } => write!(
                formatter,
                "SDF residency needs {required} bricks; limit is {limit}"
            ),
            Self::ResidentSampleBudgetExceeded { required, limit } => write!(
                formatter,
                "SDF residency needs {required} samples; limit is {limit}"
            ),
            Self::ResidentByteBudgetExceeded { required, limit } => write!(
                formatter,
                "SDF residency needs {required} bytes; limit is {limit}"
            ),
            Self::AccountingOverflow => formatter.write_str("SDF residency accounting overflow"),
            Self::AllocationFailed => formatter.write_str("SDF residency index allocation failed"),
            Self::Host(error) => write!(formatter, "scene host rejected SDF cell: {error}"),
        }
    }
}

impl<D: std::error::Error + 'static, H: std::error::Error + 'static> std::error::Error
    for SdfPhysicsCellHostError<D, H>
{
}

/// Scene-host decorator that atomically publishes sparse-SDF cell snapshots.
pub struct SdfPhysicsCellHost<'a, H: SceneHost, D: SdfCellDecoder> {
    host: &'a mut H,
    decoder: D,
    per_cell_budget: SdfPhysicsBridgeBudget,
    residency_budget: SdfResidencyBudget,
    residents: Vec<(SectorId, WorldSdfPhysicsBridge)>,
    resident_bricks: usize,
    resident_samples: usize,
    resident_bytes: usize,
}

impl<'a, H: SceneHost, D: SdfCellDecoder> SdfPhysicsCellHost<'a, H, D> {
    /// Creates an empty host with per-cell and aggregate SDF budgets.
    #[must_use]
    pub fn new(
        host: &'a mut H,
        decoder: D,
        per_cell_budget: SdfPhysicsBridgeBudget,
        residency_budget: SdfResidencyBudget,
    ) -> Self {
        Self {
            host,
            decoder,
            per_cell_budget,
            residency_budget,
            residents: Vec::new(),
            resident_bricks: 0,
            resident_samples: 0,
            resident_bytes: 0,
        }
    }

    /// Returns the decoder.
    #[must_use]
    pub const fn decoder(&self) -> &D {
        &self.decoder
    }

    /// Returns mutable decoder access.
    pub fn decoder_mut(&mut self) -> &mut D {
        &mut self.decoder
    }

    /// Returns the resident snapshot for one cell.
    #[must_use]
    pub fn resident(&self, sector: SectorId) -> Option<&SdfPhysicsSnapshot> {
        self.residents
            .binary_search_by_key(&sector, |(key, _)| *key)
            .ok()
            .and_then(|index| self.residents[index].1.current())
    }

    /// Returns aggregate resident `(cells, bricks, samples, bytes)`.
    #[must_use]
    pub fn resident_usage(&self) -> (usize, usize, usize, usize) {
        (
            self.residents.len(),
            self.resident_bricks,
            self.resident_samples,
            self.resident_bytes,
        )
    }

    /// Queries one resident cell. A missing cell is explicitly `Unknown`.
    pub fn query_sample(
        &self,
        sector: SectorId,
        voxel: [i32; 3],
        mip: u8,
    ) -> Result<SdfQueryResult<i16>, SdfError> {
        self.resident(sector)
            .map_or(Ok(SdfQueryResult::Unknown), |snapshot| {
                snapshot.query().query_sample(voxel, mip)
            })
    }

    /// Queries signed distance in millimetres at a world-space millimetre
    /// position. A missing cell or brick remains `Unknown`.
    pub fn query_world_distance_mm(
        &self,
        sector: SectorId,
        world_position_mm: [i64; 3],
        mip: u8,
    ) -> Result<SdfQueryResult<i64>, SdfBridgeError> {
        self.resident(sector)
            .map_or(Ok(SdfQueryResult::Unknown), |snapshot| {
                snapshot.query_world_distance_mm(world_position_mm, mip)
            })
    }

    /// Queries trilinear distance and its world-space derivative from one
    /// committed snapshot. A missing cell or interpolation corner is `Unknown`.
    /// Failed loads, reloads and evictions retain the previous query result.
    pub fn query_world_distance_gradient_mm(
        &self,
        sector: SectorId,
        world_position_mm: [i64; 3],
        mip: u8,
    ) -> Result<SdfQueryResult<SdfDistanceGradient>, SdfBridgeError> {
        self.resident(sector)
            .map_or(Ok(SdfQueryResult::Unknown), |snapshot| {
                snapshot.query_world_distance_gradient_mm(world_position_mm, mip)
            })
    }

    /// Queries conservative surface bounds for one resident cell.
    pub fn query_surface_region(
        &self,
        sector: SectorId,
        min_voxel: [i32; 3],
        max_voxel: [i32; 3],
        mip: u8,
    ) -> Result<SdfQueryResult<SdfBounds>, SdfError> {
        self.resident(sector)
            .map_or(Ok(SdfQueryResult::Unknown), |snapshot| {
                snapshot
                    .query()
                    .query_surface_region(min_voxel, max_voxel, mip)
            })
    }

    fn stage(
        &mut self,
        manifest: &CellManifest,
        decoded: SdfCellSnapshot,
    ) -> Result<
        (WorldSdfPhysicsBridge, (usize, usize, usize)),
        SdfPhysicsCellHostError<D::Error, H::Error>,
    > {
        if decoded.sector() != manifest.sector {
            return Err(SdfPhysicsCellHostError::ManifestSectorMismatch {
                manifest: manifest.sector,
                decoded: decoded.sector(),
            });
        }
        if let Some(current) = self.resident(manifest.sector)
            && decoded.revision() <= current.revision()
        {
            return Err(SdfPhysicsCellHostError::StaleRevision {
                current: current.revision(),
                incoming: decoded.revision(),
            });
        }
        let mut bridge = WorldSdfPhysicsBridge::new(self.per_cell_budget);
        let staged = bridge
            .publish(decoded.into_snapshot())
            .map_err(SdfPhysicsCellHostError::Physics)?;
        let old = self.resident(manifest.sector);
        let old_bricks = old.map_or(0, |value| value.grid().brick_count());
        let old_samples = old
            .map(|value| {
                value
                    .grid()
                    .sample_count()
                    .checked_mul(value.grid().brick_count())
                    .ok_or(SdfPhysicsCellHostError::AccountingOverflow)
            })
            .transpose()?
            .unwrap_or(0);
        let old_bytes = old.map_or(0, SdfPhysicsSnapshot::logical_bytes);
        let base_bricks = self
            .resident_bricks
            .checked_sub(old_bricks)
            .ok_or(SdfPhysicsCellHostError::AccountingOverflow)?;
        let base_samples = self
            .resident_samples
            .checked_sub(old_samples)
            .ok_or(SdfPhysicsCellHostError::AccountingOverflow)?;
        let base_bytes = self
            .resident_bytes
            .checked_sub(old_bytes)
            .ok_or(SdfPhysicsCellHostError::AccountingOverflow)?;
        let next_bricks = base_bricks
            .checked_add(staged.grid().brick_count())
            .ok_or(SdfPhysicsCellHostError::AccountingOverflow)?;
        let next_samples = base_samples
            .checked_add(
                staged
                    .grid()
                    .sample_count()
                    .checked_mul(staged.grid().brick_count())
                    .ok_or(SdfPhysicsCellHostError::AccountingOverflow)?,
            )
            .ok_or(SdfPhysicsCellHostError::AccountingOverflow)?;
        let next_bytes = base_bytes
            .checked_add(staged.logical_bytes())
            .ok_or(SdfPhysicsCellHostError::AccountingOverflow)?;
        if old.is_none() && self.residents.len() >= self.residency_budget.max_cells {
            return Err(SdfPhysicsCellHostError::ResidentCellCapacityExceeded {
                required: self.residents.len().saturating_add(1),
                capacity: self.residency_budget.max_cells,
            });
        }
        if next_bricks > self.residency_budget.max_bricks {
            return Err(SdfPhysicsCellHostError::ResidentBrickBudgetExceeded {
                required: next_bricks,
                limit: self.residency_budget.max_bricks,
            });
        }
        if next_samples > self.residency_budget.max_samples {
            return Err(SdfPhysicsCellHostError::ResidentSampleBudgetExceeded {
                required: next_samples,
                limit: self.residency_budget.max_samples,
            });
        }
        if next_bytes > self.residency_budget.max_bytes {
            return Err(SdfPhysicsCellHostError::ResidentByteBudgetExceeded {
                required: next_bytes,
                limit: self.residency_budget.max_bytes,
            });
        }
        if old.is_none() {
            self.residents
                .try_reserve(1)
                .map_err(|_| SdfPhysicsCellHostError::AllocationFailed)?;
        }
        Ok((bridge, (next_bricks, next_samples, next_bytes)))
    }

    fn commit(
        &mut self,
        sector: SectorId,
        staged: WorldSdfPhysicsBridge,
        usage: (usize, usize, usize),
    ) {
        match self
            .residents
            .binary_search_by_key(&sector, |(key, _)| *key)
        {
            Ok(index) => self.residents[index].1 = staged,
            Err(index) => self.residents.insert(index, (sector, staged)),
        }
        (
            self.resident_bricks,
            self.resident_samples,
            self.resident_bytes,
        ) = usage;
    }
}

impl<H: SceneHost, D: SdfCellDecoder> SceneHost for SdfPhysicsCellHost<'_, H, D> {
    type Error = SdfPhysicsCellHostError<D::Error, H::Error>;

    fn load_cell(&mut self, manifest: &CellManifest, payload: &[u8]) -> Result<(), Self::Error> {
        let decoded = self
            .decoder
            .decode(manifest, payload)
            .map_err(SdfPhysicsCellHostError::Decoder)?;
        let (staged, usage) = self.stage(manifest, decoded)?;
        self.host
            .load_cell(manifest, payload)
            .map_err(SdfPhysicsCellHostError::Host)?;
        self.commit(manifest.sector, staged, usage);
        Ok(())
    }

    fn evict_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        let index = self
            .residents
            .binary_search_by_key(&sector, |(key, _)| *key)
            .ok();
        let next_usage = if let Some(index) = index {
            let resident = self.residents[index]
                .1
                .current()
                .ok_or(SdfPhysicsCellHostError::AccountingOverflow)?;
            let bricks = resident.grid().brick_count();
            let samples = resident
                .grid()
                .sample_count()
                .checked_mul(bricks)
                .ok_or(SdfPhysicsCellHostError::AccountingOverflow)?;
            Some((
                self.resident_bricks
                    .checked_sub(bricks)
                    .ok_or(SdfPhysicsCellHostError::AccountingOverflow)?,
                self.resident_samples
                    .checked_sub(samples)
                    .ok_or(SdfPhysicsCellHostError::AccountingOverflow)?,
                self.resident_bytes
                    .checked_sub(resident.logical_bytes())
                    .ok_or(SdfPhysicsCellHostError::AccountingOverflow)?,
            ))
        } else {
            None
        };
        self.host
            .evict_cell(sector)
            .map_err(SdfPhysicsCellHostError::Host)?;
        if let (Some(index), Some((bricks, samples, bytes))) = (index, next_usage) {
            self.residents.remove(index);
            self.resident_bricks = bricks;
            self.resident_samples = samples;
            self.resident_bytes = bytes;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use prism_world_residency::{BrickCoord, SdfQueryResult, SparseSdfBrickGrid};
    use prism_world_stream::{AabbI64, DataLayerMask, GlobalPos};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct TestError(&'static str);

    impl fmt::Display for TestError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(self.0)
        }
    }

    impl std::error::Error for TestError {}

    #[derive(Default)]
    struct Host {
        reject_load: bool,
        reject_evict: bool,
        loads: usize,
        evictions: usize,
    }

    impl SceneHost for Host {
        type Error = TestError;

        fn load_cell(&mut self, _: &CellManifest, _: &[u8]) -> Result<(), Self::Error> {
            if self.reject_load {
                return Err(TestError("load rejected"));
            }
            self.loads += 1;
            Ok(())
        }

        fn evict_cell(&mut self, _: SectorId) -> Result<(), Self::Error> {
            if self.reject_evict {
                return Err(TestError("evict rejected"));
            }
            self.evictions += 1;
            Ok(())
        }
    }

    #[derive(Clone)]
    struct Decoder {
        snapshot: SdfCellSnapshot,
        reject: bool,
    }

    impl SdfCellDecoder for Decoder {
        type Error = TestError;

        fn decode(&mut self, _: &CellManifest, _: &[u8]) -> Result<SdfCellSnapshot, Self::Error> {
            if self.reject {
                Err(TestError("decode rejected"))
            } else {
                Ok(self.snapshot.clone())
            }
        }
    }

    fn sector(x: i64) -> SectorId {
        SectorId::new(0, x, 0, 0)
    }

    fn manifest(sector: SectorId) -> CellManifest {
        CellManifest {
            sector,
            bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1))
                .expect("valid bounds"),
            scene_key: 1,
            content_version: 1,
            derived_hash: 1,
            io_bytes: 1,
            data_layers: DataLayerMask::EMPTY,
        }
    }

    fn snapshot(sector: SectorId, revision: u64, value: i16) -> SdfCellSnapshot {
        let mut grid = SparseSdfBrickGrid::new(2, 1).expect("valid grid");
        grid.insert_brick(BrickCoord::new(-1, 0, 0, 0), vec![value; 8])
            .expect("valid brick");
        let source = WorldSdfSnapshot::new(
            prism_world_physics_bridge::WorldSdfTransform::new([0, 0, 0], 500, 2)
                .expect("valid transform"),
            revision,
            grid,
        );
        SdfCellSnapshot::new(sector, source)
    }

    fn host<'a>(world: &'a mut Host, decoder: Decoder) -> SdfPhysicsCellHost<'a, Host, Decoder> {
        SdfPhysicsCellHost::new(
            world,
            decoder,
            SdfPhysicsBridgeBudget::new(1, 8, 96).expect("per-cell budget"),
            SdfResidencyBudget::new(2, 1, 8, 192),
        )
    }

    #[test]
    fn load_query_and_missing_cell_preserve_unknown() {
        let first_sector = sector(0);
        let mut world = Host::default();
        let mut adapter = host(
            &mut world,
            Decoder {
                snapshot: snapshot(first_sector, 1, -3),
                reject: false,
            },
        );
        adapter
            .load_cell(&manifest(first_sector), b"cell")
            .expect("load");
        assert_eq!(adapter.resident_usage(), (1, 1, 8, 80));
        assert_eq!(
            adapter.query_sample(first_sector, [-2, 0, 0], 0),
            Ok(SdfQueryResult::Value(-3))
        );
        assert_eq!(
            adapter.query_world_distance_mm(first_sector, [-1_000, 0, 0], 0),
            Ok(SdfQueryResult::Value(-6))
        );
        assert_eq!(
            adapter.query_sample(sector(1), [-2, 0, 0], 0),
            Ok(SdfQueryResult::Unknown)
        );
    }

    #[test]
    fn host_failure_and_decode_failure_keep_old_snapshot() {
        let sector = sector(0);
        let mut world = Host::default();
        let mut adapter = host(
            &mut world,
            Decoder {
                snapshot: snapshot(sector, 1, -1),
                reject: false,
            },
        );
        adapter
            .load_cell(&manifest(sector), b"first")
            .expect("load");
        adapter.decoder_mut().snapshot = snapshot(sector, 2, -2);
        adapter.host.reject_load = true;
        assert!(matches!(
            adapter.load_cell(&manifest(sector), b"reload"),
            Err(SdfPhysicsCellHostError::Host(TestError("load rejected")))
        ));
        assert_eq!(adapter.resident(sector).expect("resident").revision(), 1);
        adapter.host.reject_load = false;
        adapter.decoder_mut().reject = true;
        assert!(matches!(
            adapter.load_cell(&manifest(sector), b"decode"),
            Err(SdfPhysicsCellHostError::Decoder(TestError(
                "decode rejected"
            )))
        ));
        assert_eq!(adapter.resident(sector).expect("resident").revision(), 1);
    }

    #[test]
    fn revision_and_aggregate_budget_fail_before_host_call() {
        let first_sector = sector(0);
        let second = sector(1);
        let mut world = Host::default();
        let mut adapter = host(
            &mut world,
            Decoder {
                snapshot: snapshot(first_sector, 2, 1),
                reject: false,
            },
        );
        adapter
            .load_cell(&manifest(first_sector), b"first")
            .expect("load");
        adapter.decoder_mut().snapshot = snapshot(first_sector, 1, 2);
        assert!(matches!(
            adapter.load_cell(&manifest(first_sector), b"stale"),
            Err(SdfPhysicsCellHostError::StaleRevision {
                current: 2,
                incoming: 1
            })
        ));
        adapter.decoder_mut().snapshot = snapshot(second, 1, 3);
        assert!(matches!(
            adapter.load_cell(&manifest(second), b"budget"),
            Err(SdfPhysicsCellHostError::ResidentBrickBudgetExceeded {
                required: 2,
                limit: 1
            })
        ));
        assert_eq!(world.loads, 1);
    }

    #[test]
    fn failed_eviction_keeps_snapshot_and_success_removes_it() {
        let sector = sector(0);
        let mut world = Host::default();
        let mut adapter = host(
            &mut world,
            Decoder {
                snapshot: snapshot(sector, 1, 1),
                reject: false,
            },
        );
        adapter
            .load_cell(&manifest(sector), b"first")
            .expect("load");
        adapter.host.reject_evict = true;
        assert!(matches!(
            adapter.evict_cell(sector),
            Err(SdfPhysicsCellHostError::Host(TestError("evict rejected")))
        ));
        assert!(adapter.resident(sector).is_some());
        adapter.host.reject_evict = false;
        adapter.evict_cell(sector).expect("evict");
        assert_eq!(adapter.resident_usage(), (0, 0, 0, 0));
    }
}
