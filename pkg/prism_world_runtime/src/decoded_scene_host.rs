use alloc::collections::BTreeMap;
use alloc::string::ToString;
use core::fmt;

use prism_world_stream::{CellManifest, SectorId};

use crate::{FailureStage, SceneHost};

/// Independent limits for decoded cell payloads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CellDecodeLimits {
    /// Maximum logical bytes retained by one decoded cell.
    pub max_cell_bytes: u64,
    /// Maximum logical bytes retained across all committed decoded cells.
    pub max_resident_bytes: u64,
    /// Maximum resident cell count, including cells with a zero-byte charge.
    pub max_resident_cells: usize,
}

impl CellDecodeLimits {
    /// Creates decoded-cell limits without clamping caller values.
    #[must_use]
    pub const fn new(
        max_cell_bytes: u64,
        max_resident_bytes: u64,
        max_resident_cells: usize,
    ) -> Self {
        Self {
            max_cell_bytes,
            max_resident_bytes,
            max_resident_cells,
        }
    }
}

/// An application-decoded cell and its caller-accounted resident footprint.
pub struct DecodedCell<T> {
    value: T,
    resident_bytes: u64,
}

impl<T> DecodedCell<T> {
    /// Wraps a decoded value with its conservative logical resident-byte charge.
    ///
    /// The application decoder owns the accounting definition; it should include
    /// all cell-owned arrays and retained backing storage, but not shared data.
    #[must_use]
    pub const fn new(value: T, resident_bytes: u64) -> Self {
        Self {
            value,
            resident_bytes,
        }
    }

    /// Returns the logical resident-byte charge.
    #[must_use]
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Returns the decoded application value.
    #[must_use]
    pub fn value(&self) -> &T {
        &self.value
    }
}

/// Application-owned, format-specific cell payload decoder.
pub trait CellPayloadDecoder {
    /// Decoded value passed to the typed host.
    type Cell;
    /// Decoder-specific error.
    type Error: fmt::Display;

    /// Returns a conservative resident-byte charge before decoded allocations.
    ///
    /// This pass should inspect bounded wire metadata only. The adapter reserves
    /// against the returned charge before calling [`Self::decode`].
    fn preflight(&mut self, manifest: &CellManifest, payload: &[u8]) -> Result<u64, Self::Error>;

    /// Decodes a payload after the runtime has validated its manifest header.
    fn decode(
        &mut self,
        manifest: &CellManifest,
        payload: &[u8],
    ) -> Result<DecodedCell<Self::Cell>, Self::Error>;
}

/// Application-owned typed cell lifecycle.
pub trait DecodedCellHost {
    /// Application cell type.
    type Cell;
    /// Staged replacement that is not yet externally visible.
    type Prepared;
    /// Typed staging error.
    type Error: fmt::Display;

    /// Validates and stages a load/reload without changing visible resident state.
    fn prepare(
        &mut self,
        manifest: &CellManifest,
        cell: Self::Cell,
    ) -> Result<Self::Prepared, Self::Error>;

    /// Publishes a prepared replacement without failure or allocation.
    ///
    /// All validation and fallible allocation must happen in [`Self::prepare`].
    /// This method must not panic; it is the single publication point.
    fn commit(&mut self, prepared: Self::Prepared);

    /// Removes a resident cell. Failure must leave the old cell available.
    fn evict(&mut self, sector: SectorId) -> Result<(), Self::Error>;
}

/// Runtime [`SceneHost`] adapter for application-owned payload decoders and typed hosts.
pub struct DecodedSceneHost<D, H>
where
    D: CellPayloadDecoder,
    H: DecodedCellHost<Cell = D::Cell>,
{
    decoder: D,
    host: H,
    limits: CellDecodeLimits,
    resident: BTreeMap<SectorId, u64>,
    resident_bytes: u64,
}

/// Typed decoding, budget, prepare, commit, or eviction failure.
#[derive(Debug, Eq, PartialEq)]
pub enum DecodedSceneHostError<DecodeError, HostError> {
    /// Application schema rejected bytes.
    Decode(DecodeError),
    /// Per-cell decoded footprint exceeds the configured bound.
    CellBudget {
        /// Preflight logical bytes requested by the cell.
        required: u64,
        /// Maximum logical bytes allowed for one cell.
        limit: u64,
    },
    /// Aggregate resident decoded footprint exceeds the configured bound.
    ResidentBudget {
        /// Aggregate logical bytes requested after replacement.
        required: u64,
        /// Maximum aggregate logical bytes allowed.
        limit: u64,
    },
    /// Resident-byte accounting overflowed.
    AccountingOverflow,
    /// Resident cell count exceeds the configured bound.
    ResidentCellBudget {
        /// Resident cell count requested after replacement.
        required: usize,
        /// Maximum resident cell count.
        limit: usize,
    },
    /// The decoder reported a footprint larger than the preflight reservation.
    PreflightUnderestimated {
        /// Actual decoded logical bytes.
        actual: u64,
        /// Bytes admitted by preflight.
        reserved: u64,
    },
    /// Typed host staging rejected the candidate cell.
    Prepare(HostError),
    /// Typed host rejected eviction.
    Evict(HostError),
}

impl<DecodeError: fmt::Display, HostError: fmt::Display> fmt::Display
    for DecodedSceneHostError<DecodeError, HostError>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(error) => write!(formatter, "cell payload decode failed: {error}"),
            Self::CellBudget { required, limit } => write!(
                formatter,
                "decoded cell requires {required} bytes; per-cell limit is {limit}"
            ),
            Self::ResidentBudget { required, limit } => write!(
                formatter,
                "decoded resident requires {required} bytes; aggregate limit is {limit}"
            ),
            Self::AccountingOverflow => formatter.write_str("decoded resident accounting overflow"),
            Self::ResidentCellBudget { required, limit } => write!(
                formatter,
                "decoded resident requires {required} cells; count limit is {limit}"
            ),
            Self::PreflightUnderestimated { actual, reserved } => write!(
                formatter,
                "decoder produced a {actual}-byte cell after reserving {reserved} bytes"
            ),
            Self::Prepare(error) => write!(formatter, "typed cell preparation failed: {error}"),
            Self::Evict(error) => write!(formatter, "typed cell eviction failed: {error}"),
        }
    }
}

impl<D, H> DecodedSceneHost<D, H>
where
    D: CellPayloadDecoder,
    H: DecodedCellHost<Cell = D::Cell>,
{
    /// Creates an empty typed host adapter with independent decoded-byte budgets.
    pub fn new(decoder: D, host: H, limits: CellDecodeLimits) -> Self {
        Self {
            decoder,
            host,
            limits,
            resident: BTreeMap::new(),
            resident_bytes: 0,
        }
    }

    /// Returns the wrapped application decoder.
    #[must_use]
    pub const fn decoder(&self) -> &D {
        &self.decoder
    }

    /// Returns mutable access to the wrapped application decoder.
    pub fn decoder_mut(&mut self) -> &mut D {
        &mut self.decoder
    }

    /// Returns the wrapped typed host.
    #[must_use]
    pub const fn host(&self) -> &H {
        &self.host
    }

    /// Returns mutable access to the wrapped typed host.
    pub fn host_mut(&mut self) -> &mut H {
        &mut self.host
    }

    /// Returns the current aggregate decoded-byte charge.
    #[must_use]
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Returns the number of typed cells tracked by this adapter.
    #[must_use]
    pub fn resident_cells(&self) -> usize {
        self.resident.len()
    }

    fn candidate_total(
        &self,
        sector: SectorId,
        candidate_bytes: u64,
    ) -> Result<u64, DecodedSceneHostError<D::Error, H::Error>> {
        if candidate_bytes > self.limits.max_cell_bytes {
            return Err(DecodedSceneHostError::CellBudget {
                required: candidate_bytes,
                limit: self.limits.max_cell_bytes,
            });
        }
        let previous = self.resident.get(&sector).copied();
        if previous.is_none() && self.resident.len() >= self.limits.max_resident_cells {
            return Err(DecodedSceneHostError::ResidentCellBudget {
                required: self.resident.len().saturating_add(1),
                limit: self.limits.max_resident_cells,
            });
        }
        let base = self
            .resident_bytes
            .checked_sub(previous.unwrap_or(0))
            .ok_or(DecodedSceneHostError::AccountingOverflow)?;
        let total = base
            .checked_add(candidate_bytes)
            .ok_or(DecodedSceneHostError::AccountingOverflow)?;
        if total > self.limits.max_resident_bytes {
            return Err(DecodedSceneHostError::ResidentBudget {
                required: total,
                limit: self.limits.max_resident_bytes,
            });
        }
        Ok(total)
    }
}

impl<D, H> SceneHost for DecodedSceneHost<D, H>
where
    D: CellPayloadDecoder,
    D::Error: fmt::Display,
    H: DecodedCellHost<Cell = D::Cell>,
    H::Error: fmt::Display,
{
    type Error = DecodedSceneHostError<D::Error, H::Error>;

    fn load_cell(&mut self, manifest: &CellManifest, payload: &[u8]) -> Result<(), Self::Error> {
        let reserved_bytes = self
            .decoder
            .preflight(manifest, payload)
            .map_err(DecodedSceneHostError::Decode)?;
        let reserved_total = self.candidate_total(manifest.sector, reserved_bytes)?;
        let decoded = self
            .decoder
            .decode(manifest, payload)
            .map_err(DecodedSceneHostError::Decode)?;
        if decoded.resident_bytes > reserved_bytes {
            return Err(DecodedSceneHostError::PreflightUnderestimated {
                actual: decoded.resident_bytes,
                reserved: reserved_bytes,
            });
        }
        let next_total = reserved_total
            .checked_sub(reserved_bytes)
            .and_then(|base| base.checked_add(decoded.resident_bytes))
            .ok_or(DecodedSceneHostError::AccountingOverflow)?;
        if next_total > self.limits.max_resident_bytes {
            return Err(DecodedSceneHostError::ResidentBudget {
                required: next_total,
                limit: self.limits.max_resident_bytes,
            });
        }
        let prepared = self
            .host
            .prepare(manifest, decoded.value)
            .map_err(DecodedSceneHostError::Prepare)?;
        self.resident
            .insert(manifest.sector, decoded.resident_bytes);
        self.resident_bytes = next_total;
        self.host.commit(prepared);
        Ok(())
    }

    fn load_failure_stage(&self, error: &Self::Error) -> FailureStage {
        match error {
            DecodedSceneHostError::Decode(_) => FailureStage::PayloadDecode,
            DecodedSceneHostError::CellBudget { .. }
            | DecodedSceneHostError::ResidentBudget { .. }
            | DecodedSceneHostError::ResidentCellBudget { .. }
            | DecodedSceneHostError::PreflightUnderestimated { .. }
            | DecodedSceneHostError::AccountingOverflow => FailureStage::PayloadBudget,
            DecodedSceneHostError::Prepare(_) => FailureStage::CellPrepare,
            DecodedSceneHostError::Evict(_) => FailureStage::LoadHost,
        }
    }

    fn load_failure_message(&self, error: &Self::Error) -> alloc::string::String {
        error.to_string()
    }

    fn evict_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        let previous = self.resident.get(&sector).copied();
        let next_total = match previous {
            Some(bytes) => Some(
                self.resident_bytes
                    .checked_sub(bytes)
                    .ok_or(DecodedSceneHostError::AccountingOverflow)?,
            ),
            None => None,
        };
        self.host
            .evict(sector)
            .map_err(DecodedSceneHostError::Evict)?;
        if previous.is_some() {
            self.resident.remove(&sector);
            self.resident_bytes = next_total.expect("previous resident has an accounted total");
        }
        Ok(())
    }
}
