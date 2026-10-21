//! Cell provider backed by a validated runtime asset-container manifest.
//!
//! The provider owns an immutable [`prism_asset_container::RuntimeManifest`]
//! and [`prism_asset_container::OverlaySet`]. Each request maps the cell's
//! `scene_key` to a [`prism_asset::StableGuid`], validates the routed entry
//! length before allocating, reads through [`prism_asset_container::ManifestReader`],
//! and reuses the runtime cell-header validator before returning bytes.
//!
//! This is a bounded synchronous compatibility provider. Cancellation is
//! observed before binding, before the read, and after the read; a blocking
//! container read cannot be interrupted once it has started. It performs no
//! GPU upload or GPU-direct transfer.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;

use prism_asset::StableGuid;
use prism_asset_container::{
    ManifestRangeRequest, ManifestReadError, ManifestReader, OverlaySet, RangeBatchPolicy,
    RuntimeManifest,
};
use prism_world_io::checksum;
use prism_world_stream::CellManifest;

use crate::{
    validate_header, CancellationToken, CellProvider, CellReadRequest, HeaderError,
    CELL_HEADER_SIZE,
};

/// Default maximum size of one complete cell stream, including its header.
pub const DEFAULT_ASSET_CONTAINER_MAX_IO_BYTES: u64 = 64 * 1024 * 1024;
/// Default maximum decoded bytes in one logical cell range request.
pub const DEFAULT_ASSET_CONTAINER_MAX_RANGE_BYTES: u64 = 64 * 1024;
/// Default maximum number of logical ranges used to assemble one cell.
pub const DEFAULT_ASSET_CONTAINER_MAX_RANGES: usize = 1024;

/// Admission limits for [`AssetContainerCellProvider`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AssetContainerProviderLimits {
    /// Maximum declared size of one complete cell stream, including the fixed
    /// [`crate::CELL_HEADER_SIZE`] byte header.
    pub max_io_bytes: u64,
    /// Maximum decoded bytes in one logical range request.
    pub max_range_bytes: u64,
    /// Maximum number of logical ranges used to assemble one cell.
    pub max_ranges: usize,
    /// Maximum physical bytes in one container batch.
    pub max_batch_bytes: u64,
    /// Maximum physical gap coalesced between adjacent ranges.
    pub max_gap_bytes: u64,
}

impl AssetContainerProviderLimits {
    /// Creates limits for one complete cell stream.
    #[must_use]
    pub const fn new(max_io_bytes: u64) -> Self {
        Self {
            max_io_bytes,
            max_range_bytes: DEFAULT_ASSET_CONTAINER_MAX_RANGE_BYTES,
            max_ranges: DEFAULT_ASSET_CONTAINER_MAX_RANGES,
            max_batch_bytes: 1024 * 1024,
            max_gap_bytes: 4096,
        }
    }

    /// Creates limits with explicit range and physical batch budgets.
    #[must_use]
    pub const fn with_ranges(
        max_io_bytes: u64,
        max_range_bytes: u64,
        max_ranges: usize,
        max_batch_bytes: u64,
        max_gap_bytes: u64,
    ) -> Self {
        Self {
            max_io_bytes,
            max_range_bytes,
            max_ranges,
            max_batch_bytes,
            max_gap_bytes,
        }
    }
}

impl Default for AssetContainerProviderLimits {
    fn default() -> Self {
        Self::new(DEFAULT_ASSET_CONTAINER_MAX_IO_BYTES)
    }
}

/// Errors returned by [`AssetContainerCellProvider`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AssetContainerProviderError {
    /// The request was cancelled at a provider boundary.
    Cancelled,
    /// The configured per-cell bound is smaller than the fixed cell header.
    InvalidMaxIoBytes {
        /// Configured maximum.
        max_io_bytes: u64,
        /// Minimum legal complete stream length.
        minimum: u64,
    },
    /// The configured range budget is invalid.
    InvalidRangeLimits {
        /// Configured maximum logical range size.
        max_range_bytes: u64,
        /// Configured maximum number of ranges.
        max_ranges: usize,
        /// Configured maximum physical batch size.
        max_batch_bytes: u64,
    },
    /// The request declares fewer bytes than the fixed cell header requires.
    InvalidDeclaredLength {
        /// Declared complete stream length.
        io_bytes: u64,
        /// Minimum legal complete stream length.
        minimum: u64,
    },
    /// The request contains a non-zero file/range offset, which is not valid
    /// for a manifest-routed logical cell payload.
    InvalidByteOffset {
        /// Unsupported request offset.
        byte_offset: u64,
    },
    /// The request exceeds the provider's explicit admission bound.
    RequestTooLarge {
        /// Declared complete stream length.
        io_bytes: u64,
        /// Configured maximum.
        max_io_bytes: u64,
    },
    /// The cell cannot be represented by the configured range-count budget.
    RangeCountExceeded {
        /// Complete cell stream length.
        io_bytes: u64,
        /// Maximum logical range size.
        max_range_bytes: u64,
        /// Configured maximum number of ranges.
        max_ranges: usize,
    },
    /// The declared length cannot be represented by the host allocator.
    LengthOverflow {
        /// Declared complete stream length.
        io_bytes: u64,
    },
    /// The manifest entry's decoded length does not match the cell manifest's
    /// complete stream length.
    PayloadLengthMismatch {
        /// Cell scene key represented by the route.
        scene_key: u128,
        /// Length declared by [`CellManifest`].
        expected_io_bytes: u64,
        /// Length reported by the validated container entry.
        actual_payload_bytes: u64,
    },
    /// The container read returned a buffer with an unexpected length.
    ReadLengthMismatch {
        /// Cell scene key represented by the route.
        scene_key: u128,
        /// Expected complete stream length.
        expected_io_bytes: u64,
        /// Actual returned length.
        actual_bytes: usize,
    },
    /// The returned bytes could not be reserved without an allocator failure.
    AllocationFailed {
        /// Requested complete stream capacity.
        requested_bytes: usize,
    },
    /// The manifest route, overlay, container identity, or range read failed.
    Manifest(Box<ManifestReadError>),
    /// The returned complete stream failed the existing `PRWC` header check.
    Header(HeaderError),
    /// The returned bytes failed the optional request checksum.
    ChecksumMismatch {
        /// Checksum requested by the caller.
        expected: u64,
        /// Checksum computed from the complete returned cell stream.
        actual: u64,
    },
}

impl fmt::Display for AssetContainerProviderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("asset-container cell read was cancelled"),
            Self::InvalidMaxIoBytes {
                max_io_bytes,
                minimum,
            } => write!(
                formatter,
                "asset-container max_io_bytes {max_io_bytes} is below minimum {minimum}"
            ),
            Self::InvalidRangeLimits {
                max_range_bytes,
                max_ranges,
                max_batch_bytes,
            } => write!(
                formatter,
                "invalid asset-container range limits: range {max_range_bytes}, ranges {max_ranges}, batch {max_batch_bytes}"
            ),
            Self::InvalidDeclaredLength { io_bytes, minimum } => write!(
                formatter,
                "cell declares {io_bytes} bytes, below minimum {minimum}"
            ),
            Self::InvalidByteOffset { byte_offset } => write!(
                formatter,
                "manifest-routed cell request has unsupported byte offset {byte_offset}"
            ),
            Self::RequestTooLarge {
                io_bytes,
                max_io_bytes,
            } => write!(
                formatter,
                "cell declares {io_bytes} bytes, above provider limit {max_io_bytes}"
            ),
            Self::RangeCountExceeded {
                io_bytes,
                max_range_bytes,
                max_ranges,
            } => write!(
                formatter,
                "cell declares {io_bytes} bytes but needs more than {max_ranges} ranges at {max_range_bytes} bytes each"
            ),
            Self::LengthOverflow { io_bytes } => write!(
                formatter,
                "cell length {io_bytes} cannot be represented by the host allocator"
            ),
            Self::PayloadLengthMismatch {
                scene_key,
                expected_io_bytes,
                actual_payload_bytes,
            } => write!(
                formatter,
                "asset {scene_key:#x} has payload length {actual_payload_bytes}, expected complete cell length {expected_io_bytes}"
            ),
            Self::ReadLengthMismatch {
                scene_key,
                expected_io_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "asset {scene_key:#x} returned {actual_bytes} bytes, expected {expected_io_bytes}"
            ),
            Self::AllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve {requested_bytes} bytes for asset-container cell"
            ),
            Self::Manifest(error) => error.fmt(formatter),
            Self::Header(error) => error.fmt(formatter),
            Self::ChecksumMismatch { expected, actual } => write!(
                formatter,
                "asset-container cell checksum mismatch: expected {expected:#x}, got {actual:#x}"
            ),
        }
    }
}

/// A deterministic, budget-checked logical range plan for one complete cell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetContainerCellRangePlan {
    manifest: CellManifest,
    requests: Vec<ManifestRangeRequest>,
}

impl AssetContainerCellRangePlan {
    /// Returns the complete cell manifest represented by this plan.
    #[must_use]
    pub const fn manifest(&self) -> CellManifest {
        self.manifest
    }

    /// Returns logical requests in deterministic declaration order.
    #[must_use]
    pub fn requests(&self) -> &[ManifestRangeRequest] {
        &self.requests
    }

    /// Returns the number of bytes assembled by the plan.
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.manifest.io_bytes
    }
}

impl std::error::Error for AssetContainerProviderError {}

/// A bounded cell provider over a validated asset-container overlay.
#[derive(Clone, Debug)]
pub struct AssetContainerCellProvider {
    manifest: RuntimeManifest,
    overlays: OverlaySet,
    limits: AssetContainerProviderLimits,
}

impl AssetContainerCellProvider {
    /// Creates a provider with [`AssetContainerProviderLimits::default`].
    #[must_use]
    pub fn new(manifest: RuntimeManifest, overlays: OverlaySet) -> Self {
        Self {
            manifest,
            overlays,
            limits: AssetContainerProviderLimits::default(),
        }
    }

    /// Creates a provider with explicit bounded allocation admission.
    pub fn with_limits(
        manifest: RuntimeManifest,
        overlays: OverlaySet,
        limits: AssetContainerProviderLimits,
    ) -> Result<Self, AssetContainerProviderError> {
        let minimum = u64::try_from(CELL_HEADER_SIZE).expect("cell header size fits u64");
        if limits.max_io_bytes < minimum {
            return Err(AssetContainerProviderError::InvalidMaxIoBytes {
                max_io_bytes: limits.max_io_bytes,
                minimum,
            });
        }
        if limits.max_range_bytes == 0
            || limits.max_ranges == 0
            || limits.max_batch_bytes == 0
            || limits.max_range_bytes > limits.max_batch_bytes
        {
            return Err(AssetContainerProviderError::InvalidRangeLimits {
                max_range_bytes: limits.max_range_bytes,
                max_ranges: limits.max_ranges,
                max_batch_bytes: limits.max_batch_bytes,
            });
        }
        Ok(Self {
            manifest,
            overlays,
            limits,
        })
    }

    /// Creates a provider with an explicit maximum complete cell size.
    pub fn with_max_io_bytes(
        manifest: RuntimeManifest,
        overlays: OverlaySet,
        max_io_bytes: u64,
    ) -> Result<Self, AssetContainerProviderError> {
        Self::with_limits(
            manifest,
            overlays,
            AssetContainerProviderLimits::new(max_io_bytes),
        )
    }

    /// Returns the immutable runtime manifest used for logical routing.
    #[must_use]
    pub fn manifest(&self) -> &RuntimeManifest {
        &self.manifest
    }

    /// Returns the immutable mounted overlay used for physical reads.
    #[must_use]
    pub fn overlays(&self) -> &OverlaySet {
        &self.overlays
    }

    /// Returns the provider's admission limits.
    #[must_use]
    pub const fn limits(&self) -> AssetContainerProviderLimits {
        self.limits
    }

    fn validate_request(
        &self,
        request: &CellReadRequest,
    ) -> Result<(CellManifest, usize, Option<u64>), AssetContainerProviderError> {
        let manifest = request.manifest();
        let io_request = request.io_request();
        if io_request.offset() != 0 {
            return Err(AssetContainerProviderError::InvalidByteOffset {
                byte_offset: io_request.offset(),
            });
        }
        if io_request.length() != manifest.io_bytes {
            return Err(AssetContainerProviderError::LengthOverflow {
                io_bytes: manifest.io_bytes,
            });
        }
        let minimum = u64::try_from(CELL_HEADER_SIZE).expect("cell header size fits u64");
        let payload_bytes = manifest.io_bytes.checked_sub(minimum).ok_or(
            AssetContainerProviderError::InvalidDeclaredLength {
                io_bytes: manifest.io_bytes,
                minimum,
            },
        )?;
        let reconstructed = minimum.checked_add(payload_bytes).ok_or(
            AssetContainerProviderError::LengthOverflow {
                io_bytes: manifest.io_bytes,
            },
        )?;
        if reconstructed != manifest.io_bytes {
            return Err(AssetContainerProviderError::LengthOverflow {
                io_bytes: manifest.io_bytes,
            });
        }
        if manifest.io_bytes > self.limits.max_io_bytes {
            return Err(AssetContainerProviderError::RequestTooLarge {
                io_bytes: manifest.io_bytes,
                max_io_bytes: self.limits.max_io_bytes,
            });
        }
        let length = usize::try_from(manifest.io_bytes).map_err(|_| {
            AssetContainerProviderError::LengthOverflow {
                io_bytes: manifest.io_bytes,
            }
        })?;
        Ok((manifest, length, io_request.expected_checksum()))
    }

    fn make_plan(
        &self,
        manifest: CellManifest,
        entry_len: u64,
    ) -> Result<AssetContainerCellRangePlan, AssetContainerProviderError> {
        if entry_len != manifest.io_bytes {
            return Err(AssetContainerProviderError::PayloadLengthMismatch {
                scene_key: manifest.scene_key,
                expected_io_bytes: manifest.io_bytes,
                actual_payload_bytes: entry_len,
            });
        }
        let range_count = manifest
            .io_bytes
            .saturating_add(self.limits.max_range_bytes - 1)
            / self.limits.max_range_bytes;
        let range_count_usize = usize::try_from(range_count).map_err(|_| {
            AssetContainerProviderError::RangeCountExceeded {
                io_bytes: manifest.io_bytes,
                max_range_bytes: self.limits.max_range_bytes,
                max_ranges: self.limits.max_ranges,
            }
        })?;
        if range_count_usize > self.limits.max_ranges {
            return Err(AssetContainerProviderError::RangeCountExceeded {
                io_bytes: manifest.io_bytes,
                max_range_bytes: self.limits.max_range_bytes,
                max_ranges: self.limits.max_ranges,
            });
        }
        let guid = StableGuid::from_u128(manifest.scene_key);
        let mut requests = Vec::new();
        requests.try_reserve_exact(range_count_usize).map_err(|_| {
            AssetContainerProviderError::AllocationFailed {
                requested_bytes: range_count_usize
                    .saturating_mul(core::mem::size_of::<ManifestRangeRequest>()),
            }
        })?;
        let mut offset = 0_u64;
        while offset < manifest.io_bytes {
            let len = (manifest.io_bytes - offset).min(self.limits.max_range_bytes);
            requests.push(ManifestRangeRequest::new(guid, offset, len));
            offset =
                offset
                    .checked_add(len)
                    .ok_or(AssetContainerProviderError::LengthOverflow {
                        io_bytes: manifest.io_bytes,
                    })?;
        }
        Ok(AssetContainerCellRangePlan { manifest, requests })
    }

    /// Plans deterministic logical ranges after validating the current route.
    pub fn plan_cell_ranges(
        &self,
        request: &CellReadRequest,
    ) -> Result<AssetContainerCellRangePlan, AssetContainerProviderError> {
        let (manifest, _, _) = self.validate_request(request)?;
        let reader = self
            .manifest
            .bind(&self.overlays)
            .map_err(|error| AssetContainerProviderError::Manifest(Box::new(error)))?;
        let guid = StableGuid::from_u128(manifest.scene_key);
        let entry = reader
            .entry_info(guid)
            .map_err(|error| AssetContainerProviderError::Manifest(Box::new(error)))?;
        self.make_plan(manifest, entry.raw_len)
    }

    fn assemble_ranges(
        &self,
        reader: ManifestReader<'_, '_>,
        plan: &AssetContainerCellRangePlan,
    ) -> Result<Vec<u8>, AssetContainerProviderError> {
        let policy = RangeBatchPolicy {
            max_batch_bytes: self.limits.max_batch_bytes,
            max_gap_bytes: self.limits.max_gap_bytes,
        };
        let length = usize::try_from(plan.total_bytes()).map_err(|_| {
            AssetContainerProviderError::LengthOverflow {
                io_bytes: plan.total_bytes(),
            }
        })?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length).map_err(|_| {
            AssetContainerProviderError::AllocationFailed {
                requested_bytes: length,
            }
        })?;
        bytes.resize(length, 0);
        reader
            .read_ranges_into(plan.requests(), policy, &mut bytes)
            .map_err(|error| AssetContainerProviderError::Manifest(Box::new(error)))?;
        validate_header(plan.manifest(), &bytes).map_err(AssetContainerProviderError::Header)?;
        Ok(bytes)
    }

    /// Reads and assembles a complete cell through bounded logical ranges.
    pub fn read_cell_ranges(
        &self,
        request: &CellReadRequest,
        cancel: CancellationToken,
    ) -> Result<Vec<u8>, AssetContainerProviderError> {
        if cancel.is_cancelled() {
            return Err(AssetContainerProviderError::Cancelled);
        }
        let (manifest, _, expected_checksum) = self.validate_request(request)?;
        if cancel.is_cancelled() {
            return Err(AssetContainerProviderError::Cancelled);
        }
        let reader = self
            .manifest
            .bind(&self.overlays)
            .map_err(|error| AssetContainerProviderError::Manifest(Box::new(error)))?;
        let guid = StableGuid::from_u128(manifest.scene_key);
        let entry = reader
            .entry_info(guid)
            .map_err(|error| AssetContainerProviderError::Manifest(Box::new(error)))?;
        let plan = self.make_plan(manifest, entry.raw_len)?;
        if cancel.is_cancelled() {
            return Err(AssetContainerProviderError::Cancelled);
        }
        let bytes = self.assemble_ranges(reader, &plan)?;
        if cancel.is_cancelled() {
            return Err(AssetContainerProviderError::Cancelled);
        }
        if let Some(expected) = expected_checksum {
            let actual = checksum(&bytes);
            if actual != expected {
                return Err(AssetContainerProviderError::ChecksumMismatch { expected, actual });
            }
        }
        Ok(bytes)
    }

    fn read_bound(
        &self,
        reader: ManifestReader<'_, '_>,
        manifest: CellManifest,
        length: usize,
    ) -> Result<Vec<u8>, AssetContainerProviderError> {
        let guid = StableGuid::from_u128(manifest.scene_key);
        let entry = reader
            .entry_info(guid)
            .map_err(|error| AssetContainerProviderError::Manifest(Box::new(error)))?;
        let expected =
            u64::try_from(length).map_err(|_| AssetContainerProviderError::LengthOverflow {
                io_bytes: manifest.io_bytes,
            })?;
        if entry.raw_len != expected {
            return Err(AssetContainerProviderError::PayloadLengthMismatch {
                scene_key: manifest.scene_key,
                expected_io_bytes: expected,
                actual_payload_bytes: entry.raw_len,
            });
        }

        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length).map_err(|_| {
            AssetContainerProviderError::AllocationFailed {
                requested_bytes: length,
            }
        })?;
        bytes.resize(length, 0);
        reader
            .read_range(guid, 0, &mut bytes)
            .map_err(|error| AssetContainerProviderError::Manifest(Box::new(error)))?;
        if bytes.len() != length {
            return Err(AssetContainerProviderError::ReadLengthMismatch {
                scene_key: manifest.scene_key,
                expected_io_bytes: expected,
                actual_bytes: bytes.len(),
            });
        }
        validate_header(manifest, &bytes).map_err(AssetContainerProviderError::Header)?;
        Ok(bytes)
    }
}

impl CellProvider for AssetContainerCellProvider {
    type Error = AssetContainerProviderError;

    fn read_cell(
        &self,
        request: &CellReadRequest,
        cancel: CancellationToken,
    ) -> Result<Vec<u8>, Self::Error> {
        if cancel.is_cancelled() {
            return Err(AssetContainerProviderError::Cancelled);
        }
        let (manifest, length, expected_checksum) = self.validate_request(request)?;
        if cancel.is_cancelled() {
            return Err(AssetContainerProviderError::Cancelled);
        }
        let reader = self
            .manifest
            .bind(&self.overlays)
            .map_err(|error| AssetContainerProviderError::Manifest(Box::new(error)))?;
        if cancel.is_cancelled() {
            return Err(AssetContainerProviderError::Cancelled);
        }
        let guid = StableGuid::from_u128(manifest.scene_key);
        let entry = reader
            .entry_info(guid)
            .map_err(|error| AssetContainerProviderError::Manifest(Box::new(error)))?;
        let plan = self.make_plan(manifest, entry.raw_len)?;
        let bytes = if plan.requests().len() == 1 {
            self.read_bound(reader, manifest, length)?
        } else {
            self.assemble_ranges(reader, &plan)?
        };
        if cancel.is_cancelled() {
            return Err(AssetContainerProviderError::Cancelled);
        }
        if let Some(expected) = expected_checksum {
            let actual = checksum(&bytes);
            if actual != expected {
                return Err(AssetContainerProviderError::ChecksumMismatch { expected, actual });
            }
        }
        Ok(bytes)
    }
}
