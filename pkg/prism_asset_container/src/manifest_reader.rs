//! Reads logical assets through a validated [`RuntimeManifest`].
//!
//! A manifest route is an explicit binding, not a hint for the normal overlay
//! lookup. This module therefore checks the named mount, its priority, the
//! current winning source, and the route's container-local GUID before any
//! payload read is attempted.

use core::fmt;
use core::mem::size_of;

use prism_asset::StableGuid;

use crate::range_batch::{checked_physical_range_within, validate_range, RangeBatchPolicy};
use crate::{
    Codec, ContainerError, ContainerReader, EntryInfo, ManifestRoute, OverlaySet, RangeRequest,
    RuntimeManifest, CHUNK_FORMAT_VERSION,
};

#[cfg(feature = "gpu")]
use crate::{
    GpuBlockDecodeBackend, GpuResidentBlockPayload, GpuResidentBlockRange, GpuResidentBlockSlice,
};

/// Errors raised while binding or reading a logical manifest route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ManifestReadError {
    /// The requested logical GUID has no route in the manifest.
    MissingRoute(StableGuid),
    /// The route names a container that is not mounted.
    MissingContainer {
        /// Logical GUID requested by the caller.
        logical_guid: StableGuid,
        /// Container name recorded by the route.
        container_name: String,
    },
    /// The route's priority differs from the mounted route source.
    PriorityMismatch {
        /// Logical GUID represented by the route.
        logical_guid: StableGuid,
        /// Container name recorded by the route.
        container_name: String,
        /// Priority recorded in the manifest.
        manifest_priority: i32,
        /// Priority of the mounted source.
        mounted_priority: i32,
    },
    /// The route's named source is not the current overlay winner.
    SourceMismatch {
        /// Logical GUID represented by the route.
        logical_guid: StableGuid,
        /// Container name recorded by the route.
        expected: String,
        /// Current winner for the route's container-local GUID, if any.
        actual: Option<String>,
    },
    /// The named container is mounted, but does not contain the route's local GUID.
    MissingContainerGuid {
        /// Logical GUID represented by the route.
        logical_guid: StableGuid,
        /// Container name recorded by the route.
        container_name: String,
        /// GUID expected inside the named container.
        container_guid: StableGuid,
    },
    /// The named mount has the right route but different immutable content.
    ContainerIdentityMismatch {
        /// Logical GUID represented by the route.
        logical_guid: StableGuid,
        /// Container name recorded by the route.
        container_name: String,
        /// Length recorded by the manifest.
        expected_len: u64,
        /// Length observed from the mounted container.
        actual_len: u64,
        /// Hash recorded by the manifest.
        expected_hash: [u8; 32],
        /// Hash observed from the mounted container.
        actual_hash: [u8; 32],
        /// Generation recorded by the manifest.
        expected_generation: u64,
        /// Generation observed from the mounted container.
        actual_generation: u64,
    },
    /// A route has no immutable container identity binding.
    MissingContainerIdentity {
        /// Logical GUID represented by the route.
        logical_guid: StableGuid,
        /// Container name recorded by the route.
        container_name: String,
    },
    /// The requested byte range is outside the routed payload.
    InvalidRange {
        /// Logical GUID requested by the caller.
        logical_guid: StableGuid,
        /// Requested payload offset.
        offset: u64,
        /// Requested output length.
        len: usize,
    },
    /// The routed container failed an already-validated container read.
    Container(ContainerError),
}

impl From<ContainerError> for ManifestReadError {
    fn from(error: ContainerError) -> Self {
        Self::Container(error)
    }
}

impl fmt::Display for ManifestReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRoute(guid) => write!(f, "manifest route for {guid:?} is missing"),
            Self::MissingContainer {
                logical_guid,
                container_name,
            } => write!(
                f,
                "manifest route for {logical_guid:?} names missing container {container_name:?}"
            ),
            Self::PriorityMismatch {
                logical_guid,
                container_name,
                manifest_priority,
                mounted_priority,
            } => write!(f, "manifest route for {logical_guid:?} has priority {manifest_priority}, but container {container_name:?} is mounted at priority {mounted_priority}"),
            Self::SourceMismatch {
                logical_guid,
                expected,
                actual,
            } => write!(f, "manifest route for {logical_guid:?} expects source {expected:?}, but the current source is {actual:?}"),
            Self::MissingContainerGuid {
                logical_guid,
                container_name,
                container_guid,
            } => write!(f, "manifest route for {logical_guid:?} cannot find container GUID {container_guid:?} in {container_name:?}"),
            Self::ContainerIdentityMismatch {
                logical_guid,
                container_name,
                expected_len,
                actual_len,
                expected_hash,
                actual_hash,
                expected_generation,
                actual_generation,
            } => write!(
                f,
                "manifest route for {logical_guid:?} has container identity mismatch in {container_name:?}: len {expected_len}/{actual_len}, hash {:02x?}/{:02x?}, generation {expected_generation}/{actual_generation}",
                expected_hash,
                actual_hash,
            ),
            Self::MissingContainerIdentity {
                logical_guid,
                container_name,
            } => write!(
                f,
                "manifest route for {logical_guid:?} has no identity for container {container_name:?}"
            ),
            Self::InvalidRange {
                logical_guid,
                offset,
                len,
            } => write!(
                f,
                "range offset {offset} length {len} is invalid for logical asset {logical_guid:?}"
            ),
            Self::Container(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ManifestReadError {}

/// A validated logical-GUID reader backed by a mounted [`OverlaySet`].
#[derive(Clone, Copy, Debug)]
pub struct ManifestReader<'manifest, 'overlay> {
    manifest: &'manifest RuntimeManifest,
    overlays: &'overlay OverlaySet,
}

/// One logical range request routed through a runtime manifest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ManifestRangeRequest {
    /// Logical asset identity in the manifest.
    pub logical_guid: StableGuid,
    /// Offset within the routed payload.
    pub offset: u64,
    /// Number of bytes requested.
    pub len: u64,
}

/// A logical manifest asset authorized for GPU-resident `BlockDeflate` reads.
///
/// Binding captures the current immutable route. Remounting does not revoke an
/// existing capability; bind again to select a new winning revision. Reads
/// validate asset-relative bounds, but returned buffers contain whole decoded
/// blocks: selected byte ranges are not an access-control boundary between
/// entries sharing a chunk. Mapped files must remain immutable for the entire
/// capability lifetime, including after the reader is dropped.
#[cfg(feature = "gpu")]
pub struct GpuResidentManifestAsset {
    logical_guid: StableGuid,
    entry: EntryInfo,
    payload: GpuResidentBlockPayload,
}

#[cfg(feature = "gpu")]
impl GpuResidentManifestAsset {
    /// Logical GUID used to authorize this routed asset.
    #[must_use]
    pub const fn logical_guid(&self) -> StableGuid {
        self.logical_guid
    }

    /// Validated container-local entry metadata.
    #[must_use]
    pub const fn entry(&self) -> EntryInfo {
        self.entry
    }

    /// Shared context on which returned GPU buffers were allocated.
    #[must_use]
    pub fn context(&self) -> &std::sync::Arc<prism_gdeflate_gpu::GpuContext> {
        self.payload.context()
    }

    /// Decodes an asset-relative range into owned GPU-resident block outputs.
    ///
    /// Admission bounds one batch, not the lifetime aggregate of GPU outputs.
    pub fn read_range_resident(
        &self,
        offset: u64,
        len: u64,
        admission: prism_gdeflate_gpu::DeflateBatchLimits,
    ) -> Result<GpuResidentAssetRange, ManifestReadError> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| self.invalid_range(offset, len))?;
        if offset > self.entry.raw_len || end > self.entry.raw_len {
            return Err(self.invalid_range(offset, len));
        }
        let chunk_offset = self
            .entry
            .raw_offset
            .checked_add(offset)
            .ok_or_else(|| self.invalid_range(offset, len))?;
        self.payload
            .read_range_resident(chunk_offset, len, admission)
            .map(|blocks| GpuResidentAssetRange {
                logical_guid: self.logical_guid,
                range: offset..end,
                blocks,
            })
            .map_err(ManifestReadError::Container)
    }

    fn invalid_range(&self, offset: u64, len: u64) -> ManifestReadError {
        ManifestReadError::InvalidRange {
            logical_guid: self.logical_guid,
            offset,
            len: usize::try_from(len).unwrap_or(usize::MAX),
        }
    }
}

/// An owned resident result whose requested range is relative to one asset,
/// rather than to the physical chunk shared by multiple entries.
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct GpuResidentAssetRange {
    logical_guid: StableGuid,
    range: core::ops::Range<u64>,
    blocks: GpuResidentBlockRange,
}

#[cfg(feature = "gpu")]
impl GpuResidentAssetRange {
    /// Logical asset identity authorized when the capability was bound.
    #[must_use]
    pub const fn logical_guid(&self) -> StableGuid {
        self.logical_guid
    }

    /// Exact requested range relative to the decoded asset.
    #[must_use]
    pub fn range(&self) -> core::ops::Range<u64> {
        self.range.clone()
    }

    /// Whole decoded blocks and exact selected byte ranges. Slice output
    /// offsets are relative to the beginning of this request, not the chunk.
    #[must_use]
    pub fn slices(&self) -> &[GpuResidentBlockSlice] {
        self.blocks.slices()
    }

    /// Transfers outputs independently of the manifest, reader and capability.
    #[must_use]
    pub fn into_slices(self) -> Vec<GpuResidentBlockSlice> {
        self.blocks.into_slices()
    }
}

impl ManifestRangeRequest {
    /// Creates a logical manifest range request.
    #[must_use]
    pub const fn new(logical_guid: StableGuid, offset: u64, len: u64) -> Self {
        Self {
            logical_guid,
            offset,
            len,
        }
    }
}

#[expect(
    clippy::result_large_err,
    reason = "manifest errors retain complete route and container identity diagnostics"
)]
impl<'manifest, 'overlay> ManifestReader<'manifest, 'overlay> {
    /// Validates every route against the current overlay before creating a reader.
    pub fn bind(
        manifest: &'manifest RuntimeManifest,
        overlays: &'overlay OverlaySet,
    ) -> Result<Self, ManifestReadError> {
        let reader = Self { manifest, overlays };
        for route in manifest.routes() {
            reader.resolve(route)?;
        }
        Ok(reader)
    }

    /// Reads the complete payload selected by a logical stable GUID.
    pub fn read_all(&self, logical_guid: StableGuid) -> Result<Vec<u8>, ManifestReadError> {
        let route = self.route(logical_guid)?;
        let reader = self.resolve(route)?;
        reader
            .read_all(route.container_guid)
            .map_err(|error| self.map_read_error(logical_guid, 0, 0, error))
    }

    /// Reads a payload range selected by a logical stable GUID.
    pub fn read_range(
        &self,
        logical_guid: StableGuid,
        offset: u64,
        out: &mut [u8],
    ) -> Result<(), ManifestReadError> {
        let route = self.route(logical_guid)?;
        let reader = self.resolve(route)?;
        reader
            .read_range(route.container_guid, offset, out)
            .map_err(|error| self.map_read_error(logical_guid, offset, out.len(), error))
    }

    /// Reads an owned range under the selected container's output/I/O budget.
    pub fn read_range_owned(
        &self,
        logical_guid: StableGuid,
        offset: u64,
        len: u64,
    ) -> Result<Vec<u8>, ManifestReadError> {
        let route = self.route(logical_guid)?;
        let reader = self.resolve(route)?;
        reader
            .read_range_owned(route.container_guid, offset, len)
            .map_err(|error| {
                self.map_read_error(
                    logical_guid,
                    offset,
                    usize::try_from(len).unwrap_or(usize::MAX),
                    error,
                )
            })
    }

    /// Reads logical ranges with deterministic per-container coalescing.
    ///
    /// Every route is resolved before any payload read. Requests targeting the
    /// same mounted container share a physical batch; results remain in input
    /// order even when logical identities map to different local GUIDs.
    pub fn read_ranges(
        &self,
        requests: &[ManifestRangeRequest],
        policy: RangeBatchPolicy,
    ) -> Result<Vec<Vec<u8>>, ManifestReadError> {
        policy.validate().map_err(ManifestReadError::Container)?;

        struct Group<'a> {
            name: String,
            reader: &'a ContainerReader,
            requests: Vec<(usize, ManifestRangeRequest, RangeRequest)>,
        }

        let mut outputs = vec![Vec::new(); requests.len()];
        let mut groups = Vec::<Group<'_>>::new();
        for (index, request) in requests.iter().copied().enumerate() {
            let route = self.route(request.logical_guid)?;
            let reader = self.resolve(route)?;
            let local = RangeRequest::new(route.container_guid, request.offset, request.len);
            policy
                .validate_request_len(request.len)
                .map_err(ManifestReadError::Container)?;
            let entry = reader.entry(route.container_guid).ok_or_else(|| {
                ManifestReadError::MissingContainerGuid {
                    logical_guid: request.logical_guid,
                    container_name: route.container_name.clone(),
                    container_guid: route.container_guid,
                }
            })?;
            validate_range(request.offset, request.len, entry.raw_len).map_err(|error| {
                self.map_read_error(
                    request.logical_guid,
                    request.offset,
                    usize::try_from(request.len).unwrap_or(usize::MAX),
                    error,
                )
            })?;
            // Raw payloads are served by direct physical range I/O. A
            // Deflate payload is decoded from its complete stored record, so
            // its logical raw offset must not be compared with the shorter
            // compressed byte range in the container.
            if entry.codec == Codec::Raw {
                checked_physical_range_within(
                    entry
                        .offset
                        .checked_add(if reader.info().version == CHUNK_FORMAT_VERSION {
                            entry.raw_offset
                        } else {
                            0
                        })
                        .ok_or(ContainerError::IntegerOverflow)
                        .map_err(|error| {
                            self.map_read_error(
                                request.logical_guid,
                                request.offset,
                                usize::try_from(request.len).unwrap_or(usize::MAX),
                                error,
                            )
                        })?,
                    request.offset,
                    request.len,
                    reader.info().total_len,
                )
                .map_err(|error| {
                    self.map_read_error(
                        request.logical_guid,
                        request.offset,
                        usize::try_from(request.len).unwrap_or(usize::MAX),
                        error,
                    )
                })?;
            }
            reader.validate_range_request(local).map_err(|error| {
                self.map_read_error(
                    request.logical_guid,
                    request.offset,
                    usize::try_from(request.len).unwrap_or(usize::MAX),
                    error,
                )
            })?;
            if let Some(group) = groups
                .iter_mut()
                .find(|group| group.name == route.container_name)
            {
                group.requests.push((index, request, local));
            } else {
                groups.push(Group {
                    name: route.container_name.clone(),
                    reader,
                    requests: vec![(index, request, local)],
                });
            }
        }

        // Group order must not depend on the caller's logical request order.
        // The per-container reader still restores each group's results to
        // caller order, while this makes cross-container source I/O stable.
        groups.sort_by(|left, right| left.name.cmp(&right.name));
        for group in groups {
            let local_requests: Vec<_> =
                group.requests.iter().map(|(_, _, local)| *local).collect();
            let bytes = group
                .reader
                .read_ranges(&local_requests, policy)
                .map_err(|error| {
                    if let Some((_, logical, _)) = group.requests.first() {
                        self.map_read_error(
                            logical.logical_guid,
                            logical.offset,
                            usize::try_from(logical.len).unwrap_or(usize::MAX),
                            error,
                        )
                    } else {
                        ManifestReadError::Container(error)
                    }
                })?;
            for ((index, _, _), data) in group.requests.into_iter().zip(bytes) {
                outputs[index] = data;
            }
        }
        Ok(outputs)
    }

    /// Reads logical ranges directly into one packed caller buffer.
    ///
    /// The buffer layout concatenates requests in input order. All routes and
    /// ranges are resolved and validated before the first payload read. The
    /// output may be partially written on error and must then be discarded.
    pub fn read_ranges_into(
        &self,
        requests: &[ManifestRangeRequest],
        policy: RangeBatchPolicy,
        output: &mut [u8],
    ) -> Result<(), ManifestReadError> {
        policy.validate().map_err(ManifestReadError::Container)?;
        let offsets_bytes =
            requests
                .len()
                .checked_mul(size_of::<usize>())
                .ok_or(ManifestReadError::Container(
                    ContainerError::IntegerOverflow,
                ))?;
        let mut output_offsets = Vec::new();
        output_offsets
            .try_reserve_exact(requests.len())
            .map_err(|_| {
                ManifestReadError::Container(ContainerError::AllocationFailed {
                    bytes: u64::try_from(offsets_bytes).unwrap_or(u64::MAX),
                })
            })?;
        let mut total = 0_u64;
        for request in requests {
            output_offsets.push(
                usize::try_from(total)
                    .map_err(|_| ManifestReadError::Container(ContainerError::IntegerOverflow))?,
            );
            total = total
                .checked_add(request.len)
                .ok_or(ManifestReadError::Container(
                    ContainerError::IntegerOverflow,
                ))?;
        }
        if usize::try_from(total)
            .map_err(|_| ManifestReadError::Container(ContainerError::IntegerOverflow))?
            != output.len()
        {
            return Err(ManifestReadError::Container(ContainerError::LengthMismatch));
        }

        struct Group<'a> {
            name: String,
            reader: &'a ContainerReader,
            requests: Vec<(usize, ManifestRangeRequest, RangeRequest)>,
        }

        let mut groups = Vec::<Group<'_>>::new();
        for (index, request) in requests.iter().copied().enumerate() {
            let route = self.route(request.logical_guid)?;
            let reader = self.resolve(route)?;
            let local = RangeRequest::new(route.container_guid, request.offset, request.len);
            policy
                .validate_request_len(request.len)
                .map_err(ManifestReadError::Container)?;
            let entry = reader.entry(route.container_guid).ok_or_else(|| {
                ManifestReadError::MissingContainerGuid {
                    logical_guid: request.logical_guid,
                    container_name: route.container_name.clone(),
                    container_guid: route.container_guid,
                }
            })?;
            validate_range(request.offset, request.len, entry.raw_len).map_err(|error| {
                self.map_read_error(
                    request.logical_guid,
                    request.offset,
                    usize::try_from(request.len).unwrap_or(usize::MAX),
                    error,
                )
            })?;
            if entry.codec == Codec::Raw {
                let base = entry
                    .offset
                    .checked_add(if reader.info().version == CHUNK_FORMAT_VERSION {
                        entry.raw_offset
                    } else {
                        0
                    })
                    .ok_or(ContainerError::IntegerOverflow)
                    .map_err(|error| {
                        self.map_read_error(
                            request.logical_guid,
                            request.offset,
                            usize::try_from(request.len).unwrap_or(usize::MAX),
                            error,
                        )
                    })?;
                checked_physical_range_within(
                    base,
                    request.offset,
                    request.len,
                    reader.info().total_len,
                )
                .map_err(|error| {
                    self.map_read_error(
                        request.logical_guid,
                        request.offset,
                        usize::try_from(request.len).unwrap_or(usize::MAX),
                        error,
                    )
                })?;
            }
            reader.validate_range_request(local).map_err(|error| {
                self.map_read_error(
                    request.logical_guid,
                    request.offset,
                    usize::try_from(request.len).unwrap_or(usize::MAX),
                    error,
                )
            })?;
            if let Some(group) = groups
                .iter_mut()
                .find(|group| group.name == route.container_name)
            {
                group.requests.push((index, request, local));
            } else {
                groups.push(Group {
                    name: route.container_name.clone(),
                    reader,
                    requests: alloc::vec![(index, request, local)],
                });
            }
        }

        groups.sort_by(|left, right| left.name.cmp(&right.name));
        for group in groups {
            let mut local_requests = Vec::new();
            let mut local_offsets = Vec::new();
            let request_bytes = group
                .requests
                .len()
                .checked_mul(size_of::<RangeRequest>())
                .ok_or(ManifestReadError::Container(
                    ContainerError::IntegerOverflow,
                ))?;
            local_requests
                .try_reserve_exact(group.requests.len())
                .map_err(|_| {
                    ManifestReadError::Container(ContainerError::AllocationFailed {
                        bytes: u64::try_from(request_bytes).unwrap_or(u64::MAX),
                    })
                })?;
            local_offsets
                .try_reserve_exact(group.requests.len())
                .map_err(|_| {
                    ManifestReadError::Container(ContainerError::AllocationFailed {
                        bytes: u64::try_from(
                            group.requests.len().saturating_mul(size_of::<usize>()),
                        )
                        .unwrap_or(u64::MAX),
                    })
                })?;
            for (index, _, request) in &group.requests {
                local_requests.push(*request);
                local_offsets.push(output_offsets[*index]);
            }
            group
                .reader
                .read_ranges_into_at(&local_requests, policy, &local_offsets, output)
                .map_err(|error| {
                    if let Some((_, logical, _)) = group.requests.first() {
                        self.map_read_error(
                            logical.logical_guid,
                            logical.offset,
                            usize::try_from(logical.len).unwrap_or(usize::MAX),
                            error,
                        )
                    } else {
                        ManifestReadError::Container(error)
                    }
                })?;
        }
        Ok(())
    }

    /// Returns validated metadata for the payload selected by a logical GUID.
    ///
    /// This is used by the [`ManifestAssetReader`](crate::ManifestAssetReader)
    /// adapter so the runtime can perform size accounting without reading the
    /// complete payload first.
    pub fn entry_info(&self, logical_guid: StableGuid) -> Result<EntryInfo, ManifestReadError> {
        let route = self.route(logical_guid)?;
        let reader = self.resolve(route)?;
        reader
            .entry(route.container_guid)
            .ok_or_else(|| ManifestReadError::MissingContainerGuid {
                logical_guid,
                container_name: route.container_name.clone(),
                container_guid: route.container_guid,
            })
    }

    /// Authorizes a logical manifest asset for GPU-resident `BlockDeflate`
    /// reads using the winning reader's own immutable backing and budget.
    #[cfg(feature = "gpu")]
    pub fn bind_gpu_resident_asset(
        &self,
        logical_guid: StableGuid,
        backend: &GpuBlockDecodeBackend,
    ) -> Result<GpuResidentManifestAsset, ManifestReadError> {
        let route = self.route(logical_guid)?;
        let reader = self.resolve(route)?;
        let entry = reader.entry(route.container_guid).ok_or_else(|| {
            ManifestReadError::MissingContainerGuid {
                logical_guid,
                container_name: route.container_name.clone(),
                container_guid: route.container_guid,
            }
        })?;
        if reader.info().version != CHUNK_FORMAT_VERSION || entry.codec != Codec::BlockDeflate {
            return Err(ManifestReadError::Container(
                ContainerError::CodecNotSupportedForLayout(entry.codec),
            ));
        }
        let payload = backend.bind_validated_reader_chunk(reader, entry.chunk_id)?;
        Ok(GpuResidentManifestAsset {
            logical_guid,
            entry,
            payload,
        })
    }

    fn route(&self, logical_guid: StableGuid) -> Result<&ManifestRoute, ManifestReadError> {
        self.manifest
            .route(logical_guid)
            .ok_or(ManifestReadError::MissingRoute(logical_guid))
    }

    fn resolve(
        &self,
        route: &ManifestRoute,
    ) -> Result<&'overlay ContainerReader, ManifestReadError> {
        self.overlays
            .latest_mount_named(&route.container_name)
            .ok_or_else(|| ManifestReadError::MissingContainer {
                logical_guid: route.guid,
                container_name: route.container_name.clone(),
            })?;

        let winner = self.overlays.winner_mount(route.container_guid);
        let winner = winner.ok_or_else(|| ManifestReadError::MissingContainerGuid {
            logical_guid: route.guid,
            container_name: route.container_name.clone(),
            container_guid: route.container_guid,
        })?;

        if winner.name != route.container_name {
            return Err(ManifestReadError::SourceMismatch {
                logical_guid: route.guid,
                expected: route.container_name.clone(),
                actual: Some(winner.name.clone()),
            });
        }

        if winner.priority != route.priority {
            return Err(ManifestReadError::PriorityMismatch {
                logical_guid: route.guid,
                container_name: route.container_name.clone(),
                manifest_priority: route.priority,
                mounted_priority: winner.priority,
            });
        }

        let info = winner.reader.info();
        let Some(expected) = self.manifest.container_identity(&route.container_name) else {
            return Err(ManifestReadError::MissingContainerIdentity {
                logical_guid: route.guid,
                container_name: route.container_name.clone(),
            });
        };
        if info.total_len != expected.byte_len
            || info.content_hash != expected.content_hash
            || info.generation != expected.generation
        {
            return Err(ManifestReadError::ContainerIdentityMismatch {
                logical_guid: route.guid,
                container_name: route.container_name.clone(),
                expected_len: expected.byte_len,
                actual_len: info.total_len,
                expected_hash: expected.content_hash,
                actual_hash: info.content_hash,
                expected_generation: expected.generation,
                actual_generation: info.generation,
            });
        }

        Ok(&winner.reader)
    }

    fn map_read_error(
        &self,
        logical_guid: StableGuid,
        offset: u64,
        len: usize,
        error: ContainerError,
    ) -> ManifestReadError {
        match error {
            ContainerError::InvalidRange | ContainerError::IntegerOverflow => {
                ManifestReadError::InvalidRange {
                    logical_guid,
                    offset,
                    len,
                }
            }
            ContainerError::MissingAsset(_) => ManifestReadError::MissingContainerGuid {
                logical_guid,
                container_name: self
                    .manifest
                    .route(logical_guid)
                    .map(|route| route.container_name.clone())
                    .unwrap_or_default(),
                container_guid: self
                    .manifest
                    .route(logical_guid)
                    .map(|route| route.container_guid)
                    .unwrap_or(StableGuid::NIL),
            },
            other => ManifestReadError::Container(other),
        }
    }
}
