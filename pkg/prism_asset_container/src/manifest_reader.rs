//! Reads logical assets through a validated [`RuntimeManifest`].
//!
//! A manifest route is an explicit binding, not a hint for the normal overlay
//! lookup. This module therefore checks the named mount, its priority, the
//! current winning source, and the route's container-local GUID before any
//! payload read is attempted.

use core::fmt;

use prism_asset::StableGuid;

use crate::{ContainerError, ContainerReader, ManifestRoute, OverlaySet, RuntimeManifest};
use crate::{RangeBatchPolicy, RangeRequest};

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

    /// Returns validated metadata for the payload selected by a logical GUID.
    ///
    /// This is used by the [`ManifestAssetReader`](crate::ManifestAssetReader)
    /// adapter so the runtime can perform size accounting without reading the
    /// complete payload first.
    pub fn entry_info(
        &self,
        logical_guid: StableGuid,
    ) -> Result<crate::EntryInfo, ManifestReadError> {
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
