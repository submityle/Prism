//! Strict, deterministic runtime routing manifest.
//!
//! The manifest is an in-memory content index serialized independently from a
//! container. It routes a stable asset identity to a container name, priority,
//! and the GUID used by that container. It does not load containers or perform
//! I/O.

use alloc::collections::{BTreeMap, BTreeSet};
use core::fmt;

use prism_asset::StableGuid;

use crate::{ManifestReadError, ManifestReader, OverlaySet};

/// Eight-byte discriminator for runtime manifests.
pub const MANIFEST_MAGIC: [u8; 8] = *b"PRSMAN\0\0";
/// Current runtime manifest wire version.
pub const MANIFEST_VERSION: u16 = 3;
const HEADER_LEN: usize = 20;
const CONTAINER_FIXED_LEN: usize = 50;
const ROUTE_FIXED_LEN: usize = 38;
const ROUTE_METADATA_FIXED_LEN: usize = 24;
const HASH_LEN: usize = 32;

/// One asset routing record. `container_guid` may differ from `guid` when an
/// external logical identity maps to a container-local record identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestRoute {
    /// Stable logical asset identity.
    pub guid: StableGuid,
    /// Name used to identify the mounted container.
    pub container_name: String,
    /// Larger priorities take precedence when resolving competing routes.
    pub priority: i32,
    /// GUID of the record inside `container_name`.
    pub container_guid: StableGuid,
}

/// Runtime metadata persisted for one logical route.
///
/// The dependency lists are route GUIDs, not build keys, so the runtime never
/// has to reconstruct build-time identity. Zero residency/streaming/capability
/// values mean that the package has no additional admission requirement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestMetadata {
    /// Required route dependencies that must be available before this route.
    pub hard_dependencies: Vec<StableGuid>,
    /// Optional route dependencies that may be prefetched or installed later.
    pub optional_dependencies: Vec<StableGuid>,
    /// Initial residency tier requested by the package.
    pub initial_residency: u8,
    /// Streaming quality/LOD tier available at startup.
    pub streaming_level: u8,
    /// Payload codec declared for this route.
    pub codec: crate::Codec,
    /// Runtime schema version for the route payload.
    pub schema: u32,
    /// Platform installation capability bitset.
    pub install_capabilities: u64,
}

impl Default for ManifestMetadata {
    fn default() -> Self {
        Self {
            hard_dependencies: Vec::new(),
            optional_dependencies: Vec::new(),
            initial_residency: 0,
            streaming_level: 0,
            codec: crate::Codec::Raw,
            schema: 0,
            install_capabilities: 0,
        }
    }
}

/// Immutable identity of one runtime container referenced by a manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContainerIdentity {
    /// Exact byte length of the container.
    pub byte_len: u64,
    /// BLAKE3 digest of the exact container bytes.
    pub content_hash: [u8; 32],
    /// Stable content-derived generation token.
    pub generation: u64,
}

/// Canonical GUID-sorted content routing index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeManifest {
    routes: BTreeMap<StableGuid, ManifestRoute>,
    metadata: BTreeMap<StableGuid, ManifestMetadata>,
    containers: BTreeMap<String, ContainerIdentity>,
}

/// Errors specific to manifest encoding and decoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ManifestError {
    /// Input ended before the declared manifest was complete.
    Truncated,
    /// Manifest discriminator did not match.
    BadMagic,
    /// Manifest uses an unsupported wire version.
    UnsupportedVersion(u16),
    /// Header, field encoding, or canonical ordering is invalid.
    InvalidFormat,
    /// A route names a codec not implemented by this manifest reader.
    UnsupportedCodec(u8),
    /// A stable GUID has more than one route.
    DuplicateGuid(StableGuid),
    /// A container name has more than one immutable identity.
    ContainerIdentityConflict(String),
    /// A route references a container absent from the identity table.
    MissingContainerIdentity(String),
    /// The identity table contains a container that no route references.
    UnreferencedContainerIdentity(String),
    /// Name length cannot be represented by the wire format.
    NameTooLong,
    /// Name is empty, contains a NUL, or is not UTF-8.
    InvalidName,
    /// BLAKE3 digest for the manifest body does not match.
    HashMismatch,
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("runtime manifest is truncated"),
            Self::BadMagic => f.write_str("invalid runtime manifest magic"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported runtime manifest version {v}"),
            Self::InvalidFormat => f.write_str("invalid runtime manifest format"),
            Self::UnsupportedCodec(codec) => {
                write!(f, "unsupported runtime manifest codec {codec}")
            }
            Self::DuplicateGuid(g) => write!(f, "duplicate manifest guid {g:?}"),
            Self::ContainerIdentityConflict(name) => {
                write!(f, "conflicting identity for runtime container {name:?}")
            }
            Self::MissingContainerIdentity(name) => {
                write!(f, "runtime manifest has no identity for container {name:?}")
            }
            Self::UnreferencedContainerIdentity(name) => {
                write!(
                    f,
                    "runtime manifest has an unreferenced container identity {name:?}"
                )
            }
            Self::NameTooLong => f.write_str("runtime manifest container name is too long"),
            Self::InvalidName => f.write_str("invalid runtime manifest container name"),
            Self::HashMismatch => f.write_str("runtime manifest hash mismatch"),
        }
    }
}

impl std::error::Error for ManifestError {}

impl RuntimeManifest {
    /// Creates an empty manifest.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a unique logical GUID route. Duplicate identities are rejected.
    pub fn insert(&mut self, route: ManifestRoute) -> Result<(), ManifestError> {
        self.insert_with_metadata(route, ManifestMetadata::default())
    }

    /// Adds a route and its persisted runtime metadata.
    pub fn insert_with_metadata(
        &mut self,
        route: ManifestRoute,
        metadata: ManifestMetadata,
    ) -> Result<(), ManifestError> {
        validate_name(&route.container_name)?;
        validate_metadata(&metadata)?;
        if self.routes.contains_key(&route.guid) {
            return Err(ManifestError::DuplicateGuid(route.guid));
        }
        self.metadata.insert(route.guid, metadata);
        self.routes.insert(route.guid, route);
        Ok(())
    }

    /// Replaces the metadata for an existing route.
    pub fn set_metadata(
        &mut self,
        guid: StableGuid,
        metadata: ManifestMetadata,
    ) -> Result<(), ManifestError> {
        if !self.routes.contains_key(&guid) {
            return Err(ManifestError::InvalidFormat);
        }
        validate_metadata(&metadata)?;
        self.metadata.insert(guid, metadata);
        Ok(())
    }

    /// Returns the persisted runtime metadata for a route.
    #[must_use]
    pub fn metadata(&self, guid: StableGuid) -> Option<&ManifestMetadata> {
        self.metadata.get(&guid)
    }

    /// Binds one immutable container identity to the routes that reference it.
    pub fn bind_container(
        &mut self,
        name: impl Into<String>,
        identity: ContainerIdentity,
    ) -> Result<(), ManifestError> {
        let name = name.into();
        validate_name(&name)?;
        if let Some(existing) = self.containers.get(&name)
            && *existing != identity
        {
            return Err(ManifestError::ContainerIdentityConflict(name));
        }
        self.containers.insert(name, identity);
        Ok(())
    }

    /// Returns the immutable identity bound to a container name.
    #[must_use]
    pub fn container_identity(&self, name: &str) -> Option<ContainerIdentity> {
        self.containers.get(name).copied()
    }

    /// Looks up the route for a logical stable GUID.
    #[must_use]
    pub fn route(&self, guid: StableGuid) -> Option<&ManifestRoute> {
        self.routes.get(&guid)
    }

    /// Applies a delta manifest to this snapshot and removes the supplied
    /// logical routes from the base revision.
    ///
    /// The operation is transactional: if the patch introduces an invalid
    /// dependency, container identity, or route, the receiver is unchanged.
    /// Patch routes are applied after removals, so a later patch may restore a
    /// GUID that an older patch tombstoned.
    pub fn apply_patch(
        &mut self,
        patch: &RuntimeManifest,
        removed_guids: &[StableGuid],
    ) -> Result<(), ManifestError> {
        let mut next = self.clone();
        let mut previous = None;
        for guid in removed_guids {
            if guid.is_nil() || previous.is_some_and(|previous| previous >= *guid) {
                return Err(ManifestError::InvalidFormat);
            }
            previous = Some(*guid);
            next.routes.remove(guid);
            next.metadata.remove(guid);
        }

        for (name, identity) in &patch.containers {
            if let Some(existing) = next.containers.get(name)
                && existing != identity
            {
                return Err(ManifestError::ContainerIdentityConflict(name.clone()));
            }
            next.containers.insert(name.clone(), *identity);
        }

        for (guid, route) in &patch.routes {
            let metadata = patch
                .metadata
                .get(guid)
                .ok_or(ManifestError::InvalidFormat)?;
            validate_name(&route.container_name)?;
            validate_metadata(metadata)?;
            if !next.containers.contains_key(&route.container_name) {
                return Err(ManifestError::MissingContainerIdentity(
                    route.container_name.clone(),
                ));
            }
            next.routes.insert(*guid, route.clone());
            next.metadata.insert(*guid, metadata.clone());
        }

        let referenced = next
            .routes
            .values()
            .map(|route| route.container_name.as_str())
            .collect::<BTreeSet<_>>();
        next.containers
            .retain(|name, _| referenced.contains(name.as_str()));
        validate_dependency_graph(&next.routes, &next.metadata)?;
        for route in next.routes.values() {
            if !next.containers.contains_key(&route.container_name) {
                return Err(ManifestError::MissingContainerIdentity(
                    route.container_name.clone(),
                ));
            }
        }
        *self = next;
        Ok(())
    }

    /// Iterates routes in canonical `StableGuid` order.
    pub fn routes(&self) -> impl Iterator<Item = &ManifestRoute> {
        self.routes.values()
    }

    /// Constructs a snapshot of the currently winning routes in an overlay.
    #[must_use = "the derived manifest is the overlay snapshot"]
    pub fn from_overlay(overlays: &OverlaySet) -> Result<Self, ManifestError> {
        overlays.runtime_manifest()
    }

    /// Binds this logical routing table to the currently mounted containers.
    ///
    /// Binding validates every route before returning. The returned reader also
    /// validates the selected route before each read, so an unmount or remount
    /// cannot silently redirect a logical asset to a different container.
    #[expect(
        clippy::result_large_err,
        reason = "manifest binding errors retain complete route and identity diagnostics"
    )]
    pub fn bind<'manifest, 'overlay>(
        &'manifest self,
        overlays: &'overlay OverlaySet,
    ) -> Result<ManifestReader<'manifest, 'overlay>, ManifestReadError> {
        ManifestReader::bind(self, overlays)
    }

    /// Encodes canonical little-endian bytes with a BLAKE3 digest over the
    /// complete header and route body.
    pub fn encode(&self) -> Result<Vec<u8>, ManifestError> {
        validate_dependency_graph(&self.routes, &self.metadata)?;
        let container_count =
            u32::try_from(self.containers.len()).map_err(|_| ManifestError::InvalidFormat)?;
        let route_count =
            u32::try_from(self.routes.len()).map_err(|_| ManifestError::InvalidFormat)?;
        let mut out = Vec::new();
        out.extend_from_slice(&MANIFEST_MAGIC);
        out.extend_from_slice(&MANIFEST_VERSION.to_le_bytes());
        out.extend_from_slice(&0_u16.to_le_bytes());
        out.extend_from_slice(&container_count.to_le_bytes());
        out.extend_from_slice(&route_count.to_le_bytes());
        for (name, identity) in &self.containers {
            validate_name(name)?;
            let name_bytes = name.as_bytes();
            let name_len =
                u16::try_from(name_bytes.len()).map_err(|_| ManifestError::NameTooLong)?;
            out.extend_from_slice(&name_len.to_le_bytes());
            out.extend_from_slice(&identity.byte_len.to_le_bytes());
            out.extend_from_slice(&identity.content_hash);
            out.extend_from_slice(&identity.generation.to_le_bytes());
            out.extend_from_slice(name_bytes);
        }
        for route in self.routes.values() {
            validate_name(&route.container_name)?;
            if !self.containers.contains_key(&route.container_name) {
                return Err(ManifestError::MissingContainerIdentity(
                    route.container_name.clone(),
                ));
            }
            let name = route.container_name.as_bytes();
            let name_len = u16::try_from(name.len()).map_err(|_| ManifestError::NameTooLong)?;
            out.extend_from_slice(&route.guid.to_u128().to_le_bytes());
            out.extend_from_slice(&route.container_guid.to_u128().to_le_bytes());
            out.extend_from_slice(&route.priority.to_le_bytes());
            out.extend_from_slice(&name_len.to_le_bytes());
            out.extend_from_slice(name);
            let metadata = self
                .metadata
                .get(&route.guid)
                .ok_or(ManifestError::InvalidFormat)?;
            validate_metadata(metadata)?;
            let hard_count = u32::try_from(metadata.hard_dependencies.len())
                .map_err(|_| ManifestError::InvalidFormat)?;
            let optional_count = u32::try_from(metadata.optional_dependencies.len())
                .map_err(|_| ManifestError::InvalidFormat)?;
            out.extend_from_slice(&hard_count.to_le_bytes());
            out.extend_from_slice(&optional_count.to_le_bytes());
            out.push(metadata.initial_residency);
            out.push(metadata.streaming_level);
            out.push(metadata.codec as u8);
            out.push(0);
            out.extend_from_slice(&metadata.schema.to_le_bytes());
            out.extend_from_slice(&metadata.install_capabilities.to_le_bytes());
            for dependency in metadata
                .hard_dependencies
                .iter()
                .chain(&metadata.optional_dependencies)
            {
                if !self.routes.contains_key(dependency) {
                    return Err(ManifestError::InvalidFormat);
                }
                out.extend_from_slice(&dependency.to_u128().to_le_bytes());
            }
        }
        for name in self.containers.keys() {
            if !self
                .routes
                .values()
                .any(|route| route.container_name == *name)
            {
                return Err(ManifestError::UnreferencedContainerIdentity(name.clone()));
            }
        }
        let digest = blake3::hash(&out);
        out.extend_from_slice(digest.as_bytes());
        Ok(out)
    }

    /// Decodes and strictly validates one complete manifest.
    pub fn decode(bytes: &[u8]) -> Result<Self, ManifestError> {
        if bytes.len() < HEADER_LEN {
            return Err(ManifestError::Truncated);
        }
        if bytes[..8] != MANIFEST_MAGIC {
            return Err(ManifestError::BadMagic);
        }
        let version = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
        if version != 2 && version != MANIFEST_VERSION {
            return Err(ManifestError::UnsupportedVersion(version));
        }
        if bytes[10..12] != [0, 0] {
            return Err(ManifestError::InvalidFormat);
        }
        let container_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let route_count = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
        let body_end = bytes
            .len()
            .checked_sub(HASH_LEN)
            .ok_or(ManifestError::Truncated)?;
        if body_end < HEADER_LEN {
            return Err(ManifestError::Truncated);
        }
        let mut cursor = HEADER_LEN;
        let mut containers = BTreeMap::new();
        let mut previous_container = None;
        for _ in 0..container_count {
            let fixed_end = cursor
                .checked_add(CONTAINER_FIXED_LEN)
                .ok_or(ManifestError::InvalidFormat)?;
            let fixed = bytes
                .get(cursor..fixed_end)
                .filter(|_| fixed_end <= body_end)
                .ok_or(ManifestError::Truncated)?;
            let name_len = u16::from_le_bytes(fixed[..2].try_into().unwrap()) as usize;
            let byte_len = u64::from_le_bytes(fixed[2..10].try_into().unwrap());
            let content_hash = fixed[10..42].try_into().unwrap();
            let generation = u64::from_le_bytes(fixed[42..50].try_into().unwrap());
            let name_end = fixed_end
                .checked_add(name_len)
                .ok_or(ManifestError::InvalidFormat)?;
            if name_end > body_end {
                return Err(ManifestError::Truncated);
            }
            let name = std::str::from_utf8(&bytes[fixed_end..name_end])
                .map_err(|_| ManifestError::InvalidName)?;
            validate_name(name)?;
            if previous_container.is_some_and(|previous| previous >= name) {
                return Err(ManifestError::InvalidFormat);
            }
            previous_container = Some(name);
            if containers
                .insert(
                    name.to_owned(),
                    ContainerIdentity {
                        byte_len,
                        content_hash,
                        generation,
                    },
                )
                .is_some()
            {
                return Err(ManifestError::InvalidFormat);
            }
            cursor = name_end;
        }
        let mut routes = BTreeMap::new();
        let mut metadata_map = BTreeMap::new();
        let mut previous = None;
        for _ in 0..route_count {
            let fixed_end = cursor
                .checked_add(ROUTE_FIXED_LEN)
                .ok_or(ManifestError::InvalidFormat)?;
            let fixed = bytes
                .get(cursor..fixed_end)
                .filter(|_| fixed_end <= body_end)
                .ok_or(ManifestError::Truncated)?;
            let guid = StableGuid::from_u128(u128::from_le_bytes(fixed[..16].try_into().unwrap()));
            let container_guid =
                StableGuid::from_u128(u128::from_le_bytes(fixed[16..32].try_into().unwrap()));
            let priority = i32::from_le_bytes(fixed[32..36].try_into().unwrap());
            let name_len = u16::from_le_bytes(fixed[36..38].try_into().unwrap()) as usize;
            let name_end = fixed_end
                .checked_add(name_len)
                .ok_or(ManifestError::InvalidFormat)?;
            if name_end > body_end {
                return Err(ManifestError::Truncated);
            }
            let name = std::str::from_utf8(&bytes[fixed_end..name_end])
                .map_err(|_| ManifestError::InvalidName)?;
            validate_name(name)?;
            if routes.contains_key(&guid) {
                return Err(ManifestError::DuplicateGuid(guid));
            }
            if previous.is_some_and(|value| value >= guid) {
                return Err(ManifestError::InvalidFormat);
            }
            previous = Some(guid);
            routes.insert(
                guid,
                ManifestRoute {
                    guid,
                    container_name: name.to_owned(),
                    priority,
                    container_guid,
                },
            );
            cursor = name_end;
            let metadata = if version == 2 {
                ManifestMetadata::default()
            } else {
                let metadata_end = cursor
                    .checked_add(ROUTE_METADATA_FIXED_LEN)
                    .ok_or(ManifestError::InvalidFormat)?;
                let fixed = bytes
                    .get(cursor..metadata_end)
                    .filter(|_| metadata_end <= body_end)
                    .ok_or(ManifestError::Truncated)?;
                let hard_count = u32::from_le_bytes(fixed[..4].try_into().unwrap()) as usize;
                let optional_count = u32::from_le_bytes(fixed[4..8].try_into().unwrap()) as usize;
                let total = hard_count
                    .checked_add(optional_count)
                    .ok_or(ManifestError::InvalidFormat)?;
                let dep_bytes = total.checked_mul(16).ok_or(ManifestError::InvalidFormat)?;
                let deps_end = metadata_end
                    .checked_add(dep_bytes)
                    .ok_or(ManifestError::InvalidFormat)?;
                if deps_end > body_end {
                    return Err(ManifestError::Truncated);
                }
                let mut dependencies = Vec::with_capacity(total);
                let mut dep_cursor = metadata_end;
                for _ in 0..total {
                    dependencies.push(StableGuid::from_u128(u128::from_le_bytes(
                        bytes[dep_cursor..dep_cursor + 16].try_into().unwrap(),
                    )));
                    dep_cursor += 16;
                }
                cursor = deps_end;
                let metadata = ManifestMetadata {
                    hard_dependencies: dependencies[..hard_count].to_vec(),
                    optional_dependencies: dependencies[hard_count..].to_vec(),
                    initial_residency: fixed[8],
                    streaming_level: fixed[9],
                    codec: crate::Codec::decode(fixed[10]).map_err(|error| match error {
                        crate::ContainerError::UnsupportedCodec(codec) => {
                            ManifestError::UnsupportedCodec(codec)
                        }
                        _ => ManifestError::InvalidFormat,
                    })?,
                    schema: u32::from_le_bytes(fixed[12..16].try_into().unwrap()),
                    install_capabilities: u64::from_le_bytes(fixed[16..24].try_into().unwrap()),
                };
                validate_metadata(&metadata)?;
                metadata
            };
            metadata_map.insert(guid, metadata);
        }
        if cursor > body_end {
            return Err(ManifestError::Truncated);
        }
        if cursor < body_end {
            return Err(ManifestError::InvalidFormat);
        }
        let expected_hash = blake3::hash(&bytes[..body_end]);
        if expected_hash.as_bytes() != &bytes[body_end..] {
            return Err(ManifestError::HashMismatch);
        }
        for route in routes.values() {
            if !containers.contains_key(&route.container_name) {
                return Err(ManifestError::MissingContainerIdentity(
                    route.container_name.clone(),
                ));
            }
        }
        for metadata in metadata_map.values() {
            for dependency in metadata
                .hard_dependencies
                .iter()
                .chain(&metadata.optional_dependencies)
            {
                if !routes.contains_key(dependency) {
                    return Err(ManifestError::InvalidFormat);
                }
            }
        }
        validate_dependency_graph(&routes, &metadata_map)?;
        for name in containers.keys() {
            if !routes.values().any(|route| route.container_name == *name) {
                return Err(ManifestError::UnreferencedContainerIdentity(name.clone()));
            }
        }
        Ok(Self {
            routes,
            metadata: metadata_map,
            containers,
        })
    }
}

fn validate_dependency_graph(
    routes: &BTreeMap<StableGuid, ManifestRoute>,
    metadata: &BTreeMap<StableGuid, ManifestMetadata>,
) -> Result<(), ManifestError> {
    let mut indegree = routes
        .keys()
        .copied()
        .map(|guid| (guid, 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut dependents = BTreeMap::<StableGuid, BTreeSet<StableGuid>>::new();
    for (guid, route_metadata) in metadata {
        for dependency in route_metadata
            .hard_dependencies
            .iter()
            .chain(&route_metadata.optional_dependencies)
        {
            if *dependency == *guid || !routes.contains_key(dependency) {
                return Err(ManifestError::InvalidFormat);
            }
            if dependents.entry(*dependency).or_default().insert(*guid) {
                *indegree.get_mut(guid).ok_or(ManifestError::InvalidFormat)? += 1;
            }
        }
    }
    let mut ready = indegree
        .iter()
        .filter_map(|(guid, degree)| (*degree == 0).then_some(*guid))
        .collect::<BTreeSet<_>>();
    let mut visited = 0usize;
    while let Some(guid) = ready.pop_first() {
        visited += 1;
        if let Some(children) = dependents.get(&guid) {
            for child in children {
                let degree = indegree
                    .get_mut(child)
                    .ok_or(ManifestError::InvalidFormat)?;
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(*child);
                }
            }
        }
    }
    (visited == routes.len())
        .then_some(())
        .ok_or(ManifestError::InvalidFormat)
}

fn validate_metadata(metadata: &ManifestMetadata) -> Result<(), ManifestError> {
    let mut seen = BTreeMap::new();
    for dependency in &metadata.hard_dependencies {
        if dependency.is_nil() || seen.insert(*dependency, ()).is_some() {
            return Err(ManifestError::InvalidFormat);
        }
    }
    for dependency in &metadata.optional_dependencies {
        if dependency.is_nil() || seen.insert(*dependency, ()).is_some() {
            return Err(ManifestError::InvalidFormat);
        }
    }
    Ok(())
}

fn validate_name(name: &str) -> Result<(), ManifestError> {
    if name.is_empty()
        || name.as_bytes().contains(&0)
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
    {
        return Err(ManifestError::InvalidName);
    }
    if name.len() > u16::MAX as usize {
        return Err(ManifestError::NameTooLong);
    }
    Ok(())
}
