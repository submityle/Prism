use alloc::collections::BTreeMap;

use prism_asset::StableGuid;

use crate::{
    ContainerError, ContainerReader, ManifestError, ManifestReadError, ManifestReader,
    ManifestRoute, RuntimeManifest,
};

/// A mounted container with explicit higher-wins priority.
#[derive(Clone, Debug)]
pub struct MountedContainer {
    /// Larger values take precedence over smaller values.
    pub priority: i32,
    /// Caller-provided name. `unmount` removes every mount with this name.
    pub name: String,
    /// Immutable reader whose bytes and TOC have been validated.
    pub reader: ContainerReader,
}

/// Runtime container overlay resolver. Higher priorities win; equal priorities
/// use most-recent mount order as the deterministic tie-breaker.
#[derive(Clone, Debug, Default)]
pub struct OverlaySet {
    mounts: Vec<MountedContainer>,
    winners: BTreeMap<StableGuid, usize>,
}

impl OverlaySet {
    /// Creates an empty overlay resolver.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mounts a container. Higher priority wins; ties go to the latest mount.
    pub fn mount(&mut self, name: impl Into<String>, priority: i32, reader: ContainerReader) {
        self.mounts.push(MountedContainer {
            priority,
            name: name.into(),
            reader,
        });
        self.rebuild();
    }

    /// Removes every mount with `name`, rebuilding winners. Returns whether
    /// anything was removed.
    pub fn unmount(&mut self, name: &str) -> bool {
        let before = self.mounts.len();
        self.mounts.retain(|mount| mount.name != name);
        let removed = before != self.mounts.len();
        if removed {
            self.rebuild();
        }
        removed
    }

    fn rebuild(&mut self) {
        self.winners.clear();
        let mut order: Vec<usize> = (0..self.mounts.len()).collect();
        order.sort_by(|a, b| {
            self.mounts[*a]
                .priority
                .cmp(&self.mounts[*b].priority)
                .then(a.cmp(b))
        });
        for index in order {
            for guid in self.mounts[index].reader.guids() {
                self.winners.insert(guid, index);
            }
        }
    }

    #[must_use]
    /// Returns whether any mounted container provides `guid`.
    pub fn contains(&self, guid: StableGuid) -> bool {
        self.winners.contains_key(&guid)
    }

    #[must_use]
    /// Returns the name of the container currently winning for `guid`.
    pub fn source(&self, guid: StableGuid) -> Option<&str> {
        self.winners
            .get(&guid)
            .map(|index| self.mounts[*index].name.as_str())
    }

    /// Exports the current winning GUID map as a runtime manifest. The
    /// manifest is a snapshot; later mounts or unmounts do not mutate it.
    pub fn runtime_manifest(&self) -> Result<RuntimeManifest, ManifestError> {
        let mut manifest = RuntimeManifest::new();
        for (guid, index) in &self.winners {
            let mount = &self.mounts[*index];
            manifest.insert(ManifestRoute {
                guid: *guid,
                container_name: mount.name.clone(),
                priority: mount.priority,
                container_guid: *guid,
            })?;
        }
        Ok(manifest)
    }

    /// Binds a runtime manifest to this set of mounted containers.
    pub fn bind_manifest<'manifest, 'overlay>(
        &'overlay self,
        manifest: &'manifest RuntimeManifest,
    ) -> Result<ManifestReader<'manifest, 'overlay>, ManifestReadError> {
        ManifestReader::bind(manifest, self)
    }

    pub(crate) fn latest_mount_named(&self, name: &str) -> Option<&MountedContainer> {
        self.mounts.iter().rev().find(|mount| mount.name == name)
    }

    pub(crate) fn winner_mount(&self, guid: StableGuid) -> Option<&MountedContainer> {
        self.winners.get(&guid).map(|index| &self.mounts[*index])
    }

    /// Reads a byte range from the winning container.
    pub fn read_range(
        &self,
        guid: StableGuid,
        offset: u64,
        out: &mut [u8],
    ) -> Result<(), ContainerError> {
        let index = self
            .winners
            .get(&guid)
            .ok_or(ContainerError::MissingAsset(guid))?;
        self.mounts[*index].reader.read_range(guid, offset, out)
    }

    /// Reads and verifies the complete payload from the winning container.
    pub fn read_all(&self, guid: StableGuid) -> Result<Vec<u8>, ContainerError> {
        let index = self
            .winners
            .get(&guid)
            .ok_or(ContainerError::MissingAsset(guid))?;
        self.mounts[*index].reader.read_all(guid)
    }
}
