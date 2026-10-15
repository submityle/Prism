use alloc::collections::{BTreeMap, BTreeSet};

use prism_asset::StableGuid;

use crate::{
    ContainerError, ContainerIdentity, ContainerReader, ManifestError, ManifestReadError,
    ManifestReader, ManifestRoute, RuntimeManifest,
};

/// A mounted container with explicit higher-wins priority.
#[derive(Clone, Debug)]
pub struct MountedContainer {
    /// Larger values take precedence over smaller values.
    pub priority: i32,
    /// Caller-provided name. Mounting the same name replaces the previous
    /// generation, so a remount cannot leave stale GUID winners reachable.
    pub name: String,
    serial: u64,
    /// Immutable reader whose bytes and TOC have been validated.
    pub reader: ContainerReader,
}

#[derive(Clone, Debug)]
struct TombstoneMount {
    priority: i32,
    name: String,
    serial: u64,
    guids: BTreeSet<StableGuid>,
}

#[derive(Clone, Copy, Debug)]
enum GuidWinner {
    Container(usize),
    Tombstone,
}

/// Runtime container overlay resolver. Higher priorities win; equal priorities
/// use most-recent mount order as the deterministic tie-breaker.
#[derive(Clone, Debug, Default)]
pub struct OverlaySet {
    mounts: Vec<MountedContainer>,
    tombstones: Vec<TombstoneMount>,
    winners: BTreeMap<StableGuid, GuidWinner>,
    next_serial: u64,
}

impl OverlaySet {
    /// Creates an empty overlay resolver.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mounts a container. Higher priority wins; ties go to the latest mount.
    pub fn mount(&mut self, name: impl Into<String>, priority: i32, reader: ContainerReader) {
        let name = name.into();
        self.mounts.retain(|mount| mount.name != name);
        self.tombstones.retain(|mount| mount.name != name);
        let serial = self.take_serial();
        self.mounts.push(MountedContainer {
            priority,
            name,
            serial,
            reader,
        });
        self.rebuild();
    }

    /// Mounts explicit patch tombstones. A tombstone hides lower-priority
    /// containers for the listed GUIDs without allocating a fake payload.
    /// Higher-priority later mounts may still restore a GUID, and unmounting
    /// this named tombstone mount restores the lower-priority winner.
    pub fn mount_tombstones(
        &mut self,
        name: impl Into<String>,
        priority: i32,
        guids: impl IntoIterator<Item = StableGuid>,
    ) {
        let name = name.into();
        self.mounts.retain(|mount| mount.name != name);
        self.tombstones.retain(|mount| mount.name != name);
        let serial = self.take_serial();
        self.tombstones.push(TombstoneMount {
            priority,
            name,
            serial,
            guids: guids.into_iter().collect(),
        });
        self.rebuild();
    }

    /// Removes every mount with `name`, rebuilding winners. Returns whether
    /// anything was removed.
    pub fn unmount(&mut self, name: &str) -> bool {
        let before = self.mounts.len();
        self.mounts.retain(|mount| mount.name != name);
        let tombstones_before = self.tombstones.len();
        self.tombstones.retain(|mount| mount.name != name);
        let removed = before != self.mounts.len() || tombstones_before != self.tombstones.len();
        if removed {
            self.rebuild();
        }
        removed
    }

    fn rebuild(&mut self) {
        self.winners.clear();
        let mut order = Vec::with_capacity(self.mounts.len() + self.tombstones.len());
        for (index, mount) in self.mounts.iter().enumerate() {
            order.push((mount.priority, mount.serial, GuidAction::Container(index)));
        }
        for (index, mount) in self.tombstones.iter().enumerate() {
            order.push((mount.priority, mount.serial, GuidAction::Tombstone(index)));
        }
        order.sort_by_key(|(priority, serial, _)| (*priority, *serial));
        for (_, _, action) in order {
            match action {
                GuidAction::Container(index) => {
                    for guid in self.mounts[index].reader.guids() {
                        self.winners.insert(guid, GuidWinner::Container(index));
                    }
                }
                GuidAction::Tombstone(index) => {
                    for guid in &self.tombstones[index].guids {
                        self.winners.insert(*guid, GuidWinner::Tombstone);
                    }
                }
            }
        }
    }

    fn take_serial(&mut self) -> u64 {
        let serial = self.next_serial;
        self.next_serial = self.next_serial.saturating_add(1);
        serial
    }

    #[must_use]
    /// Returns whether any mounted container provides `guid`.
    pub fn contains(&self, guid: StableGuid) -> bool {
        matches!(self.winners.get(&guid), Some(GuidWinner::Container(_)))
    }

    #[must_use]
    /// Returns the name of the container currently winning for `guid`.
    pub fn source(&self, guid: StableGuid) -> Option<&str> {
        self.winners.get(&guid).and_then(|winner| match winner {
            GuidWinner::Container(index) => Some(self.mounts[*index].name.as_str()),
            GuidWinner::Tombstone => None,
        })
    }

    /// Exports the current winning GUID map as a runtime manifest. The
    /// manifest is a snapshot; later mounts or unmounts do not mutate it.
    pub fn runtime_manifest(&self) -> Result<RuntimeManifest, ManifestError> {
        let mut manifest = RuntimeManifest::new();
        for (guid, winner) in &self.winners {
            let GuidWinner::Container(index) = winner else {
                continue;
            };
            let mount = &self.mounts[*index];
            manifest.insert(ManifestRoute {
                guid: *guid,
                container_name: mount.name.clone(),
                priority: mount.priority,
                container_guid: *guid,
            })?;
            let info = mount.reader.info();
            manifest.bind_container(
                mount.name.clone(),
                ContainerIdentity {
                    byte_len: info.total_len,
                    content_hash: info.content_hash,
                    generation: info.generation,
                },
            )?;
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
        match self.winners.get(&guid) {
            Some(GuidWinner::Container(index)) => Some(&self.mounts[*index]),
            Some(GuidWinner::Tombstone) | None => None,
        }
    }

    /// Reads a byte range from the winning container.
    pub fn read_range(
        &self,
        guid: StableGuid,
        offset: u64,
        out: &mut [u8],
    ) -> Result<(), ContainerError> {
        match self.winners.get(&guid) {
            Some(GuidWinner::Container(index)) => {
                self.mounts[*index].reader.read_range(guid, offset, out)
            }
            Some(GuidWinner::Tombstone) | None => Err(ContainerError::MissingAsset(guid)),
        }
    }

    /// Reads and verifies the complete payload from the winning container.
    pub fn read_all(&self, guid: StableGuid) -> Result<Vec<u8>, ContainerError> {
        match self.winners.get(&guid) {
            Some(GuidWinner::Container(index)) => self.mounts[*index].reader.read_all(guid),
            Some(GuidWinner::Tombstone) | None => Err(ContainerError::MissingAsset(guid)),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum GuidAction {
    Container(usize),
    Tombstone(usize),
}
