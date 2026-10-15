use std::collections::HashMap;

use bevy_asset::AssetId;
use bevy_ecs::prelude::*;
use bevy_mesh::Mesh;
use prism_render_architecture::{geometry::GeometryRecord, gpu_scene::GeometryHandle};

/// CPU owner of stable geometry metadata; physical buffers stay backend-owned.
#[derive(Resource, Default)]
pub struct RenderGeometryRegistry {
    records: HashMap<u32, GeometryRecord>,
    assets: HashMap<AssetId<Mesh>, GeometryHandle>,
    base_assets: HashMap<AssetId<Mesh>, GeometryHandle>,
    lod_assets: HashMap<(u32, u32), AssetId<Mesh>>,
    lod_buffer_classes: HashMap<(u32, u32), (u32, u32)>,
    dirty: Vec<u32>,
    next_buffer_class: u32,
    buffer_classes: HashMap<bevy_render::render_resource::BufferId, u32>,
}

impl RenderGeometryRegistry {
    pub fn upsert(&mut self, asset: AssetId<Mesh>, record: GeometryRecord) -> bool {
        let owner = record.handle;
        let index = record.handle.index;
        if self
            .assets
            .get(&asset)
            .is_some_and(|mapped| *mapped != owner)
            || self
                .base_assets
                .get(&asset)
                .is_some_and(|mapped| *mapped != owner)
        {
            return false;
        }

        let preserved_lods: Vec<_> = self
            .lod_assets
            .iter()
            .filter_map(|(&(mapped_index, level), mapped_asset)| {
                if mapped_index != index
                    || level == 0
                    || self.assets.get(mapped_asset) != Some(&owner)
                    || !record
                        .lods
                        .iter()
                        .any(|lod| lod.level == level && lod.resident)
                {
                    return None;
                }
                Some((
                    level,
                    *mapped_asset,
                    self.lod_buffer_classes.get(&(mapped_index, level)).copied(),
                ))
            })
            .collect();

        self.assets.retain(|_, mapped| mapped.index != index);
        self.base_assets.retain(|_, mapped| mapped.index != index);
        self.lod_assets
            .retain(|(mapped_index, _), _| *mapped_index != index);
        self.lod_buffer_classes
            .retain(|(mapped_index, _), _| *mapped_index != index);

        self.assets.insert(asset, record.handle);
        self.base_assets.insert(asset, record.handle);
        for lod in &record.lods {
            if lod.resident {
                let preserved = preserved_lods
                    .iter()
                    .find(|(level, _, _)| *level == lod.level)
                    .copied();
                let mapped_asset = preserved.map_or(asset, |(_, mapped_asset, _)| mapped_asset);
                self.lod_assets.insert((index, lod.level), mapped_asset);
                self.lod_buffer_classes.insert(
                    (index, lod.level),
                    preserved.map_or(
                        (record.vertex_buffer_class, record.index_buffer_class),
                        |(_, _, classes)| {
                            classes
                                .unwrap_or((record.vertex_buffer_class, record.index_buffer_class))
                        },
                    ),
                );
                if mapped_asset != asset {
                    self.assets.insert(mapped_asset, owner);
                }
            }
        }
        self.records.insert(index, record);
        self.dirty.push(index);
        true
    }

    /// Registers a separately prepared asset for one resident LOD of an
    /// existing geometry record.
    ///
    /// The base asset remains the source of the geometry record; this mapping
    /// only changes which prepared Bevy asset supplies the physical buffers
    /// for the selected level.
    pub fn upsert_lod_asset(
        &mut self,
        handle: GeometryHandle,
        asset: AssetId<Mesh>,
        level: u32,
        vertex_buffer_class: u32,
        index_buffer_class: u32,
    ) -> bool {
        let Some(record) = self.record(handle) else {
            return false;
        };
        if !record
            .lods
            .iter()
            .any(|lod| lod.level == level && lod.resident)
        {
            return false;
        }
        if level == 0
            || self.base_assets.contains_key(&asset)
            || self
                .assets
                .get(&asset)
                .is_some_and(|mapped| *mapped != handle)
        {
            return false;
        }
        let previous = self.lod_assets.remove(&(handle.index, level));
        let removed_levels: Vec<_> = self
            .lod_assets
            .iter()
            .filter_map(|(key, mapped)| (*mapped == asset).then_some(*key))
            .collect();
        self.lod_assets.retain(|_, mapped| *mapped != asset);
        for key in removed_levels {
            self.lod_buffer_classes.remove(&key);
        }
        self.assets.insert(asset, handle);
        self.lod_assets.insert((handle.index, level), asset);
        self.lod_buffer_classes.insert(
            (handle.index, level),
            (vertex_buffer_class, index_buffer_class),
        );
        if let Some(previous) = previous
            && previous != asset
            && !self.lod_assets.values().any(|mapped| *mapped == previous)
        {
            self.assets.remove(&previous);
        }
        self.dirty.push(handle.index);
        true
    }

    pub fn retire(&mut self, asset: AssetId<Mesh>) -> Option<GeometryHandle> {
        let handle = self.base_assets.remove(&asset)?;
        self.assets.remove(&asset);
        self.records.remove(&handle.index);
        self.lod_assets
            .retain(|(index, _), mapped| *index != handle.index && *mapped != asset);
        self.lod_buffer_classes
            .retain(|(index, _), _| *index != handle.index);
        self.assets.retain(|_, mapped| *mapped != handle);
        self.dirty.push(handle.index);
        Some(handle)
    }

    /// Retires a separately prepared non-base LOD asset without destroying the
    /// shared geometry record.
    pub fn retire_lod_asset(&mut self, asset: AssetId<Mesh>) -> bool {
        if self.base_assets.contains_key(&asset) {
            return false;
        }
        let Some(handle) = self.assets.remove(&asset) else {
            return false;
        };
        let removed_levels: Vec<_> = self
            .lod_assets
            .iter()
            .filter_map(|(key, mapped)| (*mapped == asset).then_some(*key))
            .collect();
        self.lod_assets.retain(|_, mapped| *mapped != asset);
        for key in removed_levels {
            self.lod_buffer_classes.remove(&key);
        }
        self.dirty.push(handle.index);
        true
    }

    pub fn record(&self, handle: GeometryHandle) -> Option<&GeometryRecord> {
        self.records
            .get(&handle.index)
            .filter(|record| record.handle.generation == handle.generation)
    }

    pub fn handle(&self, asset: AssetId<Mesh>) -> Option<GeometryHandle> {
        self.assets.get(&asset).copied()
    }

    /// Returns the prepared asset that owns one resident LOD of `handle`.
    #[must_use]
    pub fn lod_asset(&self, handle: GeometryHandle, level: u32) -> Option<AssetId<Mesh>> {
        self.lod_assets.get(&(handle.index, level)).copied()
    }

    /// Returns the logical LOD level owned by a prepared mesh asset.
    pub(crate) fn lod_level_for_asset(
        &self,
        handle: GeometryHandle,
        asset: AssetId<Mesh>,
    ) -> Option<u32> {
        if self.base_assets.get(&asset) == Some(&handle) {
            return Some(0);
        }
        self.lod_assets
            .iter()
            .find_map(|(&(index, level), &mapped)| {
                (index == handle.index && mapped == asset).then_some(level)
            })
    }

    /// Returns the backend buffer classes for one resident LOD.
    pub fn buffer_classes_for_lod(&self, handle: GeometryHandle, level: u32) -> Option<(u32, u32)> {
        self.record(handle)
            .and_then(|_| self.lod_buffer_classes.get(&(handle.index, level)).copied())
    }

    pub(crate) fn take_dirty(&mut self) -> Vec<u32> {
        self.dirty.sort_unstable();
        self.dirty.dedup();
        std::mem::take(&mut self.dirty)
    }

    pub(crate) fn is_dirty(&self) -> bool {
        !self.dirty.is_empty()
    }

    pub fn version(&self) -> u32 {
        1
    }

    pub(crate) fn records_for_upload(&self) -> impl Iterator<Item = &GeometryRecord> {
        self.records.values()
    }

    pub(crate) fn buffer_class(&mut self, buffer: &bevy_render::render_resource::Buffer) -> u32 {
        let id = buffer.id();
        if let Some(class) = self.buffer_classes.get(&id) {
            return *class;
        }
        self.next_buffer_class = self.next_buffer_class.saturating_add(1).max(1);
        self.buffer_classes.insert(id, self.next_buffer_class);
        self.next_buffer_class
    }

    pub fn buffer_classes(&self, handle: GeometryHandle) -> Option<(u32, u32)> {
        self.record(handle)
            .map(|record| (record.vertex_buffer_class, record.index_buffer_class))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::uuid::Uuid;
    use prism_render_architecture::{abi::GenerationalHandle, geometry::GeometryLodRecord};

    #[test]
    fn registry_rejects_stale_generation_and_deduplicates_dirty_rows() {
        let asset = AssetId::Uuid {
            uuid: Uuid::from_u128(7),
        };
        let handle = GenerationalHandle {
            index: 3,
            generation: 2,
        };
        let mut registry = RenderGeometryRegistry::default();
        registry.upsert(
            asset,
            GeometryRecord {
                handle,
                ..Default::default()
            },
        );
        registry.upsert(
            asset,
            GeometryRecord {
                handle,
                revision: 2,
                ..Default::default()
            },
        );
        assert_eq!(registry.record(handle).unwrap().revision, 2);
        assert_eq!(registry.take_dirty(), vec![3]);
        assert!(registry
            .record(GenerationalHandle {
                generation: 1,
                ..handle
            })
            .is_none());
        assert_eq!(registry.retire(asset), Some(handle));
        assert_eq!(registry.take_dirty(), vec![3]);
    }

    #[test]
    fn registry_maps_each_resident_lod_to_its_prepared_asset() {
        let base = AssetId::Uuid {
            uuid: Uuid::from_u128(10),
        };
        let coarse = AssetId::Uuid {
            uuid: Uuid::from_u128(11),
        };
        let handle = GenerationalHandle::new(4, 1);
        let mut registry = RenderGeometryRegistry::default();
        registry.upsert(
            base,
            GeometryRecord {
                handle,
                lods: vec![
                    GeometryLodRecord {
                        level: 0,
                        resident: true,
                        fallback: true,
                        ..Default::default()
                    },
                    GeometryLodRecord {
                        level: 1,
                        resident: true,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
        );
        registry.upsert_lod_asset(handle, coarse, 1, 2, 3);

        assert_eq!(registry.lod_asset(handle, 0), Some(base));
        assert_eq!(registry.lod_asset(handle, 1), Some(coarse));
        assert_eq!(registry.handle(coarse), Some(handle));
    }

    #[test]
    fn replacing_a_lod_asset_removes_the_stale_mapping() {
        let first = AssetId::Uuid {
            uuid: Uuid::from_u128(20),
        };
        let second = AssetId::Uuid {
            uuid: Uuid::from_u128(21),
        };
        let handle = GenerationalHandle::new(5, 1);
        let mut registry = RenderGeometryRegistry::default();
        registry.upsert(
            first,
            GeometryRecord {
                handle,
                lods: vec![
                    GeometryLodRecord {
                        level: 0,
                        resident: true,
                        fallback: true,
                        ..Default::default()
                    },
                    GeometryLodRecord {
                        level: 1,
                        resident: true,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
        );
        registry.upsert_lod_asset(handle, second, 1, 2, 3);

        assert_eq!(registry.lod_asset(handle, 0), Some(first));
        assert_eq!(registry.lod_asset(handle, 1), Some(second));
        assert_eq!(registry.handle(first), Some(handle));
        assert_eq!(registry.handle(second), Some(handle));
    }

    #[test]
    fn replacing_a_lod_asset_keeps_the_base_mapping() {
        let base = AssetId::Uuid {
            uuid: Uuid::from_u128(30),
        };
        let first = AssetId::Uuid {
            uuid: Uuid::from_u128(31),
        };
        let second = AssetId::Uuid {
            uuid: Uuid::from_u128(32),
        };
        let handle = GenerationalHandle::new(6, 1);
        let mut registry = RenderGeometryRegistry::default();
        registry.upsert(
            base,
            GeometryRecord {
                handle,
                lods: vec![
                    GeometryLodRecord {
                        level: 0,
                        resident: true,
                        fallback: true,
                        ..Default::default()
                    },
                    GeometryLodRecord {
                        level: 1,
                        resident: true,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
        );
        assert!(registry.upsert_lod_asset(handle, first, 1, 2, 3));
        assert!(registry.upsert_lod_asset(handle, second, 1, 4, 5));

        assert_eq!(registry.lod_asset(handle, 0), Some(base));
        assert_eq!(registry.lod_asset(handle, 1), Some(second));
        assert_eq!(registry.handle(base), Some(handle));
        assert_eq!(registry.handle(first), None);
        assert_eq!(registry.buffer_classes_for_lod(handle, 1), Some((4, 5)));
    }

    #[test]
    fn moving_one_lod_asset_clears_its_old_level_metadata() {
        let base = AssetId::Uuid {
            uuid: Uuid::from_u128(33),
        };
        let lod = AssetId::Uuid {
            uuid: Uuid::from_u128(34),
        };
        let handle = GenerationalHandle::new(6, 2);
        let mut registry = RenderGeometryRegistry::default();
        assert!(registry.upsert(
            base,
            GeometryRecord {
                handle,
                lods: vec![
                    GeometryLodRecord {
                        level: 0,
                        resident: true,
                        fallback: true,
                        ..Default::default()
                    },
                    GeometryLodRecord {
                        level: 1,
                        resident: true,
                        ..Default::default()
                    },
                    GeometryLodRecord {
                        level: 2,
                        resident: true,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
        ));
        assert!(registry.upsert_lod_asset(handle, lod, 1, 7, 8));
        assert!(registry.upsert_lod_asset(handle, lod, 2, 9, 10));

        assert_eq!(registry.lod_asset(handle, 1), None);
        assert_eq!(registry.buffer_classes_for_lod(handle, 1), None);
        assert_eq!(registry.lod_asset(handle, 2), Some(lod));
        assert_eq!(registry.buffer_classes_for_lod(handle, 2), Some((9, 10)));
    }

    #[test]
    fn replacing_a_base_record_preserves_still_resident_independent_lods() {
        let first_base = AssetId::Uuid {
            uuid: Uuid::from_u128(40),
        };
        let lod = AssetId::Uuid {
            uuid: Uuid::from_u128(41),
        };
        let second_base = AssetId::Uuid {
            uuid: Uuid::from_u128(42),
        };
        let handle = GenerationalHandle::new(7, 1);
        let mut registry = RenderGeometryRegistry::default();
        let record = GeometryRecord {
            handle,
            lods: vec![
                GeometryLodRecord {
                    level: 0,
                    resident: true,
                    fallback: true,
                    ..Default::default()
                },
                GeometryLodRecord {
                    level: 1,
                    resident: true,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert!(registry.upsert(first_base, record.clone()));
        assert!(registry.upsert_lod_asset(handle, lod, 1, 8, 9));

        let replacement = GeometryRecord {
            revision: 2,
            ..record
        };
        assert!(registry.upsert(second_base, replacement));

        assert_eq!(registry.record(handle).unwrap().revision, 2);
        assert_eq!(registry.lod_asset(handle, 0), Some(second_base));
        assert_eq!(registry.lod_asset(handle, 1), Some(lod));
        assert_eq!(registry.handle(first_base), None);
        assert_eq!(registry.handle(lod), Some(handle));
        assert_eq!(registry.buffer_classes_for_lod(handle, 1), Some((8, 9)));
    }

    #[test]
    fn registry_rejects_cross_owner_asset_conflicts_without_mutation() {
        let first_base = AssetId::Uuid {
            uuid: Uuid::from_u128(50),
        };
        let shared_lod = AssetId::Uuid {
            uuid: Uuid::from_u128(51),
        };
        let second_base = AssetId::Uuid {
            uuid: Uuid::from_u128(52),
        };
        let first = GenerationalHandle::new(8, 1);
        let second = GenerationalHandle::new(9, 1);
        let record = |handle| GeometryRecord {
            handle,
            lods: vec![
                GeometryLodRecord {
                    level: 0,
                    resident: true,
                    fallback: true,
                    ..Default::default()
                },
                GeometryLodRecord {
                    level: 1,
                    resident: true,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let mut registry = RenderGeometryRegistry::default();
        assert!(registry.upsert(first_base, record(first)));
        assert!(registry.upsert_lod_asset(first, shared_lod, 1, 10, 11));
        assert!(registry.upsert(second_base, record(second)));

        assert!(!registry.upsert_lod_asset(second, shared_lod, 1, 12, 13));
        assert!(!registry.upsert(shared_lod, record(second)));

        assert_eq!(registry.handle(shared_lod), Some(first));
        assert_eq!(registry.lod_asset(first, 1), Some(shared_lod));
        assert_eq!(registry.lod_asset(second, 0), Some(second_base));
        assert_eq!(registry.lod_asset(second, 1), Some(second_base));
        assert_eq!(registry.record(second).unwrap().handle, second);
    }

    #[test]
    fn retiring_a_base_removes_its_independent_lod_assets() {
        let base = AssetId::Uuid {
            uuid: Uuid::from_u128(60),
        };
        let lod = AssetId::Uuid {
            uuid: Uuid::from_u128(61),
        };
        let handle = GenerationalHandle::new(10, 1);
        let mut registry = RenderGeometryRegistry::default();
        assert!(registry.upsert(
            base,
            GeometryRecord {
                handle,
                lods: vec![
                    GeometryLodRecord {
                        level: 0,
                        resident: true,
                        fallback: true,
                        ..Default::default()
                    },
                    GeometryLodRecord {
                        level: 1,
                        resident: true,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
        ));
        assert!(registry.upsert_lod_asset(handle, lod, 1, 14, 15));
        let _ = registry.take_dirty();

        assert_eq!(registry.retire(base), Some(handle));
        assert!(registry.record(handle).is_none());
        assert_eq!(registry.handle(base), None);
        assert_eq!(registry.handle(lod), None);
        assert_eq!(registry.lod_asset(handle, 1), None);
        assert_eq!(registry.buffer_classes_for_lod(handle, 1), None);
        assert_eq!(registry.take_dirty(), vec![10]);
    }
}
