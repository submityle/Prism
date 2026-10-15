use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use prism_ecs::entity::Entity;
use prism_math::Aabb3;
use prism_replication::aoi::{AoiError, AoiGrid};

/// Incremental adapter around the existing replication XZ point-interest grid.
pub struct GridIndex {
    grid: AoiGrid,
    slots: BTreeMap<Entity, u32>,
    owners: Vec<Option<Entity>>,
    free: Vec<u32>,
}

impl GridIndex {
    pub(super) fn new(cell_size: f32) -> Result<Self, AoiError> {
        Ok(Self {
            grid: AoiGrid::new(cell_size)?,
            slots: BTreeMap::new(),
            owners: Vec::new(),
            free: Vec::new(),
        })
    }

    /// Existing backend for read-only diagnostics. Backend IDs are adapter slots;
    /// use this adapter's query methods for generational entity results.
    pub fn grid(&self) -> &AoiGrid {
        &self.grid
    }

    /// Last committed XZ centre for a full generational entity.
    pub fn position(&self, entity: Entity) -> Option<(f32, f32)> {
        self.slots
            .get(&entity)
            .and_then(|&slot| self.grid.position(slot))
    }

    /// Point-interest query, sorted by full entity identity.
    /// Uses the existing grid's finite-input and positive-radius policy.
    pub fn query_circle(&self, x: f32, z: f32, radius: f32) -> Vec<Entity> {
        let mut entities: Vec<_> = self
            .grid
            .query_circle(x, z, radius)
            .into_iter()
            .filter_map(|slot| self.owners[slot as usize])
            .collect();
        entities.sort_unstable();
        entities
    }

    pub(super) fn upsert(&mut self, entity: Entity, bounds: Aabb3) -> bool {
        // Multiplying before addition avoids overflow for same-sign corners.
        let x = bounds.min.x * 0.5 + bounds.max.x * 0.5;
        let z = bounds.min.z * 0.5 + bounds.max.z * 0.5;
        if self.position(entity) == Some((x, z)) {
            return false;
        }
        let slot = *self.slots.entry(entity).or_insert_with(|| {
            let slot = self.free.pop().unwrap_or_else(|| {
                let slot = u32::try_from(self.owners.len()).expect("AOI slot capacity");
                self.owners.push(None);
                slot
            });
            self.owners[slot as usize] = Some(entity);
            slot
        });
        self.grid
            .insert(slot, x, z)
            .expect("validated finite centre");
        true
    }

    pub(super) fn remove(&mut self, entity: Entity) {
        if let Some(slot) = self.slots.remove(&entity) {
            self.grid.remove(slot);
            self.owners[slot as usize] = None;
            self.free.push(slot);
        }
    }
}
