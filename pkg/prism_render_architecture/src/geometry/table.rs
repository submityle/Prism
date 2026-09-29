//! Geometry-table registration/query, `ABI` validation, and cull metadata.
//!
//! The geometry table is the stable, `CPU`-side registry that maps a
//! generational [`GeometryHandle`] to its [`GeometryRecord`]. Slots are recycled
//! with a generation bump so a stale handle can never resolve to a recycled
//! record. The table also validates the geometry `ABI` version at registration
//! and can emit compact per-geometry culling metadata for the visibility pass.
//!
//! Physical `GPU` buffers stay backend-owned and are pending the GPU backend;
//! this module owns only the `CPU`-verifiable bookkeeping.

use crate::abi::GenerationalHandle;
use crate::gpu_scene::GeometryHandle;

use super::{GeometryRecord, GEOMETRY_ABI_VERSION};

/// Reason a geometry `ABI` version was rejected.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GeometryAbiError {
    /// The record predates the minimum supported `ABI` version.
    TooOld { found: u32, minimum: u32 },
    /// The record is newer than this build understands.
    TooNew { found: u32, current: u32 },
}

/// Returns `Ok` when `version` is compatible with [`GEOMETRY_ABI_VERSION`].
///
/// Compatibility is exact for the v1 contract: only the current version is
/// accepted. Older versions are [`GeometryAbiError::TooOld`], newer ones are
/// [`GeometryAbiError::TooNew`].
pub const fn check_abi(version: u32) -> Result<(), GeometryAbiError> {
    if version < GEOMETRY_ABI_VERSION {
        return Err(GeometryAbiError::TooOld {
            found: version,
            minimum: GEOMETRY_ABI_VERSION,
        });
    }
    if version > GEOMETRY_ABI_VERSION {
        return Err(GeometryAbiError::TooNew {
            found: version,
            current: GEOMETRY_ABI_VERSION,
        });
    }
    Ok(())
}

/// Error returned when resolving a handle against the table fails.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GeometryLookupError {
    /// The handle's index is outside the table.
    OutOfRange,
    /// The slot is empty (never registered or already removed).
    Vacant,
    /// The slot is live but holds a different generation (stale handle).
    StaleGeneration { expected: u32, found: u32 },
}

/// Compact per-geometry metadata consumed by the batch culling pass.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CullBatchEntry {
    /// Handle addressing the source record.
    pub handle: GeometryHandle,
    /// Total primitives across the geometry.
    pub primitive_count: u32,
    /// Number of `LOD` levels the geometry exposes.
    pub lod_count: u32,
    /// Smallest screen error among resident `LOD`s (finest available quality).
    pub finest_resident_error: f32,
    /// Whether any `LOD` is currently resident and drawable.
    pub any_resident: bool,
}

struct Slot {
    generation: u32,
    record: Option<GeometryRecord>,
}

/// Registry mapping generational handles to geometry records.
#[derive(Default)]
pub struct GeometryTable {
    slots: Vec<Slot>,
    free: Vec<u32>,
    live: u32,
}

impl GeometryTable {
    /// Creates an empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of live records currently registered.
    #[must_use]
    pub const fn len(&self) -> u32 {
        self.live
    }

    /// True when no records are registered.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Registers `record` and returns the handle that addresses it.
    ///
    /// A free slot is reused when available; otherwise the table grows. The
    /// stored record's own `handle` field is overwritten with the assigned
    /// handle so it round-trips through [`Self::get`].
    pub fn register(&mut self, mut record: GeometryRecord) -> GeometryHandle {
        let handle = if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            let handle = GenerationalHandle::new(index, slot.generation);
            record.handle = handle;
            slot.record = Some(record);
            handle
        } else {
            let index = self.slots.len() as u32;
            let handle = GenerationalHandle::new(index, 0);
            record.handle = handle;
            self.slots.push(Slot {
                generation: 0,
                record: Some(record),
            });
            handle
        };
        self.live += 1;
        handle
    }

    /// Resolves a handle to its record, distinguishing the failure modes.
    pub fn get(&self, handle: GeometryHandle) -> Result<&GeometryRecord, GeometryLookupError> {
        let slot = self
            .slots
            .get(handle.index as usize)
            .ok_or(GeometryLookupError::OutOfRange)?;
        if slot.generation != handle.generation {
            return Err(GeometryLookupError::StaleGeneration {
                expected: slot.generation,
                found: handle.generation,
            });
        }
        slot.record.as_ref().ok_or(GeometryLookupError::Vacant)
    }

    /// Convenience accessor returning `None` on any lookup failure.
    #[must_use]
    pub fn try_get(&self, handle: GeometryHandle) -> Option<&GeometryRecord> {
        self.get(handle).ok()
    }

    /// Removes and returns the record for `handle`, recycling the slot.
    ///
    /// The slot's generation is bumped so the old handle can never resolve
    /// again. Returns the failure mode when the handle does not resolve.
    pub fn remove(
        &mut self,
        handle: GeometryHandle,
    ) -> Result<GeometryRecord, GeometryLookupError> {
        let index = handle.index as usize;
        let slot = self
            .slots
            .get_mut(index)
            .ok_or(GeometryLookupError::OutOfRange)?;
        if slot.generation != handle.generation {
            return Err(GeometryLookupError::StaleGeneration {
                expected: slot.generation,
                found: handle.generation,
            });
        }
        let record = slot.record.take().ok_or(GeometryLookupError::Vacant)?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(handle.index);
        self.live -= 1;
        Ok(record)
    }

    /// Iterates the live records in slot order.
    pub fn iter(&self) -> impl Iterator<Item = &GeometryRecord> {
        self.slots.iter().filter_map(|slot| slot.record.as_ref())
    }

    /// Builds compact culling metadata for every live record, in slot order.
    #[must_use]
    pub fn cull_batch(&self) -> Vec<CullBatchEntry> {
        self.iter().map(cull_entry).collect()
    }
}

/// Derives the culling metadata for one record.
fn cull_entry(record: &GeometryRecord) -> CullBatchEntry {
    let mut finest = f32::INFINITY;
    let mut any_resident = false;
    for lod in &record.lods {
        if lod.resident {
            any_resident = true;
            if lod.screen_error < finest {
                finest = lod.screen_error;
            }
        }
    }
    let finest_resident_error = if any_resident { finest } else { 0.0 };
    CullBatchEntry {
        handle: record.handle,
        primitive_count: record.primitive_count,
        lod_count: record.lods.len() as u32,
        finest_resident_error,
        any_resident,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::GeometryLodRecord;

    fn record_with(primitive_count: u32, lods: Vec<GeometryLodRecord>) -> GeometryRecord {
        GeometryRecord {
            primitive_count,
            lods,
            ..Default::default()
        }
    }

    fn lod(level: u32, screen_error: f32, resident: bool) -> GeometryLodRecord {
        GeometryLodRecord {
            level,
            screen_error,
            resident,
            ..Default::default()
        }
    }

    #[test]
    fn abi_check_accepts_current_rejects_others() {
        assert_eq!(check_abi(GEOMETRY_ABI_VERSION), Ok(()));
        assert_eq!(
            check_abi(0),
            Err(GeometryAbiError::TooOld {
                found: 0,
                minimum: GEOMETRY_ABI_VERSION,
            })
        );
        assert_eq!(
            check_abi(GEOMETRY_ABI_VERSION + 1),
            Err(GeometryAbiError::TooNew {
                found: GEOMETRY_ABI_VERSION + 1,
                current: GEOMETRY_ABI_VERSION,
            })
        );
    }

    #[test]
    fn register_and_get_roundtrips_handle() {
        let mut table = GeometryTable::new();
        let handle = table.register(record_with(12, vec![]));
        assert_eq!(table.len(), 1);
        assert!(!table.is_empty());
        let got = table.get(handle).unwrap();
        assert_eq!(got.handle, handle);
        assert_eq!(got.primitive_count, 12);
    }

    #[test]
    fn remove_recycles_slot_and_bumps_generation() {
        let mut table = GeometryTable::new();
        let first = table.register(record_with(1, vec![]));
        let removed = table.remove(first).unwrap();
        assert_eq!(removed.primitive_count, 1);
        assert_eq!(table.len(), 0);

        // The recycled slot reuses the index with a higher generation.
        let second = table.register(record_with(2, vec![]));
        assert_eq!(second.index, first.index);
        assert_eq!(second.generation, first.generation + 1);

        // The stale handle no longer resolves.
        assert_eq!(
            table.get(first),
            Err(GeometryLookupError::StaleGeneration {
                expected: second.generation,
                found: first.generation,
            })
        );
        assert!(table.get(second).is_ok());
    }

    #[test]
    fn lookup_failure_modes() {
        let mut table = GeometryTable::new();
        let handle = table.register(record_with(1, vec![]));
        let out_of_range = GenerationalHandle::new(99, 0);
        assert_eq!(
            table.get(out_of_range),
            Err(GeometryLookupError::OutOfRange)
        );
        table.remove(handle).unwrap();
        // Removal bumps the generation, so the old handle now reads as stale.
        assert!(matches!(
            table.get(handle),
            Err(GeometryLookupError::StaleGeneration { .. })
        ));
        assert_eq!(table.try_get(handle), None);
    }

    #[test]
    fn remove_stale_handle_errors() {
        let mut table = GeometryTable::new();
        let first = table.register(record_with(1, vec![]));
        table.remove(first).unwrap();
        table.register(record_with(2, vec![]));
        assert!(matches!(
            table.remove(first),
            Err(GeometryLookupError::StaleGeneration { .. })
        ));
    }

    #[test]
    fn cull_batch_summarizes_records() {
        let mut table = GeometryTable::new();
        table.register(record_with(
            100,
            vec![lod(0, 0.5, false), lod(1, 1.5, true), lod(2, 3.0, true)],
        ));
        table.register(record_with(50, vec![lod(0, 0.2, false)]));

        let batch = table.cull_batch();
        assert_eq!(batch.len(), 2);

        assert_eq!(batch[0].primitive_count, 100);
        assert_eq!(batch[0].lod_count, 3);
        assert!(batch[0].any_resident);
        assert!((batch[0].finest_resident_error - 1.5).abs() <= 1.0e-6);

        assert_eq!(batch[1].primitive_count, 50);
        assert!(!batch[1].any_resident);
        assert!((batch[1].finest_resident_error - 0.0).abs() <= 1.0e-6);
    }

    #[test]
    fn iter_yields_only_live_records() {
        let mut table = GeometryTable::new();
        let a = table.register(record_with(1, vec![]));
        let _b = table.register(record_with(2, vec![]));
        table.remove(a).unwrap();
        let counts: Vec<u32> = table.iter().map(|r| r.primitive_count).collect();
        assert_eq!(counts, vec![2]);
    }
}
