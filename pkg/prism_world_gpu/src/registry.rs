//! Fixed-capacity CPU ownership and dense storage for sector proxies.

use std::collections::{BTreeMap, TryReserveError};
use std::fmt;
use std::ops::Range;
#[cfg(feature = "gpu")]
use std::sync::Arc;

use prism_world_stream::SectorId;

use crate::ProxyInstance;

/// Stable identity across dense compaction, valid only in its issuing registry.
///
/// Do not use a handle with another registry: registry identity is deliberately
/// not encoded, so an unrelated registry can have the same slot and generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProxyHandle {
    slot: u32,
    generation: u64,
}

impl ProxyHandle {
    /// Returns the stable registry slot, not the current dense index.
    #[must_use]
    pub const fn slot(self) -> u32 {
        self.slot
    }

    /// Returns the slot generation used to reject stale handles.
    #[must_use]
    pub const fn generation(self) -> u64 {
        self.generation
    }
}

/// Recoverable failures; no returned error changes registry state.
#[derive(Debug)]
pub enum ProxyRegistryError {
    /// Capacity does not fit the handle's slot ABI.
    CapacityTooLarge {
        /// Requested logical capacity.
        capacity: usize,
    },
    /// Requested proxies exceed reusable slots, excluding retired slots.
    CapacityExceeded {
        /// Number of proxies in the insertion.
        requested: usize,
        /// Number of currently reusable slots.
        available: usize,
    },
    /// A fallible vector reservation failed before mutation.
    AllocationFailed(TryReserveError),
    /// Even an empty registered sector must be removed before reinsertion.
    SectorAlreadyExists {
        /// Sector already registered, possibly with no proxies.
        sector: SectorId,
    },
    /// Registered sector count, including empty sectors, exhausted its budget.
    SectorCapacityExceeded {
        /// Maximum number of simultaneously registered sectors.
        capacity: usize,
    },
    /// Proxy IDs must be unique within one sector, but not across sectors.
    DuplicateProxyId {
        /// Repeated identity within the insertion.
        proxy_id: u32,
    },
    /// Proxy floating-point fields violate the CPU/GPU contract.
    InvalidProxy {
        /// Identity of the rejected proxy.
        proxy_id: u32,
    },
    /// The handle is stale, retired, or outside this registry's capacity.
    InvalidHandle {
        /// Handle that did not resolve to a live row.
        handle: ProxyHandle,
    },
    /// Updates cannot change a proxy's identity within its sector.
    ProxyIdChanged {
        /// Existing proxy identity.
        expected: u32,
        /// Identity supplied to the update.
        actual: u32,
    },
}

impl fmt::Display for ProxyRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityTooLarge { capacity } => {
                write!(f, "proxy capacity {capacity} exceeds u32::MAX")
            }
            Self::CapacityExceeded {
                requested,
                available,
            } => {
                write!(
                    f,
                    "requested {requested} proxy slots, only {available} reusable"
                )
            }
            Self::AllocationFailed(error) => write!(f, "proxy vector reservation failed: {error}"),
            Self::SectorAlreadyExists { sector } => write!(f, "sector {sector:?} already exists"),
            Self::SectorCapacityExceeded { capacity } => {
                write!(f, "registered sector count exceeds capacity {capacity}")
            }
            Self::DuplicateProxyId { proxy_id } => {
                write!(f, "duplicate sector proxy ID {proxy_id}")
            }
            Self::InvalidProxy { proxy_id } => write!(f, "invalid proxy values for ID {proxy_id}"),
            Self::InvalidHandle { handle } => write!(f, "invalid proxy handle {handle:?}"),
            Self::ProxyIdChanged { expected, actual } => {
                write!(f, "proxy update changes ID {expected} to {actual}")
            }
        }
    }
}

impl std::error::Error for ProxyRegistryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AllocationFailed(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Slot {
    generation: u64,
    dense: Option<usize>,
}

/// Sector ownership mapped onto a compact, fixed-capacity proxy prefix.
///
/// Core slot, dense, free-list and dirty vectors reserve once at construction using
/// `try_reserve_exact` and never grow beyond the requested logical capacity.
/// Per-insertion vectors also reserve fallibly before any state change.
/// Empty and nonempty sectors each consume one unit of an independent sector
/// count budget; removal returns that unit. `new` defaults this budget to
/// `capacity.max(1)`; `with_sector_capacity` configures it independently, including
/// zero to reject all sector insertions. Zero proxy capacity still supports empty
/// sectors whenever the sector budget permits.
///
/// This bounds logical proxy and sector counts, not physical metadata RSS.
/// BTreeMap metadata uses ordinary Rust allocation (OOM may abort), with allocator
/// overhead and retained memory outside any physical hard-limit guarantee.
/// With `gpu` enabled, each instance also allocates a private `Arc` identity using
/// ordinary Rust allocation (OOM may abort); non-GPU builds omit it entirely.
///
/// Handles must never cross registry instances. Removal increments generations
/// with checked arithmetic; slots reaching `u64::MAX` are permanently retired,
/// reducing reusable capacity without changing the configured `capacity()`.
/// Dense indices and ordering can change on every sector removal.
///
/// The GPU owner must upload dirty ranges **and the current count**, then
/// acknowledge only after successful synchronization. A tail-only deletion
/// changes the count even when `dirty_ranges()` is empty. Merely reading ranges
/// never acknowledges uploads.
#[derive(Debug)]
pub struct ProxyRegistry {
    #[cfg(feature = "gpu")]
    identity: Arc<()>,
    slots: Vec<Slot>,
    dense: Vec<ProxyInstance>,
    dense_slots: Vec<u32>,
    free: Vec<u32>,
    dirty: Vec<bool>,
    sectors: BTreeMap<SectorId, Vec<ProxyHandle>>,
    sector_capacity: usize,
    pending_uploads: bool,
}

/// A fully validated sector insertion or replacement that has not changed the registry yet.
///
/// The plan keeps the registry exclusively borrowed, so no edit can invalidate its reserved
/// slots between preparation and [`Self::commit`]. Dropping the plan is a no-op. Committing uses
/// only storage reserved by the registry constructor and by preparation; it has no recoverable
/// allocation or validation failure after an authoritative host accepts the corresponding cell.
pub struct PreparedSectorUpdate<'a> {
    registry: &'a mut ProxyRegistry,
    sector: SectorId,
    proxies: Vec<ProxyInstance>,
    handles: Vec<ProxyHandle>,
    existed: bool,
}

impl PreparedSectorUpdate<'_> {
    /// Atomically publishes the prepared sector and invalidates handles from its old generation.
    pub fn commit(mut self) {
        let old_handles = if self.existed {
            std::mem::take(
                self.registry
                    .sectors
                    .get_mut(&self.sector)
                    .expect("prepared sector still exists while registry is exclusively borrowed"),
            )
        } else {
            Vec::new()
        };
        self.registry.remove_handles(old_handles);

        for (&handle, proxy) in self.handles.iter().zip(self.proxies.drain(..)) {
            let index = self.registry.dense.len();
            let slot = self
                .registry
                .free
                .pop()
                .expect("prepared sector reserved enough reusable slots");
            debug_assert_eq!(slot, handle.slot);
            debug_assert_eq!(
                self.registry.slots[slot as usize].generation,
                handle.generation
            );
            self.registry.slots[slot as usize].dense = Some(index);
            self.registry.dense.push(proxy);
            self.registry.dense_slots.push(slot);
            self.registry.dirty[index] = true;
        }
        self.registry.pending_uploads |= !self.handles.is_empty();

        if self.existed {
            *self
                .registry
                .sectors
                .get_mut(&self.sector)
                .expect("prepared sector map entry remains reserved") = self.handles;
        } else {
            self.registry.sectors.insert(self.sector, self.handles);
        }
    }
}

impl ProxyRegistry {
    /// Reserves proxy storage and budgets `capacity.max(1)` registered sectors.
    /// Zero proxy capacity therefore still allows one empty sector.
    pub fn new(capacity: usize) -> Result<Self, ProxyRegistryError> {
        Self::with_sector_capacity(capacity, capacity.max(1))
    }

    /// Reserves proxy storage with an independent registered-sector count budget.
    ///
    /// Every sector, including an empty sector, consumes one budget unit until
    /// removed. A zero sector budget rejects all insertions. Map nodes are
    /// allocated on demand rather than preallocated to this count budget.
    pub fn with_sector_capacity(
        proxy_capacity: usize,
        sector_capacity: usize,
    ) -> Result<Self, ProxyRegistryError> {
        if proxy_capacity > u32::MAX as usize {
            return Err(ProxyRegistryError::CapacityTooLarge {
                capacity: proxy_capacity,
            });
        }
        let mut slots = reserved(proxy_capacity)?;
        let dense = reserved(proxy_capacity)?;
        let dense_slots = reserved(proxy_capacity)?;
        let mut free = reserved(proxy_capacity)?;
        let mut dirty = reserved(proxy_capacity)?;
        slots.resize(
            proxy_capacity,
            Slot {
                generation: 0,
                dense: None,
            },
        );
        free.extend((0..proxy_capacity as u32).rev());
        dirty.resize(proxy_capacity, false);
        Ok(Self {
            #[cfg(feature = "gpu")]
            identity: Arc::new(()),
            slots,
            dense,
            dense_slots,
            free,
            dirty,
            sectors: BTreeMap::new(),
            sector_capacity,
            pending_uploads: false,
        })
    }

    /// Registers one sector atomically, returning handles in input order.
    ///
    /// Checks for an existing sector first, then available sector budget, then
    /// proxy capacity and validity (including repeated IDs within this input).
    /// Other sectors may use the same IDs. All returned errors, including vector
    /// allocation errors, occur before any registry state changes.
    pub fn insert_sector(
        &mut self,
        sector: SectorId,
        proxies: &[ProxyInstance],
    ) -> Result<Vec<ProxyHandle>, ProxyRegistryError> {
        if self.sectors.contains_key(&sector) {
            return Err(ProxyRegistryError::SectorAlreadyExists { sector });
        }
        if self.sector_count() >= self.sector_capacity {
            return Err(ProxyRegistryError::SectorCapacityExceeded {
                capacity: self.sector_capacity,
            });
        }
        if proxies.len() > self.free.len() {
            return Err(ProxyRegistryError::CapacityExceeded {
                requested: proxies.len(),
                available: self.free.len(),
            });
        }
        let mut ids = reserved(proxies.len())?;
        for proxy in proxies {
            if !proxy.is_finite() {
                return Err(ProxyRegistryError::InvalidProxy {
                    proxy_id: proxy.proxy_id,
                });
            }
            ids.push(proxy.proxy_id);
        }
        ids.sort_unstable();
        if let Some(pair) = ids.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(ProxyRegistryError::DuplicateProxyId { proxy_id: pair[0] });
        }
        let mut handles = reserved(proxies.len())?;
        let mut sector_handles = reserved(proxies.len())?;
        for &slot in self.free.iter().rev().take(proxies.len()) {
            handles.push(ProxyHandle {
                slot,
                generation: self.slots[slot as usize].generation,
            });
        }
        sector_handles.extend_from_slice(&handles);

        // No recoverable failures or vector growth after this point. The map's
        // allocator follows ordinary Rust OOM behavior, not a fallible API.
        self.sectors.insert(sector, sector_handles);
        for (&handle, &proxy) in handles.iter().zip(proxies) {
            let index = self.dense.len();
            let slot = self.free.pop().expect("validated reusable slot count");
            debug_assert_eq!(slot, handle.slot);
            self.slots[slot as usize].dense = Some(index);
            self.dense.push(proxy);
            self.dense_slots.push(slot);
            self.dirty[index] = true;
        }
        self.pending_uploads |= !proxies.is_empty();
        Ok(handles)
    }

    /// Prepares an atomic insertion or same-sector replacement without changing registry state.
    ///
    /// Existing sector slots are treated as reusable capacity, except slots whose next generation
    /// would be `u64::MAX` and therefore must retire. Proxy validation, duplicate detection, handle
    /// storage allocation, sector-count admission, and capacity admission all complete before this
    /// method returns. Dropping the returned plan preserves every registry field and dirty bit.
    pub fn prepare_sector_update(
        &mut self,
        sector: SectorId,
        proxies: Vec<ProxyInstance>,
    ) -> Result<PreparedSectorUpdate<'_>, ProxyRegistryError> {
        let old_handles = self.sectors.get(&sector);
        let existed = old_handles.is_some();
        if !existed && self.sector_count() >= self.sector_capacity {
            return Err(ProxyRegistryError::SectorCapacityExceeded {
                capacity: self.sector_capacity,
            });
        }

        let reusable_old = old_handles
            .into_iter()
            .flatten()
            .filter(|handle| {
                self.slots[handle.slot as usize]
                    .generation
                    .checked_add(1)
                    .is_some_and(|generation| generation != u64::MAX)
            })
            .count();
        let available = self.free.len() + reusable_old;
        if proxies.len() > available {
            return Err(ProxyRegistryError::CapacityExceeded {
                requested: proxies.len(),
                available,
            });
        }

        let mut ids = reserved(proxies.len())?;
        for proxy in &proxies {
            if !proxy.is_finite() {
                return Err(ProxyRegistryError::InvalidProxy {
                    proxy_id: proxy.proxy_id,
                });
            }
            ids.push(proxy.proxy_id);
        }
        ids.sort_unstable();
        if let Some(pair) = ids.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(ProxyRegistryError::DuplicateProxyId { proxy_id: pair[0] });
        }

        let mut handles = reserved(proxies.len())?;
        if let Some(old_handles) = old_handles {
            for handle in old_handles.iter().rev() {
                let Some(generation) = self.slots[handle.slot as usize].generation.checked_add(1)
                else {
                    continue;
                };
                if generation != u64::MAX {
                    handles.push(ProxyHandle {
                        slot: handle.slot,
                        generation,
                    });
                    if handles.len() == proxies.len() {
                        break;
                    }
                }
            }
        }
        if handles.len() < proxies.len() {
            for &slot in self.free.iter().rev().take(proxies.len() - handles.len()) {
                handles.push(ProxyHandle {
                    slot,
                    generation: self.slots[slot as usize].generation,
                });
            }
        }
        debug_assert_eq!(handles.len(), proxies.len());

        Ok(PreparedSectorUpdate {
            registry: self,
            sector,
            proxies,
            handles,
            existed,
        })
    }

    /// Updates all fields except `proxy_id`; errors preserve data and dirtiness.
    pub fn update(
        &mut self,
        handle: ProxyHandle,
        proxy: ProxyInstance,
    ) -> Result<(), ProxyRegistryError> {
        let index = self
            .dense_slot(handle)
            .ok_or(ProxyRegistryError::InvalidHandle { handle })?;
        let expected = self.dense[index].proxy_id;
        if proxy.proxy_id != expected {
            return Err(ProxyRegistryError::ProxyIdChanged {
                expected,
                actual: proxy.proxy_id,
            });
        }
        if !proxy.is_finite() {
            return Err(ProxyRegistryError::InvalidProxy {
                proxy_id: proxy.proxy_id,
            });
        }
        self.dense[index] = proxy;
        self.dirty[index] = true;
        self.pending_uploads = true;
        Ok(())
    }

    /// Resolves a live handle in this registry.
    #[must_use]
    pub fn get(&self, handle: ProxyHandle) -> Option<&ProxyInstance> {
        self.dense_slot(handle).map(|index| &self.dense[index])
    }

    /// Returns the current dense index, rejecting stale and retired handles.
    #[must_use]
    pub fn dense_slot(&self, handle: ProxyHandle) -> Option<usize> {
        let slot = self.slots.get(handle.slot as usize)?;
        (slot.generation == handle.generation)
            .then_some(slot.dense)
            .flatten()
    }

    /// Resolves a current dense index back into a stable handle.
    #[must_use]
    pub fn handle_at_dense(&self, index: usize) -> Option<ProxyHandle> {
        let &slot = self.dense_slots.get(index)?;
        Some(ProxyHandle {
            slot,
            generation: self.slots[slot as usize].generation,
        })
    }

    /// Returns handles in original insertion order, including empty sectors.
    #[must_use]
    pub fn sector_handles(&self, sector: SectorId) -> Option<&[ProxyHandle]> {
        self.sectors.get(&sector).map(Vec::as_slice)
    }

    /// Returns the live dense prefix; indices are not stable identities.
    #[must_use]
    pub fn proxies(&self) -> &[ProxyInstance] {
        &self.dense
    }

    /// Returns the number of live proxies, not registered sectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    /// Returns whether there are no live proxies.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    /// Returns the configured capacity, including permanently retired slots.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Returns the independent budget for registered sectors, including empties.
    #[must_use]
    pub const fn sector_capacity(&self) -> usize {
        self.sector_capacity
    }

    /// Returns the number of registered sectors, including empty sectors, in O(1).
    #[must_use]
    pub fn sector_count(&self) -> usize {
        self.sectors.len()
    }

    /// Removes a sector and compacts storage without invalidating other handles.
    /// Returns false only when the sector was not registered.
    pub fn remove_sector(&mut self, sector: SectorId) -> bool {
        let Some(handles) = self.sectors.remove(&sector) else {
            return false;
        };
        self.remove_handles(handles);
        true
    }

    fn remove_handles(&mut self, handles: Vec<ProxyHandle>) {
        self.pending_uploads |= !handles.is_empty();
        for handle in handles {
            let index = self.slots[handle.slot as usize]
                .dense
                .take()
                .expect("live sector handle");
            self.dense.swap_remove(index);
            self.dense_slots.swap_remove(index);
            let tail = self.dense.len();
            if index < tail {
                let moved = self.dense_slots[index];
                self.slots[moved as usize].dense = Some(index);
                self.dirty[index] = true;
            }
            self.dirty[tail] = false;
            let slot = &mut self.slots[handle.slot as usize];
            let Some(next_generation) = slot.generation.checked_add(1) else {
                // Already exhausted (also covers private test construction).
                continue;
            };
            slot.generation = next_generation;
            if slot.generation != u64::MAX {
                self.free.push(handle.slot);
            }
        }
    }

    /// Coalesces adjacent dirty rows, scanning only the active dense prefix.
    /// Count changes can require synchronization even if this iterator is empty.
    pub fn dirty_ranges(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        let dirty = &self.dirty[..self.len()];
        let mut cursor = 0;
        std::iter::from_fn(move || {
            while cursor < dirty.len() && !dirty[cursor] {
                cursor += 1;
            }
            if cursor == dirty.len() {
                return None;
            }
            let start = cursor;
            while cursor < dirty.len() && dirty[cursor] {
                cursor += 1;
            }
            Some(start..cursor)
        })
    }

    /// O(1) check for pending data or count changes; does not scan dirty rows.
    #[must_use]
    pub const fn has_pending_uploads(&self) -> bool {
        self.pending_uploads
    }

    /// Forces re-upload of all live rows without clearing a pending count change.
    pub fn mark_all_dirty(&mut self) {
        let len = self.len();
        self.dirty[..len].fill(true);
        self.pending_uploads |= len != 0;
    }

    /// The GPU owner retains a clone and checks `Arc::ptr_eq` before trusting
    /// upload state or capacity. Keeping that clone alive prevents allocation ABA
    /// if safe mutable access replaces the entire registry. Moves preserve it.
    #[cfg(feature = "gpu")]
    pub(crate) fn identity(&self) -> &Arc<()> {
        &self.identity
    }

    /// Called by the GPU owner only after synchronizing both rows and count.
    #[cfg(any(feature = "gpu", test))]
    pub(crate) fn acknowledge_upload(&mut self) {
        let len = self.len();
        self.dirty[..len].fill(false);
        self.pending_uploads = false;
    }
}

fn reserved<T>(capacity: usize) -> Result<Vec<T>, ProxyRegistryError> {
    let mut vector = Vec::new();
    vector
        .try_reserve_exact(capacity)
        .map_err(ProxyRegistryError::AllocationFailed)?;
    Ok(vector)
}

#[cfg(test)]
mod registry_tests {
    include!("registry_tests.rs");
}
