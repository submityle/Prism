//! Deterministic CPU contract for virtual lighting-page streaming (P4).
//!
//! The module is self-contained so it can be tested before being registered
//! in the lighting module tree. All ordering and eviction decisions are stable.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

const SNAPSHOT_VERSION: u32 = 1;

/// Stable identity of a spatial streaming cell.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CellId(pub u64);

/// Stable identity of a virtual representation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RepId(pub u64);

/// Content source for a representation. The token is opaque to the streamer.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RepProvider {
    pub kind: ProviderKind,
    pub token: u64,
    pub generation: u32,
}

impl RepProvider {
    #[must_use]
    pub const fn new(kind: ProviderKind, token: u64, generation: u32) -> Self {
        Self {
            kind,
            token,
            generation,
        }
    }
}

/// Provider domain for a virtual representation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ProviderKind {
    Asset,
    Generated,
    External,
}

/// Page family attached to a cell/representation pair.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PageKind {
    Shadow,
    Gi,
    Cache,
}

/// Stable virtual page address.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct PageId {
    pub kind: PageKind,
    pub rep: RepId,
    pub page: u32,
}

/// Per-representation epoch used to reject stale page data.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct RepEpoch(pub u64);

/// Cell binding to one representation and its page range in each page family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CellBinding {
    pub cell: CellId,
    pub rep: RepId,
    pub provider: RepProvider,
    pub epoch: RepEpoch,
    pub shadow: Option<PageRange>,
    pub gi: Option<PageRange>,
    pub cache: Option<PageRange>,
}

/// Half-open virtual page range. Empty ranges are rejected when bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageRange {
    pub first: u32,
    pub count: u32,
}

impl PageRange {
    #[must_use]
    pub const fn new(first: u32, count: u32) -> Self {
        Self { first, count }
    }

    fn end(self) -> Option<u32> {
        self.first.checked_add(self.count)
    }
}

/// Deterministic importance; larger values are serviced first.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct RequestPriority(pub u32);

/// A consumer request for a bound virtual page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageRequest {
    pub cell: CellId,
    pub page: PageId,
    pub epoch: RepEpoch,
    pub priority: RequestPriority,
    pub frame: u64,
}

/// Resident/nonresident page state, including the explicit fallback decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Residency {
    Resident {
        physical: u32,
        epoch: RepEpoch,
    },
    Nonresident {
        fallback: Fallback,
        reason: NonresidentReason,
    },
}

/// Rendering strategy used when a virtual page cannot be read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fallback {
    ShadowCascade,
    ShadowRay,
    GiProbe,
    GiCoarse,
    CacheRecompute,
    CacheUnfiltered,
}

/// Explicit cause for a page being unavailable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NonresidentReason {
    NotRequested,
    Capacity,
    StaleEpoch,
    DeviceLost,
    UnknownBinding,
}

/// Configured bound for the shared physical-page pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PagePoolConfig {
    pub capacity: u32,
}

/// Errors from configuration, binding, request, or restore validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamError {
    InvalidCapacity,
    InvalidRange,
    DuplicateBinding,
    UnknownBinding,
    PageNotBound,
    StaleRequest,
    SnapshotVersion,
    SnapshotCapacity,
    DuplicateSnapshotPage,
    InvalidPhysicalPage,
}

/// One deterministic snapshot mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapshotPage {
    pub page: PageId,
    pub physical: u32,
    pub epoch: RepEpoch,
    pub priority: RequestPriority,
    pub last_used_frame: u64,
}

/// Versioned device-recovery snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamSnapshot {
    pub version: u32,
    pub capacity: u32,
    pub device_epoch: u64,
    pub pages: Vec<SnapshotPage>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResidentPage {
    physical: u32,
    epoch: RepEpoch,
    priority: RequestPriority,
    last_used_frame: u64,
}

/// Bounded virtual page table and deterministic allocator.
#[derive(Clone, Debug)]
pub struct VirtualPagePool {
    config: PagePoolConfig,
    bindings: BTreeMap<(CellId, RepId), CellBinding>,
    residents: BTreeMap<PageId, ResidentPage>,
    device_epoch: u64,
    device_lost: bool,
}

impl VirtualPagePool {
    pub fn new(config: PagePoolConfig) -> Result<Self, StreamError> {
        if config.capacity == 0 {
            return Err(StreamError::InvalidCapacity);
        }
        Ok(Self {
            config,
            bindings: BTreeMap::new(),
            residents: BTreeMap::new(),
            device_epoch: 0,
            device_lost: false,
        })
    }

    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.config.capacity
    }

    #[must_use]
    pub fn resident_count(&self) -> usize {
        self.residents.len()
    }

    pub fn bind(&mut self, binding: CellBinding) -> Result<(), StreamError> {
        for range in [binding.shadow, binding.gi, binding.cache]
            .into_iter()
            .flatten()
        {
            if range.count == 0 || range.end().is_none() {
                return Err(StreamError::InvalidRange);
            }
        }
        let key = (binding.cell, binding.rep);
        if self.bindings.contains_key(&key) {
            return Err(StreamError::DuplicateBinding);
        }
        self.bindings.insert(key, binding);
        Ok(())
    }

    /// Bumps a representation epoch and removes all of its stale resident pages.
    pub fn invalidate(&mut self, rep: RepId, new_epoch: RepEpoch) -> usize {
        for binding in self.bindings.values_mut().filter(|b| b.rep == rep) {
            binding.epoch = new_epoch;
        }
        let before = self.residents.len();
        self.residents
            .retain(|page, resident| page.rep != rep || resident.epoch == new_epoch);
        before - self.residents.len()
    }

    /// Merge duplicate page requests (maximum priority and newest frame), then
    /// allocate in priority-descending/page-address order.
    pub fn request(
        &mut self,
        requests: &[PageRequest],
    ) -> Result<Vec<(PageId, Residency)>, StreamError> {
        let mut merged: BTreeMap<PageId, PageRequest> = BTreeMap::new();
        for request in requests {
            if self.device_lost {
                continue;
            }
            let binding = self
                .bindings
                .get(&(request.cell, request.page.rep))
                .ok_or(StreamError::UnknownBinding)?;
            if binding.epoch != request.epoch {
                return Err(StreamError::StaleRequest);
            }
            if !page_is_bound(*binding, request.page) {
                return Err(StreamError::PageNotBound);
            }
            merged
                .entry(request.page)
                .and_modify(|current| {
                    if request.priority > current.priority {
                        current.priority = request.priority;
                    }
                    current.frame = current.frame.max(request.frame);
                    current.epoch = request.epoch;
                })
                .or_insert(*request);
        }
        let mut ordered: Vec<PageRequest> = merged.into_values().collect();
        ordered.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.page.cmp(&b.page))
        });
        let mut result = Vec::with_capacity(ordered.len());
        for request in ordered {
            if let Some(resident) = self.residents.get_mut(&request.page) {
                if resident.epoch == request.epoch {
                    resident.priority = request.priority;
                    resident.last_used_frame = request.frame;
                    result.push((
                        request.page,
                        Residency::Resident {
                            physical: resident.physical,
                            epoch: resident.epoch,
                        },
                    ));
                    continue;
                }
            }
            let physical = self.allocate(request);
            if let Some(physical) = physical {
                result.push((
                    request.page,
                    Residency::Resident {
                        physical,
                        epoch: request.epoch,
                    },
                ));
            } else {
                result.push((
                    request.page,
                    Residency::Nonresident {
                        fallback: fallback_for(request.page.kind),
                        reason: NonresidentReason::Capacity,
                    },
                ));
            }
        }
        Ok(result)
    }

    fn allocate(&mut self, request: PageRequest) -> Option<u32> {
        let occupied: BTreeMap<u32, ()> =
            self.residents.values().map(|r| (r.physical, ())).collect();
        if let Some(physical) =
            (0..self.config.capacity).find(|index| !occupied.contains_key(index))
        {
            self.residents.insert(
                request.page,
                ResidentPage {
                    physical,
                    epoch: request.epoch,
                    priority: request.priority,
                    last_used_frame: request.frame,
                },
            );
            return Some(physical);
        }
        let victim = self
            .residents
            .iter()
            .min_by(|(ka, a), (kb, b)| {
                a.priority
                    .cmp(&b.priority)
                    .then_with(|| a.last_used_frame.cmp(&b.last_used_frame))
                    .then_with(|| ka.cmp(kb))
            })
            .map(|(key, resident)| (*key, *resident));
        let (key, old) = victim?;
        if old.priority > request.priority {
            return None;
        }
        self.residents.remove(&key);
        self.residents.insert(
            request.page,
            ResidentPage {
                physical: old.physical,
                epoch: request.epoch,
                priority: request.priority,
                last_used_frame: request.frame,
            },
        );
        Some(old.physical)
    }

    /// Returns resident state or a well-defined fallback for every requested page.
    pub fn lookup(&self, page: PageId, epoch: RepEpoch) -> Residency {
        if self.device_lost {
            return Residency::Nonresident {
                fallback: fallback_for(page.kind),
                reason: NonresidentReason::DeviceLost,
            };
        }
        match self.residents.get(&page) {
            Some(r) if r.epoch == epoch => Residency::Resident {
                physical: r.physical,
                epoch,
            },
            Some(_) => Residency::Nonresident {
                fallback: fallback_for(page.kind),
                reason: NonresidentReason::StaleEpoch,
            },
            None => Residency::Nonresident {
                fallback: fallback_for(page.kind),
                reason: NonresidentReason::NotRequested,
            },
        }
    }

    pub fn device_lost(&mut self) {
        self.device_lost = true;
        self.residents.clear();
    }

    pub fn snapshot(&self) -> StreamSnapshot {
        let pages = self
            .residents
            .iter()
            .map(|(page, r)| SnapshotPage {
                page: *page,
                physical: r.physical,
                epoch: r.epoch,
                priority: r.priority,
                last_used_frame: r.last_used_frame,
            })
            .collect();
        StreamSnapshot {
            version: SNAPSHOT_VERSION,
            capacity: self.config.capacity,
            device_epoch: self.device_epoch,
            pages,
        }
    }

    /// Rebuilds mappings on a recovered device. Stale mappings are discarded;
    /// snapshots exceeding the configured pool are rejected atomically.
    pub fn restore(
        &mut self,
        snapshot: &StreamSnapshot,
        new_device_epoch: u64,
    ) -> Result<usize, StreamError> {
        if snapshot.version != SNAPSHOT_VERSION {
            return Err(StreamError::SnapshotVersion);
        }
        if snapshot.capacity != self.config.capacity
            || snapshot.pages.len() > self.config.capacity as usize
        {
            return Err(StreamError::SnapshotCapacity);
        }
        let mut pages = snapshot.pages.clone();
        pages.sort_by_key(|p| p.page);
        let mut seen = BTreeMap::new();
        let mut physical_seen = BTreeMap::new();
        for p in &pages {
            if seen.insert(p.page, ()).is_some() {
                return Err(StreamError::DuplicateSnapshotPage);
            }
            if p.physical >= self.config.capacity {
                return Err(StreamError::InvalidPhysicalPage);
            }
            if physical_seen.insert(p.physical, ()).is_some() {
                return Err(StreamError::InvalidPhysicalPage);
            }
        }
        let mut restored = BTreeMap::new();
        let mut dropped = 0;
        for p in pages {
            let valid = self
                .bindings
                .values()
                .any(|b| b.rep == p.page.rep && b.epoch == p.epoch && page_is_bound(*b, p.page));
            if valid {
                restored.insert(
                    p.page,
                    ResidentPage {
                        physical: p.physical,
                        epoch: p.epoch,
                        priority: p.priority,
                        last_used_frame: p.last_used_frame,
                    },
                );
            } else {
                dropped += 1;
            }
        }
        self.residents = restored;
        self.device_epoch = new_device_epoch;
        self.device_lost = false;
        Ok(dropped)
    }
}

fn page_is_bound(binding: CellBinding, page: PageId) -> bool {
    let range = match page.kind {
        PageKind::Shadow => binding.shadow,
        PageKind::Gi => binding.gi,
        PageKind::Cache => binding.cache,
    };
    range.is_some_and(|r| page.page >= r.first && r.end().is_some_and(|end| page.page < end))
}

const fn fallback_for(kind: PageKind) -> Fallback {
    match kind {
        PageKind::Shadow => Fallback::ShadowCascade,
        PageKind::Gi => Fallback::GiProbe,
        PageKind::Cache => Fallback::CacheRecompute,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(cell: u64, rep: u64, epoch: u64) -> CellBinding {
        CellBinding {
            cell: CellId(cell),
            rep: RepId(rep),
            provider: RepProvider::new(ProviderKind::Asset, rep, 1),
            epoch: RepEpoch(epoch),
            shadow: Some(PageRange::new(0, 4)),
            gi: Some(PageRange::new(8, 4)),
            cache: Some(PageRange::new(16, 4)),
        }
    }

    fn req(
        cell: u64,
        kind: PageKind,
        rep: u64,
        page: u32,
        priority: u32,
        frame: u64,
    ) -> PageRequest {
        PageRequest {
            cell: CellId(cell),
            page: PageId {
                kind,
                rep: RepId(rep),
                page,
            },
            epoch: RepEpoch(1),
            priority: RequestPriority(priority),
            frame,
        }
    }

    #[test]
    fn rejects_invalid_capacity_and_ranges() {
        assert_eq!(
            VirtualPagePool::new(PagePoolConfig { capacity: 0 }).err(),
            Some(StreamError::InvalidCapacity)
        );
        let mut pool = VirtualPagePool::new(PagePoolConfig { capacity: 1 }).unwrap();
        let mut b = binding(1, 1, 1);
        b.gi = Some(PageRange::new(u32::MAX, 2));
        assert_eq!(pool.bind(b), Err(StreamError::InvalidRange));
    }

    #[test]
    fn binds_three_page_families_and_rejects_duplicate() {
        let mut pool = VirtualPagePool::new(PagePoolConfig { capacity: 2 }).unwrap();
        pool.bind(binding(1, 7, 1)).unwrap();
        assert_eq!(
            pool.bind(binding(1, 7, 1)),
            Err(StreamError::DuplicateBinding)
        );
        for (kind, page) in [
            (PageKind::Shadow, 1),
            (PageKind::Gi, 9),
            (PageKind::Cache, 17),
        ] {
            assert!(pool.request(&[req(1, kind, 7, page, 1, 0)]).is_ok());
        }
    }

    #[test]
    fn duplicate_requests_merge_max_priority_and_allocate_deterministically() {
        let mut pool = VirtualPagePool::new(PagePoolConfig { capacity: 2 }).unwrap();
        pool.bind(binding(1, 1, 1)).unwrap();
        let out = pool
            .request(&[
                req(1, PageKind::Shadow, 1, 2, 2, 1),
                req(1, PageKind::Shadow, 1, 2, 9, 3),
                req(1, PageKind::Shadow, 1, 1, 5, 2),
            ])
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0.page, 2);
        assert_eq!(
            out[0].1,
            Residency::Resident {
                physical: 0,
                epoch: RepEpoch(1)
            }
        );
    }

    #[test]
    fn capacity_uses_priority_then_age_then_address_and_fallback() {
        let mut pool = VirtualPagePool::new(PagePoolConfig { capacity: 1 }).unwrap();
        pool.bind(binding(1, 1, 1)).unwrap();
        pool.request(&[req(1, PageKind::Shadow, 1, 0, 2, 10)])
            .unwrap();
        let low = pool
            .request(&[req(1, PageKind::Shadow, 1, 1, 1, 11)])
            .unwrap();
        assert_eq!(
            low[0].1,
            Residency::Nonresident {
                fallback: Fallback::ShadowCascade,
                reason: NonresidentReason::Capacity
            }
        );
        let high = pool.request(&[req(1, PageKind::Gi, 1, 8, 3, 12)]).unwrap();
        assert_eq!(
            high[0].1,
            Residency::Resident {
                physical: 0,
                epoch: RepEpoch(1)
            }
        );
    }

    #[test]
    fn rejects_unbound_pages_and_stale_requests() {
        let mut pool = VirtualPagePool::new(PagePoolConfig { capacity: 1 }).unwrap();
        pool.bind(binding(1, 1, 1)).unwrap();
        assert_eq!(
            pool.request(&[req(1, PageKind::Cache, 1, 0, 1, 0)]),
            Err(StreamError::PageNotBound)
        );
        let mut stale = req(1, PageKind::Shadow, 1, 0, 1, 0);
        stale.epoch = RepEpoch(2);
        assert_eq!(pool.request(&[stale]), Err(StreamError::StaleRequest));
    }

    #[test]
    fn invalidation_removes_stale_residency() {
        let mut pool = VirtualPagePool::new(PagePoolConfig { capacity: 2 }).unwrap();
        pool.bind(binding(1, 1, 1)).unwrap();
        pool.request(&[req(1, PageKind::Gi, 1, 8, 1, 0)]).unwrap();
        assert_eq!(pool.invalidate(RepId(1), RepEpoch(2)), 1);
        assert_eq!(
            pool.lookup(
                PageId {
                    kind: PageKind::Gi,
                    rep: RepId(1),
                    page: 8
                },
                RepEpoch(2)
            ),
            Residency::Nonresident {
                fallback: Fallback::GiProbe,
                reason: NonresidentReason::NotRequested
            }
        );
    }

    #[test]
    fn snapshot_restore_preserves_valid_pages_and_drops_stale_epochs() {
        let mut source = VirtualPagePool::new(PagePoolConfig { capacity: 2 }).unwrap();
        source.bind(binding(1, 1, 1)).unwrap();
        source
            .request(&[req(1, PageKind::Cache, 1, 16, 4, 8)])
            .unwrap();
        let snapshot = source.snapshot();
        let mut recovered = VirtualPagePool::new(PagePoolConfig { capacity: 2 }).unwrap();
        recovered.bind(binding(1, 1, 1)).unwrap();
        assert_eq!(recovered.restore(&snapshot, 9), Ok(0));
        assert_eq!(
            recovered.lookup(
                PageId {
                    kind: PageKind::Cache,
                    rep: RepId(1),
                    page: 16
                },
                RepEpoch(1)
            ),
            Residency::Resident {
                physical: 0,
                epoch: RepEpoch(1)
            }
        );
        recovered.invalidate(RepId(1), RepEpoch(2));
        assert_eq!(recovered.restore(&snapshot, 10), Ok(1));
    }

    #[test]
    fn restore_rejects_corrupt_or_over_capacity_snapshot_atomically() {
        let mut pool = VirtualPagePool::new(PagePoolConfig { capacity: 1 }).unwrap();
        pool.bind(binding(1, 1, 1)).unwrap();
        let original = pool.snapshot();
        let mut corrupt = original.clone();
        corrupt.version += 1;
        assert_eq!(pool.restore(&corrupt, 1), Err(StreamError::SnapshotVersion));
        let mut full = original;
        for page in [8, 9] {
            full.pages.push(SnapshotPage {
                page: PageId {
                    kind: PageKind::Gi,
                    rep: RepId(1),
                    page,
                },
                physical: 0,
                epoch: RepEpoch(1),
                priority: RequestPriority(0),
                last_used_frame: 0,
            });
        }
        assert_eq!(pool.restore(&full, 1), Err(StreamError::SnapshotCapacity));
    }

    #[test]
    fn device_loss_clears_residency_until_restore() {
        let mut pool = VirtualPagePool::new(PagePoolConfig { capacity: 1 }).unwrap();
        pool.bind(binding(1, 1, 1)).unwrap();
        let page = PageId {
            kind: PageKind::Shadow,
            rep: RepId(1),
            page: 0,
        };
        pool.request(&[req(1, PageKind::Shadow, 1, 0, 1, 0)])
            .unwrap();
        pool.device_lost();
        assert_eq!(
            pool.lookup(page, RepEpoch(1)),
            Residency::Nonresident {
                fallback: Fallback::ShadowCascade,
                reason: NonresidentReason::DeviceLost
            }
        );
        assert_eq!(pool.resident_count(), 0);
    }

    #[test]
    fn merged_order_is_independent_of_input_order() {
        let mut a = VirtualPagePool::new(PagePoolConfig { capacity: 3 }).unwrap();
        a.bind(binding(1, 1, 1)).unwrap();
        let mut b = VirtualPagePool::new(PagePoolConfig { capacity: 3 }).unwrap();
        b.bind(binding(1, 1, 1)).unwrap();
        let requests = [
            req(1, PageKind::Shadow, 1, 0, 3, 1),
            req(1, PageKind::Gi, 1, 8, 3, 1),
            req(1, PageKind::Cache, 1, 16, 3, 1),
        ];
        let lhs = a.request(&requests).unwrap();
        let mut reversed = requests;
        reversed.reverse();
        let rhs = b.request(&reversed).unwrap();
        assert_eq!(lhs, rhs);
    }
}
