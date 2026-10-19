//! Deterministic shadow-slot planning and virtual-shadow residency.
//!
//! This file is intentionally self-contained.  It describes the P3 shadow
//! policy without depending on a crate module declaration, so the contract
//! can be reviewed and tested before it is wired into the render graph.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

/// Stable identity of a shadow-producing light or view slot.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ShadowSlotId(pub u32);

/// Virtual address of one shadow page.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ShadowPageKey {
    pub slot: ShadowSlotId,
    pub mip: u8,
    pub x: u16,
    pub y: u16,
}

/// Shadow representation selected for a slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadowMethod {
    RayTrace,
    VirtualShadowMap,
    CascadeMap,
}

/// A fallback that remains valid when a virtual page is unavailable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShadowFallback {
    /// A stable raster cascade visibility value.
    Cascade { visibility: f32 },
    /// A receiver-space approximation supplied by the caller.
    ScreenSpace { visibility: f32 },
    /// Fully lit is the last-resort conservative output for this contract.
    Unshadowed,
}

impl ShadowFallback {
    fn visibility(self) -> f32 {
        match self {
            Self::Cascade { visibility } | Self::ScreenSpace { visibility } => {
                clamp_visibility(visibility)
            }
            Self::Unshadowed => 1.0,
        }
    }
}

/// Fixed-point cost used by the planner.  Time is expressed in microseconds
/// to keep budget decisions independent of floating-point ordering.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ShadowCost {
    pub gpu_micros: u64,
    pub bandwidth_bytes: u64,
    pub resident_bytes: u64,
    pub io_ops: u32,
}

impl ShadowCost {
    fn fits_within(self, budget: ShadowCost) -> bool {
        self.gpu_micros <= budget.gpu_micros
            && self.bandwidth_bytes <= budget.bandwidth_bytes
            && self.resident_bytes <= budget.resident_bytes
            && self.io_ops <= budget.io_ops
    }

    fn saturating_add(self, rhs: Self) -> Self {
        Self {
            gpu_micros: self.gpu_micros.saturating_add(rhs.gpu_micros),
            bandwidth_bytes: self.bandwidth_bytes.saturating_add(rhs.bandwidth_bytes),
            resident_bytes: self.resident_bytes.saturating_add(rhs.resident_bytes),
            io_ops: self.io_ops.saturating_add(rhs.io_ops),
        }
    }

    fn saturating_sub(self, rhs: Self) -> Self {
        Self {
            gpu_micros: self.gpu_micros.saturating_sub(rhs.gpu_micros),
            bandwidth_bytes: self.bandwidth_bytes.saturating_sub(rhs.bandwidth_bytes),
            resident_bytes: self.resident_bytes.saturating_sub(rhs.resident_bytes),
            io_ops: self.io_ops.saturating_sub(rhs.io_ops),
        }
    }
}

/// Hard per-frame limits for shadow planning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadowBudget {
    pub work: ShadowCost,
    pub max_new_pages: u32,
}

/// Cost and capacity of a physical shadow-page pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadowPoolConfig {
    pub physical_pages: u32,
    pub page_bytes: u64,
    pub minimum_resident_frames: u64,
}

/// Slot declaration consumed by the planner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowSlotInput {
    pub id: ShadowSlotId,
    pub method: ShadowMethod,
    pub caster_epoch: u64,
    pub priority: u32,
    pub cost: ShadowCost,
    pub fallback: ShadowFallback,
    /// Number of valid cascade levels for [`ShadowMethod::CascadeMap`].
    pub cascade_count: u8,
}

/// One receiver's request for a virtual shadow page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadowPageRequest {
    pub key: ShadowPageKey,
    pub caster_epoch: u64,
    pub priority: u32,
    pub frame: u64,
    pub cost: ShadowCost,
}

/// A merged request after duplicate keys have been reduced deterministically.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MergedPageRequest {
    pub key: ShadowPageKey,
    pub caster_epoch: u64,
    pub priority: u32,
    pub last_requested_frame: u64,
    pub cost: ShadowCost,
}

/// Result of one page request in the current frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageDecision {
    Hit {
        key: ShadowPageKey,
        physical_page: u32,
    },
    Allocated {
        key: ShadowPageKey,
        physical_page: u32,
    },
    EvictedAndAllocated {
        key: ShadowPageKey,
        physical_page: u32,
        evicted: ShadowPageKey,
    },
    Clipped {
        key: ShadowPageKey,
        reason: ClipReason,
    },
}

/// Why a request did not receive a physical page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClipReason {
    Budget,
    NewPageLimit,
    NoEvictablePage,
    UnknownSlot,
    MethodDoesNotUsePages,
    StaleEpoch,
}

/// Why an existing mapping was removed before allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InvalidationReason {
    CasterEpoch,
    ExplicitSlotRemoval,
}

/// A mapping removed from the resident page table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Invalidation {
    pub key: ShadowPageKey,
    pub physical_page: u32,
    pub reason: InvalidationReason,
}

/// Residency counters emitted with every plan.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ShadowResidency {
    pub physical_pages: u32,
    pub resident_pages: u32,
    pub free_pages: u32,
    pub resident_bytes: u64,
    pub page_hits: u32,
    pub page_misses: u32,
    pub stale_pages: u32,
    pub evictions: u32,
    pub clipped_pages: u32,
}

/// One slot's resolved method and current epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannedShadowSlot {
    pub id: ShadowSlotId,
    pub method: ShadowMethod,
    pub caster_epoch: u64,
    pub priority: u32,
    pub fallback: ShadowFallback,
    pub cascade_count: u8,
}

/// Complete deterministic output of one shadow planning pass.
#[derive(Clone, Debug, PartialEq)]
pub struct ShadowSlotPlan {
    pub frame: u64,
    pub slots: Vec<PlannedShadowSlot>,
    pub requests: Vec<MergedPageRequest>,
    pub decisions: Vec<PageDecision>,
    pub invalidations: Vec<Invalidation>,
    pub cost: ShadowCost,
    pub residency: ShadowResidency,
    pub budget_clipped: bool,
}

/// Input to one planning pass.
#[derive(Clone, Debug, PartialEq)]
pub struct ShadowFrameInput {
    pub frame: u64,
    pub slots: Vec<ShadowSlotInput>,
    pub requests: Vec<ShadowPageRequest>,
    pub budget: ShadowBudget,
}

/// A visibility query supplied by the method-specific sampling pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VisibilitySample {
    RayTrace { visibility: f32 },
    VirtualShadowMap { key: ShadowPageKey, visibility: f32 },
    CascadeMap { cascade: u8, visibility: f32 },
}

/// Which source produced the returned visibility.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisibilitySource {
    RayTrace,
    VirtualShadowMap,
    CascadeMap,
    Fallback,
}

/// Verifiable visibility result; `fallback` is explicit and never inferred
/// from a magic visibility value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisibilityResult {
    pub visibility: f32,
    pub source: VisibilitySource,
    pub fallback: bool,
}

/// Planner state.  All maps are ordered to make both allocation and output
/// reproducible across platforms and process runs.
#[derive(Clone, Debug)]
pub struct ShadowPlanner {
    pool: ShadowPoolConfig,
    resident: BTreeMap<ShadowPageKey, ResidentPage>,
    physical_owner: BTreeMap<u32, ShadowPageKey>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResidentPage {
    physical_page: u32,
    caster_epoch: u64,
    priority: u32,
    last_used_frame: u64,
    resident_since: u64,
}

impl ShadowPlanner {
    /// Creates an empty deterministic page planner.
    #[must_use]
    pub fn new(pool: ShadowPoolConfig) -> Self {
        Self {
            pool,
            resident: BTreeMap::new(),
            physical_owner: BTreeMap::new(),
        }
    }

    /// Plans one frame, retaining resident pages between calls.
    pub fn plan(&mut self, input: ShadowFrameInput) -> Result<ShadowSlotPlan, ShadowPlanError> {
        self.validate_slots(&input.slots)?;
        let slots = sorted_slots(&input.slots);
        let slot_map: BTreeMap<ShadowSlotId, ShadowSlotInput> =
            slots.iter().copied().map(|slot| (slot.id, slot)).collect();
        let mut invalidations = self.remove_invalid_pages(&slot_map);

        let (requests, mut decisions) = merge_requests(&input.requests, &slot_map);
        let mut cost = slots.iter().fold(ShadowCost::default(), |total, slot| {
            total.saturating_add(slot.cost)
        });
        let mut page_hits = 0_u32;
        let mut page_misses = 0_u32;
        let mut stale_pages = invalidations
            .iter()
            .filter(|event| event.reason == InvalidationReason::CasterEpoch)
            .count() as u32;
        let mut evictions = 0_u32;
        let mut clipped_pages = 0_u32;
        let mut new_pages = 0_u32;
        let mut budget_clipped = !cost.fits_within(input.budget.work);

        for (index, request) in requests.iter().enumerate() {
            let slot = match slot_map.get(&request.key.slot) {
                Some(slot) => slot,
                None => {
                    decisions[index] = PageDecision::Clipped {
                        key: request.key,
                        reason: ClipReason::UnknownSlot,
                    };
                    clipped_pages = clipped_pages.saturating_add(1);
                    continue;
                }
            };
            if slot.method != ShadowMethod::VirtualShadowMap {
                decisions[index] = PageDecision::Clipped {
                    key: request.key,
                    reason: ClipReason::MethodDoesNotUsePages,
                };
                clipped_pages = clipped_pages.saturating_add(1);
                continue;
            }
            if request.caster_epoch != slot.caster_epoch {
                decisions[index] = PageDecision::Clipped {
                    key: request.key,
                    reason: ClipReason::StaleEpoch,
                };
                clipped_pages = clipped_pages.saturating_add(1);
                continue;
            }
            if let Some(existing) = self.resident.get_mut(&request.key) {
                if existing.caster_epoch == request.caster_epoch {
                    existing.last_used_frame = input.frame;
                    existing.priority = request.priority;
                    decisions[index] = PageDecision::Hit {
                        key: request.key,
                        physical_page: existing.physical_page,
                    };
                    page_hits = page_hits.saturating_add(1);
                    continue;
                }
                let old = *existing;
                self.remove_mapping(request.key, old.physical_page);
                invalidations.push(Invalidation {
                    key: request.key,
                    physical_page: old.physical_page,
                    reason: InvalidationReason::CasterEpoch,
                });
                stale_pages = stale_pages.saturating_add(1);
            }
            page_misses = page_misses.saturating_add(1);
            if new_pages >= input.budget.max_new_pages {
                decisions[index] = PageDecision::Clipped {
                    key: request.key,
                    reason: ClipReason::NewPageLimit,
                };
                clipped_pages = clipped_pages.saturating_add(1);
                budget_clipped = true;
                continue;
            }
            let next_cost = cost.saturating_add(request.cost);
            if !next_cost.fits_within(input.budget.work) {
                decisions[index] = PageDecision::Clipped {
                    key: request.key,
                    reason: ClipReason::Budget,
                };
                clipped_pages = clipped_pages.saturating_add(1);
                budget_clipped = true;
                continue;
            }
            let allocation = self.allocate_page(request, input.frame, &requests, index);
            match allocation {
                Allocation::Free(physical_page) => {
                    decisions[index] = PageDecision::Allocated {
                        key: request.key,
                        physical_page,
                    };
                    new_pages = new_pages.saturating_add(1);
                    cost = next_cost;
                }
                Allocation::Evicted {
                    physical_page,
                    evicted,
                } => {
                    decisions[index] = PageDecision::EvictedAndAllocated {
                        key: request.key,
                        physical_page,
                        evicted,
                    };
                    new_pages = new_pages.saturating_add(1);
                    evictions = evictions.saturating_add(1);
                    cost = next_cost;
                }
                Allocation::Unavailable => {
                    decisions[index] = PageDecision::Clipped {
                        key: request.key,
                        reason: ClipReason::NoEvictablePage,
                    };
                    clipped_pages = clipped_pages.saturating_add(1);
                }
            }
        }

        invalidations.sort_by_key(|event| (event.physical_page, event.key));
        let resident_pages = self.resident.len() as u32;
        let physical_pages = self.pool.physical_pages;
        let residency = ShadowResidency {
            physical_pages,
            resident_pages,
            free_pages: physical_pages.saturating_sub(resident_pages),
            resident_bytes: (resident_pages as u64).saturating_mul(self.pool.page_bytes),
            page_hits,
            page_misses,
            stale_pages,
            evictions,
            clipped_pages,
        };
        let resident_cost = ShadowCost {
            resident_bytes: residency.resident_bytes,
            ..ShadowCost::default()
        };
        cost = cost
            .saturating_add(resident_cost)
            .saturating_sub(ShadowCost {
                resident_bytes: slots.iter().map(|slot| slot.cost.resident_bytes).sum(),
                ..ShadowCost::default()
            });
        Ok(ShadowSlotPlan {
            frame: input.frame,
            slots: slots
                .iter()
                .map(|slot| PlannedShadowSlot {
                    id: slot.id,
                    method: slot.method,
                    caster_epoch: slot.caster_epoch,
                    priority: slot.priority,
                    fallback: slot.fallback,
                    cascade_count: slot.cascade_count,
                })
                .collect(),
            requests,
            decisions,
            invalidations,
            cost,
            residency,
            budget_clipped,
        })
    }

    fn validate_slots(&self, slots: &[ShadowSlotInput]) -> Result<(), ShadowPlanError> {
        let mut seen = BTreeSet::new();
        for slot in slots {
            if !seen.insert(slot.id) {
                return Err(ShadowPlanError::DuplicateSlot(slot.id));
            }
            if slot.method == ShadowMethod::CascadeMap && slot.cascade_count == 0 {
                return Err(ShadowPlanError::CascadeWithoutLevels(slot.id));
            }
            if !slot.cost.fits_within(ShadowCost {
                gpu_micros: u64::MAX,
                bandwidth_bytes: u64::MAX,
                resident_bytes: u64::MAX,
                io_ops: u32::MAX,
            }) {
                return Err(ShadowPlanError::InvalidCost(slot.id));
            }
        }
        Ok(())
    }

    fn remove_invalid_pages(
        &mut self,
        slots: &BTreeMap<ShadowSlotId, ShadowSlotInput>,
    ) -> Vec<Invalidation> {
        let stale: Vec<(ShadowPageKey, ResidentPage, InvalidationReason)> = self
            .resident
            .iter()
            .filter_map(|(key, page)| {
                let reason = match slots.get(&key.slot) {
                    None => InvalidationReason::ExplicitSlotRemoval,
                    Some(slot) if slot.method != ShadowMethod::VirtualShadowMap => {
                        InvalidationReason::ExplicitSlotRemoval
                    }
                    Some(slot) if slot.caster_epoch != page.caster_epoch => {
                        InvalidationReason::CasterEpoch
                    }
                    Some(_) => return None,
                };
                Some((*key, *page, reason))
            })
            .collect();
        let mut invalidations = Vec::with_capacity(stale.len());
        for (key, page, reason) in stale {
            self.remove_mapping(key, page.physical_page);
            invalidations.push(Invalidation {
                key,
                physical_page: page.physical_page,
                reason,
            });
        }
        invalidations
    }

    fn allocate_page(
        &mut self,
        request: &MergedPageRequest,
        frame: u64,
        requests: &[MergedPageRequest],
        request_index: usize,
    ) -> Allocation {
        if let Some(physical_page) =
            (0..self.pool.physical_pages).find(|page| !self.physical_owner.contains_key(page))
        {
            self.insert_mapping(request, physical_page, frame);
            return Allocation::Free(physical_page);
        }
        let requested_keys: BTreeSet<ShadowPageKey> = requests[..request_index]
            .iter()
            .filter(|other| {
                other.key != request.key
                    && self
                        .resident
                        .get(&other.key)
                        .is_some_and(|page| page.caster_epoch == other.caster_epoch)
            })
            .map(|other| other.key)
            .collect();
        let candidate = self
            .resident
            .iter()
            .filter(|(key, page)| {
                !requested_keys.contains(key)
                    && frame.saturating_sub(page.resident_since)
                        >= self.pool.minimum_resident_frames
            })
            .min_by(|(key_a, page_a), (key_b, page_b)| {
                eviction_order(**key_a, **page_a).cmp(&eviction_order(**key_b, **page_b))
            })
            .map(|(key, page)| (*key, *page));
        let Some((evicted, page)) = candidate else {
            return Allocation::Unavailable;
        };
        self.remove_mapping(evicted, page.physical_page);
        self.insert_mapping(request, page.physical_page, frame);
        Allocation::Evicted {
            physical_page: page.physical_page,
            evicted,
        }
    }

    fn insert_mapping(&mut self, request: &MergedPageRequest, physical_page: u32, frame: u64) {
        let page = ResidentPage {
            physical_page,
            caster_epoch: request.caster_epoch,
            priority: request.priority,
            last_used_frame: frame,
            resident_since: frame,
        };
        self.resident.insert(request.key, page);
        self.physical_owner.insert(physical_page, request.key);
    }

    fn remove_mapping(&mut self, key: ShadowPageKey, physical_page: u32) {
        self.resident.remove(&key);
        self.physical_owner.remove(&physical_page);
    }
}

/// Planning failures caused by contradictory slot declarations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadowPlanError {
    DuplicateSlot(ShadowSlotId),
    CascadeWithoutLevels(ShadowSlotId),
    InvalidCost(ShadowSlotId),
}

enum Allocation {
    Free(u32),
    Evicted {
        physical_page: u32,
        evicted: ShadowPageKey,
    },
    Unavailable,
}

fn sorted_slots(slots: &[ShadowSlotInput]) -> Vec<ShadowSlotInput> {
    let mut result = slots.to_vec();
    result.sort_by_key(|slot| slot.id);
    result
}

fn merge_requests(
    requests: &[ShadowPageRequest],
    slots: &BTreeMap<ShadowSlotId, ShadowSlotInput>,
) -> (Vec<MergedPageRequest>, Vec<PageDecision>) {
    let mut merged: BTreeMap<ShadowPageKey, MergedPageRequest> = BTreeMap::new();
    for request in requests {
        let entry = merged.entry(request.key).or_insert(MergedPageRequest {
            key: request.key,
            caster_epoch: request.caster_epoch,
            priority: request.priority,
            last_requested_frame: request.frame,
            cost: request.cost,
        });
        if request.priority > entry.priority
            || (request.priority == entry.priority && request.caster_epoch > entry.caster_epoch)
            || (request.priority == entry.priority
                && request.caster_epoch == entry.caster_epoch
                && request.frame > entry.last_requested_frame)
        {
            entry.caster_epoch = request.caster_epoch;
            entry.priority = request.priority;
            entry.last_requested_frame = request.frame;
            entry.cost = request.cost;
        }
    }
    let mut result: Vec<MergedPageRequest> = merged.into_values().collect();
    result.sort_by(|left, right| request_order(left, slots).cmp(&request_order(right, slots)));
    let decisions = result
        .iter()
        .map(|request| {
            if !slots.contains_key(&request.key.slot) {
                PageDecision::Clipped {
                    key: request.key,
                    reason: ClipReason::UnknownSlot,
                }
            } else {
                PageDecision::Clipped {
                    key: request.key,
                    reason: ClipReason::NoEvictablePage,
                }
            }
        })
        .collect();
    (result, decisions)
}

fn request_order(
    request: &MergedPageRequest,
    slots: &BTreeMap<ShadowSlotId, ShadowSlotInput>,
) -> (
    core::cmp::Reverse<u32>,
    core::cmp::Reverse<u64>,
    ShadowPageKey,
) {
    let slot_priority = slots.get(&request.key.slot).map_or(0, |slot| slot.priority);
    (
        core::cmp::Reverse(request.priority.max(slot_priority)),
        core::cmp::Reverse(request.last_requested_frame),
        request.key,
    )
}

fn eviction_order(key: ShadowPageKey, page: ResidentPage) -> (u32, u64, u64, ShadowPageKey) {
    (
        page.priority,
        page.last_used_frame,
        page.resident_since,
        key,
    )
}

/// Selects and validates a visibility source against the resolved plan.
#[must_use]
pub fn select_visibility(
    plan: &ShadowSlotPlan,
    slot_id: ShadowSlotId,
    sample: VisibilitySample,
) -> VisibilityResult {
    let Some(slot) = plan.slots.iter().find(|slot| slot.id == slot_id) else {
        return fallback_result(ShadowFallback::Unshadowed);
    };
    match (slot.method, sample) {
        (ShadowMethod::RayTrace, VisibilitySample::RayTrace { visibility }) => VisibilityResult {
            visibility: clamp_visibility(visibility),
            source: VisibilitySource::RayTrace,
            fallback: false,
        },
        (
            ShadowMethod::VirtualShadowMap,
            VisibilitySample::VirtualShadowMap { key, visibility },
        ) if key.slot == slot_id && page_is_resident(plan, key) => VisibilityResult {
            visibility: clamp_visibility(visibility),
            source: VisibilitySource::VirtualShadowMap,
            fallback: false,
        },
        (
            ShadowMethod::CascadeMap,
            VisibilitySample::CascadeMap {
                cascade,
                visibility,
            },
        ) if cascade < slot.cascade_count => VisibilityResult {
            visibility: clamp_visibility(visibility),
            source: VisibilitySource::CascadeMap,
            fallback: false,
        },
        _ => fallback_result(slot.fallback),
    }
}

fn page_is_resident(plan: &ShadowSlotPlan, key: ShadowPageKey) -> bool {
    let mut resident = false;
    for decision in &plan.decisions {
        match decision {
            PageDecision::Hit {
                key: resident_key, ..
            }
            | PageDecision::Allocated {
                key: resident_key, ..
            } if *resident_key == key => resident = true,
            PageDecision::EvictedAndAllocated {
                evicted,
                key: new_key,
                ..
            } => {
                if *evicted == key {
                    resident = false;
                }
                if *new_key == key {
                    resident = true;
                }
            }
            _ => {}
        }
    }
    resident
}

fn fallback_result(fallback: ShadowFallback) -> VisibilityResult {
    VisibilityResult {
        visibility: fallback.visibility(),
        source: VisibilitySource::Fallback,
        fallback: true,
    }
}

fn clamp_visibility(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE_COST: ShadowCost = ShadowCost {
        gpu_micros: 10,
        bandwidth_bytes: 100,
        resident_bytes: 0,
        io_ops: 1,
    };

    fn slot(id: u32, method: ShadowMethod, epoch: u64, priority: u32) -> ShadowSlotInput {
        ShadowSlotInput {
            id: ShadowSlotId(id),
            method,
            caster_epoch: epoch,
            priority,
            cost: ShadowCost {
                resident_bytes: 0,
                ..ShadowCost::default()
            },
            fallback: ShadowFallback::Cascade { visibility: 0.75 },
            cascade_count: 4,
        }
    }

    fn request(slot: u32, x: u16, priority: u32, frame: u64) -> ShadowPageRequest {
        ShadowPageRequest {
            key: ShadowPageKey {
                slot: ShadowSlotId(slot),
                mip: 0,
                x,
                y: 0,
            },
            caster_epoch: 1,
            priority,
            frame,
            cost: PAGE_COST,
        }
    }

    fn budget(max_new_pages: u32) -> ShadowBudget {
        ShadowBudget {
            work: ShadowCost {
                gpu_micros: 100,
                bandwidth_bytes: 1_000,
                resident_bytes: 256,
                io_ops: 10,
            },
            max_new_pages,
        }
    }

    #[test]
    fn identical_inputs_have_stable_allocation_and_order() {
        let input = ShadowFrameInput {
            frame: 1,
            slots: vec![
                slot(2, ShadowMethod::VirtualShadowMap, 1, 10),
                slot(1, ShadowMethod::VirtualShadowMap, 1, 10),
            ],
            requests: vec![request(2, 1, 20, 1), request(1, 0, 20, 1)],
            budget: budget(4),
        };
        let mut left = ShadowPlanner::new(ShadowPoolConfig {
            physical_pages: 2,
            page_bytes: 64,
            minimum_resident_frames: 0,
        });
        let mut right = ShadowPlanner::new(ShadowPoolConfig {
            physical_pages: 2,
            page_bytes: 64,
            minimum_resident_frames: 0,
        });
        assert_eq!(
            left.plan(input.clone()).unwrap(),
            right.plan(input).unwrap()
        );
    }

    #[test]
    fn low_priority_page_is_evicted_for_higher_priority_request() {
        let mut planner = ShadowPlanner::new(ShadowPoolConfig {
            physical_pages: 1,
            page_bytes: 64,
            minimum_resident_frames: 0,
        });
        let first = planner
            .plan(ShadowFrameInput {
                frame: 1,
                slots: vec![slot(1, ShadowMethod::VirtualShadowMap, 1, 1)],
                requests: vec![request(1, 0, 1, 1)],
                budget: budget(1),
            })
            .unwrap();
        assert!(matches!(
            first.decisions[0],
            PageDecision::Allocated {
                physical_page: 0,
                ..
            }
        ));
        let second = planner
            .plan(ShadowFrameInput {
                frame: 2,
                slots: vec![slot(1, ShadowMethod::VirtualShadowMap, 1, 1)],
                requests: vec![request(1, 1, 100, 2)],
                budget: budget(1),
            })
            .unwrap();
        assert!(matches!(
            second.decisions[0],
            PageDecision::EvictedAndAllocated {
                physical_page: 0,
                evicted: ShadowPageKey { x: 0, .. },
                ..
            }
        ));
        assert_eq!(second.residency.evictions, 1);
    }

    #[test]
    fn caster_epoch_invalidates_old_pages_before_reallocation() {
        let mut planner = ShadowPlanner::new(ShadowPoolConfig {
            physical_pages: 1,
            page_bytes: 64,
            minimum_resident_frames: 0,
        });
        let first = planner
            .plan(ShadowFrameInput {
                frame: 1,
                slots: vec![slot(1, ShadowMethod::VirtualShadowMap, 1, 1)],
                requests: vec![request(1, 0, 1, 1)],
                budget: budget(1),
            })
            .unwrap();
        assert!(first.invalidations.is_empty());
        let mut changed = request(1, 0, 1, 2);
        changed.caster_epoch = 2;
        let second = planner
            .plan(ShadowFrameInput {
                frame: 2,
                slots: vec![slot(1, ShadowMethod::VirtualShadowMap, 2, 1)],
                requests: vec![changed],
                budget: budget(1),
            })
            .unwrap();
        assert_eq!(second.residency.stale_pages, 1);
        assert_eq!(
            second.invalidations[0].reason,
            InvalidationReason::CasterEpoch
        );
        assert!(matches!(
            second.decisions[0],
            PageDecision::Allocated { .. }
        ));
    }

    #[test]
    fn missing_vsm_page_uses_explicit_fallback() {
        let plan = ShadowPlanner::new(ShadowPoolConfig {
            physical_pages: 0,
            page_bytes: 64,
            minimum_resident_frames: 0,
        })
        .plan(ShadowFrameInput {
            frame: 1,
            slots: vec![slot(1, ShadowMethod::VirtualShadowMap, 1, 1)],
            requests: vec![request(1, 3, 1, 1)],
            budget: budget(1),
        })
        .unwrap();
        let result = select_visibility(
            &plan,
            ShadowSlotId(1),
            VisibilitySample::VirtualShadowMap {
                key: ShadowPageKey {
                    slot: ShadowSlotId(1),
                    mip: 0,
                    x: 3,
                    y: 0,
                },
                visibility: 0.1,
            },
        );
        assert_eq!(result.source, VisibilitySource::Fallback);
        assert!(result.fallback);
        assert_eq!(result.visibility, 0.75);
    }

    #[test]
    fn visibility_rejects_a_page_evicted_later_in_the_same_plan() {
        let mut planner = ShadowPlanner::new(ShadowPoolConfig {
            physical_pages: 1,
            page_bytes: 64,
            minimum_resident_frames: 0,
        });
        planner
            .plan(ShadowFrameInput {
                frame: 1,
                slots: vec![slot(1, ShadowMethod::VirtualShadowMap, 1, 1)],
                requests: vec![request(1, 0, 1, 1)],
                budget: budget(1),
            })
            .unwrap();
        let plan = planner
            .plan(ShadowFrameInput {
                frame: 2,
                slots: vec![slot(1, ShadowMethod::VirtualShadowMap, 1, 1)],
                requests: vec![request(1, 0, 1, 2), request(1, 1, 100, 2)],
                budget: budget(2),
            })
            .unwrap();
        let result = select_visibility(
            &plan,
            ShadowSlotId(1),
            VisibilitySample::VirtualShadowMap {
                key: ShadowPageKey {
                    slot: ShadowSlotId(1),
                    mip: 0,
                    x: 0,
                    y: 0,
                },
                visibility: 0.1,
            },
        );
        assert_eq!(result.source, VisibilitySource::Fallback);
        assert!(result.fallback);
    }

    #[test]
    fn method_selection_is_verifiable_for_ray_and_cascade() {
        let plan = ShadowPlanner::new(ShadowPoolConfig {
            physical_pages: 1,
            page_bytes: 64,
            minimum_resident_frames: 0,
        })
        .plan(ShadowFrameInput {
            frame: 1,
            slots: vec![
                slot(1, ShadowMethod::RayTrace, 1, 1),
                slot(2, ShadowMethod::CascadeMap, 1, 1),
            ],
            requests: Vec::new(),
            budget: budget(0),
        })
        .unwrap();
        let ray = select_visibility(
            &plan,
            ShadowSlotId(1),
            VisibilitySample::RayTrace { visibility: 0.2 },
        );
        assert_eq!(ray.source, VisibilitySource::RayTrace);
        let cascade = select_visibility(
            &plan,
            ShadowSlotId(2),
            VisibilitySample::CascadeMap {
                cascade: 2,
                visibility: 0.4,
            },
        );
        assert_eq!(cascade.source, VisibilitySource::CascadeMap);
        let invalid = select_visibility(
            &plan,
            ShadowSlotId(2),
            VisibilitySample::CascadeMap {
                cascade: 4,
                visibility: 0.4,
            },
        );
        assert_eq!(invalid.source, VisibilitySource::Fallback);
    }

    #[test]
    fn duplicate_requests_merge_to_one_deterministic_request() {
        let mut planner = ShadowPlanner::new(ShadowPoolConfig {
            physical_pages: 2,
            page_bytes: 64,
            minimum_resident_frames: 0,
        });
        let mut lower = request(1, 0, 2, 1);
        lower.cost = ShadowCost {
            gpu_micros: 3,
            ..PAGE_COST
        };
        let mut higher = request(1, 0, 9, 2);
        higher.cost = ShadowCost {
            gpu_micros: 7,
            ..PAGE_COST
        };
        let plan = planner
            .plan(ShadowFrameInput {
                frame: 2,
                slots: vec![slot(1, ShadowMethod::VirtualShadowMap, 1, 1)],
                requests: vec![higher, lower],
                budget: budget(1),
            })
            .unwrap();
        assert_eq!(plan.requests.len(), 1);
        assert_eq!(plan.requests[0].priority, 9);
        assert_eq!(plan.requests[0].cost.gpu_micros, 7);
    }

    #[test]
    fn budget_clips_later_request_after_cost_is_consumed() {
        let mut planner = ShadowPlanner::new(ShadowPoolConfig {
            physical_pages: 2,
            page_bytes: 64,
            minimum_resident_frames: 0,
        });
        let plan = planner
            .plan(ShadowFrameInput {
                frame: 1,
                slots: vec![slot(1, ShadowMethod::VirtualShadowMap, 1, 1)],
                requests: vec![request(1, 0, 10, 1), request(1, 1, 9, 1)],
                budget: ShadowBudget {
                    work: ShadowCost {
                        gpu_micros: 15,
                        bandwidth_bytes: 150,
                        resident_bytes: 256,
                        io_ops: 2,
                    },
                    max_new_pages: 2,
                },
            })
            .unwrap();
        assert!(matches!(plan.decisions[0], PageDecision::Allocated { .. }));
        assert_eq!(
            plan.decisions[1],
            PageDecision::Clipped {
                key: ShadowPageKey {
                    slot: ShadowSlotId(1),
                    mip: 0,
                    x: 1,
                    y: 0
                },
                reason: ClipReason::Budget,
            }
        );
        assert!(plan.budget_clipped);
    }

    #[test]
    fn minimum_residency_prevents_same_frame_thrash() {
        let mut planner = ShadowPlanner::new(ShadowPoolConfig {
            physical_pages: 1,
            page_bytes: 64,
            minimum_resident_frames: 3,
        });
        planner
            .plan(ShadowFrameInput {
                frame: 1,
                slots: vec![slot(1, ShadowMethod::VirtualShadowMap, 1, 1)],
                requests: vec![request(1, 0, 1, 1)],
                budget: budget(1),
            })
            .unwrap();
        let plan = planner
            .plan(ShadowFrameInput {
                frame: 2,
                slots: vec![slot(1, ShadowMethod::VirtualShadowMap, 1, 1)],
                requests: vec![request(1, 1, 100, 2)],
                budget: budget(1),
            })
            .unwrap();
        assert_eq!(
            plan.decisions[0],
            PageDecision::Clipped {
                key: ShadowPageKey {
                    slot: ShadowSlotId(1),
                    mip: 0,
                    x: 1,
                    y: 0
                },
                reason: ClipReason::NoEvictablePage,
            }
        );
    }

    #[test]
    fn invalid_cascade_declaration_is_rejected() {
        let error = ShadowPlanner::new(ShadowPoolConfig {
            physical_pages: 1,
            page_bytes: 64,
            minimum_resident_frames: 0,
        })
        .plan(ShadowFrameInput {
            frame: 1,
            slots: vec![ShadowSlotInput {
                cascade_count: 0,
                ..slot(1, ShadowMethod::CascadeMap, 1, 1)
            }],
            requests: Vec::new(),
            budget: budget(0),
        })
        .unwrap_err();
        assert_eq!(
            error,
            ShadowPlanError::CascadeWithoutLevels(ShadowSlotId(1))
        );
    }
}
