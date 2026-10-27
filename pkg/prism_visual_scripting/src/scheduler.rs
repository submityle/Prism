//! Bounded, fair, deadline-aware admission for durable runs.

use crate::{VsError, VsResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet, VecDeque};

pub const MAX_SCHEDULER_ID_BYTES: usize = 128;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerLimits {
    pub max_active: usize,
    pub max_active_per_tenant: usize,
    pub max_queued: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduledRun {
    pub run_id: String,
    pub tenant: String,
    pub priority: u8,
    pub deadline_ms: Option<u64>,
    pub sequence: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum AdmitResult {
    Queued,
    Duplicate,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerSnapshot {
    pub active: BTreeMap<String, String>,
    pub queued: BTreeMap<String, VecDeque<ScheduledRun>>,
    pub tenant_order: VecDeque<String>,
    pub next_sequence: u64,
}

pub struct FairScheduler {
    limits: SchedulerLimits,
    state: SchedulerSnapshot,
    active_tenants: BTreeMap<String, usize>,
    run_ids: HashSet<String>,
}

impl FairScheduler {
    pub fn new(limits: SchedulerLimits) -> VsResult<Self> {
        if limits.max_active == 0 || limits.max_active_per_tenant == 0 || limits.max_queued == 0 {
            return Err(VsError::Validate(
                "scheduler limits must be greater than zero".into(),
            ));
        }
        Ok(Self {
            limits,
            state: SchedulerSnapshot {
                next_sequence: 1,
                ..Default::default()
            },
            active_tenants: BTreeMap::new(),
            run_ids: HashSet::new(),
        })
    }

    pub fn admit(
        &mut self,
        run_id: impl Into<String>,
        tenant: impl Into<String>,
        priority: u8,
        deadline_ms: Option<u64>,
    ) -> VsResult<AdmitResult> {
        let run_id = run_id.into();
        let tenant = tenant.into();
        if run_id.is_empty() || tenant.is_empty() {
            return Err(VsError::Validate(
                "run and tenant identities must not be empty".into(),
            ));
        }
        if run_id.len() > MAX_SCHEDULER_ID_BYTES || tenant.len() > MAX_SCHEDULER_ID_BYTES {
            return Err(VsError::Validate(format!(
                "run and tenant identities must not exceed {MAX_SCHEDULER_ID_BYTES} bytes"
            )));
        }
        if self.run_ids.contains(&run_id) {
            return Ok(AdmitResult::Duplicate);
        }
        if self.queued_len() >= self.limits.max_queued {
            return Err(VsError::Runtime("scheduler queue is full".into()));
        }
        let indexed_run_id = run_id.clone();
        let run = ScheduledRun {
            run_id,
            tenant: tenant.clone(),
            priority,
            deadline_ms,
            sequence: self.state.next_sequence,
        };
        self.state.next_sequence = self
            .state
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| VsError::Runtime("scheduler sequence exhausted".into()))?;
        self.state
            .queued
            .entry(tenant.clone())
            .or_default()
            .push_back(run);
        self.run_ids.insert(indexed_run_id);
        if !self.state.tenant_order.contains(&tenant) {
            self.state.tenant_order.push_back(tenant);
        }
        Ok(AdmitResult::Queued)
    }

    pub fn poll(&mut self, now_ms: u64) -> Vec<ScheduledRun> {
        self.drop_expired(now_ms);
        let mut ready = Vec::new();
        let mut stalled_tenants = 0;
        while self.state.active.len() < self.limits.max_active {
            let Some(tenant) = self.state.tenant_order.pop_front() else {
                break;
            };
            let can_run = self
                .active_tenants
                .get(&tenant)
                .copied()
                .unwrap_or_default()
                < self.limits.max_active_per_tenant;
            let run = if can_run {
                self.state.queued.get_mut(&tenant).and_then(pop_best)
            } else {
                None
            };
            if let Some(run) = run {
                self.state.active.insert(run.run_id.clone(), tenant.clone());
                *self.active_tenants.entry(tenant.clone()).or_default() += 1;
                ready.push(run);
                stalled_tenants = 0;
            } else {
                stalled_tenants += 1;
            }
            if self
                .state
                .queued
                .get(&tenant)
                .is_some_and(|queue| !queue.is_empty())
            {
                self.state.tenant_order.push_back(tenant);
            } else {
                self.state.queued.remove(&tenant);
            }
            if self.state.tenant_order.is_empty() {
                break;
            }
            if stalled_tenants >= self.state.tenant_order.len().max(1) {
                break;
            }
        }
        ready
    }

    pub fn complete(&mut self, run_id: &str) -> VsResult<()> {
        let tenant = self
            .state
            .active
            .remove(run_id)
            .ok_or_else(|| VsError::Runtime(format!("run {run_id} is not active")))?;
        self.run_ids.remove(run_id);
        if let Some(count) = self.active_tenants.get_mut(&tenant) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.active_tenants.remove(&tenant);
            }
        }
        Ok(())
    }

    pub fn queued_len(&self) -> usize {
        self.state.queued.values().map(VecDeque::len).sum()
    }
    pub fn snapshot(&self) -> SchedulerSnapshot {
        self.state.clone()
    }

    fn drop_expired(&mut self, now_ms: u64) {
        let mut expired = Vec::new();
        self.state.queued.retain(|_, queue| {
            queue.retain(|run| {
                let keep = run.deadline_ms.is_none_or(|deadline| deadline > now_ms);
                if !keep {
                    expired.push(run.run_id.clone());
                }
                keep
            });
            !queue.is_empty()
        });
        for run_id in expired {
            self.run_ids.remove(&run_id);
        }
        self.state
            .tenant_order
            .retain(|tenant| self.state.queued.contains_key(tenant));
    }
}

fn pop_best(queue: &mut VecDeque<ScheduledRun>) -> Option<ScheduledRun> {
    let best = queue
        .iter()
        .enumerate()
        .max_by_key(|(_, run)| (run.priority, std::cmp::Reverse(run.sequence)))
        .map(|(index, _)| index)?;
    queue.remove(best)
}
