//! Deterministic what-if execution primitives.

use crate::effect::ExecutionClock;
use crate::{DeterminismGate, RunnerClass};
use crate::{
    ExecutionEvent, ExecutionEventSink, ExecutionReplay, InMemoryEventLog, VirtualClock, VsError,
    VsResult,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationTrace {
    pub seed: u64,
    pub now_ms: u64,
    pub draws: Vec<u64>,
    pub event_digest: [u8; 32],
}

pub struct SimulationHarness {
    pub clock: Arc<VirtualClock>,
    initial_seed: u64,
    seed: u64,
    draws: Vec<u64>,
    pub events: InMemoryEventLog,
}

#[derive(Clone, Debug, Default)]
pub struct SimulationDeterminismGate {
    allowed: BTreeSet<(String, RunnerClass)>,
}

impl SimulationDeterminismGate {
    pub fn allow(&mut self, def_id: impl Into<String>, class: RunnerClass) {
        self.allowed.insert((def_id.into(), class));
    }
}

impl DeterminismGate for SimulationDeterminismGate {
    fn check_runner(&self, def_id: &str, class: RunnerClass) -> VsResult<()> {
        if self.allowed.contains(&(def_id.to_string(), class)) {
            Ok(())
        } else {
            Err(VsError::Runtime(format!(
                "runner {def_id} is not admitted for deterministic execution"
            )))
        }
    }
}

impl SimulationHarness {
    pub fn new(seed: u64, now_ms: u64) -> Self {
        Self {
            clock: Arc::new(VirtualClock::new(now_ms)),
            initial_seed: seed,
            seed,
            draws: Vec::new(),
            events: InMemoryEventLog::default(),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut value = self.seed;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.seed = value;
        self.draws.push(value);
        value
    }

    pub fn advance(&self, millis: u64) {
        self.clock.advance(millis);
    }

    pub fn record(&self, event: ExecutionEvent) -> VsResult<()> {
        self.events.append(event)
    }

    pub fn trace(&self, seed: u64, events: &[ExecutionEvent]) -> VsResult<SimulationTrace> {
        let execution_id = events
            .first()
            .map_or("simulation", |event| event.execution_id.as_str());
        let report = ExecutionReplay::validate(events, execution_id)?;
        Ok(SimulationTrace {
            seed,
            now_ms: self.clock.now_millis(),
            draws: self.draws.clone(),
            event_digest: report.digest,
        })
    }

    pub fn digest(&self) -> VsResult<[u8; 32]> {
        let trace = self.trace(self.initial_seed, &self.events.events())?;
        let bytes = serde_json::to_vec(&trace)
            .map_err(|error| VsError::Runtime(format!("encode simulation trace: {error}")))?;
        Ok(Sha256::digest(bytes).into())
    }
}
