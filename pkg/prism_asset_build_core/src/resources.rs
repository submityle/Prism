//! Admission control for declared processor working sets and scarce devices.
//!
//! Tokens bound concurrent declared demand. They do not sandbox a processor
//! which allocates more than it declares, or account for retained DAG outputs.

use crate::JobId;
use std::fmt;

/// Simultaneously reserved resources. Byte counts represent peak working
/// memory and I/O in flight, rather than a bandwidth promise.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuildResources {
    pub cpu_threads: u64,
    pub memory_bytes: u64,
    pub disk_read_bytes: u64,
    pub disk_write_bytes: u64,
    pub network_bytes: u64,
    pub external_processes: u64,
    pub gpu_slots: u64,
    pub gpu_memory_bytes: u64,
    pub tool_seats: u64,
}

impl Default for BuildResources {
    fn default() -> Self {
        Self {
            cpu_threads: 1,
            memory_bytes: 0,
            disk_read_bytes: 0,
            disk_write_bytes: 0,
            network_bytes: 0,
            external_processes: 0,
            gpu_slots: 0,
            gpu_memory_bytes: 0,
            tool_seats: 0,
        }
    }
}

impl BuildResources {
    /// Compatibility budget for callers which only limit worker count.
    pub const UNLIMITED: Self = Self {
        cpu_threads: u64::MAX,
        memory_bytes: u64::MAX,
        disk_read_bytes: u64::MAX,
        disk_write_bytes: u64::MAX,
        network_bytes: u64::MAX,
        external_processes: u64::MAX,
        gpu_slots: u64::MAX,
        gpu_memory_bytes: u64::MAX,
        tool_seats: u64::MAX,
    };

    fn components(self) -> [u64; 9] {
        [
            self.cpu_threads,
            self.memory_bytes,
            self.disk_read_bytes,
            self.disk_write_bytes,
            self.network_bytes,
            self.external_processes,
            self.gpu_slots,
            self.gpu_memory_bytes,
            self.tool_seats,
        ]
    }

    pub(crate) fn fits(self, budget: Self) -> bool {
        self.components()
            .into_iter()
            .zip(budget.components())
            .all(|(a, b)| a <= b)
    }

    pub(crate) fn reserve(self, remaining: &mut Self) -> bool {
        if !self.fits(*remaining) {
            return false;
        }
        remaining.cpu_threads -= self.cpu_threads;
        remaining.memory_bytes -= self.memory_bytes;
        remaining.disk_read_bytes -= self.disk_read_bytes;
        remaining.disk_write_bytes -= self.disk_write_bytes;
        remaining.network_bytes -= self.network_bytes;
        remaining.external_processes -= self.external_processes;
        remaining.gpu_slots -= self.gpu_slots;
        remaining.gpu_memory_bytes -= self.gpu_memory_bytes;
        remaining.tool_seats -= self.tool_seats;
        true
    }
}

/// A deterministic rejection before any processor or CAS mutation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceError {
    ZeroCpuBudget,
    ZeroCpuRequest(JobId),
    ExceedsBudget {
        job: JobId,
        requested: BuildResources,
        budget: BuildResources,
    },
}

impl fmt::Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCpuBudget => f.write_str("build CPU budget must be positive"),
            Self::ZeroCpuRequest(id) => write!(f, "job {} must reserve a CPU thread", id.0),
            Self::ExceedsBudget {
                job,
                requested,
                budget,
            } => write!(
                f,
                "job {} resources {requested:?} exceed budget {budget:?}",
                job.0
            ),
        }
    }
}
impl std::error::Error for ResourceError {}
