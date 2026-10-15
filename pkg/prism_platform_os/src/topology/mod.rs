//! Real-OS CPU-topology probe filling [`prism_platform::topology::CpuTopology`].
//!
//! [`prism_platform`]'s [`CpuTopology::detect`] is a deliberately conservative
//! fallback: it reports the logical-core *count* but refuses to invent a
//! `P`/`E` split, `SMT` siblings, `NUMA` geometry, or cache domains it has not
//! actually read, so [`CpuTopology::is_probed`] stays `false`. This module is
//! the real probe that earns `is_probed() == true` by reading the host OS.
//!
//! ## Honest boundary
//!
//! Linux uses the validated sysfs parser in `prism_platform` and therefore
//! reports the process-allowed CPU/NUMA snapshot, including sparse node ids,
//! SMT siblings, and optional distance/cache data. Apple Silicon macOS keeps
//! its dedicated sysctl backend. Other targets return
//! [`ProbeError::Unsupported`] rather than guessing.
//!
//! [`CpuTopology`]: prism_platform::topology::CpuTopology
//! [`CpuTopology::detect`]: prism_platform::topology::CpuTopology::detect
//! [`CpuTopology::is_probed`]: prism_platform::topology::CpuTopology::is_probed

use crate::error::ProbeError;
use prism_platform::topology::CpuTopology;

/// One online Linux NUMA node. `cpus` may be empty for a memory-only node;
/// such a node is real and enumerable but cannot be used as a CPU-affinity
/// target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NumaNode {
    /// Stable OS node id. Node ids may be sparse.
    pub id: u16,
    /// Logical CPUs reported by the node's `cpulist` sysfs file.
    pub cpus: Vec<u32>,
}

/// Probe the online NUMA nodes exposed by the host OS.
///
/// On Linux this reads `/sys/devices/system/node/online` and every node's
/// `cpulist`, so memory-only nodes are retained instead of being inferred from
/// the CPU topology. On targets without a verified backend it returns
/// [`ProbeError::Unsupported`]. If Linux is built without a usable NUMA sysfs
/// view, it returns [`ProbeError::Skipped`] rather than an empty successful
/// list.
pub fn probe_numa_nodes() -> Result<Vec<NumaNode>, ProbeError> {
    #[cfg(target_os = "linux")]
    {
        linux::probe_numa_nodes()
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err(ProbeError::Unsupported)
    }
}

/// Why a node-affinity request did not take effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumaAffinityError {
    /// The target has no verified node-affinity backend.
    Unsupported,
    /// Linux has a backend, but required topology data was unavailable or
    /// changed during the operation.
    Skipped {
        /// Structured reason for skipping the operation.
        reason: crate::ProbeSkip,
    },
    /// The requested node is not online or has no CPUs to bind to.
    InvalidNode,
    /// The OS exposed malformed or contradictory topology data.
    InvalidTopology,
    /// The underlying OS rejected the affinity mask.
    SystemError(i32),
}

impl core::fmt::Display for NumaAffinityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unsupported => f.write_str("NUMA CPU affinity is unsupported on this target"),
            Self::Skipped { reason } => write!(f, "NUMA CPU affinity skipped: {reason:?}"),
            Self::InvalidNode => f.write_str("requested NUMA node is not a CPU-affinity target"),
            Self::InvalidTopology => f.write_str("OS exposed invalid NUMA topology data"),
            Self::SystemError(code) => write!(f, "OS rejected NUMA CPU affinity (code {code})"),
        }
    }
}

impl std::error::Error for NumaAffinityError {}

/// Bind the calling thread to the CPUs belonging to an online NUMA node.
///
/// This is a real affinity operation, not a scheduler hint. A memory-only
/// node returns [`NumaAffinityError::InvalidNode`], and platforms without a
/// verified affinity backend return [`NumaAffinityError::Unsupported`].
pub fn bind_current_thread_to_node(node: u16) -> Result<(), NumaAffinityError> {
    #[cfg(target_os = "linux")]
    {
        let nodes = probe_numa_nodes().map_err(|error| match error {
            ProbeError::Unsupported => NumaAffinityError::Unsupported,
            ProbeError::Skipped { reason, .. } => NumaAffinityError::Skipped { reason },
            ProbeError::Os(error) => {
                NumaAffinityError::SystemError(error.raw_os_error().unwrap_or(-1))
            }
            ProbeError::Invalid(_) => NumaAffinityError::InvalidTopology,
        })?;
        let node = nodes
            .iter()
            .find(|candidate| candidate.id == node)
            .ok_or(NumaAffinityError::InvalidNode)?;
        if node.cpus.is_empty() {
            return Err(NumaAffinityError::InvalidNode);
        }
        let allowed = linux::allowed_cpus().map_err(|error| match error {
            ProbeError::Unsupported => NumaAffinityError::Unsupported,
            ProbeError::Skipped { reason, .. } => NumaAffinityError::Skipped { reason },
            ProbeError::Os(error) => {
                NumaAffinityError::SystemError(error.raw_os_error().unwrap_or(-1))
            }
            ProbeError::Invalid(_) => NumaAffinityError::InvalidTopology,
        })?;
        let cpus: Vec<usize> = node
            .cpus
            .iter()
            .copied()
            .filter(|cpu| allowed.binary_search(cpu).is_ok())
            .map(|cpu| cpu as usize)
            .collect();
        if cpus.is_empty() {
            return Err(NumaAffinityError::InvalidNode);
        }
        prism_platform::set_current_thread_affinity_mask(&cpus).map_err(|error| match error {
            prism_platform::AffinityError::Unsupported => NumaAffinityError::Unsupported,
            prism_platform::AffinityError::InvalidCore => NumaAffinityError::InvalidNode,
            prism_platform::AffinityError::SystemError(code) => {
                NumaAffinityError::SystemError(code)
            }
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = node;
        Err(NumaAffinityError::Unsupported)
    }
}

/// Probe the host OS for a genuine [`CpuTopology`] whose
/// [`CpuTopology::is_probed`] is `true`.
///
/// Verified on Apple Silicon macOS; see the module docs for the honest
/// boundary on other targets.
///
/// # Errors
///
/// Returns [`ProbeError::Unsupported`] on targets without a verified backend,
/// [`ProbeError::Os`] if an OS query fails, or [`ProbeError::Invalid`] if the
/// probed values are internally inconsistent.
///
/// [`CpuTopology`]: prism_platform::topology::CpuTopology
/// [`CpuTopology::is_probed`]: prism_platform::topology::CpuTopology::is_probed
pub fn probe_topology() -> Result<CpuTopology, ProbeError> {
    #[cfg(target_os = "linux")]
    {
        linux::probe()
    }
    #[cfg(all(not(target_os = "linux"), target_os = "macos", target_arch = "aarch64"))]
    {
        macos_arm::probe()
    }
    #[cfg(not(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64"))))]
    {
        Err(ProbeError::Unsupported)
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::NumaNode;
    use super::ProbeError;
    use crate::error::ProbeSkip;
    use prism_platform::topology;

    const ONLINE: &str = "/sys/devices/system/node/online";
    const STATUS: &str = "/proc/thread-self/status";
    const MAX_IDS: u32 = 65_536;
    const NUMA_CAPABILITY: &str = "linux NUMA node enumeration";

    fn parse_ids(text: &str) -> Option<Vec<u32>> {
        let mut ids = Vec::new();
        if text.trim().is_empty() {
            return Some(ids);
        }
        for part in text.trim().split(',') {
            let mut bounds = part.split('-');
            let start: u32 = bounds.next()?.parse().ok()?;
            let end = match bounds.next() {
                Some(end) => end.parse().ok()?,
                None => start,
            };
            if bounds.next().is_some()
                || start > end
                || end >= MAX_IDS
                || ids
                    .len()
                    .checked_add((end - start + 1) as usize)
                    .is_none_or(|len| len > MAX_IDS as usize)
            {
                return None;
            }
            ids.extend(start..=end);
        }
        ids.sort_unstable();
        if ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return None;
        }
        Some(ids)
    }

    pub fn allowed_cpus() -> Result<Vec<u32>, ProbeError> {
        let status = std::fs::read_to_string(STATUS).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                ProbeError::Skipped {
                    capability: "linux NUMA CPU affinity",
                    reason: ProbeSkip::MissingOsData,
                }
            } else {
                ProbeError::Os(error)
            }
        })?;
        let list = status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
            .ok_or_else(|| ProbeError::Invalid("missing Cpus_allowed_list".into()))?;
        parse_ids(list)
            .filter(|ids| !ids.is_empty())
            .ok_or_else(|| ProbeError::Invalid("invalid or empty Cpus_allowed_list".into()))
    }

    pub fn probe_numa_nodes() -> Result<Vec<NumaNode>, ProbeError> {
        let online = match std::fs::read_to_string(ONLINE) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(ProbeError::Skipped {
                    capability: NUMA_CAPABILITY,
                    reason: ProbeSkip::MissingOsData,
                });
            }
            Err(error) => return Err(ProbeError::Os(error)),
        };
        let ids = parse_ids(&online)
            .filter(|ids| !ids.is_empty())
            .ok_or_else(|| ProbeError::Invalid("invalid Linux NUMA online node list".into()))?;
        let mut nodes = Vec::with_capacity(ids.len());
        for &id in &ids {
            let path = format!("/sys/devices/system/node/node{id}/cpulist");
            let cpulist = std::fs::read_to_string(&path).map_err(|error| {
                if error.kind() == std::io::ErrorKind::NotFound {
                    ProbeError::Skipped {
                        capability: NUMA_CAPABILITY,
                        reason: ProbeSkip::ChangedDuringProbe,
                    }
                } else {
                    ProbeError::Os(error)
                }
            })?;
            let cpus = parse_ids(&cpulist).ok_or_else(|| {
                ProbeError::Invalid(format!("invalid Linux NUMA cpulist for node {id}"))
            })?;
            nodes.push(NumaNode {
                id: u16::try_from(id)
                    .map_err(|_| ProbeError::Invalid(format!("NUMA node id {id} exceeds u16")))?,
                cpus,
            });
        }

        let online_after = std::fs::read_to_string(ONLINE).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                ProbeError::Skipped {
                    capability: NUMA_CAPABILITY,
                    reason: ProbeSkip::ChangedDuringProbe,
                }
            } else {
                ProbeError::Os(error)
            }
        })?;
        let ids_after = parse_ids(&online_after)
            .filter(|ids| !ids.is_empty())
            .ok_or_else(|| {
                ProbeError::Invalid("invalid Linux NUMA online node list after probe".into())
            })?;
        if ids_after != ids {
            return Err(ProbeError::Skipped {
                capability: NUMA_CAPABILITY,
                reason: ProbeSkip::ChangedDuringProbe,
            });
        }
        for node in &nodes {
            let path = format!("/sys/devices/system/node/node{}/cpulist", node.id);
            let cpulist = std::fs::read_to_string(&path).map_err(|error| {
                if error.kind() == std::io::ErrorKind::NotFound {
                    ProbeError::Skipped {
                        capability: NUMA_CAPABILITY,
                        reason: ProbeSkip::ChangedDuringProbe,
                    }
                } else {
                    ProbeError::Os(error)
                }
            })?;
            let cpus_after = parse_ids(&cpulist).ok_or_else(|| {
                ProbeError::Invalid(format!(
                    "invalid Linux NUMA cpulist for node {} after probe",
                    node.id
                ))
            })?;
            if cpus_after != node.cpus {
                return Err(ProbeError::Skipped {
                    capability: NUMA_CAPABILITY,
                    reason: ProbeSkip::ChangedDuringProbe,
                });
            }
        }
        Ok(nodes)
    }

    pub fn probe() -> Result<topology::CpuTopology, ProbeError> {
        match topology::linux::probe(|path| std::fs::read_to_string(path).ok()) {
            Ok(topology) => Ok(topology),
            Err(error) => match error.kind {
                topology::linux::ProbeErrorKind::Unavailable => Err(ProbeError::Skipped {
                    capability: "linux topology",
                    reason: ProbeSkip::MissingOsData,
                }),
                topology::linux::ProbeErrorKind::Changed => Err(ProbeError::Skipped {
                    capability: "linux topology",
                    reason: ProbeSkip::ChangedDuringProbe,
                }),
                topology::linux::ProbeErrorKind::Invalid => {
                    Err(ProbeError::Invalid(error.to_string()))
                }
            },
        }
    }

    #[cfg(test)]
    mod tests {
        use super::parse_ids;

        #[test]
        fn parser_preserves_sparse_and_memory_only_node_lists() {
            assert_eq!(parse_ids("0,4-5\n"), Some(vec![0, 4, 5]));
            assert_eq!(parse_ids("\n"), Some(Vec::new()));
        }

        #[test]
        fn parser_rejects_duplicate_or_out_of_range_ids() {
            assert_eq!(parse_ids("1,1"), None);
            assert_eq!(parse_ids("65536"), None);
            assert_eq!(parse_ids("2-1"), None);
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod macos_arm {
    use super::ProbeError;
    use crate::sysctl;
    use prism_platform::topology::{CoreKind, CpuTopology, TopologyBuilder, TopologyCore};

    /// Build a verified Apple-Silicon topology from the `hw.perflevel*` sysctl
    /// tree.
    ///
    /// Apple Silicon enumerates performance levels from fastest (`perflevel0`,
    /// the `P` cluster) to slowest (`perflevel1`, the `E` cluster) and has no
    /// `SMT`, so each logical core is its own physical core. Memory is unified,
    /// so there is a single `NUMA` node; the shared-`L2` cluster size
    /// (`cpusperl2`) defines the last-level-cache domains.
    pub fn probe() -> Result<CpuTopology, ProbeError> {
        let levels = sysctl::read_uint("hw.nperflevels").map_err(ProbeError::Os)?;
        let mut builder = TopologyBuilder::new();
        let mut logical_id: u32 = 0;
        let mut llc_domain: u16 = 0;

        for level in 0..levels {
            let count = sysctl::read_uint(&format!("hw.perflevel{level}.logicalcpu"))
                .map_err(ProbeError::Os)?;
            // Cores sharing one L2 slice form a cache domain; fall back to the
            // whole level if the kernel omits the key.
            let per_l2 = sysctl::read_uint(&format!("hw.perflevel{level}.cpusperl2"))
                .unwrap_or(count)
                .max(1);
            let kind = match (levels, level) {
                (0 | 1, _) => CoreKind::Unknown,
                (_, 0) => CoreKind::Performance,
                (_, 1) => CoreKind::Efficiency,
                _ => CoreKind::Unknown,
            };

            let mut in_domain: u64 = 0;
            for _ in 0..count {
                builder = builder.core(TopologyCore::new(
                    logical_id, logical_id, // no SMT on Apple Silicon: physical == logical.
                    0,          // unified memory: single NUMA node.
                    llc_domain, kind,
                ));
                logical_id += 1;
                in_domain += 1;
                if in_domain == per_l2 {
                    in_domain = 0;
                    llc_domain += 1;
                }
            }
            // Close any partial cluster so the next level starts a fresh L2 domain.
            if in_domain != 0 {
                llc_domain += 1;
            }
        }

        builder
            .build()
            .map_err(|e| ProbeError::Invalid(e.to_string()))
    }
}
