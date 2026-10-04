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
//! The genuine read is implemented and verified only for **Apple Silicon
//! macOS** (`aarch64`), where the `hw.perflevel{0,1}.*` sysctl tree gives an
//! exact, SMT-free description of the performance (`P`) and efficiency (`E`)
//! clusters and their shared `L2` domains. On every other target
//! [`probe_topology`] returns [`ProbeError::Unsupported`] rather than guessing
//! at an `SMT`/`NUMA` layout it cannot confirm on this host.
//!
//! [`CpuTopology`]: prism_platform::topology::CpuTopology
//! [`CpuTopology::detect`]: prism_platform::topology::CpuTopology::detect
//! [`CpuTopology::is_probed`]: prism_platform::topology::CpuTopology::is_probed

use crate::error::ProbeError;
use prism_platform::topology::CpuTopology;

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
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        macos_arm::probe()
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        Err(ProbeError::Unsupported)
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
                    logical_id,
                    logical_id, // no SMT on Apple Silicon: physical == logical.
                    0,          // unified memory: single NUMA node.
                    llc_domain,
                    kind,
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
