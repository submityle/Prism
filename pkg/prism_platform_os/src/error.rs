//! Shared error type for the real-OS probes in this crate.

/// Why an otherwise supported probe was skipped without fabricating data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeSkip {
    /// Required sysfs/procfs data is not exposed in this environment.
    MissingOsData,
    /// The topology changed while a coherent snapshot was being read.
    ChangedDuringProbe,
}

/// Why a real OS probe could not produce a genuine (`is_probed() == true`)
/// result.
#[derive(Debug)]
pub enum ProbeError {
    /// The current target has no verified real-probe backend; callers should
    /// fall back to the portable best-effort model in `prism_platform`.
    Unsupported,
    /// The backend exists, but this process/environment does not expose a
    /// stable data source for it. This is a structured skip, not success.
    Skipped {
        /// Capability whose probe was skipped.
        capability: &'static str,
        /// Structured reason for skipping the probe.
        reason: ProbeSkip,
    },
    /// A required OS query failed (for example a syscall denied by a sandbox).
    Os(std::io::Error),
    /// The OS values were read but did not form a valid, self-consistent model.
    Invalid(String),
}

impl core::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ProbeError::Unsupported => f.write_str("no verified real OS probe for this target"),
            ProbeError::Skipped { capability, reason } => {
                write!(f, "{capability} probe skipped: {reason:?}")
            }
            ProbeError::Os(e) => write!(f, "OS probe query failed: {e}"),
            ProbeError::Invalid(msg) => write!(f, "probed values rejected: {msg}"),
        }
    }
}

impl std::error::Error for ProbeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProbeError::Os(e) => Some(e),
            ProbeError::Unsupported | ProbeError::Skipped { .. } | ProbeError::Invalid(_) => None,
        }
    }
}
