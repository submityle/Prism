//! Shared admission limits, checked before narrowing IDs or growing tables.

use crate::CompileError;

/// Limits for either declaration frontend. Counts must also fit u32 handles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphLimits {
    /// Maximum resources in each resource table.
    pub resources: usize,
    /// Maximum declared passes, including passes subsequently culled.
    pub passes: usize,
    /// Maximum accesses in one pass.
    pub accesses_per_pass: usize,
    /// Maximum SSA version of a resource.
    pub versions: u32,
}

impl Default for GraphLimits {
    fn default() -> Self {
        Self {
            resources: u32::MAX as usize,
            passes: u32::MAX as usize,
            accesses_per_pass: u32::MAX as usize,
            versions: u32::MAX,
        }
    }
}

impl GraphLimits {
    /// Checks a prospective table count before any cast or allocation.
    pub fn check_count(kind: &'static str, count: usize, limit: usize) -> Result<(), CompileError> {
        let limit = limit.min(u32::MAX as usize);
        if count > limit {
            Err(CompileError::Capacity { kind, count, limit })
        } else {
            Ok(())
        }
    }

    pub(crate) fn next_id(
        kind: &'static str,
        len: usize,
        limit: usize,
    ) -> Result<u32, CompileError> {
        let count = len.checked_add(1).ok_or(CompileError::Capacity {
            kind,
            count: len,
            limit: limit.min(u32::MAX as usize),
        })?;
        Self::check_count(kind, count, limit)?;
        Ok(len as u32)
    }

    pub(crate) fn next_version(self, current: u32) -> Result<u32, CompileError> {
        current
            .checked_add(1)
            .filter(|v| *v <= self.versions)
            .ok_or(CompileError::VersionOverflow {
                version: current,
                limit: self.versions,
            })
    }
}
