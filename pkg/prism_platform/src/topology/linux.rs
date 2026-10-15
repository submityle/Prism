//! Linux topology probe with injectable file reads, usable with `core`+`alloc`.
//!
//! The reader accepts absolute sysfs/procfs paths and returns `None` for an
//! unreadable file. Tests/hosts can supply a virtual filesystem. No FFI or
//! hardware assumptions are required. CPU/node lists are bounded to 65536 ids;
//! distance matrices to 256 accessible nodes, and cache scans to 32 indices per
//! CPU. Optional data outside those bounds is omitted, not guessed.
//!
//! CPU classes use the kernel's relative `cpu_capacity`, never frequencies or
//! CPU numbering: with complete heterogeneous capacity data, maximum capacity
//! is Performance, minimum is Efficiency and intermediate classes stay Unknown.
//! Equal or incomplete capacities leave *all* classes Unknown. NUMA/SMT are
//! mandatory; optional invalid distances/caches do not invalidate that geometry.

use alloc::{format, string::String, vec::Vec};
use core::fmt;

use super::{CoreKind, CpuTopology, TopologyBuilder, TopologyCache, TopologyCore, TopologySource};

const CPU_ONLINE: &str = "/sys/devices/system/cpu/online";
const NODE_ONLINE: &str = "/sys/devices/system/node/online";
const STATUS: &str = "/proc/thread-self/status";
const MAX_IDS: u32 = 65536;

/// A required file could not be read, parsed, or held stable during the probe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeError {
    /// File or topology relationship which failed validation.
    pub path: String,
    /// Nature of the failure.
    pub kind: ProbeErrorKind,
}

/// Why the Linux topology probe failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeErrorKind {
    /// Required data is not readable.
    Unavailable,
    /// Data is malformed, out of bounds, empty, or contradictory.
    Invalid,
    /// CPU/node availability changed while reading the snapshot.
    Changed,
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "topology {:?}: {}", self.kind, self.path)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ProbeError {}

fn error(path: &str, kind: ProbeErrorKind) -> ProbeError {
    ProbeError { path: path.into(), kind }
}

fn required(read: &mut impl FnMut(&str) -> Option<String>, path: &str) -> Result<String, ProbeError> {
    read(path).ok_or_else(|| error(path, ProbeErrorKind::Unavailable))
}

// Bounded, sorted and duplicate-free. Overlaps/duplicates in kernel lists
// indicate corruption rather than something to silently normalize away.
fn ids(text: &str) -> Option<Vec<u32>> {
    let mut ids = Vec::new();
    if text.trim().is_empty() { return Some(ids); }
    for part in text.trim().split(',') {
        let mut bounds = part.split('-');
        let start: u32 = bounds.next()?.parse().ok()?;
        let end = match bounds.next() { Some(end) => end.parse().ok()?, None => start };
        if bounds.next().is_some() || start > end || end >= MAX_IDS
            || ids.len().checked_add((end - start + 1) as usize)? > MAX_IDS as usize {
            return None;
        }
        ids.extend(start..=end);
    }
    ids.sort_unstable();
    if ids.windows(2).any(|pair| pair[0] == pair[1]) { return None; }
    Some(ids)
}

fn list(read: &mut impl FnMut(&str) -> Option<String>, path: &str) -> Result<Vec<u32>, ProbeError> {
    ids(&required(read, path)?).ok_or_else(|| error(path, ProbeErrorKind::Invalid))
}

fn allowed(read: &mut impl FnMut(&str) -> Option<String>) -> Result<Vec<u32>, ProbeError> {
    let status = required(read, STATUS)?;
    status.lines().find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .and_then(ids).filter(|ids| !ids.is_empty())
        .ok_or_else(|| error(STATUS, ProbeErrorKind::Invalid))
}

/// Probe Linux geometry, or explicitly fall back without losing known CPU ids.
///
/// A missing/malformed geometry file must not replace a restricted affinity set
/// (e.g. CPUs 8 and 12) with invented CPU ids 0 and 1. If even the allowed/online
/// lists are unreadable, ids are only uniform placeholders; failed affinity
/// operations must still be treated as unsupported placement, never success.
pub fn probe_or_fallback(mut read: impl FnMut(&str) -> Option<String>, fallback_count: usize) -> CpuTopology {
    if let Ok(topology) = probe(&mut read) { return topology; }
    let mut topology = CpuTopology::fallback(fallback_count.max(1), super::TopologyFallback::ProbeFailed);
    if let (Ok(online), Ok(allowed)) = (list(&mut read, CPU_ONLINE), allowed(&mut read)) {
        let cores: Vec<_> = online.into_iter().filter(|id| allowed.binary_search(id).is_ok())
            .map(|id| TopologyCore::new(id, id, 0, 0, CoreKind::Unknown)).collect();
        if !cores.is_empty() { topology.cores = cores; }
    }
    topology
}

/// Read a coherent snapshot of online, thread-allowed Linux CPU geometry.
///
/// Logical CPU and NUMA ids remain OS ids (possibly sparse). Physical ids are
/// canonical sibling representatives, so package-local `core_id` collisions
/// cannot merge cores across sockets. A final online/allowed-list reread rejects
/// detectable hotplug/cpuset races. Callers may retry after a `Changed` error;
/// no OS snapshot can guarantee that hotplug does not happen after this call.
///
/// # Errors
/// Returns [`ProbeError`] for unavailable/invalid mandatory geometry, overlapping
/// node membership or inconsistent SMT siblings. Missing NUMA sysfs on kernels
/// built without NUMA support is an explicit failure, not a fabricated probe.
pub fn probe(mut read: impl FnMut(&str) -> Option<String>) -> Result<CpuTopology, ProbeError> {
    let online = list(&mut read, CPU_ONLINE)?;
    let permitted = allowed(&mut read)?;
    let cpus: Vec<_> = online.iter().copied().filter(|id| permitted.binary_search(id).is_ok()).collect();
    if cpus.is_empty() { return Err(error(CPU_ONLINE, ProbeErrorKind::Invalid)); }
    let nodes = list(&mut read, NODE_ONLINE)?;
    if nodes.is_empty() { return Err(error(NODE_ONLINE, ProbeErrorKind::Invalid)); }
    let mut membership = Vec::new();
    for &node in &nodes {
        let path = format!("/sys/devices/system/node/node{node}/cpulist");
        for cpu in list(&mut read, &path)? {
            if cpus.binary_search(&cpu).is_ok() { membership.push((cpu, node as u16)); }
        }
    }
    membership.sort_unstable();
    if membership.len() != cpus.len()
        || !membership.iter().zip(&cpus).all(|((cpu, _), expected)| cpu == expected) {
        return Err(error(NODE_ONLINE, ProbeErrorKind::Invalid));
    }

    let mut cores = Vec::new();
    let mut siblings = Vec::new();
    let mut capacities = Vec::new();
    for &(cpu, node) in &membership {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        let group = list(&mut read, &path)?;
        if group.binary_search(&cpu).is_err() { return Err(error(&path, ProbeErrorKind::Invalid)); }
        cores.push(TopologyCore::new(cpu, group[0], node, 0, CoreKind::Unknown));
        siblings.push(group);
        capacities.push(read(&format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity"))
            .and_then(|text| text.trim().parse::<u64>().ok()).filter(|&n| n > 0));
    }
    for (i, group) in siblings.iter().enumerate() {
        for &cpu in group {
            if let Ok(j) = cpus.binary_search(&cpu) {
                if siblings[j] != *group { return Err(error("SMT siblings", ProbeErrorKind::Invalid)); }
                // SMT siblings are one core, so capacity cannot disagree.
                if capacities[i] != capacities[j] { capacities.fill(None); }
            }
        }
    }
    if capacities.iter().all(Option::is_some) {
        let min = capacities.iter().flatten().copied().min().unwrap_or(0);
        let max = capacities.iter().flatten().copied().max().unwrap_or(0);
        if min < max {
            for (core, capacity) in cores.iter_mut().zip(&capacities) {
                core.kind = if *capacity == Some(max) { CoreKind::Performance }
                    else if *capacity == Some(min) { CoreKind::Efficiency }
                    else { CoreKind::Unknown };
            }
        }
    }

    let mut topology = TopologyBuilder::new().with_cores(cores).build()
        .map_err(|_| error("SMT/NUMA geometry", ProbeErrorKind::Invalid))?;
    topology.source = TopologySource::LinuxSysfs;
    let accessible = topology.node_ids();
    topology.distances = distances(&mut read, &nodes, &accessible);
    topology.caches = caches(&mut read, &cpus);
    // Legacy scalar LLC ids are useful only for cores with a known last cache.
    // Unknown cores retain 0; consumers requiring evidence must use caches().
    let mut llcs: Vec<Vec<u32>> = Vec::new();
    for index in 0..topology.cores.len() {
        let physical_id = topology.cores[index].physical_id;
        if let Some(cache) = topology.caches.iter().filter(|cache| topology.cores.iter()
            .filter(|core| core.physical_id == physical_id)
            .all(|core| cache.cores.binary_search(&core.logical_id).is_ok()))
            .max_by_key(|cache| cache.level) {
            let llc_index = match llcs.iter().position(|cores| cores == &cache.cores) {
                Some(index) => index,
                None => { llcs.push(cache.cores.clone()); llcs.len() - 1 }
            };
            topology.cores[index].llc_domain = llc_index as u16;
        }
    }
    if list(&mut read, CPU_ONLINE)? != online || allowed(&mut read)? != permitted
        || list(&mut read, NODE_ONLINE)? != nodes {
        return Err(error("online/allowed CPUs or nodes", ProbeErrorKind::Changed));
    }
    Ok(topology)
}

fn distances(read: &mut impl FnMut(&str) -> Option<String>, nodes: &[u32], accessible: &[u16]) -> Option<Vec<Vec<u16>>> {
    if accessible.len() > 256 { return None; }
    let mut rows = Vec::new();
    for (row_index, &node) in accessible.iter().enumerate() {
        let text = read(&format!("/sys/devices/system/node/node{node}/distance"))?;
        let all: Vec<u16> = text.split_whitespace().map(str::parse).collect::<Result<_, _>>().ok()?;
        if all.len() != nodes.len() { return None; }
        let row: Vec<_> = accessible.iter().map(|&node| nodes.binary_search(&(node as u32)).ok().map(|index| all[index]))
            .collect::<Option<_>>()?;
        if row.iter().any(|&value| value == 0 || value < row[row_index]) { return None; }
        rows.push(row);
    }
    Some(rows)
}

fn size_bytes(text: &str) -> Option<u64> {
    let text = text.trim();
    let (digits, multiplier) = if let Some(digits) = text.strip_suffix('K') { (digits, 1024) }
        else if let Some(digits) = text.strip_suffix('M') { (digits, 1024 * 1024) }
        else { (text, 1) };
    digits.parse::<u64>().ok()?.checked_mul(multiplier).filter(|&size| size > 0)
}

fn caches(read: &mut impl FnMut(&str) -> Option<String>, cpus: &[u32]) -> Vec<TopologyCache> {
    let mut caches: Vec<(u32, TopologyCache)> = Vec::new();
    for &cpu in cpus {
        for index in 0..32 {
            let base = format!("/sys/devices/system/cpu/cpu{cpu}/cache/index{index}");
            let record = (|| {
                let level = read(&format!("{base}/level"))?.trim().parse::<u8>().ok().filter(|&n| n > 0)?;
                let kind = read(&format!("{base}/type"))?;
                if !matches!(kind.trim(), "Data" | "Unified") { return None; }
                let size_bytes = size_bytes(&read(&format!("{base}/size"))?)?;
                let line_bytes = read(&format!("{base}/coherency_line_size"))?.trim().parse::<u32>().ok().filter(|n| n.is_power_of_two())?;
                let sharers = ids(&read(&format!("{base}/shared_cpu_list"))?)?;
                if sharers.binary_search(&cpu).is_err() { return None; }
                let cores = sharers.into_iter().filter(|id| cpus.binary_search(id).is_ok()).collect();
                Some(TopologyCache { level, size_bytes, line_bytes, cores })
            })();
            if let Some(record) = record { caches.push((cpu, record)); }
        }
    }
    // Only publish a cache when every available sharer reports the same
    // geometry exactly once. Conflicting or partial hotplug records are omitted.
    let mut result = Vec::new();
    for (_, record) in &caches {
        let overlaps = |other: &TopologyCache| other.level == record.level
            && other.cores.iter().any(|id| record.cores.binary_search(id).is_ok());
        let valid = caches.iter().filter(|(_, other)| overlaps(other)).all(|(_, other)| other == record)
            && record.cores.iter().all(|cpu| caches.iter().filter(|(reporter, other)| reporter == cpu && other == record).count() == 1);
        if valid && !result.contains(record) { result.push(record.clone()); }
    }
    result
}

#[cfg(test)]
mod tests;
