use super::*;
use alloc::{collections::BTreeMap, vec};

fn fixture() -> BTreeMap<String, String> {
    let mut files = BTreeMap::new();
    files.insert(CPU_ONLINE.into(), "0,2,4,6".into());
    files.insert(STATUS.into(), "Name:\ttest\nCpus_allowed_list:\t2,4,6-7\n".into());
    files.insert(NODE_ONLINE.into(), "2,9,12".into());
    for (node, cpus, distance) in [(2, "0,2", "10 30 50"), (9, "4,6", "25 10 60"), (12, "", "50 60 10")] {
        files.insert(format!("/sys/devices/system/node/node{node}/cpulist"), cpus.into());
        files.insert(format!("/sys/devices/system/node/node{node}/distance"), distance.into());
    }
    for (cpu, siblings, capacity) in [(2, "0,2", "1024"), (4, "4,6", "512"), (6, "4,6", "512")] {
        files.insert(format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"), siblings.into());
        files.insert(format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity"), capacity.into());
        let base = format!("/sys/devices/system/cpu/cpu{cpu}/cache/index3");
        for (field, value) in [("level", "3"), ("type", "Unified"), ("size", "2M"), ("coherency_line_size", "64"), ("shared_cpu_list", siblings)] {
            files.insert(format!("{base}/{field}"), value.into());
        }
    }
    files
}

fn run(files: &BTreeMap<String, String>) -> Result<CpuTopology, ProbeError> {
    probe(|path| files.get(path).cloned())
}

#[test]
fn sparse_ids_allowed_online_smt_capacity_cache_and_distances() {
    let topology = run(&fixture()).unwrap();
    assert_eq!(topology.source(), TopologySource::LinuxSysfs);
    assert!(topology.is_probed());
    assert!(topology.is_hybrid());
    assert_eq!(topology.cores().iter().map(|c| c.logical_id).collect::<Vec<_>>(), vec![2, 4, 6]);
    assert_eq!(topology.node_ids(), vec![2, 9]);
    assert_eq!(topology.numa_node_count(), 2);
    assert_eq!(topology.physical_core_count(), 2);
    assert!(!topology.has_smt_sibling(2)); // its other sibling is not permitted
    assert!(topology.has_smt_sibling(4));
    assert_eq!(topology.cores()[0].physical_id, 0); // OS sibling representative
    assert_eq!(topology.cores()[0].kind, CoreKind::Performance);
    assert_eq!(topology.cores()[1].kind, CoreKind::Efficiency);
    assert_eq!(topology.numa_distances().unwrap(), &[vec![10, 30], vec![25, 10]]);
    assert_eq!(topology.caches().len(), 2);
    assert_eq!(topology.caches()[0].cores, vec![2]);
    assert_eq!(topology.caches()[1].cores, vec![4, 6]);
    assert_eq!(topology.caches()[1].size_bytes, 2 * 1024 * 1024);
    assert_eq!(topology.caches()[1].line_bytes, 64);
}

#[test]
fn cpu_list_parser_rejects_unbounded_malformed_and_duplicate_ids() {
    for text in ["-1", "3-1", "0-65536", "4294967295", "1,1", "1-3,2", "1-2-3", "1,", "1, 2", "0-65535,0"] {
        assert_eq!(ids(text), None, "{text}");
    }
    assert_eq!(ids("6,2-3\n"), Some(vec![2, 3, 6]));
    assert_eq!(ids("65535"), Some(vec![65535]));
    assert_eq!(ids("\n"), Some(vec![]));
}

#[test]
fn missing_required_data_is_not_a_fake_single_node_probe() {
    for missing in [CPU_ONLINE, NODE_ONLINE, STATUS, "/sys/devices/system/cpu/cpu4/topology/thread_siblings_list", "/sys/devices/system/node/node9/cpulist"] {
        let mut files = fixture();
        files.remove(missing);
        let error = run(&files).unwrap_err();
        assert_eq!(error.path, missing);
        assert_eq!(error.kind, ProbeErrorKind::Unavailable);
    }
}

#[test]
fn rejects_no_online_allowed_cpu_and_invalid_status() {
    for status in ["Cpus_allowed_list:\t7", "Cpus_allowed_list:\t", "Cpus_allowed:\tff", "Cpus_allowed_list:\t2-1"] {
        let mut files = fixture();
        files.insert(STATUS.into(), status.into());
        assert_eq!(run(&files).unwrap_err().kind, ProbeErrorKind::Invalid);
    }
}

#[test]
fn rejects_overlapping_missing_or_cross_node_smt_membership() {
    for cpus in ["2,4,6", "4", "2,4"] {
        let mut files = fixture();
        files.insert("/sys/devices/system/node/node9/cpulist".into(), cpus.into());
        assert_eq!(run(&files).unwrap_err().kind, ProbeErrorKind::Invalid);
    }
    let mut files = fixture();
    files.insert("/sys/devices/system/node/node2/cpulist".into(), "0,2,4".into());
    files.insert("/sys/devices/system/node/node9/cpulist".into(), "6".into());
    assert_eq!(run(&files).unwrap_err().kind, ProbeErrorKind::Invalid);
}

#[test]
fn rejects_asymmetric_or_missing_self_smt_lists() {
    for siblings in ["4", "2,4,6", "2"] {
        let mut files = fixture();
        files.insert("/sys/devices/system/cpu/cpu4/topology/thread_siblings_list".into(), siblings.into());
        assert_eq!(run(&files).unwrap_err().kind, ProbeErrorKind::Invalid);
    }
}

#[test]
fn rejects_detectable_hotplug_and_allowed_set_changes() {
    for changing in [CPU_ONLINE, NODE_ONLINE, STATUS] {
        let files = fixture();
        let mut reads = 0;
        let result = probe(|path| {
            if path == changing {
                reads += 1;
                if reads > 1 {
                    return Some(if path == STATUS { "Cpus_allowed_list: 2" } else { "2" }.into());
                }
            }
            files.get(path).cloned()
        });
        assert_eq!(result.unwrap_err().kind, ProbeErrorKind::Changed, "{changing}");
    }
}

#[test]
fn optional_missing_or_inconsistent_capacities_remain_unknown() {
    for value in [None, Some("0"), Some("garbage"), Some("1024")] {
        let mut files = fixture();
        let path = "/sys/devices/system/cpu/cpu4/cpu_capacity";
        match value { Some(value) => { files.insert(path.into(), value.into()); }, None => { files.remove(path); } }
        let topology = run(&files).unwrap();
        assert!(!topology.is_hybrid());
        assert!(topology.cores().iter().all(|c| c.kind == CoreKind::Unknown));
    }
}

#[test]
fn equal_capacity_does_not_infer_performance_class() {
    let mut files = fixture();
    files.insert("/sys/devices/system/cpu/cpu2/cpu_capacity".into(), "512".into());
    assert!(run(&files).unwrap().cores().iter().all(|c| c.kind == CoreKind::Unknown));
}

#[test]
fn invalid_optional_distances_do_not_replace_real_numa_geometry() {
    for row in ["10 30", "0 30 50", "30 10 50", "10 bad 50", "65536 30 50"] {
        let mut files = fixture();
        files.insert("/sys/devices/system/node/node2/distance".into(), row.into());
        let topology = run(&files).unwrap();
        assert!(topology.numa_distances().is_none());
        assert_eq!(topology.node_ids(), vec![2, 9]);
        assert!(topology.is_probed());
    }
}

#[test]
fn incomplete_conflicting_or_instruction_only_cache_is_omitted() {
    for (field, value) in [("size", "1M"), ("shared_cpu_list", "6"), ("coherency_line_size", "3"), ("type", "Instruction")] {
        let mut files = fixture();
        files.insert(format!("/sys/devices/system/cpu/cpu6/cache/index3/{field}"), value.into());
        let topology = run(&files).unwrap();
        assert_eq!(topology.caches().len(), 1, "{field}");
        assert_eq!(topology.caches()[0].cores, vec![2]);
    }
}

#[test]
fn cache_size_parser_is_checked() {
    assert_eq!(size_bytes("32K\n"), Some(32768));
    assert_eq!(size_bytes("2M"), Some(2097152));
    assert_eq!(size_bytes("4096"), Some(4096));
    for text in ["0", "-1K", "1G", "18446744073709551615M"] { assert_eq!(size_bytes(text), None); }
}

#[test]
fn fallback_provenance_never_claims_hardware_geometry() {
    for reason in [super::super::TopologyFallback::NoStd, super::super::TopologyFallback::UnsupportedPlatform, super::super::TopologyFallback::ProbeFailed] {
        let topology = CpuTopology::fallback(3, reason);
        assert_eq!(topology.source(), TopologySource::Fallback(reason));
        assert!(!topology.is_probed());
        assert!(!topology.is_hybrid());
        assert_eq!(topology.numa_node_count(), 1);
        assert!(topology.caches().is_empty());
        assert!(topology.numa_distances().is_none());
        assert!(topology.cores().iter().all(|core| core.kind == CoreKind::Unknown));
    }
}

#[test]
fn live_probe_reports_real_source_or_explicit_fallback() {
    let topology = CpuTopology::detect();
    assert!(topology.logical_core_count() > 0);
    if matches!(topology.source(), TopologySource::Fallback(_)) {
        assert!(!topology.is_probed());
        assert_eq!(topology.numa_node_count(), 1);
        assert!(topology.cores().iter().all(|core| core.kind == CoreKind::Unknown));
    } else {
        assert_eq!(topology.source(), TopologySource::LinuxSysfs);
        assert!(topology.is_probed());
    }
    // No hardware-success assertion: on macOS this must be a fallback.
}

#[test]
fn fallback_preserves_sparse_allowed_ids_and_handles_unreadable_files() {
    let mut files = fixture();
    files.remove(NODE_ONLINE);
    let topology = probe_or_fallback(|path| files.get(path).cloned(), 2);
    assert!(!topology.is_probed());
    assert_eq!(topology.cores().iter().map(|core| core.logical_id).collect::<Vec<_>>(), vec![2, 4, 6]);
    assert_eq!(topology.numa_node_count(), 1);
    let topology = probe_or_fallback(|_| None, 0);
    assert_eq!(topology.logical_core_count(), 1);
    assert!(!topology.is_probed());
}
