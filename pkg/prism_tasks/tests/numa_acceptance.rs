//! Real-OS NUMA/affinity acceptance evidence.
//!
//! This test deliberately does not use the portable topology model as proof of
//! hardware support. It asks `prism_platform_os` for a probed topology, reads
//! the calling thread's effective affinity mask, exercises a real node/core
//! binding when the host exposes one, and then checks the evidence published by
//! `TaskPool` and its frame arenas.
//!
//! The default mode is diagnostic: unsupported or sandbox-denied operations are
//! reported as `SKIPPED` and the test remains useful on hosts without NUMA.
//! `PRISM_REQUIRE_NUMA=1` turns every missing real capability or rejected
//! binding into a failure. No synthetic topology, timing value, or requested
//! placement is counted as a success.

#![expect(
    clippy::print_stderr,
    reason = "acceptance diagnostics are intentionally emitted to stderr"
)]

use prism_platform::{current_thread_affinity, set_current_thread_affinity_mask};
use prism_platform_os::{bind_current_thread_to_node, probe_numa_nodes, probe_topology};
use prism_tasks::{CoreClassPolicy, TaskPool, TaskPoolConfig};

fn require_numa() -> bool {
    matches!(std::env::var("PRISM_REQUIRE_NUMA"), Ok(value) if value == "1")
}

fn skip_or_fail(message: impl std::fmt::Display) {
    if require_numa() {
        panic!("PRISM_REQUIRE_NUMA=1: {message}");
    }
    eprintln!("SKIPPED NUMA/affinity acceptance: {message}");
}

#[test]
fn real_os_numa_affinity_and_worker_placement_are_verified() {
    let topology = match probe_topology() {
        Ok(topology) => topology,
        Err(error) => {
            skip_or_fail(format!("real topology probe unavailable: {error}"));
            return;
        }
    };
    assert!(
        topology.is_probed(),
        "probe_topology returned data without real probe provenance"
    );
    let capabilities = topology.capabilities();
    eprintln!(
        "NUMA topology: source={:?} logical_cores={} physical_cores={} nodes={} capabilities={{cpu_affinity:{}, numa_memory_binding:{}}}",
        topology.source(),
        topology.logical_core_count(),
        topology.physical_core_count(),
        topology.numa_node_count(),
        capabilities.cpu_affinity,
        capabilities.numa_memory_binding,
    );

    let original_mask = match current_thread_affinity() {
        Ok(mask) if !mask.is_empty() => Some(mask),
        Ok(_) => {
            skip_or_fail("OS returned an empty effective CPU-affinity mask");
            None
        }
        Err(error) => {
            skip_or_fail(format!("effective affinity read-back unavailable: {error}"));
            None
        }
    };

    if capabilities.cpu_affinity {
        let Some(original_mask) = original_mask.as_deref() else {
            return;
        };
        let core = topology
            .cores()
            .iter()
            .map(|core| core.logical_id as usize)
            .find(|core| original_mask.contains(core));
        let Some(core) = core else {
            skip_or_fail("real topology has no core in the process effective affinity mask");
            return;
        };
        if let Err(error) = set_current_thread_affinity_mask(&[core]) {
            skip_or_fail(format!(
                "OS rejected verified single-core affinity: {error}"
            ));
            return;
        }
        let effective = current_thread_affinity()
            .expect("affinity was accepted but effective mask read-back failed");
        assert_eq!(
            effective,
            vec![core],
            "accepted singleton affinity did not read back as the requested mask"
        );
        eprintln!("PASS CPU affinity: requested_core={core} effective_mask={effective:?}");
        set_current_thread_affinity_mask(original_mask)
            .expect("restore of the acceptance test thread affinity failed");
    } else {
        skip_or_fail("real topology is available but CPU affinity capability is unsupported");
    }

    let nodes = match probe_numa_nodes() {
        Ok(nodes) if !nodes.is_empty() => nodes,
        Ok(_) => {
            skip_or_fail("NUMA probe returned no online nodes");
            Vec::new()
        }
        Err(error) => {
            skip_or_fail(format!("NUMA node probe unavailable: {error}"));
            Vec::new()
        }
    };
    if !nodes.is_empty() && capabilities.cpu_affinity {
        let Some(original_mask) = original_mask.as_deref() else {
            return;
        };
        let node = nodes.iter().find(|node| {
            node.cpus
                .iter()
                .any(|cpu| original_mask.contains(&(*cpu as usize)))
        });
        let Some(node) = node else {
            skip_or_fail("no probed NUMA node intersects the process affinity mask");
            return;
        };
        match bind_current_thread_to_node(node.id) {
            Ok(()) => {
                let effective = current_thread_affinity()
                    .expect("NUMA binding was accepted but mask read-back failed");
                assert!(!effective.is_empty());
                assert!(effective
                    .iter()
                    .all(|cpu| node.cpus.contains(&(*cpu as u32))));
                eprintln!(
                    "PASS NUMA CPU binding: node={} effective_mask={effective:?}",
                    node.id
                );
                set_current_thread_affinity_mask(original_mask)
                    .expect("restore after NUMA binding acceptance failed");
            }
            Err(error) => {
                skip_or_fail(format!("NUMA node binding was rejected: {error}"));
            }
        }
    }

    let worker_count = topology.logical_core_count().clamp(1, 2);
    let pool = TaskPool::with_config(TaskPoolConfig {
        threads: worker_count,
        pin_workers: true,
        core_class_policy: CoreClassPolicy::Flat,
    });
    let Some(assignments) = pool.worker_core_assignments() else {
        skip_or_fail("task pool could not produce a real worker placement plan");
        return;
    };
    let Some(records) = pool.worker_binding_outcomes() else {
        skip_or_fail("task pool could not publish real worker binding evidence");
        return;
    };
    assert_eq!(assignments.len(), worker_count);
    assert_eq!(records.len(), assignments.len());
    let bound = records
        .iter()
        .filter(|record| record.outcome.is_bound())
        .count();
    for (index, record) in records.iter().enumerate() {
        let assignment = record
            .outcome
            .assignment()
            .expect("every planned worker needs an assignment");
        assert_eq!(assignment, assignments[index]);
        if record.outcome.is_bound() {
            if let Some(mask) = &record.effective_cpus {
                assert!(mask.contains(&assignment.core));
            }
        } else {
            eprintln!(
                "SKIPPED worker binding: worker={} core={} outcome={:?}",
                index, assignment.core, record.outcome
            );
        }
    }
    if bound == records.len() {
        eprintln!("PASS worker affinity: bound={bound}/{}", records.len());
    } else {
        skip_or_fail(format!("worker affinity bound={bound}/{}", records.len()));
    }

    let arenas = pool.new_frame_arenas(4096);
    let placement = arenas.placement_summary();
    eprintln!(
        "NUMA arena placement: bound={} fallback={} requested_arenas={}",
        placement.bound,
        placement.fallback,
        arenas.len()
    );
    if placement.fallback == 0 {
        eprintln!("PASS NUMA arena placement: every arena has an OS binding");
    } else {
        skip_or_fail(format!(
            "{} of {} frame arenas used explicit ordinary-memory fallback",
            placement.fallback,
            arenas.len()
        ));
    }
}
