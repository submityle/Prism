//! Real-machine checks for the OS topology probe.
//!
//! These are not golden-vs-fixed-input oracles: they probe the *actual* host
//! and assert the result is internally consistent and agrees with the std
//! runtime. On a target without a verified backend the probe reports
//! [`ProbeError::Unsupported`] and the checks degrade to confirming that honest
//! refusal rather than a fabricated topology.
//!
//! Note: `sysctl` can be denied inside a sandbox, in which case the probe
//! surfaces [`ProbeError::Os`]; run these unsandboxed to exercise the real
//! read.

use crate::{probe_topology, ProbeError};
use prism_platform::topology::CoreKind;

#[test]
fn probe_agrees_with_runtime_or_refuses_honestly() {
    match probe_topology() {
        Ok(topo) => {
            // A real probe must mark itself as such, unlike `detect()`.
            assert!(topo.is_probed(), "a successful probe must set is_probed()");

            // Logical-core count must match what the std runtime schedules on.
            let runtime = std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1);
            assert_eq!(
                topo.logical_core_count(),
                runtime,
                "probed logical cores must match std available_parallelism"
            );

            // Physical cores cannot exceed logical cores, and both are >= 1.
            assert!(topo.logical_core_count() >= 1);
            assert!(topo.physical_core_count() >= 1);
            assert!(topo.physical_core_count() <= topo.logical_core_count());

            // At least one NUMA node always exists.
            assert!(topo.numa_node_count() >= 1);

            // Every core carries a NUMA node within the reported range.
            for core in topo.cores() {
                assert!(core.numa_node < topo.numa_node_count());
            }
        }
        // Honest refusal on an unverified target is an acceptable outcome.
        Err(ProbeError::Unsupported) => {}
        Err(e) => panic!("unexpected probe failure: {e}"),
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn apple_silicon_is_hybrid_smt_free_single_node() {
    let topo = match probe_topology() {
        Ok(t) => t,
        // sysctl denied (sandbox): cannot exercise the real read here.
        Err(ProbeError::Os(_)) => return,
        Err(e) => panic!("probe failed on Apple Silicon: {e}"),
    };

    // Apple Silicon ships hybrid P/E cores.
    assert!(topo.is_hybrid(), "Apple Silicon must report a P/E split");
    assert!(
        topo.cores_of_kind(CoreKind::Performance).count() >= 1,
        "expected at least one P core"
    );
    assert!(
        topo.cores_of_kind(CoreKind::Efficiency).count() >= 1,
        "expected at least one E core"
    );

    // No SMT: physical core count equals logical core count, and no logical
    // core reports a sibling.
    assert_eq!(
        topo.physical_core_count(),
        topo.logical_core_count(),
        "Apple Silicon has no SMT"
    );
    for core in topo.cores() {
        assert!(
            !topo.has_smt_sibling(core.logical_id),
            "Apple Silicon core {} must have no SMT sibling",
            core.logical_id
        );
    }

    // Unified memory: exactly one NUMA node.
    assert_eq!(topo.numa_node_count(), 1, "unified memory is a single node");

    // P and E clusters must live in distinct last-level-cache domains: no
    // performance core may share an `llc_domain` with any efficiency core.
    for p in topo.cores_of_kind(CoreKind::Performance) {
        for e in topo.cores_of_kind(CoreKind::Efficiency) {
            assert_ne!(
                p.llc_domain, e.llc_domain,
                "P core {} and E core {} must not share an L2 domain",
                p.logical_id, e.logical_id
            );
        }
    }
}
