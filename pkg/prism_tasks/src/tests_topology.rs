//! §24.6 NUMA / hybrid-core topology data-model tests.
//!
//! Anti-vacuous contract: every pure function is checked against an
//! independently hand-computed oracle on fixed inputs — the distance matrix's
//! validation and symmetry, the distance-aware node ranking / nearest / load-
//! balanced tie-break, the workload-class worker placement (P-core for
//! latency, E-core for background, NUMA-local-first ordering) with its memory-
//! affinity suggestions, the P/E partition and its homogeneous fallback, and
//! the cache-hierarchy closeness queries.

use alloc::vec;
use alloc::vec::Vec;

use crate::{
    choose_balanced_node, nearest_node, plan_partitioned, plan_placement,
    rank_nodes_by_distance, shared_l3, CacheInfo, CacheLevel, CacheTopology, CoreClass, CoreInfo,
    DistanceError, NumaDistanceMatrix, NumaNodeId, Topology, TopologyDescriptor, TopologyError,
    WorkloadClass, LOCAL_DISTANCE,
};

fn node(index: u16) -> NumaNodeId {
    NumaNodeId::new(index)
}

fn pcore(id: usize, n: u16) -> CoreInfo {
    CoreInfo {
        id,
        node: node(n),
        class: CoreClass::Performance,
    }
}

fn ecore(id: usize, n: u16) -> CoreInfo {
    CoreInfo {
        id,
        node: node(n),
        class: CoreClass::Efficiency,
    }
}

// ---------------------------------------------------------------------------
// Distance matrix
// ---------------------------------------------------------------------------

#[test]
fn single_matrix_is_one_local_node() {
    let m = NumaDistanceMatrix::single();
    assert_eq!(m.node_count(), 1);
    assert_eq!(m.distance(node(0), node(0)), LOCAL_DISTANCE);
    assert!(m.is_symmetric());
}

#[test]
fn uniform_matrix_diagonal_local_offdiagonal_remote() {
    let m = NumaDistanceMatrix::uniform(3, 20);
    assert_eq!(m.node_count(), 3);
    for a in 0..3u16 {
        for b in 0..3u16 {
            let expected = if a == b { LOCAL_DISTANCE } else { 20 };
            assert_eq!(m.distance(node(a), node(b)), expected);
        }
    }
    assert!(m.is_symmetric());
}

#[test]
fn uniform_matrix_clamps_remote_below_local() {
    // A remote value cheaper than local is nonsensical; it is clamped up.
    let m = NumaDistanceMatrix::uniform(2, 5);
    assert_eq!(m.distance(node(0), node(1)), LOCAL_DISTANCE);
}

#[test]
fn from_rows_builds_asymmetric_matrix() {
    let m = NumaDistanceMatrix::from_rows(vec![
        vec![10, 20, 30],
        vec![20, 10, 25],
        vec![30, 24, 10],
    ])
    .unwrap();
    assert_eq!(m.node_count(), 3);
    assert_eq!(m.distance(node(0), node(2)), 30);
    assert_eq!(m.distance(node(2), node(1)), 24);
    // 24 != 25, so the matrix is asymmetric.
    assert!(!m.is_symmetric());
    assert_eq!(m.get(node(5), node(0)), None);
}

#[test]
fn from_rows_rejects_bad_input() {
    assert_eq!(NumaDistanceMatrix::from_rows(vec![]), Err(DistanceError::Empty));
    assert_eq!(
        NumaDistanceMatrix::from_rows(vec![vec![10, 20], vec![20]]),
        Err(DistanceError::NotSquare {
            rows: 2,
            cols: 1,
            row: 1
        })
    );
    assert_eq!(
        NumaDistanceMatrix::from_rows(vec![vec![10, 0], vec![20, 10]]),
        Err(DistanceError::ZeroDistance { from: 0, to: 1 })
    );
    // Node 1's self-distance (15) is larger than its remote entry (12).
    assert_eq!(
        NumaDistanceMatrix::from_rows(vec![vec![10, 20], vec![12, 15]]),
        Err(DistanceError::DiagonalNotMinimal { node: 1 })
    );
}

#[test]
#[should_panic(expected = "out of range")]
fn distance_panics_out_of_range() {
    let m = NumaDistanceMatrix::uniform(2, 20);
    let _ = m.distance(node(0), node(9));
}

// ---------------------------------------------------------------------------
// Distance-aware scheduling
// ---------------------------------------------------------------------------

fn asym_3() -> NumaDistanceMatrix {
    NumaDistanceMatrix::from_rows(vec![
        vec![10, 20, 30],
        vec![20, 10, 25],
        vec![30, 25, 10],
    ])
    .unwrap()
}

#[test]
fn rank_nodes_orders_by_distance() {
    let m = asym_3();
    // From node 0: distances 10,20,30 -> already sorted.
    assert_eq!(
        rank_nodes_by_distance(&m, node(0)),
        vec![node(0), node(1), node(2)]
    );
    // From node 2: distances to 0,1,2 = 30,25,10 -> [2,1,0].
    assert_eq!(
        rank_nodes_by_distance(&m, node(2)),
        vec![node(2), node(1), node(0)]
    );
    // Out-of-range source -> empty.
    assert!(rank_nodes_by_distance(&m, node(9)).is_empty());
}

#[test]
fn rank_nodes_breaks_ties_by_index() {
    // From node 0, nodes 1 and 2 are both distance 20; index decides.
    let m = NumaDistanceMatrix::from_rows(vec![
        vec![10, 20, 20],
        vec![20, 10, 20],
        vec![20, 20, 10],
    ])
    .unwrap();
    assert_eq!(
        rank_nodes_by_distance(&m, node(0)),
        vec![node(0), node(1), node(2)]
    );
}

#[test]
fn nearest_node_picks_closest_candidate() {
    let m = asym_3();
    // From node 2, candidates {0,1}: distances 30,25 -> node 1.
    assert_eq!(
        nearest_node(&m, node(2), &[node(0), node(1)]),
        Some(node(1))
    );
    // Empty candidates -> None.
    assert_eq!(nearest_node(&m, node(2), &[]), None);
    // Out-of-range candidates are skipped; only node 0 remains.
    assert_eq!(
        nearest_node(&m, node(1), &[node(9), node(0)]),
        Some(node(0))
    );
}

#[test]
fn choose_balanced_node_prefers_distance_then_load_then_index() {
    // From node 0, nodes 0 and 1 are both distance 10 (local tie); node 2 is 20.
    let m = NumaDistanceMatrix::from_rows(vec![
        vec![10, 10, 20],
        vec![10, 10, 20],
        vec![20, 20, 10],
    ])
    .unwrap();
    // Distance ties between nodes 0 and 1 -> least loaded (node 1, load 2) wins.
    assert_eq!(
        choose_balanced_node(&m, node(0), &[5, 2, 9]),
        Some(node(1))
    );
    // Equal distance AND equal load -> lowest index (node 0) wins.
    assert_eq!(
        choose_balanced_node(&m, node(0), &[3, 3, 0]),
        Some(node(0))
    );
    // Node 2 is farther; even at zero load it loses to the nearer nodes.
    assert_eq!(
        choose_balanced_node(&m, node(0), &[7, 7, 0]),
        Some(node(0))
    );
}

#[test]
fn choose_balanced_node_edge_cases() {
    let m = NumaDistanceMatrix::uniform(3, 20);
    // Empty loads -> None.
    assert_eq!(choose_balanced_node(&m, node(0), &[]), None);
    // Out-of-range home -> None.
    assert_eq!(choose_balanced_node(&m, node(9), &[1, 1, 1]), None);
    // Fewer loads than nodes: only the first two nodes are considered.
    // From node 0 the local node 0 (distance 10) always wins over node 1 (20).
    assert_eq!(choose_balanced_node(&m, node(0), &[4, 1]), Some(node(0)));
}

// ---------------------------------------------------------------------------
// Workload-class placement
// ---------------------------------------------------------------------------

// A 2-node hybrid: P-cores id 0 (node 0) / 1 (node 1); E-cores id 2 (node 0) /
// 3 (node 1).
fn hybrid_2node() -> TopologyDescriptor {
    TopologyDescriptor::new(
        vec![pcore(0, 0), pcore(1, 1), ecore(2, 0), ecore(3, 1)],
        NumaDistanceMatrix::uniform(2, 20),
        CacheTopology::empty(),
    )
    .unwrap()
}

#[test]
fn latency_sensitive_fills_pcores_first() {
    let topo = hybrid_2node();
    let plan = plan_placement(&topo, WorkloadClass::LatencySensitive, 4);
    // Ordering: P-cores by node then id (0,1), then E-cores (2,3).
    let cores: Vec<usize> = plan.placements().iter().map(|p| p.core).collect();
    assert_eq!(cores, vec![0, 1, 2, 3]);
    // First two workers land on performance cores.
    assert_eq!(plan.placement(0).unwrap().class, CoreClass::Performance);
    assert_eq!(plan.placement(1).unwrap().class, CoreClass::Performance);
    assert_eq!(plan.placement(2).unwrap().class, CoreClass::Efficiency);
    // Memory node equals the assigned core's node.
    assert_eq!(plan.worker_nodes(), vec![node(0), node(1), node(0), node(1)]);
    assert_eq!(plan.memory_nodes(), plan.worker_nodes());
}

#[test]
fn background_fills_ecores_first() {
    let topo = hybrid_2node();
    let plan = plan_placement(&topo, WorkloadClass::Background, 2);
    let cores: Vec<usize> = plan.placements().iter().map(|p| p.core).collect();
    // E-cores first: ids 2 (node 0) then 3 (node 1).
    assert_eq!(cores, vec![2, 3]);
    assert!(plan
        .placements()
        .iter()
        .all(|p| p.class == CoreClass::Efficiency));
}

#[test]
fn throughput_matches_performance_first_ordering() {
    let topo = hybrid_2node();
    let plan = plan_placement(&topo, WorkloadClass::Throughput, 4);
    let cores: Vec<usize> = plan.placements().iter().map(|p| p.core).collect();
    assert_eq!(cores, vec![0, 1, 2, 3]);
}

#[test]
fn placement_oversubscribes_round_robin() {
    let topo = hybrid_2node();
    let plan = plan_placement(&topo, WorkloadClass::LatencySensitive, 6);
    let cores: Vec<usize> = plan.placements().iter().map(|p| p.core).collect();
    // 6 workers over 4 cores (ordered 0,1,2,3) wraps around.
    assert_eq!(cores, vec![0, 1, 2, 3, 0, 1]);
    assert_eq!(plan.len(), 6);
    assert!(!plan.is_empty());
}

#[test]
fn placement_numa_local_first_within_class() {
    // All P-cores, two per node: node-local packing must group node 0 first.
    let topo = TopologyDescriptor::new(
        vec![pcore(0, 0), pcore(1, 1), pcore(2, 0), pcore(3, 1)],
        NumaDistanceMatrix::uniform(2, 20),
        CacheTopology::empty(),
    )
    .unwrap();
    let plan = plan_placement(&topo, WorkloadClass::LatencySensitive, 4);
    let cores: Vec<usize> = plan.placements().iter().map(|p| p.core).collect();
    // Node 0 cores (ids 0,2) come before node 1 cores (ids 1,3).
    assert_eq!(cores, vec![0, 2, 1, 3]);
}

#[test]
fn empty_or_zero_placement_is_empty() {
    let topo = hybrid_2node();
    assert!(plan_placement(&topo, WorkloadClass::Throughput, 0).is_empty());
}

#[test]
fn partition_splits_pcores_and_ecores_disjointly() {
    let topo = hybrid_2node();
    let (latency, background) = plan_partitioned(&topo, 2, 2);
    assert!(latency
        .placements()
        .iter()
        .all(|p| p.class == CoreClass::Performance));
    assert!(background
        .placements()
        .iter()
        .all(|p| p.class == CoreClass::Efficiency));
    // Disjoint core sets.
    let lat: Vec<usize> = latency.placements().iter().map(|p| p.core).collect();
    let bg: Vec<usize> = background.placements().iter().map(|p| p.core).collect();
    assert_eq!(lat, vec![0, 1]);
    assert_eq!(bg, vec![2, 3]);
    assert!(lat.iter().all(|c| !bg.contains(c)));
}

#[test]
fn partition_falls_back_on_homogeneous_machine() {
    // All P-cores: background has no E-cores and spills onto the P-cores.
    let topo = TopologyDescriptor::uniform(4);
    let (latency, background) = plan_partitioned(&topo, 2, 2);
    assert!(latency
        .placements()
        .iter()
        .all(|p| p.class == CoreClass::Performance));
    assert!(background
        .placements()
        .iter()
        .all(|p| p.class == CoreClass::Performance));
}

#[test]
fn workload_class_preferences() {
    assert_eq!(
        WorkloadClass::LatencySensitive.core_class_preference(),
        [CoreClass::Performance, CoreClass::Efficiency]
    );
    assert_eq!(
        WorkloadClass::Throughput.core_class_preference(),
        [CoreClass::Performance, CoreClass::Efficiency]
    );
    assert_eq!(
        WorkloadClass::Background.core_class_preference(),
        [CoreClass::Efficiency, CoreClass::Performance]
    );
}

// ---------------------------------------------------------------------------
// Topology descriptor
// ---------------------------------------------------------------------------

#[test]
fn uniform_descriptor_is_single_node_all_performance() {
    let topo = TopologyDescriptor::uniform(3);
    assert_eq!(topo.core_count(), 3);
    assert_eq!(topo.node_count(), 1);
    assert_eq!(topo.performance_core_count(), 3);
    assert_eq!(topo.efficiency_core_count(), 0);
    assert!(!topo.is_hybrid());
    assert_eq!(topo.distances().node_count(), 1);
    assert!(topo.caches().caches().is_empty());
}

#[test]
fn descriptor_accessors_and_hybrid() {
    let topo = hybrid_2node();
    assert_eq!(topo.core_count(), 4);
    assert_eq!(topo.node_count(), 2);
    assert_eq!(topo.performance_core_count(), 2);
    assert_eq!(topo.efficiency_core_count(), 2);
    assert!(topo.is_hybrid());
    assert_eq!(topo.node_of_core(3), Some(node(1)));
    assert_eq!(topo.node_of_core(99), None);
    assert_eq!(topo.cores_on_node(node(0)).count(), 2);
    assert_eq!(topo.cores_of_class(CoreClass::Efficiency).count(), 2);
}

#[test]
fn descriptor_new_rejects_bad_input() {
    assert_eq!(
        TopologyDescriptor::new(
            vec![],
            NumaDistanceMatrix::single(),
            CacheTopology::empty()
        )
        .unwrap_err(),
        TopologyError::NoCores
    );
    assert_eq!(
        TopologyDescriptor::new(
            vec![pcore(0, 0), pcore(0, 0)],
            NumaDistanceMatrix::single(),
            CacheTopology::empty()
        )
        .unwrap_err(),
        TopologyError::DuplicateCoreId { id: 0 }
    );
    // Cores span 2 nodes but matrix describes only 1.
    assert_eq!(
        TopologyDescriptor::new(
            vec![pcore(0, 0), pcore(1, 1)],
            NumaDistanceMatrix::single(),
            CacheTopology::empty()
        )
        .unwrap_err(),
        TopologyError::DistanceNodeMismatch {
            cores_nodes: 2,
            matrix_nodes: 1
        }
    );
    // Cache references a core id that does not exist.
    let caches = CacheTopology::new(vec![CacheInfo {
        level: CacheLevel::L3,
        size_bytes: 1024,
        line_bytes: 64,
        cores: vec![0, 7],
    }]);
    assert_eq!(
        TopologyDescriptor::new(vec![pcore(0, 0)], NumaDistanceMatrix::single(), caches)
            .unwrap_err(),
        TopologyError::CacheUnknownCore { id: 7 }
    );
}

#[test]
fn descriptor_from_numa_topology_bridges() {
    let numa = Topology::from_cores(vec![pcore(0, 0), ecore(1, 1)]);
    let topo = TopologyDescriptor::from_numa_topology(&numa, 25);
    assert_eq!(topo.node_count(), 2);
    assert_eq!(topo.distances().distance(node(0), node(1)), 25);
    assert_eq!(topo.distances().distance(node(0), node(0)), LOCAL_DISTANCE);
    assert!(topo.caches().caches().is_empty());
    assert!(topo.is_hybrid());

    // Single-node source -> single-node distance matrix.
    let one = Topology::uniform(4);
    let d = TopologyDescriptor::from_numa_topology(&one, 25);
    assert_eq!(d.node_count(), 1);
    assert_eq!(d.distances().node_count(), 1);
}

// ---------------------------------------------------------------------------
// Cache hierarchy
// ---------------------------------------------------------------------------

fn cache_topo() -> CacheTopology {
    CacheTopology::new(vec![
        CacheInfo {
            level: CacheLevel::L2,
            size_bytes: 512 * 1024,
            line_bytes: 64,
            cores: vec![1, 0], // unsorted on purpose; new() sorts.
        },
        CacheInfo {
            level: CacheLevel::L2,
            size_bytes: 512 * 1024,
            line_bytes: 64,
            cores: vec![2, 3],
        },
        CacheInfo {
            level: CacheLevel::L3,
            size_bytes: 8 * 1024 * 1024,
            line_bytes: 64,
            cores: vec![0, 1, 2, 3],
        },
    ])
}

#[test]
fn empty_cache_shares_nothing() {
    let c = CacheTopology::empty();
    assert_eq!(c.innermost_shared_level(0, 1), None);
    assert!(!c.shares_cache(CacheLevel::L2, 0, 1));
    assert!(c.sharers(CacheLevel::L3, 0).is_empty());
    assert_eq!(c.line_bytes(CacheLevel::L3), None);
}

#[test]
fn innermost_shared_level_picks_closest() {
    let c = cache_topo();
    // 0 and 1 share L2 (and L3) -> innermost is L2.
    assert_eq!(c.innermost_shared_level(0, 1), Some(CacheLevel::L2));
    // 0 and 2 share only L3.
    assert_eq!(c.innermost_shared_level(0, 2), Some(CacheLevel::L3));
    // A core with itself reports its innermost known level.
    assert_eq!(c.innermost_shared_level(0, 0), Some(CacheLevel::L2));
    // A core not in any cache -> None.
    assert_eq!(c.innermost_shared_level(0, 9), None);
}

#[test]
fn cache_sharers_and_shares() {
    let c = cache_topo();
    assert!(c.shares_cache(CacheLevel::L2, 0, 1));
    assert!(!c.shares_cache(CacheLevel::L2, 0, 2));
    assert!(c.shares_cache(CacheLevel::L3, 0, 2));
    // sharers returns the sorted instance membership (including the core).
    assert_eq!(c.sharers(CacheLevel::L2, 0), vec![0, 1]);
    assert_eq!(c.sharers(CacheLevel::L3, 2), vec![0, 1, 2, 3]);
    assert_eq!(c.line_bytes(CacheLevel::L3), Some(64));
}

#[test]
fn cache_level_ordering() {
    assert!(CacheLevel::L1 < CacheLevel::L2);
    assert!(CacheLevel::L2 < CacheLevel::L3);
}

#[test]
fn shared_l3_helper_builds_sorted_instance() {
    let cores = vec![pcore(3, 0), pcore(1, 0), pcore(2, 0)];
    let info = shared_l3(&cores, 8 * 1024 * 1024, 64);
    assert_eq!(info.level, CacheLevel::L3);
    assert_eq!(info.cores, vec![1, 2, 3]);
    assert!(info.contains(2));
    assert!(!info.contains(9));
}
