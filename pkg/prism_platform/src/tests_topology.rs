//! Tests for the §24.3 hybrid-core / NUMA / cache topology data model, the
//! `QoS` lane mapping, and the power/thermal policy. All oracles are hand-
//! computed against explicitly constructed inputs so none of this needs a live
//! machine.

use crate::platform::Os;
use crate::topology::power::{PowerPolicy, PowerSource, PowerState, ThermalState};
use crate::topology::qos::{qos_hint, EngineQos};
use crate::topology::{CoreKind, CpuTopology, TopologyBuilder, TopologyCore, TopologyError};

/// An explicit 2P+2E, two-NUMA-node machine with SMT on the P cores:
/// - logical 0,1 -> physical 0 (P, node 0, llc 0)  [SMT siblings]
/// - logical 2   -> physical 1 (P, node 0, llc 0)
/// - logical 3   -> physical 2 (E, node 1, llc 1)
/// - logical 4   -> physical 3 (E, node 1, llc 1)
fn hybrid_machine() -> CpuTopology {
    TopologyBuilder::new()
        .with_cores([
            TopologyCore::new(0, 0, 0, 0, CoreKind::Performance),
            TopologyCore::new(1, 0, 0, 0, CoreKind::Performance),
            TopologyCore::new(2, 1, 0, 0, CoreKind::Performance),
            TopologyCore::new(3, 2, 1, 1, CoreKind::Efficiency),
            TopologyCore::new(4, 3, 1, 1, CoreKind::Efficiency),
        ])
        .build()
        .expect("valid hybrid topology")
}

#[test]
fn builder_counts_logical_physical_and_numa() {
    let t = hybrid_machine();
    assert_eq!(t.logical_core_count(), 5);
    // physical ids are {0,1,2,3} -> 4 physical cores (logical 0,1 collapse).
    assert_eq!(t.physical_core_count(), 4);
    assert_eq!(t.numa_node_count(), 2);
    assert!(t.is_probed());
}

#[test]
fn hybrid_detection_and_class_filters() {
    let t = hybrid_machine();
    assert!(t.is_hybrid());
    assert_eq!(t.cores_of_kind(CoreKind::Performance).count(), 3);
    assert_eq!(t.cores_of_kind(CoreKind::Efficiency).count(), 2);
    assert_eq!(t.cores_on_node(1).count(), 2);
    assert_eq!(t.cores_on_node(0).count(), 3);
}

#[test]
fn smt_sibling_detection() {
    let t = hybrid_machine();
    // logical 0 and 1 share physical 0 -> siblings.
    assert!(t.has_smt_sibling(0));
    assert!(t.has_smt_sibling(1));
    // logical 2,3,4 are each alone on their physical core.
    assert!(!t.has_smt_sibling(2));
    assert!(!t.has_smt_sibling(3));
    // unknown id -> false.
    assert!(!t.has_smt_sibling(99));
}

#[test]
fn builder_rejects_empty_and_duplicate() {
    assert_eq!(TopologyBuilder::new().build(), Err(TopologyError::NoCores));
    let dup = TopologyBuilder::new()
        .core(TopologyCore::new(5, 0, 0, 0, CoreKind::Unknown))
        .core(TopologyCore::new(5, 1, 0, 0, CoreKind::Unknown))
        .build();
    assert_eq!(dup, Err(TopologyError::DuplicateLogicalId { id: 5 }));
}

#[test]
fn detect_is_honest_about_being_unprobed() {
    let t = CpuTopology::detect();
    // At least one core, single flat node, all Unknown class, not probed.
    assert!(t.logical_core_count() >= 1);
    assert_eq!(t.numa_node_count(), 1);
    assert!(!t.is_probed());
    assert!(!t.is_hybrid());
    assert!(t.cores().iter().all(|c| c.kind == CoreKind::Unknown));
    // physical == logical count in the fallback (no SMT collapsing).
    assert_eq!(t.physical_core_count(), t.logical_core_count());
}

#[test]
fn engine_qos_importance_is_monotonic() {
    // ALL is ordered most -> least important; importance must be strictly
    // decreasing.
    let imps: Vec<u8> = EngineQos::ALL.iter().map(|q| q.importance()).collect();
    assert_eq!(imps, [100, 80, 60, 30, 10]);
    for w in imps.windows(2) {
        assert!(w[0] > w[1]);
    }
}

#[test]
fn qos_prefers_efficiency_for_low_lanes_only() {
    assert!(!EngineQos::Critical.prefers_efficiency_core());
    assert!(!EngineQos::Interactive.prefers_efficiency_core());
    assert!(!EngineQos::Default.prefers_efficiency_core());
    assert!(EngineQos::Utility.prefers_efficiency_core());
    assert!(EngineQos::Background.prefers_efficiency_core());
}

#[test]
fn qos_hint_maps_per_platform() {
    // Apple: critical+interactive both collapse to USER_INTERACTIVE.
    assert_eq!(
        qos_hint(Os::Apple, EngineQos::Critical).class,
        "QOS_CLASS_USER_INTERACTIVE"
    );
    assert_eq!(
        qos_hint(Os::Apple, EngineQos::Background).class,
        "QOS_CLASS_BACKGROUND"
    );
    // Windows ladder.
    assert_eq!(qos_hint(Os::Windows, EngineQos::Default).class, "WIN_QOS_MEDIUM");
    assert_eq!(qos_hint(Os::Windows, EngineQos::Background).class, "WIN_QOS_ECO");
    // Linux nice levels.
    assert_eq!(qos_hint(Os::Linux, EngineQos::Critical).class, "NICE_-15");
    assert_eq!(qos_hint(Os::Android, EngineQos::Background).class, "NICE_19");
    // Unsupported platforms are honest.
    assert_eq!(qos_hint(Os::Web, EngineQos::Default).class, "QOS_UNSUPPORTED");
    assert_eq!(qos_hint(Os::Unknown, EngineQos::Critical).class, "QOS_UNSUPPORTED");
    // importance passes through from the lane; only low lanes are throttlable.
    assert_eq!(qos_hint(Os::Apple, EngineQos::Critical).importance, 100);
    assert!(!qos_hint(Os::Apple, EngineQos::Critical).throttlable);
    assert!(qos_hint(Os::Apple, EngineQos::Background).throttlable);
}

#[test]
fn power_policy_nominal_ac_is_unconstrained() {
    let p = PowerState::plugged_in().policy();
    assert_eq!(
        p,
        PowerPolicy {
            background_load_scale: 100,
            prefer_efficiency_cores: false,
            frame_rate_cap: None,
        }
    );
}

#[test]
fn power_policy_sheds_load_under_serious_thermal() {
    let state = PowerState {
        source: PowerSource::Battery,
        thermal: ThermalState::Serious,
        battery_percent: Some(70),
        low_power_mode: false,
    };
    let p = state.policy();
    assert_eq!(p.background_load_scale, 30);
    assert!(p.prefer_efficiency_cores);
    assert_eq!(p.frame_rate_cap, Some(30));
}

#[test]
fn power_policy_pauses_background_when_critical() {
    let state = PowerState {
        source: PowerSource::Battery,
        thermal: ThermalState::Critical,
        battery_percent: Some(55),
        low_power_mode: false,
    };
    let p = state.policy();
    assert_eq!(p.background_load_scale, 0);
    assert_eq!(p.frame_rate_cap, Some(30));
}

#[test]
fn power_policy_low_power_mode_caps_frame_rate() {
    let state = PowerState {
        source: PowerSource::Battery,
        thermal: ThermalState::Nominal,
        battery_percent: Some(90),
        low_power_mode: true,
    };
    let p = state.policy();
    // Low-power mode constrains even with a healthy battery and no heat.
    assert_eq!(p.background_load_scale, 60);
    assert!(p.prefer_efficiency_cores);
    assert_eq!(p.frame_rate_cap, Some(30));
}

#[test]
fn battery_threshold_helpers() {
    let unknown = PowerState::default();
    assert!(!unknown.battery_at_or_below(50));
    let low = PowerState {
        battery_percent: Some(15),
        ..PowerState::default()
    };
    assert!(low.battery_at_or_below(20));
    assert!(!low.battery_at_or_below(10));
    // low battery alone triggers the constrained policy.
    let p = low.policy();
    assert!(p.prefer_efficiency_cores);
    assert_eq!(p.frame_rate_cap, Some(30));
}
