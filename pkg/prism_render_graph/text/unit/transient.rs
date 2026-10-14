use super::*;

fn region(offset: u64, size: u64, alignment: u64) -> Option<TransientRegion> {
    Some(TransientRegion {
        offset,
        size,
        alignment,
    })
}

#[test]
fn validate_alias_safety_accepts_disjoint_lifetimes_in_a_group() {
    // Two resources aliased to offset 0, lifetimes [0,1] and [2,3]: disjoint.
    let plan = TransientAllocation {
        heap_bytes: 1024,
        regions: vec![region(0, 1024, 256), region(0, 1024, 256)],
        lifetimes: vec![Some(0..2), Some(2..4)],
        alias_groups: vec![vec![ResourceId(0), ResourceId(1)]],
        individual_bytes: 2048,
    };
    assert_eq!(plan.validate_alias_safety(), Ok(()));
}

#[test]
fn validate_alias_safety_flags_overlapping_lifetimes_in_a_group() {
    // Two resources sharing offset 0 whose lifetimes [0,2] and [1,3] overlap
    // at positions 1..2 — an unsound alias the validator must reject.
    let plan = TransientAllocation {
        heap_bytes: 1024,
        regions: vec![region(0, 1024, 256), region(0, 1024, 256)],
        lifetimes: vec![Some(0..3), Some(1..4)],
        alias_groups: vec![vec![ResourceId(0), ResourceId(1)]],
        individual_bytes: 2048,
    };
    assert_eq!(
        plan.validate_alias_safety(),
        Err(AliasOverlap {
            offset: 0,
            first: ResourceId(0),
            second: ResourceId(1),
        })
    );
}

#[test]
fn touching_lifetimes_are_disjoint() {
    // [0,0] then [1,1]: last_a (0) < first_b (1), so half-open 0..1 and 1..2
    // do not overlap — the boundary case the first-fit reuse rule allows.
    assert!(!ranges_overlap(&(0..1), &(1..2)));
    assert!(ranges_overlap(&(0..2), &(1..3)));
}

#[test]
fn zero_sized_transients_never_alias_even_when_lifetimes_overlap() {
    // Topology-only graphs model transients with size 0 (byte budgets are
    // resolved later at runtime). Every such resource lands on offset 0, but
    // an empty region owns no memory and cannot corrupt a peer, so two
    // overlapping zero-byte resources must not be grouped or flagged.
    let regions = vec![region(0, 0, 16), region(0, 0, 16)];
    assert!(group_by_offset(&regions).is_empty());

    let plan = TransientAllocation {
        heap_bytes: 0,
        regions,
        lifetimes: vec![Some(0..5), Some(0..5)],
        alias_groups: group_by_offset(&[region(0, 0, 16), region(0, 0, 16)]),
        individual_bytes: 0,
    };
    assert_eq!(plan.validate_alias_safety(), Ok(()));
}

#[test]
fn peak_live_bytes_tracks_simultaneous_demand() {
    // r0 live [0,2] (1024), r1 live [1,3] (512): overlap at 1..3 => 1536.
    let plan = TransientAllocation {
        heap_bytes: 1536,
        regions: vec![region(0, 1024, 256), region(1024, 512, 256)],
        lifetimes: vec![Some(0..3), Some(1..4)],
        alias_groups: vec![vec![ResourceId(0)], vec![ResourceId(1)]],
        individual_bytes: 1536,
    };
    assert_eq!(plan.peak_live_bytes(), 1536);
    assert_eq!(plan.savings_bytes(), 0);
}
