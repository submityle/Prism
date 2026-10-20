//! Regression coverage for hard dependency ordering and optional soft references.

use prism_asset::StableGuid;
use prism_asset_container::{
    ContainerIdentity, ManifestError, ManifestMetadata, ManifestRoute, RuntimeManifest,
};

const CONTAINER: &str = "assets";
// v3 header, one container record/name, then the first route record/name.
const FIRST_METADATA_OFFSET: usize = 20 + 50 + CONTAINER.len() + 38 + CONTAINER.len();
const FIRST_DEPENDENCY_OFFSET: usize = FIRST_METADATA_OFFSET + 24;

type RouteSpec<'a> = (u128, &'a [u128], &'a [u128]);

fn metadata(hard: &[u128], optional: &[u128]) -> ManifestMetadata {
    ManifestMetadata {
        hard_dependencies: hard.iter().copied().map(StableGuid::from_u128).collect(),
        optional_dependencies: optional
            .iter()
            .copied()
            .map(StableGuid::from_u128)
            .collect(),
        ..ManifestMetadata::default()
    }
}

fn route(id: u128) -> ManifestRoute {
    ManifestRoute {
        guid: StableGuid::from_u128(id),
        container_name: CONTAINER.into(),
        priority: 0,
        container_guid: StableGuid::from_u128(id),
    }
}

fn manifest(routes: &[RouteSpec<'_>]) -> RuntimeManifest {
    let mut manifest = RuntimeManifest::new();
    manifest
        .bind_container(
            CONTAINER,
            ContainerIdentity {
                byte_len: 1,
                content_hash: [1; 32],
                generation: 1,
            },
        )
        .unwrap();
    for &(id, hard, optional) in routes {
        manifest
            .insert_with_metadata(route(id), metadata(hard, optional))
            .unwrap();
    }
    manifest
}

fn assert_round_trip(manifest: &RuntimeManifest) {
    let bytes = manifest.encode().unwrap();
    let decoded = RuntimeManifest::decode(&bytes).unwrap();
    assert_eq!(&decoded, manifest);
    assert_eq!(decoded.encode().unwrap(), bytes);
}

fn rehash(bytes: &mut [u8]) {
    let body_end = bytes.len() - 32;
    let digest = *blake3::hash(&bytes[..body_end]).as_bytes();
    bytes[body_end..].copy_from_slice(&digest);
}

#[test]
fn missing_optional_route_round_trips() {
    let manifest = manifest(&[(1, &[], &[99])]);
    assert!(manifest.route(StableGuid::from_u128(99)).is_none());
    assert_round_trip(&manifest);
}

#[test]
fn optional_self_reference_round_trips() {
    assert_round_trip(&manifest(&[(1, &[], &[1])]));
}

#[test]
fn optional_cycle_round_trips() {
    assert_round_trip(&manifest(&[(1, &[], &[2]), (2, &[], &[1])]));
}

#[test]
fn optional_edges_do_not_close_hard_dependency_cycles() {
    assert_round_trip(&manifest(&[(1, &[2], &[]), (2, &[], &[1])]));
}

#[test]
fn encode_rejects_missing_self_and_cyclic_hard_dependencies() {
    let cases: &[&[RouteSpec<'_>]] = &[
        &[(1, &[99], &[])],
        &[(1, &[1], &[])],
        &[(1, &[2], &[]), (2, &[1], &[])],
        &[(1, &[2], &[]), (2, &[3], &[]), (3, &[1], &[])],
    ];
    for routes in cases {
        assert_eq!(
            manifest(routes).encode(),
            Err(ManifestError::InvalidFormat),
            "hard dependency graph: {routes:?}",
        );
    }
}

#[test]
fn decode_rejects_missing_self_and_cyclic_hard_dependencies() {
    // Promote the first route's sole optional edge to a hard edge in otherwise
    // valid, correctly hashed bytes. The dependency GUID and byte length stay put.
    let cases: &[&[RouteSpec<'_>]] = &[
        &[(1, &[], &[99])],
        &[(1, &[], &[1])],
        &[(1, &[], &[2]), (2, &[1], &[])],
        &[(1, &[], &[2]), (2, &[3], &[]), (3, &[1], &[])],
    ];
    for routes in cases {
        let mut bytes = manifest(routes).encode().unwrap();
        bytes[FIRST_METADATA_OFFSET..FIRST_METADATA_OFFSET + 4]
            .copy_from_slice(&1_u32.to_le_bytes());
        bytes[FIRST_METADATA_OFFSET + 4..FIRST_METADATA_OFFSET + 8]
            .copy_from_slice(&0_u32.to_le_bytes());
        rehash(&mut bytes);
        assert_eq!(
            RuntimeManifest::decode(&bytes),
            Err(ManifestError::InvalidFormat),
            "hard dependency graph: {routes:?}",
        );
    }
}

#[test]
fn metadata_mutations_reject_nil_duplicates_and_overlapping_lists() {
    let cases: &[(&[u128], &[u128])] = &[
        (&[0], &[]),
        (&[], &[0]),
        (&[2, 2], &[]),
        (&[], &[2, 2]),
        (&[2], &[2]),
    ];
    for &(hard, optional) in cases {
        let invalid = metadata(hard, optional);
        let mut manifest = manifest(&[(1, &[], &[])]);
        let original = manifest.clone();
        assert_eq!(
            manifest.insert_with_metadata(route(3), invalid.clone()),
            Err(ManifestError::InvalidFormat),
        );
        assert_eq!(
            manifest.set_metadata(StableGuid::from_u128(1), invalid),
            Err(ManifestError::InvalidFormat),
        );
        assert_eq!(manifest, original);
        assert_round_trip(&manifest);
    }
}

#[test]
fn decode_rejects_nil_duplicates_and_overlapping_lists() {
    let manifest = manifest(&[
        (1, &[2, 3], &[4, 5]),
        (2, &[], &[]),
        (3, &[], &[]),
        (4, &[], &[]),
        (5, &[], &[]),
    ]);
    assert_round_trip(&manifest);
    // Dependency slots 0..2 are hard and 2..4 are optional. All non-nil
    // replacement GUIDs have routes, isolating canonical-list validation.
    for (slot, replacement) in [(0, 0_u128), (2, 0), (1, 2), (3, 4), (2, 2)] {
        let mut bytes = manifest.encode().unwrap();
        let offset = FIRST_DEPENDENCY_OFFSET + slot * 16;
        bytes[offset..offset + 16].copy_from_slice(&replacement.to_le_bytes());
        rehash(&mut bytes);
        assert_eq!(
            RuntimeManifest::decode(&bytes),
            Err(ManifestError::InvalidFormat),
            "dependency slot {slot}, replacement {replacement}",
        );
    }
}

#[test]
fn patch_can_remove_an_optional_target_but_not_a_hard_target() {
    let removed = [StableGuid::from_u128(2)];
    let mut optional = manifest(&[(1, &[], &[2]), (2, &[], &[])]);
    optional
        .apply_patch(&RuntimeManifest::new(), &removed)
        .unwrap();
    assert!(optional.route(removed[0]).is_none());
    assert_round_trip(&optional);

    let mut hard = manifest(&[(1, &[2], &[]), (2, &[], &[])]);
    let original = hard.clone();
    assert_eq!(
        hard.apply_patch(&RuntimeManifest::new(), &removed),
        Err(ManifestError::InvalidFormat),
    );
    assert_eq!(hard, original);
}
