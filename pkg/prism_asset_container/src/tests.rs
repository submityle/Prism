use prism_asset::{AssetReader, StableGuid};

use crate::{
    ChunkInput, ChunkPlanner, ChunkPolicy, ContainerError, ContainerReader, ContainerWriter,
    ManifestAssetReader, ManifestError, ManifestReadError, ManifestRoute, OverlaySet,
    RuntimeManifest, FORMAT_VERSION, MAGIC,
};

fn guid(name: &str) -> StableGuid {
    StableGuid::from_path(name)
}

fn build(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut writer = ContainerWriter::new();
    for (name, bytes) in entries {
        writer.add(guid(name), bytes.to_vec()).unwrap();
    }
    writer.finish().unwrap()
}

fn update_directory_hash(bytes: &mut [u8]) {
    let directory_start = u64::from_le_bytes(bytes[20..28].try_into().unwrap()) as usize;
    let directory_len = u64::from_le_bytes(bytes[28..36].try_into().unwrap()) as usize;
    let hash = blake3::hash(&bytes[directory_start..directory_start + directory_len]);
    bytes[52..84].copy_from_slice(hash.as_bytes());
}

#[test]
fn writer_is_deterministic_and_canonical_across_insertion_orders() {
    let mut first = ContainerWriter::new();
    first.add(guid("z.asset"), b"zed".to_vec()).unwrap();
    first.add(guid("a.asset"), b"alpha".to_vec()).unwrap();
    let mut second = ContainerWriter::new();
    second.add(guid("a.asset"), b"alpha".to_vec()).unwrap();
    second.add(guid("z.asset"), b"zed".to_vec()).unwrap();
    let first = first.finish().unwrap();
    let second = second.finish().unwrap();
    assert_eq!(first, second);
    assert_eq!(&first[..8], &MAGIC);
    assert_eq!(
        u16::from_le_bytes(first[8..10].try_into().unwrap()),
        FORMAT_VERSION
    );
    let reader = ContainerReader::new(first).unwrap();
    assert_eq!(reader.info().entry_count, 2);
    assert_eq!(reader.read_all(guid("a.asset")).unwrap(), b"alpha");
    assert_eq!(reader.read_all(guid("z.asset")).unwrap(), b"zed");
}

#[test]
fn writer_rejects_duplicate_guid_without_replacing_payload() {
    let id = guid("same.asset");
    let mut writer = ContainerWriter::new();
    writer.add(id, b"first".to_vec()).unwrap();
    assert_eq!(
        writer.add(id, b"second".to_vec()),
        Err(ContainerError::DuplicateGuid(id))
    );
    let reader = ContainerReader::new(writer.finish().unwrap()).unwrap();
    assert_eq!(reader.read_all(id).unwrap(), b"first");
}

#[test]
fn reader_handles_empty_payload_and_checks_ranges_without_overflow() {
    let bytes = build(&[("empty", b""), ("data", b"1234")]);
    let reader = ContainerReader::new(bytes).unwrap();
    assert_eq!(reader.read_all(guid("empty")).unwrap(), b"");
    let mut range = [0; 2];
    reader.read_range(guid("data"), 1, &mut range).unwrap();
    assert_eq!(&range, b"23");
    assert_eq!(
        reader.read_range(guid("data"), u64::MAX, &mut range),
        Err(ContainerError::IntegerOverflow)
    );
    assert_eq!(
        reader.read_range(guid("data"), 3, &mut range),
        Err(ContainerError::InvalidRange)
    );
    assert_eq!(
        reader.read_range(guid("absent"), 0, &mut range),
        Err(ContainerError::MissingAsset(guid("absent")))
    );
}

#[test]
fn rejects_truncation_version_and_directory_digest_corruption() {
    let canonical = build(&[("one", b"one")]);
    assert_eq!(
        ContainerReader::new(canonical[..canonical.len() - 1].to_vec()).unwrap_err(),
        ContainerError::Truncated
    );

    let mut version = canonical.clone();
    version[8..10].copy_from_slice(&99_u16.to_le_bytes());
    assert_eq!(
        ContainerReader::new(version).unwrap_err(),
        ContainerError::UnsupportedVersion(99)
    );

    let mut directory = canonical;
    directory[96] ^= 1;
    assert_eq!(
        ContainerReader::new(directory).unwrap_err(),
        ContainerError::HashMismatch
    );
}

#[test]
fn rejects_payload_hash_mismatch_and_does_not_modify_output() {
    let mut bytes = build(&[("payload", b"hello")]);
    let payload_offset = u64::from_le_bytes(bytes[36..44].try_into().unwrap()) as usize;
    bytes[payload_offset] ^= 0x20;
    assert_eq!(
        ContainerReader::new(bytes).unwrap_err(),
        ContainerError::HashMismatch
    );
}

#[test]
fn rejects_noncanonical_or_invalid_toc_records() {
    // Reserved record bytes must be zero, even when the directory digest matches.
    let mut reserved = build(&[("item", b"x")]);
    reserved[96 + 76..96 + 80].copy_from_slice(&3_u32.to_le_bytes());
    update_directory_hash(&mut reserved);
    assert_eq!(
        ContainerReader::new(reserved).unwrap_err(),
        ContainerError::InvalidAlignment
    );

    // A payload offset that creates a gap is not accepted as a valid TOC.
    let mut gap = build(&[("item", b"x")]);
    let old = u64::from_le_bytes(gap[96 + 16..96 + 24].try_into().unwrap());
    gap[96 + 16..96 + 24].copy_from_slice(&(old + 1).to_le_bytes());
    update_directory_hash(&mut gap);
    assert_eq!(
        ContainerReader::new(gap).unwrap_err(),
        ContainerError::OutOfBounds
    );
}

#[test]
fn aligned_payloads_preserve_absolute_offsets_and_zero_padding() {
    let mut writer = ContainerWriter::new();
    writer
        .add_aligned(guid("unaligned"), b"a".to_vec(), 1)
        .unwrap();
    writer
        .add_aligned(guid("aligned"), b"payload".to_vec(), 4096)
        .unwrap();
    let bytes = writer.finish().unwrap();
    let reader = ContainerReader::new(bytes).unwrap();
    let entry = reader.entry(guid("aligned")).unwrap();
    assert_eq!(entry.alignment, 4096);
    assert_eq!(entry.offset % 4096, 0);
    assert_eq!(reader.read_all(guid("aligned")).unwrap(), b"payload");
}

#[test]
fn planned_layout_is_consumed_and_round_trips_in_version_two() {
    let first = guid("planned-first");
    let second = guid("planned-second");
    let policy = ChunkPolicy {
        max_chunk_size: 16,
        chunk_alignment: 8,
        max_chunks: 2,
    };
    let plan = ChunkPlanner::new(policy)
        .plan([
            ChunkInput {
                guid: first,
                size: 3,
                alignment: 1,
                dependencies: vec![],
                priority: 0,
            },
            ChunkInput {
                guid: second,
                size: 4,
                alignment: 4,
                dependencies: vec![],
                priority: 9,
            },
        ])
        .unwrap();
    assert_eq!(plan.chunks[0].placements[0].guid, second);

    let mut writer = ContainerWriter::new();
    writer.add_aligned(first, b"one".to_vec(), 1).unwrap();
    writer.add_aligned(second, b"four".to_vec(), 4).unwrap();
    let bytes = writer
        .finish_with_chunk_plan(&plan, policy.chunk_alignment)
        .unwrap();
    assert_eq!(u16::from_le_bytes(bytes[8..10].try_into().unwrap()), 2);
    let reader = ContainerReader::new(bytes).unwrap();
    let first_entry = reader.entry(first).unwrap();
    let second_entry = reader.entry(second).unwrap();
    assert_eq!(first_entry.offset % 1, 0);
    assert_eq!(second_entry.offset % 4, 0);
    assert_eq!(first_entry.offset, second_entry.offset + 8);
    assert_eq!(reader.read_all(first).unwrap(), b"one");
    assert_eq!(reader.read_all(second).unwrap(), b"four");
}

#[test]
fn planned_layout_honors_payload_alignment_above_chunk_alignment() {
    let id = guid("high-alignment");
    let policy = ChunkPolicy {
        max_chunk_size: 16 * 1024,
        chunk_alignment: 4096,
        max_chunks: 1,
    };
    let plan = ChunkPlanner::new(policy)
        .plan([ChunkInput {
            guid: id,
            size: 3,
            alignment: 8192,
            dependencies: vec![],
            priority: 0,
        }])
        .unwrap();
    assert_eq!(plan.chunks[0].start_alignment, 8192);

    let mut writer = ContainerWriter::new();
    writer.add_aligned(id, b"abc".to_vec(), 8192).unwrap();
    let reader = ContainerReader::new(
        writer
            .finish_with_chunk_plan(&plan, policy.chunk_alignment)
            .unwrap(),
    )
    .unwrap();
    let entry = reader.entry(id).unwrap();
    assert_eq!(entry.offset % 8192, 0);
    assert_eq!(reader.read_all(id).unwrap(), b"abc");
}

#[test]
fn planned_layout_keeps_cross_chunk_dependencies_before_dependents() {
    let root = StableGuid::from_u128(1);
    let dependency = StableGuid::from_u128(2);
    let dependent = StableGuid::from_u128(3);
    let policy = ChunkPolicy {
        max_chunk_size: 8,
        chunk_alignment: 1,
        max_chunks: 2,
    };
    let plan = ChunkPlanner::new(policy)
        .plan([
            ChunkInput {
                guid: root,
                size: 5,
                alignment: 1,
                dependencies: vec![],
                priority: 0,
            },
            ChunkInput {
                guid: dependency,
                size: 4,
                alignment: 4,
                dependencies: vec![root],
                priority: 0,
            },
            ChunkInput {
                guid: dependent,
                size: 1,
                alignment: 1,
                dependencies: vec![dependency],
                priority: 0,
            },
        ])
        .unwrap();

    assert_eq!(plan.assignments[&root], 0);
    assert_eq!(plan.assignments[&dependency], 1);
    assert_eq!(plan.assignments[&dependent], 1);
    assert_eq!(
        plan.chunks[1]
            .placements
            .iter()
            .map(|placement| placement.guid)
            .collect::<Vec<_>>(),
        vec![dependency, dependent]
    );
    assert!(
        plan.chunks[plan.assignments[&dependency]].offset
            <= plan.chunks[plan.assignments[&dependent]].offset
    );
}

#[test]
fn planned_writer_rejects_a_plan_that_does_not_cover_exactly_its_payloads() {
    let id = guid("planned-only");
    let plan = ChunkPlanner::new(ChunkPolicy {
        max_chunk_size: 8,
        chunk_alignment: 4,
        max_chunks: 1,
    })
    .plan([ChunkInput {
        guid: id,
        size: 1,
        alignment: 1,
        dependencies: vec![],
        priority: 0,
    }])
    .unwrap();
    let mut writer = ContainerWriter::new();
    writer.add(id, b"different-size".to_vec()).unwrap();
    assert_eq!(
        writer.finish_with_chunk_plan(&plan, 4),
        Err(ContainerError::InvalidDirectory)
    );
}

#[test]
fn planned_writer_rejects_noncanonical_placement_and_chunk_extents() {
    let id = guid("planned-canonical-extent");
    let policy = ChunkPolicy {
        max_chunk_size: 16,
        chunk_alignment: 8,
        max_chunks: 1,
    };
    let canonical = ChunkPlanner::new(policy)
        .plan([ChunkInput {
            guid: id,
            size: 1,
            alignment: 1,
            dependencies: vec![],
            priority: 0,
        }])
        .unwrap();

    // A forged plan must not be able to force arbitrary zero-filled extents
    // around one byte of actual payload.
    let mut oversized_extent = canonical.clone();
    oversized_extent.chunks[0].placements[0].offset = 1024;
    oversized_extent.chunks[0].raw_size = 1025;
    let mut writer = ContainerWriter::new();
    writer.add(id, b"x".to_vec()).unwrap();
    assert_eq!(
        writer
            .finish_with_chunk_plan(&oversized_extent, policy.chunk_alignment)
            .unwrap_err(),
        ContainerError::InvalidDirectory
    );

    // Even a small hole is noncanonical: placements are packed at the first
    // offset satisfying the effective alignment, and raw_size ends at the
    // final placement rather than including attacker-chosen trailing bytes.
    let mut noncanonical_hole = canonical;
    noncanonical_hole.chunks[0].placements[0].offset = 8;
    noncanonical_hole.chunks[0].raw_size = 9;
    let mut writer = ContainerWriter::new();
    writer.add(id, b"x".to_vec()).unwrap();
    assert_eq!(
        writer
            .finish_with_chunk_plan(&noncanonical_hole, policy.chunk_alignment)
            .unwrap_err(),
        ContainerError::InvalidDirectory
    );
}

#[test]
fn planned_writer_rejects_noncanonical_chunk_gaps() {
    let first = guid("planned-gap-first");
    let second = guid("planned-gap-second");
    let policy = ChunkPolicy {
        max_chunk_size: 1,
        chunk_alignment: 8,
        max_chunks: 2,
    };
    let mut plan = ChunkPlanner::new(policy)
        .plan([first, second].map(|guid| ChunkInput {
            guid,
            size: 1,
            alignment: 1,
            dependencies: vec![],
            priority: 0,
        }))
        .unwrap();
    plan.chunks[1].offset += 8;

    let mut writer = ContainerWriter::new();
    writer.add(first, b"a".to_vec()).unwrap();
    writer.add(second, b"b".to_vec()).unwrap();
    assert_eq!(
        writer
            .finish_with_chunk_plan(&plan, policy.chunk_alignment)
            .unwrap_err(),
        ContainerError::InvalidDirectory
    );
}

#[test]
fn version_two_reader_rejects_nonzero_directory_alignment_padding() {
    let id = guid("padding-check");
    let plan = ChunkPlanner::new(ChunkPolicy {
        max_chunk_size: 8,
        chunk_alignment: 64,
        max_chunks: 1,
    })
    .plan([ChunkInput {
        guid: id,
        size: 1,
        alignment: 1,
        dependencies: vec![],
        priority: 0,
    }])
    .unwrap();
    let mut writer = ContainerWriter::new();
    writer.add(id, b"x".to_vec()).unwrap();
    let mut bytes = writer.finish_with_chunk_plan(&plan, 64).unwrap();
    let payload_start = u64::from_le_bytes(bytes[36..44].try_into().unwrap()) as usize;
    assert!(payload_start > 176);
    bytes[176] = 1;
    assert_eq!(
        ContainerReader::new(bytes).unwrap_err(),
        ContainerError::InvalidHeader
    );
}

#[test]
fn overlay_priority_ties_latest_mount_wins_and_unmount_falls_back() {
    let id = guid("shared.asset");
    let mut overlays = OverlaySet::new();
    overlays.mount(
        "base",
        0,
        ContainerReader::new(build(&[("shared.asset", b"base")])).unwrap(),
    );
    overlays.mount(
        "patch",
        1,
        ContainerReader::new(build(&[("shared.asset", b"patch")])).unwrap(),
    );
    overlays.mount(
        "same-priority-later",
        1,
        ContainerReader::new(build(&[("shared.asset", b"latest")])).unwrap(),
    );
    assert_eq!(overlays.source(id), Some("same-priority-later"));
    assert_eq!(overlays.read_all(id).unwrap(), b"latest");
    assert!(overlays.unmount("same-priority-later"));
    assert_eq!(overlays.read_all(id).unwrap(), b"patch");
    assert!(overlays.unmount("patch"));
    assert_eq!(overlays.read_all(id).unwrap(), b"base");
    assert!(!overlays.unmount("missing"));
}

#[test]
fn remounting_a_container_name_replaces_its_previous_guid_set() {
    let old_guid = guid("remount-old");
    let new_guid = guid("remount-new");
    let mut overlays = OverlaySet::new();
    overlays.mount(
        "shipping",
        1,
        ContainerReader::new(build(&[("remount-old", b"old")])).unwrap(),
    );
    overlays.mount(
        "shipping",
        1,
        ContainerReader::new(build(&[("remount-new", b"new")])).unwrap(),
    );
    assert!(!overlays.contains(old_guid));
    assert_eq!(overlays.read_all(new_guid).unwrap(), b"new");

    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: old_guid,
            container_name: "shipping".into(),
            priority: 1,
            container_guid: old_guid,
        })
        .unwrap();
    assert!(matches!(
        manifest.bind(&overlays),
        Err(ManifestReadError::MissingContainerGuid { .. })
    ));
}

#[test]
fn runtime_manifest_round_trips_deterministically_and_rejects_bad_input() {
    let logical = guid("logical");
    let local = guid("container-local");
    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: logical,
            container_name: "patch.container".into(),
            priority: 7,
            container_guid: local,
        })
        .unwrap();
    let encoded = manifest.encode().unwrap();
    assert_eq!(RuntimeManifest::decode(&encoded).unwrap(), manifest);
    assert_eq!(manifest.encode().unwrap(), encoded);
    assert_eq!(manifest.route(logical).unwrap().container_guid, local);
    assert_eq!(
        RuntimeManifest::decode(&encoded[..encoded.len() - 1]),
        Err(ManifestError::Truncated)
    );

    let mut tampered = encoded.clone();
    tampered[HEADER_NAME_OFFSET] ^= 1;
    assert_eq!(
        RuntimeManifest::decode(&tampered),
        Err(ManifestError::HashMismatch)
    );

    let mut unknown = encoded;
    unknown[8..10].copy_from_slice(&99_u16.to_le_bytes());
    assert_eq!(
        RuntimeManifest::decode(&unknown),
        Err(ManifestError::UnsupportedVersion(99))
    );
}

const HEADER_NAME_OFFSET: usize = 16 + 38;

#[test]
fn runtime_manifest_rejects_duplicate_guid_records_even_with_valid_hash() {
    let mut manifest = RuntimeManifest::new();
    for name in ["a", "b"] {
        manifest
            .insert(ManifestRoute {
                guid: guid(name),
                container_name: name.into(),
                priority: 0,
                container_guid: guid(name),
            })
            .unwrap();
    }
    let mut bytes = manifest.encode().unwrap();
    let first_guid = bytes[16..32].to_vec();
    // Locate the second variable-size record and replace its logical GUID.
    let first_name_len = u16::from_le_bytes(bytes[52..54].try_into().unwrap()) as usize;
    let second_guid_at = 16 + 38 + first_name_len;
    bytes[second_guid_at..second_guid_at + 16].copy_from_slice(&first_guid);
    let body_end = bytes.len() - 32;
    let digest = blake3::hash(&bytes[..body_end]);
    bytes[body_end..].copy_from_slice(digest.as_bytes());
    assert!(matches!(
        RuntimeManifest::decode(&bytes),
        Err(ManifestError::DuplicateGuid(_))
    ));
}

#[test]
fn overlay_manifest_exports_the_priority_winner_snapshot() {
    let id = guid("overlay-index");
    let mut overlays = OverlaySet::new();
    overlays.mount(
        "low",
        -2,
        ContainerReader::new(build(&[("overlay-index", b"low")])).unwrap(),
    );
    overlays.mount(
        "high",
        4,
        ContainerReader::new(build(&[("overlay-index", b"high")])).unwrap(),
    );
    let manifest = RuntimeManifest::from_overlay(&overlays).unwrap();
    let route = manifest.route(id).unwrap();
    assert_eq!(route.container_name, "high");
    assert_eq!(route.priority, 4);
    assert_eq!(route.container_guid, id);
    assert_eq!(
        RuntimeManifest::decode(&manifest.encode().unwrap()).unwrap(),
        manifest
    );
}

#[test]
fn manifest_reader_routes_logical_guid_to_container_guid() {
    let logical = guid("logical.asset");
    let local = guid("container-local.asset");
    let mut overlays = OverlaySet::new();
    overlays.mount(
        "patch",
        7,
        ContainerReader::new(build(&[("container-local.asset", b"mapped bytes")])).unwrap(),
    );

    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: logical,
            container_name: "patch".into(),
            priority: 7,
            container_guid: local,
        })
        .unwrap();
    let reader = manifest.bind(&overlays).unwrap();
    assert_eq!(reader.read_all(logical).unwrap(), b"mapped bytes");
    let mut range = [0; 5];
    reader.read_range(logical, 7, &mut range).unwrap();
    assert_eq!(&range, b"bytes");
}

#[test]
fn manifest_reader_distinguishes_route_container_guid_and_range_errors() {
    let logical = guid("logical-errors.asset");
    let local = guid("local-errors.asset");
    let absent = guid("absent-errors.asset");

    let empty_manifest = RuntimeManifest::new();
    let overlays = OverlaySet::new();
    let reader = empty_manifest.bind(&overlays).unwrap();
    assert_eq!(
        reader.read_all(logical),
        Err(ManifestReadError::MissingRoute(logical))
    );

    let mut missing_container = RuntimeManifest::new();
    missing_container
        .insert(ManifestRoute {
            guid: logical,
            container_name: "not-mounted".into(),
            priority: 0,
            container_guid: local,
        })
        .unwrap();
    assert!(matches!(
        missing_container.bind(&overlays),
        Err(ManifestReadError::MissingContainer { .. })
    ));

    let mut overlays = OverlaySet::new();
    overlays.mount(
        "mounted",
        0,
        ContainerReader::new(build(&[("other.asset", b"other")])).unwrap(),
    );
    let mut missing_guid = RuntimeManifest::new();
    missing_guid
        .insert(ManifestRoute {
            guid: logical,
            container_name: "mounted".into(),
            priority: 0,
            container_guid: absent,
        })
        .unwrap();
    assert!(matches!(
        missing_guid.bind(&overlays),
        Err(ManifestReadError::MissingContainerGuid { .. })
    ));

    let mut valid = RuntimeManifest::new();
    valid
        .insert(ManifestRoute {
            guid: logical,
            container_name: "mounted".into(),
            priority: 0,
            container_guid: guid("other.asset"),
        })
        .unwrap();
    let reader = valid.bind(&overlays).unwrap();
    let mut output = [0xA5; 2];
    assert!(matches!(
        reader.read_range(logical, 4, &mut output),
        Err(ManifestReadError::InvalidRange { .. })
    ));
    assert_eq!(output, [0xA5; 2]);
}

#[test]
fn manifest_reader_rejects_priority_and_source_drift() {
    let logical = guid("logical-drift.asset");
    let local = guid("local-drift.asset");
    let mut overlays = OverlaySet::new();
    overlays.mount(
        "base",
        1,
        ContainerReader::new(build(&[("local-drift.asset", b"base")])).unwrap(),
    );

    let mut wrong_priority = RuntimeManifest::new();
    wrong_priority
        .insert(ManifestRoute {
            guid: logical,
            container_name: "base".into(),
            priority: 2,
            container_guid: local,
        })
        .unwrap();
    assert!(matches!(
        wrong_priority.bind(&overlays),
        Err(ManifestReadError::PriorityMismatch { .. })
    ));

    overlays.mount(
        "patch",
        2,
        ContainerReader::new(build(&[("local-drift.asset", b"patch")])).unwrap(),
    );
    let mut wrong_source = RuntimeManifest::new();
    wrong_source
        .insert(ManifestRoute {
            guid: logical,
            container_name: "base".into(),
            priority: 1,
            container_guid: local,
        })
        .unwrap();
    assert!(matches!(
        wrong_source.bind(&overlays),
        Err(ManifestReadError::SourceMismatch { .. })
    ));
}

#[test]
fn manifest_asset_reader_implements_vfs_path_and_subasset_contract() {
    let parent = guid("scene.asset");
    let child = StableGuid::derive_sub(parent, "Mesh0");
    let mut writer = ContainerWriter::new();
    writer.add(parent, b"scene bytes".to_vec()).unwrap();
    writer.add(child, b"mesh bytes".to_vec()).unwrap();
    let container = ContainerReader::new(writer.finish().unwrap()).unwrap();

    let mut overlays = OverlaySet::new();
    overlays.mount("shipping", 3, container);
    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid: parent,
            container_name: "shipping".into(),
            priority: 3,
            container_guid: parent,
        })
        .unwrap();
    manifest
        .insert(ManifestRoute {
            guid: child,
            container_name: "shipping".into(),
            priority: 3,
            container_guid: child,
        })
        .unwrap();

    let source = ManifestAssetReader::new(manifest, overlays);
    assert_eq!(source.read("scene.asset").unwrap(), b"scene bytes");
    assert_eq!(
        source.read_range("scene.asset#Mesh0", 5, 5).unwrap(),
        b"bytes"
    );
    assert_eq!(
        source.read_range("scene.asset#Mesh0", u64::MAX, u64::MAX),
        Err(prism_asset::ReadError::OutOfRange {
            offset: u64::MAX,
            len: u64::MAX,
            size: 10,
        })
    );
    assert_eq!(
        source.read_range("scene.asset#Mesh0", 9, 2),
        Err(prism_asset::ReadError::OutOfRange {
            offset: 9,
            len: 2,
            size: 10,
        })
    );
    assert_eq!(source.metadata("scene.asset#Mesh0").unwrap().size, 10);
    assert_eq!(
        source.read("../scene.asset"),
        Err(prism_asset::ReadError::InvalidPath)
    );
}
