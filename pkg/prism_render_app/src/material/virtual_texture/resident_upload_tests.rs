//! Deterministic admission/gather tests; hardware consumption lives in
//! `tests/resident_vt_gpu.rs`. No device is needed for these preflight checks.
use super::*;
use prism_render_material::{
    MaterialVtFormat, MaterialVtSourceDescriptor, MaterialVtSourceSnapshot,
};

fn layout() -> MaterialVtSourceLayout {
    MaterialVtSourceLayout::new(
        MaterialVtSourceDescriptor::new(7, 19, 9, 2, 8, 5, MaterialVtFormat::Rgba8Unorm).unwrap(),
    )
    .unwrap()
}

fn key(mip: u8, layer: u16, x: u16, y: u16) -> TexturePageKey {
    TexturePageKey {
        texture: 7,
        mip,
        layer,
        x,
        y,
    }
}

#[test]
fn gather_matches_cpu_page_rows_for_npot_layers_and_reordered_pages() {
    let layout = layout();
    let raw: Vec<u8> = (0..layout.byte_len())
        .map(|i| (i * 37 + i / 13) as u8)
        .collect();
    let snapshot = MaterialVtSourceSnapshot::new(layout.descriptor(), raw.clone()).unwrap();
    let mut keys: Vec<_> = layout.page_keys().collect();
    keys.reverse();
    let pages: Vec<_> = keys
        .into_iter()
        .enumerate()
        .map(|(slot, key)| (key, slot as u32))
        .collect();
    let offset = 65_533;
    let plan = plan_gather(
        3,
        &layout,
        offset,
        offset + layout.byte_len(),
        pages.clone(),
        usize::MAX,
    )
    .unwrap();
    assert_eq!(plan.source_index, 3);
    assert_eq!(plan.pages, pages);
    let gathered: Vec<u8> = plan
        .ranges
        .iter()
        .flat_map(|range| {
            raw[(range.start - offset) as usize..(range.end - offset) as usize]
                .iter()
                .copied()
        })
        .collect();
    assert_eq!(gathered.len() as u64, plan.byte_len);
    let mut expected = Vec::new();
    for &(key, _) in &pages {
        let page = snapshot.page(key).unwrap();
        let region = plan.regions[&key];
        assert_eq!(region.offset as usize, expected.len());
        let meta = layout.page_metadata(key).unwrap();
        assert_eq!(
            (region.width, region.height),
            (meta.extent_width, meta.extent_height)
        );
        for row in page.rows() {
            expected.extend_from_slice(row.bytes);
        }
    }
    assert_eq!(gathered, expected);
}

#[test]
fn gather_merges_only_adjacent_ranges_and_checks_final_bound() {
    let layout = layout();
    // The 1x1 tail has two consecutive array layers; these merge into one range.
    let pages = vec![(key(4, 0, 0, 0), 9), (key(4, 1, 0, 0), 2)];
    let offset = 17;
    let plan = plan_gather(
        0,
        &layout,
        offset,
        offset + layout.byte_len(),
        pages.clone(),
        1,
    )
    .unwrap();
    assert_eq!(plan.ranges.len(), 1);
    assert_eq!(plan.ranges[0].end, offset + layout.byte_len());
    assert_eq!(plan.byte_len, 8);
    assert_eq!(plan.regions[&pages[1].0].offset, 4);
    assert!(matches!(
        plan_gather(0, &layout, offset, offset + layout.byte_len() - 1, pages, 1),
        Err(NativeVtResidentError::SizeOverflow)
    ));
    let disjoint = vec![(key(0, 0, 0, 0), 0)];
    assert!(matches!(
        plan_gather(0, &layout, offset, offset + layout.byte_len(), disjoint, 1),
        Err(NativeVtResidentError::SourceRangeLimit {
            requested: 2,
            limit: 1
        })
    ));
}

#[test]
fn gather_rejects_zero_range_budget_bad_keys_and_offset_overflow() {
    let layout = layout();
    let pages = vec![(key(4, 0, 0, 0), 0)];
    assert!(matches!(
        plan_gather(0, &layout, 0, layout.byte_len(), pages.clone(), 0),
        Err(NativeVtResidentError::SourceRangeLimit {
            requested: 1,
            limit: 0
        })
    ));
    assert!(matches!(
        plan_gather(0, &layout, u64::MAX, u64::MAX, pages, 1),
        Err(NativeVtResidentError::SizeOverflow)
    ));
    for bad in [
        key(5, 0, 0, 0),
        key(0, 2, 0, 0),
        key(0, 0, 3, 0),
        TexturePageKey {
            texture: 8,
            ..key(0, 0, 0, 0)
        },
    ] {
        assert!(matches!(
            plan_gather(0, &layout, 0, layout.byte_len(), vec![(bad, 0)], 32),
            Err(NativeVtResidentError::Vt(NativeVtError::Source(_)))
        ));
    }
}

#[test]
fn assembly_admits_aggregate_padded_output_and_host_descriptors() {
    let gather = GatherPlan {
        source_index: 0,
        pages: Vec::new(),
        ranges: vec![0..1, 9..13],
        regions: BTreeMap::new(),
        byte_len: 5,
    };
    let exact = ResidentAssemblyLimits {
        max_slices: 2,
        max_device_bytes: 8 + 2 * 16,
    };
    assert_eq!(admit_assembly(&gather, exact).unwrap(), 8);
    assert!(matches!(
        admit_assembly(
            &gather,
            ResidentAssemblyLimits {
                max_device_bytes: exact.max_device_bytes - 1,
                ..exact
            }
        ),
        Err(NativeVtResidentError::Assembly(
            ResidentAssemblyError::AllocationBudgetExceeded { .. }
        ))
    ));
    assert!(matches!(
        admit_assembly(
            &gather,
            ResidentAssemblyLimits {
                max_slices: 1,
                ..exact
            }
        ),
        Err(NativeVtResidentError::Assembly(
            ResidentAssemblyError::TooManySlices {
                requested: 2,
                limit: 1
            }
        ))
    ));
}

#[test]
fn assembly_rounding_overflow_is_structured() {
    let gather = GatherPlan {
        source_index: 0,
        pages: Vec::new(),
        ranges: Vec::new(),
        regions: BTreeMap::new(),
        byte_len: u64::MAX,
    };
    assert!(matches!(
        admit_assembly(
            &gather,
            ResidentAssemblyLimits {
                max_slices: usize::MAX,
                max_device_bytes: u64::MAX
            }
        ),
        Err(NativeVtResidentError::SizeOverflow)
    ));
}

#[test]
fn upload_budget_is_aggregate_and_checked() {
    let mut used = 0;
    add_budget(&mut used, 40, 80).unwrap();
    add_budget(&mut used, 40, 80).unwrap();
    assert!(matches!(
        add_budget(&mut used, 1, 80),
        Err(NativeVtResidentError::UploadBudget {
            required: 81,
            limit: 80
        })
    ));
    let mut used = u64::MAX;
    assert!(matches!(
        add_budget(&mut used, 1, u64::MAX),
        Err(NativeVtResidentError::SizeOverflow)
    ));
}

#[test]
fn neighbor_lookup_never_crosses_layer_mip_or_negative_coordinates() {
    let center = key(0, 0, 0, 0);
    let right = key(0, 0, 1, 0);
    let below = key(0, 0, 0, 1);
    let region = ResidentPageRegion {
        offset: 4,
        width: 8,
        height: 8,
    };
    let mut regions = BTreeMap::new();
    for k in [center, right, below, key(0, 1, 0, 0), key(1, 0, 0, 0)] {
        regions.insert(k, region);
    }
    // A known but nonresident neighbor must not leak into the gutter.
    let keys = BTreeSet::from([right, key(0, 1, 0, 0), key(1, 0, 0, 0)]);
    let neighbors = neighbor_regions(center, &regions, &keys);
    for (i, neighbor) in neighbors.iter().enumerate() {
        assert_eq!(neighbor.is_some(), i == 4 || i == 5);
    }
}

#[test]
fn grouping_preserves_each_textures_input_order_and_physical_slots() {
    let a = key(0, 0, 0, 0);
    let b = TexturePageKey { texture: 9, ..a };
    let c = key(4, 1, 0, 0);
    let grouped = group_pages(&[(a, 17), (b, 3), (c, 29)]).unwrap();
    assert_eq!(grouped[&7], [(a, 17), (c, 29)]);
    assert_eq!(grouped[&9], [(b, 3)]);
}
