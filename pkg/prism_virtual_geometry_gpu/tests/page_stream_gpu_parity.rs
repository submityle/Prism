//! Real-device parity for the virtual-geometry page table and page storage.

use prism_render_architecture::gpu_scene::SceneBounds;
use prism_render_architecture::paging::{RequestBatch, UNMAPPED_SLOT};
use prism_render_architecture::virtual_geometry::{
    ClusterNode, GeometryPageKey, VirtualGeometryAsset, VirtualGeometryAssetHeader,
    VirtualGeometryClusterRecord, VirtualGeometryCompression, VirtualGeometryPage,
    VirtualGeometryPageRecord, VirtualGeometryPageStream, VIRTUAL_GEOMETRY_ASSET_SCHEMA,
};
use prism_virtual_geometry_gpu::{GpuContext, GpuPageStorage, GpuPageTable};

fn stream() -> (VirtualGeometryPageStream, GeometryPageKey) {
    let key = GeometryPageKey::new(7, 0);
    let bounds = SceneBounds {
        center: [0.0; 3],
        radius: 1.0,
        half_extents: [1.0; 3],
        _padding: 0.0,
    };
    let raw = [7_u8, 8, 9, 10];
    let compressed = prism_render_architecture::compression::gdeflate_compress(&raw);
    let asset = VirtualGeometryAsset {
        header: VirtualGeometryAssetHeader {
            asset_id: 7,
            schema_version: VIRTUAL_GEOMETRY_ASSET_SCHEMA,
            page_size: 1024,
            build_settings_hash: [1; 32],
            content_hash: [2; 32],
            bounds,
            material_slot_count: 1,
            fallback_mesh_hash: [0; 32],
        },
        nodes: vec![ClusterNode::leaf(bounds, 0.0, key)],
        roots: vec![0],
        pages: vec![VirtualGeometryPage {
            record: VirtualGeometryPageRecord {
                key,
                compression: VirtualGeometryCompression::GDeflate,
                uncompressed_size: raw.len() as u32,
                priority: 1,
                cluster_start: 0,
                cluster_count: 1,
            },
            payload: compressed,
        }],
        clusters: vec![VirtualGeometryClusterRecord {
            node_index: 0,
            page: key,
            material_slot: 0,
            vertex_count: 3,
            triangle_count: 1,
            payload_offset: 0,
            payload_size: raw.len() as u32,
            bounds,
            normal_cone: [0.0, 0.0, 1.0, 1.0],
        }],
    };
    let mut stream = VirtualGeometryPageStream::from_asset(asset, 2, 2).unwrap();
    let mut requests = RequestBatch::new();
    requests.record(key, 10.0);
    stream.reconcile(&requests, 1);
    assert_eq!(stream.resident_count(), 1);
    (stream, key)
}

#[test]
fn gpu_page_table_and_storage_match_the_stream_cpu_contract() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping virtual-geometry page parity: no GPU adapter");
        return;
    };
    let (stream, key) = stream();
    let entries = stream.entries();
    let queries = vec![key, GeometryPageKey::new(7, 99)];
    let resolved = GpuPageTable::new(&ctx).resolve(&ctx, &entries, &queries).unwrap();
    assert_eq!(resolved, vec![entries[0].1, UNMAPPED_SLOT]);

    let uploads: Vec<_> = entries
        .iter()
        .map(|(_, slot)| (*slot, stream.storage().slot_words(*slot).unwrap()))
        .collect();
    let fetches = vec![(entries[0].1, 0), (entries[0].1, 1), (99, 0)];
    let fetched = GpuPageStorage::new(&ctx).round_trip(
        &ctx,
        stream.storage().page_words(),
        stream.storage().capacity(),
        &uploads,
        &fetches,
    );
    assert_eq!(
        fetched,
        vec![
            stream.storage().fetch(entries[0].1, 0).unwrap(),
            stream.storage().fetch(entries[0].1, 1).unwrap(),
            0,
        ]
    );
}
