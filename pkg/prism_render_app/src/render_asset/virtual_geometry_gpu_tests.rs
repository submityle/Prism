//! CLST vertex-pulling CPU/GPU contract coverage.

use super::*;
use alloc::vec;
use bytemuck::{cast_slice, Pod, Zeroable};
use prism_render_architecture::{
    gpu_scene::{SceneBounds, SceneMaterialHandle},
    virtual_geometry::{
        cluster_payload::{decode_cluster_payload, encode_cluster_payload_with_attributes},
        gpu::{VirtualGeometryGpuCluster, VirtualGeometryGpuPage},
        ClusterNode, ClusterVertexAttribute, GeometryPageKey, VirtualGeometryAsset,
        VirtualGeometryAssetHeader, VirtualGeometryClusterRecord, VirtualGeometryCompression,
        VirtualGeometryPage, VirtualGeometryPageRecord, NON_RESIDENT_PHYSICAL_PAGE,
        VIRTUAL_GEOMETRY_ASSET_SCHEMA,
    },
};

const SHADER: &str = include_str!("clst_vertex_pull.wgsl");

fn bounds() -> SceneBounds {
    SceneBounds {
        center: [0.0, 0.0, 0.0],
        radius: 1.0,
        half_extents: [1.0; 3],
        _padding: 0.0,
    }
}

fn fixture() -> (VirtualGeometryAsset, Vec<u8>) {
    let bounds = bounds();
    let positions = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
    let indices = [0, 1, 2];
    let normal_data = [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let payload = encode_cluster_payload_with_attributes(
        bounds,
        &positions,
        &indices,
        0,
        &[ClusterVertexAttribute {
            semantic: "Vertex_Normal".into(),
            shader_location: 1,
            format: prism_render_driver::VertexFormat::Float32x3,
            data: normal_data,
        }],
    )
    .unwrap();
    let key = GeometryPageKey::new(41, 0);
    let page_size = 1024;
    let mut page = vec![0; page_size];
    page[..payload.len()].copy_from_slice(&payload);
    let asset = VirtualGeometryAsset {
        header: VirtualGeometryAssetHeader {
            asset_id: 41,
            schema_version: VIRTUAL_GEOMETRY_ASSET_SCHEMA,
            page_size: page_size as u32,
            build_settings_hash: [1; 32],
            content_hash: [2; 32],
            bounds,
            material_slot_count: 1,
            fallback_mesh_hash: [3; 32],
        },
        nodes: vec![ClusterNode::leaf(bounds, 0.0, key)],
        roots: vec![0],
        pages: vec![VirtualGeometryPage {
            record: VirtualGeometryPageRecord {
                key,
                compression: VirtualGeometryCompression::None,
                uncompressed_size: payload.len() as u32,
                priority: 1,
                cluster_start: 0,
                cluster_count: 1,
            },
            payload: payload.clone(),
        }],
        clusters: vec![VirtualGeometryClusterRecord {
            node_index: 0,
            page: key,
            material_slot: 0,
            vertex_count: 3,
            triangle_count: 1,
            payload_offset: 0,
            payload_size: payload.len() as u32,
            bounds,
            normal_cone: [0.0, 0.0, 1.0, 1.0],
        }],
    };
    (asset, payload)
}

#[test]
fn clst_vertex_pull_records_match_the_published_abis() {
    assert_eq!(size_of::<VirtualGeometryGpuPage>(), 32);
    assert_eq!(size_of::<VirtualGeometryGpuCluster>(), 32);
    assert_eq!(size_of::<ClstPullParams>(), 40);
    assert_eq!(size_of::<ClstPulledVertex>(), 48);
    assert_eq!(NON_RESIDENT_PHYSICAL_PAGE, u32::MAX);
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClstPullParams {
    page_count: u32,
    cluster_count: u32,
    page_storage_words: u32,
    material_count: u32,
    expected_generation: [u32; 2],
    cluster_index: u32,
    triangle_index: u32,
    requested_location: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct ClstPulledVertex {
    position: [f32; 4],
    attribute: [f32; 4],
    material_index: u32,
    material_generation: u32,
    valid: u32,
    pad: u32,
}

#[test]
fn cpu_golden_decodes_triangle_positions_attributes_and_rejects_bad_inputs() {
    let (asset, payload) = fixture();
    asset.validate().unwrap();
    let decoded = decode_cluster_payload(bounds(), &payload).unwrap();
    assert_eq!(decoded.positions.len(), 3);
    assert_eq!(decoded.indices, [0, 1, 2]);
    assert_eq!(decoded.attributes[0].shader_location, 1);
    assert_eq!(decoded.attributes[0].semantic, "Vertex_Normal");
    assert!(decoded
        .positions
        .iter()
        .all(|p| p.iter().all(|v| v.is_finite())));

    for cut in [0, 1, 2] {
        let vertex = decoded.indices[cut] as usize;
        assert_eq!(decoded.positions[vertex][2], 0.0);
        assert_eq!(
            f32::from_le_bytes(
                decoded.attributes[0].data[vertex * 12..vertex * 12 + 4]
                    .try_into()
                    .unwrap()
            ),
            0.0
        );
    }
    for end in 0..payload.len() {
        assert!(decode_cluster_payload(bounds(), &payload[..end]).is_err());
    }
}

#[cfg(feature = "fixed-page-feedback")]
#[test]
fn clst_vertex_pull_shader_executes_against_gpu_and_fails_closed() {
    use prism_render_driver_wgpu::{require_hardware_gpu, WgpuInstance};

    let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
        Ok(pair) => pair,
        Err(prism_render_driver_wgpu::HardwareDeviceError::NoAdapter) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 but no adapter was found"
            );
            eprintln!("SKIPPED CLST vertex-pulling hardware test: no adapter");
            return;
        }
        Err(prism_render_driver_wgpu::HardwareDeviceError::NonHardware(evidence)) => {
            assert!(
                !require_hardware_gpu(),
                "physical GPU required, found {evidence}"
            );
            eprintln!("SKIPPED CLST physical-GPU evidence: {evidence}");
            return;
        }
        Err(error) => panic!("hardware device creation failed: {error}"),
    };
    let native = device.wgpu_device();
    let (asset, payload) = fixture();
    let revision = asset.header.content_hash;
    let bindings = [VirtualGeometryMaterialBinding {
        material_slot: 0,
        material: SceneMaterialHandle {
            index: 2,
            generation: 9,
        },
    }];
    let publication = VirtualGeometryGpuPublication::build(
        &asset,
        revision,
        0x1_0000_0007,
        &[VirtualGeometryPhysicalPage {
            key: GeometryPageKey::new(41, 0),
            physical_page: 0,
            byte_offset: 0,
            byte_len: payload.len() as u32,
            generation: 0x1_0000_0007,
        }],
        &bindings,
    )
    .unwrap();
    let mut words = vec![0u32; 1024 / 4];
    let payload_words: &[u32] = cast_slice(&payload);
    words[..payload_words.len()].copy_from_slice(payload_words);
    let mut material = [0u32; 40];
    material[0] = 9;
    material[13] = 1;

    let shader = native.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("CLST vertex-pulling contract"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = native.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("CLST pull test"),
        layout: None,
        module: &shader,
        entry_point: Some("clst_pull_contract_test"),
        compilation_options: Default::default(),
        cache: None,
    });
    let page_buffer = native.create_buffer_init("pages", cast_slice(publication.pages()));
    let cluster_buffer = native.create_buffer_init("clusters", cast_slice(publication.clusters()));
    let storage_buffer = native.create_buffer_init("page storage", cast_slice(&words));
    let material_buffer = native.create_buffer_init("material headers", cast_slice(&material));
    let output_buffer = native.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pull output"),
        size: (3 * size_of::<ClstPulledVertex>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = ClstPullParams {
        page_count: 1,
        cluster_count: 1,
        page_storage_words: words.len() as u32,
        material_count: 3,
        expected_generation: [7, 1],
        cluster_index: 0,
        triangle_index: 0,
        requested_location: 1,
        pad: 0,
    };
    let params_buffer = native.create_buffer_init("params", bytemuck::bytes_of(&params));
    let bind = native.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("CLST pull bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: page_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cluster_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: material_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });
    let execute = |params: ClstPullParams| {
        queue
            .wgpu_queue()
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
        let mut encoder = native.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.wgpu_queue().submit([encoder.finish()]);
        let readback = native.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CLST pull readback"),
            size: (3 * size_of::<ClstPulledVertex>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = native.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, readback.size());
        queue.wgpu_queue().submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap()
        });
        native.poll(wgpu::PollType::Wait).unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let values = cast_slice::<u8, ClstPulledVertex>(&mapped).to_vec();
        drop(mapped);
        readback.unmap();
        values
    };

    let pulled = execute(params);
    assert!(pulled
        .iter()
        .all(|v| v.valid == 1 && v.material_index == 2 && v.material_generation == 9));
    let decoded = decode_cluster_payload(bounds(), &payload).unwrap();
    for (gpu, index) in pulled.iter().zip(decoded.indices) {
        for axis in 0..3 {
            assert!(
                (gpu.position[axis] - decoded.positions[index as usize][axis]).abs()
                    <= 2.0 / 65535.0
            );
        }
        assert_eq!(gpu.attribute[..3], [0.0, 0.0, 1.0]);
    }

    let mut bad = params;
    bad.expected_generation = [8, 1];
    assert!(execute(bad).iter().all(|v| v.valid == 0));
    bad = params;
    bad.cluster_index = u32::MAX;
    assert!(execute(bad).iter().all(|v| v.valid == 0));
    bad = params;
    bad.requested_location = u32::MAX;
    assert!(execute(bad).iter().all(|v| v.valid == 0));

    drop((
        bind,
        pipeline,
        shader,
        page_buffer,
        cluster_buffer,
        storage_buffer,
        material_buffer,
        output_buffer,
        params_buffer,
    ));
}

#[cfg(feature = "fixed-page-feedback")]
trait WgpuBufferInit {
    fn create_buffer_init(&self, label: &str, contents: &[u8]) -> wgpu::Buffer;
}

#[cfg(feature = "fixed-page-feedback")]
impl WgpuBufferInit for wgpu::Device {
    fn create_buffer_init(&self, label: &str, contents: &[u8]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }
}
