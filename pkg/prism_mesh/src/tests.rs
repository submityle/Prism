//! Conformance tests for the mesh model, primitive generators, and upload.
//!
//! The CPU tests pin the geometry invariants a renderer relies on (in-range
//! indices, counts, unit normals, outward winding, unit-range UVs, and a layout
//! that matches the packed bytes). The upload test is GPU-gated: it skips when
//! no adapter is visible (the usual sandbox) and runs for real under
//! `PRISM_REQUIRE_GPU`, reading the uploaded buffers back to prove the bytes
//! round-trip exactly.

use crate::mesh::Mesh;
use crate::primitives::{cube, cylinder, plane, uv_sphere};
use crate::vertex::{Vertex, VertexLayout};
use prism_render_driver::{IndexFormat, VertexStepMode};
use prism_render_driver_wgpu::{WgpuDevice, WgpuInstance, WgpuQueue};

/// Tolerance for unit-length and on-surface checks.
const EPSILON: f32 = 1e-4;

/// The four primitives, all with the default (position/normal/uv) layout.
fn all_primitives() -> [Mesh; 4] {
    [
        plane(2.0),
        cube(2.0),
        uv_sphere(1.0, 8, 6),
        cylinder(1.0, 2.0, 12),
    ]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f32; 3], factor: f32) -> [f32; 3] {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

fn normalize(a: [f32; 3]) -> [f32; 3] {
    let len = length(a);
    scale(a, 1.0 / len)
}

/// Decodes the mesh's indices into triangles of decoded vertices.
fn triangles(mesh: &Mesh) -> impl Iterator<Item = [Vertex; 3]> + '_ {
    mesh.indices().chunks_exact(3).map(move |triangle| {
        [
            mesh.read_vertex(triangle[0] as usize).expect("in range"),
            mesh.read_vertex(triangle[1] as usize).expect("in range"),
            mesh.read_vertex(triangle[2] as usize).expect("in range"),
        ]
    })
}

fn assert_indices_in_range(mesh: &Mesh) {
    let count = mesh.vertex_count();
    for &index in mesh.indices() {
        assert!(index < count, "index {index} exceeds vertex count {count}");
    }
}

fn assert_layout_matches_packed_bytes(mesh: &Mesh) {
    let layout = mesh.layout();
    let stride = mesh.stride();
    assert_eq!(
        layout.array_stride, stride,
        "layout stride must match the mesh stride"
    );
    assert_eq!(layout.step_mode, VertexStepMode::Vertex);
    assert!(
        layout.is_within_stride(),
        "every attribute must fit within the stride"
    );
    assert_eq!(
        mesh.vertex_bytes().len() as u64,
        stride * u64::from(mesh.vertex_count()),
        "vertex bytes must be a whole number of strided vertices"
    );
}

fn assert_unit_normals(mesh: &Mesh) {
    for vertex in mesh.vertices() {
        let len = length(vertex.normal);
        assert!((len - 1.0).abs() < EPSILON, "normal length {len} is not 1");
    }
}

fn assert_uvs_in_unit_range(mesh: &Mesh) {
    for vertex in mesh.vertices() {
        for channel in vertex.uv {
            assert!(
                (0.0..=1.0).contains(&channel),
                "uv channel {channel} lies outside [0, 1]"
            );
        }
    }
}

/// For flat-faced primitives: every triangle's geometric normal matches the
/// vertices' normal, so the winding faces outward.
fn assert_flat_winding_matches_normals(mesh: &Mesh) {
    for triangle in triangles(mesh) {
        let geometric = normalize(cross(
            sub(triangle[1].position, triangle[0].position),
            sub(triangle[2].position, triangle[0].position),
        ));
        let average = normalize(add(
            add(triangle[0].normal, triangle[1].normal),
            triangle[2].normal,
        ));
        assert!(
            dot(geometric, average) > 0.99,
            "triangle winds away from its vertex normal"
        );
    }
}

/// For origin-centred convex primitives: every non-degenerate triangle faces
/// away from the origin, so the winding is outward.
fn assert_winding_faces_outward(mesh: &Mesh) {
    for triangle in triangles(mesh) {
        let raw = cross(
            sub(triangle[1].position, triangle[0].position),
            sub(triangle[2].position, triangle[0].position),
        );
        if length(raw) < 1e-6 {
            continue; // A degenerate pole triangle has no meaningful normal.
        }
        let centroid = scale(
            add(
                add(triangle[0].position, triangle[1].position),
                triangle[2].position,
            ),
            1.0 / 3.0,
        );
        assert!(
            dot(normalize(raw), centroid) >= 0.0,
            "triangle winds inward"
        );
    }
}

#[test]
fn primitives_have_in_range_indices_and_matching_layouts() {
    for mesh in all_primitives() {
        assert_indices_in_range(&mesh);
        assert_layout_matches_packed_bytes(&mesh);
    }
}

#[test]
fn primitive_counts_match_their_topology() {
    let plane = plane(2.0);
    assert_eq!((plane.vertex_count(), plane.index_count()), (4, 6));

    let cube = cube(2.0);
    assert_eq!((cube.vertex_count(), cube.index_count()), (24, 36));

    let (sectors, stacks) = (8u32, 6u32);
    let sphere = uv_sphere(1.0, sectors, stacks);
    assert_eq!(sphere.vertex_count(), (stacks + 1) * (sectors + 1));
    assert_eq!(sphere.index_count(), stacks * sectors * 6);

    let segments = 12u32;
    let cylinder = cylinder(1.0, 2.0, segments);
    assert_eq!(cylinder.vertex_count(), 4 * segments + 6);
    assert_eq!(cylinder.index_count(), 12 * segments);
}

#[test]
fn primitive_layout_uses_canonical_shader_locations() {
    let mesh = cube(1.0);
    let layout = mesh.layout();
    let locations: Vec<u32> = layout
        .attributes
        .iter()
        .map(|attribute| attribute.shader_location)
        .collect();
    assert_eq!(locations, vec![0, 1, 2], "position, normal, uv");
    assert_eq!(mesh.index_format(), IndexFormat::Uint32);
}

#[test]
fn cube_normals_are_unit_length_and_face_outward() {
    let cube = cube(2.0);
    assert_unit_normals(&cube);
    for vertex in cube.vertices() {
        assert!(
            dot(vertex.normal, vertex.position) > 0.0,
            "a cube normal must point away from the origin"
        );
    }
    assert_flat_winding_matches_normals(&cube);
    // Each face is flat: its four vertices share one normal.
    let vertices: Vec<_> = cube.vertices().collect();
    for face in vertices.chunks(4) {
        for vertex in face {
            assert_eq!(vertex.normal, face[0].normal, "a face must be flat");
        }
    }
}

#[test]
fn plane_normals_are_unit_length_and_point_up() {
    let plane = plane(2.0);
    assert_unit_normals(&plane);
    for vertex in plane.vertices() {
        assert_eq!(vertex.normal, [0.0, 1.0, 0.0]);
    }
    assert_flat_winding_matches_normals(&plane);
}

#[test]
fn sphere_vertices_are_on_surface_with_unit_normals() {
    let radius = 1.5;
    let sphere = uv_sphere(radius, 12, 8);
    assert_unit_normals(&sphere);
    for vertex in sphere.vertices() {
        assert!(
            dot(vertex.normal, vertex.position) > 0.0,
            "a sphere normal must point outward"
        );
        let distance = length(vertex.position);
        assert!(
            (distance - radius).abs() < EPSILON,
            "vertex sits at {distance}, not on the radius-{radius} sphere"
        );
    }
}

#[test]
fn primitive_uvs_lie_in_the_unit_range() {
    for mesh in all_primitives() {
        assert_uvs_in_unit_range(&mesh);
    }
}

#[test]
fn curved_primitives_wind_outward() {
    assert_winding_faces_outward(&uv_sphere(1.0, 12, 8));
    assert_winding_faces_outward(&cylinder(1.0, 2.0, 12));
}

#[test]
fn tangent_layout_packs_a_fourth_attribute() {
    let layout = VertexLayout::POSITION_NORMAL_UV_TANGENT;
    let vertices = [
        Vertex::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0])
            .with_tangent([1.0, 0.0, 0.0, 1.0]),
        Vertex::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0])
            .with_tangent([1.0, 0.0, 0.0, -1.0]),
        Vertex::new([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0])
            .with_tangent([1.0, 0.0, 0.0, 1.0]),
    ];
    let mesh = Mesh::from_vertices(&vertices, vec![0, 1, 2], layout);

    assert_eq!(mesh.stride(), 48, "3 + 3 + 2 + 4 floats");
    assert_eq!(mesh.vertex_count(), 3);
    assert_eq!(mesh.layout().attributes.len(), 4);
    assert!(mesh.layout().is_within_stride());
    for (index, vertex) in vertices.iter().enumerate() {
        assert_eq!(
            mesh.read_vertex(index),
            Some(*vertex),
            "vertex {index} must round-trip through the packed bytes"
        );
    }
}

/// Brings up a device/queue pair, or `None` when no GPU is visible.
fn device_or_skip() -> Option<(WgpuDevice, WgpuQueue)> {
    WgpuInstance::new().request_default_device().ok()
}

/// Whether a missing adapter must fail the test instead of skipping it.
fn require_gpu() -> bool {
    std::env::var_os("PRISM_REQUIRE_GPU").is_some()
}

#[test]
fn gpu_upload_round_trips_vertex_and_index_bytes() {
    let Some((device, queue)) = device_or_skip() else {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };

    let mesh = cube(2.0);
    let gpu = mesh.upload(&device, &queue);

    assert_eq!(gpu.index_format, IndexFormat::Uint32);
    assert_eq!(gpu.vertex_count, mesh.vertex_count());
    assert_eq!(gpu.index_count, mesh.index_count());

    assert_eq!(
        queue.read_buffer(gpu.vertices, mesh.vertex_bytes().len() as u64),
        mesh.vertex_bytes().to_vec(),
        "the uploaded vertex bytes must round-trip exactly"
    );
    assert_eq!(
        queue.read_buffer(gpu.indices, mesh.index_bytes().len() as u64),
        mesh.index_bytes(),
        "the uploaded index bytes must round-trip exactly"
    );

    gpu.destroy(&device);
}
