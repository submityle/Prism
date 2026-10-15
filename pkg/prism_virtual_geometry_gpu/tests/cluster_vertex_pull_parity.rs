//! Real-device render validation for CLST v3 vertex pulling.

use prism_render_architecture::gpu_scene::SceneBounds;
use prism_render_architecture::virtual_geometry::{
    encode_cluster_payload_with_attributes, ClusterVertexAttribute,
};
use prism_render_driver::VertexFormat;
use prism_virtual_geometry_gpu::{GpuClusterVertexPull, GpuContext};

#[test]
fn clst_v3_attribute_is_decoded_by_vertex_shader_and_reaches_color_target() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let bounds = SceneBounds {
        center: [0.0; 3],
        radius: 1.414214,
        half_extents: [1.0; 3],
        _padding: 0.0,
    };
    let payload = encode_cluster_payload_with_attributes(
        bounds,
        &[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]],
        &[0, 1, 2],
        0,
        &[ClusterVertexAttribute {
            semantic: "Vertex_Color".into(),
            shader_location: 1,
            format: VertexFormat::Unorm8x4,
            data: vec![255, 0, 0, 255].repeat(3),
        }],
    )
    .expect("valid CLST v3 fixture");
    let pixels = GpuClusterVertexPull::new(&ctx)
        .draw(&ctx, bounds, &payload, 16, 16, "Vertex_Color")
        .expect("raw CLST payload should render");
    assert_eq!(pixels.len(), 16 * 16);
    let covered = pixels
        .iter()
        .filter(|pixel| **pixel == [255, 0, 0, 255])
        .count();
    assert!(covered > 0, "the pulled triangle must cover pixels");
    assert!(
        pixels
            .iter()
            .all(|pixel| *pixel == [0, 0, 0, 255] || *pixel == [255, 0, 0, 255]),
        "all non-clear pixels must come from the decoded constant attribute"
    );
}
