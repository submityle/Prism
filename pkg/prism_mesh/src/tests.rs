//! Behavioural tests for the mesh asset.

use alloc::vec;
use alloc::vec::Vec;

use prism_render_driver::{IndexFormat, PrimitiveTopology, VertexFormat, VertexStepMode};

use crate::{
    Indices, Mesh, MeshBounds, MeshError, MeshVertexAttribute, POSITION_ATTRIBUTE,
    VertexAttributeValues,
};

const POSITION: MeshVertexAttribute =
    MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0);
const UV: MeshVertexAttribute = MeshVertexAttribute::new("Vertex_Uv", VertexFormat::Float32x2, 1);

/// A two-triangle quad: four vertices, interleaved position-then-uv.
fn quad() -> Mesh {
    Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(
            POSITION,
            VertexAttributeValues::Float32x3(vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
        )
        .expect("positions fit")
        .with_attribute(
            UV,
            VertexAttributeValues::Float32x2(vec![
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]),
        )
        .expect("uvs fit")
        .indexed(Indices::Uint16(vec![0, 1, 2, 2, 3, 0]))
}

/// The layout the pipeline binds, in shader-location order, with the offsets the
/// interleaving actually uses.
#[test]
fn the_vertex_layout_declares_the_interleaved_offsets() {
    let mesh = quad();
    let layout = mesh.vertex_layout().expect("the quad has attributes");

    assert_eq!(layout.array_stride, 20, "12 bytes of position + 8 of uv");
    assert_eq!(layout.step_mode, VertexStepMode::Vertex);
    assert_eq!(layout.attributes.len(), 2);
    assert_eq!(layout.attributes[0].shader_location, 0, "location order");
    assert_eq!(layout.attributes[0].format, VertexFormat::Float32x3);
    assert_eq!(layout.attributes[0].offset, 0);
    assert_eq!(layout.attributes[1].shader_location, 1);
    assert_eq!(layout.attributes[1].format, VertexFormat::Float32x2);
    assert_eq!(layout.attributes[1].offset, 12, "the uv follows the position");
    assert!(
        layout.is_within_stride(),
        "every attribute lies inside the stride it declares"
    );
}

/// The uploaded bytes are the interleaving the layout promises, vertex by vertex,
/// little-endian — the exact bytes a GPU fetch would read.
#[test]
fn the_vertex_buffer_bytes_match_the_declared_layout() {
    let mesh = quad();
    let data = mesh.vertex_buffer_data();
    assert_eq!(data.len(), 4 * 20, "one stride per vertex");

    let mut expected: Vec<u8> = Vec::new();
    for (position, uv) in [
        ([0.0_f32, 0.0, 0.0], [0.0_f32, 0.0]),
        ([1.0, 0.0, 0.0], [1.0, 0.0]),
        ([1.0, 1.0, 0.0], [1.0, 1.0]),
        ([0.0, 1.0, 0.0], [0.0, 1.0]),
    ] {
        for value in position {
            expected.extend_from_slice(&value.to_le_bytes());
        }
        for value in uv {
            expected.extend_from_slice(&value.to_le_bytes());
        }
    }
    assert_eq!(data, expected, "the interleaving is exact");

    // The second vertex starts one stride in, at the position it was authored
    // with: `1.0` as `f32` little-endian.
    let second = &data[20..24];
    assert_eq!(second, 1.0_f32.to_le_bytes());
}

/// An attribute whose storage does not match its declared format is refused, so
/// the GPU can never fetch reinterpreted bytes.
#[test]
fn an_attribute_that_does_not_match_its_format_is_refused() {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
    let mismatch = mesh.insert_attribute(
        POSITION,
        VertexAttributeValues::Float32x2(vec![[0.0, 0.0]]),
    );
    assert_eq!(
        mismatch,
        Err(MeshError::FormatMismatch {
            attribute: POSITION_ATTRIBUTE
        })
    );
    assert_eq!(mesh.attribute_count(), 0, "nothing was stored");
}

/// Attributes index the same vertices, so a short one is refused rather than
/// reading past its own data for the last vertices.
#[test]
fn attributes_must_agree_on_the_vertex_count() {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
    mesh.insert_attribute(
        POSITION,
        VertexAttributeValues::Float32x3(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    .expect("the first attribute defines the count");

    let mismatch = mesh.insert_attribute(UV, VertexAttributeValues::Float32x2(vec![[0.0, 0.0]]));
    assert_eq!(
        mismatch,
        Err(MeshError::VertexCountMismatch {
            attribute: "Vertex_Uv",
            expected: 2,
            found: 1,
        })
    );
    assert_eq!(mesh.attribute_count(), 1, "the bad attribute was not stored");
}

/// Indices stay at the width they were authored in, and the byte encoding
/// follows that width.
#[test]
fn indices_keep_their_authored_width() {
    let sixteen = Indices::Uint16(vec![0, 1, 2]);
    assert_eq!(sixteen.format(), IndexFormat::Uint16);
    assert_eq!(sixteen.to_bytes().len(), 6);
    assert_eq!(sixteen.to_u32(), vec![0, 1, 2]);
    assert_eq!(sixteen.max_index(), Some(2));

    let thirty_two = Indices::Uint32(vec![0, 70_000]);
    assert_eq!(thirty_two.format(), IndexFormat::Uint32);
    assert_eq!(thirty_two.to_bytes().len(), 8);
    assert_eq!(
        thirty_two.max_index(),
        Some(70_000),
        "a wide index survives the round trip"
    );

    assert_eq!(Indices::Uint16(Vec::new()).max_index(), None);
}

/// An index past the vertex range would read another mesh's vertices on the GPU,
/// and a mesh with no attributes has no vertex stage at all: both are refused
/// before a device ever sees them.
#[test]
fn validate_refuses_unusable_meshes() {
    let mut out_of_range = quad();
    out_of_range.with_indices(Indices::Uint16(vec![0, 1, 9]));
    assert_eq!(
        out_of_range.validate(),
        Err(MeshError::IndexOutOfRange {
            max_index: 9,
            vertex_count: 4,
        })
    );

    let empty = Mesh::new(PrimitiveTopology::TriangleList);
    assert_eq!(empty.validate(), Err(MeshError::NoAttributes));
    assert!(empty.vertex_layout().is_none());
    assert!(empty.vertex_buffer_data().is_empty());
    assert!(empty.bounds().is_none());

    assert_eq!(quad().validate(), Ok(()), "the fixture mesh is valid");
}

/// Bounds come from the positions, ignore non-finite vertices, and report the
/// center/radius a culling sphere needs.
#[test]
fn bounds_follow_the_positions() {
    let mesh = quad();
    let bounds = mesh.bounds().expect("the quad has positions");
    assert_eq!(bounds.min, [0.0, 0.0, 0.0]);
    assert_eq!(bounds.max, [1.0, 1.0, 0.0]);
    assert_eq!(bounds.center(), [0.5, 0.5, 0.0]);
    assert_eq!(bounds.extents(), [1.0, 1.0, 0.0]);
    assert!((bounds.radius() - core::f32::consts::SQRT_2 * 0.5).abs() < 1e-6);

    let cluster = MeshBounds::from_positions(&[[0.0, 0.0, 0.0], [2.0, -1.0, 4.0]]).expect("finite");
    assert_eq!(cluster.min, [0.0, -1.0, 0.0]);
    assert_eq!(cluster.max, [2.0, 0.0, 4.0]);

    let polluted = MeshBounds::from_positions(&[[0.0, 0.0, 0.0], [f32::NAN, 1.0, 1.0]]);
    assert_eq!(
        polluted.map(|bounds| bounds.max),
        Some([0.0, 0.0, 0.0]),
        "a non-finite vertex does not poison the box"
    );
    assert_eq!(MeshBounds::from_positions(&[]), None);
}

/// A mesh is an asset: the renderer stores it in an `Assets<Mesh>` arena and
/// prepares from it, so it carries the asset contract's identity.
#[test]
fn a_mesh_is_an_asset() {
    assert_eq!(<Mesh as prism_asset::Asset>::TYPE_NAME, "prism_mesh::Mesh");
}

