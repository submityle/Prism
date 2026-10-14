//! Preparing and releasing a mesh's GPU buffers.

use prism_asset::AssetId;
use prism_mesh::Mesh;
use prism_render_driver::{
    BufferDescriptor, BufferId, BufferUsages, IndexFormat, RenderDevice, RenderQueue,
};

use crate::render_asset::{PrepareAssetError, RenderAsset};
use crate::scene::MeshBuffers;

use super::{RenderMesh, RenderMeshBufferInfo};

/// A prepared mesh's buffers plus the capacities a re-prepare compares against.
pub struct MeshBuffersAllocation {
    /// The buffers themselves.
    pub buffers: MeshBuffers,
    /// The vertex buffer's byte capacity.
    pub vertex_capacity: Vec<u64>,
    /// The index buffer's byte capacity (`0` when non-indexed).
    pub index_capacity: u64,
}

/// Creates, reuses, or releases a mesh's buffers and uploads its bytes.
///
/// The ownership rule from [`RenderAsset::prepare`] lives here: a buffer from
/// `previous` that is large enough and has the right format is reused, while a
/// replacement is created and uploaded before the old buffer is released.
pub fn prepare_mesh_buffers(
    device: &dyn RenderDevice,
    queue: &dyn RenderQueue,
    vertex_streams: &[Vec<u8>],
    index_bytes: Option<(&[u8], IndexFormat)>,
    previous: Option<&RenderMesh>,
) -> MeshBuffersAllocation {
    let mut vertices = Vec::with_capacity(vertex_streams.len());
    let mut vertex_capacity = Vec::with_capacity(vertex_streams.len());
    for (stream_index, bytes) in vertex_streams.iter().enumerate() {
        let byte_len = bytes.len() as u64;
        let previous_buffer =
            previous.and_then(|mesh| mesh.buffers.vertices.get(stream_index).copied());
        let previous_capacity = previous
            .and_then(|mesh| mesh.vertex_capacity.get(stream_index).copied())
            .unwrap_or(0);
        let reusable = previous_buffer.filter(|_| previous_capacity >= byte_len);
        let buffer = reusable.unwrap_or_else(|| {
            create_buffer(device, "mesh vertex stream", byte_len, BufferUsages::VERTEX)
        });
        queue.write_buffer(buffer, 0, bytes);
        if reusable.is_none() {
            if let Some(old) = previous_buffer {
                device.destroy_buffer(old);
            }
        }
        vertices.push(buffer);
        vertex_capacity.push(reusable.map_or(byte_len.max(4), |_| previous_capacity));
    }
    if let Some(previous) = previous {
        for old in previous.buffers.vertices.iter().skip(vertex_streams.len()) {
            device.destroy_buffer(*old);
        }
    }

    let previous_index = previous.and_then(|p| p.buffers.indices);
    let reusable_index = match (index_bytes, previous_index) {
        (Some((bytes, format)), Some((id, previous_format)))
            if previous_format == format
                && previous.is_some_and(|p| p.index_capacity >= bytes.len() as u64) =>
        {
            Some(id)
        }
        _ => None,
    };

    let (indices, index_capacity) = match index_bytes {
        Some((bytes, format)) => {
            let bytes_len = bytes.len() as u64;
            let id = reusable_index.unwrap_or_else(|| {
                create_buffer(device, "mesh index buffer", bytes_len, BufferUsages::INDEX)
            });
            queue.write_buffer(id, 0, bytes);
            if reusable_index.is_none() {
                if let Some((old, _)) = previous_index {
                    device.destroy_buffer(old);
                }
            }
            (
                Some((id, format)),
                if reusable_index.is_some() {
                    previous.map_or(bytes_len.max(4), |p| p.index_capacity)
                } else {
                    bytes_len.max(4)
                },
            )
        }
        None => {
            if let Some((old, _)) = previous_index {
                device.destroy_buffer(old);
            }
            (None, 0)
        }
    };

    MeshBuffersAllocation {
        buffers: MeshBuffers { vertices, indices },
        vertex_capacity,
        index_capacity,
    }
}

/// Creates a buffer big enough to hold `size` bytes of `usage`.
///
/// An empty upload would ask for a zero-size object, which no backend accepts;
/// the fourth byte is the smallest buffer that can hold nothing useful.
fn create_buffer(
    device: &dyn RenderDevice,
    label: &str,
    size: u64,
    usage: BufferUsages,
) -> BufferId {
    device.create_buffer(&BufferDescriptor {
        label: Some(label.into()),
        size: size.max(4),
        usage: usage | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl RenderAsset for RenderMesh {
    type SourceAsset = Mesh;

    fn prepare(
        source: &Mesh,
        id: AssetId<Mesh>,
        device: &dyn RenderDevice,
        queue: &dyn RenderQueue,
        previous: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Mesh>> {
        // An unusable mesh is a content bug, not a missing dependency: waiting
        // will not fix it, so it is reported as unpreparable and the previous
        // representation (if any) is left untouched.
        if source.validate().is_err() {
            return Err(PrepareAssetError::Unpreparable { id });
        }
        let stream_layouts = source.stream_layouts(&prism_mesh::StreamLayoutProfile::default());
        let mut vertex_streams = Vec::with_capacity(stream_layouts.len());
        for layout in &stream_layouts {
            let Some(bytes) = source.stream_buffer_data(layout) else {
                return Err(PrepareAssetError::Unpreparable { id });
            };
            vertex_streams.push(bytes);
        }
        let index_bytes = source
            .indices()
            .map(|indices| (indices.to_bytes(), indices.format()));
        let allocation = prepare_mesh_buffers(
            device,
            queue,
            &vertex_streams,
            index_bytes
                .as_ref()
                .map(|(bytes, format)| (bytes.as_slice(), *format)),
            previous,
        );

        let buffer_info = match index_bytes {
            Some((bytes, format)) => RenderMeshBufferInfo::Indexed {
                index_format: format,
                index_count: u32::try_from(bytes.len() / format.size() as usize)
                    .unwrap_or(u32::MAX),
            },
            None => RenderMeshBufferInfo::NonIndexed,
        };

        Ok(Self {
            buffers: allocation.buffers,
            buffer_info,
            vertex_count: u32::try_from(source.vertex_count()).unwrap_or(u32::MAX),
            layout: stream_layouts
                .iter()
                .map(prism_mesh::VertexStreamLayout::rhi_layout)
                .collect(),
            topology: source.primitive_topology(),
            bounds: source.bounds(),
            vertex_capacity: allocation.vertex_capacity,
            index_capacity: allocation.index_capacity,
        })
    }

    fn unload(prepared: &Self, device: &dyn RenderDevice) {
        for buffer in &prepared.buffers.vertices {
            device.destroy_buffer(*buffer);
        }
        if let Some((indices, _)) = prepared.buffers.indices {
            device.destroy_buffer(indices);
        }
    }
}
