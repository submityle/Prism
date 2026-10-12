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
    pub vertex_capacity: u64,
    /// The index buffer's byte capacity (`0` when non-indexed).
    pub index_capacity: u64,
}

/// Creates, reuses, or releases a mesh's buffers and uploads its bytes.
///
/// The ownership rule from [`RenderAsset::prepare`] lives here: a buffer from
/// `previous` that is large enough and has the right format is reused, and one
/// that is not is released through `device` before a replacement is created — so
/// exactly one of the two GPU objects exists per buffer kind, whatever the mesh
/// did between frames.
pub fn prepare_mesh_buffers(
    device: &dyn RenderDevice,
    queue: &dyn RenderQueue,
    vertex_bytes: &[u8],
    index_bytes: Option<(&[u8], IndexFormat)>,
    previous: Option<&RenderMesh>,
) -> MeshBuffersAllocation {
    let vertex_bytes_len = vertex_bytes.len() as u64;
    let vertices = match previous {
        Some(previous) if previous.vertex_capacity >= vertex_bytes_len => {
            previous.buffers.vertices
        }
        Some(previous) => {
            device.destroy_buffer(previous.buffers.vertices);
            create_buffer(device, "mesh vertex buffer", vertex_bytes_len, BufferUsages::VERTEX)
        }
        None => create_buffer(device, "mesh vertex buffer", vertex_bytes_len, BufferUsages::VERTEX),
    };
    queue.write_buffer(vertices, 0, vertex_bytes);

    let reusable_index = match (index_bytes, previous.and_then(|p| p.buffers.indices)) {
        (Some((bytes, format)), Some((id, previous_format)))
            if previous_format == format
                && previous.is_some_and(|p| p.index_capacity >= bytes.len() as u64) =>
        {
            Some(id)
        }
        (_, previous_buffer) => {
            // Either the mesh stopped being indexed, changed width, or grew: the
            // old buffer goes, and a new one is created only if still needed.
            if let Some((id, _)) = previous_buffer {
                device.destroy_buffer(id);
            }
            None
        }
    };

    let (indices, index_capacity) = match index_bytes {
        Some((bytes, format)) => {
            let bytes_len = bytes.len() as u64;
            let id = reusable_index.unwrap_or_else(|| {
                create_buffer(device, "mesh index buffer", bytes_len, BufferUsages::INDEX)
            });
            queue.write_buffer(id, 0, bytes);
            (Some((id, format)), bytes_len)
        }
        None => (None, 0),
    };

    MeshBuffersAllocation {
        buffers: MeshBuffers { vertices, indices },
        vertex_capacity: vertex_bytes_len,
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
        let Some(layout) = source.vertex_layout() else {
            // `validate` already rejects an attribute-less mesh; the check is
            // repeated so the layout is never assumed to exist.
            return Err(PrepareAssetError::Unpreparable { id });
        };

        let vertex_bytes = source.vertex_buffer_data();
        let index_bytes = source
            .indices()
            .map(|indices| (indices.to_bytes(), indices.format()));
        let allocation = prepare_mesh_buffers(
            device,
            queue,
            &vertex_bytes,
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
            layout,
            topology: source.primitive_topology(),
            bounds: source.bounds(),
            vertex_capacity: allocation.vertex_capacity,
            index_capacity: allocation.index_capacity,
        })
    }

    fn unload(prepared: &Self, device: &dyn RenderDevice) {
        device.destroy_buffer(prepared.buffers.vertices);
        if let Some((indices, _)) = prepared.buffers.indices {
            device.destroy_buffer(indices);
        }
    }
}

