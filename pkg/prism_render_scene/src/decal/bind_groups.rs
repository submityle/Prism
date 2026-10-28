//! Exact group-0 assembly for authenticated Decal domain resources.

use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries, BufferBinding},
    renderer::RenderDevice,
};
use std::num::NonZeroU64;

use crate::{
    fabric_domains::{
        gpu::{DomainRecordReadyResource, DomainRecordReadyView},
        resources::{
            DECAL_CLUSTER_INDEX_RESOURCE_ROLE, DECAL_CLUSTER_RANGE_RESOURCE_ROLE,
            DECAL_INSTANCE_RESOURCE_ROLE, DECAL_MATERIAL_RESOURCE_ROLE,
            DECAL_RECEIVER_RESOURCE_ROLE, DECAL_RESOLVE_RESOURCE_ROLE,
        },
    },
    gpu::rhi::SceneRhi,
};

use super::pipeline::DecalResolvePipeline;

fn binding_for<'a>(
    view: &'a DomainRecordReadyView,
    role: prism_render_architecture::render_fabric::PayloadRole,
) -> Result<BufferBinding<'a>, &'static str> {
    let resource = view
        .resources()
        .iter()
        .find(|resource| resource.declaration().payload_role() == role)
        .ok_or("missing Decal resource role")?;
    match resource
        .record_ready()
        .map_err(|_| "resource is not record-ready")?
    {
        DomainRecordReadyResource::Buffer {
            owner,
            offset,
            bytes,
            ..
        } => Ok(BufferBinding {
            buffer: owner.buffer(),
            offset,
            size: NonZeroU64::new(bytes),
        }),
        DomainRecordReadyResource::Texture { .. } => Err("Decal resource must be a buffer"),
    }
}

pub(crate) fn create_decal_bind_group(
    _rhi: &SceneRhi,
    device: &RenderDevice,
    pipeline: &DecalResolvePipeline,
    view: &DomainRecordReadyView,
) -> Result<BindGroup, &'static str> {
    let entries = (
        binding_for(view, DECAL_INSTANCE_RESOURCE_ROLE)?,
        binding_for(view, DECAL_MATERIAL_RESOURCE_ROLE)?,
        binding_for(view, DECAL_RECEIVER_RESOURCE_ROLE)?,
        binding_for(view, DECAL_CLUSTER_RANGE_RESOURCE_ROLE)?,
        binding_for(view, DECAL_CLUSTER_INDEX_RESOURCE_ROLE)?,
        binding_for(view, DECAL_RESOLVE_RESOURCE_ROLE)?,
    );
    Ok(device.create_bind_group(
        "prism decal resolve",
        pipeline.layout(),
        &BindGroupEntries::sequential(entries),
    ))
}
