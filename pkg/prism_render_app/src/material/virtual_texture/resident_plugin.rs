//! Opt-in compressed resident uploads on the renderer's original native tuple.

use prism_app::{plugin::PluginDependency, App, Plugin};

use super::{
    NativeVtError, NativeVtResidentError, NativeVtResidentLimits, NativeVtResidentUploader,
    NativeVtUploadDriver,
};
use crate::{
    PrismRenderPlugin, RenderApp, RenderRuntimeResource, ResidentRenderDevice, ResidentWgpuInstance,
};

/// Installs resident Manifest-to-atlas staging into the ordinary render prepare
/// schedule. Add after `PrismRenderPlugin<ResidentWgpuInstance>`. CPU and resident
/// sources may coexist; CPU-only plans retain the ordinary upload fast path.
/// Decoder/gather/packer pipelines are cached per committed device generation.
/// This is a resident decode/upload bridge, not a direct-storage I/O backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeVtResidentPlugin {
    pub limits: NativeVtResidentLimits,
}

impl Plugin for NativeVtResidentPlugin {
    fn dependencies(&self) -> Vec<PluginDependency> {
        vec![PluginDependency::on::<
            PrismRenderPlugin<ResidentWgpuInstance>,
        >()]
    }

    fn build(&self, app: &mut App) {
        let render = app
            .sub_app_mut(RenderApp)
            .expect("NativeVtResidentPlugin requires PrismRenderPlugin<ResidentWgpuInstance>");
        assert!(
            render
                .world
                .get_resource::<RenderRuntimeResource<ResidentWgpuInstance>>()
                .is_some(),
            "NativeVtResidentPlugin requires the resident renderer's committed runtime"
        );
        assert!(
            render
                .world
                .get_resource::<NativeVtUploadDriver<ResidentWgpuInstance>>()
                .is_none(),
            "native VT upload driver is already installed"
        );
        render.world.insert_resource(resident_driver(self.limits));
    }
}

fn resident_driver(limits: NativeVtResidentLimits) -> NativeVtUploadDriver<ResidentWgpuInstance> {
    let mut cached: Option<(u64, NativeVtResidentUploader)> = None;
    NativeVtUploadDriver::new(move |runtime, owner, sources, heap, residency| {
        let resident = residency
            .pages
            .iter()
            .map(|(key, _)| key.texture)
            .chain(residency.portable.iter().copied())
            .any(|texture| {
                sources
                    .get(texture)
                    .is_some_and(|source| source.resident().is_some())
            });
        if !resident {
            return owner.stage(runtime, sources, heap, residency);
        }
        let generation = runtime.device_generation();
        if cached
            .as_ref()
            .is_none_or(|(previous, _)| *previous != generation)
        {
            // Discard old native pipelines even if reconstruction fails. A later
            // frame can retry; no failed candidate becomes the upload context.
            cached = None;
            let shared =
                ResidentRenderDevice::from_runtime(runtime).ok_or(NativeVtError::DeviceMismatch)?;
            let uploader = NativeVtResidentUploader::new(shared).map_err(stage_error)?;
            cached = Some((generation, uploader));
        }
        cached
            .as_ref()
            .expect("resident uploader initialized")
            .1
            .stage(owner, runtime, sources, heap, residency, limits)
            .map_err(stage_error)
    })
}

fn stage_error(error: NativeVtResidentError) -> NativeVtError {
    match error {
        NativeVtResidentError::Vt(error) => error,
        error => NativeVtError::ResidentUpload(error.to_string()),
    }
}
