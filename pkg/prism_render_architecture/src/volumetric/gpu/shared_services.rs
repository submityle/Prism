//! The shared-base-service wiring for each volumetric `GPU` kernel (design
//! section 17, milestone M8: "接入共享服务签名").
//!
//! The volumetric subsystem does not own the engine's advanced lighting base:
//! hybrid global illumination, `ReSTIR` many-light resampling, the virtual
//! shadow map, the froxel light-shaft volume, the atmosphere-scattering `LUT`,
//! the offline path-traced reference, and the temporal upsampling history are
//! all *shared services* (design section 5 / 9 boundary rules). The volumetric
//! kernels **consume** those services (sampling a `LUT`, reading ambient `GI`)
//! and in two cases **write into** a shared target (injecting into the froxel
//! volume for god rays, writing the cloud contribution into the virtual shadow
//! map) — but they never re-implement them.
//!
//! This module owns the *signature* of that coupling: which
//! [`SharedBaseServices`] each [`VolumetricKernel`] binds and whether the
//! binding is a read-only sample or a write into the shared resource. It is
//! pure integer/enum bookkeeping, so the wiring table is a deterministic,
//! `CPU`-testable function of a kernel tag.
//!
//! **Device-verified twin.** The `WESL` twin
//! (`shaders/volumetric_clouds.wesl`, device-verified by
//! `prism_render_scene::shading::volumetric_clouds`) declares the kernels these
//! bindings describe; the render-graph backend there resolves them to real
//! bind-group resources and the eight kernels pass on-device Metal parity, so
//! this access table is the contract the wired dispatch pass already fulfils.

use super::super::SharedBaseServices;
use super::kernels::VolumetricKernel;

/// How a kernel touches a shared service resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ServiceAccess {
    /// Read-only sample of a shared `LUT` / probe field (atmosphere `LUT`,
    /// hybrid `GI`, the path-traced reference for calibration).
    SampleReadOnly,
    /// Write (inject) into the shared froxel light-shaft volume for
    /// crepuscular / god-ray accumulation.
    InjectWrite,
    /// Write the cloud contribution into the shared virtual shadow map.
    ShadowWrite,
    /// Read the previous-frame history and write the reconstructed frame back
    /// into the shared temporal-upsampling history.
    HistoryReadWrite,
}

impl ServiceAccess {
    /// `true` when the access writes into the shared resource (as opposed to a
    /// read-only sample), so the render graph must order it as a producer of
    /// that shared service.
    #[must_use]
    pub fn is_write(self) -> bool {
        matches!(
            self,
            ServiceAccess::InjectWrite
                | ServiceAccess::ShadowWrite
                | ServiceAccess::HistoryReadWrite
        )
    }
}

/// One shared-service binding a kernel declares: which single service and how
/// it is accessed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SharedServiceBinding {
    /// The single shared service this binding names (a one-bit
    /// [`SharedBaseServices`] set).
    pub service: SharedBaseServices,
    /// How the kernel touches it.
    pub access: ServiceAccess,
}

/// The view ray-march samples the atmosphere `LUT` and hybrid `GI` ambient and
/// injects its scattering into the shared froxel volume for god rays.
const RAYMARCH_BINDINGS: [SharedServiceBinding; 3] = [
    SharedServiceBinding {
        service: SharedBaseServices::ATMOSPHERE_LUT,
        access: ServiceAccess::SampleReadOnly,
    },
    SharedServiceBinding {
        service: SharedBaseServices::HYBRID_GI,
        access: ServiceAccess::SampleReadOnly,
    },
    SharedServiceBinding {
        service: SharedBaseServices::FROXEL_VOLUME,
        access: ServiceAccess::InjectWrite,
    },
];

/// The octave-scatter resolve folds in the hybrid `GI` ambient term and is
/// calibrated against the offline path-traced reference.
const SCATTER_BINDINGS: [SharedServiceBinding; 2] = [
    SharedServiceBinding {
        service: SharedBaseServices::HYBRID_GI,
        access: ServiceAccess::SampleReadOnly,
    },
    SharedServiceBinding {
        service: SharedBaseServices::RT_REFERENCE,
        access: ServiceAccess::SampleReadOnly,
    },
];

/// The cloud-shadow march writes the cloud contribution into the shared virtual
/// shadow map.
const SHADOW_BINDINGS: [SharedServiceBinding; 1] = [SharedServiceBinding {
    service: SharedBaseServices::VSM_SHADOW,
    access: ServiceAccess::ShadowWrite,
}];

/// The temporal upsample reads and writes the shared upsampling history.
const UPSAMPLE_BINDINGS: [SharedServiceBinding; 1] = [SharedServiceBinding {
    service: SharedBaseServices::TEMPORAL_UPSAMPLE,
    access: ServiceAccess::HistoryReadWrite,
}];

/// No shared-service bindings (the internal bakes: weather, noise, density,
/// multiple-scatter `LUT`).
const NO_BINDINGS: [SharedServiceBinding; 0] = [];

/// The shared-service bindings a kernel declares, in a stable order.
///
/// The four internal bakes (weather advect, noise bake, density modelling,
/// multiple-scatter `LUT`) touch no shared service and return an empty slice.
#[must_use]
pub fn shared_service_bindings(kernel: VolumetricKernel) -> &'static [SharedServiceBinding] {
    match kernel {
        VolumetricKernel::WeatherAdvect
        | VolumetricKernel::NoiseBake
        | VolumetricKernel::Modeling
        | VolumetricKernel::MultiscatterLutBake => &NO_BINDINGS,
        VolumetricKernel::Raymarch => &RAYMARCH_BINDINGS,
        VolumetricKernel::ScatterResolve => &SCATTER_BINDINGS,
        VolumetricKernel::ShadowMarch => &SHADOW_BINDINGS,
        VolumetricKernel::Upsample => &UPSAMPLE_BINDINGS,
    }
}

/// The union of every shared service a kernel consumes or produces.
#[must_use]
pub fn touched_services(kernel: VolumetricKernel) -> SharedBaseServices {
    shared_service_bindings(kernel)
        .iter()
        .fold(SharedBaseServices::NONE, |acc, b| acc.union(b.service))
}

/// The union of the shared services a kernel *writes into* (froxel injection,
/// virtual-shadow write, history write-back), so the render graph can order it
/// as a producer of those services.
#[must_use]
pub fn produced_services(kernel: VolumetricKernel) -> SharedBaseServices {
    shared_service_bindings(kernel)
        .iter()
        .filter(|b| b.access.is_write())
        .fold(SharedBaseServices::NONE, |acc, b| acc.union(b.service))
}

#[cfg(test)]
mod tests {
    use super::{
        produced_services, shared_service_bindings, touched_services, ServiceAccess,
        SharedBaseServices, VolumetricKernel,
    };

    #[test]
    fn internal_bakes_touch_no_shared_service() {
        for kernel in [
            VolumetricKernel::WeatherAdvect,
            VolumetricKernel::NoiseBake,
            VolumetricKernel::Modeling,
            VolumetricKernel::MultiscatterLutBake,
        ] {
            assert!(shared_service_bindings(kernel).is_empty());
            assert!(touched_services(kernel).is_empty());
            assert!(produced_services(kernel).is_empty());
        }
    }

    #[test]
    fn every_binding_names_exactly_one_service_within_the_base() {
        for kernel in VolumetricKernel::ALL {
            for binding in shared_service_bindings(kernel) {
                assert_eq!(
                    binding.service.len(),
                    1,
                    "{kernel:?} binding not a single bit"
                );
                assert!(
                    SharedBaseServices::ALL.contains(binding.service),
                    "{kernel:?} binding outside the shared base"
                );
            }
        }
    }

    #[test]
    fn raymarch_samples_atmosphere_and_gi_and_injects_froxel() {
        let touched = touched_services(VolumetricKernel::Raymarch);
        assert!(touched.contains(SharedBaseServices::ATMOSPHERE_LUT));
        assert!(touched.contains(SharedBaseServices::HYBRID_GI));
        assert!(touched.contains(SharedBaseServices::FROXEL_VOLUME));
        // Only the froxel injection is a write.
        assert_eq!(
            produced_services(VolumetricKernel::Raymarch),
            SharedBaseServices::FROXEL_VOLUME
        );
    }

    #[test]
    fn shadow_march_writes_the_virtual_shadow_map() {
        assert_eq!(
            produced_services(VolumetricKernel::ShadowMarch),
            SharedBaseServices::VSM_SHADOW
        );
        let bindings = shared_service_bindings(VolumetricKernel::ShadowMarch);
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].access, ServiceAccess::ShadowWrite);
    }

    #[test]
    fn upsample_reads_and_writes_temporal_history() {
        let bindings = shared_service_bindings(VolumetricKernel::Upsample);
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].access, ServiceAccess::HistoryReadWrite);
        assert_eq!(
            produced_services(VolumetricKernel::Upsample),
            SharedBaseServices::TEMPORAL_UPSAMPLE
        );
    }

    #[test]
    fn scatter_resolve_only_samples_never_writes() {
        assert!(produced_services(VolumetricKernel::ScatterResolve).is_empty());
        for b in shared_service_bindings(VolumetricKernel::ScatterResolve) {
            assert_eq!(b.access, ServiceAccess::SampleReadOnly);
            assert!(!b.access.is_write());
        }
    }

    #[test]
    fn produced_is_a_subset_of_touched() {
        for kernel in VolumetricKernel::ALL {
            let produced = produced_services(kernel);
            let touched = touched_services(kernel);
            assert!(
                touched.contains(produced),
                "{kernel:?} produces outside touched"
            );
        }
    }

    #[test]
    fn bindings_are_deterministic() {
        for kernel in VolumetricKernel::ALL {
            assert_eq!(
                shared_service_bindings(kernel),
                shared_service_bindings(kernel)
            );
            assert_eq!(touched_services(kernel), touched_services(kernel));
        }
    }
}
