//! Trace-backend selection with capability-driven fallback.
//!
//! Ray queries can be served by three backends of decreasing fidelity:
//! `HardwareRayQuery` (dedicated ray-tracing cores), `SoftwareBvh` (a compute
//! traversal over a `CPU`/`GPU` `BVH`), and `ScreenSpace` (depth-buffer marching
//! that only sees on-screen geometry). Selection walks that chain top-down and
//! returns the first backend whose required capabilities are present.
//!
//! The `GPU` traversal kernels are pending the GPU backend; this module encodes
//! only the deterministic, `CPU`-verifiable selection policy.

/// Backend used to resolve ray queries, ordered by fidelity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceBackend {
    /// Depth-buffer marching; only on-screen surfaces are visible.
    ScreenSpace,
    /// Compute traversal over a bounding-volume hierarchy (`BVH`).
    SoftwareBvh,
    /// Dedicated hardware ray-query acceleration.
    HardwareRayQuery,
}

impl TraceBackend {
    /// Relative fidelity rank; higher is better.
    #[must_use]
    pub const fn fidelity_rank(self) -> u8 {
        match self {
            Self::ScreenSpace => 0,
            Self::SoftwareBvh => 1,
            Self::HardwareRayQuery => 2,
        }
    }

    /// True when off-screen geometry can be intersected (everything but the
    /// screen-space backend).
    #[must_use]
    pub const fn resolves_offscreen(self) -> bool {
        !matches!(self, Self::ScreenSpace)
    }
}

/// Hardware/driver capabilities that gate the backends.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BackendCapabilities {
    /// Dedicated ray-query hardware is present and enabled.
    pub hardware_ray_query: bool,
    /// A compute-built `BVH` is resident and traversable.
    pub software_bvh: bool,
    /// A depth buffer suitable for screen-space marching exists.
    pub depth_buffer: bool,
}

impl BackendCapabilities {
    /// Capability set for a fully featured ray-tracing `GPU`.
    #[must_use]
    pub const fn full() -> Self {
        Self {
            hardware_ray_query: true,
            software_bvh: true,
            depth_buffer: true,
        }
    }

    /// True when `backend` can run under these capabilities.
    #[must_use]
    pub const fn supports(self, backend: TraceBackend) -> bool {
        match backend {
            TraceBackend::HardwareRayQuery => self.hardware_ray_query,
            TraceBackend::SoftwareBvh => self.software_bvh,
            TraceBackend::ScreenSpace => self.depth_buffer,
        }
    }
}

/// What a ray-query workload demands from its backend.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TraceRequirements {
    /// The workload needs to intersect geometry outside the view frustum
    /// (e.g. reflections of off-screen surfaces), forbidding `ScreenSpace`.
    pub needs_offscreen: bool,
    /// A hard floor on backend fidelity; backends below this rank are skipped.
    pub min_fidelity: u8,
}

/// Reason a backend was rejected during selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendRejection {
    /// The capability set does not expose this backend.
    Unavailable(TraceBackend),
    /// The backend cannot satisfy the off-screen requirement.
    OffscreenUnsupported(TraceBackend),
    /// The backend's fidelity is below the requested floor.
    BelowMinFidelity(TraceBackend),
}

/// Outcome of a backend selection, including the rejected candidates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendSelection {
    /// The chosen backend, or `None` when nothing qualified.
    pub selected: Option<TraceBackend>,
    /// Backends considered and rejected, in evaluation order.
    pub rejected: Vec<BackendRejection>,
}

/// Preference order walked during selection: best fidelity first.
const PREFERENCE_ORDER: [TraceBackend; 3] = [
    TraceBackend::HardwareRayQuery,
    TraceBackend::SoftwareBvh,
    TraceBackend::ScreenSpace,
];

/// Selects the highest-fidelity backend satisfying `requirements` under `caps`.
///
/// The chain is walked from `HardwareRayQuery` down to `ScreenSpace`; the first
/// candidate that is available, meets the off-screen requirement, and clears the
/// fidelity floor wins. Every earlier rejection is recorded for diagnostics.
#[must_use]
pub fn select_backend(
    caps: BackendCapabilities,
    requirements: TraceRequirements,
) -> BackendSelection {
    let mut rejected = Vec::new();
    for backend in PREFERENCE_ORDER {
        if backend.fidelity_rank() < requirements.min_fidelity {
            rejected.push(BackendRejection::BelowMinFidelity(backend));
            continue;
        }
        if !caps.supports(backend) {
            rejected.push(BackendRejection::Unavailable(backend));
            continue;
        }
        if requirements.needs_offscreen && !backend.resolves_offscreen() {
            rejected.push(BackendRejection::OffscreenUnsupported(backend));
            continue;
        }
        return BackendSelection {
            selected: Some(backend),
            rejected,
        };
    }
    BackendSelection {
        selected: None,
        rejected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_hardware_when_available() {
        let sel = select_backend(BackendCapabilities::full(), TraceRequirements::default());
        assert_eq!(sel.selected, Some(TraceBackend::HardwareRayQuery));
        assert!(sel.rejected.is_empty());
    }

    #[test]
    fn falls_back_to_software_bvh() {
        let caps = BackendCapabilities {
            hardware_ray_query: false,
            software_bvh: true,
            depth_buffer: true,
        };
        let sel = select_backend(caps, TraceRequirements::default());
        assert_eq!(sel.selected, Some(TraceBackend::SoftwareBvh));
        assert_eq!(
            sel.rejected,
            vec![BackendRejection::Unavailable(
                TraceBackend::HardwareRayQuery
            )]
        );
    }

    #[test]
    fn falls_back_to_screen_space() {
        let caps = BackendCapabilities {
            hardware_ray_query: false,
            software_bvh: false,
            depth_buffer: true,
        };
        let sel = select_backend(caps, TraceRequirements::default());
        assert_eq!(sel.selected, Some(TraceBackend::ScreenSpace));
        assert_eq!(sel.rejected.len(), 2);
    }

    #[test]
    fn offscreen_requirement_skips_screen_space() {
        let caps = BackendCapabilities {
            hardware_ray_query: false,
            software_bvh: false,
            depth_buffer: true,
        };
        let req = TraceRequirements {
            needs_offscreen: true,
            min_fidelity: 0,
        };
        let sel = select_backend(caps, req);
        assert_eq!(sel.selected, None);
        assert!(sel
            .rejected
            .contains(&BackendRejection::OffscreenUnsupported(
                TraceBackend::ScreenSpace
            )));
    }

    #[test]
    fn min_fidelity_floor_rejects_low_backends() {
        let caps = BackendCapabilities::full();
        let req = TraceRequirements {
            needs_offscreen: false,
            min_fidelity: TraceBackend::SoftwareBvh.fidelity_rank(),
        };
        let sel = select_backend(caps, req);
        assert_eq!(sel.selected, Some(TraceBackend::HardwareRayQuery));

        let caps_soft = BackendCapabilities {
            hardware_ray_query: false,
            software_bvh: false,
            depth_buffer: true,
        };
        let sel_soft = select_backend(caps_soft, req);
        assert_eq!(sel_soft.selected, None);
        assert!(sel_soft
            .rejected
            .contains(&BackendRejection::BelowMinFidelity(
                TraceBackend::ScreenSpace
            )));
    }

    #[test]
    fn no_capabilities_selects_nothing() {
        let sel = select_backend(BackendCapabilities::default(), TraceRequirements::default());
        assert_eq!(sel.selected, None);
        assert_eq!(sel.rejected.len(), 3);
    }

    #[test]
    fn fidelity_and_offscreen_flags_are_consistent() {
        assert!(
            TraceBackend::HardwareRayQuery.fidelity_rank()
                > TraceBackend::SoftwareBvh.fidelity_rank()
        );
        assert!(
            TraceBackend::SoftwareBvh.fidelity_rank() > TraceBackend::ScreenSpace.fidelity_rank()
        );
        assert!(TraceBackend::HardwareRayQuery.resolves_offscreen());
        assert!(TraceBackend::SoftwareBvh.resolves_offscreen());
        assert!(!TraceBackend::ScreenSpace.resolves_offscreen());
    }
}
