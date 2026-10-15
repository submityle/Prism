//! Engine-neutral CPU contract mirrored by the WGSL kernel.

/// Screen-space error quantisation uses 1/256 pixel units.
pub const ERROR_QUANTUM: f32 = 256.0;

/// Error thresholds for LOD 0, 1, and 2. LOD 3 is the coarsest result.
pub const LOD_THRESHOLDS: [f32; 3] = [64.0, 16.0, 4.0];

/// A normalised plane in the form `dot(normal, position) + distance >= 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrustumPlane {
    /// Inward-facing plane normal.
    pub normal: [f32; 3],
    /// Signed plane distance.
    pub distance: f32,
}

impl FrustumPlane {
    /// Creates a plane without silently normalising it.
    #[must_use]
    pub const fn new(normal: [f32; 3], distance: f32) -> Self {
        Self { normal, distance }
    }

    /// Evaluates the signed distance expression used by CPU and WGSL.
    #[must_use]
    pub fn signed_distance(self, position: [f32; 3]) -> f32 {
        self.normal[0] * position[0]
            + self.normal[1] * position[1]
            + self.normal[2] * position[2]
            + self.distance
    }
}

/// One GPU-resident world proxy instance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProxyInstance {
    /// Stable content identity carried through the visibility result.
    pub proxy_id: u32,
    /// Stable cluster identity for higher-level batching.
    pub cluster_id: u32,
    /// Sector-local centre in the camera-relative render coordinate space.
    pub position: [f32; 3],
    /// Bounding-sphere radius in the same space as `position`.
    pub radius: f32,
    /// Geometric error in world units.
    pub geometric_error: f32,
    /// Per-proxy maximum visibility distance.
    pub max_distance: f32,
    /// Reserved flags carried by the SoA contract.
    pub flags: u32,
}

impl ProxyInstance {
    /// Constructs a proxy with no flags.
    #[must_use]
    pub const fn new(
        proxy_id: u32,
        cluster_id: u32,
        position: [f32; 3],
        radius: f32,
        geometric_error: f32,
        max_distance: f32,
    ) -> Self {
        Self {
            proxy_id,
            cluster_id,
            position,
            radius,
            geometric_error,
            max_distance,
            flags: 0,
        }
    }

    /// Returns whether every numeric field can be represented by the GPU ABI.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.position.iter().all(|value| value.is_finite())
            && self.radius.is_finite()
            && self.geometric_error.is_finite()
            && self.max_distance.is_finite()
            && self.radius >= 0.0
            && self.geometric_error >= 0.0
            && self.max_distance >= 0.0
    }
}

/// One spatial aggregate used by CPU batching and higher-level HLOD systems.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cluster {
    /// Stable cluster identity.
    pub cluster_id: u32,
    /// First proxy index in the cluster's deterministic input order.
    pub first_proxy: u32,
    /// Number of proxies in the cluster.
    pub proxy_count: u32,
    /// Cluster bounding-sphere centre.
    pub center: [f32; 3],
    /// Cluster bounding-sphere radius.
    pub radius: f32,
    /// Conservative aggregate geometric error.
    pub geometric_error: f32,
}

impl Cluster {
    /// Returns whether the cluster is safe to upload.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.center.iter().all(|value| value.is_finite())
            && self.radius.is_finite()
            && self.geometric_error.is_finite()
            && self.radius >= 0.0
            && self.geometric_error >= 0.0
    }
}

/// Per-dispatch camera and projection constants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisibilityInput {
    /// Camera origin in the proxy coordinate space.
    pub camera_position: [f32; 3],
    /// Six inward-facing frustum planes.
    pub frustum: [FrustumPlane; 6],
    /// Vertical projection scale, normally `viewport_height / (2*tan(fov/2))`.
    pub projection_scale: f32,
    /// Global range limit applied before the per-proxy limit.
    pub max_distance: f32,
}

impl VisibilityInput {
    /// Creates a contract from camera position, frustum, viewport height, and
    /// a caller-provided projection scale.
    ///
    /// `projection_scale` is supplied directly so authoring and runtime code
    /// can use one already-quantised projection value and the CPU/GPU path
    /// does not repeat trigonometric work.
    #[must_use]
    pub fn from_projection_scale(
        camera_position: [f32; 3],
        frustum: [FrustumPlane; 6],
        projection_scale: f32,
        max_distance: f32,
    ) -> Self {
        Self {
            camera_position,
            frustum,
            projection_scale,
            max_distance,
        }
    }

    /// Creates a perspective contract with a deterministic `f32` projection
    /// scale and a finite global range.
    #[must_use]
    pub fn new(
        camera_position: [f32; 3],
        frustum: [FrustumPlane; 6],
        viewport_height: u32,
        max_distance: f32,
    ) -> Self {
        let projection_scale = viewport_height as f32 / 2.0;
        Self::from_projection_scale(camera_position, frustum, projection_scale, max_distance)
    }

    /// Evaluates one proxy using the exact scalar sequence mirrored in WGSL.
    #[must_use]
    pub fn evaluate(self, proxy: &ProxyInstance) -> VisibilityOutput {
        let mut visible = true;
        for plane in self.frustum {
            if plane.signed_distance(proxy.position) < -proxy.radius {
                visible = false;
                break;
            }
        }

        let dx = proxy.position[0] - self.camera_position[0];
        let dy = proxy.position[1] - self.camera_position[1];
        let dz = proxy.position[2] - self.camera_position[2];
        let distance_squared = dx * dx + dy * dy + dz * dz;
        let global_limit_squared = self.max_distance * self.max_distance;
        let proxy_limit_squared = proxy.max_distance * proxy.max_distance;
        if distance_squared > global_limit_squared || distance_squared > proxy_limit_squared {
            visible = false;
        }

        if !visible {
            return VisibilityOutput::culled(proxy.proxy_id);
        }

        let distance = distance_squared.max(1.0e-8).sqrt();
        let screen_error = proxy.geometric_error * self.projection_scale / distance;
        VisibilityOutput::visible(proxy.proxy_id, screen_error)
    }
}

/// CPU result and canonical four-word GPU readback record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisibilityOutput {
    /// Stable proxy identity.
    pub proxy_id: u32,
    /// Whether the proxy survives range and frustum tests.
    pub visible: bool,
    /// Selected LOD, where zero is the highest detail.
    pub lod: u32,
    /// Unquantised screen-space geometric error in pixels.
    pub screen_error: f32,
    /// Quantised error used for exact CPU/GPU parity checks.
    pub error_quantized: u32,
}

impl VisibilityOutput {
    fn culled(proxy_id: u32) -> Self {
        Self {
            proxy_id,
            visible: false,
            lod: 0,
            screen_error: 0.0,
            error_quantized: 0,
        }
    }

    fn visible(proxy_id: u32, screen_error: f32) -> Self {
        Self {
            proxy_id,
            visible: true,
            lod: lod_for_error(screen_error),
            screen_error,
            error_quantized: quantise_error(screen_error),
        }
    }

    /// Returns the exact storage words written by the WGSL kernel.
    #[must_use]
    pub const fn words(self) -> [u32; 4] {
        [
            if self.visible { 1 } else { 0 },
            self.lod,
            self.error_quantized,
            self.proxy_id,
        ]
    }
}

/// Maps the error to one of four deterministic detail levels.
#[must_use]
pub fn lod_for_error(screen_error: f32) -> u32 {
    if screen_error >= LOD_THRESHOLDS[0] {
        0
    } else if screen_error >= LOD_THRESHOLDS[1] {
        1
    } else if screen_error >= LOD_THRESHOLDS[2] {
        2
    } else {
        3
    }
}

/// Quantises a non-negative error to the contract's fixed-point readback word.
#[must_use]
pub fn quantise_error(screen_error: f32) -> u32 {
    let scaled = (screen_error * ERROR_QUANTUM + 0.5).floor();
    scaled.clamp(0.0, u16::MAX as f32) as u32
}
