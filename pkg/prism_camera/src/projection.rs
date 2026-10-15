use core::f32::consts::PI;

use prism_math::{Mat4, Vec3A};

use crate::{CameraError, CameraErrorCode};

/// Axis that remains stable when the render target aspect changes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FovAxis {
    Vertical,
    Horizontal,
    Auto,
}

/// Perspective projection intent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerspectiveProjection {
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: Option<f32>,
    pub fov_axis: FovAxis,
    pub reverse_z: bool,
}

impl Default for PerspectiveProjection {
    fn default() -> Self {
        Self {
            fov_y: PI / 3.0,
            aspect: 16.0 / 9.0,
            near: 0.05,
            far: None,
            fov_axis: FovAxis::Vertical,
            reverse_z: true,
        }
    }
}

/// Orthographic projection intent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrthographicProjection {
    pub width: f32,
    pub height: f32,
    pub near: f32,
    pub far: f32,
    pub reverse_z: bool,
}

impl Default for OrthographicProjection {
    fn default() -> Self {
        Self {
            width: 16.0,
            height: 9.0,
            near: 0.0,
            far: 1000.0,
            reverse_z: true,
        }
    }
}

/// Projection family used by a camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    Perspective(PerspectiveProjection),
    Orthographic(OrthographicProjection),
    Custom {
        clip_from_view: Mat4,
        near: f32,
        far: Option<f32>,
        reverse_z: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectionError {
    InvalidFov,
    InvalidAspect,
    InvalidNear,
    InvalidFar,
    NonFiniteMatrix,
}

impl Projection {
    pub fn validate(self) -> Result<(), ProjectionError> {
        match self {
            Self::Perspective(p) => {
                if !p.fov_y.is_finite() || p.fov_y <= 0.0 || p.fov_y >= PI {
                    return Err(ProjectionError::InvalidFov);
                }
                if !p.aspect.is_finite() || p.aspect <= 0.0 {
                    return Err(ProjectionError::InvalidAspect);
                }
                if !p.near.is_finite() || p.near <= 0.0 {
                    return Err(ProjectionError::InvalidNear);
                }
                if let Some(far) = p.far {
                    if !far.is_finite() || far <= p.near {
                        return Err(ProjectionError::InvalidFar);
                    }
                }
            }
            Self::Orthographic(p) => {
                if !p.width.is_finite() || !p.height.is_finite() || p.width <= 0.0 || p.height <= 0.0 {
                    return Err(ProjectionError::InvalidAspect);
                }
                if !p.near.is_finite() || !p.far.is_finite() || p.far <= p.near {
                    return Err(ProjectionError::InvalidFar);
                }
            }
            Self::Custom { clip_from_view, near, far, .. } => {
                if !matrix_is_finite(clip_from_view) {
                    return Err(ProjectionError::NonFiniteMatrix);
                }
                if !near.is_finite() || near <= 0.0 {
                    return Err(ProjectionError::InvalidNear);
                }
                if let Some(far) = far {
                    if !far.is_finite() || far <= near {
                        return Err(ProjectionError::InvalidFar);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn clip_from_view(self) -> Result<Mat4, CameraError> {
        self.validate().map_err(|e| match e {
            ProjectionError::InvalidFov => CameraError::new(CameraErrorCode::InvalidProjection, "fov_y", 0.0),
            ProjectionError::InvalidAspect => CameraError::new(CameraErrorCode::InvalidProjection, "aspect", 0.0),
            ProjectionError::InvalidNear => CameraError::new(CameraErrorCode::InvalidProjection, "near", 0.0),
            ProjectionError::InvalidFar => CameraError::new(CameraErrorCode::InvalidProjection, "far", 0.0),
            ProjectionError::NonFiniteMatrix => CameraError::new(CameraErrorCode::NonFiniteDerivedView, "projection", 0.0),
        })?;
        Ok(match self {
            Self::Perspective(p) if p.reverse_z && p.far.is_none() =>
                prism_math::perspective_infinite_reverse_z_rh(p.fov_y, p.aspect, p.near),
            Self::Perspective(p) if p.reverse_z =>
                prism_math::perspective_reverse_z_rh(p.fov_y, p.aspect, p.near, p.far.unwrap()),
            Self::Perspective(p) =>
                prism_math::perspective_rh(p.fov_y, p.aspect, p.near, p.far.unwrap()),
            Self::Orthographic(p) => {
                let mut m = prism_math::orthographic_rh(
                    -p.width * 0.5, p.width * 0.5, -p.height * 0.5, p.height * 0.5, p.near, p.far,
                );
                if p.reverse_z {
                    m = prism_math::orthographic_rh(
                        -p.width * 0.5, p.width * 0.5, -p.height * 0.5, p.height * 0.5, p.far, p.near,
                    );
                }
                m
            }
            Self::Custom { clip_from_view, .. } => clip_from_view,
        })
    }

    pub fn near(self) -> f32 {
        match self {
            Self::Perspective(p) => p.near,
            Self::Orthographic(p) => p.near,
            Self::Custom { near, .. } => near,
        }
    }

    pub fn far(self) -> Option<f32> {
        match self {
            Self::Perspective(p) => p.far,
            Self::Orthographic(p) => Some(p.far),
            Self::Custom { far, .. } => far,
        }
    }

    pub fn reverse_z(self) -> bool {
        match self {
            Self::Perspective(p) => p.reverse_z,
            Self::Orthographic(p) => p.reverse_z,
            Self::Custom { reverse_z, .. } => reverse_z,
        }
    }

    pub fn frustum_corners(self, z_near: f32, z_far: f32) -> Result<[Vec3A; 8], CameraError> {
        let p = match self {
            Self::Perspective(p) => p,
            _ => return Err(CameraError::new(CameraErrorCode::InvalidProjection, "frustum_corners", 0.0)),
        };
        let tan_half = libm::tanf(p.fov_y * 0.5);
        let nh = tan_half * z_near;
        let nw = nh * p.aspect;
        let fh = tan_half * z_far;
        let fw = fh * p.aspect;
        Ok([
            Vec3A::new(-nw, -nh, -z_near), Vec3A::new(nw, -nh, -z_near),
            Vec3A::new(nw, nh, -z_near), Vec3A::new(-nw, nh, -z_near),
            Vec3A::new(-fw, -fh, -z_far), Vec3A::new(fw, -fh, -z_far),
            Vec3A::new(fw, fh, -z_far), Vec3A::new(-fw, fh, -z_far),
        ])
    }
}

fn matrix_is_finite(matrix: Mat4) -> bool {
    matrix.x_axis.is_finite()
        && matrix.y_axis.is_finite()
        && matrix.z_axis.is_finite()
        && matrix.w_axis.is_finite()
}
