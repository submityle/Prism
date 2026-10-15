use prism_math::Vec2;

use crate::{CameraError, CameraErrorCode};

/// Render destination identity. Resource creation remains owned by the window
/// and renderer crates; this value is only a stable routing key.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RenderTarget {
    PrimaryWindow,
    Window(u64),
    Texture(u64),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_viewport_resolves_to_target_pixels() {
        let rect = Viewport::FULL.resolve([1920, 1080]).unwrap();
        assert_eq!(rect.origin, [0, 0]);
        assert_eq!(rect.size, [1920, 1080]);
    }

    #[test]
    fn invalid_viewport_is_rejected() {
        let viewport = Viewport::new(Vec2::ZERO, Vec2::new(0.0, 1.0));
        assert_eq!(
            viewport.resolve([100, 100]).unwrap_err().code,
            CameraErrorCode::InvalidViewport
        );
    }
}

/// How a view affects the target before drawing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClearMode {
    Inherit,
    ColorAndDepth,
    DepthOnly,
    None,
}

/// Logical viewport in target pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub origin: Vec2,
    pub size: Vec2,
}

/// Validated integer viewport used by picking and GPU upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ViewportRect {
    pub origin: [u32; 2],
    pub size: [u32; 2],
}

impl Viewport {
    pub const FULL: Self = Self {
        origin: Vec2::ZERO,
        size: Vec2::ONE,
    };

    pub const fn new(origin: Vec2, size: Vec2) -> Self {
        Self { origin, size }
    }

    /// Builds a viewport expressed directly in physical target pixels.
    pub const fn from_pixels(origin: [u32; 2], size: [u32; 2]) -> Self {
        Self {
            origin: Vec2::new(origin[0] as f32, origin[1] as f32),
            size: Vec2::new(size[0] as f32, size[1] as f32),
        }
    }

    pub fn resolve(self, target: [u32; 2]) -> Result<ViewportRect, CameraError> {
        if !self.origin.is_finite()
            || !self.size.is_finite()
            || self.size.x <= 0.0
            || self.size.y <= 0.0
        {
            return Err(CameraError::new(
                CameraErrorCode::InvalidViewport,
                "viewport",
                self.size.x,
            ));
        }
        let normalized = self.origin.x <= 1.0
            && self.origin.y <= 1.0
            && self.size.x <= 1.0
            && self.size.y <= 1.0;
        let (origin, size) = if normalized {
            (
                Vec2::new(
                    self.origin.x * target[0] as f32,
                    self.origin.y * target[1] as f32,
                ),
                Vec2::new(
                    self.size.x * target[0] as f32,
                    self.size.y * target[1] as f32,
                ),
            )
        } else {
            (self.origin, self.size)
        };
        let x0 = origin.x.max(0.0).floor() as u32;
        let y0 = origin.y.max(0.0).floor() as u32;
        let x1 = (origin.x + size.x).min(target[0] as f32).ceil() as u32;
        let y1 = (origin.y + size.y).min(target[1] as f32).ceil() as u32;
        if x1 <= x0 || y1 <= y0 || x0 >= target[0] || y0 >= target[1] {
            return Err(CameraError::new(
                CameraErrorCode::InvalidViewport,
                "viewport",
                0.0,
            ));
        }
        Ok(ViewportRect {
            origin: [x0, y0],
            size: [x1 - x0, y1 - y0],
        })
    }
}
