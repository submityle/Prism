use prism_math::{Ray3, Vec2, Vec3, Vec4};

use crate::{ViewData, ViewportRect};

/// Errors are explicit so callers never consume a fabricated ray or world point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PickError {
    OutsideViewport,
    InvalidDepth,
    NonInvertibleView,
}

fn viewport_local(viewport: ViewportRect, pixel: Vec2) -> Result<Vec2, PickError> {
    let min = Vec2::new(viewport.origin[0] as f32, viewport.origin[1] as f32);
    let size = Vec2::new(viewport.size[0] as f32, viewport.size[1] as f32);
    if pixel.x < min.x || pixel.y < min.y || pixel.x >= min.x + size.x || pixel.y >= min.y + size.y
    {
        return Err(PickError::OutsideViewport);
    }
    Ok(Vec2::new(
        (pixel.x - min.x) / size.x,
        (pixel.y - min.y) / size.y,
    ))
}

impl ViewData {
    pub fn viewport_to_world_ray(&self, pixel: Vec2) -> Result<Ray3, PickError> {
        let local = viewport_local(self.viewport_px, pixel)?;
        let ndc = Vec3::new(local.x * 2.0 - 1.0, 1.0 - local.y * 2.0, 0.0);
        let inverse = self.clip_from_world_unjittered.inverse();
        let near = inverse.mul_vec4(Vec4::new(ndc.x, ndc.y, 1.0, 1.0));
        let far = inverse.mul_vec4(Vec4::new(ndc.x, ndc.y, 0.0, 1.0));
        if !near.is_finite() || !far.is_finite() || near.w.abs() <= 1.0e-8 {
            return Err(PickError::NonInvertibleView);
        }
        let origin = Vec3::new(near.x / near.w, near.y / near.w, near.z / near.w);
        let direction = if far.w.abs() <= 1.0e-8 {
            Vec3::new(far.x, far.y, far.z).normalize()
        } else {
            let point = Vec3::new(far.x / far.w, far.y / far.w, far.z / far.w);
            (point - origin).normalize()
        };
        if !direction.is_finite() || direction.length_squared() <= 1.0e-12 {
            return Err(PickError::NonInvertibleView);
        }
        Ok(Ray3::new_normalized(origin, direction))
    }

    pub fn viewport_to_world(&self, pixel: Vec2, ndc_depth: f32) -> Result<Vec3, PickError> {
        let local = viewport_local(self.viewport_px, pixel)?;
        if !ndc_depth.is_finite() || !(0.0..=1.0).contains(&ndc_depth) {
            return Err(PickError::InvalidDepth);
        }
        let inverse = self.clip_from_world_unjittered.inverse();
        let point = inverse.mul_vec4(Vec4::new(
            local.x * 2.0 - 1.0,
            1.0 - local.y * 2.0,
            ndc_depth,
            1.0,
        ));
        if !point.is_finite() || point.w.abs() <= 1.0e-8 {
            return Err(PickError::NonInvertibleView);
        }
        Ok(Vec3::new(
            point.x / point.w,
            point.y / point.w,
            point.z / point.w,
        ))
    }

    pub fn world_to_viewport(&self, world: Vec3) -> Result<Option<Vec2>, PickError> {
        let clip = self.clip_from_world_unjittered.mul_vec4(world.extend(1.0));
        if !clip.is_finite() || clip.w.abs() <= 1.0e-8 {
            return Err(PickError::NonInvertibleView);
        }
        if clip.w <= 0.0 {
            return Ok(None);
        }
        let ndc = Vec2::new(clip.x / clip.w, clip.y / clip.w);
        if !ndc.x.is_finite() || !ndc.y.is_finite() {
            return Ok(None);
        }
        Ok(Some(Vec2::new(
            self.viewport_px.origin[0] as f32
                + (ndc.x * 0.5 + 0.5) * self.viewport_px.size[0] as f32,
            self.viewport_px.origin[1] as f32
                + (0.5 - ndc.y * 0.5) * self.viewport_px.size[1] as f32,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{derive_view, CameraIntent, CameraPose, ViewHistory};

    #[test]
    fn ray_and_projection_round_trip() {
        let view = derive_view(
            CameraIntent::default(),
            CameraPose::default(),
            0,
            [100, 100],
            &mut ViewHistory::default(),
        )
        .unwrap();
        let ray = view.viewport_to_world_ray(Vec2::new(50.0, 50.0)).unwrap();
        assert!(ray.direction.z < 0.0);
        let pixel = view
            .world_to_viewport(Vec3::new(0.0, 0.0, -1.0))
            .unwrap()
            .unwrap();
        assert!((pixel.x - 50.0).abs() < 0.1 && (pixel.y - 50.0).abs() < 0.1);
    }

    #[test]
    fn invalid_pick_inputs_are_rejected() {
        let view = derive_view(
            CameraIntent::default(),
            CameraPose::default(),
            0,
            [100, 100],
            &mut ViewHistory::default(),
        )
        .unwrap();
        assert_eq!(
            view.viewport_to_world_ray(Vec2::new(-1.0, 50.0)),
            Err(PickError::OutsideViewport)
        );
        assert_eq!(
            view.viewport_to_world(Vec2::new(50.0, 50.0), f32::NAN),
            Err(PickError::InvalidDepth)
        );
    }
}
