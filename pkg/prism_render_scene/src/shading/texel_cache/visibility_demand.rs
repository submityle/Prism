//! Typed visibility-to-shade demand conversion.
//!
//! The caller must provide a sample reconstructed from the real visibility
//! buffer and the validated per-face chart extent. This module performs only
//! deterministic LOD/page math; it never invents visibility or geometry.

use prism_render_visibility::texture_lod;

use super::{ShadeChannel, ShadePageKey, ShadePageRequest};

/// One real visibility-buffer sample for object/texture-space shading.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadeVisibilitySample {
    /// Stable render-object identity.
    pub object_id: u32,
    /// Stable material identity.
    pub material_id: u32,
    /// Face/primitive identity in the owning geometry.
    pub face_id: u32,
    /// Interpolated chart UV at the covered pixel.
    pub uv: [f32; 2],
    /// Screen-space UV derivative in X.
    pub uv_ddx: [f32; 2],
    /// Screen-space UV derivative in Y.
    pub uv_ddy: [f32; 2],
    /// Fixed-point screen importance from the visibility producer.
    pub screen_importance: u16,
}

/// Fixed chart/physical parameters needed to lower one visibility sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadeDemandConfig {
    /// Chart resolution at mip zero in texels.
    pub chart_extent: [u32; 2],
    /// Logical page edge in texels.
    pub page_size: u32,
    /// Physical atlas bytes consumed by one page.
    pub page_byte_cost: u64,
    /// Frame number used for deterministic request aging.
    pub frame: u64,
    /// Cached view-independent response channel.
    pub channel: ShadeChannel,
    /// Highest mip the chart producer is willing to request.
    pub max_mip: u8,
}

/// Converts one validated visibility sample into one deterministic page request.
pub fn shade_page_request_from_visibility(
    sample: ShadeVisibilitySample,
    config: ShadeDemandConfig,
    material_epoch: u64,
) -> Option<ShadePageRequest> {
    if config.chart_extent.contains(&0)
        || config.page_size == 0
        || config.page_byte_cost == 0
        || !sample
            .uv
            .into_iter()
            .chain(sample.uv_ddx)
            .chain(sample.uv_ddy)
            .all(f32::is_finite)
    {
        return None;
    }
    let face = u16::try_from(sample.face_id).ok()?;
    let raw_lod = texture_lod(
        sample.uv_ddx,
        sample.uv_ddy,
        config.chart_extent[0],
        config.chart_extent[1],
    );
    if !raw_lod.is_finite() {
        return None;
    }
    let mip = (raw_lod.max(0.0).floor() as u32).min(u32::from(config.max_mip));
    let width = (config.chart_extent[0] >> mip).max(1);
    let height = (config.chart_extent[1] >> mip).max(1);
    let pages_x = width.div_ceil(config.page_size);
    let pages_y = height.div_ceil(config.page_size);
    // ShadePageKey stores page coordinates as u16. Reject a chart that cannot
    // be represented losslessly instead of truncating a valid UV into an
    // unrelated page.
    let max_page_count = u32::from(u16::MAX) + 1;
    if pages_x > max_page_count || pages_y > max_page_count {
        return None;
    }
    let page_x = (sample.uv[0].clamp(0.0, 1.0) * width as f32 / config.page_size as f32)
        .floor()
        .clamp(0.0, pages_x.saturating_sub(1) as f32) as u16;
    let page_y = (sample.uv[1].clamp(0.0, 1.0) * height as f32 / config.page_size as f32)
        .floor()
        .clamp(0.0, pages_y.saturating_sub(1) as f32) as u16;
    let mip = u8::try_from(mip).ok()?;
    let priority = (u64::from(sample.screen_importance) << 16) | u64::from(u8::MAX - mip);
    Some(ShadePageRequest {
        key: ShadePageKey {
            object_id: sample.object_id,
            material_id: sample.material_id,
            material_epoch,
            face,
            mip,
            channel: config.channel,
            page_x,
            page_y,
        },
        priority,
        byte_cost: config.page_byte_cost,
        frame: config.frame,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ShadeDemandConfig {
        ShadeDemandConfig {
            chart_extent: [1024, 512],
            page_size: 128,
            page_byte_cost: 65_536,
            frame: 9,
            channel: ShadeChannel::LitDiffuse,
            max_mip: 8,
        }
    }

    #[test]
    fn lowers_real_uv_and_derivatives_to_a_clamped_page() {
        let request = shade_page_request_from_visibility(
            ShadeVisibilitySample {
                object_id: 4,
                material_id: 7,
                face_id: 3,
                uv: [1.0, 0.5],
                uv_ddx: [0.01, 0.0],
                uv_ddy: [0.0, 0.01],
                screen_importance: 12,
            },
            config(),
            11,
        )
        .unwrap();
        assert_eq!(request.key.object_id, 4);
        assert_eq!(request.key.material_epoch, 11);
        assert_eq!(request.key.face, 3);
        assert_eq!(request.key.mip, 3);
        assert_eq!((request.key.page_x, request.key.page_y), (0, 0));
        assert_eq!(request.frame, 9);
        assert!(request.priority > 0);
    }

    #[test]
    fn rejects_non_finite_and_zero_physical_inputs() {
        let mut sample = ShadeVisibilitySample {
            object_id: 1,
            material_id: 2,
            face_id: 0,
            uv: [0.5, 0.5],
            uv_ddx: [0.0, 0.0],
            uv_ddy: [0.0, 0.0],
            screen_importance: 1,
        };
        sample.uv[0] = f32::NAN;
        assert!(shade_page_request_from_visibility(sample, config(), 1).is_none());
        sample.uv[0] = 0.5;
        assert!(shade_page_request_from_visibility(
            sample,
            ShadeDemandConfig {
                page_byte_cost: 0,
                ..config()
            },
            1
        )
        .is_none());
    }

    #[test]
    fn rejects_page_grids_that_do_not_fit_the_key_coordinate_abi() {
        let result = shade_page_request_from_visibility(
            ShadeVisibilitySample {
                object_id: 1,
                material_id: 2,
                face_id: 0,
                uv: [0.5, 0.5],
                uv_ddx: [0.0, 0.0],
                uv_ddy: [0.0, 0.0],
                screen_importance: 1,
            },
            ShadeDemandConfig {
                chart_extent: [u32::MAX, 128],
                page_size: 1,
                ..config()
            },
            1,
        );
        assert!(result.is_none());
    }
}
