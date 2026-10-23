//! Camera-local cinematic parameter normalization used by the HDR compositor.

use prism_camera_ecs::PublishedCameraRenderParams;

/// Converts authored exposure into scene-linear compensation stops.
///
/// EV100 is the logarithmic exposure value of camera settings, not the renderer
/// multiplier. Prism defines that multiplier as 1 / (1.2 * 2^EV100), so the
/// equivalent multiplier in stops is -EV100 - log2(1.2). An optional PostStack
/// exposure node adds authored compensation in EV. Invalid values fail neutral.
pub(super) fn camera_exposure_stops(params: &PublishedCameraRenderParams) -> f32 {
    let base = params
        .exposure
        .map_or(0.0, |exposure| -exposure.ev100() - 0.263_034_4);
    let post = params
        .post_stack
        .as_ref()
        .and_then(|stack| {
            stack.nodes.iter().find_map(|node| match *node {
                prism_camera::PostNode::Exposure(settings) => Some(settings.compensation_ev),
                _ => None,
            })
        })
        .unwrap_or(0.0);
    if base.is_finite() && post.is_finite() {
        (base + post).clamp(-24.0, 24.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs::prelude::Entity;

    fn params(
        exposure: prism_camera::Exposure,
        post_stack: Option<prism_camera::PostStack>,
    ) -> PublishedCameraRenderParams {
        PublishedCameraRenderParams {
            entity: Entity::from_bits(1_u64 << 32).unwrap(),
            focal_length_mm: Some(50.0),
            exposure: Some(exposure),
            shutter: None,
            post_stack,
        }
    }

    #[test]
    fn ev100_maps_to_the_photometric_multiplier_in_stops() {
        for ev100 in [-12.0, -2.0, 0.0, 8.0, 18.0] {
            let exposure = prism_camera::Exposure::ManualEv100 { ev100 };
            let stops = camera_exposure_stops(&params(exposure, None));
            let actual = 2.0_f32.powf(stops);
            let expected = exposure.multiplier();
            assert!((actual - expected).abs() <= expected.abs() * 2.0e-6);
        }
    }

    #[test]
    fn post_stack_exposure_is_applied_in_camera_linear_hdr_domain() {
        let mut stack = prism_camera::PostStack::default();
        stack
            .push(prism_camera::PostNode::Exposure(
                prism_camera::ExposureSettings {
                    compensation_ev: 2.0,
                    domain: prism_camera::PassDomain::PostUpscale,
                },
            ))
            .unwrap();
        stack.normalize().unwrap();
        let input = params(
            prism_camera::Exposure::ManualEv100 { ev100: 0.0 },
            Some(stack),
        );
        assert!((camera_exposure_stops(&input) - (2.0 - 0.263_034_4)).abs() < 1.0e-6);
    }

    #[test]
    fn invalid_exposure_fails_neutral_and_large_compensation_is_bounded() {
        let invalid = params(
            prism_camera::Exposure::ManualEv100 { ev100: f32::NAN },
            None,
        );
        assert_eq!(camera_exposure_stops(&invalid), 0.0);
        let mut stack = prism_camera::PostStack::default();
        stack
            .push(prism_camera::PostNode::Exposure(
                prism_camera::ExposureSettings {
                    compensation_ev: 100.0,
                    domain: prism_camera::PassDomain::PostUpscale,
                },
            ))
            .unwrap();
        stack.normalize().unwrap();
        assert_eq!(
            camera_exposure_stops(&params(
                prism_camera::Exposure::ManualEv100 { ev100: 0.0 },
                Some(stack),
            )),
            24.0
        );
    }
}
