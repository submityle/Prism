use alloc::vec::Vec;

/// Which side of a temporal upscaler consumes a post-process node.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PassDomain {
    PreUpscale,
    PostUpscale,
}

/// Fixed physical order used by the renderer.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PostStage {
    MotionBlur,
    DepthOfField,
    Bloom,
    Exposure,
    Tonemap,
    ColorGrade,
    Stylize,
    AntiAliasing,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureSettings {
    pub compensation_ev: f32,
    pub domain: PassDomain,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BloomSettings {
    pub threshold: f32,
    pub intensity: f32,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DofSettings {
    pub focus_distance_m: f32,
    pub max_coc_mm: f32,
    pub aperture_blades: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionBlurSettings {
    pub shutter_angle: f32,
    pub max_velocity_pixels: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tonemapper {
    Aces,
    AgX,
    Reinhard,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorGradeSettings {
    pub exposure_ev: f32,
    pub contrast: f32,
    pub saturation: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PostNode {
    Exposure(ExposureSettings),
    Bloom(BloomSettings),
    DepthOfField(DofSettings),
    MotionBlur(MotionBlurSettings),
    ChromaticAberration(f32),
    Vignette(f32),
    FilmGrain(f32),
    Tonemap(Tonemapper),
    ColorGrade(ColorGradeSettings),
    Fxaa,
    Smaa(u8),
    Raw {
        id: u32,
        stage: PostStage,
        domain: PassDomain,
    },
}

impl PostNode {
    pub const fn stage(self) -> PostStage {
        match self {
            Self::MotionBlur(_) => PostStage::MotionBlur,
            Self::DepthOfField(_) => PostStage::DepthOfField,
            Self::Bloom(_) => PostStage::Bloom,
            Self::Exposure(_) => PostStage::Exposure,
            Self::Tonemap(_) => PostStage::Tonemap,
            Self::ColorGrade(_) => PostStage::ColorGrade,
            Self::ChromaticAberration(_) | Self::Vignette(_) | Self::FilmGrain(_) => {
                PostStage::Stylize
            }
            Self::Fxaa | Self::Smaa(_) => PostStage::AntiAliasing,
            Self::Raw { stage, .. } => stage,
        }
    }

    pub const fn domain(self) -> PassDomain {
        match self {
            Self::MotionBlur(_) | Self::DepthOfField(_) | Self::Bloom(_) => PassDomain::PreUpscale,
            Self::Raw { domain, .. } => domain,
            _ => PassDomain::PostUpscale,
        }
    }

    fn exclusive_key(self) -> Option<u64> {
        match self {
            Self::Exposure(_) => Some(0),
            Self::Tonemap(_) => Some(1),
            Self::ColorGrade(_) => Some(2),
            Self::Fxaa | Self::Smaa(_) => Some(3),
            Self::Raw { id, .. } => Some(0x1_0000_0000 | id as u64),
            _ => None,
        }
    }

    fn validate(self) -> bool {
        match self {
            Self::Exposure(settings) => settings.compensation_ev.is_finite(),
            Self::Bloom(settings) => {
                settings.threshold.is_finite()
                    && settings.intensity.is_finite()
                    && settings.radius.is_finite()
                    && settings.threshold >= 0.0
                    && settings.intensity >= 0.0
                    && settings.radius >= 0.0
            }
            Self::DepthOfField(settings) => {
                settings.focus_distance_m.is_finite()
                    && settings.max_coc_mm.is_finite()
                    && settings.focus_distance_m > 0.0
                    && settings.max_coc_mm >= 0.0
                    && settings.aperture_blades >= 3
            }
            Self::MotionBlur(settings) => {
                settings.shutter_angle.is_finite()
                    && settings.max_velocity_pixels.is_finite()
                    && settings.shutter_angle >= 0.0
                    && settings.max_velocity_pixels >= 0.0
            }
            Self::ChromaticAberration(value) | Self::Vignette(value) | Self::FilmGrain(value) => {
                value.is_finite() && value >= 0.0
            }
            Self::ColorGrade(settings) => {
                settings.exposure_ev.is_finite()
                    && settings.contrast.is_finite()
                    && settings.saturation.is_finite()
            }
            Self::Smaa(level) => level > 0,
            Self::Tonemap(_) | Self::Fxaa | Self::Raw { .. } => true,
        }
    }

    fn sort_key(self) -> (PostStage, PassDomain, u64) {
        let kind = match self {
            Self::MotionBlur(_) => 0,
            Self::DepthOfField(_) => 1,
            Self::Bloom(_) => 2,
            Self::Exposure(_) => 3,
            Self::Tonemap(_) => 4,
            Self::ColorGrade(_) => 5,
            Self::ChromaticAberration(_) => 6,
            Self::Vignette(_) => 7,
            Self::FilmGrain(_) => 8,
            Self::Fxaa => 9,
            Self::Smaa(_) => 10,
            Self::Raw { id, .. } => 0x1_0000_0000 | id as u64,
        };
        (self.stage(), self.domain(), kind)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PostStackError {
    CapacityExceeded,
    DuplicateExclusiveNode,
    InvalidDomain,
    InvalidSettings,
}

/// Bounded authoring stack. Normalization is deterministic and allocation-free
/// after the caller has supplied the node storage.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PostStack {
    pub nodes: Vec<PostNode>,
}

impl PostStack {
    pub const MAX_NODES: usize = 32;

    pub fn normalize(&mut self) -> Result<(), PostStackError> {
        if self.nodes.len() > Self::MAX_NODES {
            return Err(PostStackError::CapacityExceeded);
        }
        for node in &self.nodes {
            if !node.validate() {
                return Err(PostStackError::InvalidSettings);
            }
            if node.domain() == PassDomain::PreUpscale && node.stage() >= PostStage::Tonemap {
                return Err(PostStackError::InvalidDomain);
            }
        }
        let mut seen = [0_u64; Self::MAX_NODES];
        let mut seen_count = 0;
        for node in &self.nodes {
            if let Some(key) = node.exclusive_key() {
                if seen[..seen_count].contains(&key) {
                    return Err(PostStackError::DuplicateExclusiveNode);
                }
                seen[seen_count] = key;
                seen_count += 1;
            }
        }
        self.nodes.sort_by_key(|node| node.sort_key());
        Ok(())
    }

    pub fn push(&mut self, node: PostNode) -> Result<(), PostStackError> {
        if self.nodes.len() >= Self::MAX_NODES {
            return Err(PostStackError::CapacityExceeded);
        }
        self.nodes.push(node);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_orders_nodes_and_rejects_duplicate_exclusive_nodes() {
        let mut stack = PostStack { nodes: Vec::new() };
        stack.push(PostNode::Tonemap(Tonemapper::AgX)).unwrap();
        stack
            .push(PostNode::Bloom(BloomSettings {
                threshold: 1.0,
                intensity: 1.0,
                radius: 1.0,
            }))
            .unwrap();
        assert!(stack.normalize().is_ok());
        assert!(matches!(stack.nodes[0], PostNode::Bloom(_)));
        stack.push(PostNode::Tonemap(Tonemapper::Aces)).unwrap();
        assert_eq!(
            stack.normalize(),
            Err(PostStackError::DuplicateExclusiveNode)
        );
    }

    #[test]
    fn raw_node_ids_do_not_collide_with_builtin_exclusive_nodes() {
        let mut stack = PostStack { nodes: Vec::new() };
        stack
            .push(PostNode::Raw {
                id: 0,
                stage: PostStage::Stylize,
                domain: PassDomain::PostUpscale,
            })
            .unwrap();
        stack
            .push(PostNode::Exposure(ExposureSettings {
                compensation_ev: 0.0,
                domain: PassDomain::PostUpscale,
            }))
            .unwrap();
        assert!(stack.normalize().is_ok());
    }

    #[test]
    fn normalization_rejects_non_finite_or_invalid_settings() {
        let mut stack = PostStack {
            nodes: vec![PostNode::Bloom(BloomSettings {
                threshold: f32::NAN,
                intensity: 1.0,
                radius: 1.0,
            })],
        };
        assert_eq!(stack.normalize(), Err(PostStackError::InvalidSettings));

        stack.nodes[0] = PostNode::DepthOfField(DofSettings {
            focus_distance_m: 2.0,
            max_coc_mm: 1.0,
            aperture_blades: 2,
        });
        assert_eq!(stack.normalize(), Err(PostStackError::InvalidSettings));
    }

    #[test]
    fn normalization_has_deterministic_tie_order() {
        let mut stack = PostStack {
            nodes: vec![
                PostNode::Raw {
                    id: 20,
                    stage: PostStage::Stylize,
                    domain: PassDomain::PostUpscale,
                },
                PostNode::Vignette(0.5),
                PostNode::Raw {
                    id: 3,
                    stage: PostStage::Stylize,
                    domain: PassDomain::PostUpscale,
                },
            ],
        };
        stack.normalize().unwrap();
        assert!(matches!(stack.nodes[0], PostNode::Vignette(_)));
        assert!(matches!(stack.nodes[1], PostNode::Raw { id: 3, .. }));
        assert!(matches!(stack.nodes[2], PostNode::Raw { id: 20, .. }));
    }
}
