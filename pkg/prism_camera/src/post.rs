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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PostStackError {
    CapacityExceeded,
    DuplicateExclusiveNode,
    InvalidDomain,
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
        self.nodes.sort_by(|a, b| {
            a.stage()
                .cmp(&b.stage())
                .then_with(|| a.domain().cmp(&b.domain()))
        });
        Ok(())
    }

    pub fn push(&mut self, node: PostNode) -> Result<(), PostStackError> {
        if self.nodes.len() == Self::MAX_NODES {
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
}
