use crate::axis::{Illumination, SpecializationId};
use crate::{MaterialFeatureFlags, MaterialRenderClass, MaterialValidationError};
use alloc::collections::{BTreeMap, BTreeSet};

/// Maximum number of `Layer`/`Mix` nodes on any path from the output. The
/// closure stack is a *bounded slab* (design doc §3.2), never an unbounded
/// Substrate; exceeding this is a validation error rather than a silent
/// performance cliff.
pub const MAX_CLOSURE_SLAB_DEPTH: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MaterialNodeId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaterialValue {
    Scalar(f32),
    Vector([f32; 4]),
    Boolean(bool),
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClosureKind {
    Diffuse,
    Conductor,
    Dielectric,
    ClearCoat,
    Sheen,
    Subsurface,
    Transmission,
    Emission,
    Hair,
    Volume,
    Npr,
    Custom,
    Mix,
    Layer,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MaterialNode {
    Constant(MaterialValue),
    Parameter {
        slot: u32,
        default: MaterialValue,
    },
    TextureSample {
        texture_slot: u32,
        uv: MaterialNodeId,
    },
    Add(MaterialNodeId, MaterialNodeId),
    Multiply(MaterialNodeId, MaterialNodeId),
    Closure {
        kind: ClosureKind,
        inputs: Vec<MaterialNodeId>,
    },
}

#[derive(Clone, Debug, Default)]
pub struct MaterialGraph {
    pub nodes: BTreeMap<MaterialNodeId, MaterialNode>,
    pub output: Option<MaterialNodeId>,
}

/// The lowered, live-pruned material. The closure graph is **preserved** as an
/// ordered node list; it is no longer collapsed into a single shading model.
/// Style lives on the orthogonal [`Illumination`] axis and the compiled
/// permutation identity is the [`SpecializationId`].
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedMaterial {
    pub nodes: Vec<(MaterialNodeId, MaterialNode)>,
    pub output: MaterialNodeId,
    pub features: MaterialFeatureFlags,
    pub render_class: MaterialRenderClass,
    pub illumination: Illumination,
    pub closure_mask: u32,
    pub specialization_id: SpecializationId,
}

impl MaterialGraph {
    pub fn normalize(&self) -> Result<NormalizedMaterial, MaterialValidationError> {
        crate::validate_graph(self)?;
        let output = self.output.expect("validated output");

        // Enforce the bounded closure slab depth on the live output cone.
        let slab_depth = self.slab_depth(output, &mut BTreeMap::new());
        if slab_depth > MAX_CLOSURE_SLAB_DEPTH {
            return Err(MaterialValidationError::ClosureSlabTooDeep {
                node: output,
                depth: slab_depth,
                max: MAX_CLOSURE_SLAB_DEPTH,
            });
        }

        let mut live = BTreeSet::new();
        collect_live(self, output, &mut live);
        let nodes = live
            .iter()
            .filter_map(|id| self.nodes.get(id).cloned().map(|node| (*id, node)))
            .collect();

        let mut closure_mask = 0_u32;
        let mut features = MaterialFeatureFlags::default();
        let mut illumination = Illumination::Lit;
        for id in &live {
            match self.nodes[id] {
                MaterialNode::TextureSample { .. } => features |= MaterialFeatureFlags::NORMAL_MAP,
                MaterialNode::Closure { kind, .. } => {
                    closure_mask |= 1 << kind as u32;
                    if matches!(kind, ClosureKind::Transmission) {
                        features |= MaterialFeatureFlags::TRANSMISSION;
                    }
                    if matches!(kind, ClosureKind::Emission) {
                        features |= MaterialFeatureFlags::EMISSIVE;
                    }
                    // Illumination is an orthogonal axis, not a collapse: an NPR
                    // closure marks the surface as `Stylized`, a custom closure
                    // as `Custom`. The rest of the closure graph is preserved.
                    if matches!(kind, ClosureKind::Npr) {
                        illumination = Illumination::Stylized;
                    }
                    if matches!(kind, ClosureKind::Custom) {
                        illumination = Illumination::Custom;
                    }
                    if matches!(kind, ClosureKind::Layer | ClosureKind::Mix) {
                        features |= MaterialFeatureFlags::COMPLEX_CLOSURE;
                    }
                }
                _ => {}
            }
        }

        // Blend family derives from transmission only. Style (NPR/custom) no
        // longer leaks into `render_class`; that flattening was the bug.
        let render_class = if features.contains(MaterialFeatureFlags::TRANSMISSION) {
            MaterialRenderClass::Transmissive
        } else {
            MaterialRenderClass::Opaque
        };

        let specialization_id =
            SpecializationId::new(illumination, closure_mask, render_class as u32);

        Ok(NormalizedMaterial {
            nodes,
            output,
            features,
            render_class,
            illumination,
            closure_mask,
            specialization_id,
        })
    }

    /// Longest chain of `Layer`/`Mix` nodes on any path from `id`, memoized.
    fn slab_depth(&self, id: MaterialNodeId, memo: &mut BTreeMap<MaterialNodeId, u32>) -> u32 {
        if let Some(cached) = memo.get(&id) {
            return *cached;
        }
        // Guard against cycles (already rejected by validation, but keep the
        // recursion total): seed with 0 before descending.
        memo.insert(id, 0);
        let depth = match &self.nodes[&id] {
            MaterialNode::TextureSample { uv, .. } => self.slab_depth(*uv, memo),
            MaterialNode::Add(a, b) | MaterialNode::Multiply(a, b) => {
                self.slab_depth(*a, memo).max(self.slab_depth(*b, memo))
            }
            MaterialNode::Closure { kind, inputs } => {
                let child = inputs
                    .iter()
                    .map(|input| self.slab_depth(*input, memo))
                    .max()
                    .unwrap_or(0);
                if matches!(kind, ClosureKind::Layer | ClosureKind::Mix) {
                    child + 1
                } else {
                    child
                }
            }
            _ => 0,
        };
        memo.insert(id, depth);
        depth
    }
}

fn collect_live(graph: &MaterialGraph, id: MaterialNodeId, live: &mut BTreeSet<MaterialNodeId>) {
    if !live.insert(id) {
        return;
    }
    match &graph.nodes[&id] {
        MaterialNode::TextureSample { uv, .. } => collect_live(graph, *uv, live),
        MaterialNode::Add(a, b) | MaterialNode::Multiply(a, b) => {
            collect_live(graph, *a, live);
            collect_live(graph, *b, live);
        }
        MaterialNode::Closure { inputs, .. } => {
            for input in inputs {
                collect_live(graph, *input, live);
            }
        }
        _ => {}
    }
}
