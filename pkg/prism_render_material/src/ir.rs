use crate::{
    BuiltinProgram, MaterialFeatureFlags, MaterialRenderClass, MaterialValidationError, ShadingPlan,
};
use alloc::collections::{BTreeMap, BTreeSet};

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
    /// Explicit lowering intent for registered programs and response composition.
    /// A graph with no override selects Lit or Npr from its live closures.
    pub plan: Option<ShadingPlan>,
}

/// The lowered, live-pruned material. The closure graph is **preserved** as an
/// ordered node list; it is no longer collapsed into a single shading model.
/// Response selection is an explicit plan. Packed lobes are layout data;
/// executable module/permutation identity is assigned by package compilation.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedMaterial {
    pub nodes: Vec<(MaterialNodeId, MaterialNode)>,
    pub output: MaterialNodeId,
    pub features: MaterialFeatureFlags,
    pub render_class: MaterialRenderClass,
    pub plan: ShadingPlan,
    pub closure_mask: u32,
}

impl MaterialGraph {
    pub fn normalize(&self) -> Result<NormalizedMaterial, MaterialValidationError> {
        crate::validate_graph(self)?;
        let output = self.output.expect("validated output");

        let mut live = BTreeSet::new();
        collect_live(self, output, &mut live);
        let nodes = live
            .iter()
            .filter_map(|id| self.nodes.get(id).cloned().map(|node| (*id, node)))
            .collect();

        let mut closure_mask = 0_u32;
        let mut features = MaterialFeatureFlags::default();
        let mut npr = false;
        let mut custom = false;
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
                    if matches!(kind, ClosureKind::Npr) {
                        npr = true;
                    }
                    if matches!(kind, ClosureKind::Custom) {
                        custom = true;
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

        let plan = self.plan.clone().unwrap_or_else(|| {
            ShadingPlan::single_builtin(if npr {
                BuiltinProgram::Npr
            } else {
                BuiltinProgram::Lit
            })
        });
        if custom && !plan.contains_registered() {
            return Err(MaterialValidationError::MissingCustomProgram);
        }

        Ok(NormalizedMaterial {
            nodes,
            output,
            features,
            render_class,
            plan,
            closure_mask,
        })
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
