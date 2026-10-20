//! Generic ragdoll authoring and deterministic self-collision policy.
//!
//! [`RagdollBuilder`] is intentionally independent of gameplay names and ECS
//! entities. A caller supplies stable part identifiers, body descriptions,
//! local anchors, and [`super::SwingTwistJoint`] contracts. The result is an
//! immutable, sorted definition that a world integration can instantiate.
//! [`HumanoidPreset`] is opt-in and only supplies a conventional starting
//! topology; it is not part of the generic API.

use std::collections::{BTreeMap, BTreeSet};

use glam::Vec3;

use crate::math::transform::Isometry;
use crate::state::body::BodyDesc;
use crate::state::handle::BodyHandle;

use super::SwingTwistJoint;

const EPS: f32 = 1.0e-6;

/// Stable application-defined identifier for a ragdoll part.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RagdollPartId(u16);

impl RagdollPartId {
    /// Creates an identifier from a stable numeric value.
    #[must_use]
    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    /// Returns the stable numeric value.
    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// A body description and authoring transform for one ragdoll part.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RagdollPart {
    /// Stable part identifier.
    pub id: RagdollPartId,
    /// Body state used when the part is instantiated.
    pub body: BodyDesc,
    /// Part-local transform relative to the ragdoll root.
    pub local_pose: Isometry,
}

/// A parent-child link with a solver-neutral swing-twist contract.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RagdollLink {
    /// Parent part identifier.
    pub parent: RagdollPartId,
    /// Child part identifier.
    pub child: RagdollPartId,
    /// Anchor in parent-local coordinates.
    pub anchor_parent: Vec3,
    /// Anchor in child-local coordinates.
    pub anchor_child: Vec3,
    /// Rotational constraint for this link.
    pub joint: SwingTwistJoint,
}

/// Self-collision policy for parts belonging to one ragdoll.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SelfCollisionPolicy {
    /// Do not generate self-collision pairs.
    Disabled,
    /// Generate every distinct part pair.
    All,
    /// Exclude directly linked parent-child pairs.
    ExcludeAdjacent,
}

impl Default for SelfCollisionPolicy {
    fn default() -> Self {
        Self::ExcludeAdjacent
    }
}

/// Deterministic pair filter generated from a ragdoll topology.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SelfCollisionFilter {
    policy: SelfCollisionPolicy,
    adjacent_pairs: BTreeSet<(RagdollPartId, RagdollPartId)>,
}

impl SelfCollisionFilter {
    /// Returns the policy used to construct this filter.
    #[must_use]
    pub const fn policy(&self) -> SelfCollisionPolicy {
        self.policy
    }

    /// Returns whether two distinct parts should produce a self-collision
    /// pair.
    #[must_use]
    pub fn should_collide(&self, a: RagdollPartId, b: RagdollPartId) -> bool {
        if a == b {
            return false;
        }
        match self.policy {
            SelfCollisionPolicy::Disabled => false,
            SelfCollisionPolicy::All => true,
            SelfCollisionPolicy::ExcludeAdjacent => {
                !self.adjacent_pairs.contains(&ordered_pair(a, b))
            }
        }
    }

    /// Returns all collision pairs for a sorted part list.
    #[must_use]
    pub fn pairs(&self, parts: &[RagdollPart]) -> Vec<(RagdollPartId, RagdollPartId)> {
        let mut pairs = Vec::new();
        for (index, a) in parts.iter().enumerate() {
            for b in parts.iter().skip(index + 1) {
                if self.should_collide(a.id, b.id) {
                    pairs.push((a.id, b.id));
                }
            }
        }
        pairs
    }
}

/// Threshold controlling mass-ratio diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MassRatioPolicy {
    /// Warn when a finite parent/child mass ratio exceeds this value.
    pub warn_above: f32,
    /// Warn when a finite principal-inertia ratio exceeds this value.
    pub inertia_warn_above: f32,
}

impl Default for MassRatioPolicy {
    fn default() -> Self {
        Self {
            warn_above: 100.0,
            inertia_warn_above: 100.0,
        }
    }
}

/// One mass or inertia conditioning warning emitted by [`RagdollBuilder`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MassRatioWarning {
    /// Parent part in the problematic link.
    pub parent: RagdollPartId,
    /// Child part in the problematic link.
    pub child: RagdollPartId,
    /// Larger-to-smaller finite mass ratio, or zero when not applicable.
    pub mass_ratio: f32,
    /// Larger-to-smaller finite principal inertia ratio, or zero when not applicable.
    pub inertia_ratio: f32,
}

/// Diagnostics produced during ragdoll construction.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RagdollDiagnostics {
    /// Conditioning warnings in deterministic link order.
    pub mass_ratio_warnings: Vec<MassRatioWarning>,
}

/// An immutable, sorted ragdoll definition ready for world instantiation.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RagdollDefinition {
    /// Parts sorted by stable identifier.
    pub parts: Vec<RagdollPart>,
    /// Links sorted by `(parent, child)`.
    pub links: Vec<RagdollLink>,
    /// Deterministic self-collision filter.
    pub self_collision: SelfCollisionFilter,
    /// Build-time conditioning diagnostics.
    pub diagnostics: RagdollDiagnostics,
}

/// A live-body Swing-Twist constraint generated from one ragdoll link.
///
/// This is the solver-facing form of a link. Keeping it separate from the
/// authoring definition means the XPBD hot loop only reads compact handles,
/// anchors, and the validated response law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RagdollConstraint {
    /// Parent rigid body.
    pub body_a: BodyHandle,
    /// Child rigid body.
    pub body_b: BodyHandle,
    /// Parent-local anchor point.
    pub local_point_a: Vec3,
    /// Child-local anchor point.
    pub local_point_b: Vec3,
    /// Swing-Twist limit, friction, and drive response.
    pub joint: SwingTwistJoint,
}

/// A validated mapping from a built ragdoll definition to live rigid bodies.
///
/// The mapping is intentionally kept outside [`RagdollDefinition`]: the
/// definition is reusable asset data, while handles belong to one simulation
/// instance. XPBD consumes this binding directly during its island solve.
#[derive(Clone, Debug, PartialEq)]
pub struct RagdollBinding {
    definition: RagdollDefinition,
    bodies: Vec<BodyHandle>,
    constraints: Vec<RagdollConstraint>,
}

/// Errors returned when a definition is bound to simulation bodies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RagdollBindingError {
    /// The number of body handles differs from the number of definition parts.
    PartCountMismatch { expected: usize, actual: usize },
    /// A body handle is invalid and cannot participate in an XPBD solve.
    InvalidBodyHandle { index: usize },
}

impl RagdollBinding {
    /// Binds bodies to parts in the definition's stable identifier order.
    pub fn try_new(
        definition: RagdollDefinition,
        bodies: Vec<BodyHandle>,
    ) -> Result<Self, RagdollBindingError> {
        if bodies.len() != definition.parts.len() {
            return Err(RagdollBindingError::PartCountMismatch {
                expected: definition.parts.len(),
                actual: bodies.len(),
            });
        }
        if let Some(index) = bodies.iter().position(|body| *body == BodyHandle::INVALID) {
            return Err(RagdollBindingError::InvalidBodyHandle { index });
        }
        let constraints = definition
            .links
            .iter()
            .map(|link| RagdollConstraint {
                body_a: body_for_part(&definition, &bodies, link.parent),
                body_b: body_for_part(&definition, &bodies, link.child),
                local_point_a: link.anchor_parent,
                local_point_b: link.anchor_child,
                joint: link.joint,
            })
            .collect();
        Ok(Self {
            definition,
            bodies,
            constraints,
        })
    }

    /// Returns the immutable authoring definition.
    #[must_use]
    pub const fn definition(&self) -> &RagdollDefinition {
        &self.definition
    }

    /// Returns the body handle corresponding to a stable part identifier.
    #[must_use]
    pub fn body(&self, id: RagdollPartId) -> Option<BodyHandle> {
        let index = self
            .definition
            .parts
            .binary_search_by_key(&id, |part| part.id)
            .ok()?;
        self.bodies.get(index).copied()
    }

    /// Returns all definition links paired with their mapped body handles.
    #[must_use]
    pub fn links(&self) -> impl Iterator<Item = (&RagdollLink, BodyHandle, BodyHandle)> {
        self.definition
            .links
            .iter()
            .filter_map(|link| Some((link, self.body(link.parent)?, self.body(link.child)?)))
    }

    /// Returns the compact constraints consumed by the XPBD solver.
    #[must_use]
    pub fn constraints(&self) -> &[RagdollConstraint] {
        &self.constraints
    }
}

fn body_for_part(
    definition: &RagdollDefinition,
    bodies: &[BodyHandle],
    id: RagdollPartId,
) -> BodyHandle {
    let index = definition
        .parts
        .binary_search_by_key(&id, |part| part.id)
        .expect("validated ragdoll link must reference a definition part");
    bodies[index]
}

impl RagdollDefinition {
    /// Finds a part by stable identifier.
    #[must_use]
    pub fn part(&self, id: RagdollPartId) -> Option<&RagdollPart> {
        self.parts
            .binary_search_by_key(&id, |part| part.id)
            .ok()
            .map(|index| &self.parts[index])
    }

    /// Returns the number of parts.
    #[must_use]
    pub fn part_count(&self) -> usize {
        self.parts.len()
    }

    /// Returns the number of links.
    #[must_use]
    pub fn link_count(&self) -> usize {
        self.links.len()
    }
}

/// Construction errors for a generic ragdoll definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum RagdollBuildError {
    /// A part identifier was already inserted.
    DuplicatePart(RagdollPartId),
    /// A link referenced a part that was not inserted.
    UnknownPart(RagdollPartId),
    /// A link connected a part to itself.
    SelfLink(RagdollPartId),
    /// Two links used the same unordered pair.
    DuplicateLink(RagdollPartId, RagdollPartId),
    /// The links do not form a tree/forest.
    Cycle(RagdollPartId, RagdollPartId),
    /// A body transform or vector contained invalid values.
    InvalidPart(RagdollPartId),
    /// The mass-ratio policy was invalid.
    InvalidMassRatioPolicy,
}

/// Generic deterministic ragdoll builder.
#[derive(Clone, Debug, Default)]
pub struct RagdollBuilder {
    parts: BTreeMap<RagdollPartId, RagdollPart>,
    links: Vec<RagdollLink>,
    self_collision_policy: SelfCollisionPolicy,
    mass_ratio_policy: MassRatioPolicy,
}

impl RagdollBuilder {
    /// Creates an empty generic builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one part. Part identifiers must be unique.
    pub fn add_part(&mut self, part: RagdollPart) -> Result<(), RagdollBuildError> {
        if !valid_part(part) {
            return Err(RagdollBuildError::InvalidPart(part.id));
        }
        if self.parts.contains_key(&part.id) {
            return Err(RagdollBuildError::DuplicatePart(part.id));
        }
        self.parts.insert(part.id, part);
        Ok(())
    }

    /// Replaces the body description and local pose of an existing part.
    pub fn replace_part(&mut self, part: RagdollPart) -> Result<(), RagdollBuildError> {
        if !valid_part(part) {
            return Err(RagdollBuildError::InvalidPart(part.id));
        }
        if !self.parts.contains_key(&part.id) {
            return Err(RagdollBuildError::UnknownPart(part.id));
        }
        self.parts.insert(part.id, part);
        Ok(())
    }

    /// Adds a parent-child link. The part references are checked at build time
    /// so callers can author links before all parts are inserted.
    pub fn add_link(&mut self, link: RagdollLink) -> Result<(), RagdollBuildError> {
        if link.parent == link.child {
            return Err(RagdollBuildError::SelfLink(link.parent));
        }
        let pair = ordered_pair(link.parent, link.child);
        if self
            .links
            .iter()
            .any(|existing| ordered_pair(existing.parent, existing.child) == pair)
        {
            return Err(RagdollBuildError::DuplicateLink(pair.0, pair.1));
        }
        self.links.push(link);
        Ok(())
    }

    /// Sets the self-collision policy.
    #[must_use]
    pub const fn with_self_collision_policy(mut self, policy: SelfCollisionPolicy) -> Self {
        self.self_collision_policy = policy;
        self
    }

    /// Sets the conditioning warning thresholds.
    #[must_use]
    pub fn with_mass_ratio_policy(mut self, policy: MassRatioPolicy) -> Self {
        self.mass_ratio_policy = policy;
        self
    }

    /// Builds a sorted, validated definition and computes diagnostics.
    pub fn build(mut self) -> Result<RagdollDefinition, RagdollBuildError> {
        if !valid_ratio_policy(self.mass_ratio_policy) {
            return Err(RagdollBuildError::InvalidMassRatioPolicy);
        }
        for link in &self.links {
            if !self.parts.contains_key(&link.parent) {
                return Err(RagdollBuildError::UnknownPart(link.parent));
            }
            if !self.parts.contains_key(&link.child) {
                return Err(RagdollBuildError::UnknownPart(link.child));
            }
        }

        let mut parts: Vec<_> = self.parts.into_values().collect();
        parts.sort_by_key(|part| part.id);
        self.links.sort_by_key(|link| (link.parent, link.child));
        detect_cycles(&parts, &self.links)?;

        let adjacent_pairs = self
            .links
            .iter()
            .map(|link| ordered_pair(link.parent, link.child))
            .collect();
        let self_collision = SelfCollisionFilter {
            policy: self.self_collision_policy,
            adjacent_pairs,
        };
        let diagnostics = RagdollDiagnostics {
            mass_ratio_warnings: mass_ratio_warnings(&parts, &self.links, self.mass_ratio_policy),
        };
        Ok(RagdollDefinition {
            parts,
            links: self.links,
            self_collision,
            diagnostics,
        })
    }
}

/// Conventional humanoid part identifiers used only by [`HumanoidPreset`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum HumanoidPart {
    Pelvis,
    Spine,
    Chest,
    Head,
    UpperArmLeft,
    LowerArmLeft,
    HandLeft,
    UpperArmRight,
    LowerArmRight,
    HandRight,
    UpperLegLeft,
    LowerLegLeft,
    FootLeft,
    UpperLegRight,
    LowerLegRight,
    FootRight,
}

impl HumanoidPart {
    /// Returns a stable numeric identifier for the preset part.
    #[must_use]
    pub const fn id(self) -> RagdollPartId {
        Self::id_value(self)
    }

    const fn id_value(value: Self) -> RagdollPartId {
        RagdollPartId::new(match value {
            Self::Pelvis => 0,
            Self::Spine => 1,
            Self::Chest => 2,
            Self::Head => 3,
            Self::UpperArmLeft => 4,
            Self::LowerArmLeft => 5,
            Self::HandLeft => 6,
            Self::UpperArmRight => 7,
            Self::LowerArmRight => 8,
            Self::HandRight => 9,
            Self::UpperLegLeft => 10,
            Self::LowerLegLeft => 11,
            Self::FootLeft => 12,
            Self::UpperLegRight => 13,
            Self::LowerLegRight => 14,
            Self::FootRight => 15,
        })
    }

    const ALL: [Self; 16] = [
        Self::Pelvis,
        Self::Spine,
        Self::Chest,
        Self::Head,
        Self::UpperArmLeft,
        Self::LowerArmLeft,
        Self::HandLeft,
        Self::UpperArmRight,
        Self::LowerArmRight,
        Self::HandRight,
        Self::UpperLegLeft,
        Self::LowerLegLeft,
        Self::FootLeft,
        Self::UpperLegRight,
        Self::LowerLegRight,
        Self::FootRight,
    ];
}

/// Opt-in conventional humanoid topology.
pub struct HumanoidPreset;

impl HumanoidPreset {
    /// Returns a generic builder seeded with 16 named-by-ID humanoid parts and
    /// 15 tree links. Callers may replace body descriptions before building.
    #[must_use]
    pub fn builder() -> RagdollBuilder {
        let mut builder = RagdollBuilder::new();
        for part in HumanoidPart::ALL {
            let id = part.id();
            let body = BodyDesc::dynamic_at(Vec3::ZERO);
            let inserted = builder.add_part(RagdollPart {
                id,
                body,
                local_pose: Isometry::IDENTITY,
            });
            debug_assert!(inserted.is_ok());
        }

        let links = [
            (HumanoidPart::Pelvis, HumanoidPart::Spine),
            (HumanoidPart::Spine, HumanoidPart::Chest),
            (HumanoidPart::Chest, HumanoidPart::Head),
            (HumanoidPart::Chest, HumanoidPart::UpperArmLeft),
            (HumanoidPart::UpperArmLeft, HumanoidPart::LowerArmLeft),
            (HumanoidPart::LowerArmLeft, HumanoidPart::HandLeft),
            (HumanoidPart::Chest, HumanoidPart::UpperArmRight),
            (HumanoidPart::UpperArmRight, HumanoidPart::LowerArmRight),
            (HumanoidPart::LowerArmRight, HumanoidPart::HandRight),
            (HumanoidPart::Pelvis, HumanoidPart::UpperLegLeft),
            (HumanoidPart::UpperLegLeft, HumanoidPart::LowerLegLeft),
            (HumanoidPart::LowerLegLeft, HumanoidPart::FootLeft),
            (HumanoidPart::Pelvis, HumanoidPart::UpperLegRight),
            (HumanoidPart::UpperLegRight, HumanoidPart::LowerLegRight),
            (HumanoidPart::LowerLegRight, HumanoidPart::FootRight),
        ];
        for (parent, child) in links {
            let config =
                super::SwingTwistConfig::symmetric(Vec3::X, Vec3::X, Vec3::Y, Vec3::Y, 0.75, 0.8);
            let joint = SwingTwistJoint::try_new(config).unwrap_or_else(|_| unreachable!());
            let inserted = builder.add_link(RagdollLink {
                parent: parent.id(),
                child: child.id(),
                anchor_parent: Vec3::ZERO,
                anchor_child: Vec3::ZERO,
                joint,
            });
            debug_assert!(inserted.is_ok());
        }
        builder
    }
}

fn ordered_pair(a: RagdollPartId, b: RagdollPartId) -> (RagdollPartId, RagdollPartId) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn valid_part(part: RagdollPart) -> bool {
    part.body.position.is_finite()
        && part.body.orientation.is_finite()
        && part.body.orientation.length_squared() > EPS
        && part.body.linear_velocity.is_finite()
        && part.body.angular_velocity.is_finite()
        && part.body.mass_properties.inv_mass.is_finite()
        && part.body.mass_properties.inv_mass >= 0.0
        && part.body.mass_properties.inv_inertia.is_finite()
        && part.body.mass_properties.inv_inertia.min_element() >= 0.0
        && part.local_pose.translation.is_finite()
        && part.local_pose.rotation.is_finite()
        && part.local_pose.rotation.length_squared() > EPS
}

fn valid_ratio_policy(policy: MassRatioPolicy) -> bool {
    policy.warn_above.is_finite()
        && policy.warn_above >= 1.0
        && policy.inertia_warn_above.is_finite()
        && policy.inertia_warn_above >= 1.0
}

fn detect_cycles(parts: &[RagdollPart], links: &[RagdollLink]) -> Result<(), RagdollBuildError> {
    let index: BTreeMap<_, _> = parts
        .iter()
        .enumerate()
        .map(|(i, part)| (part.id, i))
        .collect();
    let mut parent: Vec<usize> = (0..parts.len()).collect();
    for link in links {
        let a = index[&link.parent];
        let b = index[&link.child];
        let root_a = find_root(&mut parent, a);
        let root_b = find_root(&mut parent, b);
        if root_a == root_b {
            return Err(RagdollBuildError::Cycle(link.parent, link.child));
        }
        parent[root_a] = root_b;
    }
    Ok(())
}

fn find_root(parent: &mut [usize], value: usize) -> usize {
    let mut root = value;
    while parent[root] != root {
        root = parent[root];
    }
    let mut current = value;
    while parent[current] != current {
        let next = parent[current];
        parent[current] = root;
        current = next;
    }
    root
}

fn mass_ratio_warnings(
    parts: &[RagdollPart],
    links: &[RagdollLink],
    policy: MassRatioPolicy,
) -> Vec<MassRatioWarning> {
    let by_id: BTreeMap<_, _> = parts.iter().map(|part| (part.id, part)).collect();
    let mut warnings = Vec::new();
    for link in links {
        let a = by_id[&link.parent].body.mass_properties;
        let b = by_id[&link.child].body.mass_properties;
        let mass_ratio = finite_ratio(a.inv_mass, b.inv_mass);
        let inertia_ratio = inertia_ratio(a.inv_inertia, b.inv_inertia);
        if mass_ratio > policy.warn_above || inertia_ratio > policy.inertia_warn_above {
            warnings.push(MassRatioWarning {
                parent: link.parent,
                child: link.child,
                mass_ratio,
                inertia_ratio,
            });
        }
    }
    warnings
}

fn finite_ratio(inverse_a: f32, inverse_b: f32) -> f32 {
    if inverse_a > EPS && inverse_b > EPS {
        let mass_a = 1.0 / inverse_a;
        let mass_b = 1.0 / inverse_b;
        (mass_a / mass_b).max(mass_b / mass_a)
    } else {
        0.0
    }
}

fn inertia_ratio(inverse_a: Vec3, inverse_b: Vec3) -> f32 {
    let mut ratio: f32 = 0.0;
    for (a, b) in inverse_a.to_array().into_iter().zip(inverse_b.to_array()) {
        if a > EPS && b > EPS {
            let ia = 1.0 / a;
            let ib = 1.0 / b;
            ratio = ratio.max((ia / ib).max(ib / ia));
        }
    }
    ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::body::MassProperties;

    fn swing_joint() -> SwingTwistJoint {
        SwingTwistJoint::try_new(super::super::SwingTwistConfig::symmetric(
            Vec3::X,
            Vec3::X,
            Vec3::Y,
            Vec3::Y,
            0.6,
            0.7,
        ))
        .unwrap()
    }

    #[test]
    fn generic_builder_sorts_and_filters_adjacent_pairs() {
        let mut builder = RagdollBuilder::new();
        for id in [
            RagdollPartId::new(2),
            RagdollPartId::new(1),
            RagdollPartId::new(3),
        ] {
            builder
                .add_part(RagdollPart {
                    id,
                    body: BodyDesc::dynamic_at(Vec3::ZERO),
                    local_pose: Isometry::IDENTITY,
                })
                .unwrap();
        }
        builder
            .add_link(RagdollLink {
                parent: RagdollPartId::new(2),
                child: RagdollPartId::new(1),
                anchor_parent: Vec3::ZERO,
                anchor_child: Vec3::ZERO,
                joint: swing_joint(),
            })
            .unwrap();
        let definition = builder.build().unwrap();
        assert_eq!(
            definition
                .parts
                .iter()
                .map(|part| part.id.get())
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!(!definition
            .self_collision
            .should_collide(RagdollPartId::new(1), RagdollPartId::new(2)));
        assert!(definition
            .self_collision
            .should_collide(RagdollPartId::new(1), RagdollPartId::new(3)));
    }

    #[test]
    fn mass_ratio_warning_is_real_and_deterministic() {
        let mut builder = RagdollBuilder::new();
        builder
            .add_part(RagdollPart {
                id: RagdollPartId::new(1),
                body: BodyDesc::dynamic_at(Vec3::ZERO).with_mass_properties(MassProperties {
                    inv_mass: 1.0,
                    inv_inertia: Vec3::ONE,
                }),
                local_pose: Isometry::IDENTITY,
            })
            .unwrap();
        builder
            .add_part(RagdollPart {
                id: RagdollPartId::new(2),
                body: BodyDesc::dynamic_at(Vec3::ZERO).with_mass_properties(MassProperties {
                    inv_mass: 0.001,
                    inv_inertia: Vec3::splat(0.001),
                }),
                local_pose: Isometry::IDENTITY,
            })
            .unwrap();
        builder
            .add_link(RagdollLink {
                parent: RagdollPartId::new(1),
                child: RagdollPartId::new(2),
                anchor_parent: Vec3::ZERO,
                anchor_child: Vec3::ZERO,
                joint: swing_joint(),
            })
            .unwrap();
        let definition = builder.build().unwrap();
        assert_eq!(definition.diagnostics.mass_ratio_warnings.len(), 1);
        assert!((definition.diagnostics.mass_ratio_warnings[0].mass_ratio - 1000.0).abs() < 1.0e-3);
    }

    #[test]
    fn cycles_are_rejected() {
        let mut builder = RagdollBuilder::new();
        for id in 0..3 {
            builder
                .add_part(RagdollPart {
                    id: RagdollPartId::new(id),
                    body: BodyDesc::dynamic_at(Vec3::ZERO),
                    local_pose: Isometry::IDENTITY,
                })
                .unwrap();
        }
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            builder
                .add_link(RagdollLink {
                    parent: RagdollPartId::new(a),
                    child: RagdollPartId::new(b),
                    anchor_parent: Vec3::ZERO,
                    anchor_child: Vec3::ZERO,
                    joint: swing_joint(),
                })
                .unwrap();
        }
        assert!(matches!(
            builder.build(),
            Err(RagdollBuildError::Cycle(_, _))
        ));
    }

    #[test]
    fn humanoid_is_opt_in_and_has_a_tree_topology() {
        let definition = HumanoidPreset::builder().build().unwrap();
        assert_eq!(definition.part_count(), 16);
        assert_eq!(definition.link_count(), 15);
        assert!(definition.diagnostics.mass_ratio_warnings.is_empty());
    }

    #[test]
    fn binding_maps_stable_part_order_into_solver_constraints() {
        let mut builder = RagdollBuilder::new();
        for id in [RagdollPartId::new(20), RagdollPartId::new(10)] {
            builder
                .add_part(RagdollPart {
                    id,
                    body: BodyDesc::dynamic_at(Vec3::ZERO),
                    local_pose: Isometry::IDENTITY,
                })
                .unwrap();
        }
        builder
            .add_link(RagdollLink {
                parent: RagdollPartId::new(20),
                child: RagdollPartId::new(10),
                anchor_parent: Vec3::X,
                anchor_child: -Vec3::X,
                joint: swing_joint(),
            })
            .unwrap();
        let definition = builder.build().unwrap();
        let first = BodyHandle::new(3, 7);
        let second = BodyHandle::new(9, 2);
        let binding = RagdollBinding::try_new(definition, vec![first, second]).unwrap();
        assert_eq!(binding.body(RagdollPartId::new(10)), Some(first));
        assert_eq!(binding.body(RagdollPartId::new(20)), Some(second));
        assert_eq!(binding.constraints().len(), 1);
        assert_eq!(binding.constraints()[0].body_a, second);
        assert_eq!(binding.constraints()[0].body_b, first);
        assert_eq!(binding.constraints()[0].local_point_a, Vec3::X);
    }

    #[test]
    fn binding_rejects_wrong_count_and_invalid_handle() {
        let mut builder = RagdollBuilder::new();
        builder
            .add_part(RagdollPart {
                id: RagdollPartId::new(1),
                body: BodyDesc::dynamic_at(Vec3::ZERO),
                local_pose: Isometry::IDENTITY,
            })
            .unwrap();
        let definition = builder.build().unwrap();
        assert_eq!(
            RagdollBinding::try_new(definition.clone(), Vec::new()),
            Err(RagdollBindingError::PartCountMismatch {
                expected: 1,
                actual: 0
            })
        );
        assert_eq!(
            RagdollBinding::try_new(definition, vec![BodyHandle::INVALID]),
            Err(RagdollBindingError::InvalidBodyHandle { index: 0 })
        );
    }
}
