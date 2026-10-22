//! Deterministic layered resolution, immutable snapshots, diffs, and explain data.
use crate::{
    CrossRule, Domain, LayerKind, PrecedenceProfile, SettingId, SettingKey, SettingsError,
    SettingsRegistry, TrustLevel, Value,
};
use alloc::{collections::BTreeMap, string::String, sync::Arc, vec::Vec};

/// One deterministic input source attached to a domain layer.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerSource {
    /// Layer occupied by the source.
    pub layer: LayerKind,
    /// Trust attached by the loader/caller.
    pub trust: TrustLevel,
    /// Stable source identity, also the final tie-breaker.
    pub source_id: String,
    /// Higher order wins when multiple sources target one layer.
    pub order: i32,
    /// Candidate values keyed by stable setting key.
    pub values: BTreeMap<SettingKey, Value>,
}
impl LayerSource {
    /// Creates an empty named layer source.
    pub fn new(layer: LayerKind, trust: TrustLevel, source_id: impl Into<String>) -> Self {
        Self {
            layer,
            trust,
            source_id: source_id.into(),
            order: 0,
            values: BTreeMap::new(),
        }
    }
    /// Inserts or replaces a candidate value in this source.
    pub fn set(&mut self, key: impl Into<SettingKey>, value: Value) {
        self.values.insert(key.into(), value);
    }
}

/// Provenance for an effective value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueSource {
    /// Winning layer.
    pub layer: LayerKind,
    /// Trust class.
    pub trust: TrustLevel,
    /// Stable source identity.
    pub source_id: String,
}

#[derive(Clone, Debug, PartialEq)]
struct SnapshotData {
    domain: Domain,
    generation: u64,
    values: Vec<Option<Value>>,
    sources: Vec<Option<ValueSource>>,
}

/// Frozen effective values. Cloning is O(1); reads borrow immutable data.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot(Arc<SnapshotData>);
impl Snapshot {
    /// Domain this snapshot resolves.
    pub fn domain(&self) -> Domain {
        self.0.domain
    }
    /// Monotonically supplied generation number.
    pub fn generation(&self) -> u64 {
        self.0.generation
    }
    /// Borrowed effective value by dense ID.
    pub fn get(&self, id: SettingId) -> Option<&Value> {
        self.0.values.get(id.0 as usize)?.as_ref()
    }
    /// Provenance by dense ID.
    pub fn source(&self, id: SettingId) -> Option<&ValueSource> {
        self.0.sources.get(id.0 as usize)?.as_ref()
    }
    /// Number of registered slots represented by this snapshot.
    pub fn len(&self) -> usize {
        self.0.values.len()
    }
    /// Whether the registry had no definitions.
    pub fn is_empty(&self) -> bool {
        self.0.values.is_empty()
    }
}

/// Explanation of one definition across every legal layer.
#[derive(Clone, Debug, PartialEq)]
pub struct InspectResult {
    /// Effective value, or `None` only when a registry slot was not resolved.
    pub effective: Option<Value>,
    /// Winning source, absent when default is effective.
    pub source: Option<ValueSource>,
    /// Schema default.
    pub default: Value,
    /// Winning candidate for each legal layer.
    pub by_layer: BTreeMap<LayerKind, Option<(Value, ValueSource)>>,
    /// Candidates excluded by precedence or rejected by source/value policy.
    pub diagnostics: Vec<SettingsError>,
    /// Source which enforces the effective value, when Mandatory wins.
    pub locked_by: Option<ValueSource>,
    /// Whether a user/project/runtime candidate could change the effective value.
    pub modifiable: bool,
}

/// A single changed slot between two snapshots.
#[derive(Clone, Debug, PartialEq)]
pub struct SettingChange {
    /// Dense setting ID.
    pub id: SettingId,
    /// Previous value.
    pub before: Option<Value>,
    /// New value.
    pub after: Option<Value>,
    /// Previous provenance.
    pub before_source: Option<ValueSource>,
    /// New provenance.
    pub after_source: Option<ValueSource>,
}
/// Deterministically ordered snapshot difference.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SettingsDiff {
    /// Changes in ascending SettingId order.
    pub changes: Vec<SettingChange>,
}

/// Resolves registered settings for one domain.
pub struct Resolver<'a> {
    registry: &'a SettingsRegistry,
}
impl<'a> Resolver<'a> {
    /// Binds a resolver to a frozen registry.
    pub const fn new(registry: &'a SettingsRegistry) -> Self {
        Self { registry }
    }

    /// Resolves sources in deterministic precedence order and validates the final cross-setting state.
    pub fn resolve(
        &self,
        domain: Domain,
        sources: &[LayerSource],
        generation: u64,
    ) -> Result<Snapshot, Vec<SettingsError>> {
        let mut errors = Vec::new();
        let stack = domain.layers();
        let mut identities = BTreeMap::new();
        for source in sources {
            if !stack.contains(&source.layer) {
                errors.push(SettingsError::WrongLayer {
                    layer: source.layer,
                    domain,
                });
            }
            let identity = (source.layer, source.source_id.clone());
            if identities.insert(identity, ()).is_some() {
                errors.push(SettingsError::DuplicateSource {
                    layer: source.layer,
                    source_id: source.source_id.clone(),
                });
            }
        }
        // Convert key maps to dense ID-indexed borrowed slots once. The hot
        // per-definition loop then performs only bounded source iteration and
        // direct indexing; it does not allocate or search BTreeMaps.
        let mut prepared = Vec::with_capacity(sources.len());
        let defs = self.registry.defs();
        for source in sources {
            let mut slots = alloc::vec![None; self.registry.len()];
            if stack.contains(&source.layer) {
                let mut canonical_index = 0;
                for (key, value) in &source.values {
                    while canonical_index < defs.len() && defs[canonical_index].key.cmp(key).is_lt()
                    {
                        canonical_index += 1;
                    }
                    let canonical = defs
                        .get(canonical_index)
                        .filter(|def| def.key == *key)
                        .map(|def| (SettingId(canonical_index as u32), def));
                    let resolved = canonical.or_else(|| self.registry.get(key));
                    let Some((id, def)) = resolved else {
                        errors.push(SettingsError::UnknownKey(key.clone()));
                        continue;
                    };
                    if !def.domains.contains(domain) {
                        errors.push(SettingsError::WrongDomain {
                            key: def.key.clone(),
                            domain,
                        });
                        continue;
                    }
                    slots[id.0 as usize] = Some(value);
                }
            }
            prepared.push((source, slots));
        }
        let profiles = [
            PrecedenceProfile::UserChoiceWins,
            PrecedenceProfile::ServerWins,
            PrecedenceProfile::DeviceScoped,
            PrecedenceProfile::Locked,
        ];
        let mut source_orders = Vec::with_capacity(profiles.len());
        for profile in profiles {
            let order = crate::layer_order(domain, profile);
            let mut indices: Vec<usize> = prepared
                .iter()
                .enumerate()
                .filter(|(_, (source, _))| order.contains(&source.layer))
                .map(|(index, _)| index)
                .collect();
            indices.sort_by(|a, b| {
                let (left, _) = &prepared[*a];
                let (right, _) = &prepared[*b];
                order
                    .iter()
                    .position(|layer| *layer == left.layer)
                    .cmp(&order.iter().position(|layer| *layer == right.layer))
                    .then_with(|| left.order.cmp(&right.order))
                    .then_with(|| right.source_id.cmp(&left.source_id))
            });
            source_orders.push(indices);
        }
        let mut values = Vec::with_capacity(self.registry.len());
        let mut resolved_sources = Vec::with_capacity(self.registry.len());
        for (idx, def) in self.registry.defs().iter().enumerate() {
            if !def.domains.contains(domain) {
                values.push(None);
                resolved_sources.push(None);
                continue;
            }
            let id = SettingId(idx as u32);
            let mut winner: Option<(Value, &LayerSource)> = None;
            for source_index in &source_orders[profile_index(def.precedence)] {
                let (source, slots) = &prepared[*source_index];
                if let Some(value) = slots[idx] {
                    if let Some(error) = source_rejection(def, source) {
                        errors.push(error);
                        continue;
                    }
                    match self.registry.validate_value(id, value.clone()) {
                        Ok(value) => winner = Some((value, source)),
                        Err(e) => errors.push(e),
                    }
                }
            }
            if let Some((value, source)) = winner {
                values.push(Some(value));
                resolved_sources.push(Some(ValueSource {
                    layer: source.layer,
                    trust: source.trust,
                    source_id: source.source_id.clone(),
                }));
            } else {
                if def.flags.required {
                    errors.push(SettingsError::MissingRequiredSetting(def.key.clone()));
                }
                values.push(Some(def.default.clone()));
                resolved_sources.push(Some(ValueSource {
                    layer: LayerKind::EngineDefault,
                    trust: TrustLevel::Trusted,
                    source_id: "schema-default".into(),
                }));
            }
        }
        if errors.is_empty() {
            self.validate_cross(domain, &values, &mut errors);
        }
        if !errors.is_empty() {
            return Err(errors);
        }
        Ok(Snapshot(Arc::new(SnapshotData {
            domain,
            generation,
            values,
            sources: resolved_sources,
        })))
    }

    fn validate_cross(
        &self,
        domain: Domain,
        values: &[Option<Value>],
        errors: &mut Vec<SettingsError>,
    ) {
        for (idx, def) in self.registry.defs().iter().enumerate() {
            if !def.domains.contains(domain) {
                continue;
            }
            let Some(value) = values[idx].as_ref() else {
                continue;
            };
            for rule in &def.cross_rules {
                let (key, condition, requires) = match rule {
                    CrossRule::Requires { key, when } => (key, *when, true),
                    CrossRule::Conflicts { key, when } => (key, *when, false),
                };
                let active = value.is_enabled() == condition;
                if !active {
                    continue;
                }
                let Some((other_id, _)) = self.registry.get(key) else {
                    continue;
                };
                let other = values[other_id.0 as usize]
                    .as_ref()
                    .is_some_and(Value::is_enabled);
                if (requires && !other) || (!requires && other) {
                    errors.push(SettingsError::CrossConstraint {
                        key: def.key.clone(),
                        referenced: key.clone(),
                    });
                }
            }
        }
    }

    /// Inspects effective and per-layer values without changing resolver or snapshot state.
    pub fn inspect(
        &self,
        snapshot: &Snapshot,
        key: &SettingKey,
        sources: &[LayerSource],
    ) -> Result<InspectResult, SettingsError> {
        let (id, def) = self
            .registry
            .get(key)
            .ok_or_else(|| SettingsError::InvalidKey(key.clone()))?;
        if !def.domains.contains(snapshot.domain()) {
            return Err(SettingsError::WrongDomain {
                key: def.key.clone(),
                domain: snapshot.domain(),
            });
        }
        let mut by_layer = BTreeMap::new();
        let mut diagnostics = Vec::new();
        let allowed = crate::layer_order(snapshot.domain(), def.precedence);
        for layer in snapshot.domain().layers() {
            let mut candidates: Vec<(&LayerSource, &Value)> = Vec::new();
            for source in sources.iter().filter(|s| s.layer == *layer) {
                if let Some((candidate_key, value)) = source
                    .values
                    .iter()
                    .find(|(candidate_key, _)| self.registry.id(candidate_key) == Some(id))
                {
                    let _ = candidate_key;
                    if !allowed.contains(layer) {
                        diagnostics.push(SettingsError::PrecedenceExcluded {
                            key: def.key.clone(),
                            layer: *layer,
                            profile: def.precedence,
                        });
                    } else {
                        candidates.push((source, value));
                    }
                }
            }
            candidates.sort_by(|a, b| {
                a.0.order
                    .cmp(&b.0.order)
                    .then_with(|| b.0.source_id.cmp(&a.0.source_id))
            });
            let mut selected = None;
            for (source, value) in candidates {
                if let Some(error) = source_rejection(def, source) {
                    diagnostics.push(error);
                    continue;
                }
                match self.registry.validate_value(id, value.clone()) {
                    Ok(value) => {
                        selected = Some((
                            value,
                            ValueSource {
                                layer: source.layer,
                                trust: source.trust,
                                source_id: source.source_id.clone(),
                            },
                        ))
                    }
                    Err(error) => diagnostics.push(error),
                }
            }
            if selected.is_none() && *layer == LayerKind::EngineDefault {
                selected = Some((
                    def.default.clone(),
                    ValueSource {
                        layer: LayerKind::EngineDefault,
                        trust: TrustLevel::Trusted,
                        source_id: "schema-default".into(),
                    },
                ));
            }
            by_layer.insert(*layer, selected);
        }
        let source = snapshot.source(id).cloned();
        let locked_by = source
            .as_ref()
            .filter(|s| s.layer == LayerKind::Mandatory)
            .cloned();
        Ok(InspectResult {
            effective: snapshot.get(id).cloned(),
            source,
            default: def.default.clone(),
            by_layer,
            diagnostics,
            modifiable: locked_by.is_none()
                && !def.flags.read_only
                && def.precedence != PrecedenceProfile::Locked,
            locked_by,
        })
    }
}

#[cfg(test)]
mod required_setting_tests {
    use super::*;
    use crate::{RegistryBuilder, SettingDef};

    #[test]
    fn required_setting_rejects_engine_default_but_accepts_a_source_value() {
        let mut definition =
            SettingDef::new("render.device", "render", Value::String("auto".into()));
        definition.flags.required = true;
        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test").unwrap();
        builder.register(definition);
        let registry = builder.finish().unwrap();

        let missing = Resolver::new(&registry).resolve(Domain::Runtime, &[], 1);
        assert!(matches!(
            missing,
            Err(errors) if errors.iter().any(|error| matches!(
                error,
                SettingsError::MissingRequiredSetting(key) if key.as_str() == "render.device"
            ))
        ));

        let mut source = LayerSource::new(LayerKind::User, TrustLevel::User, "user");
        source.set("render.device", Value::String("gpu".into()));
        assert!(Resolver::new(&registry)
            .resolve(Domain::Runtime, &[source], 2)
            .is_ok());
    }
}

fn source_rejection(def: &crate::SettingDef, source: &LayerSource) -> Option<SettingsError> {
    if source.layer == LayerKind::Mandatory && source.trust != TrustLevel::Trusted {
        return Some(SettingsError::UntrustedMandatory {
            source_id: source.source_id.clone(),
            trust: source.trust,
        });
    }
    if def.flags.execution_class
        && source.layer != LayerKind::Mandatory
        && source.trust != TrustLevel::Trusted
    {
        return Some(SettingsError::UntrustedExecutionSetting {
            key: def.key.clone(),
            trust: source.trust,
        });
    }
    if def.flags.read_only && source.layer != LayerKind::Mandatory {
        return Some(SettingsError::ReadOnly(def.key.clone()));
    }
    if matches!(source.layer, LayerKind::Project | LayerKind::ProjectLocal)
        && !def.flags.project_overridable
    {
        return Some(SettingsError::ProjectOverrideDenied(def.key.clone()));
    }
    None
}

fn profile_index(profile: PrecedenceProfile) -> usize {
    match profile {
        PrecedenceProfile::UserChoiceWins => 0,
        PrecedenceProfile::ServerWins => 1,
        PrecedenceProfile::DeviceScoped => 2,
        PrecedenceProfile::Locked => 3,
    }
}

/// Computes a deterministic slot-by-slot diff.
pub fn diff(before: &Snapshot, after: &Snapshot) -> SettingsDiff {
    let mut changes = Vec::new();
    for i in 0..core::cmp::min(before.len(), after.len()) {
        let a = before.values_ref(i);
        let b = after.values_ref(i);
        let sa = before.sources_ref(i);
        let sb = after.sources_ref(i);
        if a != b || sa != sb {
            changes.push(SettingChange {
                id: SettingId(i as u32),
                before: a.cloned(),
                after: b.cloned(),
                before_source: sa.cloned(),
                after_source: sb.cloned(),
            });
        }
    }
    SettingsDiff { changes }
}

impl Snapshot {
    fn values_ref(&self, i: usize) -> Option<&Value> {
        self.0.values.get(i)?.as_ref()
    }
    fn sources_ref(&self, i: usize) -> Option<&ValueSource> {
        self.0.sources.get(i)?.as_ref()
    }
}

// Kept public so callers can compute the policy-adjusted layer order in tests and tooling.
/// Returns a profile-adjusted legal layer order for one definition.
pub fn layer_order(domain: Domain, profile: PrecedenceProfile) -> Vec<LayerKind> {
    let mut order = domain.layers().to_vec();
    if profile == PrecedenceProfile::Locked {
        order.retain(|l| matches!(l, LayerKind::EngineDefault | LayerKind::Mandatory));
    }
    let move_above = |order: &mut Vec<LayerKind>, high: LayerKind, low: LayerKind| {
        if let (Some(h), Some(l)) = (
            order.iter().position(|x| *x == high),
            order.iter().position(|x| *x == low),
        ) && h < l
        {
            let item = order.remove(h);
            let new_l = order.iter().position(|x| *x == low).unwrap_or(l);
            order.insert(new_l + 1, item);
        }
    };
    match profile {
        PrecedenceProfile::ServerWins => move_above(&mut order, LayerKind::Remote, LayerKind::User),
        PrecedenceProfile::DeviceScoped => {
            move_above(&mut order, LayerKind::Device, LayerKind::User)
        }
        _ => {}
    }
    order
}
