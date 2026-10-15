//! Versioned prefab archives are checked through the public World API.
#![cfg(feature = "serialize")]

use std::collections::BTreeMap;

use prism_ecs::component::Component;
use prism_ecs::entity::Entity;
use prism_ecs::prefab::reflect::{PrefabComponentOverride, PrefabOverrides, PrefabReflectError};
use prism_ecs::prefab::PrefabError;
use prism_ecs::reflect_archive::{
    ReflectArchiveError as Error, ReflectArchiveLoadOptions as Options,
};
use prism_ecs::world::World;
use prism_reflect::{
    FromReflect, ParsedPath, Patch, Reflect, ReflectMut, StableTypeId, TypeRegistry,
};

// Independent envelope codec for adversarial fixtures. Reflection payloads and
// schema descriptors are retained byte-for-byte from a known valid archive.
#[derive(Clone, Debug)]
struct RawComponent {
    name: String,
    id: u64,
    value: Vec<u8>,
}
#[derive(Clone, Debug)]
struct RawField {
    canonical: Vec<u8>,
    spelling: Vec<u8>,
    value: RawComponent,
}
#[derive(Clone, Debug)]
struct RawPolicy {
    name: String,
    id: u64,
    schema: Vec<u8>,
    tag: u8,
    fields: Vec<RawField>,
}
#[derive(Clone, Debug)]
struct RawEntity {
    source: u64,
    components: Vec<RawComponent>,
    parents: Vec<u64>,
    policies: Vec<RawPolicy>,
}

struct Cursor<'a>(&'a [u8]);
impl Cursor<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        a
    }
    fn u32(&mut self) -> usize {
        u32::from_le_bytes(self.take(4).try_into().unwrap()) as usize
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn blob(&mut self) -> Vec<u8> {
        let n = self.u32();
        self.take(n).to_vec()
    }
    fn ty(&mut self) -> (String, u64) {
        let n = u16::from_le_bytes(self.take(2).try_into().unwrap()) as usize;
        let name = String::from_utf8(self.take(n).to_vec()).unwrap();
        (name, self.u64())
    }
    fn component(&mut self) -> RawComponent {
        let (name, id) = self.ty();
        RawComponent {
            name,
            id,
            value: self.blob(),
        }
    }
}
fn unpack(bytes: &[u8]) -> Vec<RawEntity> {
    let mut c = Cursor(bytes);
    assert_eq!(c.take(5), b"PECA\x03");
    let n = c.u32();
    let records = (0..n)
        .map(|_| {
            let source = c.u64();
            let components = (0..c.u32()).map(|_| c.component()).collect();
            let parents = (0..c.u32()).map(|_| c.u64()).collect();
            let policies = (0..c.u32())
                .map(|_| {
                    let (name, id) = c.ty();
                    let schema = c.blob();
                    let tag = c.take(1)[0];
                    let fields = if tag == 1 {
                        (0..c.u32())
                            .map(|_| RawField {
                                canonical: c.blob(),
                                spelling: c.blob(),
                                value: c.component(),
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    RawPolicy {
                        name,
                        id,
                        schema,
                        tag,
                        fields,
                    }
                })
                .collect();
            RawEntity {
                source,
                components,
                parents,
                policies,
            }
        })
        .collect();
    assert!(c.0.is_empty());
    records
}
fn count(out: &mut Vec<u8>, n: usize) {
    out.extend_from_slice(&(n as u32).to_le_bytes());
}
fn blob(out: &mut Vec<u8>, bytes: &[u8]) {
    count(out, bytes.len());
    out.extend_from_slice(bytes);
}
fn ty(out: &mut Vec<u8>, name: &str, id: u64) {
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(&id.to_le_bytes());
}
fn component(out: &mut Vec<u8>, c: &RawComponent) {
    ty(out, &c.name, c.id);
    blob(out, &c.value);
}
fn pack(records: &[RawEntity], version: u8) -> Vec<u8> {
    let mut out = b"PECA".to_vec();
    out.push(version);
    count(&mut out, records.len());
    for r in records {
        out.extend_from_slice(&r.source.to_le_bytes());
        count(&mut out, r.components.len());
        for c in &r.components {
            component(&mut out, c);
        }
        if version == 2 {
            continue;
        }
        count(&mut out, r.parents.len());
        for parent in &r.parents {
            out.extend_from_slice(&parent.to_le_bytes());
        }
        count(&mut out, r.policies.len());
        for p in &r.policies {
            ty(&mut out, &p.name, p.id);
            blob(&mut out, &p.schema);
            out.push(p.tag);
            if p.tag == 1 {
                count(&mut out, p.fields.len());
                for f in &p.fields {
                    blob(&mut out, &f.canonical);
                    blob(&mut out, &f.spelling);
                    component(&mut out, &f.value);
                }
            }
        }
    }
    out
}
fn fixture() -> Vec<RawEntity> {
    let (mut source, id) = world();
    let parent = source.spawn(Stats { hp: 1, mana: 2 });
    let child = source.spawn_instance_of(parent);
    source
        .override_prefab_field(child, id, "hp", &9i32)
        .unwrap();
    unpack(&source.save_reflect_archive([child]).unwrap())
}

#[test]
fn allocating_reversed_missing_source_ids_never_merges_records() {
    let (mut source, _) = world();
    let a = source.spawn(Stats { hp: 1, mana: 2 });
    let b = source.spawn(Stats { hp: 3, mana: 4 });
    let bytes = source.save_reflect_archive([b, a]).unwrap();
    let (mut target, _) = world();
    let map = target
        .load_reflect_archive_with_options(&bytes, Options::default())
        .unwrap();
    assert_ne!(map[0].1, map[1].1);
    assert_eq!(target.entity_count(), 2);
    assert_eq!(target.get::<Stats>(map[0].1).unwrap().hp, 3);
    assert_eq!(target.get::<Stats>(map[1].1).unwrap().hp, 1);
}

#[derive(Reflect, Clone, Debug, PartialEq)]
struct Extra(i32);
impl Component for Extra {}
#[derive(Debug, PartialEq)]
struct Private(i32);
impl Component for Private {}

#[test]
fn v3_replaces_reflected_state_and_prefabs_while_v2_preserves_merge_behavior() {
    let (mut source, _) = world();
    let entity = source.spawn(Stats { hp: 1, mana: 2 });
    let records = unpack(&source.save_reflect_archive([entity]).unwrap());
    for version in [2, 3] {
        let (mut target, id) = world();
        target.register_reflect_component::<Extra>();
        let live = target.spawn((Stats { hp: 3, mana: 4 }, Extra(7), Private(8)));
        assert_eq!(live, entity);
        let parent = target.spawn(Stats { hp: 5, mana: 6 });
        target.add_prefab(live, parent);
        target
            .override_prefab_field(live, id, "hp", &99i32)
            .unwrap();
        assert_eq!(
            target
                .load_reflect_archive(&pack(&records, version))
                .unwrap(),
            vec![live]
        );
        assert_eq!(target.get::<Stats>(live).unwrap().hp, 1);
        assert_eq!(target.get::<Private>(live), Some(&Private(8)));
        assert_eq!(target.has::<Extra>(live), version == 2);
        assert_eq!(target.has::<PrefabOverrides>(live), version == 2);
        if version == 2 {
            assert_eq!(target.prefabs_of(live), &[parent]);
        } else {
            assert!(target.prefabs_of(live).is_empty());
        }
    }
}

#[derive(Reflect, Clone, Debug, PartialEq)]
struct Links {
    target_bits: u64,
    ordinary: u64,
}
impl Component for Links {}
fn remap_links(
    id: StableTypeId,
    path: Option<&ParsedPath>,
    value: &mut dyn Reflect,
    map: &BTreeMap<Entity, Entity>,
) -> Result<(), Error> {
    if id != StableTypeId::of_path(core::any::type_name::<Links>()) {
        return Ok(());
    }
    let bits = match path {
        None => match value.reflect_mut() {
            ReflectMut::Struct(s) => s
                .field_mut("target_bits")
                .unwrap()
                .downcast_mut::<u64>()
                .unwrap(),
            _ => unreachable!(),
        },
        Some(p) if p == &ParsedPath::parse("target_bits").unwrap() => {
            value.downcast_mut::<u64>().unwrap()
        }
        _ => return Ok(()),
    };
    let source = Entity::from_bits(*bits).ok_or(Error::InvalidEntity(*bits))?;
    *bits = map
        .get(&source)
        .ok_or(Error::MissingPrefabBase(source))?
        .to_bits();
    Ok(())
}

#[test]
fn explicit_remap_visits_concrete_components_and_selected_override_subtrees() {
    let mut source = World::new();
    let id = source.register_reflect_component::<Links>();
    let referenced = source.spawn(());
    let parent = source.spawn(Links {
        target_bits: referenced.to_bits(),
        ordinary: referenced.to_bits(),
    });
    // Make the empty referenced entity an ancestor so the archive includes it.
    source.add_prefab(parent, referenced);
    let child = source.spawn_instance_of(parent);
    source
        .override_prefab_field(child, id, "target_bits", &parent.to_bits())
        .unwrap();
    let bytes = source.save_reflect_archive([child]).unwrap();
    let mut target = World::new();
    target.register_reflect_component::<Links>();
    for _ in 0..5 {
        target.spawn(());
    }
    let map: BTreeMap<_, _> = target
        .load_reflect_archive_with_options(
            &bytes,
            Options {
                spawn_fresh: true,
                remap: Some(remap_links),
                ..Options::default()
            },
        )
        .unwrap()
        .into_iter()
        .collect();
    let concrete = target.get::<Links>(map[&parent]).unwrap();
    assert_eq!(concrete.target_bits, map[&referenced].to_bits());
    assert_eq!(
        concrete.ordinary,
        referenced.to_bits(),
        "ordinary integers are never remapped implicitly"
    );
    let value = target
        .get_inherited_reflect(map[&child], id)
        .unwrap()
        .unwrap();
    let resolved = Links::from_reflect(&*value).unwrap();
    assert_eq!(resolved.target_bits, map[&parent].to_bits());
    assert_eq!(resolved.ordinary, referenced.to_bits());
}

#[test]
fn remap_failure_leaves_live_records_untouched_and_reclaims_new_entities() {
    fn fail(
        _: StableTypeId,
        _: Option<&ParsedPath>,
        _: &mut dyn Reflect,
        _: &BTreeMap<Entity, Entity>,
    ) -> Result<(), Error> {
        Err(Error::InvalidPrefabMetadata(
            "application reference missing",
        ))
    }
    let (mut source, _) = world();
    let a = source.spawn(Stats { hp: 1, mana: 2 });
    let b = source.spawn(Stats { hp: 3, mana: 4 });
    let bytes = source.save_reflect_archive([a, b]).unwrap();
    let (mut target, _) = world();
    let live = target.spawn(Stats { hp: 99, mana: 88 });
    assert_eq!(a, live);
    assert!(target
        .load_reflect_archive_with_options(
            &bytes,
            Options {
                remap: Some(fail),
                ..Options::default()
            }
        )
        .is_err());
    assert_eq!(target.entity_count(), 1);
    assert_eq!(target.get::<Stats>(live), Some(&Stats { hp: 99, mana: 88 }));
}

#[test]
fn missing_parents_and_direct_or_indirect_cycles_fail_before_destination_writes() {
    let records = fixture();
    for case in 0..4 {
        let mut bad = records.clone();
        match case {
            0 => {
                bad.pop();
            }
            1 => bad[0].parents = vec![bad[0].source],
            2 => bad[1].parents = vec![bad[0].source],
            _ => {
                let parent = bad[0].parents[0];
                bad[0].parents.push(parent);
            }
        }
        let (mut target, _) = world();
        let live = target.spawn(Stats { hp: 50, mana: 60 });
        // Even a live destination entity cannot supply an omitted archive base.
        target.spawn(());
        let result = target.load_reflect_archive(&pack(&bad, 3));
        match case {
            0 => assert!(matches!(result, Err(Error::MissingPrefabBase(_)))),
            1 | 2 => assert!(matches!(
                result,
                Err(Error::Prefab(PrefabReflectError::Ancestry(
                    PrefabError::Cycle(_)
                )))
            )),
            _ => assert!(matches!(
                result,
                Err(Error::InvalidPrefabMetadata("duplicate parent"))
            )),
        }
        assert_eq!(target.entity_count(), 2);
        assert_eq!(target.get::<Stats>(live).unwrap().hp, 50);
    }
}

#[test]
fn malformed_policies_paths_types_and_duplicate_records_are_rejected() {
    let records = fixture();
    let cases: &[fn(&mut Vec<RawEntity>)] = &[
        |r| r.push(r[0].clone()),
        |r| {
            let c = r[1].components[0].clone();
            r[1].components.push(c);
        },
        |r| {
            let p = r[0].policies[0].clone();
            r[0].policies.push(p);
        },
        |r| r[0].policies[0].fields.clear(),
        |r| r[0].policies[0].tag = 255,
        |r| r[0].policies[0].schema[0] ^= 1,
        |r| {
            let f = r[0].policies[0].fields[0].clone();
            r[0].policies[0].fields.push(f);
        },
        |r| r[0].policies[0].fields[0].canonical = b"unknown".to_vec(),
        |r| r[0].policies[0].fields[0].spelling = b"mana".to_vec(),
        |r| r[0].policies[0].fields[0].canonical.clear(),
        |r| r[0].policies[0].fields[0].spelling = vec![255],
        |r| r[0].policies[0].fields[0].value.id ^= 1,
        |r| {
            let f = &mut r[0].policies[0].fields[0];
            f.value.name = "u32".into();
            f.value.id = StableTypeId::of_path("u32").value();
        },
        |r| r[0].policies[0].fields[0].value.value.clear(),
        |r| r[0].source = 0,
    ];
    for (i, mutate) in cases.iter().enumerate() {
        let mut bad = records.clone();
        mutate(&mut bad);
        let (mut target, _) = world();
        let live = target.spawn(Stats { hp: 50, mana: 60 });
        let result = target.load_reflect_archive(&pack(&bad, 3));
        assert!(result.is_err(), "malformed case {i} unexpectedly accepted");
        assert_eq!(
            target.entity_count(),
            1,
            "malformed case {i} allocated entities"
        );
        assert_eq!(target.get::<Stats>(live).unwrap().hp, 50);
    }
}

#[test]
fn every_truncation_trailing_bytes_and_envelope_budgets_are_rejected() {
    use prism_ecs::reflect_archive::ReflectArchiveLimits;
    let bytes = pack(&fixture(), 3);
    for end in 0..bytes.len() {
        let (mut target, _) = world();
        assert!(
            target.load_reflect_archive(&bytes[..end]).is_err(),
            "accepted prefix {end}"
        );
        assert_eq!(target.entity_count(), 0);
    }
    let (mut target, _) = world();
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(matches!(
        target.load_reflect_archive(&trailing),
        Err(Error::TrailingData)
    ));
    let limits = [
        ReflectArchiveLimits {
            max_bytes: bytes.len() - 1,
            ..Default::default()
        },
        ReflectArchiveLimits {
            max_entities: 1,
            ..Default::default()
        },
        ReflectArchiveLimits {
            max_records: 2,
            ..Default::default()
        },
        ReflectArchiveLimits {
            max_path_depth: 0,
            ..Default::default()
        },
        ReflectArchiveLimits {
            max_validation_steps: 0,
            ..Default::default()
        },
    ];
    for limit in limits {
        assert!(matches!(
            target.load_reflect_archive_with_options(
                &bytes,
                Options {
                    limits: limit,
                    ..Default::default()
                }
            ),
            Err(Error::LimitExceeded(_))
        ));
        assert_eq!(target.entity_count(), 0);
    }
}

#[test]
fn dormant_invalid_paths_are_rejected_without_a_concrete_base() {
    let mut records = fixture();
    records[1].components.clear();
    let (mut target, _) = world();
    assert!(target.load_reflect_archive(&pack(&records, 3)).is_ok());
    records[0].policies[0].fields[0].canonical = b"absent".to_vec();
    records[0].policies[0].fields[0].spelling = b"absent".to_vec();
    let (mut target, _) = world();
    assert!(matches!(
        target.load_reflect_archive(&pack(&records, 3)),
        Err(Error::Prefab(PrefabReflectError::MissingField(_)))
    ));
    assert_eq!(target.entity_count(), 0);
}

#[test]
fn ownership_equality_uses_paths_values_and_order_independent_policies() {
    let (mut source, id) = world();
    let a = source.spawn(Stats { hp: 1, mana: 2 });
    let b = source.spawn(Stats { hp: 3, mana: 4 });
    source.override_prefab_field(a, id, ".hp", &9i32).unwrap();
    source.override_prefab_field(a, id, "mana", &8i32).unwrap();
    source.override_prefab_field(b, id, ".mana", &8i32).unwrap();
    source.override_prefab_field(b, id, "hp", &9i32).unwrap();
    assert_eq!(
        source.get::<PrefabOverrides>(a),
        source.get::<PrefabOverrides>(b)
    );
    source.override_prefab_field(b, id, "hp", &10i32).unwrap();
    assert_ne!(
        source.get::<PrefabOverrides>(a),
        source.get::<PrefabOverrides>(b)
    );
    source.remove_prefab_component(b, id).unwrap();
    assert_ne!(
        source.get::<PrefabOverrides>(a),
        source.get::<PrefabOverrides>(b)
    );
}

#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
struct Stats {
    hp: i32,
    mana: i32,
}
impl Component for Stats {}

fn world() -> (World, StableTypeId) {
    let mut world = World::new();
    let id = world.register_reflect_component::<Stats>();
    (world, id)
}

fn stats(world: &World, entity: Entity, id: StableTypeId) -> Option<Stats> {
    world
        .get_inherited_reflect(entity, id)
        .unwrap()
        .map(|value| Stats::from_reflect(&*value).unwrap())
}

fn fresh(world: &mut World, bytes: &[u8]) -> BTreeMap<Entity, Entity> {
    world
        .load_reflect_archive_with_options(
            bytes,
            Options {
                spawn_fresh: true,
                ..Options::default()
            },
        )
        .unwrap()
        .into_iter()
        .collect()
}

#[test]
fn inherited_fields_and_explicit_ownership_survive_fresh_entity_remap() {
    let (mut source, id) = world();
    let base = source.spawn(Stats { hp: 10, mana: 20 });
    let instance = source.spawn_instance_of(base);
    source
        .override_prefab_field(instance, id, "hp", &99i32)
        .unwrap();
    let bytes = source.save_reflect_archive([instance, instance]).unwrap();
    assert_eq!(bytes[4], 3);
    assert_eq!(bytes, source.save_reflect_archive([instance]).unwrap());

    let (mut target, _) = world();
    for _ in 0..5 {
        target.spawn(());
    }
    let mapping = fresh(&mut target, &bytes);
    assert_eq!(
        mapping.len(),
        2,
        "instance-only save includes its base once"
    );
    let (child, parent) = (mapping[&instance], mapping[&base]);
    assert_eq!(target.prefabs_of(child), &[parent]);
    assert_ne!(parent, base);
    assert!(target.get::<Stats>(child).is_none());
    assert_eq!(
        target.get::<PrefabOverrides>(child),
        source.get::<PrefabOverrides>(instance)
    );
    target.insert(parent, Stats { hp: 30, mana: 40 });
    assert_eq!(stats(&target, child, id), Some(Stats { hp: 99, mana: 40 }));
    assert!(target
        .clear_prefab_field_override(child, id, ".hp")
        .unwrap());
    assert_eq!(stats(&target, child, id), Some(Stats { hp: 30, mana: 40 }));
}

#[test]
fn ordered_multiple_parents_diamonds_and_empty_ancestors_round_trip() {
    let (mut source, id) = world();
    let base = source.spawn(Stats { hp: 1, mana: 2 });
    let left = source.spawn_instance_of(base);
    let right = source.spawn_instance_of(base);
    let empty = source.spawn(());
    source.override_prefab_field(left, id, "hp", &3i32).unwrap();
    source
        .override_prefab_field(right, id, "hp", &4i32)
        .unwrap();
    let leaf = source.spawn_instance_of(empty);
    source.add_prefab(leaf, right);
    source.add_prefab(leaf, left);
    let bytes = source.save_reflect_archive([leaf]).unwrap();
    let (mut target, _) = world();
    let landed = target
        .load_reflect_archive_with_options(&bytes, Options::default())
        .unwrap();
    assert_eq!(
        landed.iter().map(|p| p.0).collect::<Vec<_>>(),
        vec![leaf, empty, right, left, base]
    );
    let map: BTreeMap<_, _> = landed.into_iter().collect();
    assert_eq!(
        target.prefabs_of(map[&leaf]),
        &[map[&empty], map[&right], map[&left]]
    );
    assert_eq!(target.prefabs_of(map[&left]), &[map[&base]]);
    assert_eq!(target.prefabs_of(map[&right]), &[map[&base]]);
    assert_eq!(
        stats(&target, map[&leaf], id),
        Some(Stats { hp: 4, mana: 2 })
    );
}

#[test]
fn removal_only_and_dormant_field_policies_reactivate_and_reset() {
    let (mut source, id) = world();
    let base = source.spawn(Stats { hp: 1, mana: 2 });
    let dormant = source.spawn_instance_of(base);
    source
        .override_prefab_field(dormant, id, "mana", &8i32)
        .unwrap();
    let removed = source.spawn_instance_of(base);
    source.remove_prefab_component(removed, id).unwrap();
    let standalone = source.spawn(());
    source.remove_prefab_component(standalone, id).unwrap();
    source.remove::<Stats>(base);
    let bytes = source
        .save_reflect_archive([dormant, removed, standalone])
        .unwrap();
    let (mut target, _) = world();
    let map = fresh(&mut target, &bytes);
    assert_eq!(map.len(), 4, "empty base and removal-only entity survive");
    assert_eq!(stats(&target, map[&dormant], id), None);
    target.insert(map[&base], Stats { hp: 10, mana: 20 });
    assert_eq!(
        stats(&target, map[&dormant], id),
        Some(Stats { hp: 10, mana: 8 })
    );
    assert_eq!(stats(&target, map[&removed], id), None);
    assert!(target.reset_prefab_component(map[&removed], id).unwrap());
    assert_eq!(
        stats(&target, map[&removed], id),
        Some(Stats { hp: 10, mana: 20 })
    );
    assert!(matches!(
        target
            .get::<PrefabOverrides>(map[&standalone])
            .unwrap()
            .component(id),
        Some(PrefabComponentOverride::Removed)
    ));
    assert!(target.reset_prefab_component(map[&standalone], id).unwrap());
    assert!(!target.has::<PrefabOverrides>(map[&standalone]));
}

#[test]
fn concrete_local_components_coexist_with_fields_and_removal_metadata() {
    let (mut source, id) = world();
    let local = source.spawn(Stats { hp: 1, mana: 2 });
    source
        .override_prefab_field(local, id, "hp", &8i32)
        .unwrap();
    let removed = source.spawn(());
    source.remove_prefab_component(removed, id).unwrap();
    source.insert(removed, Stats { hp: 3, mana: 4 });
    let bytes = source.save_reflect_archive([local, removed]).unwrap();
    let (mut target, _) = world();
    let map = fresh(&mut target, &bytes);
    assert_eq!(
        target.get::<Stats>(map[&local]),
        Some(&Stats { hp: 1, mana: 2 })
    );
    assert_eq!(
        stats(&target, map[&local], id),
        Some(Stats { hp: 8, mana: 2 })
    );
    assert_eq!(
        target.get::<Stats>(map[&removed]),
        Some(&Stats { hp: 3, mana: 4 })
    );
    assert_eq!(stats(&target, map[&removed], id), None);
}

#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
enum Mode {
    Named { x: i32, y: i32 },
    Off,
}
#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
struct Config {
    numbers: Vec<i32>,
    mode: Mode,
    nested: Stats,
}
impl Component for Config {}

fn config_types() -> TypeRegistry {
    let mut types = TypeRegistry::new();
    types.register::<Vec<i32>>();
    types.register::<Mode>();
    types.register::<Stats>();
    types
}

#[test]
fn nested_container_replacements_and_named_enum_aliases_are_field_only() {
    let mut source = World::new();
    let id = source.register_reflect_component::<Config>();
    let base = source.spawn(Config {
        numbers: vec![1, 2, 3],
        mode: Mode::Named { x: 1, y: 2 },
        nested: Stats { hp: 3, mana: 4 },
    });
    let child = source.spawn_instance_of(base);
    source
        .override_prefab_field(child, id, "numbers", &vec![9i32])
        .unwrap();
    source
        .override_prefab_field(child, id, "mode#0", &7i32)
        .unwrap();
    source
        .override_prefab_field(child, id, "nested.hp", &6i32)
        .unwrap();
    let switched = source.spawn_instance_of(base);
    source
        .override_prefab_field(switched, id, "mode", &Mode::Off)
        .unwrap();
    let types = config_types();
    let bytes = source
        .save_reflect_archive_with_types([child, switched], &types)
        .unwrap();
    let mut target = World::new();
    target.register_reflect_component::<Config>();
    let map: BTreeMap<_, _> = target
        .load_reflect_archive_with_options(
            &bytes,
            Options {
                types: Some(&types),
                ..Options::default()
            },
        )
        .unwrap()
        .into_iter()
        .collect();
    let value = target
        .get_inherited_reflect(map[&child], id)
        .unwrap()
        .unwrap();
    let config = Config::from_reflect(&*value).unwrap();
    assert_eq!(config.numbers, vec![9]);
    assert_eq!(config.mode, Mode::Named { x: 7, y: 2 });
    assert_eq!(config.nested, Stats { hp: 6, mana: 4 });
    let PrefabComponentOverride::Fields(fields) = target
        .get::<PrefabOverrides>(map[&child])
        .unwrap()
        .component(id)
        .unwrap()
    else {
        panic!()
    };
    assert!(fields
        .iter()
        .all(|f| matches!(f.patch(), Patch::Replace(_))));
    assert!(fields
        .iter()
        .any(|f| f.path_str() == "mode#0" && f.path() == &ParsedPath::parse("mode.x").unwrap()));
    let value = target
        .get_inherited_reflect(map[&switched], id)
        .unwrap()
        .unwrap();
    assert_eq!(Config::from_reflect(&*value).unwrap().mode, Mode::Off);
    target.insert(
        map[&base],
        Config {
            numbers: vec![3, 4],
            mode: Mode::Off,
            nested: Stats { hp: 5, mana: 6 },
        },
    );
    assert!(target
        .clear_prefab_field_override(map[&child], id, "mode#0")
        .unwrap());
    assert!(target.get_inherited_reflect(map[&child], id).is_ok());
}

#[test]
fn reordered_dynamic_struct_and_named_enum_overrides_keep_named_values() {
    use prism_reflect::{DynamicEnum, DynamicStruct, DynamicVariant};
    let mut source = World::new();
    let id = source.register_reflect_component::<Config>();
    let base = source.spawn(Config {
        numbers: vec![],
        mode: Mode::Named { x: 1, y: 2 },
        nested: Stats { hp: 3, mana: 4 },
    });
    let child = source.spawn_instance_of(base);
    let mut nested = DynamicStruct::new();
    nested.set_represented_type_name(core::any::type_name::<Stats>());
    nested.insert("mana", 80i32);
    nested.insert("hp", 70i32);
    let mut mode = DynamicEnum::new(
        0,
        "Named",
        DynamicVariant::Struct(vec![("y", Box::new(60i32)), ("x", Box::new(50i32))]),
    );
    mode.set_represented_type_name(core::any::type_name::<Mode>());
    source
        .override_prefab_field(child, id, "nested", &nested)
        .unwrap();
    source
        .override_prefab_field(child, id, "mode", &mode)
        .unwrap();
    let types = config_types();
    let bytes = source
        .save_reflect_archive_with_types([child], &types)
        .unwrap();
    let mut target = World::new();
    target.register_reflect_component::<Config>();
    let map: BTreeMap<_, _> = target
        .load_reflect_archive_with_options(
            &bytes,
            Options {
                types: Some(&types),
                ..Options::default()
            },
        )
        .unwrap()
        .into_iter()
        .collect();
    let value = target
        .get_inherited_reflect(map[&child], id)
        .unwrap()
        .unwrap();
    let restored = Config::from_reflect(&*value).unwrap();
    assert_eq!(restored.nested, Stats { hp: 70, mana: 80 });
    assert_eq!(restored.mode, Mode::Named { x: 50, y: 60 });
    assert_eq!(
        target.get::<PrefabOverrides>(map[&child]),
        source.get::<PrefabOverrides>(child)
    );
}

#[test]
fn remapping_can_reorder_dynamic_fields_but_cannot_change_their_schema() {
    use prism_reflect::DynamicStruct;
    fn reorder(
        _: StableTypeId,
        _: Option<&ParsedPath>,
        value: &mut dyn Reflect,
        _: &BTreeMap<Entity, Entity>,
    ) -> Result<(), Error> {
        let value = value.downcast_mut::<DynamicStruct>().unwrap();
        let hp = value.remove("hp").unwrap();
        value.insert_boxed("hp", hp);
        Ok(())
    }
    fn corrupt(
        _: StableTypeId,
        _: Option<&ParsedPath>,
        value: &mut dyn Reflect,
        _: &BTreeMap<Entity, Entity>,
    ) -> Result<(), Error> {
        let value = value.downcast_mut::<DynamicStruct>().unwrap();
        value.remove("hp");
        value.insert("unknown", 7i32);
        Ok(())
    }
    fn rename(
        _: StableTypeId,
        _: Option<&ParsedPath>,
        value: &mut dyn Reflect,
        _: &BTreeMap<Entity, Entity>,
    ) -> Result<(), Error> {
        value
            .downcast_mut::<DynamicStruct>()
            .unwrap()
            .set_represented_type_name("not::Stats");
        Ok(())
    }
    let (mut source, _) = world();
    let entity = source.spawn(Stats { hp: 10, mana: 20 });
    let bytes = source.save_reflect_archive([entity]).unwrap();
    let (mut target, _) = world();
    let landed = target
        .load_reflect_archive_with_options(
            &bytes,
            Options {
                remap: Some(reorder),
                ..Options::default()
            },
        )
        .unwrap();
    assert_eq!(
        target.get::<Stats>(landed[0].1),
        Some(&Stats { hp: 10, mana: 20 })
    );
    for remap in [
        corrupt as prism_ecs::reflect_archive::ReflectArchiveRemapFn,
        rename,
    ] {
        let live = landed[0].1;
        target.insert(live, Stats { hp: 30, mana: 40 });
        assert!(target
            .load_reflect_archive_with_options(
                &bytes,
                Options {
                    remap: Some(remap),
                    ..Options::default()
                }
            )
            .is_err());
        assert_eq!(target.entity_count(), 1);
        assert_eq!(target.get::<Stats>(live), Some(&Stats { hp: 30, mana: 40 }));
    }
}

#[test]
fn nested_schema_registration_and_reflection_value_limits_fail_explicitly() {
    use prism_ecs::reflect_archive::ReflectArchiveLimits;
    use prism_reflect::{DeserializeError, DeserializeLimits};
    let mut source = World::new();
    let id = source.register_reflect_component::<Config>();
    let base = source.spawn(Config {
        numbers: vec![1, 2, 3],
        mode: Mode::Off,
        nested: Stats { hp: 1, mana: 2 },
    });
    let child = source.spawn_instance_of(base);
    source
        .override_prefab_field(child, id, "numbers", &vec![4i32, 5])
        .unwrap();
    assert!(matches!(
        source.save_reflect_archive([child]),
        Err(Error::Decode(DeserializeError::UnregisteredType(_)))
    ));
    let types = config_types();
    let bytes = source
        .save_reflect_archive_with_types([child], &types)
        .unwrap();
    let mut target = World::new();
    target.register_reflect_component::<Config>();
    assert!(matches!(
        target.load_reflect_archive(&bytes),
        Err(Error::Decode(DeserializeError::UnregisteredType(_)))
    ));
    assert_eq!(target.entity_count(), 0);
    for (values, collection) in [
        (DeserializeLimits::new(128, 1), true),
        (DeserializeLimits::new(1, 100), false),
    ] {
        let error = target
            .load_reflect_archive_with_options(
                &bytes,
                Options {
                    types: Some(&types),
                    limits: ReflectArchiveLimits {
                        values,
                        ..Default::default()
                    },
                    ..Options::default()
                },
            )
            .unwrap_err();
        if collection {
            assert!(matches!(
                error,
                Error::Decode(DeserializeError::CollectionTooLarge { .. })
            ));
        } else {
            assert!(matches!(
                error,
                Error::Decode(DeserializeError::DepthLimitExceeded { .. })
            ));
        }
        assert_eq!(target.entity_count(), 0);
    }
}
