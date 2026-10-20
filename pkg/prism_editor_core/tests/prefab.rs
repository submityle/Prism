//! Public API regression tests for actual authored composition and transactions.
extern crate alloc;
use alloc::{collections::BTreeMap, sync::Arc};
use prism_editor_core::{prefab::*, CommandStack};

fn aid(n: u128) -> PrefabId {
    PrefabId::from_raw(n)
}
fn iid(n: u128) -> PrefabInstanceId {
    PrefabInstanceId::from_raw(n)
}
fn ep(n: u128) -> PrefabEntityPath {
    PrefabEntityPath::local(PrefabEntityId::from_raw(n))
}
fn owner(n: u128) -> PrefabOwner {
    PrefabOwner::Asset(aid(n))
}
fn io(n: u128) -> PrefabOwner {
    PrefabOwner::Instance(iid(n))
}
fn property() -> PrefabProperty {
    PrefabProperty {
        entity: ep(1),
        component: "generic/schema@1".into(),
        fields: vec!["x".into()],
    }
}
fn val(n: i64) -> PrefabOverride {
    PrefabOverride::Set(Arc::new(PrefabValue::Signed(n)))
}
fn pin(o: PrefabOwner, n: i64) -> PrefabEdit {
    PrefabEdit::Override {
        owner: o,
        property: property(),
        value: val(n),
    }
}
fn asset(n: u128, base: Option<u128>) -> PrefabAsset {
    let mut layer = PrefabLayer::default();
    if base.is_none() {
        layer.entities.insert(
            PrefabEntityId::from_raw(1),
            PrefabEntity {
                parent: None,
                components: BTreeMap::from([(
                    "generic/schema@1".into(),
                    Arc::new(PrefabValue::Object(BTreeMap::from([
                        ("x".into(), PrefabValue::Signed(1)),
                        ("y".into(), PrefabValue::Signed(2)),
                    ]))),
                )]),
            },
        );
    }
    PrefabAsset {
        id: aid(n),
        base: base.map(aid),
        layer,
    }
}
fn insert(a: PrefabAsset) -> PrefabEdit {
    PrefabEdit::InsertAsset(Arc::new(a))
}
fn inst(n: u128, p: u128) -> PrefabEdit {
    PrefabEdit::InsertInstance(Arc::new(PrefabInstance {
        id: iid(n),
        prefab: aid(p),
        layer: PrefabLayer::default(),
    }))
}
fn batch(s: &mut PrefabStore, edits: &[PrefabEdit]) -> PrefabReceipt {
    s.transact(s.revision(), edits).unwrap()
}
fn read(s: &PrefabStore, o: PrefabOwner) -> i64 {
    match s.compose(o).unwrap().value(&property()).unwrap() {
        PrefabValue::Signed(n) => *n,
        _ => panic!("wrong exact type"),
    }
}
fn library() -> PrefabStore {
    let mut s = PrefabStore::new(PrefabLimits::default()).unwrap();
    batch(
        &mut s,
        &[
            insert(asset(1, None)),
            insert(asset(2, Some(1))),
            insert(asset(3, Some(2))),
            insert(asset(9, None)),
            inst(1, 3),
            inst(2, 3),
            inst(3, 9),
        ],
    );
    s
}

#[test]
fn live_inheritance_explicit_equal_pins_and_unrelated_identity() {
    let mut s = library();
    batch(&mut s, &[pin(io(1), 1)]); // Equality still pins.
    let untouched = s.instance(iid(3)).unwrap().clone();
    let original_value = s.asset(aid(1)).unwrap().layer.entities[&PrefabEntityId::from_raw(1)]
        .components["generic/schema@1"]
        .clone();
    let receipt = batch(&mut s, &[pin(owner(1), 8)]);
    assert_eq!(receipt.changed_records(), 1);
    assert_eq!(
        (read(&s, io(1)), read(&s, io(2)), read(&s, io(3))),
        (1, 8, 1)
    );
    assert!(Arc::ptr_eq(&untouched, s.instance(iid(3)).unwrap()));
    assert!(Arc::ptr_eq(
        &original_value,
        &s.asset(aid(1)).unwrap().layer.entities[&PrefabEntityId::from_raw(1)].components
            ["generic/schema@1"]
    ));
    batch(
        &mut s,
        &[PrefabEdit::Revert {
            owner: io(1),
            property: property(),
        }],
    );
    assert_eq!(read(&s, io(1)), 8);
}

#[test]
fn nested_variant_slots_are_collision_free_and_preserve_mount_parents() {
    let mut s = library();
    batch(&mut s, &[pin(owner(2), 7)]);
    let mut container = asset(10, None);
    for n in [41, 42] {
        container.layer.slots.insert(
            PrefabSlotId::from_raw(n),
            Some(PrefabSlot {
                prefab: aid(2),
                parent: Some(ep(1)),
            }),
        );
    }
    batch(&mut s, &[insert(container), inst(10, 10)]);
    let result = s.compose(io(10)).unwrap();
    assert_eq!(result.entities.len(), 3);
    for n in [41, 42] {
        let mut p = property();
        p.entity.slots.push(PrefabSlotId::from_raw(n));
        assert_eq!(result.value(&p), Some(&PrefabValue::Signed(7)));
        assert_eq!(result.entities[&p.entity].parent, Some(ep(1)));
    }
    let mut p = property();
    p.entity.slots.push(PrefabSlotId::from_raw(41));
    batch(
        &mut s,
        &[
            PrefabEdit::Override {
                owner: io(10),
                property: p.clone(),
                value: val(99),
            },
            pin(owner(1), 6),
        ],
    );
    assert_eq!(
        s.compose(io(10)).unwrap().value(&p),
        Some(&PrefabValue::Signed(99))
    );
    // A nested variant can itself contain another stable slot chain.
    let mut outer = asset(11, Some(10));
    outer.layer.slots.insert(
        PrefabSlotId::from_raw(43),
        Some(PrefabSlot {
            prefab: aid(10),
            parent: None,
        }),
    );
    batch(&mut s, &[insert(outer)]);
    p.entity.slots = vec![PrefabSlotId::from_raw(43), PrefabSlotId::from_raw(42)];
    assert_eq!(
        s.compose(owner(11)).unwrap().value(&p),
        Some(&PrefabValue::Signed(7))
    );
}

#[test]
fn apply_to_base_moves_override_and_rejects_intervening_conflicts_atomically() {
    let mut s = library();
    batch(&mut s, &[pin(io(1), 20)]);
    batch(
        &mut s,
        &[PrefabEdit::ApplyToBase {
            source: io(1),
            target: aid(1),
            property: property(),
        }],
    );
    assert!(!s.layer(io(1)).unwrap().overrides.contains_key(&property()));
    assert_eq!(
        (read(&s, io(1)), read(&s, io(2)), read(&s, io(3))),
        (20, 20, 1)
    );
    batch(&mut s, &[pin(owner(2), 5), pin(io(1), 30)]);
    let before = s.encode().unwrap();
    assert!(s
        .transact(
            s.revision(),
            &[PrefabEdit::ApplyToBase {
                source: io(1),
                target: aid(1),
                property: property()
            }]
        )
        .is_err());
    assert_eq!(s.encode().unwrap(), before);
    assert!(s
        .transact(
            s.revision(),
            &[PrefabEdit::ApplyToBase {
                source: io(1),
                target: aid(9),
                property: property()
            }]
        )
        .is_err());
}

#[test]
fn selected_down_propagation_keeps_other_instance_pins_and_variant_behavior() {
    let mut s = library();
    batch(
        &mut s,
        &[
            pin(owner(1), 10),
            pin(owner(2), 20),
            pin(io(1), 30),
            pin(io(2), 40),
        ],
    );
    let other = s.instance(iid(2)).unwrap().clone();
    batch(
        &mut s,
        &[PrefabEdit::Propagate {
            source: aid(1),
            property: property(),
            recipients: vec![io(1)],
        }],
    );
    assert_eq!(
        (read(&s, io(1)), read(&s, io(2)), read(&s, owner(2))),
        (10, 40, 20)
    );
    assert!(Arc::ptr_eq(&other, s.instance(iid(2)).unwrap()));
    batch(
        &mut s,
        &[PrefabEdit::Propagate {
            source: aid(1),
            property: property(),
            recipients: vec![owner(2)],
        }],
    );
    assert_eq!(read(&s, owner(3)), 10);
    assert_eq!(read(&s, io(2)), 40);
    let before = s.encode().unwrap();
    for recipients in [vec![io(3)], vec![io(1), io(1)]] {
        assert!(s
            .transact(
                s.revision(),
                &[PrefabEdit::Propagate {
                    source: aid(1),
                    property: property(),
                    recipients
                }]
            )
            .is_err());
        assert_eq!(s.encode().unwrap(), before);
    }
}

#[test]
fn broader_override_conflicts_without_losing_sibling_fields() {
    let mut s = library();
    let mut whole = property();
    whole.fields.clear();
    batch(
        &mut s,
        &[PrefabEdit::Override {
            owner: io(1),
            property: whole,
            value: PrefabOverride::Set(Arc::new(PrefabValue::Object(BTreeMap::from([
                ("x".into(), PrefabValue::Signed(50)),
                ("sibling".into(), PrefabValue::Bool(true)),
            ])))),
        }],
    );
    let before = s.encode().unwrap();
    assert!(s
        .transact(
            s.revision(),
            &[PrefabEdit::Propagate {
                source: aid(1),
                property: property(),
                recipients: vec![io(1)]
            }]
        )
        .is_err());
    assert_eq!(s.encode().unwrap(), before);
    assert!(s.transact(s.revision(), &[pin(io(1), 3)]).is_err());
    assert_eq!(s.encode().unwrap(), before);
}

#[test]
fn remove_revert_apply_and_absence_propagation_are_real() {
    let mut s = library();
    batch(
        &mut s,
        &[PrefabEdit::Override {
            owner: io(1),
            property: property(),
            value: PrefabOverride::Remove,
        }],
    );
    assert!(s.compose(io(1)).unwrap().value(&property()).is_none());
    batch(
        &mut s,
        &[PrefabEdit::ApplyToBase {
            source: io(1),
            target: aid(2),
            property: property(),
        }],
    );
    assert!(s.compose(io(2)).unwrap().value(&property()).is_none());
    batch(&mut s, &[pin(io(1), 80)]);
    batch(
        &mut s,
        &[PrefabEdit::Propagate {
            source: aid(2),
            property: property(),
            recipients: vec![io(1)],
        }],
    );
    assert!(s.compose(io(1)).unwrap().value(&property()).is_none());
    batch(
        &mut s,
        &[PrefabEdit::Revert {
            owner: owner(2),
            property: property(),
        }],
    );
    assert_eq!(read(&s, io(1)), 1);
    // Missing removals are not silently accepted.
    let mut absent = property();
    absent.fields = vec!["absent".into()];
    assert!(s
        .transact(
            s.revision(),
            &[PrefabEdit::Override {
                owner: io(1),
                property: absent,
                value: PrefabOverride::Remove
            }]
        )
        .is_err());
}

#[test]
fn atomic_structural_edits_reparent_and_remove_nested_slots() {
    let mut s = library();
    let mut a = asset(10, None);
    a.layer.slots.insert(
        PrefabSlotId::from_raw(1),
        Some(PrefabSlot {
            prefab: aid(1),
            parent: Some(ep(1)),
        }),
    );
    batch(&mut s, &[insert(a)]);
    let nested = PrefabEntityPath {
        slots: vec![PrefabSlotId::from_raw(1)],
        entity: PrefabEntityId::from_raw(1),
    };
    let mut v = asset(11, Some(10));
    v.layer.removed.insert(ep(1));
    v.layer.parents.insert(nested.clone(), None);
    batch(&mut s, &[insert(v)]);
    assert_eq!(s.compose(owner(11)).unwrap().entities[&nested].parent, None);
    let mut v = s.asset(aid(11)).unwrap().as_ref().clone();
    v.layer.parents.clear();
    v.layer.slots.insert(PrefabSlotId::from_raw(1), None);
    batch(&mut s, &[PrefabEdit::ReplaceAsset(Arc::new(v))]);
    assert!(s.compose(owner(11)).unwrap().entities.is_empty());
    let before = s.encode().unwrap();
    assert!(s
        .transact(s.revision(), &[PrefabEdit::RemoveAsset(aid(1))])
        .is_err());
    assert_eq!(s.encode().unwrap(), before);
}

#[test]
fn cycles_missing_references_duplicate_ids_and_late_failure_roll_back() {
    let mut s = library();
    let before = s.encode().unwrap();
    let mut cyclic = asset(10, None);
    cyclic.layer.slots.insert(
        PrefabSlotId::from_raw(1),
        Some(PrefabSlot {
            prefab: aid(11),
            parent: None,
        }),
    );
    let errors = [
        vec![pin(owner(1), 5), insert(asset(1, None))],
        vec![insert(cyclic), insert(asset(11, Some(10)))],
        vec![insert(asset(10, Some(999)))],
        vec![PrefabEdit::RemoveInstance(iid(999))],
        vec![
            pin(owner(1), 5),
            PrefabEdit::Revert {
                owner: io(1),
                property: property(),
            },
        ],
    ];
    for edits in errors {
        assert!(s.transact(s.revision(), &edits).is_err());
        assert_eq!(s.encode().unwrap(), before);
    }
    let mut cyclic = asset(10, None);
    cyclic
        .layer
        .entities
        .get_mut(&PrefabEntityId::from_raw(1))
        .unwrap()
        .parent = Some(ep(1));
    assert!(matches!(
        s.transact(s.revision(), &[insert(cyclic)]),
        Err(PrefabError::Cycle)
    ));
    assert!(matches!(
        s.transact(0, &[]),
        Err(PrefabError::Revision { .. })
    ));
    assert_eq!(s.encode().unwrap(), before);
}

#[test]
fn command_stack_sparse_inverse_history_and_concurrent_conflicts() {
    let mut s = library();
    let mut stack = CommandStack::default();
    for n in [4, 5] {
        stack
            .execute(
                Box::new(PrefabCommand::new(
                    "generic field",
                    s.revision(),
                    vec![pin(io(1), n)],
                )),
                &mut s,
            )
            .unwrap();
    }
    assert_eq!(read(&s, io(1)), 5);
    batch(&mut s, &[pin(io(3), 90)]);
    for n in [4, 1] {
        assert!(stack.undo(&mut s).unwrap());
        assert_eq!(read(&s, io(1)), n);
        assert_eq!(read(&s, io(3)), 90);
    }
    for n in [4, 5] {
        assert!(stack.redo(&mut s).unwrap());
        assert_eq!(read(&s, io(1)), n);
    }
    batch(&mut s, &[pin(io(1), 99)]);
    let before = s.encode().unwrap();
    assert!(stack.undo(&mut s).is_err());
    assert_eq!(stack.undo_len(), 2);
    assert_eq!(s.encode().unwrap(), before);
    let count = stack.undo_len();
    assert!(stack
        .execute(
            Box::new(PrefabCommand::new(
                "bad",
                s.revision(),
                vec![pin(io(2), 8), PrefabEdit::RemoveAsset(aid(1))]
            )),
            &mut s
        )
        .is_err());
    assert_eq!(stack.undo_len(), count);
    assert_eq!(s.encode().unwrap(), before);
}

#[test]
fn inverse_validates_surviving_dependencies_and_is_store_scoped() {
    let mut s = library();
    let receipt = batch(&mut s, &[insert(asset(10, None))]);
    batch(&mut s, &[inst(10, 10)]);
    let before = s.encode().unwrap();
    assert!(receipt.undo(&mut s).is_err());
    assert_eq!(s.encode().unwrap(), before);
    let mut other = PrefabStore::decode(&before, PrefabLimits::default()).unwrap();
    assert!(receipt.undo(&mut other).is_err());
    batch(&mut s, &[PrefabEdit::RemoveInstance(iid(10))]);
    receipt.undo(&mut s).unwrap();
    receipt.redo(&mut s).unwrap();
}

#[test]
fn limits_are_explicit_and_failed_batches_preserve_state() {
    let configurations = [
        PrefabLimits {
            max_assets: 1,
            ..Default::default()
        },
        PrefabLimits {
            max_instances: 1,
            ..Default::default()
        },
        PrefabLimits {
            max_entities: 1,
            ..Default::default()
        },
        PrefabLimits {
            max_value_nodes: 1,
            ..Default::default()
        },
        PrefabLimits {
            max_value_bytes: 1,
            ..Default::default()
        },
        PrefabLimits {
            max_steps: 1,
            ..Default::default()
        },
        PrefabLimits {
            max_edits: 1,
            ..Default::default()
        },
    ];
    for limits in configurations {
        let mut s = PrefabStore::new(limits).unwrap();
        assert!(matches!(
            s.transact(
                0,
                &[
                    insert(asset(1, None)),
                    insert(asset(2, Some(1))),
                    inst(1, 2),
                    inst(2, 2)
                ]
            ),
            Err(PrefabError::Budget(_))
        ));
        assert_eq!(s.revision(), 0);
        assert!(s.asset(aid(1)).is_none());
    }
    let mut s = PrefabStore::new(PrefabLimits {
        max_edits: 3,
        ..Default::default()
    })
    .unwrap();
    batch(&mut s, &[insert(asset(1, None)), inst(1, 1), inst(2, 1)]);
    let before = s.encode().unwrap();
    assert!(s
        .transact(
            s.revision(),
            &[
                pin(owner(1), 2),
                PrefabEdit::Propagate {
                    source: aid(1),
                    property: property(),
                    recipients: vec![io(1), io(2)]
                }
            ]
        )
        .is_err());
    assert_eq!(s.encode().unwrap(), before);
    assert!(PrefabStore::new(PrefabLimits {
        max_depth: 129,
        ..Default::default()
    })
    .is_err());
}

#[test]
fn dependency_value_and_field_depth_are_bounded() {
    let mut s = PrefabStore::new(PrefabLimits {
        max_depth: 2,
        ..Default::default()
    })
    .unwrap();
    batch(&mut s, &[insert(asset(1, None))]);
    let before = s.encode().unwrap();
    assert!(s
        .transact(
            s.revision(),
            &[insert(asset(2, Some(1))), insert(asset(3, Some(2)))]
        )
        .is_err());
    let mut p = property();
    p.fields = vec!["a".into(); 3];
    assert!(s
        .transact(
            s.revision(),
            &[PrefabEdit::Override {
                owner: owner(1),
                property: p,
                value: val(4)
            }]
        )
        .is_err());
    let v = PrefabValue::Array(vec![PrefabValue::Array(vec![PrefabValue::Array(vec![
        PrefabValue::Null,
    ])])]);
    assert!(s
        .transact(
            s.revision(),
            &[PrefabEdit::Override {
                owner: owner(1),
                property: property(),
                value: PrefabOverride::Set(Arc::new(v))
            }]
        )
        .is_err());
    assert_eq!(s.encode().unwrap(), before);
}

#[test]
fn exact_value_mapping_and_deterministic_codec_roundtrip() {
    let mut s = library();
    let mut p = property();
    p.fields = vec![" \n字段./ ".into()];
    let value = PrefabValue::Array(vec![
        PrefabValue::Null,
        PrefabValue::Bool(true),
        PrefabValue::Signed(i64::MIN),
        PrefabValue::Unsigned(u64::MAX),
        PrefabValue::FloatBits(0x7ff8_0000_0000_0042),
        PrefabValue::FloatBits(1u64 << 63),
        PrefabValue::String(" \n路径/\\😀 ".into()),
        PrefabValue::Bytes(vec![0, 255, 13, 10]),
        PrefabValue::Object(BTreeMap::from([("".into(), PrefabValue::Bool(false))])),
    ]);
    batch(
        &mut s,
        &[PrefabEdit::Override {
            owner: io(1),
            property: p.clone(),
            value: PrefabOverride::Set(Arc::new(value.clone())),
        }],
    );
    let bytes = s.encode().unwrap();
    let restored = PrefabStore::decode(&bytes, PrefabLimits::default()).unwrap();
    assert_eq!(restored.encode().unwrap(), bytes);
    assert_eq!(restored.revision(), s.revision());
    assert_eq!(restored.compose(io(1)).unwrap().value(&p), Some(&value));
    let mut a = PrefabStore::new(PrefabLimits::default()).unwrap();
    let mut b = PrefabStore::new(PrefabLimits::default()).unwrap();
    batch(&mut a, &[insert(asset(1, None)), insert(asset(9, None))]);
    batch(&mut b, &[insert(asset(9, None)), insert(asset(1, None))]);
    assert_eq!(a.encode().unwrap(), b.encode().unwrap());
}

#[test]
fn malformed_codec_never_yields_a_partial_store() {
    let s = library();
    let bytes = s.encode().unwrap();
    for len in 0..bytes.len() {
        assert!(
            PrefabStore::decode(&bytes[..len], PrefabLimits::default()).is_err(),
            "prefix {len}"
        );
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(PrefabStore::decode(&trailing, PrefabLimits::default()).is_err());
    let mut magic = bytes.clone();
    magic[4] = 9;
    assert!(PrefabStore::decode(&magic, PrefabLimits::default()).is_err());
    let mut count = bytes.clone();
    count[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(PrefabStore::decode(&count, PrefabLimits::default()).is_err());
    let mut identity = bytes.clone();
    identity[40] ^= 1;
    assert!(PrefabStore::decode(&identity, PrefabLimits::default()).is_err());
    assert!(PrefabStore::decode(
        &bytes,
        PrefabLimits {
            max_codec_bytes: bytes.len() - 1,
            ..Default::default()
        }
    )
    .is_err());
    assert!(PrefabStore::decode(
        &bytes,
        PrefabLimits {
            max_assets: 1,
            ..Default::default()
        }
    )
    .is_err());
    // Revision exhaustion is checked before publication, including history.
    let mut exhausted = bytes;
    exhausted[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut s = PrefabStore::decode(&exhausted, PrefabLimits::default()).unwrap();
    assert!(s.transact(u64::MAX, &[pin(io(1), 88)]).is_err());
    assert_eq!(s.encode().unwrap(), exhausted);
}

#[test]
fn canonical_codec_rejects_duplicates_invalid_tags_utf8_and_graph_edges() {
    let mut s = PrefabStore::new(PrefabLimits::default()).unwrap();
    let empty = |n| PrefabAsset {
        id: aid(n),
        base: None,
        layer: PrefabLayer::default(),
    };
    batch(&mut s, &[insert(empty(1)), insert(empty(2))]);
    let bytes = s.encode().unwrap();
    // Format v1: header/revision/count=24; each empty asset is
    // key16 + id16 + base1 + five map/set counts(40) = 73 bytes.
    assert_eq!(bytes.len(), 24 + 73 * 2 + 8);
    let mut duplicate = bytes.clone();
    duplicate[97..113].copy_from_slice(&1u128.to_le_bytes());
    duplicate[113..129].copy_from_slice(&1u128.to_le_bytes());
    assert!(matches!(
        PrefabStore::decode(&duplicate, PrefabLimits::default()),
        Err(PrefabError::Invalid(_))
    ));
    let mut tag = bytes.clone();
    tag[56] = 2;
    assert!(PrefabStore::decode(&tag, PrefabLimits::default()).is_err());
    // Replace None base with Some(nonexistent), keeping all lengths canonical.
    let mut missing = bytes;
    missing[56] = 1;
    missing.splice(57..57, 999u128.to_le_bytes());
    assert!(matches!(
        PrefabStore::decode(&missing, PrefabLimits::default()),
        Err(PrefabError::Missing(_))
    ));

    let mut s = library();
    let mut p = property();
    p.fields = vec!["codec-value-unique".into()];
    batch(
        &mut s,
        &[PrefabEdit::Override {
            owner: io(1),
            property: p,
            value: PrefabOverride::Set(Arc::new(PrefabValue::String("utf8-payload-unique".into()))),
        }],
    );
    let bytes = s.encode().unwrap();
    let pos = bytes
        .windows(b"utf8-payload-unique".len())
        .position(|w| w == b"utf8-payload-unique")
        .unwrap();
    let mut invalid_utf8 = bytes.clone();
    invalid_utf8[pos] = 255;
    assert!(PrefabStore::decode(&invalid_utf8, PrefabLimits::default()).is_err());
    // String tag is before its u64 length. Unknown tags never coerce to Null.
    let mut invalid_tag = bytes;
    invalid_tag[pos - 9] = 255;
    assert!(PrefabStore::decode(&invalid_tag, PrefabLimits::default()).is_err());
}

#[test]
fn codec_writer_and_decoder_depth_budgets_are_enforced() {
    let mut s = PrefabStore::new(PrefabLimits {
        max_codec_bytes: 64,
        ..Default::default()
    })
    .unwrap();
    batch(&mut s, &[insert(asset(1, None))]);
    assert!(matches!(
        s.encode(),
        Err(PrefabError::Budget("codec bytes"))
    ));
    let mut s = library();
    let deep = PrefabValue::Array(vec![PrefabValue::Array(vec![PrefabValue::Array(vec![
        PrefabValue::Null,
    ])])]);
    batch(
        &mut s,
        &[PrefabEdit::Override {
            owner: io(1),
            property: property(),
            value: PrefabOverride::Set(Arc::new(deep)),
        }],
    );
    let bytes = s.encode().unwrap();
    assert!(matches!(
        PrefabStore::decode(
            &bytes,
            PrefabLimits {
                max_depth: 2,
                ..Default::default()
            }
        ),
        Err(PrefabError::Budget("depth"))
    ));
    assert!(matches!(
        PrefabStore::decode(
            &bytes,
            PrefabLimits {
                max_value_nodes: 1,
                ..Default::default()
            }
        ),
        Err(PrefabError::Budget("value nodes"))
    ));
}

#[test]
fn qualified_nested_override_applies_to_containing_base_without_retargeting_source() {
    let mut s = library();
    let mut a = asset(10, None);
    a.layer.slots.insert(
        PrefabSlotId::from_raw(5),
        Some(PrefabSlot {
            prefab: aid(2),
            parent: None,
        }),
    );
    batch(&mut s, &[insert(a), inst(10, 10), inst(11, 10)]);
    let mut p = property();
    p.entity.slots.push(PrefabSlotId::from_raw(5));
    batch(
        &mut s,
        &[PrefabEdit::Override {
            owner: io(10),
            property: p.clone(),
            value: val(70),
        }],
    );
    batch(
        &mut s,
        &[PrefabEdit::ApplyToBase {
            source: io(10),
            target: aid(10),
            property: p.clone(),
        }],
    );
    assert_eq!(
        s.compose(io(11)).unwrap().value(&p),
        Some(&PrefabValue::Signed(70))
    );
    assert_eq!(read(&s, owner(2)), 1);
    assert!(!s.layer(io(10)).unwrap().overrides.contains_key(&p));
    let before = s.encode().unwrap();
    // Moving across a nested namespace requires an explicit host remapping;
    // this API refuses to reinterpret [slot5]/entity1 as local entity1.
    assert!(s
        .transact(
            s.revision(),
            &[PrefabEdit::ApplyToBase {
                source: owner(10),
                target: aid(2),
                property: p
            }]
        )
        .is_err());
    assert_eq!(s.encode().unwrap(), before);
}

#[test]
fn noops_preserve_revision_record_identity_and_existing_history() {
    let mut s = library();
    let receipt = batch(&mut s, &[pin(io(1), 4)]);
    let record = s.instance(iid(1)).unwrap().clone();
    let revision = s.revision();
    let noop = batch(&mut s, &[pin(io(1), 4)]);
    assert_eq!(noop.changed_records(), 0);
    assert_eq!(s.revision(), revision);
    assert!(Arc::ptr_eq(&record, s.instance(iid(1)).unwrap()));
    assert_eq!(receipt.owners().collect::<Vec<_>>(), vec![io(1)]);
    receipt.undo(&mut s).unwrap();
    assert_eq!(read(&s, io(1)), 1);
    assert_eq!(s.assets().count(), 4);
    assert_eq!(s.instances().count(), 3);
    assert_eq!(s.limits(), PrefabLimits::default());
}

#[test]
fn absent_down_propagation_is_independent_of_recipient_order() {
    let mut s = library();
    batch(
        &mut s,
        &[
            PrefabEdit::Override {
                owner: owner(1),
                property: property(),
                value: PrefabOverride::Remove,
            },
            pin(owner(2), 20),
            pin(io(1), 30),
        ],
    );
    let before = s.encode().unwrap();
    let mut other = PrefabStore::decode(&before, PrefabLimits::default()).unwrap();
    for (store, recipients) in [
        (&mut s, vec![io(1), owner(2)]),
        (&mut other, vec![owner(2), io(1)]),
    ] {
        batch(
            store,
            &[PrefabEdit::Propagate {
                source: aid(1),
                property: property(),
                recipients,
            }],
        );
        assert!(store.compose(io(1)).unwrap().value(&property()).is_none());
        assert!(!store
            .layer(io(1))
            .unwrap()
            .overrides
            .contains_key(&property()));
    }
    assert_eq!(s.encode().unwrap(), other.encode().unwrap());
}
