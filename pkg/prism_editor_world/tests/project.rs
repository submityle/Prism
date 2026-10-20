//! Production membership codec, real world identities and sparse inverse history.
use prism_asset::StableGuid;
use prism_ecs::partition::{cell::CellCoord, data_layer::DataLayerId};
use prism_editor_world::{
    project::{
        ProjectEdit, ProjectError, ProjectHistory, ProjectLayer, ProjectLimits, WorldProject,
    },
    CellResidency, DataLayer, EditorWorldData, EditorWorldLimits,
};
use std::collections::BTreeSet;

const CHANGE_BYTES: usize = 8 * 1024 * 1024;
fn id(n: u128) -> StableGuid {
    StableGuid::from_u128(n)
}
fn layer_id(n: u32) -> DataLayerId {
    DataLayerId::new(n)
}
fn cell(n: i32) -> CellCoord {
    CellCoord::new(n, -7, i32::MIN)
}
fn limits() -> ProjectLimits {
    ProjectLimits::default()
}
fn layer(n: u32, name: &str, parent: Option<u32>) -> ProjectLayer {
    ProjectLayer {
        id: layer_id(n),
        name: name.into(),
        parent: parent.map(layer_id),
        visible: true,
        editable: true,
    }
}
fn put_layer(value: ProjectLayer) -> ProjectEdit {
    ProjectEdit::Layer {
        id: value.id,
        value: Some(value),
    }
}
fn put_cell(n: i32, memberships: &[u32]) -> ProjectEdit {
    ProjectEdit::Cell {
        cell: cell(n),
        layers: Some(memberships.iter().copied().map(layer_id).collect()),
    }
}
fn put_actor(actor: u128, owner: i32) -> ProjectEdit {
    ProjectEdit::Actor {
        actor: id(actor),
        cell: Some(cell(owner)),
    }
}
fn apply(
    project: &mut WorldProject,
    edits: &[ProjectEdit],
) -> prism_editor_world::project::ProjectChange {
    project
        .apply(project.revision(), edits, limits(), CHANGE_BYTES)
        .unwrap()
}
fn authored() -> WorldProject {
    let mut project = WorldProject::new(id(77)).unwrap();
    apply(
        &mut project,
        &[
            put_layer(layer(9, "父层", None)),
            put_layer(layer(0, "child", Some(9))),
            put_cell(-10, &[9, 0]),
            put_cell(20, &[]),
            put_actor(101, -10),
            put_actor(u128::MAX, 20),
        ],
    );
    project
}
fn rehash(bytes: &mut [u8]) {
    let offset = bytes.len() - 8;
    let hash = bytes[..offset]
        .iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3)
        });
    bytes[offset..].copy_from_slice(&hash.to_le_bytes());
}
fn unchanged_error(project: &mut WorldProject, edits: &[ProjectEdit]) -> ProjectError {
    let before = project.encode(limits()).unwrap();
    let token = project.token();
    let error = project
        .apply(project.revision(), edits, limits(), CHANGE_BYTES)
        .unwrap_err();
    assert_eq!(before, project.encode(limits()).unwrap());
    assert!(project.matches_token(&token));
    error
}

#[test]
fn real_editor_world_capture_roundtrip_materializes_complete_clean_actor_membership() {
    let world_limits = EditorWorldLimits::new(4, 1024, 8, 1, 1).unwrap();
    let mut world = EditorWorldData::new(id(77), world_limits).unwrap();
    world
        .add_layer(DataLayer::new(layer_id(9), "父层").unwrap())
        .unwrap();
    world
        .add_layer(DataLayer::with_parent(layer_id(0), "child", layer_id(9)).unwrap())
        .unwrap();
    world.set_layer_visible(layer_id(9), false).unwrap();
    world.set_layer_editable(layer_id(0), false).unwrap();
    world.add_cell(cell(20), []).unwrap();
    world
        .add_cell(cell(-10), [layer_id(9), layer_id(0)])
        .unwrap();
    world.register_actor(id(u128::MAX), cell(20)).unwrap();
    world.register_actor(id(101), cell(-10)).unwrap();
    assert!(
        world.plan_ofpa().operations().is_empty(),
        "clean actors must still be captured"
    );
    let project = WorldProject::capture(&world, limits()).unwrap();
    assert_eq!(project.actors().count(), 2);
    let bytes = project.encode(limits()).unwrap();
    let decoded = WorldProject::decode(&bytes, limits()).unwrap();
    assert_eq!(project, decoded);
    assert_eq!(decoded.encode(limits()).unwrap(), bytes);
    assert!(
        !decoded.matches_token(&project.token()),
        "adopted replacement has its own fence"
    );
    let restored = decoded.materialize(world_limits, limits()).unwrap();
    assert_eq!(WorldProject::capture(&restored, limits()).unwrap(), project);
    for old in world.model().cells() {
        let new = restored.model().cell(old.coord()).unwrap();
        assert_eq!(old.file(), new.file());
        assert_eq!(new.residency(), CellResidency::Unloaded);
        assert_eq!(
            world.model().is_cell_visible(old.coord()),
            restored.model().is_cell_visible(old.coord())
        );
        assert_eq!(
            old.layers().collect::<Vec<_>>(),
            new.layers().collect::<Vec<_>>()
        );
    }
    for (actor, owner) in project.actors() {
        assert_eq!(
            restored.actor_record(actor).unwrap().identity(),
            world.actor_record(actor).unwrap().identity()
        );
        assert_eq!(
            restored.actor_record(actor).unwrap().identity().cell(),
            owner
        );
        assert!(restored.actor_payload(actor).is_none());
    }
}

#[test]
fn canonical_bytes_ignore_command_order_and_cover_empty_extreme_coordinates() {
    let empty = WorldProject::new(id(7)).unwrap();
    assert_eq!(empty.encode(limits()).unwrap().len(), 64);
    assert_eq!(
        empty,
        WorldProject::decode(&empty.encode(limits()).unwrap(), limits()).unwrap()
    );
    let edits = [
        put_cell(i32::MAX, &[3]),
        put_actor(u128::MAX, i32::MAX),
        put_layer(layer(3, "layer", None)),
        put_cell(i32::MIN, &[]),
    ];
    let mut a = WorldProject::new(id(7)).unwrap();
    let mut b = WorldProject::new(id(7)).unwrap();
    apply(&mut a, &edits);
    apply(&mut b, &edits.into_iter().rev().collect::<Vec<_>>());
    assert_eq!(a.encode(limits()).unwrap(), b.encode(limits()).unwrap());
    assert_eq!(
        a.cells().map(|(c, _)| c.x).collect::<Vec<_>>(),
        [i32::MIN, i32::MAX]
    );
}

#[test]
fn malformed_wire_never_accepts_truncation_corruption_order_flags_or_duplicates() {
    let bytes = authored().encode(limits()).unwrap();
    for n in 0..bytes.len() {
        assert!(
            WorldProject::decode(&bytes[..n], limits()).is_err(),
            "prefix {n}"
        );
    }
    for offset in 0..bytes.len() {
        let mut bad = bytes.clone();
        bad[offset] ^= 0x40;
        assert!(
            WorldProject::decode(&bad, limits()).is_err(),
            "corrupt byte {offset}"
        );
    }
    for (offset, replacement) in [(8, 2u32), (44, u32::MAX), (48, u32::MAX), (52, u32::MAX)] {
        let mut bad = bytes.clone();
        bad[offset..offset + 4].copy_from_slice(&replacement.to_le_bytes());
        rehash(&mut bad);
        assert!(WorldProject::decode(&bad, limits()).is_err());
    }
    // First layer is id=0, name=child, parent=9; next layer id must be >0.
    for offset in [56 + 4 + 4 + 5, 56 + 4 + 4 + 5 + 1 + 4] {
        let mut bad = bytes.clone();
        bad[offset] = 7;
        rehash(&mut bad);
        assert!(WorldProject::decode(&bad, limits()).is_err());
    }
    let second = 56 + 10 + 5 + 4;
    let mut bad = bytes.clone();
    bad[second..second + 4].copy_from_slice(&0u32.to_le_bytes());
    rehash(&mut bad);
    assert!(matches!(
        WorldProject::decode(&bad, limits()),
        Err(ProjectError::Malformed("layer order"))
    ));
    let mut bad = bytes;
    bad.extend([0; 1]);
    let len = bad.len() as u64;
    bad[12..20].copy_from_slice(&len.to_le_bytes());
    rehash(&mut bad);
    assert!(WorldProject::decode(&bad, limits()).is_err());
}

#[test]
fn codec_enforces_each_budget_and_rejects_dangling_and_nil_wire_identities() {
    let project = authored();
    let bytes = project.encode(limits()).unwrap();
    for constrained in [
        ProjectLimits {
            max_bytes: bytes.len() - 1,
            ..limits()
        },
        ProjectLimits {
            max_layers: 1,
            ..limits()
        },
        ProjectLimits {
            max_cells: 1,
            ..limits()
        },
        ProjectLimits {
            max_actors: 1,
            ..limits()
        },
        ProjectLimits {
            max_memberships: 1,
            ..limits()
        },
        ProjectLimits {
            max_name_bytes: 3,
            ..limits()
        },
    ] {
        assert!(matches!(
            project.encode(constrained),
            Err(ProjectError::Budget(_))
        ));
        assert!(matches!(
            WorldProject::decode(&bytes, constrained),
            Err(ProjectError::Budget(_))
        ));
    }
    assert_eq!(
        WorldProject::decode(
            &bytes,
            ProjectLimits {
                max_bytes: usize::MAX,
                ..limits()
            }
        ),
        Err(ProjectError::InvalidLimits)
    );
    let mut bad = bytes.clone();
    bad[20..36].fill(0);
    rehash(&mut bad);
    assert!(matches!(
        WorldProject::decode(&bad, limits()),
        Err(ProjectError::Invalid(_))
    ));
    // Rewrite first parent to an unknown layer while keeping a valid checksum.
    let mut bad = bytes;
    bad[70..74].copy_from_slice(&99u32.to_le_bytes());
    rehash(&mut bad);
    assert!(matches!(
        WorldProject::decode(&bad, limits()),
        Err(ProjectError::Invalid(_))
    ));
}

#[test]
fn atomic_authoring_validates_final_relationships_and_inverse_repairs_every_record() {
    let mut project = authored();
    let before = project.clone();
    let inverse = apply(
        &mut project,
        &[
            ProjectEdit::Layer {
                id: layer_id(9),
                value: None,
            },
            put_layer(layer(0, "renamed", None)),
            put_cell(30, &[0]),
            put_actor(101, 30),
            ProjectEdit::Cell {
                cell: cell(-10),
                layers: None,
            },
        ],
    );
    assert_eq!(inverse.record_count(), 5);
    assert_eq!(project.actor_cell(id(101)), Some(cell(30)));
    assert!(project.cell_layers(cell(-10)).is_none());
    inverse.undo(&mut project, limits()).unwrap();
    let revision = project.revision();
    assert_eq!(
        project.layers().collect::<Vec<_>>(),
        before.layers().collect::<Vec<_>>()
    );
    assert_eq!(
        project.cells().collect::<Vec<_>>(),
        before.cells().collect::<Vec<_>>()
    );
    assert_eq!(
        project.actors().collect::<Vec<_>>(),
        before.actors().collect::<Vec<_>>()
    );
    inverse.redo(&mut project, limits()).unwrap();
    assert_eq!(project.revision(), revision + 1);
}

#[test]
fn illegal_names_hierarchy_references_duplicates_and_budgets_are_atomic() {
    let mut project = authored();
    for edits in [
        vec![put_layer(layer(5, "child", None))],
        vec![put_layer(layer(5, "bad/name", None))],
        vec![put_layer(layer(9, "parent", Some(0)))],
        vec![put_layer(layer(9, "parent", Some(99)))],
        vec![put_cell(50, &[99])],
        vec![put_actor(101, 999)],
        vec![put_actor(0, 20)],
        vec![ProjectEdit::Layer {
            id: layer_id(9),
            value: None,
        }],
        vec![ProjectEdit::Cell {
            cell: cell(-10),
            layers: None,
        }],
        vec![put_actor(101, 20), put_actor(101, 20)],
        vec![ProjectEdit::Layer {
            id: layer_id(1),
            value: Some(layer(2, "bad id", None)),
        }],
    ] {
        unchanged_error(&mut project, &edits);
    }
    assert_eq!(
        unchanged_error(&mut project, &[put_actor(101, -10)]),
        ProjectError::NoChange
    );
    let before = project.clone();
    assert!(matches!(
        project.apply(0, &[put_actor(101, 20)], limits(), CHANGE_BYTES),
        Err(ProjectError::Conflict)
    ));
    assert!(matches!(
        project.apply(project.revision(), &[put_actor(101, 20)], limits(), 1),
        Err(ProjectError::Budget(_))
    ));
    assert!(project
        .apply(
            project.revision(),
            &[put_actor(101, 20), put_cell(40, &[])],
            ProjectLimits {
                max_edits: 1,
                ..limits()
            },
            CHANGE_BYTES
        )
        .is_err());
    assert!(project
        .apply(
            project.revision(),
            &[put_cell(40, &[])],
            ProjectLimits {
                max_cells: 2,
                ..limits()
            },
            CHANGE_BYTES
        )
        .is_err());
    assert_eq!(project, before);
}

#[test]
fn sparse_inverse_preserves_unrelated_edits_and_rejects_touched_conflicts() {
    let mut project = authored();
    let inverse = apply(&mut project, &[put_actor(101, 20)]);
    apply(
        &mut project,
        &[put_layer(layer(0, "unrelated rename", Some(9)))],
    );
    inverse.undo(&mut project, limits()).unwrap();
    assert_eq!(project.layer(layer_id(0)).unwrap().name, "unrelated rename");
    inverse.redo(&mut project, limits()).unwrap();
    apply(&mut project, &[put_cell(30, &[]), put_actor(101, 30)]);
    let before = project.clone();
    let token = project.token();
    assert_eq!(
        inverse.undo(&mut project, limits()),
        Err(ProjectError::Conflict)
    );
    assert_eq!(project, before);
    assert!(project.matches_token(&token));
    let mut foreign = WorldProject::new(id(999)).unwrap();
    assert_eq!(
        inverse.undo(&mut foreign, limits()),
        Err(ProjectError::Conflict)
    );
}

#[test]
fn inverse_dangling_reference_failure_rolls_back_and_history_fences_follow_revisions() {
    let mut project = authored();
    let inverse = apply(&mut project, &[put_cell(30, &[])]);
    apply(&mut project, &[put_actor(123, 30)]);
    let before = project.clone();
    let token = project.token();
    assert!(matches!(
        inverse.undo(&mut project, limits()),
        Err(ProjectError::Invalid(_))
    ));
    assert_eq!(project, before);
    assert!(project.matches_token(&token));
    let old_token = project.token();
    let mut history = ProjectHistory::new(2, 2048).unwrap();
    history
        .execute(
            &mut project,
            before.revision(),
            &[put_actor(101, 20)],
            limits(),
        )
        .unwrap();
    history.undo(&mut project, limits()).unwrap();
    assert!(
        !project.matches_token(&old_token),
        "ABA identical content must not accept old save receipts"
    );
    let undone = project.clone();
    assert!(history
        .execute(
            &mut project,
            undone.revision(),
            &[put_actor(101, 999)],
            limits()
        )
        .is_err());
    assert_eq!((history.undo_len(), history.redo_len()), (0, 1));
    assert_eq!(project, undone);
    history.redo(&mut project, limits()).unwrap();
    let revision = project.revision();
    history
        .execute(&mut project, revision, &[put_actor(101, 30)], limits())
        .unwrap();
    let bytes = project.encode(limits()).unwrap();
    let revision = project.revision();
    assert!(matches!(
        history.execute(&mut project, revision, &[put_actor(101, -10)], limits()),
        Err(ProjectError::Budget(_))
    ));
    assert_eq!(project.encode(limits()).unwrap(), bytes);
}

#[test]
fn inverse_charge_is_independent_of_unrelated_world_size_and_history_bytes_reject_before_edit() {
    let mut small = authored();
    let mut large = small.clone();
    let cells: Vec<_> = (1000..2000).map(|n| put_cell(n, &[])).collect();
    apply(&mut large, &cells);
    let a = apply(&mut small, &[put_actor(101, 20)]);
    let b = apply(&mut large, &[put_actor(101, 20)]);
    assert_eq!(a.record_count(), 1);
    assert_eq!(b.record_count(), 1);
    assert_eq!(a.charged_bytes(), b.charged_bytes());
    assert!(b.charged_bytes() < 1024);
    let mut history = ProjectHistory::new(10, b.charged_bytes() - 1).unwrap();
    let before = large.clone();
    let revision = large.revision();
    assert!(matches!(
        history.execute(&mut large, revision, &[put_actor(101, -10)], limits()),
        Err(ProjectError::Budget(_))
    ));
    assert_eq!(large, before);
    assert_eq!(history.charged_bytes(), 0);
    assert_eq!(
        history.undo(&mut large, limits()),
        Err(ProjectError::EmptyHistory)
    );
}

#[test]
fn deep_hierarchy_uses_bounded_nonrecursive_validation_and_world_limits_are_real() {
    let mut project = WorldProject::new(id(1)).unwrap();
    let edits: Vec<_> = (0..1024)
        .map(|n| {
            put_layer(layer(
                n,
                &format!("layer {n}"),
                (n != 1023).then_some(n + 1),
            ))
        })
        .collect();
    apply(&mut project, &edits);
    assert_eq!(
        WorldProject::decode(&project.encode(limits()).unwrap(), limits()).unwrap(),
        project
    );
    assert!(unchanged_error(
        &mut project,
        &[put_layer(layer(1023, "layer 1023", Some(0)))]
    )
    .to_string()
    .contains("cycle"));
    let mut project = authored();
    let mut bytes = project.encode(limits()).unwrap();
    bytes[36..44].copy_from_slice(&u64::MAX.to_le_bytes());
    rehash(&mut bytes);
    project = WorldProject::decode(&bytes, limits()).unwrap();
    assert_eq!(
        unchanged_error(&mut project, &[put_actor(101, 20)]),
        ProjectError::RevisionExhausted
    );
    assert!(matches!(
        project.materialize(EditorWorldLimits::new(1, 16, 1, 1, 1).unwrap(), limits()),
        Err(ProjectError::Budget(_))
    ));
    assert_eq!(project.cell_layers(cell(20)), Some(&BTreeSet::new()));
}

#[test]
fn valid_checksum_cannot_hide_duplicate_memberships_cells_actors_or_missing_owner() {
    let bytes = authored().encode(limits()).unwrap();
    // Header (56), child (19), UTF-8 parent layer (16), two cells (24+16).
    let cells = 91;
    let actors = cells + 24 + 16;
    for kind in 0..5 {
        let mut bad = bytes.clone();
        match kind {
            0 => bad[cells + 20..cells + 24].copy_from_slice(&0u32.to_le_bytes()),
            1 => {
                let first = bad[cells..cells + 12].to_vec();
                bad[cells + 24..cells + 36].copy_from_slice(&first);
            }
            2 => bad[actors + 28..actors + 44].copy_from_slice(&101u128.to_le_bytes()),
            3 => bad[actors + 16..actors + 20].copy_from_slice(&999i32.to_le_bytes()),
            4 => bad[actors..actors + 16].fill(0),
            _ => unreachable!(),
        }
        rehash(&mut bad);
        assert!(
            WorldProject::decode(&bad, limits()).is_err(),
            "malformed relation {kind}"
        );
    }
}

#[test]
fn exact_history_budget_failure_and_inverse_budget_failure_preserve_every_fence() {
    let mut p = authored();
    let change = apply(&mut p, &[put_actor(101, 20)]);
    change.undo(&mut p, limits()).unwrap();
    let mut history = ProjectHistory::new(4, change.charged_bytes()).unwrap();
    let revision = p.revision();
    history
        .execute(&mut p, revision, &[put_actor(101, 20)], limits())
        .unwrap();
    let before = p.clone();
    let token = p.token();
    let revision = p.revision();
    assert_eq!(
        history.execute(&mut p, revision, &[put_actor(101, -10)], limits()),
        Err(ProjectError::Budget("history bytes"))
    );
    assert_eq!(p, before);
    assert!(p.matches_token(&token));
    assert_eq!(history.undo_len(), 1);
    // A policy change may make an otherwise valid inverse exceed current caps.
    assert!(matches!(
        history.undo(
            &mut p,
            ProjectLimits {
                max_cells: 1,
                ..limits()
            }
        ),
        Err(ProjectError::Budget(_))
    ));
    assert_eq!(p, before);
    assert!(p.matches_token(&token));
    assert_eq!((history.undo_len(), history.redo_len()), (1, 0));
}

#[test]
fn real_world_capture_checks_each_catalog_budget_including_wire_before_copying() {
    let source = authored()
        .materialize(EditorWorldLimits::editor_default(), limits())
        .unwrap();
    for policy in [
        ProjectLimits {
            max_bytes: 64,
            ..limits()
        },
        ProjectLimits {
            max_layers: 1,
            ..limits()
        },
        ProjectLimits {
            max_cells: 1,
            ..limits()
        },
        ProjectLimits {
            max_actors: 1,
            ..limits()
        },
        ProjectLimits {
            max_memberships: 1,
            ..limits()
        },
        ProjectLimits {
            max_name_bytes: 1,
            ..limits()
        },
    ] {
        assert!(matches!(
            WorldProject::capture(&source, policy),
            Err(ProjectError::Budget(_))
        ));
    }
    for policy in [
        ProjectLimits {
            max_bytes: 63,
            ..limits()
        },
        ProjectLimits {
            max_layers: 0,
            ..limits()
        },
        ProjectLimits {
            max_cells: 0,
            ..limits()
        },
        ProjectLimits {
            max_actors: 0,
            ..limits()
        },
        ProjectLimits {
            max_memberships: 0,
            ..limits()
        },
        ProjectLimits {
            max_name_bytes: 0,
            ..limits()
        },
        ProjectLimits {
            max_edits: 0,
            ..limits()
        },
    ] {
        assert_eq!(
            WorldProject::capture(&source, policy),
            Err(ProjectError::InvalidLimits)
        );
    }
}
