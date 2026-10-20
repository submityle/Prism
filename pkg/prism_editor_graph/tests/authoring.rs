//! Public authoring protocol, bounded inverse history and real typed evaluation.
use prism_editor_graph::*;
extern crate alloc;
use alloc::collections::BTreeMap;

fn node(n: u64, first_port: u64, ty: &str) -> GraphEdit {
    GraphEdit::CreateNode(GraphNodeDraft {
        id: NodeId::from_raw(n),
        name: format!("node {n}"),
        payload: vec![n as u8],
        position: GraphPoint {
            x: n as i32 * 200,
            y: 0,
        },
        ports: vec![
            GraphPortDraft {
                id: PortId::from_raw(first_port),
                spec: PortSpec::input("in", PortType::named(ty)),
            },
            GraphPortDraft {
                id: PortId::from_raw(first_port + 1),
                spec: PortSpec::output("out", PortType::named(ty)),
            },
        ],
    })
}
fn edge(e: u64, o: u64, i: u64) -> GraphEdit {
    GraphEdit::Connect {
        edge: EdgeId::from_raw(e),
        output: PortId::from_raw(o),
        input: PortId::from_raw(i),
    }
}
fn editor() -> GraphEditor {
    GraphEditor::new(GraphAuthoringLimits::default()).unwrap()
}
fn run(e: &mut GraphEditor, edits: Vec<GraphEdit>) {
    e.execute(&GraphCommand::Edit(edits)).unwrap();
}

#[test]
fn failure_rolls_back_all_records_highwater_revision_and_history() {
    let mut e = editor();
    run(&mut e, vec![node(1, 1, "number")]);
    e.execute(&GraphCommand::Undo).unwrap();
    let before = (
        e.next_ids(),
        e.revision(),
        e.undo_depth(),
        e.redo_depth(),
        e.history_bytes(),
    );
    assert!(e
        .execute(&GraphCommand::Edit(vec![
            node(2, 3, "number"),
            GraphEdit::MoveNode {
                node: NodeId::from_raw(2),
                position: GraphPoint { x: 42, y: 21 }
            },
            GraphEdit::SetPayload {
                node: NodeId::from_raw(2),
                payload: vec![7; 20]
            },
            GraphEdit::DeleteNode(NodeId::from_raw(999))
        ]))
        .is_err());
    assert_eq!(
        before,
        (
            e.next_ids(),
            e.revision(),
            e.undo_depth(),
            e.redo_depth(),
            e.history_bytes()
        )
    );
    assert_eq!(e.graph().node_count(), 0);
    e.execute(&GraphCommand::Redo).unwrap();
    assert_eq!(
        e.graph().node(NodeId::from_raw(1)).unwrap().payload(),
        &vec![1]
    );
    e.graph().validate().unwrap();
}
#[test]
fn deletion_undo_restores_incident_edges_ports_ids_and_evaluation() {
    let mut e = editor();
    run(
        &mut e,
        vec![
            node(1, 1, "n"),
            node(2, 3, "n"),
            node(3, 5, "n"),
            edge(1, 2, 3),
            edge(2, 4, 5),
        ],
    );
    let ids = e.next_ids();
    run(&mut e, vec![GraphEdit::DeleteNode(NodeId::from_raw(2))]);
    assert_eq!(e.graph().edge_count(), 0);
    let charge = e.history_bytes();
    for _ in 0..3 {
        e.execute(&GraphCommand::Undo).unwrap();
        e.graph().validate().unwrap();
        assert_eq!(e.graph().edge_count(), 2);
        assert_eq!(
            e.graph().port_named(NodeId::from_raw(2), "in"),
            Some(PortId::from_raw(3))
        );
        let mut values = BTreeMap::<NodeId, u32>::new();
        let order = e
            .graph()
            .evaluate(|n| {
                let input = n
                    .input_edges()
                    .iter()
                    .map(|edge| values[&e.graph().port(edge.output()).unwrap().node()])
                    .sum::<u32>();
                values.insert(n.node().id(), input + u32::from(n.node().payload()[0]));
                Ok::<(), ()>(())
            })
            .unwrap();
        assert_eq!(
            order,
            vec![
                NodeId::from_raw(1),
                NodeId::from_raw(2),
                NodeId::from_raw(3)
            ]
        );
        assert_eq!(values[&NodeId::from_raw(3)], 6);
        e.execute(&GraphCommand::Redo).unwrap();
        assert_eq!(e.next_ids(), ids);
        assert_eq!(charge, e.history_bytes());
    }
}
#[test]
fn identities_never_reuse_after_undo_or_branch() {
    let mut e = editor();
    run(&mut e, vec![node(10, 100, "n")]);
    e.execute(&GraphCommand::Undo).unwrap();
    assert!(e
        .execute(&GraphCommand::Edit(vec![node(10, 100, "n")]))
        .is_err());
    run(&mut e, vec![node(11, 102, "n")]);
    assert_eq!(e.redo_depth(), 0);
    assert_eq!(e.next_ids().0.raw(), 12);
    for raw in [0, u64::MAX] {
        assert!(e
            .execute(&GraphCommand::Edit(vec![node(raw, 104, "n")]))
            .is_err());
    }
}
#[test]
fn types_cardinality_direction_and_cycle_are_atomic() {
    let mut e = editor();
    run(
        &mut e,
        vec![node(1, 1, "n"), node(2, 3, "n"), node(3, 5, "text")],
    );
    for edits in [
        vec![edge(1, 2, 5)],
        vec![edge(1, 1, 3)],
        vec![edge(1, 2, 3), edge(2, 2, 3)],
        vec![edge(1, 2, 3), edge(2, 4, 1)],
    ] {
        let before = (e.revision(), e.next_ids(), e.history_bytes());
        assert!(e.execute(&GraphCommand::Edit(edits)).is_err());
        assert_eq!(before, (e.revision(), e.next_ids(), e.history_bytes()));
        assert_eq!(e.graph().edge_count(), 0);
        e.graph().validate().unwrap();
    }
}
#[test]
fn bounded_history_evicts_oldest_and_rejects_unadmittable_inverse() {
    let mut e = GraphEditor::new(GraphAuthoringLimits {
        history_entries: 2,
        ..Default::default()
    })
    .unwrap();
    run(&mut e, vec![node(1, 1, "n")]);
    for x in 1..4 {
        run(
            &mut e,
            vec![GraphEdit::MoveNode {
                node: NodeId::from_raw(1),
                position: GraphPoint { x, y: 0 },
            }],
        );
    }
    assert_eq!(e.undo_depth(), 2);
    e.execute(&GraphCommand::Undo).unwrap();
    e.execute(&GraphCommand::Undo).unwrap();
    assert_eq!(e.position(NodeId::from_raw(1)).unwrap().x, 1);
    assert_eq!(
        e.execute(&GraphCommand::Undo),
        Err(GraphAuthoringError::EmptyHistory)
    );
    let mut e = GraphEditor::new(GraphAuthoringLimits {
        history_bytes: 100,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(
        e.execute(&GraphCommand::Edit(vec![node(1, 1, "n")])),
        Err(GraphAuthoringError::Limit)
    );
    assert_eq!(e.graph().node_count(), 0);
    assert_eq!(e.next_ids().0.raw(), 1);
}
#[test]
fn codec_canonical_replay_including_history_operations() {
    let limits = GraphAuthoringLimits::default();
    let mut a = editor();
    let mut b = editor();
    let commands = [
        GraphCommand::Edit(vec![node(1, 1, "n"), node(2, 3, "n"), edge(1, 2, 3)]),
        GraphCommand::Edit(vec![
            GraphEdit::SetPayload {
                node: NodeId::from_raw(2),
                payload: vec![9, 8, 7],
            },
            GraphEdit::Disconnect(EdgeId::from_raw(1)),
        ]),
        GraphCommand::Undo,
        GraphCommand::Redo,
        GraphCommand::Undo,
    ];
    for c in commands {
        let bytes = c.encode(limits).unwrap();
        assert_eq!(bytes, c.encode(limits).unwrap());
        let decoded = GraphCommand::decode(&bytes, limits).unwrap();
        assert_eq!(c, decoded);
        a.execute(&c).unwrap();
        b.execute(&decoded).unwrap();
    }
    assert_eq!(a.next_ids(), b.next_ids());
    assert_eq!(a.revision(), b.revision());
    assert_eq!(a.history_bytes(), b.history_bytes());
    for n in a.graph().node_ids() {
        assert_eq!(
            a.graph().node(n).unwrap().payload(),
            b.graph().node(n).unwrap().payload()
        );
        assert_eq!(a.position(n), b.position(n));
    }
    assert_eq!(
        a.graph().edges().collect::<Vec<_>>(),
        b.graph().edges().collect::<Vec<_>>()
    );
}
#[test]
fn codec_rejects_malformed_and_resource_excess_before_execution() {
    let l = GraphAuthoringLimits::default();
    let bytes = GraphCommand::Edit(vec![node(1, 1, "n")]).encode(l).unwrap();
    assert_eq!(&bytes[..7], b"PGRF\x01\x00\x00");
    for end in 0..bytes.len() {
        assert!(GraphCommand::decode(&bytes[..end], l).is_err());
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(GraphCommand::decode(&trailing, l).is_err());
    for (offset, value) in [(4, 2), (6, 255), (11, 255), (24, 255)] {
        let mut bad = bytes.clone();
        bad[offset] = value;
        assert!(GraphCommand::decode(&bad, l).is_err(), "offset {offset}");
    }
    let mut count = bytes.clone();
    count[7..11].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(GraphCommand::decode(&count, l).is_err());
    assert!(GraphCommand::decode(
        &bytes,
        GraphAuthoringLimits {
            command_bytes: 10,
            ..l
        }
    )
    .is_err());
    assert!(GraphCommand::Edit(vec![]).encode(l).is_err());
    let GraphEdit::CreateNode(mut draft) = node(1, 1, "n") else {
        unreachable!()
    };
    draft.position.x = i32::MIN;
    assert!(GraphCommand::Edit(vec![GraphEdit::CreateNode(draft)])
        .encode(l)
        .is_err());
}
#[test]
fn live_bounds_and_duplicate_port_names_rollback() {
    let mut e = GraphEditor::new(GraphAuthoringLimits {
        nodes: 1,
        ..Default::default()
    })
    .unwrap();
    assert!(e
        .execute(&GraphCommand::Edit(vec![node(1, 1, "n"), node(2, 3, "n")]))
        .is_err());
    assert_eq!(e.graph().node_count(), 0);
    let GraphEdit::CreateNode(mut d) = node(1, 1, "n") else {
        unreachable!()
    };
    d.ports[1].spec = d.ports[0].spec.clone();
    assert!(e
        .execute(&GraphCommand::Edit(vec![GraphEdit::CreateNode(d)]))
        .is_err());
    assert_eq!(e.next_ids().1.raw(), 1);
}

#[test]
fn deterministic_long_replay_preserves_atomic_errors_and_inverse_indexes() {
    let limits = GraphAuthoringLimits {
        history_entries: 8,
        history_bytes: 32768,
        ..Default::default()
    };
    let mut a = GraphEditor::new(limits).unwrap();
    let mut b = GraphEditor::new(limits).unwrap();
    let mut seed = 0x71848_u64;
    for step in 0..300 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let ids = a.graph().node_ids();
        let n = ids
            .get((seed as usize) % ids.len().max(1))
            .copied()
            .unwrap_or(NodeId::from_raw(999));
        let cmd = match seed % 8 {
            0 => GraphCommand::Undo,
            1 => GraphCommand::Redo,
            2 => GraphCommand::Edit(vec![GraphEdit::DeleteNode(n)]),
            3 => GraphCommand::Edit(vec![GraphEdit::SetPayload {
                node: n,
                payload: vec![(seed >> 8) as u8; (seed as usize) % 256],
            }]),
            4 => GraphCommand::Edit(vec![GraphEdit::MoveNode {
                node: n,
                position: GraphPoint { x: step, y: -step },
            }]),
            5 if ids.len() > 1 => {
                let other = ids[(step as usize) % ids.len()];
                GraphCommand::Edit(vec![GraphEdit::Connect {
                    edge: a.next_ids().2,
                    output: a.graph().node(n).unwrap().ports()[1],
                    input: a.graph().node(other).unwrap().ports()[0],
                }])
            }
            _ => GraphCommand::Edit(vec![node(a.next_ids().0.raw(), a.next_ids().1.raw(), "n")]),
        };
        let before = format!("{a:?}");
        let result = a.execute(&cmd);
        if result.is_err() {
            assert_eq!(before, format!("{a:?}"), "atomic error step {step}");
        }
        let decoded = GraphCommand::decode(&cmd.encode(limits).unwrap(), limits).unwrap();
        assert_eq!(result, b.execute(&decoded));
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        assert!(a.history_bytes() <= limits.history_bytes);
        assert!(a.undo_depth() + a.redo_depth() <= limits.history_entries);
        a.graph().validate().unwrap();
    }
}
