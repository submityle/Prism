use prism_visual_scripting::prelude::*;

fn request(id: &str, condition: AwaitCondition) -> AwaitRequest {
    AwaitRequest {
        id: id.into(),
        condition,
        payload: Value::Unit,
    }
}

#[test]
fn broker_matches_typed_waits_in_deterministic_order() {
    let broker = InMemoryAwaitBroker::default();
    broker
        .register(request(
            "signal-b",
            AwaitCondition::Signal {
                topic: "orders".into(),
                correlation: "42".into(),
            },
        ))
        .unwrap();
    broker
        .register(request(
            "signal-a",
            AwaitCondition::Signal {
                topic: "orders".into(),
                correlation: "42".into(),
            },
        ))
        .unwrap();
    broker
        .register(request("timer", AwaitCondition::Timer { deadline_ms: 100 }))
        .unwrap();

    assert_eq!(broker.tick(99).unwrap(), 0);
    assert_eq!(broker.tick(100).unwrap(), 1);
    assert_eq!(
        broker
            .publish_signal("orders", "42", Value::I64(7))
            .unwrap(),
        2
    );
    let ready = broker.drain_ready().unwrap();
    assert_eq!(
        ready
            .iter()
            .map(|item| item.request.id.as_str())
            .collect::<Vec<_>>(),
        vec!["signal-a", "signal-b", "timer"]
    );
}

#[test]
fn broker_rejects_pending_ready_identity_overlap() {
    let broker = InMemoryAwaitBroker::default();
    broker
        .register(request("stable", AwaitCondition::Timer { deadline_ms: 9 }))
        .unwrap();
    let snapshot = AwaitBrokerState {
        pending: std::collections::BTreeMap::from([(
            "overlap".into(),
            request("overlap", AwaitCondition::Timer { deadline_ms: 10 }),
        )]),
        ready: std::collections::BTreeMap::from([(
            "overlap".into(),
            AwaitReady {
                request: request("overlap", AwaitCondition::Timer { deadline_ms: 10 }),
                resolution: SuspensionResolution {
                    outputs: vec![Value::Bool(true)],
                    next: ExecNext::End,
                },
            },
        )]),
    };
    assert!(matches!(
        broker.restore(snapshot),
        Err(VsError::Validate(message)) if message.contains("both pending and ready")
    ));
    let preserved = broker.snapshot().unwrap();
    assert!(preserved.pending.contains_key("stable"));
    assert!(preserved.ready.is_empty());
}
#[test]
fn broker_is_idempotent_and_checkpointable() {
    let broker = InMemoryAwaitBroker::default();
    let approval = request(
        "approval-1",
        AwaitCondition::Approval {
            subject: "deploy".into(),
        },
    );
    assert!(broker.register(approval.clone()).unwrap());
    assert!(!broker.register(approval.clone()).unwrap());
    let changed = AwaitRequest {
        payload: Value::I64(1),
        ..approval
    };
    assert!(broker
        .register(changed)
        .unwrap_err()
        .to_string()
        .contains("different content"));

    let snapshot = broker.snapshot().unwrap();
    let restored = InMemoryAwaitBroker::default();
    restored.restore(snapshot).unwrap();
    assert_eq!(restored.approve("deploy", Value::Bool(true)).unwrap(), 1);
    assert_eq!(restored.drain_ready().unwrap().len(), 1);
}
