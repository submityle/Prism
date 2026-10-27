use prism_visual_scripting::prelude::*;

fn event(sequence: u64, node_id: u32) -> ExecutionEvent {
    ExecutionEvent {
        sequence,
        execution_id: "run-replay".into(),
        task_id: 1,
        frame_depth: 0,
        node_id,
        timestamp_ms: sequence,
        event: ExecutionEventKind::NodeStarted {
            def_id: "test.node".into(),
        },
    }
}

#[test]
fn replay_verifier_rejects_tampering_and_reports_divergence() {
    let expected = vec![event(1, 1), event(2, 2)];
    let report = ExecutionReplay::validate(&expected, "run-replay").unwrap();
    assert_eq!(report.event_count, 2);
    assert_ne!(report.digest, [0; 32]);

    let mut actual = expected.clone();
    actual[1].node_id = 9;
    let divergence = ExecutionReplay::compare(&expected, &actual)
        .unwrap()
        .unwrap();
    assert_eq!(divergence.sequence, 2);

    let mut invalid = expected.clone();
    invalid[0].sequence = 7;
    assert!(ExecutionReplay::validate(&invalid, "run-replay").is_err());
    invalid[0].sequence = 1;
    invalid[0].execution_id = "other-run".into();
    assert!(ExecutionReplay::validate(&invalid, "run-replay").is_err());

    let mut different_execution = expected.clone();
    different_execution[0].execution_id = "other-run".into();
    assert!(ExecutionReplay::compare(&expected, &different_execution).is_err());
}
