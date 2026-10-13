use alloc::vec;

use super::{
    BindingPath, BindingPathError, BindingSlot, DependencyError, DependencyPlan,
    EvaluatorDependency,
};

#[test]
fn binding_paths_are_canonical_and_strict() {
    assert_eq!(
        BindingPath::parse("combat.target.health").unwrap().as_str(),
        "combat.target.health"
    );
    assert_eq!(
        BindingPath::parse("combat..health"),
        Err(BindingPathError::InvalidSegment { index: 1 })
    );
    assert_eq!(
        BindingPath::parse("combat.target-health"),
        Err(BindingPathError::InvalidSegment { index: 1 })
    );
}

#[test]
fn dependency_plan_dirties_each_evaluator_once_in_id_order() {
    let plan = DependencyPlan::compile(vec![
        EvaluatorDependency {
            evaluator: 20,
            inputs: vec![BindingSlot(2), BindingSlot(0)],
        },
        EvaluatorDependency {
            evaluator: 10,
            inputs: vec![BindingSlot(0), BindingSlot(1)],
        },
    ])
    .unwrap();
    assert_eq!(
        plan.dirty_evaluators(&[BindingSlot(0), BindingSlot(2), BindingSlot(0)]),
        vec![10, 20]
    );
    assert_eq!(
        plan.evaluators()[0].inputs,
        vec![BindingSlot(0), BindingSlot(1)]
    );
}

#[test]
fn dependency_plan_rejects_duplicate_declarations() {
    assert_eq!(
        DependencyPlan::compile(vec![
            EvaluatorDependency {
                evaluator: 1,
                inputs: vec![]
            },
            EvaluatorDependency {
                evaluator: 1,
                inputs: vec![]
            },
        ]),
        Err(DependencyError::DuplicateEvaluator(1))
    );
    assert_eq!(
        DependencyPlan::compile(vec![EvaluatorDependency {
            evaluator: 1,
            inputs: vec![BindingSlot(3), BindingSlot(3)],
        }]),
        Err(DependencyError::DuplicateInput(1))
    );
}
