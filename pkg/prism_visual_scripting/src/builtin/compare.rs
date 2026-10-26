//! 比较节点：greater。

use super::pins::{data_in, data_out};
use crate::registry::{ExecNext, FnRunner, NodeDefinition, NodeLibrary, Purity, RunnerKind};
use crate::value::{Value, ValueType};

pub fn register(lib: &mut NodeLibrary) {
    lib.register(NodeDefinition {
        id: "cmp.greater".into(),
        purity: Purity::Pure,
        inputs: vec![
            data_in("a", ValueType::F64, Some(Value::F64(0.0))),
            data_in("b", ValueType::F64, Some(Value::F64(0.0))),
        ],
        outputs: vec![data_out("out", ValueType::Bool)],
        runner: RunnerKind::sync(FnRunner(|inp, out, _cx| {
            let a = inp.first().and_then(|v| v.as_f64()).unwrap_or(0.0);
            let b = inp.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            out[0] = Value::Bool(a > b);
            Ok(ExecNext::End)
        })),
    });
}
