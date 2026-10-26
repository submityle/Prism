//! 数学节点：add / mul。

use super::pins::{data_in, data_out};
use crate::registry::{ExecNext, FnRunner, NodeDefinition, NodeLibrary, Purity, RunnerKind};
use crate::value::{Value, ValueType};

pub fn register(lib: &mut NodeLibrary) {
    lib.register(NodeDefinition {
        id: "math.add".into(),
        purity: Purity::Pure,
        inputs: vec![
            data_in("a", ValueType::F64, Some(Value::F64(0.0))),
            data_in("b", ValueType::F64, Some(Value::F64(0.0))),
        ],
        outputs: vec![data_out("sum", ValueType::F64)],
        runner: RunnerKind::sync(FnRunner(|inp, out, _cx| {
            let a = inp.first().and_then(|v| v.as_f64()).unwrap_or(0.0);
            let b = inp.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            out[0] = Value::F64(a + b);
            Ok(ExecNext::End)
        })),
    });

    lib.register(NodeDefinition {
        id: "math.mul".into(),
        purity: Purity::Pure,
        inputs: vec![
            data_in("a", ValueType::F64, Some(Value::F64(1.0))),
            data_in("b", ValueType::F64, Some(Value::F64(1.0))),
        ],
        outputs: vec![data_out("product", ValueType::F64)],
        runner: RunnerKind::sync(FnRunner(|inp, out, _cx| {
            let a = inp.first().and_then(|v| v.as_f64()).unwrap_or(0.0);
            let b = inp.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            out[0] = Value::F64(a * b);
            Ok(ExecNext::End)
        })),
    });
}
