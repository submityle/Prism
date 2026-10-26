//! 流程控制节点：分支 / 直通。

use super::pins::{data_in, exec_in, exec_out};
use crate::registry::{ExecNext, FnRunner, NodeDefinition, NodeLibrary, Purity, RunnerKind};
use crate::value::{Value, ValueType};

pub fn register(lib: &mut NodeLibrary) {
    // flow.branch: cond -> true / false
    lib.register(NodeDefinition {
        id: "flow.branch".into(),
        purity: Purity::Impure,
        inputs: vec![
            exec_in(),
            data_in("cond", ValueType::Bool, Some(Value::Bool(false))),
        ],
        outputs: vec![exec_out("true"), exec_out("false")],
        runner: RunnerKind::sync(FnRunner(|inp, _out, _cx| {
            let cond = inp.first().and_then(|v| v.as_bool()).unwrap_or(false);
            Ok(ExecNext::Pin(if cond { "true" } else { "false" }.into()))
        })),
    });

    // flow.passthrough: 单路直通（占位，未来可扩展为 sequence 多路）
    lib.register(NodeDefinition {
        id: "flow.passthrough".into(),
        purity: Purity::Impure,
        inputs: vec![exec_in()],
        outputs: vec![exec_out("then")],
        runner: RunnerKind::sync(FnRunner(|_in, _out, _cx| Ok(ExecNext::Pin("then".into())))),
    });
}
