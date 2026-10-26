//! 调试节点：print。

use super::pins::{data_in, exec_in, exec_out};
use crate::registry::{
    ExecNext, FnRunner, LogLevel, NodeDefinition, NodeLibrary, Purity, RunnerKind,
};
use crate::value::{Value, ValueType};
use std::sync::Arc;

pub fn register(lib: &mut NodeLibrary) {
    lib.register(NodeDefinition {
        id: "debug.print".into(),
        purity: Purity::Impure,
        inputs: vec![
            exec_in(),
            data_in("msg", ValueType::String, Some(Value::String(Arc::from("")))),
        ],
        outputs: vec![exec_out("then")],
        runner: RunnerKind::sync(FnRunner(|inp, _out, cx| {
            let msg = inp.first().and_then(|v| v.as_str()).unwrap_or("");
            cx.host.log(LogLevel::Info, msg);
            Ok(ExecNext::Pin("then".into()))
        })),
    });
}
