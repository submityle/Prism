//! 变量节点：固定计数器演示。
//!
//! TODO：等 IR 接入 `LoadVar/StoreVar` 指令后，把这里换成通用 `var.set` / `var.get`，
//! 变量名通过 `Node.config` 传入。

use super::pins::{data_out, exec_in, exec_out};
use crate::registry::{ExecNext, FnRunner, NodeDefinition, NodeLibrary, Purity, RunnerKind};
use crate::value::{Value, ValueType};

pub fn register(lib: &mut NodeLibrary) {
    lib.register(NodeDefinition {
        id: "var.inc_counter".into(),
        purity: Purity::Impure,
        inputs: vec![exec_in()],
        outputs: vec![exec_out("then"), data_out("after", ValueType::F64)],
        runner: RunnerKind::sync(FnRunner(|_in, out, cx| {
            let cur = cx
                .host
                .var_get("counter")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let next = cur + 1.0;
            cx.host.var_set("counter", Value::F64(next));
            out[0] = Value::F64(next);
            Ok(ExecNext::Pin("then".into()))
        })),
    });
}
