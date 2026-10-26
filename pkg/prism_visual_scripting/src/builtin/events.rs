//! 事件入口节点。

use super::pins::exec_out;
use crate::registry::{ExecNext, FnRunner, NodeDefinition, NodeLibrary, Purity, RunnerKind};

pub fn register(lib: &mut NodeLibrary) {
    lib.register(NodeDefinition {
        id: "event.begin".into(),
        purity: Purity::Impure,
        inputs: vec![],
        outputs: vec![exec_out("then")],
        runner: RunnerKind::sync(FnRunner(|_in, _out, _cx| Ok(ExecNext::Pin("then".into())))),
    });
}
