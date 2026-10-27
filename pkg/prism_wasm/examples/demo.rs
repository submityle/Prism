//! 单沙箱端到端示例（Component Model + WASI Preview 2）。
//!
//! 演示流程与核心模块版本一致，但底层全部走 component：
//!   1. 构造引擎（已默认开启 component model）+ 零信任策略；
//!   2. 链式加载器（内存优先、文件兜底）；
//!   3. 热插拔管理器登记一个内存 component；
//!   4. 沙箱默认接入 WASI p2，无需用户手动配置；
//!   5. 类型化调用 `add: (s32, s32) -> s32`；
//!   6. 热插拔：把加法换成乘法，再调一次。

use std::sync::Arc;
use std::time::Duration;

use prism_wasm::loader::Loader;
use prism_wasm::{
    Attestor, ChainLoader, FileLoader, HotSwap, MemoryLoader, Policy, Sandbox, SandboxEngine,
};

/// 最简 component：导出顶层 `add: (s32, s32) -> s32`，运算符通过参数注入。
fn adder_component(op: &str) -> anyhow::Result<Vec<u8>> {
    let wat = format!(
        r#"(component
             (core module $m
               (func (export "op") (param i32 i32) (result i32)
                 local.get 0 local.get 1 {op}))
             (core instance $i (instantiate $m))
             (func (export "add") (param "a" s32) (param "b" s32) (result s32)
               (canon lift (core func $i "op"))))"#
    );
    Ok(wat::parse_str(&wat)?)
}

fn main() -> anyhow::Result<()> {
    // 1. 引擎 + 零信任策略
    let engine = Arc::new(SandboxEngine::new()?);
    let policy = Policy::builder()
        .fuel(2_000_000)
        .deadline(Duration::from_millis(500))
        .memory_pages(64)
        .build();

    // 2. 链式加载器
    let mem = Arc::new(MemoryLoader::new());
    mem.insert("adder", adder_component("i32.add")?);
    let chain = Arc::new(
        ChainLoader::new()
            .push(mem.clone())
            .push(Arc::new(FileLoader::new(["./modules", "/usr/local/wasm"]))),
    );

    // 3. 热插拔管理器
    let hs = HotSwap::new(
        engine.clone(),
        chain.clone() as Arc<dyn Loader>,
        policy.clone(),
    );
    hs.track("adder", None)?;

    // 4. 沙箱（默认已接入 WASI p2）
    let attestor = Arc::new(Attestor::ephemeral());
    let sb = Sandbox::new("default", engine.clone(), policy, attestor.clone());

    // 5. 类型化调用
    let r = sb.call_typed::<(i32, i32), (i32,)>("adder", "add", (11, 2))?;
    println!("返回值        = {:?}", r.returns);
    println!("消耗燃料      = {:?}", r.fuel_consumed);
    println!("耗时          = {} ms", r.elapsed_ms);
    println!("Component摘要 = {}", hex::encode(r.receipt.module_digest));
    println!("聚合承诺      = {}", r.receipt.commitment_hex());
    println!("回执验签      = {}", attestor.verify(&r.receipt));

    // 6. 热插拔：加法 → 乘法
    mem.insert("adder", adder_component("i32.mul")?);
    if let Some(new_digest) = hs.reload("adder")? {
        println!("已热插拔到摘要 {}", hex::encode(new_digest));
    }
    let r2 = sb.call_typed::<(i32, i32), (i32,)>("adder", "add", (6, 7))?;
    println!("热插拔后调用结果 = {:?}", r2.returns);
    let r3 = sb.call_typed::<(i32, i32), (i32,)>("adder", "add", (6, 8))?;
    println!(
        "再调一次       = {:?}  // 加法应得 14、乘法应得 48",
        r3.returns
    );
    Ok(())
}
