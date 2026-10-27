//! 单租户多沙箱 Demo：同一份 WASM component，挂载不同策略 / Linker 配置的沙箱。
//!
//! # 场景
//!
//! 一个"计算服务"只有一个 WASM 模块（`calc`），但需要向不同调用方提供
//! 差异化的执行隔离——通过多个 `Sandbox` 实现，共享同一个 `SandboxEngine`：
//!
//! ```text
//!              ┌──────── SandboxEngine ────────┐
//!  calc.wasm ──┤  内容寻址缓存，只 JIT 编译一次 │
//!  heavy.wasm──┤  component → CachedComponent  │
//!              └───────────────────────────────┘
//!                            │ shared Arc<Engine>
//!            ┌───────────────┼────────────────┐
//!            ▼               ▼                ▼
//!       sandbox_A        sandbox_B        sandbox_C
//!    (默认宽松策略)   (燃料上限 800)   (禁用 WASI 注入)
//!    InstancePre_A    InstancePre_B    InstancePre_C
//! ```
//!
//! 演示要点：
//! 1. 三个沙箱共享 `SandboxEngine`——component 只被 JIT 编译一次，后续均复用
//! 2. 每个沙箱独立持有自己的 `InstancePre` 缓存，Linker 配置互不干扰
//! 3. 沙箱 B 的极低燃料（800）会在 `heavy` 循环中耗尽，不影响 A / C 的正常运行
//! 4. 沙箱 C 通过 `.with_wasi(false)` 禁用 WASI p2 注入，适用于纯计算 component；
//!    其 `InstancePre` 与 A / B 的不同（Linker 内容不同），各自隔离
//! 5. 三个沙箱在阶段 1 并发执行，通过 `JoinSet` 收集各自结果

use std::sync::Arc;
use std::time::Duration;

use prism_wasm::{Attestor, ModuleSource, Policy, Sandbox, SandboxEngine};
use tokio::task::JoinSet;

// ── WASM component 工厂 ───────────────────────────────────────────────────────

/// 普通加法 component：导出 `compute(x: s32, y: s32) -> s32`，直接映射到 `i32.add`。
///
/// 指令数极少（约 3 条），任何合理燃料限制下均可正常完成。
fn make_add_component() -> anyhow::Result<Vec<u8>> {
    let wat = r#"(component
         (core module $m
           (func (export "add") (param i32 i32) (result i32)
             local.get 0 local.get 1 i32.add))
         (core instance $i (instantiate $m))
         (func (export "compute") (param "x" s32) (param "y" s32) (result s32)
           (canon lift (core func $i "add"))))"#;
    Ok(wat::parse_str(wat)?)
}

/// 计算密集型 component：导出 `compute(x: s32, y: s32) -> s32`。
///
/// 循环体每轮约 12 条 wasm 指令（wasmtime 每条消耗 1 fuel）：
///
/// ```text
/// 循环轮数 = x × y（运行时决定，Cranelift 无法常量折叠）
/// 总 fuel  ≈ 12 × (x × y)
/// ```
///
/// 使用 `x=10, y=100` 时：循环 1,000 次，消耗约 12,000 fuel。
/// 沙箱 B 的燃料上限为 800，必然在循环过程中耗尽并返回 `OutOfFuel` 错误。
fn make_heavy_component() -> anyhow::Result<Vec<u8>> {
    let wat = r#"(component
         (core module $m
           (func (export "heavy") (param i32 i32) (result i32)
             (local $acc i32)
             (local $i   i32)
             ;; i = x * y（运行时乘法，阻止常量折叠）
             local.get 0
             local.get 1
             i32.mul
             local.set $i
             (block $done
               (loop $loop
                 ;; 循环头：检查计数器是否归零
                 local.get $i
                 i32.eqz
                 br_if $done
                 ;; 循环体：acc += x；i -= 1
                 local.get $acc
                 local.get 0
                 i32.add
                 local.set $acc
                 local.get $i
                 i32.const 1
                 i32.sub
                 local.set $i
                 br $loop
               )
             )
             local.get $acc
           )
         )
         (core instance $i (instantiate $m))
         (func (export "compute") (param "x" s32) (param "y" s32) (result s32)
           (canon lift (core func $i "heavy"))))"#;
    Ok(wat::parse_str(wat)?)
}

// ── 主程序 ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("═══════════════════════════════════════════");
    println!(" 单租户多沙箱 Demo");
    println!("═══════════════════════════════════════════\n");

    // ── 1. 创建共享引擎，编译两个 component ──────────────────────────────────
    //
    // `SandboxEngine` 持有 wasmtime `Engine`（含 Cranelift 编译器）和
    // 内容寻址 component 缓存（key = SHA-256 摘要）。
    // 同一份字节只会被 JIT 编译一次，后续所有沙箱均复用 `CachedComponent`。

    let engine = Arc::new(SandboxEngine::new()?);

    // `Attestor::ephemeral()` 生成只在进程生命周期内有效的临时签名密钥，
    // 适合演示；生产环境应使用持久化密钥并验证 Receipt。
    let attestor = Arc::new(Attestor::ephemeral());

    // `ModuleSource` 封装 (逻辑名称, 字节内容)，安装时引擎以 SHA-256 摘要去重。
    let add_src = ModuleSource::new("calc", make_add_component()?);
    let heavy_src = ModuleSource::new("heavy", make_heavy_component()?);

    // 启动期同步安装（阻塞可接受），返回 `CachedComponent`（含摘要供后续日志展示）。
    let add_cached = engine.install(add_src, &Policy::default())?;
    let _heavy_cached = engine.install(heavy_src, &Policy::default())?;

    println!(
        "[引擎] 'calc'  已编译，摘要(前8B): {}",
        hex::encode(&add_cached.digest[..8])
    );
    println!("[引擎] 'heavy' 已编译\n");

    // ── 2. 构建三个沙箱（共享同一引擎，策略各异） ────────────────────────────
    //
    // 每个 `Sandbox` 独立持有：
    //   - `Policy`：燃料 / 内存页 / 超时限制
    //   - `configure_linker` 回调：追加宿主导入
    //   - `enable_wasi`：是否向 Linker 注入 WASI p2 接口
    //   - `instance_pre_cache`：已预实例化的 `InstancePre`（按 digest 缓存）
    //
    // 不同沙箱的 `InstancePre` 彼此隔离——相同 component 在不同 Linker 配置下
    // 会生成不同的 `InstancePre`，不能跨沙箱共享。

    // 沙箱 A：默认 `Policy`（10M fuel，1s 超时，无内存页上限）
    //         WASI p2 默认开启；适合宽松的通用计算场景。
    let sandbox_a = Arc::new(Sandbox::new(
        "sandbox_A [宽松]",
        engine.clone(),
        Policy::default(),
        attestor.clone(),
    ));

    // 沙箱 B：极低燃料（800），短超时（200ms），内存上限 4 页（256 KiB）。
    //         调用 `heavy(10, 100)` 循环约 1,000 次（~12,000 fuel）必然耗尽，
    //         但对简单加法（约 3 条指令）仍可正常完成。
    let policy_strict = Policy::builder()
        .fuel(800)
        .deadline(Duration::from_millis(200))
        .memory_pages(4)
        .build();
    let sandbox_b = Arc::new(Sandbox::new(
        "sandbox_B [燃料=800]",
        engine.clone(),
        policy_strict,
        attestor.clone(),
    ));

    // 沙箱 C：通过 `.with_wasi(false)` 完全跳过 WASI p2 向 Linker 的注入。
    //         适用于不调用任何 WASI 接口的纯计算 component：
    //           - 构建 `InstancePre` 时少遍历大量 WASI 函数定义
    //           - 运行时每次调用省去 `default_wasi_ctx()` 内的 3 次 dup() 系统调用
    //           - 沙箱能力更严格：即便 component 尝试调用 WASI，也会在链接阶段失败
    let sandbox_c = Arc::new(
        Sandbox::new(
            "sandbox_C [no-WASI]",
            engine.clone(),
            Policy::default(),
            attestor.clone(),
        )
        .with_wasi(false),
    );

    println!("三个沙箱已创建，共享同一个 SandboxEngine（不重复 JIT 编译）\n");

    // ── 3. 阶段 1：并发调用三个沙箱 ─────────────────────────────────────────
    //
    // `call_typed_async` 将阻塞的 WASM 执行包裹进 `tokio::task::spawn_blocking`，
    // 避免阻塞 tokio 的异步执行器线程。三个 spawn_blocking 并行提交到线程池。

    println!("─── 阶段 1：并发调用（各沙箱独立运行）───\n");

    let mut tasks: JoinSet<()> = JoinSet::new();

    // 沙箱 A：调用 calc.compute(42, 58)，预期返回 100。
    {
        let sb = sandbox_a.clone();
        tasks.spawn(async move {
            match Arc::clone(&sb)
                .call_typed_async::<(i32, i32), (i32,)>("calc", "compute", (42, 58))
                .await
            {
                Ok(r) => println!(
                    "[{}] compute(42, 58) = {}  fuel_consumed={:?}  elapsed={}ms  receipt={}",
                    sb.name(),
                    r.returns.0,
                    r.fuel_consumed,
                    r.elapsed_ms,
                    hex::encode(&r.receipt.commitment[..6]),
                ),
                Err(e) => println!("[{}] 错误: {e:#}", sb.name()),
            }
        });
    }

    // 沙箱 B：调用 heavy.compute(10, 100) → 循环 1,000 次，消耗约 12,000 fuel，
    //         远超燃料上限 800，预期以 OutOfFuel 错误失败。
    {
        let sb = sandbox_b.clone();
        tasks.spawn(async move {
            match Arc::clone(&sb)
                .call_typed_async::<(i32, i32), (i32,)>("heavy", "compute", (10, 100))
                .await
            {
                Ok(r) => println!(
                    "[{}] compute(10, 100) = {}  fuel_consumed={:?}",
                    sb.name(),
                    r.returns.0,
                    r.fuel_consumed,
                ),
                Err(e) => println!(
                    "[{}] 预期失败（燃料耗尽）✓  错误: {}",
                    sb.name(),
                    // 只取第一行，避免长栈信息污染输出
                    e.to_string().lines().next().unwrap_or("unknown"),
                ),
            }
        });
    }

    // 沙箱 C：调用 calc.compute(100, 200)，no-WASI 模式下纯计算同样正常运行，
    //         预期返回 300。WASI 接口未被注入到 Linker，但 calc 本身不调用 WASI，
    //         因此不影响结果。
    {
        let sb = sandbox_c.clone();
        tasks.spawn(async move {
            match Arc::clone(&sb)
                .call_typed_async::<(i32, i32), (i32,)>("calc", "compute", (100, 200))
                .await
            {
                Ok(r) => println!(
                    "[{}] compute(100, 200) = {}  fuel_consumed={:?}",
                    sb.name(),
                    r.returns.0,
                    r.fuel_consumed,
                ),
                Err(e) => println!("[{}] 错误: {e:#}", sb.name()),
            }
        });
    }

    // 等待三个并发任务全部完成（任意顺序）
    while let Some(r) = tasks.join_next().await {
        r?;
    }

    // ── 4. 阶段 2：隔离验证（沙箱 B 失败不影响 A / C） ──────────────────────
    //
    // 验证沙箱之间的状态完全隔离：B 在 heavy 调用中燃料耗尽，
    // 不会污染 A / C 的 Store，后续调用均可正常完成。

    println!("\n─── 阶段 2：隔离验证（B 失败后，A/C 仍可正常运行）───\n");

    let ra = sandbox_a.call_typed::<(i32, i32), (i32,)>("calc", "compute", (999, 1))?;
    println!("[sandbox_A] 999 + 1 = {}  ✓", ra.returns.0);
    assert_eq!(ra.returns.0, 1000);

    // 沙箱 B 对简单加法（约 3 条指令，远低于 800 fuel）仍可正常运行，
    // 证明 B 并未因上一次 heavy 失败而进入损坏状态。
    let rb = sandbox_b.call_typed::<(i32, i32), (i32,)>("calc", "compute", (3, 4))?;
    println!(
        "[sandbox_B] 3 + 4 = {}  ✓（简单加法燃料充足）",
        rb.returns.0
    );
    assert_eq!(rb.returns.0, 7);

    let rc = sandbox_c.call_typed::<(i32, i32), (i32,)>("calc", "compute", (50, 50))?;
    println!("[sandbox_C] 50 + 50 = {}  ✓", rc.returns.0);
    assert_eq!(rc.returns.0, 100);

    // ── 5. 阶段 3：InstancePre 缓存验证 ──────────────────────────────────────
    //
    // `resolve_pre` 采用双重检查锁：
    //   - 快路径（读锁）：第 2 次及以后命中缓存，无需重新构建 InstancePre
    //   - 慢路径（写锁）：首次调用时构建并写入缓存
    //
    // 连续 4 次调用中，第 1 次触发慢路径（elapsed 略高），
    // 第 2-4 次均走快路径（elapsed 更低），可从输出中观察到差异。

    println!("\n─── 阶段 3：InstancePre 缓存（沙箱 A 连续调用）───\n");
    for i in 1..=10000000_i32 {
        let r = sandbox_a.call_typed::<(i32, i32), (i32,)>("calc", "compute", (i, i * 10))?;
        println!(
            "  call #{i}: compute({i}, {}) = {}  fuel_consumed={:?}  elapsed={}ms",
            i * 10,
            r.returns.0,
            r.fuel_consumed,
            r.elapsed_ms,
        );
        assert_eq!(r.returns.0, i + i * 10);
    }

    // ── 汇总 ─────────────────────────────────────────────────────────────────

    println!("\n═══════════════════════════════════════════");
    println!(" 汇总");
    println!("═══════════════════════════════════════════");
    println!("沙箱数量    : 3（共享 1 个 SandboxEngine）");
    println!("JIT 编译    : 2 次（calc + heavy，各自只编译一次）");
    println!("策略差异    : A=默认宽松  /  B=燃料 800  /  C=禁用 WASI 注入");
    println!("隔离性      : B 的 heavy 燃料耗尽，不影响 A / C 的正常调用");
    println!("InstancePre : 每个沙箱独立缓存，Linker 配置差异不跨沙箱污染");
    println!("✓ 所有断言通过");

    Ok(())
}
