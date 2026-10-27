# prism_wasm

通用、可嵌入的 **WebAssembly 沙箱引擎** —— 基于 wasmtime 的 Component Model + WASI Preview 2，
为「多租户、零信任、可审计、可热插拔」的 WASM 执行环境打底。

模块本身对载荷不做语义假设；它只关心**可加载、可隔离、可观测、可回滚、可证明**。

---

## 特性

- **Component Model + WASI p2** — 默认即开，类型化调用 `(params) -> results`
- **零信任策略** — Capability 白名单 + Fuel / Deadline / 内存页 / Stdio 资源上限
- **内容寻址** — 所有进入引擎的字节流强制 SHA-256 指纹，作为缓存键与审计 ID
- **可插拔加载器** — `FileLoader` / `MemoryLoader` / `ChainLoader`，HTTP / OCI 自行实现 trait 即可
- **热插拔** — 基于 mtime 轮询 + 原子指针替换，旧调用零中断
- **多沙箱注册表** — 多租户隔离，单租户也可挂多沙箱做版本/影子
- **金丝雀发布** — 流量按权重 / 哈希分发，配合 `SelfHealing` 自动回退坏版本
- **执行回执（Attestation）** — 「零知识风格」承诺：模块摘要 + 入参/出参摘要 + 资源用量
- **eBPF 双桥** — 把内核 syscall 事件与 WASM host-call 事件做交叉验证
- **时间旅行调试** — 快照 / 倒带 / 差分 / 重放

非目标：分布式调度、跨节点复制、领域逻辑。这些应在更上层构建。

---

## 模块布局

```
src/
├── lib.rs              纯 facade + 顶层 re-export
├── runtime/            运行时层
│   ├── engine.rs           wasmtime 引擎 + Component / InstancePre 缓存 + WASI p2 状态
│   ├── sandbox.rs          单沙箱（policy + linker + attestor）
│   ├── registry.rs         多沙箱（多租户）注册表
│   └── canary.rs           金丝雀路由 + 自愈
├── loader/             入口层
│   ├── mod.rs              Loader trait + FileLoader / MemoryLoader / ChainLoader
│   └── hotswap.rs          mtime 轮询 + 原子指针热插拔
├── security/           安全层
│   ├── policy.rs           零信任能力策略 + 资源上限
│   ├── attestation.rs      「零知识风格」执行回执
│   └── ebpf_bridge.rs      eBPF + WASM 双重可信链验证
└── debug/              调试层
    └── timetravel.rs       快照 / 倒带 / 差分 / 重放
```

---

## 快速开始

```rust
use std::sync::Arc;
use std::time::Duration;
use prism_wasm::{
    Attestor, ChainLoader, FileLoader, HotSwap, MemoryLoader,
    Policy, Sandbox, SandboxEngine,
};
use prism_wasm::loader::Loader;

// 1. 引擎 + 零信任策略
let engine = Arc::new(SandboxEngine::new()?);
let policy = Policy::builder()
    .fuel(2_000_000)
    .deadline(Duration::from_millis(500))
    .memory_pages(64)
    .build();

// 2. 链式加载器：内存优先、文件兜底
let mem = Arc::new(MemoryLoader::new());
mem.insert("adder", adder_component_bytes);
let chain = Arc::new(
    ChainLoader::new()
        .push(mem.clone())
        .push(Arc::new(FileLoader::new(["./modules"]))),
);

// 3. 热插拔管理器
let hs = HotSwap::new(engine.clone(), chain.clone() as Arc<dyn Loader>, policy.clone());
hs.track("adder", None)?;

// 4. 沙箱（默认接入 WASI p2）
let attestor = Arc::new(Attestor::ephemeral());
let sb = Sandbox::new("default", engine.clone(), policy, attestor.clone());

// 5. 类型化调用
let r = sb.call_typed::<(i32, i32), (i32,)>("adder", "add", (11, 2))?;
assert_eq!(r.returns, (13,));
assert!(attestor.verify(&r.receipt));
```

完整示例：

```bash
cargo run --example demo                          # 单沙箱端到端
cargo run --example multi                         # 多模块
cargo run --example concurrent                    # 并发调用
cargo run --example multi_sandbox                 # 多租户注册表
cargo run --example single_tenant_multi_sandbox   # 单租户多沙箱（版本/影子）
```

---

## 核心 API

### 引擎与沙箱

```rust
let engine = Arc::new(SandboxEngine::new()?);
let sb = Sandbox::new(name, engine, policy, attestor);
sb.call_typed::<Params, Returns>(module, export, args)?;   // 单次调用 → CallReceipt
```

### 加载器

```rust
FileLoader::new(["./modules", "/usr/local/wasm"]);          // 多搜索根，自动探 .wasm/.wat
MemoryLoader::new(); mem.insert("name", bytes);             // 运行期注入
ChainLoader::new().push(a).push(b);                         // 顺序兜底
```

### 热插拔

```rust
let hs = HotSwap::new(engine, loader, policy);
hs.track("module", None)?;          // 起一个 mtime 轮询
hs.reload("module")?;                // 手动触发，原子指针替换
```

### 多租户 / 金丝雀

```rust
let reg: SharedRegistry = SandboxRegistry::shared();
reg.create("tenant-a", engine.clone(), policy.clone(), attestor.clone());

let router = CanaryRouter::new()
    .add_version("v1", 90)
    .add_version("v2", 10);
let healer = SelfHealing::new(router, /* error_rate_threshold */ 0.2);
```

### 安全策略

```rust
Policy::builder()
    .fuel(2_000_000)                 // 燃料上限
    .deadline(Duration::from_millis(500))
    .memory_pages(64)                // 线性内存页（每页 64KiB）
    .allow(Capability::Stdout)       // Capability 白名单
    .build();
```

### 回执 / eBPF

```rust
let receipt: Receipt = call_receipt.receipt;
attestor.verify(&receipt);                          // 验签
let bridge = EbpfBridge::new();
bridge.verify(&receipt, &syscall_events, &host_events);  // 双链交叉
```

### 时间旅行调试

```rust
let mut session = TimeTravelSession::new();
session.snapshot(&store);            // 任意时刻打点
session.rewind(snapshot_id);         // 倒带
session.diff(a, b);                  // 差分
session.replay(from, to);            // 重放
```

---

## 设计要点

- **内容寻址先行**：所有 `Loader::load` 出口都强制 SHA-256，让 Policy 白名单、引擎缓存、回执三方共用同一把 ID。
- **InstancePre 缓存**：编译产物按 (engine, digest) 缓存，热路径只做 `instantiate_pre + call`。
- **Linker 配置随策略走**：WASI p2 默认接入，但能力按 `Policy` 收紧；不在白名单内的 host call 直接拒绝。
- **热插拔零中断**：旧的 `InstancePre` 在最后一个调用结束后才释放，新调用看见新指针。
- **可证明执行**：`CallReceipt` 把模块摘要、入参/出参摘要、资源用量、wall clock 一起承诺，签名后即审计证据。

---

## 测试

```bash
cargo test -p prism_wasm
```

当前测试：0 个（二进制包，无单元测试）。

---

## License

WIP — internal use.
