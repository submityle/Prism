# E5：真实 Prism ECS remote adapter

实现位于 `pkg/prism_editor_remote/src/ecs.rs`，通过可选 feature `ecs` 启用；
依赖原生 `prism_ecs` 的 `reflect` bridge 与 `prism_reflect`，不依赖 Bevy。
`remote` 的依赖、feature 与 `pub mod ecs` 已接入，无需主线程补导出。

## 数据来源与宿主接线

`EcsSyncStore<'w>` 独占借用真实 `prism_ecs::world::World`。读取经
`World::get_reflect` 访问 typed column；写入在转换全部成功后，通过
`World::get_mut::<T>` 交换真实组件。内部 `WorldSnapshot` 仅用于协议投影、
版本比较和历史缓存，不是替代 engine 的组件存储。

公开入口：

- `EcsSyncStore::new(&mut World, TargetId, EcsSyncLimits)`。
- `register_component::<T>() -> ComponentId`：白名单开放组件，以 Rust 完整
  `type_name::<T>()` 为协议键；要求 `Component + Reflect + FromReflect +
  GetTypeRegistration + Clone`。
- `register_value_type::<T>()`：登记嵌套复合类型，包括列表、固定数组、枚举、
  map 及其嵌套元素；标量无需登记。未登记形状可以读取，但 typed 写入明确失败。
- `world()`、`world_mut()`：访问同一个 engine world。
- `entity_id(Entity)`、`synchronize()` 与现有 `SyncStore` 的 snapshot、changes、
  apply 接口。
- `PrismEcsRuntimeAdapter::new(TargetId)`：供现有 `RuntimeBridge` 启动真实宿主。

通用绑定示例（业务组件由调用者指定）：

```rust
use prism_ecs::{component::Component, world::World};
use prism_editor_remote::{ecs::{EcsSyncLimits, EcsSyncStore}, SyncEndpoint, SyncError, TargetId};
use prism_reflect::{FromReflect, GetTypeRegistration, Reflect};

fn bind_editor_world<T>(
    world: &mut World, target: TargetId,
) -> Result<SyncEndpoint<EcsSyncStore<'_>>, SyncError>
where
    T: Component + Reflect + FromReflect + GetTypeRegistration + Clone,
{
    let mut store = EcsSyncStore::new(world, target, EcsSyncLimits::default())?;
    store.register_component::<T>();
    // 在这里为 T 使用的复合字段调用 register_value_type。
    Ok(SyncEndpoint::new(store))
}
```

宿主在自己的 ECS 线程、schedule barrier 处理请求。`World` 的独占借用防止
同步与 systems 同时写入；不要求把 engine world 移入 transport worker。
宿主阶段通过 `endpoint.store_mut().world_mut()?` 操作 World，再服务下一批
请求。endpoint 可直接交给现有 `JsonRpcServer` / `JsonRpcServerSession`。
宿主自行选择场景加载、可编辑组件、线程调度与进程入口；本模块不注入业务模型。

`PrismEcsRuntimeAdapter` 保留调用者的 executable、args 和 env，拒绝空 executable。
启动时在同一个总 deadline 内完成 `prism/initialize` 与真实
`prism/world/readSnapshot`，验证协议主版本、`prism-ecs` endpoint、目标与六项
同步能力。RPC 错误及无效 snapshot 会使启动失败；不能用仅返回初始化成功的
宿主冒充就绪。进程、帧大小、队列、取消和超时沿用现有 RuntimeBridge 限制。
本次未修改 native app；UI、场景持久化与应用的启动菜单仍由宿主集成。

## 精确值映射

协议值以登记的 Rust schema 为准，不猜类型、不进行截断或有损数值转换。

| Rust 形状 | 协议 `Value` | 边界 |
| --- | --- | --- |
| bool | Bool | 严格匹配 |
| i8/i16/i32/i64/isize | 负数 Signed，零与正数 Unsigned | 按具体 Rust 宽度检查；正数 Signed 非规范，拒绝 |
| u8/u16/u32/u64/usize | Unsigned | 按具体 Rust 宽度检查 |
| f32 | Unsigned，保存原始 u32 bits | 保留 NaN payload 与负零，不接受普通数值 float 替代 |
| f64 | Float，保存原始 u64 bits | JSON 使用现有 `$prism.floatBits` 标签 |
| i128/u128 | Bytes，恰好 16 字节小端 | JSON 使用现有 `$prism.bytes` 标签 |
| String / char | String | char 必须恰好一个 Unicode scalar；不 trim 文本 |
| named struct | Object | 必须包含全部且仅包含声明字段 |
| tuple struct | Array | 长度必须与声明相同 |
| Vec / 固定数组 | Array | 列表可增减；固定数组长度严格一致 |
| enum | `{variant: String, fields: ...}` | unit 为 Null、tuple 为 Array、named 为 Object；变体及字段精确匹配 |
| String-key map | Object | 完整重建，支持真正删除条目 |

非负有符号整数采用 Unsigned，是为了与 JSON 的非负整数解析保持一致，
而非丢弃 Rust 类型信息：写入始终以白名单组件及其字段 schema 验证。
`isize` / `usize` 遵循宿主位宽。组件路径和 schema 必须与宿主构建一致；
本层不提供跨 schema 版本迁移。

仅支持 String map key；`$prism.bytes`、`$prism.floatBits` 两个 map key 因与
现有 JSON 标签冲突而明确拒绝，避免往返后静默改变值种类。集合、非 String
key map、未知 opaque leaf 及其他未支持的 reflection shape 返回明确错误，
不会省略字段后返回成功快照。

写入采用完整 `FromReflect` 重建组件，包含 `SetProperty`。因此开放组件必须
能够由公开反射形状完整重建：被忽略字段遵循 `FromReflect` 的默认值，
不承诺保留其旧值。含重要隐藏状态的组件不能直接作为可写投影，应由宿主
提供适合编辑的组件或自定义反射契约。自定义 Clone / FromReflect / Drop /
reflection 实现属于可信宿主代码，必须无 panic、无外部副作用。

## 原子提交与 engine 行为

每个请求首先核对真实 ECS 的刷新边界，再验证精确 base revision。批次依次执行
命令预算、实体身份、组件白名单、存在性、存储写权限与路径预算检查，
在协议投影上计算最终值。全部动态值转换为具体 `T`，再编码核对精确相等，
并验证最终完整投影预算。最终 delta、revision、历史与替换容器都在写入前准备好。

仅在全部准备成功后交换所有 typed 值，再发布协议状态并释放旧值。
后一个实体/组件转换失败时，前一个组件也不会写入；拒绝不会部分推进 revision。
真正无值变化的批次不推进 revision，也不改变 ECS change tick。

可远程修改已有 Table / Sparse 组件；Shared 组件只读。通过普通 `Mut<T>`
标记 Changed，不触发结构性 on_insert/on_replace hook。远程创建/删除实体、
增删组件以及向缺失组件执行 SetComponent 均明确拒绝。任意结构 hook 的
副作用无法由当前协议可靠回滚，因此结构修改交由宿主在 barrier 执行，
其真实 spawn/despawn/add/remove 会被后续 snapshot / delta 观察到。
原子承诺针对正常返回的协议成功/错误，不是对 OOM 或可信用户代码 panic 的回滚。

## revision、实体与刷新边界

首次同步、登记组件、取得 `world_mut()` 或 archetype generation 改变后扫描真实列。
其余查询复用已验证投影，不遍历实体或再次编码组件；真实远程批写已在发布前
验证并更新投影，因此其后的查询也可直接复用。独占 World 借用保证所有宿主
可变访问经过边界。每次请求仍以常数成本检查 World 身份、reservation 和实体数。
扫描失败不替换已缓存状态，也不清除刷新要求；后续请求继续扫描并报错，
不会把旧缓存包装成成功。`world_mut()` 在失败后仍可修复数据。实体 reservation
必须先 flush 再由 spawn_at 完成放置；未 flush 或仍 live/unplaced 均拒绝同步。

`projection_work()` 返回实际累计扫描次数、访问的实体行与编码的组件数，
统计包含失败扫描且饱和不回绕。空闲 `read_changes` 无历史数据时不复制完整世界；
`read_snapshot` 仍复制完整协议投影，宿主刷新与远程批写的 staging/最终预算检查
仍与完整投影大小相关，不能据此宣称全部路径成本都只随变化量增长。

每次成功取得 `world_mut()` 都标记宿主边界；下次同步使旧 revision 失效，
即使值相同或经历 A→B→A。未观察的多个宿主阶段可以合并成一个 revision。
无值变化的宿主边界会清空增量历史，旧客户端必须重新读 snapshot。
这是显式版本围栏，不虚构实体变动来填补空增量。

wire EntityId 高 64 位为 adapter session / restore namespace，低 64 位
保留 engine 的 index32/generation32。despawn 后重用 index 不接受旧 ID；
World restore / archetype generation 改变会轮换 namespace。
整个 World 被替换时拒绝继续服务，必须新建 adapter。

每次 adapter 重建、进程重启或重连必须由宿主分配新的 TargetId，不能用旧
target 重新从零发版。namespace 计数器只在当前进程唯一，不是持久标识；
engine 自身 generation32 的回绕上限仍是其边界。revision / namespace
计数耗尽则明确报错，要求新 session，不允许溢出混用。

## 预算与历史

`EcsSyncLimits` 的六项限制必须非零，默认：16,384 个实体、262,144 个
value node、8MiB 字符串/字节 payload、深度 64、65,536 条命令、16 批历史。

节点/字节限制针对每次输入、完整投影和合并增量响应；字段名、组件路径、
map key、字符串与 byte payload 均计入字节预算。它不是进程 RSS 或总 wire
frame 大小限制，后者由 codec/transport limits 控制。投影缓存、事务 staging
和历史批次会各自占用内存，不能把默认 8MiB 当成 adapter 总内存上限。

历史只保留有界批次；合并响应还受命令数、节点数、字节数与深度约束。
过旧 revision、宿主无变化围栏、历史不足或增量预算超限，调用者需重新
读取完整 snapshot。宿主一次结构变动超过 delta 命令预算时丢弃历史，
仍允许在投影预算内返回真实完整 snapshot。未来 revision 明确返回冲突。

## 验证记录（2026-10-21）

在共享 checkout 使用独立 target `/private/tmp/prism-remote-ecs.Vyz4fq`，
并固定 `CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0`：

```sh
CARGO_TARGET_DIR=/private/tmp/prism-remote-ecs.Vyz4fq cargo test -p prism_editor_remote --all-features --offline
CARGO_TARGET_DIR=/private/tmp/prism-remote-ecs.Vyz4fq cargo clippy -p prism_editor_remote --all-targets --all-features --no-deps --offline -- -D warnings
cargo clean --target-dir /private/tmp/prism-remote-ecs.Vyz4fq
```

实际通过 32 项包内测试、19 项 ECS 测试（包含子进程宿主入口）、9 项进程桥
回归；strict Clippy 成功。依赖 prism_ecs 的 reflect-only 构建仍打印已有
`apply_from_reflect` dead_code 警告，未修改或屏蔽依赖源码。

ECS 回归覆盖 Table / Sparse / Shared、真实变更 tick、原子多实体失败、
结构拒绝、列表收缩/增长、enum 切换、map 删除、保留键拒绝、嵌套类型登记、
数值精度与越界、ABA 冲突、历史与聚合预算、实体重用、restore、reservation、
World 替换以及失败扫描后的宿主修复。Unix 子进程测试启动真实 ECS World，
经 RuntimeBridge → TCP → JSON-RPC → EcsSyncStore 修改具体组件，再验证
宿主直接修改被客户端读回；未用 InMemoryStore / Loopback 替代 engine。
子进程测试的 libtest 启动包装使用 `/bin/sh`，仅在 Unix 执行；此处验证
平台为 macOS，未声明完成 Windows 运行验收。

验证结束后仅清理上述自有 target，cargo 报告回收 99.7MiB / 275 个文件；
没有保留活动验证命令，未清理其他任务的 target。

同日的空闲查询优化在共享 target `/private/tmp/prism-editor-joint-cr1ODK` 验证：
`cargo test --offline -p prism_editor_remote --all-features --all-targets` 通过
32 项包内、21 项 ECS 和 9 项进程桥测试；对应全部 targets/features 的
`cargo clippy --offline -p prism_editor_remote --all-targets --all-features --no-deps -- -D warnings`
通过。新增回归在 4095 个真实实体上连续执行 1000 次增量查询和身份查询，
确认无额外投影；验证远程写入后复用、宿主写入后仅刷新一次，以及空世界
首次同步和连续失败刷新不能返回陈旧成功。共享 target 仍供其他编辑器验证使用，
尚未清理。
