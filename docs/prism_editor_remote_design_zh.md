# Prism 编辑器运行时桥设计方案（prism_editor_remote）
> v1 / 顶级次世代 AAA 级 — 编辑器 ↔ 运行时的协议、传输、订阅、安全、稳定一体化设计

> 面向 Prism(Bevy fork)的**运行时解耦、跨进程、可内省、规模无关**的编辑器远程协议层。
> 本 crate 是 Loom Studio 编辑器(`prism_editor_framework_design_zh.md` §4/§13 的 BRP 前置待建项)的**运行时桥地基**:
> 让编辑器(可同进程、可独立进程、可远程机)**查询 / 变更 / 订阅**运行时 World 的实体、组件、资源、调度与事件,
> 并承载命令(撤销/重做)、资产操作、拾取/Gizmo 旁路、协同同步的底层通路。
>
> 借形态不抄码,系统性对标:
> **Unreal**（Remote Control API / Multi-User Editing / Live Link / Concert）、
> **Unity**（Editor Remote / Live Link / UnityLink）、
> **Omniverse**（Nucleus / USD Live Layer / 实时分层协同）、
> **Chrome DevTools Protocol(CDP)**（域/方法/事件三元模型、会话多路复用）、
> **LSP / DAP**（JSON-RPC 能力协商、请求/通知/双向）、
> **Figma·Zed**（CRDT 增量、低延迟会话）、**gRPC·FlatBuffers·Cap'n Proto**（schema 化、零拷贝）、
> **QUIC·WebTransport**（多路复用、0-RTT 重连、拥塞控制）、**Redis RESP·PostgreSQL wire**（紧凑帧 + 流水线）。
>
> 本文为**设计规格**。地基层(`bevy_remote` 的 BRP 实现、`prism_reflect` 反射/序列化/安全边界、`prism_tasks` 异步、
> `prism_ui_store`/`prism_ui_async` 客户端消费)为**已交付**(SHIPPED);**`prism_editor_remote` 本体为全新规划项(PLANNED)**。
> 全文严格区分「已实现并通过测试」与「规划中」,不把未落地能力描述为已落地。
>
> **非目标**:不含 AI / ML / 神经网络 / LLM;不替代通用 RPC 框架(gRPC/tonic),而是**面向编辑器语义、绑定 Prism 反射与 ECS**的专用桥;
> 协同编辑的**冲突消解(CRDT/OT)不在本层**,本层只保证可靠、有序、可订阅的消息通路(见 §13)。

- 版本: v1(初始设计)
- 适用引擎: Prism / Bevy ECS 生态
- Crate 名: `prism_editor_remote`
- 关键地基 — 已就绪(SHIPPED): `bevy_remote`(BRP / JSON-RPC 2.0 / `world.*` + `+watch` / HTTP 传输)、
  `prism_reflect`(`TypeRegistry` + `StableTypeId` + `ReflectSerialize` + 反射驱动序列化 + schema 版本化迁移 + `net_delta` 字段增量 + 不可信反序列化安全边界)、
  `prism_tasks`(work-stealing 线程池 + `FrameArenas` + `join`)、`prism_diagnostic`、`prism_app`、
  `prism_ui_store`/`prism_ui_async`(客户端状态/订阅消费)
- 关键地基 — 保留 Bevy 机制底座: `bevy_remote`(协议参照与服务端复用,过渡期桥接)、`bevy_ecs`(World/Schedule/Observer)
- 前置待建(本 crate 落地后方可被编辑器挂接): 见 §16 路线图 M0–M6
- 核心契约: 继承 Loom「**成本 ∝ 变化量**」;一切读走订阅差分,一切写走命令事务,一切长任务异步可取消,一切输入默认不可信。

---

## 目录
1. 设计哲学与核心契约
2. 顶级产品对标:借形态、取什么、不抄什么
3. 分层架构总览
4. 传输层(Transport):同进程 / IPC / TCP / QUIC / WebSocket
5. 会话层(Session):握手 / 能力协商 / 多路复用 / 心跳 / 重连
6. 消息层(Wire):帧格式 / 编码 / 批处理 / 流水线
7. 语义层(Methods):域 · 方法 · 事件三元模型
8. 订阅与增量(Watch / Delta):成本 ∝ 变化量
9. 命令事务通道(Command / Undo-Redo 协同)
10. 性能工程:零拷贝 / 帧预算 / 背压 / 采样
11. 安全:认证 / 鉴权 / 加密 / 能力模型 / DoS 防护 / 审计
12. 稳定性:版本协商 / 幂等 / 重连恢复 / 错误模型 / 降级
13. 与协同编辑 / Multi-User 的关系
14. 服务端集成(ECS systems / 调度点 / extract)
15. 客户端集成(prism_ui_store / async / Inspector)
16. Crate 结构 / 模块 / feature / 路线图(M0–M6)
17. 公共 API 草图
18. 风险与取舍
19. 诚实边界(SHIPPED vs PLANNED)
20. 术语表

---

## 1. 设计哲学与核心契约

在 Loom 编辑器七条铁律基础上,为「运行时桥」这一特定层加固:

- **R1 运行时解耦优先**:编辑器与运行时边界清晰,**可同进程(零拷贝直通)、可独立进程(IPC)、可远程机(网络)**。同一套语义 API 覆盖三种拓扑,业务层不感知传输。
- **R2 一切读走订阅差分**:编辑器 Inspector/大纲/Profiler 的实时刷新,**不轮询**;客户端注册订阅,服务端按帧只推**变化量**(复用 `prism_reflect::net_delta` 的 `DirtyMask`),成本随变化而非随 World 规模。
- **R3 一切写走命令事务**:远程变更不是裸 `insert_component`,而是封装成**可撤销、可序列化、可仲裁**的命令(对齐编辑器 §4 命令内核);即使跨进程,撤销/重做语义一致。
- **R4 反射驱动、无需手写**:组件/资源的序列化、schema、版本迁移全部走 `prism_reflect`,新增类型**零协议改动**。编辑器拿到的是类型安全视图,不是裸 JSON。
- **R5 一切输入默认不可信**:远程来的每个字节都经 `prism_reflect` 不可信反序列化安全边界(分配上限、不 panic、未知类型拒绝);传输层叠加认证/鉴权/速率/能力白名单。
- **R6 失败可见 + 优雅降级**:断线不丢状态——重连后做增量对账(resync);版本不匹配时协商降级而非崩溃;长任务异步可取消,永不冻结服务端主循环。
- **R7 可观测**:每条消息、每个订阅、每次重连都进 `prism_diagnostic`;协议本身可被 Profiler/VisLog 观测(自举)。
- **R8 规模无关**:十万实体订阅、GB 级资产流、公里级世界下,桥的延迟与带宽随**可见/变化**子集缩放,不随 World 总量线性恶化。

---

## 2. 顶级产品对标:借形态、取什么、不抄什么

| 来源 | 借鉴形态 | 取什么到 `prism_editor_remote` | 不抄什么 |
|---|---|---|---|
| **Chrome DevTools Protocol** | Domain.method + Domain.event + sessionId 多路复用 | §7 域/方法/事件三元模型;一连接多会话 | 浏览器特有域 |
| **LSP / DAP** | JSON-RPC `initialize` 能力协商、请求/通知/双向、`$/cancelRequest` | §5 握手能力协商;§9 取消语义 | 编辑文本特化语义 |
| **Unreal Remote Control** | 暴露属性/函数的可寻址 preset、HTTP+WebSocket 双通道 | §7 property/function 可寻址;§4 WebSocket 远程通道 | 蓝图特化、商业实现 |
| **Unreal Multi-User(Concert)** | 事务复制、会话快照、权威仲裁 | §9 命令事务复制;§12 重连快照对账 | 完整 CRDT/事务引擎(上层,见 §13) |
| **Unity Live Link** | 运行时 ↔ 编辑器实时数据流、低延迟源 | §8 订阅差分流 | C#/GC 运行时模型 |
| **Omniverse USD Live** | 分层(layer)增量、实时合流 | §9 命令作为可合流增量;§13 协同下沉 | USD/Nucleus 栈 |
| **gRPC / tonic** | schema 化服务、流式 RPC、拦截器(auth/日志) | §11 拦截器链;§6 流式 | protobuf 代码生成(改用反射驱动) |
| **FlatBuffers / Cap'n Proto** | 零拷贝读、无需反序列化即可寻址 | §6 二进制零拷贝帧(复用 reflect 二进制序列化) | 独立 IDL 工具链 |
| **QUIC / WebTransport** | 流多路复用、0-RTT 重连、内建 TLS、拥塞控制 | §4 QUIC 传输后端;§5 0-RTT 重连 | — |
| **Redis RESP / PG wire** | 紧凑帧 + 流水线(pipelining)+ 批 | §6 请求批处理与流水线 | — |
| **Figma / Zed** | 低延迟协同会话、CRDT 增量 | §13 为协同提供可靠有序通路 | 其 CRDT 具体实现 |

> **版权红线**:所有 Prism crate 不含任何 Unreal Engine / Unity / Omniverse 源码或衍生代码;仅借鉴**公开架构形态**与经典协议数值。协议本体为自研,过渡期在服务端复用保留的 `bevy_remote`(其为 MIT/Apache 开源)。

---

## 3. 分层架构总览

```
┌──────────────────────────────────────────────────────────────┐
│  编辑器业务层(E 系列:Inspector / 大纲 / Profiler / 内容浏览器)│  ← 不在本 crate
├──────────────────────────────────────────────────────────────┤
│  客户端 SDK(prism_editor_remote::client)                      │
│   · 类型安全 API(query/get/set/spawn/...)· 订阅句柄            │
│   · 增量应用器(net_delta apply)· 重连对账 · 喂 prism_ui_store  │
├───────────────── 语义层 Methods(§7)─────────────────────────┤
│   world.* / asset.* / command.* / pick.* / gizmo.* / diag.*    │
├───────────────── 订阅/增量(§8)+ 命令事务(§9)──────────────┤
│   DirtyMask 帧级 diff(复用 prism_reflect::net_delta)           │
├───────────────── 会话层 Session(§5)─────────────────────────┤
│   握手 · 能力协商 · 多路复用(sessionId)· 心跳 · 重连 · 取消    │
├───────────────── 消息层 Wire(§6)────────────────────────────┤
│   帧格式(len-prefixed)· 二进制/JSON 编码 · 批 · 流水线         │
├───────────────── 传输层 Transport(§4)───────────────────────┤
│   InProcess(零拷贝) · IPC(uds/pipe) · TCP · QUIC · WebSocket   │
├──────────────────────────────────────────────────────────────┤
│  服务端 SDK(prism_editor_remote::server)                      │
│   · ECS systems(调度点注入)· 命令执行器 · 订阅跟踪 · 拦截器链  │
├──────────────────────────────────────────────────────────────┤
│  Prism 运行时(bevy_ecs World / Schedule / Observer)           │  ← 被观测对象
└──────────────────────────────────────────────────────────────┘
```

分层原则:**依赖严格向下**;传输/会话/消息层**与语义无关**(可独立测试);语义层**与反射深耦合**但与传输无关;客户端/服务端 SDK 为对称镜像。

---

## 4. 传输层(Transport)

同一套 `Transport` trait,多后端;业务层经 feature 选择,**语义零改动**:

| 后端 | 场景 | 延迟/带宽 | feature |
|---|---|---|---|
| **InProcess** | 编辑器与运行时同进程(默认) | 零拷贝、无序列化(直接传 `Box<dyn Reflect>`/句柄) | 核心 |
| **IPC(UDS / Named Pipe)** | 同机独立进程(PIE 隔离、崩溃隔离) | 极低延迟、无网络栈 | `ipc` |
| **TCP** | 同机/局域网、调试友好、广兼容 | 低延迟 | `tcp` |
| **QUIC** | 远程机 / 高丢包 / 需加密复用 | 多路复用、0-RTT 重连、内建 TLS | `quic` |
| **WebSocket** | 浏览器端编辑器 / Web 工具 | 兼容 Web | `websocket` |

```rust
pub trait Transport: Send + Sync {
    type Conn: Connection;
    async fn accept(&self) -> Result<Self::Conn, TransportError>; // 服务端
    async fn connect(&self, addr: &Endpoint) -> Result<Self::Conn, TransportError>; // 客户端
}

pub trait Connection: Send {
    async fn send_frame(&mut self, frame: Frame<'_>) -> Result<(), TransportError>;
    async fn recv_frame(&mut self) -> Result<OwnedFrame, TransportError>;
    fn peer(&self) -> PeerInfo;          // 用于鉴权/审计
    fn supports_zero_copy(&self) -> bool; // InProcess=true
}
```

**关键设计**:
- **InProcess 零拷贝直通**:同进程时帧不序列化,直接以 `Arc<dyn Reflect>` / `Entity` 句柄过通道;只有跨进程才触发 §6 编码。这让「编辑器内嵌运行时」这一最常见拓扑**零协议开销**。
- **异步 IO 走 `prism_tasks`**:所有网络 IO 在 IO 任务池,work-stealing 调度;不阻塞 ECS 主循环。
- **拥塞/背压**:QUIC 原生流控;TCP/IPC 用有界通道 + §10 背压策略。

---

## 5. 会话层(Session)

### 5.1 握手与能力协商(借 LSP `initialize`)
连接建立后首帧为 `hello`,双方交换:
- 协议版本(semver)、支持的域集合、支持的编码(binary/json)、压缩能力、最大帧长。
- `prism_reflect` 的 **schema 指纹**(TypeRegistry 快照哈希):若客户端/服务端类型集不一致,进入**降级/迁移**(见 §12.1)。
- 认证凭据(§11)。

协商结果固定本连接的能力集;**不匹配项协商降级,不直接断开**。

### 5.2 多路复用(借 CDP sessionId)
单条物理连接承载多个逻辑**会话**(sessionId):例如同一编辑器开多个视口/多文档,各自独立订阅与命令流,互不阻塞。QUIC 下每会话映射独立 stream(队头阻塞隔离)。

### 5.3 心跳与存活
周期 `ping/pong`;超时判定断线,触发 §12 重连对账。心跳间隔与超时经能力协商。

### 5.4 取消(借 DAP `$/cancelRequest`)
长查询/长命令可携带 `requestId`;客户端发 `cancel(requestId)`,服务端在下个安全点中止并回 `Cancelled`。异步任务用 `prism_tasks` 的取消令牌。

---

## 6. 消息层(Wire)

### 6.1 帧格式
长度前缀帧(len-prefixed),头部紧凑:
```
[u32 len][u8 flags][u8 encoding][u16 domain][u32 request_id][payload...]
flags:    REQUEST | RESPONSE | EVENT | BATCH | COMPRESSED | END_STREAM
encoding: BINARY(reflect 紧凑) | JSON(调试/Web)
```

### 6.2 编码:二进制优先、JSON 可选
- **BINARY(默认跨进程)**:走 `prism_reflect` 二进制序列化器(定长数值整块编码、`varint` 变长、字段级增量)。**零拷贝读**路径复用 reflect 的访问器,寻址无需完整反序列化。
- **JSON(调试/Web/兼容)**:走 reflect JSON 后端,人类可读,对齐 BRP 的 JSON-RPC 形态,便于 `curl`/浏览器调试。编码由能力协商选定。

### 6.3 批处理与流水线(借 RESP pipelining / BRP Batch)
- **BATCH**:多请求打包一帧,服务端顺序执行、合并回包,摊薄往返与系统调用。
- **流水线**:客户端无需等响应即可连发;`request_id` 关联乱序响应。
- **帧合并**:一帧内多订阅的本帧增量合并下发(§8),减少小包。

### 6.4 压缩
大 payload(资产块、大查询结果)可选压缩(feature `compress`);小包不压(避免 CPU 浪费)。阈值可配。

---

## 7. 语义层(Methods):域 · 方法 · 事件

采用 CDP 式 `域.方法` 命名,**语义对齐并超集化 BRP**(`bevy_remote` 已有的 `world.*` 直接映射,过渡期可桥接):

### 7.1 `world.*`(实体/组件/资源——对齐 BRP)
| 方法 | 语义 | BRP 对应 |
|---|---|---|
| `world.query` | 按组件过滤查询实体 + 取值 | `world.query` |
| `world.get_components` | 取指定实体的组件(strict/非 strict) | `world.get_components` |
| `world.list_components` | 列实体拥有的组件类型 | `world.list_components` |
| `world.spawn` / `world.despawn` | 建/删实体 | `spawn_entity`/`despawn` |
| `world.insert` / `world.remove` / `world.mutate` | 组件增删改 | 同名 |
| `world.reparent` | 改父子层级 | `reparent_entities` |
| `world.get_resource` / `world.mutate_resource` / `list_resources` | 资源读写列 | 同名 |
| `world.trigger_event` / `world.write_message` | 触发事件/写消息 | 同名 |
| `world.registry_schema` | 取类型 schema(反射) | `registry.schema` |

> **差异点**:Prism 侧**所有写**(insert/remove/mutate/spawn/despawn/reparent)默认经 §9 命令事务,带撤销语义;BRP 的裸写仅在 `raw` 能力位开启时直通(调试用)。

### 7.2 `world.*+watch`(订阅——见 §8)
`get_components+watch`、`list_components+watch`、`query+watch`、`observe`(事件订阅)。对齐 BRP 的 `+watch` 家族,但增量走 §8 的 `DirtyMask`。

### 7.3 `command.*`(命令事务——见 §9)
`command.submit`(提交可撤销命令)、`command.undo` / `redo`、`command.transaction`(批量原子)、`command.subscribe`(协同:订阅他方命令流)。

### 7.4 `asset.*`(资产——挂接 `prism_asset`)
`asset.list` / `asset.get_meta` / `asset.import` / `asset.reimport` / `asset.dependencies` / `asset.watch`(热重载通知)/ `asset.stream_chunk`(大资产分块流)。长任务异步可取消。

### 7.5 `pick.*` / `gizmo.*`(视口旁路)
`pick.raycast`(视口射线 → 实体,GPU 拾取结果回传)、`gizmo.draw`(编辑器下发调试绘制)、`gizmo.manipulate`(操作柄变换 → 命令)。为编辑器 §7 视口服务,低延迟优先。

### 7.6 `diag.*`(诊断——挂接 `prism_diagnostic`)
`diag.frame_timings` / `diag.memory` / `diag.schedule_graph` / `diag.vislog+watch`。帧分析器/VisLog 数据流。

### 7.7 `session.*`(会话管理)
`session.hello` / `session.capabilities` / `session.ping` / `session.cancel` / `session.resync`(重连对账)。

---

## 8. 订阅与增量(Watch / Delta):成本 ∝ 变化量

**核心复用 `prism_reflect::net_delta`**(§24.5 已交付:`DirtyMask` 位集 + `varint` 字段增量编解码 + 幂等远端应用 + `ReplicationState` baseline):

- 客户端 `*+watch` 注册订阅,服务端建立 `ReplicationState` baseline。
- 每帧(或节流周期),服务端用 ECS 的**变更检测 tick**找出订阅集合里**本帧改动**的实体/组件,只对这些字段算 `DirtyMask` 增量。
- 下发**仅变化字段**;客户端 `apply` 幂等合并到本地镜像。
- **成本 ∝ 变化量**:订阅十万实体但本帧只动十个,只传十个的变化字段。

**订阅规模控制**:
- **可见性驱动**:编辑器只订阅当前视口/大纲可见的子集(虚拟化);滚动/展开时增量扩订。
- **采样率/LOD**:高频组件(Transform)可按编辑器需求降采样下发;Profiler 轨道按帧预算采样。
- **节流与合并**:同帧多次改动合并为一次增量;突发变更按 §10 背压合并。

**一致性**:每批增量带帧号/序号;客户端检测缺口 → 请求 `session.resync` 重取 baseline。

---

## 9. 命令事务通道(Command / Undo-Redo 协同)

远程写**不是裸 ECS 操作**,而是编辑器命令的传输化:

- `command.submit`:客户端提交一个**可序列化命令**(反射描述:目标、操作、前后值),服务端执行并记入**权威命令历史**。
- **撤销/重做跨进程一致**:`command.undo`/`redo` 操作权威历史;结果经 §8 增量广播给所有订阅会话。
- `command.transaction`:多命令**原子**提交(全成或全败),用于复合编辑(如批量变换)。
- **幂等 + 版本**:每命令带 `base_version`;若权威状态已前进(他人先改),服务端返回冲突,客户端重基或走 §13 协同消解。
- **本层职责边界**:本 crate 提供**可靠、有序、带版本的命令通路 + 权威历史存储接口**;**具体的命令定义、合并(coalesce)、冲突消解策略在编辑器命令内核与协同层**(见 §13),本层不内置 CRDT。

---

## 10. 性能工程:成本 ∝ 变化量的桥接落地

- **零拷贝(同进程)**:InProcess 传输不序列化,直接传反射句柄/`Arc`;最常见拓扑无协议税。
- **零拷贝读(跨进程)**:BINARY 帧复用 reflect 的访问器寻址,大查询结果**无需完整反序列化**即可按字段取用。
- **帧预算**:服务端每帧给桥一个**时间/字节预算**;超预算的增量切片到下帧续传(`END_STREAM` 标记),绝不让桥拖垮运行时帧率。
- **背压**:有界发送队列;满时按策略——`Coalesce`(合并同目标最新值)/ `Drop-oldest`(高频遥测)/ `Block`(命令等关键流,反压到客户端)。订阅声明自己的背压策略。
- **合批与流水线**:§6.3,摊薄往返与 syscall。
- **IO 卸载**:序列化/压缩/网络在 `prism_tasks` IO 池;`FrameArenas` 做每帧临时分配(bump,零 per-alloc 开销),帧末整体回收。
- **采样/LOD**:§8 的降采样;Profiler/VisLog 走独立低优先级流,不与命令流抢带宽。
- **可测性能**:桥自身暴露 `diag.*` 指标(每帧增量字节、订阅数、队列深度、重连次数),性能回归可在 CI 门控。

---

## 11. 安全

远程桥是**攻击面**,默认最小信任:

### 11.1 默认安全姿态
- **默认仅本机回环(loopback)**:TCP/QUIC 默认绑 `127.0.0.1`;对外监听需**显式配置 + 认证**,启动时 `prism_diagnostic` 显式告警(呼应安全意识:不静默创建无认证网络服务)。
- **InProcess/IPC 无网络暴露**:默认拓扑最安全。

### 11.2 认证与鉴权
- **认证**:握手携带 token(预共享密钥 / 一次性配对码 / mTLS 证书)。无 token 的远程连接拒绝。
- **鉴权(能力模型)**:每会话绑定**能力集**——只读 / 可写命令 / 可执行脚本函数 / 可资产导入 / 可触发事件。编辑器调试会话默认全能力(本机);远程/第三方工具按最小授权。
- **方法白名单**:高危方法(`world.trigger_event`、函数调用、`raw` 直写)默认关闭,按能力位开启。

### 11.3 加密
- QUIC 内建 TLS 1.3;WebSocket 走 WSS;TCP 可选 TLS。本机 IPC/InProcess 免加密(无网络面)。

### 11.4 不可信输入边界(复用 `prism_reflect` 安全层)
- 每个入站 payload 经 reflect 的**不可信反序列化安全边界**:**分配上限夹到剩余字节数**(防 OOM 炸弹)、**不 panic**(未知类型/字段/标签走 `DeserializeError`)、**未注册类型拒绝**。
- 帧长上限、批大小上限、嵌套深度上限,全部可配且有安全默认。

### 11.5 DoS 防护
- **速率限制**:每连接/每会话的请求数、字节数令牌桶。
- **订阅配额**:单会话最大订阅数/增量带宽上限。
- **慢连接保护**:发送超时 + 有界队列,慢客户端被背压或断开,不拖累服务端。
- **连接数上限**:防连接耗尽。

### 11.6 审计
- 认证失败、能力越权、速率触顶、异常断开全部进 `prism_diagnostic` 审计日志(带 `PeerInfo`),可用于安全回溯。
- 命令历史即审计轨:谁、何时、改了什么,可回放。

---

## 12. 稳定性

### 12.1 版本协商与 schema 漂移
- 协议 semver 协商;次版本差异走能力降级,主版本不兼容显式拒绝并提示。
- **类型 schema 漂移**:客户端/服务端 TypeRegistry 不一致时,逐类型用 reflect 的 **schema 版本化迁移链**升降级;不可迁移的字段隔离上报,不污染整条消息。

### 12.2 幂等
- 所有写命令幂等(带 `request_id` + `base_version`);重连重放不产生重复副作用。

### 12.3 重连与对账(resync)
- 断线后客户端重连(QUIC 0-RTT 优先);`session.resync` 重建 baseline:
  - **轻量**:若序号缺口小,补发缺失增量。
  - **全量**:缺口大或 baseline 失效,重取订阅集全量快照后转增量。
- 重连期间客户端 UI 不丢本地镜像,进入「正在对账」态而非清空(呼应失败可见 + 非阻塞)。

### 12.4 错误模型
- 结构化错误码(对齐 JSON-RPC error object + BRP `error_codes`):`InvalidParams` / `MethodNotFound` / `Unauthorized` / `RateLimited` / `VersionMismatch` / `Cancelled` / `Conflict` / `Internal`。
- **错误隔离**:单请求失败不影响同连接其他请求/订阅(非 strict 查询单组件缺失不炸整请求,对齐 BRP strict 语义)。

### 12.5 降级
- 传输不可用 → 回退更低依赖后端(QUIC 失败回退 TCP)。
- 高负载 → 自动降订阅采样率、暂停低优先级遥测流,保命令流通畅。

### 12.6 崩溃隔离
- IPC/独立进程拓扑下,运行时崩溃不拖垮编辑器 UI;编辑器检测桥断开 → 提示重启 PIE,保留未提交编辑。

---

## 13. 与协同编辑 / Multi-User 的关系

本 crate 是协同的**传输与权威通路地基**,不是协同引擎本身:

- **提供**:可靠、有序、带版本的命令流;多会话广播;权威命令历史存储接口;重连对账。
- **不提供(上层职责)**:CRDT/OT 冲突消解、意图保持、离线合并、锁/权限粒度策略。这些在编辑器协同层(编辑器文档 §13)实现,**消费**本层的命令通路。
- **仲裁模型**:默认**服务端权威**(运行时 World 为单一真相源),对齐 Unreal Concert 的事务复制形态;对等协同(P2P CRDT)可在上层基于本层多会话广播构建。

---

## 14. 服务端集成(ECS)

- **`RemoteServerPlugin`**:向 `App` 注入——传输监听(IO 任务)、入站请求队列、命令执行系统、订阅跟踪系统、出站增量系统。
- **调度点**:
  - 入站命令在 World 可变访问的**专用 Set**执行(排他,保证一致性)。
  - 增量采集系统在帧末读 ECS 变更 tick,产出 `DirtyMask`。
  - 出站下发在 IO 任务,不占 ECS 排他窗口。
- **拦截器链(借 gRPC interceptor)**:认证 → 鉴权 → 速率 → 审计 → 分发,可插拔。
- **过渡期桥接**:服务端可**复用保留的 `bevy_remote`** 的方法实现(`process_remote_*`),`prism_editor_remote` 做协议/传输/安全/增量的 prism 外壳,逐步内化。

## 15. 客户端集成

- **`RemoteClient`**:连接管理 + 请求/订阅 API,返回类型安全句柄(非裸 JSON)。
- **订阅 → `prism_ui_store`**:增量 `apply` 后更新 store 切片,UI 经 selector 精确重渲(成本 ∝ 变化量)。
- **异步 → `prism_ui_async`**:查询/命令作为 `Resource`,配 Suspense/ErrorBoundary;加载/错误/重连态天然可渲。
- **Inspector 直连**:反射视图 + 命令提交 + 订阅刷新三合一,驱动 §4 编辑器 Inspector。

---

## 16. Crate 结构 / 模块 / feature / 路线图

### 16.1 模块
```
prism_editor_remote/
├── protocol/      # 帧格式、编码、错误码、能力协商(与传输无关)
├── transport/     # Transport trait + 后端(inprocess/ipc/tcp/quic/websocket)
├── session/       # 握手、多路复用、心跳、取消、重连
├── codec/         # 绑定 prism_reflect 的 binary/json 编解码 + 安全边界
├── delta/         # 订阅跟踪 + net_delta 增量采集/应用
├── methods/       # world.*/asset.*/command.*/pick.*/gizmo.*/diag.* 语义
├── command/       # 命令事务通路 + 权威历史接口
├── security/      # 认证/鉴权/能力/速率/审计 拦截器
├── server/        # ECS 插件、系统、调度点、bevy_remote 桥
├── client/        # 客户端 SDK、store/async 集成
└── diagnostics/   # 自观测指标
```

### 16.2 features
`inprocess`(默认)/ `ipc` / `tcp` / `quic` / `websocket` / `json`(调试)/ `compress` / `tls` / `bevy_remote_bridge`(过渡)/ `std`。核心 `no_std + alloc` 友好(协议/编解码),传输/网络在 `std`。

### 16.3 路线图(M0–M6)
| 里程碑 | 内容 | 依赖 |
|---|---|---|
| **M0** | protocol + codec(复用 reflect)+ InProcess 传输 + world.* 读 | prism_reflect ✅ |
| **M1** | 订阅/增量(delta,复用 net_delta)+ world.*+watch | M0 |
| **M2** | 命令事务通路 + undo/redo + 权威历史接口 | 编辑器命令内核 |
| **M3** | IPC + TCP 传输 + 会话层(握手/心跳/取消/重连)+ 安全(认证/鉴权/速率/审计) | M0 |
| **M4** | asset.*/diag.* 挂接 + 客户端 store/async 集成 | prism_asset / prism_diagnostic |
| **M5** | pick.*/gizmo.* 视口旁路(依赖 S13 拾取 + Gizmo pkg) | 编辑器 §7 |
| **M6** | QUIC/WebSocket + 压缩 + 协同多会话广播通路 | M3 / 协同层 |

---

## 17. 公共 API 草图(示意,非最终)

```rust
// 客户端
let client = RemoteClient::connect(Endpoint::InProcess(world_handle)).await?;

// 读:类型安全查询
let rows = client.world()
    .query::<(&Transform, &Name)>()
    .with::<Visible>()
    .fetch().await?;

// 订阅:增量驱动 UI(成本 ∝ 变化量)
let mut sub = client.world()
    .watch::<Transform>(entities)
    .into_store(store.slice("transforms")); // apply → prism_ui_store

// 写:走命令事务(可撤销)
client.command()
    .mutate(entity, Transform::from_xyz(1.0, 0.0, 0.0))
    .submit().await?;          // 进权威历史
client.command().undo().await?;

// 服务端
app.add_plugins(
    RemoteServerPlugin::new()
        .transport(InProcess::shared())          // 默认零拷贝
        .transport(Tcp::loopback(15703))         // 可选,默认本机
        .security(Security::require_token(token).capabilities(Caps::EDITOR))
        .rate_limit(RateLimit::per_session(10_000))
);
```

---

## 18. 风险与取舍

1. **协议与 reflect 耦合深**:schema 漂移是主要复杂度来源;靠版本化迁移链 + 握手指纹兜底,但跨版本协同需严格 CI 覆盖。
2. **零拷贝 vs 安全**:同进程零拷贝绕过序列化,也绕过部分不可信边界——仅在**同信任域**(编辑器内嵌运行时)启用,跨进程一律走安全编解码。
3. **命令权威历史的存储成本**:长编辑会话历史增长;需配额 + 快照压缩(复用 `prism_ecs` 的 `SnapshotDelta` 思路)。
4. **BRP 桥接过渡期双实现**:`bevy_remote` 与 `prism_editor_remote` 并存期需明确哪条通路权威,避免双写;以 refactor plan 的 pkg 优先定调逐步内化。
5. **远程暴露的安全责任**:默认回环 + 显式告警,但用户配置对外监听仍可能误配;文档与运行时告警双重提示。
6. **采样/背压的可感知性**:降采样可能让编辑器显示滞后;需在 UI 明示「遥测降级中」,不伪装实时。

---

## 19. 诚实边界(SHIPPED vs PLANNED)

- ✅ **已存在底座(非本 crate)**:`bevy_remote` 的 BRP(JSON-RPC 2.0 / `world.*` + `+watch` / `observe` / HTTP:15702 / strict 语义 / `error_codes`)、`prism_reflect`(TypeRegistry / StableTypeId / ReflectSerialize / 反射驱动序列化 / schema 版本化迁移 / `net_delta` DirtyMask 字段增量 / 不可信反序列化安全边界:分配上限夹到剩余字节、不 panic、未注册类型拒绝 / property_bridge)、`prism_tasks`(work-stealing + FrameArenas + join)、`prism_ui_store` / `prism_ui_async`。
- ⬜ **全部 PLANNED(本 crate 尚未创建)**:`prism_editor_remote` 的 protocol / transport(除概念外无实现)/ session / codec 绑定 / delta 采集 / methods / command 通路 / security 拦截器 / server 插件 / client SDK。本文档所有 `rust` 代码块与方法表均为**设计意图**,不代表已实现。
- ⬜ **依赖未落地项**:`command.*` 需编辑器命令内核;`pick.*`/`gizmo.*` 需 S13 拾取迁移 + Gizmo pkg;`asset.*` 的流/热重载需 `prism_asset` 后续里程碑。
- 诚实约束:协议本体为自研;过渡期在服务端**复用**开源 `bevy_remote` 的方法实现并逐步内化,不抄任何 Unreal/Unity/Omniverse 源码,仅借公开架构形态。

---

## 20. 术语表

- **BRP**:Bevy Remote Protocol,`bevy_remote` 提供的 JSON-RPC 2.0 远程协议,本设计的参照与过渡底座。
- **DirtyMask**:`prism_reflect::net_delta` 的字段级脏位集,增量下发的核心。
- **InProcess 零拷贝**:编辑器与运行时同进程时不序列化、直传反射句柄的最优拓扑。
- **能力模型(Capabilities)**:每会话绑定的权限集(只读/可写/可脚本/可资产),鉴权基础。
- **resync 对账**:断线重连后重建订阅 baseline 的一致性恢复流程。
- **权威历史**:服务端持有的命令执行序列,撤销/重做/审计/协同仲裁的单一真相源。
- **拦截器链**:认证→鉴权→速率→审计→分发的可插拔服务端中间件(借 gRPC)。

---

> 本文为 `prism_editor_remote` 的 v1 设计规格。实现须遵循仓库 `prism_bevy_refactor_plan_zh.md` 的「pkg 优先、`prism_*` 为唯一真相源」方向,与 `prism_editor_framework_design_zh.md`(§4/§13 BRP 前置待建)、`prism_reflect_design_zh.md`(序列化/安全/net_delta)、`prism_tasks_design_zh.md`(异步/背压)、`prism_asset_design_zh.md`(资产通路)协同演进。所有 Prism crate 不含任何 Unreal Engine / Unity / Omniverse 源码或衍生代码;仅借鉴公开架构形态与经典协议数值。
