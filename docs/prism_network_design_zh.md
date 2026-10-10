# Prism Network 顶级次世代 AAA 级网络/多人复制系统设计方案

> 面向 Prism（脱离 Bevy 后的独立引擎）的 **网络/多人** 内核设计：**传输层（可靠 UDP / QUIC 多通道 + 拥塞控制 + 加密）+ 服务器权威仿真 + 事件流复制骨架（override delta 事件溯源）+ 瞬时状态快照 delta 双通道 + AOI 兴趣管理 + 客户端预测与权威和解（rollback）+ 延迟补偿 + 服务器网格分片与权威移交 + 确定性回放对账 + 权威式反作弊**。它不是 world/scene 的职责，而是**架在 `prism_scene`（内容真相）/ `prism_world`（空间组织）/ persist（事务日志）之上的横切复制层**，独立成 crate 与文档，与仓库「一子系统一文档」惯例一致。
> **架构立场**：网络不另立「网络专属真相源」。权威持久状态的复制单元**复用 persist 的 override delta 事务语义**（同一套序列化、同一套确定性合并）；网络只是把这条 delta 事件流**权威化、定序、按兴趣裁剪地跨机广播**，再叠加一条用于高频瞬时仿真态（位移/物理/动画相位）的状态快照通道。没有第二份真相。
> **模块化与可替换**：网络不是铁板一块。传输、复制策略、AOI 策略、预测策略、权威后端、反作弊都是 provider（trait + 默认实现），可整体或局部替换为自带/第三方实现；唯一不可替换的是「服务器权威 + 复用 persist delta + 确定性对账」这组不变量——provider 只能插进这套骨架、不能违反它（§4）。反作弊作为独立可替换策略模块，单独成 crate 与文档（`prism_anticheat`），本文只保留其接入网络权威回路的接缝（§14）。其中「权威事件流」这半的 durable 存储/编排后端经 `EventLogBackend` provider 暴露，可选用成熟事件网格 `uwu_event_mesh`（仅服务端，§5.1），默认实现自带、换后端是 opt-in。
> 借形态不抄码。借鉴：
> - **ECS 快照复制与预测**：Unity Netcode for Entities（ghost snapshot + 预测 + 插值 + 优先级）、UE Replication Graph / Iris（relevancy / dormancy / NetGUID / FastArray delta）
> - **延迟隐藏**：Valve Source（delta 压缩快照 + entity baseline + 客户端插值 + lag compensation 回溯命中）、Overwatch（command frame + 预测 + 回滚和解）、Rocket League（fixed-tick 确定性回滚）
> - **传输**：Valve GameNetworkingSockets / QUIC / WebTransport / ENet（可靠-不可靠混合通道、乱序交付、拥塞控制、0-RTT 重连、DTLS 加密）
> - **大规模权威**：Star Citizen Server Meshing（分片权威 + 无缝移交）、EVE（兴趣域 + 时间膨胀降级）、RTS 确定性 lockstep（AoE/星际，海量单位靠输入同步）
> - **事件流/一致性**：事件溯源（Event Sourcing / CQRS）、append-only 日志（Kafka 形态）、CRDT 冲突合并、Lamport/向量时钟定序
> - **反作弊形态**：服务器唯一权威、输入与运动约束校验、速率限制、重放序号/nonce 防重放、确定性仿真交叉对账（审计用事件日志重放）
> 本文为纯经典网络工程 + 数据结构 + 确定性算法路线（可靠 UDP ARQ、滑动窗口、delta/位域压缩、定点量化、Morton AOI 裁剪、Lamport 定序、CRDT 合并、回滚重放），**不含任何 AI/ML 内容**（反作弊为确定性权威校验与统计阈值，非机器学习），不含任何 Unreal/Unity/Valve/其他引擎源码或衍生代码。

- 版本： v0.1（设计阶段，未进入编码，无旧 API 需保留；内容源自 `prism_world_system_design_zh.md` v3.1 剥离的「网络复制 / 权威 / AOI」与「服务器网格分片」，按「一子系统一文档」惯例独立重构深化；去 Bevy、多 crate、架在 `prism_scene` / persist 契约之上；v0.2：反作弊独立为可替换 provider 并剥离至 `prism_anticheat`（本文 §14 改为接缝），确立「传输 / 复制 / AOI / 预测 / 权威 / 反作弊皆为可替换 provider」的模块化立场；v0.3：把权威事件流的底层存储抽象为具名 provider，引入可选服务端后端 `uwu_event_mesh`（权威事件流 / 回放 / 编排，`feature evmesh`，仅 `std`/`server`，不进 core/client，§5.1 / §13 / §18）；v0.4：把 RPC 提升为与事件流 / 快照并列的第三类复制原语（命令式控制面，§7.4），统一调用形态 / 定向 / 幂等去重 / 默认安全 / provider 接缝，并明确「RPC 不另立真相、权威改动仍以 override delta 固化」；v0.5：深化延迟与体验工程——时钟同步 + 命令帧提前量（§9.1）、自适应插值 / 抖动缓冲 / 可视平滑（§10.5）、motion-to-photon 延迟预算分解（§11.1）、内存零分配热路径与向量化量化（§15）、从单机到联机不分叉的开发者工作流（§17.1）；并复核 `uwu_event_mesh` 2.0.0-dev.1 的 durable log / at-least-once / timeline / Process-RPC 立场仍准确、仅用于服务端权威事件流与编排）
- 适用引擎： Prism（后 Bevy 时代，独立运行时）
- 关键依赖： `prism_scene`（override delta / `PersistentId` / spawn·despawn·materialize 契约 / 网络复制对称 §16.4）、`prism_world`（sector 树与空间索引 §5、HLOD 确定性簇 id §10、需求源 §7.1、persist 事务接口 §11）、`prism_ecs`（复制组件的列读写 / `serialize` / `determinism`）、`prism_time`（tick / 固定步长 / 时间线 §4D）、`prism_tasks`（并行序列化 / 分片编排）、`prism_math`（定点量化 / 空间编码）、`prism_reflect`（复制字段路径 / 增量脏标）、`prism_diagnostic`（网络计数器 / trace / 回放）、`prism_anticheat`（反作弊 provider 接缝，§14，另文）；可选外部后端 `uwu_event_mesh`（服务端权威事件流 / 回放 / 编排的可替换 provider 后端，`feature evmesh`，仅 `std`/`server`，§5.1 / §13 / §18）
- 层级定位： 横切复制层 L6.5；下接 `prism_scene` / persist / `prism_world` / `prism_ecs` / `prism_time`，向上为 `prism_gameplay` 提供 RPC / 复制组件 / 权威事件；与 `prism_world` 以「接缝」解耦（world §12），不侵入其 stream/residency 热路径
- 明确约束： 核心 `no_std + alloc`（仅定序 / delta / CRDT 纯逻辑）；`std` / `transport_udp` / `transport_quic` / `server` / `client` / `listen_server` / `prediction` / `rollback` / `lag_comp` / `server_mesh` / `replay` / `crypto` / `anticheat` 为 feature；工作区 `forbid(unsafe_code)`；**不依赖任何 `bevy_*` crate**；吞吐 / 延迟 / 带宽 / 并发连接数均为 design target（本机无多端实测环境，如实标注），可验证部分限于定序 / CRDT 合并 / delta 编解码 / 回滚重放的 CPU 纯函数单测
- 架构立场： **scene 是单一逻辑真相，persist 的 override delta 是唯一持久变更语义，网络复用它作权威事件流、不另立网络专属真相源**；瞬时仿真态走独立快照通道、本身不入存档

---

## 目录

1. 设计哲学与定位（为什么网络独立成 crate）
2. 参考产品取舍（AAA 网络形态）
3. 档位化（capability / feature / 拓扑）
4. 多 crate 分层架构
5. 事件流 vs 状态复制：双骨架分工（回答「事件流是否更好」）
6. 传输层（可靠 UDP / QUIC / 多通道 / 拥塞 / 加密）
7. 复制模型：权威 delta 事件流 + 瞬时状态快照 delta + RPC 控制面
8. AOI 兴趣管理（空间裁剪 / 簇分级 / relevancy / dormancy）
9. 服务器权威与仿真定序（tick / command frame / Lamport）
10. 客户端预测与权威和解（rollback / reconciliation）
11. 延迟补偿（lag compensation 回溯命中判定）
12. 服务器网格分片与权威移交（`server_mesh`，借 Star Citizen 形态）
13. 一致性模型（确定性回放 / CRDT / 事件日志对账）
14. 反作弊接缝（权威回路钩子，策略归 `prism_anticheat`）
15. 带宽与性能工程（delta 压缩 / 量化 / 优先级 / 预算调速）
16. 可扩展性接缝（大厅 / 匹配 / 中继 / 重连）
17. 易用性与 API 人体工学（声明式复制 / RPC / 事件 DSL / prelude）
18. 可观测性与诊断（网络剖析 / 回放 / 丢包注入）
19. 与各子系统的接缝（scene / persist / world / ecs / time / gameplay）
20. crate 分层与模块布局
21. 确定性与可复现
22. 契约、不变量与版本化
23. 路线图、基准即规格与诚实边界
24. 一句话总结

---

## 1. 设计哲学与定位（为什么网络独立成 crate）

多人网络要回答的问题是：**N 台机器上的玩家，如何对「同一个只有一份真相的世界」做出改动、彼此看见、且在延迟与丢包下仍感觉即时、公平、不可作弊？** 很多引擎把网络缝进世界/场景系统内部（复制逻辑散落在实体、组件、关卡里），结果是「单机构建也背着网络税、网络协议与内容真相互相渗透、权威与预测难以单独验证」。

Prism 的选择是把网络**从世界系统剥离、独立成 crate 与文档**（本文），只通过**接缝**消费上游（`prism_world` §12）：

- **网络不是真相源**：权威的持久改动就是 persist 的 override delta（scene §15、world §11）。网络不发明新的「网络状态格式」，它把这条已有的 delta 事件流**权威化 + 定序 + 按兴趣裁剪 + 跨机广播**。存档与多人是**同一套 delta 的两个消费者**。
- **瞬时态与持久态分离**：位移、速度、动画相位、瞬时特效这类高频易变、不需要进存档的状态，走**独立的状态快照通道**，用插值/预测隐藏延迟，本身不污染事务日志。
- **零侵入热路径**：无网络的单机构建里，`network` 系列 feature 编译期移除，world 的 stream/residency 不含一行网络代码。
- **可单独验证**：定序、CRDT 合并、delta 编解码、回滚重放都是 CPU 纯函数，可脱离真实网络做确定性单测（真实吞吐/延迟只能标 design target）。

一句话定位：**网络 = 架在 scene/persist/world 之上的「权威事件流复制 + 瞬时状态同步 + 延迟隐藏 + 反作弊」横切层，不碰内容真相、不侵入流送热路径。**

## 2. 参考产品取舍（AAA 网络形态）

| 维度 | 借谁的形态 | Prism 取舍 |
|---|---|---|
| ECS 快照复制 | Unity Netcode for Entities（ghost + 预测 + 插值 + 优先级） | 复制单元按 archetype 列块序列化（§7），与 `prism_ecs` 存储同构 |
| relevancy / 休眠 | UE Replication Graph / Iris（NetGUID、dormancy、FastArray delta） | AOI 空间裁剪（§8）+ 静止对象休眠、只在变更时唤醒 |
| 延迟隐藏 | Valve Source（delta 快照 + baseline + 客户端插值）、Overwatch（command frame + 回滚和解） | 瞬时态快照插值 + 可预测输入回滚和解（§10） |
| 时钟 / tick 对齐 | QuakeWorld / Overwatch（client-side clock + command frame 提前量） | 时钟同步 + 命令帧提前量，把输入延迟压到接近单机（§9.1） |
| 命中判定公平 | Valve lag compensation（服务器回溯到客户端所见时刻判定） | 服务器按历史快照环回溯（§11），回溯窗口有界防滥用 |
| 抖动吸收 | VoIP / Source 抖动缓冲（jitter buffer + 自适应插值延迟） | 快照回放缓冲 + 自适应插值延迟 + error smoothing，吸收抖动不橡皮筋（§10.5） |
| 确定性大规模 | RTS lockstep（输入同步，海量单位）、Rocket League（fixed-tick 回滚） | 可选确定性仿真（§13、§21），输入同步作为低带宽档位 |
| 大世界权威 | Star Citizen Server Meshing、EVE 兴趣域 + 时间膨胀 | sector 子树分片 + 无缝权威移交（§12），过载时降级而非崩溃 |
| 传输 | GameNetworkingSockets / QUIC / ENet（可靠-不可靠混合、乱序、拥塞、0-RTT） | 自定义可靠 UDP 为主、QUIC 为可选 feature（§6） |
| 一致性 | 事件溯源 / CRDT / Lamport（append-only、冲突合并、确定性定序） | 复用 persist 事件日志语义（§5、§13） |
| 反作弊 | 服务器权威 + 校验 + 签名 + 重放防护（不引入内核态/ML） | 独立为可替换 provider（§14 接缝 → `prism_anticheat`），默认权威式确定性校验 + 统计阈值，**无 ML** |

**不照搬**：不引入内核态反作弊（越权且非引擎职责）；不做纯客户端权威的「信任客户端」架构；不把网络状态立为独立于 persist 的真相；不引入任何 AI/ML 行为检测。

## 3. 档位化（capability / feature / 拓扑）

网络能力全部 feature 门控，默认单机零开销；拓扑按部署选择：

```text
拓扑档位:
  single          纯单机,network 全移除(默认,零网络税)
  listen_server   主机兼客户端(P2P 房主权威,小规模合作)
  dedicated       专用服务器权威(竞技/大规模)
  server_mesh     N 台专用服务器分片(行星尺度,§12)

能力 feature(可叠加):
  transport_udp   可靠 UDP(默认传输)
  transport_quic  QUIC/WebTransport(NAT 友好、0-RTT、浏览器端)
  prediction      客户端预测(移动/交互本地先行)
  rollback        回滚和解(确定性 fixed-tick,竞技)
  lag_comp        延迟补偿命中回溯
  crypto          传输加密 + 握手签名
  anticheat       启用反作弊 provider 接缝(默认实现归 prism_anticheat;可换第三方/自定义/no-op)
  replay          事件流录制与确定性回放(§13、§18 对账)
  evmesh          uwu_event_mesh 作权威事件流/回放/编排 durable 后端(仅服务端,§5.1)
```

每档只增量启用所需子系统；`single` 构建里以上 feature 全部编译期消失。质量/带宽档位（快照频率、量化位宽、AOI 半径、插值延迟）运行时可调（§15 预算调速）。

此外，传输、复制策略、AOI 策略、预测策略、权威后端、事件流后端、反作弊均以 **provider（trait + 默认实现）** 暴露，可整体或局部替换为自带/第三方实现（§4）；替换 provider 不得违反核心不变量（§22）。

## 4. 多 crate 分层架构与可替换后端（provider 模型）

```text
prism_net_core      no_std 核: 定序(Lamport/序号)、delta 编解码、CRDT 合并、
                    事件流抽象、量化/位域打包 —— 纯函数,可单测
prism_net_transport 传输: 可靠 UDP ARQ / QUIC 适配 / 多通道 / 拥塞 / 加密握手
prism_net_replicate 复制: 权威 delta 事件流 + 瞬时快照通道 + baseline + 优先级
prism_net_aoi       兴趣管理: 空间裁剪(消费 world sector/簇)、relevancy、dormancy
prism_net_predict   预测与和解: 输入缓冲、本地预测、回滚重放、延迟补偿
prism_net_authority 服务器权威: tick 定序、command frame、提议校验、仿真对账
prism_net_mesh      服务器网格: 分片划分、权威移交、重叠带、容灾(feature server_mesh)
prism_net_acheat_api 反作弊接缝: AntiCheatProvider trait + 钩子点 + 默认 no-op(实现归 prism_anticheat)
prism_net_replay    录制/回放: 事件流落盘、确定性重放、对账(feature replay)
prism_net_evmesh    uwu_event_mesh 适配: 服务端权威事件流/回放/编排后端(feature evmesh,仅 std/server)
prism_net_diag      诊断: 计数器、带宽剖析、丢包/延迟注入、网络可视化
```

依赖方向单向向下，`prism_net_core` 不依赖传输（可在无网络环境单测）。`server`/`client` 两套入口按 feature 裁剪：客户端不含权威仲裁，服务器不含渲染侧插值。

### 可替换后端（provider 模型）

网络每一层都以 **provider = trait + 默认实现** 暴露，studio 可整体或局部替换：

| 层 | provider trait | 默认实现 | 可替换为 |
|---|---|---|---|
| 传输 | `Transport` | 可靠 UDP（§6） | QUIC/WebTransport、平台 socket、自带传输 |
| 复制策略 | `ReplicationStrategy` | delta 事件流 + 快照 delta（§7） | 自定义量化/压缩/通道划分 |
| 兴趣管理 | `AoiPolicy` | sector/簇空间裁剪（§8） | 格子/房间/自定义相关性 |
| 预测和解 | `PredictionStrategy` | 回滚重放（§10） | 仅插值纠正、输入同步 lockstep |
| 权威后端 | `AuthorityBackend` | 单权威（§9） | server_mesh 分片（§12）、第三方专用服务器 |
| 事件流后端 | `EventLogBackend` | 内置 append-only 日志（§5、§13） | `uwu_event_mesh`（durable log / cursor / DLQ / timeline，仅服务端，§5.1）、PostgreSQL / RocksDB / Kafka 适配 |
| 反作弊 | `AntiCheatProvider` | 内置确定性校验（归 `prism_anticheat`，§14） | 第三方商用、自定义、no-op（可信局域网） |

**可替换的边界**：provider 只能插进「服务器权威 + 复用 persist delta + 确定性对账」这组不变量（§22）构成的骨架、**不能违反它**——例如反作弊 provider 可换，但「客户端永不被信任、结果只由权威仿真产生」不可换；传输 provider 可换，但「必达事件流与可丢快照分通道」的语义契约不可换。默认 provider 全部零配置可跑通，替换是 opt-in。

## 5. 事件流 vs 状态复制：双骨架分工（回答「事件流是否更好」）

**直接回答你的问题：事件流（event sourcing / append-only 日志）用对地方确实更好，但不能一刀切地「全用事件流」。** Prism 采用**双骨架**：持久权威改动走事件流（天然契合、强烈推荐），高频瞬时仿真态走状态快照 delta（事件流在此会劣化体验）。分工如下：

| 维度 | 事件流（权威 delta 事件溯源） | 状态快照 delta |
|---|---|---|
| 承载 | 持久、离散、因果重要的改动：破坏、建造、拾取、任务、override 编辑、门/开关、库存 | 高频连续的瞬时态：位移、速度、朝向、动画相位、瞬时特效 |
| 单元 | 复用 persist 的 override delta（scene §15、world §11），append-only，可定序可重放 | 实体组件的当前值快照，按 baseline 做 delta 压缩，可丢旧帧 |
| 可靠性 | 必达、有序、幂等（丢了要补发，顺序错会改变因果） | 可丢（下一帧覆盖），不可靠通道，最新即正确 |
| 一致性 | 全局 Lamport 定序 + CRDT 合并（§13），确定性，入存档 | 服务器权威值直接覆盖客户端预测，不入存档 |
| 延迟策略 | 可容忍到达延迟（因果正确优先） | 插值/外推 + 客户端预测隐藏延迟（即时感优先） |
| 为什么这样 | 事件流让「网络 = 存档 = 时间线」三者同源，审计/回放/反作弊对账全部免费复用 | 对位移用事件流会累积重放成本、放大抖动、必达要求拖垮延迟，体验更差 |

**事件流带来的红利（这是「更好」的实质）**：

- **三位一体同源**：存档（§11 事务日志）、多人复制（本文）、时间轴回放（world §18）消费**同一条 append-only override delta 流**，不写三套序列化。
- **天然审计与反作弊对账**：权威事件流可确定性重放（§13、§18），服务器可随时「重放这段事件验证结果是否自洽」，作弊表现为「客户端提议的事件无法在权威仿真中复现」（§14）。
- **断线续传与补发**：客户端持最后确认的事件序号，重连时服务器从该序号增量补发（§16），无需重传全量世界。
- **冲突合并有理论**：离散事件 + CRDT/Lamport 有成熟的最终一致性保证（§13），而连续状态没有「合并」语义、只有「最新覆盖」。

**边界（为什么不全用事件流）**：对每秒数十次更新的位移若也做必达有序事件，会导致重放成本累积、队头阻塞、带宽爆炸，且这些状态根本不需要进存档。所以瞬时态走**可丢的快照通道**，只有当瞬时态「固化」为持久事实（如角色死亡落点、建筑最终形变）时，才由服务器生成一条权威 delta 事件写入事件流。

**结论**：事件流是**权威持久层的骨架**（强烈采用，且 Prism 架构已天然就位）；状态快照是**瞬时表现层的骨架**（延迟隐藏必需）。二者在服务器权威处汇合——事件流决定「世界真相如何改变」，快照通道决定「玩家此刻看到什么」。（另有一类命令式交互——登录 / 匹配 / 交易 / 技能确认等一次性、常带返回值的调用——既不属事件流也不属快照，由第三类原语 **RPC** 承载，见 §7.4；RPC 改变权威持久态时仍产出 override delta 事件、不另立真相。）

### 5.1 事件流后端 provider：`uwu_event_mesh`（可选服务端）

权威持久层的事件流骨架（§5）需要一套 append-only 日志 + 定序 + 消费游标 + 补发 的底层存储。Prism 自带轻量内置实现（`prism_net_core` 纯逻辑 + 落盘），同时把这一层抽象为 **`EventLogBackend` provider**，可整体替换为成熟的事件网格 `uwu_event_mesh`（仅服务端、`feature evmesh`、重 `std`/`tokio`，**不进 `prism_net_core` 与客户端**）。

契合点（为什么它适合「权威事件流」这半）：

- **语义同构**：`uwu_event_mesh` 是 durable log + 分区内有序 + 分区间并行 + 消费游标 + at-least-once（显式 ack/nack、超时重投、DLQ）+ 幂等键，正是 §5 权威 delta 事件流要的「必达、有序、可补发、幂等」。
- **幂等天然满足**：override delta 的确定性合并（§13、persist）本就幂等，恰好满足其 at-least-once 的去重前提——重复投递不改变最终真相。
- **timeline 即回放/对账**：其时间线分支、差异比较、投影重放、快照恢复，可直接服务 §18 的事件流录制/回放与 §13/§14 的确定性对账审计。
- **编排即分片**：其租约 epoch fencing、恢复握手、消费游标续传，映射到 §12 的权威移交与 §16 的断线续传。

明确边界（为什么只用在这半）：

- **只服务端、只持久层**：仅承载权威 delta 事件流与服务端编排；瞬时态快照通道（§7.2）、客户端预测/回滚/延迟补偿（§10/§11）**绝不走它**——那些要 unreliable UDP、可丢弃、sub-tick，而 at-least-once + durable append 语义相反（会可靠重投过期帧，劣化体验）。
- **不进 core/client/`no_std`**：`uwu_event_mesh` 重 `std`/`tokio`，只允许出现在 `prism_net_evmesh` 适配 crate（§20），由 `server`/`evmesh` feature 门控；客户端与 `single` 构建里编译期不存在。
- **不另立真相**：它只是「权威事件流」的一个存储/编排后端，承载的仍是复用 persist 的 override delta（§5），不发明新的网络状态格式。
- **opt-in**：默认用内置实现、零外部依赖即可跑通；换 `uwu_event_mesh` 是 studio 需要 durable / HA / timeline 时的显式选择，且其仍处 `2.0.0-dev`（§23 诚实边界）。

跨机实时传输仍由 `prism_net_transport` 的可靠 UDP/QUIC 承担（§6）；`uwu_event_mesh` 负责的是服务端内部/节点间的权威事件流持久化、定序、补发与编排——两者分工、不重叠。

## 6. 传输层（可靠 UDP / QUIC / 多通道 / 拥塞 / 加密）

游戏网络不能直接用 TCP（队头阻塞会把一个丢包放大成全通道卡顿）。Prism 传输层以**可靠 UDP** 为默认，QUIC 为可选 feature：

```text
多通道(channel)语义,一条物理连接复用:
  ch_events_reliable   权威 delta 事件流 —— 必达、有序、幂等(ARQ 滑动窗口)
  ch_state_unreliable  瞬时态快照 —— 可丢、乱序容忍(最新序号即正确)
  ch_input_unreliable  客户端输入 —— 可丢(下一帧补),带冗余最近 N 帧抗丢
  ch_rpc_reliable      RPC/控制面 —— 必达、有序
  ch_bulk              大块传输(初始世界基线、资产句柄) —— 可靠、低优先、可抢占
```

- **可靠 UDP ARQ**：选择性确认（SACK）+ 快速重传 + RTT 自适应 RTO；可靠通道独立滑动窗口，互不队头阻塞。
- **拥塞控制**：BBR 形态（带宽-RTT 估计）优先于丢包型，避免 bufferbloat；拥塞时先降瞬时态频率（§15）而非丢事件。
- **QUIC/WebTransport**（feature `transport_quic`）：复用成熟的 0-RTT 重连、连接迁移（切网不断线）、内建 TLS，面向浏览器端与 NAT 复杂环境；语义上把上述 channel 映射到 QUIC stream（可靠）与 datagram（不可靠）。
- **NAT 穿透**：ICE/打洞 + 中继回退（§16），`listen_server` 房主优先直连、失败走中继。
- **加密**（feature `crypto`）：DTLS/QUIC-TLS 握手 + 会话密钥；握手阶段做身份签名（§14 反作弊前置）。
- **MTU 与分片**：探测路径 MTU，事件流大 delta 在应用层分片重组，避免 IP 层分片丢失放大。
- **RPC / 控制面通道**：`ch_rpc_reliable` 承载请求-响应与控制信令，per-RPC 可按需降级为 unordered 或 unreliable 单发，大载荷借 `ch_bulk` 应用层分片；完整的调用形态 / 定向 / 幂等 / 安全模型见 §7.4。

> 传输吞吐/RTT/丢包恢复时间均为 design target，本机无多端环境；可验证部分限 ARQ 窗口、SACK、重排缓冲的纯函数单测。

## 7. 复制模型：权威 delta 事件流 + 瞬时状态快照 delta + RPC 控制面

承 §5 的双骨架，复制层具体落地两条数据通路：

### 7.1 权威 delta 事件流（持久层）

- 复制单元 = persist 的 **override delta**（改了什么），按 `PersistentId` 寻址（scene §6），与存档同一套序列化。
- 客户端改动作为**提议事务**上行（ch_events_reliable）；服务器定序（Lamport/序列号）、校验（§14）、合并（CRDT §13）后，作为**权威事务**下行广播给订阅者（§8）。
- 幂等与补发：每条事件带全局序号，客户端持最后确认序号，丢失增量补发；重复事件按序号去重。

### 7.2 瞬时状态快照 delta（表现层）

- 服务器每 tick 对每个客户端的 AOI 内实体生成**快照**，相对该客户端上次确认的 **baseline** 做 delta（只发变化的字段，借 Source entity baseline 形态）。
- 字段**定点量化 + 位域打包**（§15）：位置按世界定点坐标分段量化，角度/法锥按低位编码，速度按预算位宽。
- 走不可靠通道，可丢旧帧；客户端用插值（落后一个插值延迟窗口）平滑，预测实体用外推（§10）。
- 优先级：近处/交互中/刚变化的实体高频，远处/静止的低频或休眠（§8 dormancy）。

### 7.3 两通路的汇合点

服务器仿真 tick 内：输入 → 仿真 → ①持久事实固化为 delta 事件（入事件流 + 存档）②当前瞬时态生成快照。客户端：收事件流 → 施加权威 delta 到 scene 内容；收快照 → 覆盖/校正预测的瞬时态。**事件流改真相，快照改表现，二者在同一 tick 边界对齐（§9）。**

### 7.4 RPC / 远程过程调用：命令式控制面（第三类复制原语）

§7.1 / §7.2 两条通路是**声明式状态复制**（作者标注「这个状态要复制」，框架持续同步）。但有一类交互天然是**命令式、一次性、常带返回值**的：登录鉴权、匹配 / 组队、进出房间、购买 / 交易、技能释放确认、聊天、观战请求、管理指令、服务器节点间控制。用状态复制表达它们别扭——没有「持续同步的状态」，只有「发生一次的调用 + 一个回执」。因此 Prism 把 **RPC 作为与事件流、状态快照并列的第三类复制原语**：命令式控制面，补齐「触发式、点对点 / 定向、请求-响应」语义。

关键定位（不破坏架构不变量）：**RPC 不是真相源。** 服务端 RPC 处理器若改变权威持久态，产出的仍是 persist 的 **override delta 事件**（§7.1），经事件流复制下行——RPC 只是「触发权威变更的命令入口 + 返回确认」，绝不绕过 scene 单一真相，也不另立网络专属状态格式。这保证「网络 = 存档 = 时间线」三源同一（§5）。

调用形态：

- **单发（fire-and-forget）**：不等返回，可声明可靠或不可靠（非关键提示可走不可靠单发）。
- **请求-响应（request / response）**：带 reply 的异步调用（`async` / future 形态），每次调用带 call-id 关联回执，支持**超时**与**取消**；超时后在幂等保证下可安全重试。
- **服务端流式（server → client stream）**：一次请求、持续推送（如订阅排行榜 / 聊天频道），带背压（§15 预算）。

路由 / 定向（谁调谁）：

| 方向 | 典型用途 | 约束 |
|---|---|---|
| client → server | 提议 / 请求（默认） | 恒为「提议」，须权威校验（§9 / §14） |
| server → 指定 client | 定向回执 / 私信 | 权威下行 |
| server → AOI clients | 范围广播（§8） | 受 AOI 裁剪 |
| server → all / room / party | 全局 / 房间 / 小队广播 | 按订阅集 |
| peer ↔ peer | listen_server 直连 | 失败回退中继（§16） |
| server ↔ server | server_mesh 节点间控制（§12） | 权威移交 / 编排信令 |

可靠性与有序：默认映射到 §6 的 `ch_rpc_reliable`（必达、有序）；per-RPC 可声明 unordered（无因果依赖的调用免队头阻塞）或 unreliable 单发；大载荷 RPC 走 `ch_bulk` 应用层分片（§6）。

幂等与去重：每次调用带全局调用序号（Lamport / 序列号，§9），服务端按 call-id 幂等去重——断线重连重发不会重复执行（与 §13 一致性模型、§16 断线续传对齐）；request / response 用 call-id 关联迟到的 reply，避免错配。

默认安全（RPC 是最大的受攻击面，安全性设计化而非约定）：

- **client → server 恒为提议**：不存在「客户端直接改他人 / 全局权威状态」的 RPC（§14 根基设计化），服务端校验通过后才落为 override delta 事件。
- **能力 / 权限校验**：每个 RPC 声明所需 capability（谁能调、在什么状态能调），服务端派发前校验（§14 钩子）。
- **速率限制与配额**：每连接每 RPC 的调用速率 / 并发 / 载荷大小上限（§15 `on_rate`），反 DoS 与刷接口。
- **载荷校验**：参数 schema 校验 + 反序列化资源上限，拒绝畸形 / 超限载荷（§14 反序列化安全钩子）。

schema 与版本化：RPC 签名（名字、参数、返回、方向、可靠性、capability）纳入 §22 契约版本化；参数 / 返回复用 `prism_ecs` / `prism_reflect` 序列化（与复制字段同一套编解码）；跨版本演进遵循「新增可选、不改既有语义」，握手阶段协商双方可用 RPC 集与版本。

背压与拥塞：RPC 与权威事件流共享可靠通道的带宽预算（§15 预算调速）；request 积压时按优先级 / 配额排队或**显式拒绝**（返回 busy 而非静默丢弃），避免放大拥塞。

provider 接缝：RPC 的编解码与派发是可替换 provider（§4 模型）；跨机实时 RPC 走 `prism_net_transport`（§6），而**服务端节点间**控制 RPC 可选复用 `uwu_event_mesh` 的 RPC 传输形态（§5.1、§12），由 `evmesh` feature 门控、不进 `prism_net_core` 与客户端。

> RPC 的调用率、往返延迟、失败率、重试率、超时率、限流拒绝率均接 `prism_diagnostic`（§18）；可验证部分限于 call-id 幂等去重、超时 / 重试状态机、schema 编解码的纯函数单测，端到端 RTT 为 design target。

## 8. AOI 兴趣管理（area of interest）

海量世界不能给每个客户端复制全世界。AOI 按空间裁剪每个客户端的订阅，消费 `prism_world` 的空间原料（world §12.1 接缝）：

```text
客户端订阅集 = 其 AOI 中心(复用 world §7.1 需求源)覆盖的 {sector, 数据层, HLOD 簇}
服务器只向该客户端复制:
  - 订阅 sector 内对象的权威 delta 事件(§7.1)
  - 按簇/实例粒度(world §10 确定性簇 id)订阅: 近处逐实例、远处按簇聚合
  - 进/出 AOI 发订阅增删(relevancy 变更),错峰避免瞬时风暴
```

- **按簇/实例分级**：近处交互对象逐实例复制；远处只复制 HLOD 簇级聚合状态（簇是否被破坏、数据层是否切换），带宽正比于「可感知的变化」而非世界规模。
- **兴趣中心即需求源**：AOI 中心复用 world §7.1 需求源结构，服务器既用它做复制裁剪、也可反向下发权威流送决策给客户端（world §7.5），复制与流送决策同源。
- **relevancy 与 dormancy**：静止且无事件的对象进入休眠，不占快照带宽，变更或进入近 AOI 时唤醒（借 UE dormancy 形态）。
- **优先级预算**：每客户端每 tick 有带宽预算（§15），AOI 内对象按「距离 × 新鲜度 × 交互性」排序填充预算，超出的降频或延后。

## 9. 服务器权威与仿真定序（tick / command frame / Lamport）

- **固定步长 tick**（接 `prism_time` §4D）：服务器以固定 tick 推进权威仿真，所有事件在 tick 边界定序，保证跨机因果一致。
- **command frame**：客户端输入打上「目标 tick」上行；服务器在对应 tick 消费（借 Overwatch command frame 形态），输入缓冲吸收抖动。
- **全局定序**：单权威用序列号即可；多权威（§12 server_mesh）用 Lamport/向量时钟定序跨分片事件，CRDT 合并无中央锁（§13）。
- **提议-权威回路**：客户端提议 → 服务器校验/定序/合并 → 权威下行。权威下行**始终覆盖**客户端预测（§10），服务器是唯一真相仲裁者。

### 9.1 时钟同步与命令帧提前量（client clock / command-frame lead）

「按键即响应」的前提是客户端的输入能**准点落在服务器对应 tick**。若客户端时钟与服务器 tick 不对齐，输入要么迟到（被丢或补偿）要么过早（空等），手感直接劣化。Prism 用一套时钟同步 + 命令帧提前量把输入延迟隐藏到接近单机（借 QuakeWorld / Overwatch 形态，feature `clock_sync`）：

```text
RTT 采样(周期性 ping/pong,带发送戳):
  rtt_i = recv_time - send_time
  单向延迟 ≈ rtt_i / 2(对称假设,非对称时以服务器回报的接收戳校正)
滤波:
  EWMA 跟踪均值 + 中位数/低分位滤波抑制尖刺(抖动不污染基线)
命令帧提前量(lead):
  lead = 单向延迟 + 半个 tick + 抖动余量(由近窗抖动分位数定)
  客户端把输入打上「目标 tick = 服务器当前估计 tick + lead」上行
```

- **slew 不跳变**：本地时钟估计用 slew（缓调频率）向目标收敛，绝不瞬跳，避免时钟校正自身引发可见顿挫；大偏差（如刚连上 / 切网）才一次性 resync。
- **提前量自适应**：lead 大 → 输入几乎不迟到但本地输入延迟变高；lead 小 → 延迟低但丢帧率升。按近窗丢帧率与抖动分位在二者间自适应折中（design target，按游戏类型给默认档）。
- **与回滚 / 补偿自洽**：提前量定义了「输入该落哪个 tick」，回滚和解（§10）与延迟补偿（§11）回溯用的 tick 号都以此为基准；反作弊对 tick 号的约束校验（§14）也复用同一时钟模型，伪造提前量会被统计阈值捕获。
- **纯函数可测**：RTT 滤波、lead 估计、slew 收敛均为无 I/O 纯函数，可单测收敛性与抖动鲁棒性（§21）；端到端对齐精度为 design target，本机无多端实测。

## 10. 客户端预测与权威和解（rollback / reconciliation）

延迟下仍要「按键即响应」，靠客户端预测 + 权威和解（feature `prediction`/`rollback`）：

```text
客户端(每 tick):
  1. 本地施加可预测输入(移动/即时交互) → 预测态立即可见
  2. 记录该 tick 的输入与预测结果(ring buffer)
  3. 上行输入(带 tick 号)
服务器:
  4. 在对应 tick 以权威仿真处理输入 → 下行权威态(快照 §7.2 + 必要时 delta 事件 §7.1)
客户端收到权威态:
  5. 与本地该 tick 的预测比对
     一致 → 确认,丢弃该 tick 缓冲
     冲突 → 回滚到权威态 → 重放此后所有已缓冲输入(确定性 §21) → 得到校正后的现在
```

- **回滚重放确定性**：重放依赖 `prism_ecs` `determinism` 与定点数学（§21），保证「同输入同结果」，和解收敛。
- **预测范围受限**：只预测本地可自主的改动（自身移动/交互）；他人与权威仲裁结果（命中、拾取归属）不预测，等权威，防止大幅回弹。
- **平滑校正**：小冲突用一两帧的视觉平滑（smooth correction）吸收，避免瞬移；大冲突才硬回滚。
- **非确定性档位**：不开 `rollback` 时退化为「预测 + 快照插值纠正」（Source 形态），实现更简单、适合非竞技。

### 10.5 自适应插值、抖动缓冲与可视平滑（jitter buffer / smoothing）

远端实体（他人、非预测对象）走瞬时态快照通道（§7.2），到达本就乱序 + 抖动。直接「收到即显示」会抖、会橡皮筋。Prism 用一套回放缓冲 + 自适应插值 + 误差平滑把网络抖动与丢包隐藏成平滑运动（借 Source 插值 + VoIP 抖动缓冲形态，feature `interp`）：

```text
快照回放缓冲(snapshot buffer):
  收到的快照按 tick 入缓冲,渲染端故意滞后一个插值延迟 interp_delay 回放
  在相邻两快照间插值,得到任意渲染时刻的平滑位姿
插值延迟自适应:
  interp_delay = f(近窗抖动分位数)  抖动大→延迟大(更稳),抖动小→延迟小(更跟手)
缺帧外推(extrapolation):
  下一快照迟到→按速度短时外推,但有硬上限(防高延迟下大幅外推再被拉回的橡皮筋)
```

- **error smoothing**：权威态与当前显示态有偏差时（回滚校正、外推纠偏），用指数 / 样条在数帧内吸收，而非瞬移；平滑时长是 design target，大偏差才允许可见跳变。
- **仿真态 / 渲染态分离**：预测回滚（§10）发生在仿真态上，回滚产生的 hitch 不直接写渲染态——渲染态只被平滑后的结果驱动，保证「回滚在逻辑里发生、画面不漏抖」。
- **与提前量对偶**：本地预测用命令帧提前量（§9.1）隐藏上行延迟，远端用插值延迟隐藏下行抖动，二者是「隐藏输入延迟」与「隐藏观察抖动」的对偶，共同逼近单机观感。
- **纯函数可测**：插值 / 外推 / 平滑曲线、抖动分位估计均为纯函数，可单测连续性与上限约束（§21）；主观平滑度为 design target。

## 11. 延迟补偿（lag compensation 回溯命中判定）

竞技公平的关键：服务器判定命中时，**回溯到攻击者客户端实际所见的那一刻**（feature `lag_comp`）：

```text
服务器维护每实体近 N tick 的历史位姿环(history ring,时长 = 最大补偿窗口)
收到带 tick 号的攻击/命中提议:
  1. 还原该 tick(受 RTT/插值延迟修正)时各目标的历史位姿
  2. 在该历史快照上做命中判定(射线/碰撞)
  3. 判定结果作为权威事件(§7.1)下行
```

- **窗口有界**：回溯窗口上限（design target，如 ≤ 若干百毫秒）防「高延迟优势」被滥用；超窗按当前态判定。
- **防作弊耦合**：回溯用的 tick 号须与客户端输入序号自洽（§14），伪造旧 tick 会被约束校验与统计阈值捕获。
- **体验权衡**：被击中者可能「拐角后仍被命中」（peeker's advantage 的对偶），窗口与插值延迟是可调的 design target，按游戏类型折中（§15）。

### 11.1 端到端延迟预算（motion-to-photon 分解）

把「按键到画面变化」拆成可度量的环节，才能知道每一毫秒花在哪、哪些可被预测 / 插值隐藏（均为 design target，本机无多端实测）：

```text
环节                         性质              隐藏手段
① 本地输入采样 → 处理         本地,固定         低延迟输入管线(window_input)
② 可预测改动预测先行          本地,立即可见     客户端预测(§10) —— 隐藏②之后的全部上行+服务器延迟
③ 上行 + 命令帧提前量         网络,含单向延迟   提前量对齐(§9.1),使输入准点落 tick
④ 服务器 tick 处理 → 下行     权威,固定 tick    tick 粒度量化,批处理(§15)
⑤ 下行 + 插值延迟            网络,含抖动       快照插值缓冲(§10.5) —— 隐藏⑤的抖动与丢包
⑥ 渲染 → 显示               本地,含呈现延迟   与渲染/呈现管线对齐
```

- **可预测改动**（自身移动 / 即时交互）：②预测先行后，感知延迟 ≈ ①+⑥，逼近单机手感，③④⑤对「可见响应」透明（仅在和解冲突时以平滑吸收，§10.5）。
- **不可预测改动**（命中归属、他人状态、权威仲裁）：必须等权威，感知延迟含 RTT；用延迟补偿（§11）保证判定公平、用插值（§10.5）保证他人运动平滑，以「预测性反馈」（命中特效先播、权威确认后结算）缓解等待感，但不伪造权威结果。

## 12. 服务器网格分片与权威移交（`server_mesh`，借 Star Citizen 形态）

单服务器扛不住行星尺度多人。按 `prism_world` 的 sector 树（world §5）切成**分片（shard）**，每分片一台权威服务器，玩家跨边界时**无缝移交权威**（feature `server_mesh`）：

| 机制 | 落法 |
|---|---|
| 分片划分 | 按 sector 树子树分配给服务器节点；负载不均时动态再切（split/merge 子树） |
| 权威移交 | 对象越界 → 源分片把其 override delta（§7.1）+ 瞬时仿真态打包 → 目标分片接管 → 事件流续写 |
| 边界重叠带 | 分片边界设重叠 AOI 带（§8），移交前双订阅、移交后单订阅，无可见中断 |
| 一致性 | 跨分片事件用全局 Lamport 定序 + CRDT 合并（§13），确定性，无中央锁 |
| 过载降级 | 超预算时该分片时间膨胀/降低 tick 率（借 EVE 形态）而非崩溃，优先保事件流必达 |
| 容灾 | 分片权威日志周期快照（persist §11.4），节点宕机由副本从快照 + 事件重放恢复 |

分片是**复制/权威层的编排，不改内容真相**（仍是 scene + delta）；关闭 `server_mesh` 退化单权威（§9），代码编译期移除。它消费 world 的 sector 树与 persist 的 delta，**不是 world 的功能**（world §12.2 已明确归属本文）。分片间的权威移交与跨节点命令可选用 `uwu_event_mesh` 作编排后端（§5.1）：其租约 epoch fencing 恰好对应「旧权威被 fence、新权威接管」，at-least-once + 幂等保证移交命令不丢不重；该后端仅服务端、`feature evmesh`。

## 13. 一致性模型（确定性回放 / CRDT / 事件日志对账）

- **事件日志为准**：权威真相是 append-only 的 override delta 事件流（§5），全局定序后对所有端一致。
- **CRDT 合并**：并发提议用 CRDT（last-writer-wins / 计数器 / 集合，按字段语义选）合并（复用 persist §11.3 的合并规则），保证无中央锁下的最终一致。
- **确定性回放对账**：任意端可从某快照 + 后续事件流确定性重放（§21、world §18），得到完全一致的状态；服务器用它做权威对账与反作弊审计（§14）。
- **收敛保证**：客户端预测的分叉由权威覆盖 + 回滚重放收敛（§10）；跨分片分叉由 Lamport 定序 + CRDT 收敛（§12）。
- **可选 durable 后端**：「事件日志为准」的 append-only 存储可由内置实现承担，也可换 `uwu_event_mesh` 的 durable log + 消费游标 + 幂等键 + DLQ（§5.1），获得分区有序、at-least-once 补投与 timeline 分支/快照；override delta 的确定性合并天然幂等，正好满足其去重前提。该后端仅服务端、`feature evmesh`，不进 core/client。

## 14. 反作弊接缝（权威回路钩子，策略归 `prism_anticheat`）

反作弊**不是网络协议的一部分，而是插在网络权威回路上的可替换策略模块**，独立成 crate 与文档（`prism_anticheat`，另文），与仓库「一子系统一文档」惯例一致。网络在此只定义**接缝**：提供反作弊赖以生效的权威根基与钩子点，不实现具体检测策略。

### 14.1 网络提供的根基（不可替换）

- **服务器唯一权威**：客户端只上行输入与提议，绝不上行「我造成了什么结果」，结果只由权威仿真产生（§9）。这是反作弊强度的上限，任何 provider 都绕不过也不能削弱。
- **确定性可对账**：权威事件流可确定性重放（§13、§18），provider 据此判定「客户端提议能否在权威仿真中复现」。
- **传输与会话完整性**：序号 + nonce 防重放、握手签名、会话令牌（§6 crypto）由网络提供，provider 直接复用。

### 14.2 网络暴露的钩子点（供 provider 插入）

| 钩子 | 时机 | provider 可做什么 |
|---|---|---|
| `on_proposal` | 每条上行提议定序前（§7.1） | 约束校验（移动/物理/交互/资源守恒），拒绝或放行 |
| `on_tick_audit` | 每权威 tick 后（§9） | 确定性对账、统计计数（命中率/移动一致性/拒绝率） |
| `on_rate` | 每连接提议/RPC/订阅变更（§15） | 速率限制与配额，反 DoS |
| `on_lagcomp` | 延迟补偿回溯前（§11） | 校验回溯 tick 号与输入序号自洽 |
| `on_enforce` | 命中阈值/规则时 | 处置动作（flag/throttle/kick/ban/影子），回传遥测 |

### 14.3 边界与不变量

- **网络不实现策略**：具体校验阈值、统计规则、处置分级、商用集成全在 `prism_anticheat`；网络只保证「钩子被按时、带足够上下文地调用」。
- **provider 可换、骨架不可换**：换 provider（内置/第三方/自定义/no-op）不改变「客户端永不被信任、结果只由权威仿真产生」（§4 可替换边界）。
- **零开销可裁剪**：不开 `anticheat` 时钩子退化为 no-op，编译期移除；可信局域网/单机构建无反作弊税。
- **无内核态、无 ML**：接缝不假设也不提供内核态/驱动级能力；统计为确定性可解释阈值、**明确不含机器学习**——详见 `prism_anticheat`。

> 本节只描述接缝与钩子；反作弊的威胁模型、校验策略、统计阈值、处置分级、provider 契约与 design target 见 `prism_anticheat`。

## 15. 带宽与性能工程（delta 压缩 / 量化 / 优先级 / 预算调速）

- **delta 压缩**：瞬时态相对 baseline 只发变化字段（§7.2）；事件流本就是「只发改动」（§7.1）。
- **定点量化 + 位域打包**：位置用 world 定点坐标分段量化、角度/法锥低位编码、速度按预算位宽，位域紧打包（借 Source/UE 量化形态）。
- **优先级填充预算**：每客户端每 tick 带宽预算，AOI 内对象按「距离 × 新鲜度 × 交互性」排序填充，溢出降频/延后（§8）。
- **QoS 调速**：拥塞或过载时按序降级——先降远处快照频率 → 降量化精度 → 扩插值延迟 → 分片时间膨胀（§12），**永不丢必达事件**。
- **批处理与合并**：同 tick 多事件合并一包；小包聚合（Nagle 式但带时限，不拖延迟）。
- **零拷贝序列化**：复制单元按 `prism_ecs` 列块直接编码，避免中间对象（接 scene 数据导向快路径）。
- **内存零分配热路径**：序列化 / 量化 / AOI 填充的每 tick 热路径走预分配 ring buffer / arena 与双缓冲对象池，稳态零堆分配，避免 GC 式停顿污染 tick 边界；包缓冲用定长环复用，优化尾部延迟（p99 / p999 而非仅均值，borrowed from 实时音视频工程）。
- **向量化量化**：位置 / 速度 / 角度的分段量化与位域打包按 SoA 列块批量处理，接 `prism_math` 的 SIMD 后端一次处理多实体，摊薄每实体编码开销（design target）。
- **并行**：序列化/量化/AOI 裁剪按客户端分片并行（`prism_tasks`），服务器 tick 内可横向扩展。

> 每客户端带宽、支持并发连接数、每 tick 序列化耗时均为 design target，本机无多端实测；可验证部分限量化编解码、delta 编码、优先级排序的纯函数单测。

## 16. 可扩展性接缝（大厅 / 匹配 / 中继 / 重连）

网络 crate 只做**传输 + 复制 + 权威**，大厅/匹配/账号为上层服务，经接缝对接，不内建业务：

- **连接生命周期**：握手 → 鉴权（令牌，§14）→ 初始基线下发（ch_bulk）→ 稳态复制 → 断线 → 重连续传（持事件序号增量补发，§7.1）。
- **重连续传**：客户端持最后确认事件序号与 baseline 版本，重连时服务器增量补发，不重传全量世界。
- **中继回退**：直连失败（NAT 严格）走中继节点转发（§6）；`listen_server` 房主迁移时权威移交（§12 的小规模特例）。
- **匹配/大厅接缝**：对外暴露「创建/加入会话、分配权威服务器」的接口，具体匹配策略由上层实现，网络层只认「一个会话 = 一组连接 + 一个权威拓扑」。

## 17. 易用性与 API 人体工学（声明式复制 / RPC / 事件 DSL / prelude）

易用性目标：**作者标注「这个组件要复制」「这个函数是 RPC」，复制细节由框架接管**，与 scene §16.4「网络复制对称」契约对齐：

```text
声明式复制(概念示意,非最终语法):
  标注组件为 replicated → 自动纳入快照/事件流,按字段选通路与量化
    #[replicate(channel = state, quantize = pos16)]  瞬时态走快照通道
    #[replicate(channel = event)]                    持久态走事件流
  RPC(命令式控制面,§7.4):
    #[rpc(to = server, reliable)]              客户端 → 服务器(提议,须校验)
    #[rpc(to = clients, aoi)]                  服务器 → AOI 内客户端(权威广播)
    #[rpc(to = server, reply=Result, timeout=2s, cap="trade")]
                                               请求-响应(带返回/超时/能力校验)
    #[rpc(to = client(peer), reliable)]        服务器 → 指定客户端(定向回执)
  权威事件:
    服务器仿真产出 override delta,框架自动定序 + AOI 裁剪 + 下行
```

- **默认安全**：RPC 默认「客户端→服务器为提议、须校验」，不存在「客户端直接改他人状态」的 API（§14 根基设计化）。
- **请求-响应人体工学**：带 `reply` 的 RPC 以 `async` / future 返回，作者直接 `let r = foo(..).await?`，框架接管 call-id 关联、超时、重试与幂等去重（§7.4），不手写请求-响应状态机。
- **预测透明**：标注 `predicted` 的组件自动纳入预测/回滚缓冲（§10），作者不手写回滚逻辑。
- **AOI 透明**：复制自动受 AOI 裁剪（§8），作者不手写订阅管理。
- **prelude**：常用标注、通道、量化策略、RPC 宏一处导出；单机构建里这些标注编译为零开销（§3）。
- **与 scene 对称**：复制的字段路径复用 scene/`prism_reflect` 的 override 字段路径（scene §16.4），存档与网络「同一字段同一套序列化」。

### 17.1 开发者工作流（从单机到联机不分叉）

易用性不止于 API 形态，更在于**开发闭环**——改一行能多快看到多人下的结果：

- **零配置单机起步**：默认 `single` 构建里所有复制 / RPC 标注编译为零开销直调（§3），作者先按单机写玩法，联机不需要重写——「single 与联机同一套代码」。
- **单进程多端仿真**：一个进程内起 1 服务器 + N 客户端 + 注入的网络条件（§18），本机即可复现延迟 / 抖动 / 丢包下的预测回滚与插值行为，无需多机环境。
- **确定性测试 harness**：录制一段输入 + 权威事件流即可确定性重放（§13 / §21），回归「同输入同结果」，把网络 bug 变成可复现用例而非偶现玄学。
- **录制即调试（time-travel）**：权威事件流落盘可回溯到任意 tick 复盘（§18、§5.1 timeline），结合预测 / 和解可视化定位「这一回滚为何发生」。
- **运行时热调参**：插值延迟、提前量、带宽预算、AOI 半径、QoS 档位等为运行时可调的 design target 旋钮（§15），现场调手感不重启。
- **显式错误面**：鉴权失败 / 能力拒绝 / 限流 / 版本不匹配 / 断线原因都是显式可观测结果（§16 / §18），不静默吞错，便于定位联机问题。

## 18. 可观测性与诊断（网络剖析 / 回放 / 丢包注入）

- **网络剖析**（`prism_net_diag`）：每通道带宽、包率、RTT、重传率、丢包率、AOI 订阅规模、每实体复制开销，接 `prism_diagnostic` 计数器与 trace。
- **RPC 剖析**：每 RPC 的调用率、往返延迟、失败率、重试率、超时率、限流拒绝率，定位控制面热点与被刷接口（§7.4）。
- **事件流录制与回放**（feature `replay`）：权威事件流落盘，可确定性重放整局（§13、world §18），用于复现 bug、反作弊审计、观战/回放系统。
- **durable 日志与时间线后端**（feature `evmesh`，仅服务端）：录制/回放的落盘与游标可委托 `uwu_event_mesh`，复用其 durable log、消费游标、timeline 分支/差异比较/投影重放/快照恢复（§5.1），用于长局审计、分支复盘与断点续放，无需自建第二套日志。
- **网络条件注入**：开发期注入延迟/抖动/丢包/乱序/带宽上限，本机单端模拟多端劣化条件（弥补无真实多端环境）。
- **预测/和解可视化**：显示预测态与权威态的偏差、回滚发生点与重放帧数，调参延迟补偿与插值窗口。

## 19. 与各子系统的接缝（scene / persist / world / ecs / time / gameplay）

| 子系统 | 接缝 | 方向 |
|---|---|---|
| `prism_scene` | override delta / `PersistentId` / 网络复制对称（scene §16.4） | 网络消费 scene 的 delta 与字段路径 |
| persist（world §11） | 事务日志 / CRDT 合并规则 / 快照 | 网络复用其序列化与合并，作权威事件流 |
| `prism_world` | sector 树 §5 / HLOD 簇 id §10 / 需求源 §7.1 / 事务接口 §11（world §12 接缝） | 网络消费空间原料做 AOI 与分片 |
| `prism_ecs` | 复制组件列读写 / `serialize` / `determinism` | 网络按列块序列化、靠确定性做回滚重放 |
| `prism_time` | 固定 tick / 时间线（§4D） | 网络的仿真定序与 command frame 对齐其 tick |
| `prism_gameplay` | 复制组件 / RPC / 权威事件 DSL（§17） | 网络向上提供多人编程面 |
| `prism_anticheat` | 权威回路钩子点（§14）/ 确定性对账 / 遥测 sink | 网络提供钩子与根基，反作弊 provider 插入 |
| `uwu_event_mesh`（可选外部） | 权威事件流 durable 后端 / 回放 / 编排（§5.1、§12、§18） | 网络把事件日志 / 游标 / DLQ / timeline 委托其实现（仅服务端，`feature evmesh`） |

**关键不变量**：world 不碰网络协议（world §12.2），网络不碰内容真相（§1）；一套 override delta 两个消费者（存档 + 网络）；瞬时态不入存档。

## 20. crate 分层与模块布局

```text
prism_network/              (伞 crate,re-export + prelude + feature 聚合)
  prism_net_core/           定序、delta 编解码、CRDT、量化(no_std,纯函数)
  prism_net_transport/      可靠 UDP / QUIC / 多通道 / 拥塞 / 加密
  prism_net_replicate/      事件流通路 + 快照通路 + baseline + 优先级
  prism_net_aoi/            兴趣裁剪、relevancy、dormancy(消费 world 空间)
  prism_net_predict/        输入缓冲、预测、回滚重放、延迟补偿
  prism_net_authority/      tick 定序、command frame、提议校验、对账
  prism_net_mesh/           分片、权威移交、重叠带、容灾(feature server_mesh)
  prism_net_acheat_api/     反作弊接缝: AntiCheatProvider trait + 钩子点 + 默认 no-op(实现归 prism_anticheat)
  prism_net_replay/         事件流录制/确定性回放(feature replay)
  prism_net_evmesh/         uwu_event_mesh 适配: 权威事件流/回放/编排后端(feature evmesh,仅 std/server)
  prism_net_diag/           带宽剖析、网络条件注入、预测可视化
```

单向依赖；`prism_net_core` 零传输依赖（可脱网单测）；`client`/`server` 入口 feature 裁剪；`single` 构建整伞 crate 可被编译期裁空。

## 21. 确定性与可复现

- **确定性是多人一致性的前提**：回滚重放（§10）、跨分片合并（§12）、审计对账（§14）、回放（§18）全依赖「同输入同结果」。
- **来源**：复用 `prism_ecs` `determinism`（固定迭代序、无并发非确定）与 `prism_math` 定点运算（无浮点漂移），固定步长 tick（`prism_time`）。
- **可验证**：定序、CRDT 合并、delta 编解码、量化、回滚重放均为纯函数，可做确定性单测与跨平台一致性校验；真实网络吞吐只能标 design target。
- **边界**：浮点仿真若参与权威，必须定点化或在权威端单点计算后以 delta 下行，禁止各端浮点自算后比对。

## 22. 契约、不变量与版本化

- **契约**：对上（`prism_gameplay`）暴露声明式复制 + RPC + 权威事件；对下消费 scene delta / persist 事务 / world 空间 / ecs 列 / time tick。
- **核心不变量**：①网络不另立真相（复用 persist delta）②world 不碰协议、网络不碰内容真相③一套 delta 两消费者④瞬时态不入存档⑤服务器唯一权威、客户端永不被信任⑥所有权威改动可确定性重放对账⑦传输/复制/AOI/预测/权威/反作弊皆为可替换 provider，但 provider 不得违反①—⑥（§4 可替换边界）。
- **版本化**：复制 schema 随 scene/persist 的稳定字段 ID 演进（scene §6 内容寻址）；协议版本握手协商（含双方可用 RPC 集与 RPC 签名版本，演进遵循「新增可选、不改既有语义」），拒绝不兼容版本；事件流格式向后兼容读（旧序号可被新版读）。
- **feature 兼容矩阵**：`single`/`listen_server`/`dedicated`/`server_mesh` 与各能力 feature 的合法组合在此约束，非法组合编译期报错。

## 23. 路线图、基准即规格与诚实边界

路线图（M 对齐 world/scene 的里程碑编号惯例）：

```text
M1 传输: 可靠 UDP + 多通道 + 拥塞 + 丢包/延迟注入(diag)
M2 复制骨架: 权威 delta 事件流(复用 persist) + 瞬时快照 delta + baseline
M3 AOI: 空间裁剪 + 簇分级 + relevancy/dormancy(消费 world 空间)
M4 预测与和解: 客户端预测 + 回滚重放 + 平滑校正(prediction/rollback)
M5 延迟补偿: 历史位姿环 + 回溯命中 + 窗口有界(lag_comp)
M6 反作弊接缝: 权威回路钩子点 + 确定性对账根基(provider 与默认实现归 prism_anticheat)
M7 规模: 服务器网格分片 + 权威移交 + 过载降级 + 容灾(server_mesh)
M8 工具: 事件流录制/回放 + 网络剖析 + 预测可视化 + QUIC 适配
M9 后端 provider: uwu_event_mesh 适配(durable log/游标/DLQ/timeline) 作权威事件流/回放/编排可选后端(evmesh,仅服务端)
```

基准即规格（全部 design target，本机无多端实测，如实标注）：每客户端稳态带宽、支持并发连接数、权威 tick 率、端到端输入延迟（预测隐藏后感知）、回滚重放帧数上限、延迟补偿窗口、分片移交无中断时长。可验证的 CPU 纯函数部分（定序/CRDT/delta/量化/回滚）设确定性单测门槛。

诚实边界：

- 本机无多端网络环境，一切吞吐/延迟/丢包恢复为 design target，不是实测。
- 反作弊上限由「服务器权威 + 确定性对账」决定，客户端侧检测可被破解、仅作噪声过滤；**不含任何 ML**。
- 浮点仿真参与权威必须定点化，否则无法保证跨平台确定性与回滚收敛。
- QUIC/中继/NAT 穿透依赖部署环境，设计只定接缝与回退策略，不保证所有网络拓扑可直连。
- `uwu_event_mesh` 后端仅用于服务端权威事件流 / 回放 / 编排，不触碰客户端、瞬时态快照通道与 `no_std` 核；其仍处 `2.0.0-dev`、自述会破坏性调整，故定位为可选 provider、以 `EventLogBackend` 接缝隔离，默认实现不依赖它。

## 24. 一句话总结

**Prism Network 是架在 scene/persist/world 之上的横切复制层：以 persist 的 override delta 作权威事件流骨架（存档/多人/时间轴三者同源），叠加瞬时态快照 delta 做延迟隐藏，补一条命令式 RPC 控制面处理登录 / 匹配 / 交易等触发式交互（§7.4，仍以 override delta 固化权威改动、不另立真相），用时钟同步 + 命令帧提前量 + 自适应插值把输入延迟与网络抖动隐藏到接近单机手感，用服务器权威 + 客户端预测回滚 + 延迟补偿兼顾公平与即时，用 AOI 与分片兼顾行星尺度，用确定性回放对账为反作弊提供根基（反作弊本身是可替换 provider、策略归 `prism_anticheat`）；传输 / 复制 / AOI / 预测 / 权威 / 反作弊 / 事件流后端皆为可替换 provider（durable 事件流 / 回放 / 编排后端可选 `uwu_event_mesh`，仅服务端、`feature evmesh`）——不另立网络真相、不侵入流送热路径、无任何 AI/ML。**
