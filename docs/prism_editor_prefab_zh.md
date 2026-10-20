# Prism Prefab：稳定身份、分层组合与原子命令

本实现位于 prism_editor_core::prefab，实现设计 §12 的独立作者模型：资产、单继承变体、嵌套引用、场景实例、显式覆盖、还原、应用到基底、下传播和可逆编辑。实际组合结果由 PrefabStore::compose 产生；测试直接构建资产图并检查解析后的实体/字段，不使用 runtime/loopback 替身。

core 根只增加 pub mod prefab，无新增 Cargo 依赖，不依赖 Bevy，也不要求修改共享反射接口。所有公共类型从该模块导入，无需额外根级 re-export。此模块不改 app、UI、content 或 timeline。

## 身份与值边界

资产、实例、源实体、嵌套槽各有独立 u128 ID，由宿主分配持久身份（建议 UUID），保存/重载保留原值；文件名和运行时 ECS 索引不参与身份。宿主必须避免把旧身份复用于不相关的新对象。

实体地址 PrefabEntityPath 是嵌套槽链 slots 加源实体 entity。例如 [slotA, slotB]/entity42 与 [slotC]/entity42 不冲突。同一资产可以多次嵌套相同源资产。变体继承保持已有地址；显式替换槽引用更换该命名空间的内容。指向已不存在实体的覆盖被拒绝，不静默改指其他实体。

PrefabProperty 包含 entity、component、fields。component 是宿主稳定 schema key；fields 是精确对象键序列，空序列表示整个组件。键里的空白、换行、点、斜杠、Unicode 都按字面保存，不分割、不 trim、不规范化。空 component key 无效，空对象字段键合法。数组作为原子值替换，不支持易漂移的数组下标覆盖。

PrefabValue 精确区分 Null、bool、i64、u64、f64 原始位、UTF-8 字符串、字节、数组、对象。NaN payload、负零、大整数和原始字节不经过 JSON/lossy 转换。Null 与 Remove 不同。该值模型是业务中立语义载体；宿主负责 component/schema key 的类型登记、Prism 原生反射/ECS 转换与业务类型约束。模块不猜测业务组件布局或执行任意反射内存写入。

## 实际组合顺序

一个资产先递归组合 base，再应用本层；实例先组合引用资产，再应用实例层。每层依次应用：嵌套槽增/替换/删、本地实体增添、精确实体删除、父关系替换、属性覆盖。槽挂接时给嵌套组合的每个根赋予 mount parent，内部父地址自动加槽前缀。继承/嵌套的混合依赖环、父链环、缺失引用均被拒绝。

PrefabLayer.entities 是新增实体集合，不能覆盖已继承的源实体 ID。slots 的 Some 增添/替换引用，None 删除已存在的槽；removed 精确删除实体，没有隐式子树级联。parents 显式重设存活实体父关系；删除父节点时必须在同一最终有效层里删除或重挂子节点。结构编辑通过 ReplaceAsset / ReplaceInstance 替换一个作者记录完成，组件值仍以 Arc 共享。

每个资产/变体必须自身可独立组合，不能依赖更下层来修复无效资产。同一事务允许先登记尚未存在的依赖，最终统一验证；但会立即读取组合结果的 Propagate 等操作要求前序编辑已形成可组合视图。

overrides 精确记录作者意图。与继承值相等的 Set 仍是覆盖 pin；同层的父字段与子字段覆盖互相冲突，兄弟字段可以共存。Set 可以新增最终字段或整个组件，路径的中间对象必须存在。Remove 必须对应已有字段/组件，重复删除是错误。

## 还原、应用基底和下传播

| 操作 | 实际行为 |
| --- | --- |
| 修改基底/变体 | 后续 compose 自动读取最新依赖；未覆盖字段变化，实例 pin 保留 |
| Revert | 删除本层精确属性覆盖，让继承值重新显现；没有该覆盖则报错 |
| ApplyToBase | 把一个精确作者覆盖移动到显式祖先资产，同时清除源层覆盖，保留原 qualified namespace |
| Propagate | 获取源资产当前有效值，在指定后代层写入显式 pin，覆盖其局部同路径/更细粒度 pin |

ApplyToBase 只接受继承链祖先（实例可以应用到它直接引用的资产），不接受自身资产或不相关资产。跳过的变体若有重叠覆盖会冲突；目标上更宽/更细的覆盖也必须满足同层不重叠约束。实例新添实体的覆盖不能应用到尚无该实体的基底，最终验证会拒绝。结构覆盖的还原/应用通过显式编辑相关记录并放入同一事务表达。

嵌套属性应用到包含它的外层基底时保留槽链；不会自动把 [slot]/entity 解释成被嵌套源资产的本地 entity。跨嵌套命名空间的“应用源 Prefab”需要宿主明确重映射并提交作者记录，不会被隐式执行。

Propagate 只接受唯一的正确后代：变体或引用源资产/其变体的实例。不重写中间的未选中变体来实现单实例传播，因此保留其他实例的独立覆盖。选中变体会自然影响继承该变体且无覆盖的下游；这是显式选中变体的作用域。目标有更宽的父字段覆盖时拒绝，以免丢掉同级字段。接收者按继承深度/稳定 ID 排序，祖先先于后代，不受输入顺序影响。

源属性缺失时，Propagate 清理目标局部重叠 pin，并在仍继承到该属性时写 Remove；若已缺失则不产生冗余 Remove。这是当前作者图的有效缺失状态，不是“永远禁止未来新增此字段”的墓碑。源实体缺失、目标被结构删除或字段形状不兼容会显式失败。强制传播产生的 Set 是 pin，后续基底变化不会越过它；可 Revert 恢复实时继承。

## 原子事务与逆历史

store.transact(expected_revision, edits) 在稀疏 overlay 上按序暂存，验证整个最终资产图、全部实例和预算，成功才发布。晚到的缺失/冲突/环/预算/修订溢出均不发布部分记录，也不推进 revision。等价 no-op 保留原记录 Arc 身份，不推进 revision。

PrefabCommand::new(label, revision, edits) 可直接放入现有 CommandStack<PrefabStore>；一条命令内部的多项编辑构成原子批次。要求原子的 Prefab 编辑应放在同一个 PrefabCommand 中，不依赖外层通用 Transaction 对多个独立命令的尽力回滚。现有 DocumentState/DocumentLayer 是文档层元数据；宿主负责 document ID 到此作者库/实例的映射、dirty 状态及保存调度，本实现未添加第二套 DocumentState 所有权。

PrefabReceipt 只保存变化资产/实例的 before/after Arc，不克隆整个 PrefabStore 或整幅解析图来存历史。编辑最多克隆受影响的作者记录壳与正在修改的组件语义值；未修改组件值和无关记录共享。changed_records() / owners() 暴露作者记录变化范围；它们不是全部受依赖传播影响的实例列表。

undo/redo 检查库身份与受影响记录的精确 Arc 身份，防止覆盖后续外部修改；不相关记录的修改可以保留。还原前再次验证依赖/预算，因此新增依赖可能让某次 undo 合理失败。失败时库、revision 和 CommandStack 对应历史项均保留。revision 在 commit/undo/redo 时单调递增；新解码库具有新库身份，旧 receipt 不可重放。

原子保证覆盖所有返回 Result 的错误，不把分配失败导致进程 abort/panic 宣称为可恢复事务。Store 通过独占可变借用同步使用，未附加后台线程/全局锁；主机异步提交须捕获 revision，回到所属线程检查后提交。

## 预算和确定性保存

PrefabLimits 每项必须非零，max_depth 硬上限 128。默认资产 1024、实例 4096、实体累计访问 65536、值节点累计 1048576、文本/字节累计 32MiB、深度 64、工作步数 2000000、每批编辑 4096、编码文件 32MiB。Propagate 每个接收者也占一个编辑单位。

实体/值/字节预算计算遍历和中间组合的累计访问量，不是唯一实体数、压缩文件大小或 RSS。验证所有资产与实例时共享组合预算；深层变体和多实例会重复访问上游，有界拒绝优先于无界展开。预算限制作者输入、依赖深度、字段/父链/值深度和嵌套放大。组合临时值不进入逆历史。编码、解码也受预算，宿主应配置足够余量；同一作者图可能在更昂贵操作或更紧预算下显式拒绝，不能只按最终实体数量设置限制。

encode() / decode(bytes, limits) 使用 PRPF + 版本 1 的二进制格式，u128 ID、u64 长度/修订和整数均为 little endian。记录、对象、集合按 BTree 稳定顺序保存；同样的作者内容和 revision 产生相同字节，与插入顺序无关。只保存作者记录与 revision，不保存组合缓存、receipt、runtime entity、预算或平台指针。

解码前检查文件大小；序列长度使用剩余字节/工作预算约束，不按不可信数量预留容量。字符串在分配前检查字节预算，值递归在读取时检查深度/节点。严格拒绝重复/乱序 key、身份键不匹配、未知 tag/版本、无效 UTF-8、截断/尾随字节及非法依赖图；整个文件验证成功后返回新 Store。编码缓冲逐次检查字节上限。I/O 和磁盘原子替换由宿主负责，模块只提供有界字节 codec。

## 验证与联测入口

公共集成测试位于 pkg/prism_editor_core/tests/prefab.rs，覆盖多层变体/多重嵌套、挂接父关系、精确值/Unicode、覆盖传播/未选中实例保留、还原/应用、删除/重挂、混合环、冲突/预算失败原子性、稀疏 undo/redo、跨库/并发历史拒绝、no-op 和确定性/恶意 codec。

2026-10-21 验证命令（独立 target，单 job，关闭 debug/incremental）：

结果：core 既有 5 项测试、Prefab 公共 API 19 项回归全部通过；all-targets/all-features/no-deps strict Clippy 通过。

    CARGO_TARGET_DIR=/private/tmp/prism-prefab.4K2HLO CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0 cargo test -p prism_editor_core --all-features --offline
    CARGO_TARGET_DIR=/private/tmp/prism-prefab.4K2HLO CARGO_BUILD_JOBS=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 cargo clippy -p prism_editor_core --all-targets --all-features --no-deps --offline -- -D warnings

联合测试可直接导入 prism_editor_core::prefab::*，创建 Store，将资产/实例登记在一个事务内，用 compose 驱动层级/Inspector 读模型，用 PrefabCommand 驱动编辑与历史。assets()/instances() 提供加载后的枚举。无需新增依赖或 root 接线。Native ECS 实体生成/销毁、Prism 反射类型映射、资源解析/异步加载、UI、文件写入仍由各自宿主适配，本提交不把这些边界宣称为已经联通。
