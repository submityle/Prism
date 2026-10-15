# 响应身份、动态分桶与 CPU 执行

运行期不再按九种 BSDF 组合判定着色身份。Lit、Npr、Unlit 与注册程序均来自 `ProgramRef`，活动瓣只影响程序内部的响应数学。Lit 在同一积分器内叠合次表面、清漆与绒毛，不用互斥 class 丢弃其他活动瓣。

## 分桶与校验

`ResponseBucketTable::from_execution_tables` 从已验证的计划端点导出桶，按 `(execution_family, program_id, render_phase)` 排序去重。表、计数、偏移与间接分派参数均为变长结构；没有九槽上限。显式桶表构建失败时拒绝该表，不静默换成内建表。

内建 id 为 Lit=0、Npr=1、Unlit=2；注册 id 0 合法。相同执行族中不同程序引用若重用同一个桶身份，登记会报歧义。二值区域选择只切换 Lit/Npr，不能改写 Unlit 或注册程序。部分绑定、越界索引、错误 package、无效 permutation 与过期材质均拒绝。

opaque 路径仅接收 render class 0–3 的 Surface/Hair；Transmissive、Transparent、Additive 和其他运行域不能进入该桶。

## CPU 注册响应

应用通过 `CpuResponseRegistry` 为已验证的注册程序绑定真实 CPU evaluator。调用 `resolve_pixel_with_cpu_responses` 时，先验证执行代次、计划所有权、package 与 permutation，再将采样后的表面、重建 frame、位置和共享光照环境传入 evaluator。输出包含五个辐射通道与独立 alpha，非有限或非法输出报错。没有 evaluator 就明确失败，不以程序 id 合成颜色。

这个接口执行宿主函数，不解释 SPIR-V，也不证明 GPU 注册程序已经执行。`PrincipledDirectChannels` 在源项处分开 direct/transmission；两通道不含 emission，兼容的 aggregate radiance 包含一次 emission。

## 验证记录

2026-10-15，`CARGO_INCREMENTAL=0 cargo test -p prism_render_shading --locked --quiet`：3385 单元测试和 27 集成测试通过，6 项既有测试 ignored。

生产分类 shader 的独立 GPU 测试运行 classify → prefix → scatter 并读回，验证 68 个确定性桶、注册 id 0、混合活动瓣、区域端点、错误绑定、容量与溢出以及 opaque eligibility；它不代替响应函数执行测试。
