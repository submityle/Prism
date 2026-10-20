# prism_asset_container

Prism 运行时容器、Manifest 路由、overlay 与本地安装事务。支持 v1
Raw/Deflate payload 和 v2 独立 Raw/Deflate Chunk；不执行 Cook 或 GPU 上传。

读取不可信文件时，可把同一个 `ContainerReadBudget` 传给多个 mount，限制
元数据、条目数、单元大小、展开比例、挂载校验工作量、输出与并发字节缓冲。
`read_range_owned` 在分配前执行门禁；返回的 `Vec` 由调用方驻留预算负责。

- [设计与安全边界](docs/prism_asset_container_design_zh.md)
- [版本记录](docs/version/README.md)
- 验证：`cargo test -p prism_asset_container --all-features`

range source 必须在 reader 生命周期内保持不可变。BLAKE3 提供完整性校验，
不是发布者认证；局部读取不是密码学子范围认证。
