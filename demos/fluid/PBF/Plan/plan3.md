# SDF烘焙、缓存与读取
- 默认烘焙体素间距为 **0.0075 米**，通过离线 `SDFOptions` 调整；启动读取直接使用文件内的体素信息。
- 容量参考：当前墙厚 0.25 米、坑底厚 0.20 米的容器，float32 全覆盖 NanoVDB、每侧 padding 为 3 个体素时，SDF 本体约占 140.88 MiB 显存，未压缩 `.nvdb` 约 147.72 MB；Warp 1.15 加载时本体短暂占用约 281.75 MiB，不含 CUDA、PBF 与烘焙临时资源。
- 代码实现到烘焙SDF缓存与PBF主程序与Device读入SDF为止。边界处理暂不实现。
- 烘焙代码单独实现，不要参考或抄袭仓库中已有的SDF烘焙器，那是过往实验性代码。烘焙kernel基于Warp dsl，可以借鉴官方demo或其存在于网上的文档中，标准的烘焙做法。
- 当前接入：PBF 启动时仅读取代码中固定 `STATIC_SDF_PATH` 的已有 `.nvdb`，不接受烘焙参数，不自动烘焙。独立 `bake_sdf.py` CLI 复用离线 Python API，提供必填 `--boundary-obj`、默认 `0.0075` 的 `--sdf-voxel-size`、`--sdf-force-rebuild` 和默认 `cuda:0` 的 `--device`；仅 CUDA。烘焙后输出路径并退出，不创建仿真、不修改固定路径。以下网格检查和缓存逻辑保留在离线 Python API 中：
    1. 功能1：在接收到指定容器后，按照此前容器合成的"逻辑"，检查容器是否密闭，检查obj/mesh是否闭合，以及简单检查mesh质量是否太差，过于差的拒绝(差的定义待定，搜索分析后与我讨论，搜索分析后给出公认"差"的定义)
    2. 功能2：在功能1检查之前，先检测对应文件名的obj是否存在SDF烘焙副本文件，如有则跳过烘焙，如没有则启动烘焙并保存为volume/nanovdb文件(warp能直接读取nanovdb吗，还是要转为wp.volume？nanovdb和volume是同一个东西吗？)，SDF结果保存为和obj相同的名称。

## 已确认实施约定

- 网格检查与离线烘焙位于 `boundary/mesh_validation.py`、`boundary/sdf_baker.py`；`boundary_model.py` 提供固定路径加载和设备采样辅助函数，不使用 helper 模块全局 SDF 变量。
- 两个入口选定设备后创建仿真实例，由 `Example.__init__()` 调用固定路径加载函数，并以 `self.static_sdf` 持有完整 `wp.Volume`；入口不再重复加载。`Example` 与 `create_pbf_simulation` 保留原签名，不增加边界参数；实例只读盘、不负责烘焙，`reset()` 不重新加载。直接创建实例同样需要有效的 SDF 文件和 CUDA。
- 离线输入裸文件名从 `PBF/Assets/Container` 解析，含目录的相对路径按工作目录解析；输出为 `PBF/Assets/Boundary/<stem>.nvdb`。运行时直接读取 `STATIC_SDF_PATH`，不改变 PBFConfig 或物理参数 JSON；文件缺失或损坏即报错。其他文件名烘焙后需手动修改固定路径。
- 离线缓存命中时直接加载，不读取 OBJ、不校验源内容/参数/哈希、不生成 sidecar。错误缓存报错，通过 `SDFOptions(force_rebuild=True)` 显式重建。缺失或强制重建时先检查后烘焙，成功后原子保存。
- `.nvdb` 是磁盘序列化格式；`wp.Volume` 持有其 GPU NanoVDB 数据和查询 id。实例的 `self.static_sdf` 必须保留完整 Volume，不能只存整数 id。
- 第一版仅 CUDA、float32 全覆盖场，padding 每侧 3 体素；不做窄带或压缩。保持 OBJ 坐标，负值表示实体，正值表示空腔和外部。写入所有分配的 tile 体素，查询失败不保存。
- 简单检查：非空/有限坐标、合法索引、重复面、精确退化面、边与顶点流形、闭合、绕序、一个正外壳与一个负空腔、正总体积；float32 转换后再检查。仅精确合并重合点，不修补、不近似焊接、不自动翻转。细长但非退化的三角形仅告警。
- 不接入 SDF 碰撞、不改粒子配置或原有箱体边界、不做坐标和尺寸自动对齐。

## 后续理想检查需求（本轮不实施）

- 全局三角形自交、外壳与内腔的几何包含及相交关系。
- 与体素间距对应的最小墙厚、狭窄通道及薄特征解析能力；烘焙后符号与表面偏差评估。
- 这些不能仅凭 watertight、正总体积或 Manifold 的成功返回来认证，需进一步讨论检查方法与成本。

