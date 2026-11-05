# 测试脚本

仓库测试集中在本目录，按功能分组。正式 PBF 位于根目录 `PBF/`，原有 demo 功能模块仍位于 `demos/`。

`tests/pbf/` 验证根目录正式 PBF 的 Newton 双 State、原规模逐位回归、Graph、CLI 与渲染。与 `tests/fluid/pbf/` 的原版 demo 测试分开维护。先激活 Newton/WarpSim 环境，再运行：

```powershell
python -B -m unittest discover -s tests/pbf -t . -v
python -B -m tests.pbf.benchmark
python -B -m tests.pbf.verify_viewer
python -B -m tests.pbf.verify_integrity
```

正式 PBF 测试依赖与详细说明见 [PBF README](../PBF/README.md)。

`tests/sim_ready/` 验证新离线资产管线：模型适配与放置、容器搭接和拓扑、独立 CLI、失败回滚、CUDA 烘焙。运行 `python -m unittest discover -s tests/sim_ready -t . -v`。原 `tests/fluid/boundary/` 保留归档流程的回归检查。

```text
tests/fluid/
├── boundary/          # 地形与容器生成、网格检查、SDF 烘焙与独立 CLI
├── pbf/               # PBF 配置、控制、静态资产选择与资源接入
├── mmpbf/             # MMPBF 公式与行为
└── pbf2waycoupling/   # 双向耦合、墙体钳制、渲染与接触参考实现
```

在仓库根目录运行全部测试（环境需同时满足原 demo 与正式 PBF 的 Newton 依赖；下面沿用原 `warplab` 环境名）：

```powershell
conda run -n warplab --no-capture-output python -m unittest discover -s tests -t . -v
```

仅运行 PBF 和场景边界测试：

```powershell
conda run -n warplab --no-capture-output python -m unittest discover -s tests/fluid/pbf -t . -v
conda run -n warplab --no-capture-output python -m unittest discover -s tests/fluid/boundary -t . -v
```

运行单个模块：

```powershell
conda run -n warplab --no-capture-output python -m unittest tests.fluid.boundary.test_sdf_baker -v
conda run -n warplab --no-capture-output python -m unittest tests.fluid.boundary.test_bake_sdf -v
```

NanoVDB 测试需要 CUDA，无 CUDA 时跳过；双向耦合的渲染测试需要 CUDA 和可用的 OpenGL 上下文。渲染测试的截图保存在 `outputs/pbf2way/render-check/`。`test_contact_reference.py` 是供等价性测试和接触性能基准调用的参考 kernel。
