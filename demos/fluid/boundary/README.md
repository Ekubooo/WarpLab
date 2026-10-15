# Boundary 场景生成与合成

`creator1.py` 生成并导出七种程序化地形；`scene_creator.py` 自动生成阶梯、放置 OBJ 模型，并导出布尔并集，可选简化副本。`scene_container_creator.py` 是原合成器的独立副本，生成简化景观和带四墙、顶盖的简化封闭容器。各脚本可独立运行，场景合成器会自行生成阶梯。

以下命令均在项目根目录运行，使用已配置的 Conda 环境：

```powershell
conda activate warplab
```

地形生成器依赖 `numpy`、`trimesh`。场景合成还需要 `manifold3d`、`pillow`；简化统计需要 `scipy` 或 `networkx`。当前 Conda `warplab` 环境已满足运行需求。

输入模型放在 `Assets/origin/`；程序化地形输出到 `Assets/procedural/`，合成场景输出到 `Assets/output/`。

## 默认运行命令

生成全部七种程序化地形（推荐命令，显式选择 `all`）：

```powershell
python demos/fluid/boundary/creator1.py --terrain all
```

导出 Y-up 的 `flat.obj`、`pyramid_stairs.obj`、`random_grid.obj`、`wave.obj`、`box.obj`、`gap.obj`、`heightfield.obj` 到 `Assets/procedural/`。其中高度场使用内置示例，随机地形每次可不同。无需输入 OBJ。

`all` 是 `--terrain` 的取值。省略 `--terrain` 时程序实际默认生成 `wave`。也可通过包入口生成全部地形：

```powershell
python -m demos.fluid.boundary.creator1 --terrain all
```

生成默认阶梯与兔子合成场景：

```powershell
python -m demos.fluid.boundary.scene_creator
```

当前代码中的阶梯区域为 3.1×1.6 米，台阶宽 0.15 米、高度增量 0.1 米，中心平台为 0.7×0.7 米、顶面高约 0.5 米。默认导入 `bunny_flat_bottom.obj`，按 Y-up 输入，标定倍率约为 6.12274，绕竖直轴额外旋转 0°。默认开启平台尺寸联动，实际倍率约为 `6.12274 × 0.7 / 1 = 4.28592`；兔子宽约 0.665 米、深约 0.516 米、高约 0.630 米，宽度占平台短边的 95%。

兔子的平底面积重心自动对齐平台中央，并下沉 0.003 米：底面支撑中心的内部 Z-up 坐标约为 `(1.55, 0.8, 0.497)`，默认 Y-up 导出后约为 `(1.55, 0.497, -0.8)`。定位只看支撑面，不使用整个兔子（包括耳朵、身体）的投影包围盒中心。模型完整落在当前平台范围内，整体包围盒在 X 两侧余量约为 0.014 和 0.021 米，在 Y 两侧约为 0.151 和 0.033 米。裁平副本的整个底面都位于平台内部，实体布尔并集将其删除，连接处没有兔子底面或内部空腔。默认导出：

- `Assets/output/stairs_scene.obj`：原始拼接对照，保留内部面。
- `Assets/output/stairs_scene_bool.obj`：实体布尔并集。

`bunny_flat_bottom.obj` 从 `bunny_repaired.obj` 另存而来，原始两份兔子均保留。在输入 Y-up 坐标的 `Y=0.04` 处水平裁切，删除旧凹凸底面，再封上完全共面的底面。裁掉的源模型高度约为 0.00669（世界尺寸随实际倍率改变），主体外形保持不变。平整底面在独立兔子 OBJ 中保留以保证封闭，在场景布尔结果中消失。需要重新生成副本时运行（需本地存在修复版源模型）：

```powershell
python -m demos.fluid.boundary.flatten_bunny
```

启用简化并另存 `stairs_scene_bool_simplified.obj`：

```powershell
python -m demos.fluid.boundary.scene_creator --simplify
```

直接运行脚本也支持：

```powershell
python demos/fluid/boundary/scene_creator.py
```

## 双分支封闭容器合成

新脚本保留当前 OBJ 的轴转换、平台联动缩放、支撑面定位和下沉方式。两个分支共用一次准备好的地形与 OBJ：景观分支只合成地形和模型；容器分支从未简化的地形、模型与四墙、顶盖求整体实体并集，再独立简化。

在项目根目录、Conda `warplab` 环境中无参数运行：

```powershell
python -m demos.fluid.boundary.scene_container_creator
```

也可直接运行：

```powershell
python demos/fluid/boundary/scene_container_creator.py
```

本脚本没有 CLI 参数，默认始终简化，仅导出两份 Y-up OBJ：

- `Assets/output/stairs_scene_bool_simplified.obj`：仅地形与 OBJ 的景观参考，可用于不显示墙体的渲染。
- `Assets/output/stairs_container_bool_simplified.obj`：地形、OBJ、四面墙与顶盖的整体封闭碰撞容器，内部保留流体空腔。

不导出拼接或未简化中间结果，也不删除目录中旧脚本先前生成的文件。景观输出与原合成器的默认简化副本同名，重复运行会覆盖；原 `scene_creator.py` 的代码和命令保持不变。当前没有 SDF 烘焙、表面重建或 PBF 接入。

修改 `scene_container_creator.py` 顶部配置，或调用其 Python API：

| 配置 / API 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `DEFAULT_BLOCK_SIZE` | `(3.1, 1.6)` | 地形内部 Z-up 坐标中的 X 长、Y 宽，米。 |
| `DEFAULT_STEP_WIDTH` / `step_width` | `0.15` | 同时控制台阶宽与墙厚 `w`，米。 |
| `DEFAULT_STEP_HEIGHT` | `0.1` | 每级台阶的高度增量，米。 |
| `DEFAULT_PLATFORM_WIDTH` | `0.7` | 中央平台边长，米。 |
| `DEFAULT_CONTAINER_HEIGHT` / `container_height` | `1.5` | 内顶面绝对 Z 坐标，米；不自动取包围盒高度，须高于地形和放置后的 OBJ。 |
| `DEFAULT_OVERLAP_DEPTH` / `overlap_depth` | `None` | 相交量 `δ`；`None` 自动取墙厚的一半，当前为 `0.075` 米；手动设置须满足 `0 < δ < w`。 |
| `DEFAULT_SIMPLIFY_TOLERANCE` / `simplify_tolerance` | `0.001` | 两份输出共同的简化表面容差，米；不表示目标面数。 |
| `DEFAULT_OBJ_PATH` / `obj_path` | `Assets/origin/bunny_flat_bottom.obj` | 原输入 OBJ，须为封闭实体。模型缩放等参数沿用原合成器。 |
| `output_path` / `container_output_path` | 上述两份默认输出 | 允许分别指定路径，须为 `.obj` 且不覆盖输入或彼此。 |
| `up_axis` | `"Y"` | 两份输出统一使用的向上轴，可设为 `"Z"`。 |

四面墙和顶盖使用统一外边界。墙厚为 `w`、向地形内侧进入 `δ` 时，每侧向外增加的尺寸为 `w-δ`，所以外部水平尺寸为 `L+2(w-δ)`、`W+2(w-δ)`。两面长墙横跨完整外长度；两面短墙的目标长度为 `W+δ`，若超过统一外宽度则截到该外边界。墙角交叠发生在同一外轮廓内部，不再额外加长墙头。顶盖使用相同外边界，整体外观为长方体。

墙底位于地形最低高度；墙顶比内顶面高 `δ`。顶盖厚度为 `w`，与墙顶产生实体重叠。改变相交量时，墙、盖的外尺寸同步重算，不能仅独立修改某一面墙的长度。

当前内部水平空间约为 `2.95×1.45` 米，顶盖覆盖范围约为 `3.25×1.75` 米，内顶面 `Z=1.5` 米、外顶面 `Z=1.65` 米。底面最低 `Z=-0.1` 米，因此整体外尺寸约为 `3.25×1.75×1.75` 米。生成后检查简化前后的封闭性、唯一流体空腔，以及外表面是否全部位于统一长方体的六个外平面；外表面与内腔表面不共享顶点是正常结构，不代表墙体没有融合。

Python 调用示例（位置、容器高度均采用内部 Z-up，米）：

```python
from demos.fluid.boundary.scene_container_creator import create_scene

landscape_path, container_path = create_scene(
    container_height=1.5,
    step_width=0.15,          # 台阶宽与墙厚同步改变
    overlap_depth=None,      # 使用 step_width / 2
    simplify_tolerance=0.001,
    up_axis="Y",
)
```

输入和两份输出路径在导出前检查。几何构造或简化检查失败时不开始导出；不会保留本次未简化结果。墙体尺寸与模型尺寸均应在修改配置后重新检查。

运行边界回归检查：

```powershell
conda run -n warplab --no-capture-output python -m unittest discover -s tests/fluid/boundary -t . -v
```

## 地形生成 CLI 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--terrain TYPE` | `wave` | 选择 `flat`、`pyramid_stairs`、`random_grid`、`wave`、`box`、`gap`、`heightfield`；`all` 分别导出七种地形。 |
| `--up-axis Y\|Z` | `Y` | 输出的向上轴；`Z` 保留原始地形坐标。 |
| `--seed INT` | 不固定 | 随机地形种子；用于 `random_grid` 或 `all`，固定整数可复现。 |
| `--heightfield FILE.npy` | 内置隆起和凹坑示例 | 高度场二维数组；用于 `heightfield` 或 `all`。第一维对应 X，第二维对应 Y，数值单位米。 |
| `--ground-z FLOAT` | 最低地表高度减 `0.3` 米 | 高度场底面高度，采用轴转换前的 Z-up 坐标；须严格低于最低地表。用于 `heightfield` 或 `all`。 |
| `-h` / `--help` | — | 查看完整帮助。 |

可复现的全部地形，导出为 Z-up：

```powershell
python demos/fluid/boundary/creator1.py --terrain all --seed 42 --up-axis Z
```

只生成随机地形：

```powershell
python demos/fluid/boundary/creator1.py --terrain random_grid --seed 42
```

使用自定义高度场（文件路径相对于当前工作目录）：

```powershell
python demos/fluid/boundary/creator1.py --terrain heightfield --heightfield height.npy
```

高度数组必须为有限数值二维数组，每个方向至少有 2 个采样点。当前高度场重复生成 6×6 块，每块 6×4 米，总范围为 36×24 米；其他类型各生成一块 6×4 米地形。尺寸、台阶和波浪等参数在 `creator1.py` 对应的 `generate_*()` 函数中修改。

输出目录固定为 `Assets/procedural/`，同名文件会覆盖。`flat` 和 `wave` 是开放表面；箱体类地形和多块高度场为拼接网格，导出过程不执行布尔并集，不能将所有输出直接视为单一封闭实体。

## 场景合成 CLI 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--obj PATH` | `Assets/origin/bunny_flat_bottom.obj` | 输入 OBJ 模型。 |
| `--output PATH` | `Assets/output/stairs_scene_bool.obj` | 布尔结果 OBJ 路径。 |
| `--raw-output PATH` | `Assets/output/stairs_scene.obj` | 原始拼接对照 OBJ 路径。 |
| `--simplify` / `--no-simplify` | 关闭 | 开启或显式关闭简化副本导出。 |
| `--simplify-tolerance FLOAT` | `0.001` | 简化表面容差，单位米，仅启用简化时生效。 |
| `--simplified-output PATH` | 与布尔输出同目录，文件名为 `<stem>_simplified.obj` | 简化副本路径，须与 `--simplify` 一起使用。 |
| `--scale FLOAT` | 约 `6.12274` | 默认兔子宽占平台短边的 95%。OBJ 在 1 米基准平台上的标定倍率，须大于零；开启平台联动后再乘平台短边与基准的比例。仅影响 OBJ。 |
| `--scale-with-platform` / `--no-scale-with-platform` | 开启 | 按中央平台短边等比缩放 OBJ；关闭后直接使用固定的 `--scale` 倍率。 |
| `--yaw FLOAT` | `0.0` | 绕场景竖直 Z 轴旋转的角度，单位度。 |
| `--obj-up-axis Y\|Z` | `Y` | 输入 OBJ 的向上轴。 |
| `--up-axis Y\|Z` | `Y` | 输出场景的向上轴。 |
| `--position X Y Z` | 自动放置 | 指定模型底面支撑中心，使用 Z-up 绝对坐标，单位米；指定后不再叠加 `--sink`，OBJ 尺寸仍按联动规则计算。 |
| `--sink FLOAT` | `0.003` | 自动放置时下沉的深度，单位米，须非负；不随平台尺寸缩放。 |
| `-h` / `--help` | — | 查看完整帮助。 |

表中的默认资产路径相对于本目录；用户传入的相对路径相对于运行命令时的工作目录。输出会覆盖同名文件，输入和各输出路径必须互不相同。

自定义模型和输出的示例：

```powershell
python -m demos.fluid.boundary.scene_creator --obj demos/fluid/boundary/Assets/origin/bunny_flat_bottom.obj --scale 5 --yaw 30 --up-axis Z --simplify --output outputs/boundary/bunny_scene.obj --raw-output outputs/boundary/bunny_raw.obj
```

输入模型必须是封闭、朝向一致的正体积实体；场景合成只焊接重复顶点，不会自动补洞或裁平。默认裁平兔子副本已包含在本目录中，也可用 `--obj` 指定其他模型；其他模型的底面和下沉深度需另行检查。

阶梯尺寸和台阶参数在 `scene_creator.py` 的 `generate_stairs()` 中设置，当前没有对应 CLI 参数。放置计算统一使用 Z-up 坐标，导出时按 `--up-axis` 转换整个场景。

## OBJ 与平台尺寸联动

`REFERENCE_PLATFORM_WIDTH = 1.0` 是代码中的固定标定基准，保留原来 1 米平台上已调好的模型大小。实际平台尺寸从地形最高处向上的水平面读取，以其较短边计算：

```text
实际 OBJ 倍率 = --scale × 当前平台短边 / REFERENCE_PLATFORM_WIDTH
```

当前默认兔子的标定倍率为 `0.95 / 0.15515922 ≈ 6.12274`，在默认朝向下整体宽度占平台短边的 95%；95% 指宽度，不是面积。按支撑面重心放置后，整体包围盒的两侧余量可以不同。更换模型或调整旋转后仍需独立标定 `--scale`。

平台宽度从 1 米改为 0.4 米时，OBJ 三个方向都缩为原来的 0.4 倍；从 1 米改为 2 米时都放大为 2 倍，不改变模型形状。仅改变阶梯外部范围、平台尺寸不变时，OBJ 尺寸不变。单独改变 OBJ 的路径、倍率或旋转角度不会修改地形尺寸或台阶配置。

独立调小摆件，继续跟随平台（当前平台上实际倍率为 `4 × 0.7 = 2.8`）：

```powershell
python -m demos.fluid.boundary.scene_creator --scale 4
```

关闭联动，直接使用固定倍率 2：

```powershell
python -m demos.fluid.boundary.scene_creator --no-scale-with-platform --scale 2
```

Python API 中 `load_and_place_obj()` 和 `create_scene()` 均提供仅限关键字参数 `scale_with_platform=True`；传入 `False` 可恢复固定倍率行为。自动放置将 OBJ 的支撑中心对齐实际平台中心，Z 按顶面高度减去下沉量计算。平底模型按底面三角形面积加权求重心，不受三角形或顶点密度影响；无平底的封闭模型尝试采用穿透高度处截面的面积重心，无法取得截面则采用最低接触顶点的范围中心。显式 `--position` 指定的是底面支撑中心，不隐含下沉；无平底时按最低接触顶点定位。下沉量和简化容差仍为独立的米制值，不联动。运行时会显示平台尺寸、指定倍率和实际倍率。

开启联动时，无法识别有效水平平台会报错。此规则不会自动将不同 OBJ 填满平台，也不限制主动指定的过大倍率；更换模型需用 `--scale` 单独标定，旋转后也应检查是否超出台面。阶梯布尔预处理只在副本中对齐一个 float32 舍入单位内的共用坐标，避免相邻箱体产生微小空腔，原始地形数据保持不变。


## 尺寸记录：
```
vertices, indices = create_mesh_terrain(
    grid_size=(1, 1),                   # (行数, 列数)，这里生成一块。
    block_size=(6.0, 4.0),              # X 长度和 Y 长度，米。
    terrain_types="pyramid_stairs",     # 四周低、中心高的同心阶梯。
    terrain_params={
        "pyramid_stairs": {
            "step_width": 0.5,         # 每圈台阶的水平宽度，米。
            "step_height": 0.1,        # 每圈高度增量，米；当前源码应使用正值。
            "platform_width": 1.0,      # 中心正方形平台的边长，米。
        }
    }
)
```
SDF 烘焙当前默认体素间距为 `0.0075` 米，可通过 PBF 入口的 `--sdf-voxel-size` 调整。

## 容器检查与 SDF 烘焙

`mesh_validation.py` 提供 `load_validate_container_obj(path)`；`sdf_baker.py` 提供
`load_or_bake_sdf(obj_path, cache_dir=..., options=SDFOptions(...), device="cuda:0")`。
烘焙使用 Warp DSL 的最近三角面查询和绕数符号，保存可由 Warp 直接读取的 float32 NanoVDB；两模块不依赖 PBF。

PBF 启动时调用这些 API，默认输入目录为 `PBF/Assets/Container`，缓存输出目录为 `PBF/Assets/Boundary`。将容器结果复制到输入目录后，参见 [PBF 使用说明](../PBF/README.md#静态容器-sdf烘焙缓存与读取)。缓存仅按文件名复用；修改输入或烘焙间距后显式强制重建。本轮不实现 SDF 碰撞。
