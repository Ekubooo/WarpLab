# Nvidia Warp Note

## Fluid
- 完整仿真步进
    ```
    Example.__init__
    └─ HashGrid(25,25,25)
        只创建内部结构，尚无粒子

    每个物理子步
    ├─ grid.build(x, h)
    │    根据最新位置重建桶和索引
    ├─ compute_density(grid.id, ...)
    │    point_id 重排线程
    │    query 获取候选邻居
    ├─ get_acceleration(grid.id, ...)
    │    再次查询同一位置状态
    ├─ kick
    └─ drift
        修改位置，使当前网格失效

    下一个子步
    └─ 再次 grid.build(...)
    ```