# UPDATE

## 20260409 v0.2
1. 新增g1的`ui`和`kimodo_gen`npz导出额外包含`root_position, root_rot, dof, fps`
2. 对`kimodo_gen`新增可选项`--tiny-npz`仅导出上述几个key

## 20260318 v0.1
1. 备注了注意事项[NOTE.md](docs/NOTE.md)，安装和启动方法
2. 修改了默认的`Gradio`缓存存储位置在当前项目的 `./tmp` 下面，不再去 `/tmp/gradio`（服务器上可能和别人的冲突了）
