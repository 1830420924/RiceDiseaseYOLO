# 八类别水稻病害目标检测

基于 YOLO11n、PyTorch 和 Ultralytics 的独立 PyCharm 项目。当前先建立原始模型基线，后续再按实验结果加入改进模型和检测界面。

**项目搭建不包含实际训练。训练由你在 PyCharm 中手动运行。**

## 在 PyCharm 中运行

1. 选择“打开”，打开整个 `RiceDiseaseYOLO` 项目文件夹。
2. 在“设置 → 项目 → Python 解释器”选择已有 Conda 环境的解释器：
   `D:\Miniconda3\envs\pytorch_gpu\python.exe`。
   当前环境已安装项目依赖，不需要重新创建环境或升级 PyTorch。
3. 右键运行 `scripts/check_environment.py`，检查环境；右键运行 `scripts/check_dataset.py`，检查准备好的八类数据。
4. 首次建议使用顶部运行配置 **04 - 3轮试训**，确认自己能完整运行训练。也可给 `train_baseline.py` 添加参数 `--epochs 3 --name smoke_test`。
5. 试训完成后选择 **05 - 50轮基线**。直接右键运行 `train_baseline.py` 也会按 `configs/project.yaml` 中的默认值启动训练。
6. 训练完成后运行 `evaluate.py`，默认评估最新训练结果的 `best.pt`，使用保留的 `test` 集。

项目提供的 `.run/` 运行配置只会在你点击运行时执行。路径计算以项目目录为准，不依赖 PyCharm 当前工作目录。项目内 `.runtime/` 保存库配置和缓存。

## 数据与类别

来源是共享对话中选定的 [Rice Leaf Spot Disease Annotated Dataset](https://www.kaggle.com/datasets/hadiurrahmannabil/rice-leaf-spot-disease-annotated-dataset)。以本地原始 `data.yaml` 为准，保留原始 ID 和名称：

| ID | 原始英文名称 | 中文显示名称 |
|---|---|---|
| 0 | Bacterial_Leaf_Blight | 白叶枯病 |
| 1 | Brown_Spot | 褐斑病 |
| 2 | HealthyLeaf | 健康叶片 |
| 3 | Leaf_Blast | 叶瘟病 |
| 4 | Leaf_Scald | 叶焦病 |
| 5 | Narrow_Brown_Leaf_Spot | 窄褐斑病 |
| 6 | Neck_Blast | 穗颈瘟 |
| 7 | Rice_Hispa | 稻铁甲虫危害 |

这是八个检测类别，包含健康状态、六个病害类别及一个虫害类别，不能统称为八种病害。中文名称用于显示，训练始终保留原始英文名称。

原始目录：`D:\郭佳丽\yolo官方数据集\archive\RiceLeafAnnotatedDataset`。
项目使用独立副本 `datasets/rice/`；保留原 Train/Valid/Test 划分，只扫描这三个集合，不混入源目录中额外的西瓜子目录。

| 集合 | 图片 | 标注目标 | 空标签图片 |
|---|---:|---:|---:|
| train | 2,767 | 5,466 | 17 |
| valid | 400 | 656 | 5 |
| test | 400 | 696 | 2 |
| 合计 | 3,567 | 6,818 | 24 |

实际标签含 **581 个五列检测框和 6,237 个分割多边形**。准备程序将每个多边形的 x/y 极值转为轴对齐外接框，使最终标签统一为 `class x_center y_center width height`。这一步保留原标注的目标单位，不把病斑自动变成整叶框，也不合并多个目标。原始图像和标签不作修改。

24 个空标签保留为无标注目标的背景图片，不擅自补为健康。结构检查不能证明原始标签语义完全准确，可查看 `reports/label_preview.jpg` 抽查目标范围。完整统计见 `reports/source_audit.md`。未发现跨集合字节完全相同的图片；131 组跨集合源名前缀碰撞均伴有类别差异，不能据此认定同源或数据泄漏。

## 训练设置

修改 `configs/project.yaml` 中的 `train` 部分即可：

| 参数 | 默认值 |
|---|---|
| model | yolo11n.pt |
| epochs | 50 |
| imgsz | 640 |
| batch | 8 |
| workers | 2 |
| seed | 42 |
| device | auto（可用 GPU 优先，否则 CPU） |
| patience | 30 |

`epochs=50` 是最大轮数，启用早停时可能提前结束。当前默认以 `batch=8`、`workers=2` 尝试提高吞吐，实际速度以完整轮次耗时为准。显存不足可将 batch 改为 4、2 或 1；Windows 多进程读取出错或系统内存不足时可将 workers 改回 0。需要固定设备时将 device 改为 `"0"` 或 `cpu`。

参数在每次启动训练时读取，正在运行的训练不会自动切换。使用新参数重新运行会建立独立实验目录，从官方初始化权重开始训练，并保留已有结果。

`weights/yolo11n.pt` 已备好官方通用检测预训练权重及来源、校验和记录。它用于初始化，尚不具备本项目的八类识别能力。训练输出在 `runs/train/` 中按实验建立独立目录，其中：

- `weights/best.pt`：根据验证集选出的八类水稻权重。
- `weights/last.pt`：最后一轮权重。
- `results.csv`、训练曲线、混淆矩阵：Ultralytics 训练结果。
- 项目摘要文件：实际配置、运行目录和指标。

基线训练使用 `train` 和 `valid`；`test` 仅在训练后单独评估。请先保留一份完整基线结果，再开展改进模型和消融实验。项目搭建检查不会产生真实训练精度。

当前 Ultralytics 版本启用 `amp: true` 时，首次正式训练的自动混合精度自检可能另外下载通用自检权重。需要离线训练时可将 `amp` 改为 `false`，GPU 显存占用可能增加。

## 常用命令（PyCharm 终端）

以下假设终端已经选用上述解释器，且位于项目目录：

```powershell
python scripts/check_environment.py
python scripts/check_dataset.py
python train_baseline.py --dry-run
```

上述命令只做检查。**以下训练命令需要你自行运行：**

```powershell
python train_baseline.py --epochs 3 --name smoke_test
python train_baseline.py --name baseline
```

评估默认最新训练权重，也可显式指定：

```powershell
python evaluate.py --split test
python evaluate.py --weights "runs/train/实际训练目录/weights/best.pt" --split test
python predict.py --weights "runs/train/实际训练目录/weights/best.pt" --source "D:/需要检测的图片.jpg"
```

评估保存总体与各类 Precision、Recall、mAP50 和 mAP50-95；预测保存带框图片、YOLO 文本标签及 JSON 摘要。没有检出目标不等于健康。`evaluate.py` 和 `predict.py` 需要八类水稻训练权重，不能用通用预训练权重代替。

若以后更换原数据位置，修改 `configs/project.yaml` 的 `source_root`，再运行：

```powershell
python scripts/prepare_dataset.py
python scripts/check_dataset.py
python scripts/visualize_labels.py
```

准备程序可重复运行；源数据或已准备数据发生变化时会提示，使用 `--force` 才归档旧数据并重新生成，旧目录保存为 `datasets/rice.archive-日期时间-编号/`。不要直接用原始混合标签目录作为检测训练入口。移动整个项目后，调整原数据路径并运行 `python scripts/prepare_dataset.py --force`，以更新 `data.yaml` 的绝对路径。

## 项目结构

```text
RiceDiseaseYOLO/
├── configs/project.yaml       路径及训练参数
├── datasets/rice/             已准备的八类检测数据
├── rice_yolo/                 配置、数据和模型实现
├── scripts/                   环境、数据准备、核查和标注可视化
├── models/                    后续改进模型预留目录
├── app/                       后续检测界面预留目录
├── weights/                   官方初始化权重与来源记录
├── reports/                   审计、环境、数据检查和预览
├── runs/                      训练、评估及检测结果
├── tests/                     不启动真实训练的流程测试
├── .run/                      PyCharm 运行配置
├── train_baseline.py           基线训练入口
├── evaluate.py                测试集评估入口
└── predict.py                 本地图片检测入口
```

## 开发检查

```powershell
python -B -m unittest discover -s tests -v
```

测试采用合成数据和模拟模型，验证转换规则、数据路径、训练调用参数和输出逻辑，不代表真实模型性能。

## 参考

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [YOLO 检测数据格式](https://docs.ultralytics.com/datasets/detect/)
- [训练参数](https://docs.ultralytics.com/modes/train/)

