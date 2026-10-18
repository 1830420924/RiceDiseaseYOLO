将需要检测的 JPG、PNG 等图片放在此文件夹中，训练完成后可直接运行项目根目录的 predict.py。

默认自动寻找 runs/train/ 中最近保存的八类水稻 best.pt。也可在 predict.py 顶部设置 WEIGHTS_PATH 和 SOURCE，或通过 --weights、--source 指定。

