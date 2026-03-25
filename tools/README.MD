# 1、环境搭建
```
# 创建指定Python版本的环境
uv venv --python 3.11
# 或者直接指定Python解释器路径（如果3.11已安装）
uv venv --python python3.11`
uv pip install rknn-toolkit2 -i https://pypi.mirrors.ustc.edu.cn/simple/
uv pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```

# 2、下载pth偶像
```
# RT-DETRv2-L
wget https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth
```

# 3、导出onnx模型
```
# RT-DETRv2-L
uv run export_onnx.py --config ../configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml  --resume=./rtdetrv2_r50vd_6x_coco_ema.pth
```

# 4、onnx模型推理
```
uv run onnx_infer.py
```

# 4、导出rknn模型
```
uv run export_rknn.py model.onnx rk3588
```