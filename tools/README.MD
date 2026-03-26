# 1、环境搭建
```
# 创建指定Python版本的环境
uv venv --python 3.11
# 或者直接指定Python解释器路径（如果3.11已安装）
uv venv --python python3.11
uv pip install rknn-toolkit2 -i https://pypi.mirrors.ustc.edu.cn/simple/
uv pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```

# 2、下载pth偶像
```
# Base models
# wget https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth
# Models of discrete sampling
wget https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_dsp_3x_coco.pth
```

# 3、导出onnx模型
```
# 注意事项
（1）使用Base models时。因为rknn不支持GridSample双线性插值算子，所以得切换到discrete最近邻差值。
configs/rtdetrv2/include/rtdetrv2_r50vd.yml2目录中涉及到 RTDETRTransformerv2的cross_attn_method 改成 discrete，后续rknn使用才不会报错
（2）Models of discrete sampling（RT-DETRv2-S_dsp等） 不需要任何处理。
# RT-DETRv2-L
uv run export_onnx.py --config ../configs/rtdetrv2/rtdetrv2_r18vd_dsp_3x_coco.yml  --resume=./rtdetrv2_r18vd_dsp_3x_coco.pth
```

# 4、onnx模型推理
```
uv run onnx_infer.py
```

# 4、导出rknn模型
```
uv run export_rknn.py model.onnx rk3588
```
