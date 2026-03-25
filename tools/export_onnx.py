"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
命令：python export_onnx.py --config ../configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml  --resume=./datas/rtdetrv2_r50vd_6x_coco_ema.pth
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn

from src.core import YAMLConfig, yaml_utils


def main(args, ):
    """main
    """
    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    update_dict.update({k: v for k, v in args.__dict__.items() \
                        if k not in ['update', ] and v is not None})
    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            self.orig_target_sizes = torch.tensor([[1920,1080]])

        def forward(self, images):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, self.orig_target_sizes)
            return outputs

    model = Model()
    model.eval()
    data = torch.rand(1, 3, args.input_size, args.input_size)
    _ = model(data)
    

    # dynamic_axes = {
        # 'images': {0: 'N', }
    # }
    dynamic_axes = None

    torch.onnx.export(
        model,
        data,
        args.output_file,
        input_names=['images'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=dynamic_axes,
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(args.output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx
        import onnxsim
        from onnxsim import simplify
        
        # 新版本的使用方式
        model = onnx.load(args.output_file)
        
        test_input_shapes = {
            'images': [1, 3, 640, 640]  # 测试用的输入形状
        }
        onnx_model_simplify, check = simplify(
            model,
            test_input_shapes=test_input_shapes
        )
        
        if check:
            onnx.save(onnx_model_simplify, args.output_file)
            print(f"Simplification save { args.output_file}")
        else:
            print("Simplification failed")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--resume', '-r', type=str)
    parser.add_argument('--output_file', '-o', type=str, default='model.onnx')
    parser.add_argument('--input_size', '-s', type=int, default=640)
    parser.add_argument('--check', action='store_true', default=False)
    parser.add_argument('--simplify', action='store_true', default=False)
    parser.add_argument('--update', '-u', nargs='+', help='update yaml config')

    args = parser.parse_args()

    main(args)