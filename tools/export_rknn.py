import sys
import cv2
import numpy as np
from rknn.api import RKNN
# 注释掉 matplotlib 相关导入
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

DATASET_PATH = 'datas/dataserts.txt'
DEFAULT_RKNN_PATH = 'model.rknn'
DEFAULT_QUANT = False
quantized_algorithm='normal'          # 使用MMSE算法提高量化精度： normal 、 kl_divergence 、 mmse
DEFAULT_IMAGE_PATH = 'datas/1920_1080.jpeg'  # 默认测试图片路径
DEFAULT_OUTPUT_IMAGE_PATH = 'result_detection.jpg'  # 默认输出图片路径

# COCO数据集类别名称（80类）
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# COCO数据集类别ID映射（如果模型输出的是COCO原始ID，需要映射到0-79）
COCO_ID_TO_INDEX = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
    22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
    35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
    46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
    67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
    80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
}

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)] [image_path(optional)] [output_image_path(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1103, rv1106, rv1126b, rv1109, rv1126, rk1808]")
        print("       dtype choose from [i8, fp] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1103, rv1106, rv1126b]")
        print("       dtype choose from [u8, fp] for [rv1109, rv1126, rk1808]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'u8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH
        
    # 获取图片路径参数
    if len(sys.argv) > 5:
        image_path = sys.argv[5]
    else:
        image_path = DEFAULT_IMAGE_PATH
    
    # 获取输出图片路径参数
    if len(sys.argv) > 6:
        output_image_path = sys.argv[6]
    else:
        output_image_path = DEFAULT_OUTPUT_IMAGE_PATH

    return model_path, platform, do_quant, output_path, image_path, output_image_path

def preprocess_image(image_path, input_size=None):
    """
    预处理图片，使其符合模型输入要求
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image from {image_path}")
        return None, None
    
    # 保存原始图片用于绘制
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = (img.shape[1], img.shape[0])  # (width, height)
    
    # BGR转RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 调整大小（如果需要）
    if input_size is not None:
        img = cv2.resize(img, (input_size[1], input_size[0]))
    
    # 归一化
    img = img.astype(np.float32)
    
    # 添加batch维度
    img = np.expand_dims(img, axis=0)
    
    return img, original_img, original_size

def inference_with_rknn(rknn, image_path, input_size=None):
    """
    使用RKNN模型进行推理
    """
    print('--> Loading image for inference')
    input_size = [640,640]
    
    # 预处理图片
    img, original_img, original_size = preprocess_image(image_path, input_size)
    if img is None:
        return None, None, None
    
    print(f"Input shape: {img.shape}")
    
    # 推理
    print('--> Running inference')
    outputs_data = rknn.inference(inputs=[img])
    
    return outputs_data, original_img, original_size

def get_class_name(class_id):
    """
    获取COCO类别名称
    """
    # 如果class_id在0-79范围内，直接使用
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    # 如果是COCO原始ID（1-90），尝试映射
    elif class_id in COCO_ID_TO_INDEX:
        return COCO_CLASSES[COCO_ID_TO_INDEX[class_id]]
    else:
        return f'Class_{class_id}'

def draw_detections(image, detections, confidence_threshold=0.5, save_path='result.jpg'):
    """
    在图片上绘制检测结果（仅使用OpenCV）
    """
    # 复制图片用于绘制
    img_draw = image.copy()
    
    # 定义颜色映射
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3)).tolist()
    
    # 过滤检测结果
    filtered_detections = [det for det in detections if det[1] >= confidence_threshold]
    
    print(f"\nDrawing {len(filtered_detections)} detections on image...")
    
    # 绘制每个检测框
    for det in filtered_detections:
        class_id, confidence, x1, y1, x2, y2 = det
        
        # 转换为整数坐标
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, img_draw.shape[1]))
        y1 = max(0, min(y1, img_draw.shape[0]))
        x2 = max(0, min(x2, img_draw.shape[1]))
        y2 = max(0, min(y2, img_draw.shape[0]))
        
        # 选择颜色
        color_idx = class_id % len(colors)
        color = colors[color_idx]
        
        # 绘制矩形框
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本
        class_name = get_class_name(class_id)
        label = f'{class_name}: {confidence:.2f}'
        
        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # 绘制文本背景
        cv2.rectangle(img_draw, (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width + 5, y1), color, -1)
        
        # 绘制文本
        cv2.putText(img_draw, label, (x1 + 2, y1 - baseline - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 添加图例
    if len(filtered_detections) > 0:
        legend_y = 30
        legend_x = 10
        detected_classes = set([det[0] for det in filtered_detections])
        
        # 半透明背景
        overlay = img_draw.copy()
        cv2.rectangle(overlay, (legend_x - 5, legend_y - 25), 
                     (legend_x + 200, legend_y + len(detected_classes) * 25 + 5), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img_draw, 0.3, 0, img_draw)
        
        cv2.putText(img_draw, "Detected Classes:", (legend_x, legend_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, class_id in enumerate(detected_classes):
            class_name = get_class_name(class_id)
            color_idx = class_id % len(colors)
            color = colors[color_idx]
            cv2.rectangle(img_draw, (legend_x, legend_y + i * 20), 
                         (legend_x + 15, legend_y + i * 20 + 15), color, -1)
            cv2.putText(img_draw, class_name, (legend_x + 20, legend_y + i * 20 + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # 保存图片
    cv2.imwrite(save_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
    print(f"Detection result saved to: {save_path}")
    
    return img_draw

def print_inference_results(outputs_data, original_img=None, save_path=None):
    """
    打印推理结果（COCO数据集版本）
    """
    print('--> Inference results:')
    
    # 置信度过滤
    confidence_threshold = 0.5
    
    # 打印所有输出的基本信息
    for i, output in enumerate(outputs_data):
        print(f"\nOutput {i}:")
        print(f"  Shape: {output.shape}")
        print(f"  Dtype: {output.dtype}")
        
        # 安全地计算统计信息
        try:
            print(f"  Min: {np.min(output):.4f}, Max: {np.max(output):.4f}, Mean: {np.mean(output):.4f}")
        except:
            print(f"  Min: {np.min(output)}, Max: {np.max(output)}, Mean: {np.mean(output)}")
    
    # 假设输出顺序固定
    if len(outputs_data) >= 3:
        classes = outputs_data[0]  # [1, 300]
        boxes = outputs_data[1]    # [1, 300, 4] 或 [1, 300, 6]
        scores = outputs_data[2]   # [1, 300]
        
        print("\n" + "="*80)
        print(f"COCO Detection Results (Confidence > {confidence_threshold}):")
        print("="*80)
        
        # 存储检测结果用于绘制
        detections = []
        
        try:
            # 提取数据并确保是一维数组
            if scores is not None and len(scores) > 0:
                scores_data = scores[0] if len(scores.shape) > 1 else scores
                if len(scores_data.shape) > 1:
                    scores_data = scores_data.flatten()
                
                # 提取类别数据
                classes_data = None
                if classes is not None and len(classes) > 0:
                    classes_data = classes[0] if len(classes.shape) > 1 else classes
                    if len(classes_data.shape) > 1:
                        classes_data = classes_data.flatten()
                
                # 创建布尔掩码
                mask = scores_data > confidence_threshold
                valid_indices = np.where(mask)[0]
                
                if len(valid_indices) == 0:
                    print(f"No detections with confidence > {confidence_threshold}")
                    max_conf = np.max(scores_data) if len(scores_data) > 0 else 0
                    min_conf = np.min(scores_data) if len(scores_data) > 0 else 0
                    print(f"Max confidence: {float(max_conf):.4f}")
                    print(f"Min confidence: {float(min_conf):.4f}")
                else:
                    print(f"Found {len(valid_indices)} detections:\n")
                    print(f"{'No.':<5} {'Class':<20} {'Confidence':<12} {'Box Coordinates':<40}")
                    print("-" * 80)
                    
                    for idx, idx_val in enumerate(valid_indices):
                        try:
                            confidence = float(scores_data[idx_val])
                            
                            class_id = -1
                            if classes_data is not None and idx_val < len(classes_data):
                                class_id = int(classes_data[idx_val])
                            
                            class_name = get_class_name(class_id)
                            
                            if boxes is not None and len(boxes) > 0:
                                if len(boxes.shape) == 3 and idx_val < boxes.shape[1]:
                                    box = boxes[0][idx_val]
                                    if len(box) >= 4:
                                        coords = [float(box[j]) for j in range(4)]
                                        print(f"{idx+1:<5} {class_name:<20} {confidence:.4f}      [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}, {coords[3]:.2f}]")
                                        detections.append([class_id, confidence] + coords)
                                    else:
                                        print(f"{idx+1:<5} {class_name:<20} {confidence:.4f}      {box}")
                                else:
                                    print(f"{idx+1:<5} {class_name:<20} {confidence:.4f}")
                            else:
                                print(f"{idx+1:<5} {class_name:<20} {confidence:.4f}")
                        
                        except Exception as e:
                            print(f"Error processing detection {idx}: {e}")
                            continue
                    
                    # 打印统计信息
                    print("-" * 80)
                    print(f"Statistics:")
                    print(f"  Total detections: {len(scores_data)}")
                    print(f"  Detections > {confidence_threshold}: {len(valid_indices)}")
                    print(f"  Max confidence: {float(np.max(scores_data)):.4f}")
                    print(f"  Min confidence: {float(np.min(scores_data)):.4f}")
                    print(f"  Mean confidence: {float(np.mean(scores_data)):.4f}")
                    
                    # 统计每个类别的检测数量
                    if len(detections) > 0:
                        class_counts = {}
                        for det in detections:
                            class_id = det[0]
                            class_name = get_class_name(class_id)
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        print(f"\nDetected classes summary:")
                        for class_name, count in class_counts.items():
                            print(f"  {class_name}: {count}")
            else:
                print("No valid scores data found")
                
        except Exception as e:
            print(f"Error processing detection results: {e}")
            import traceback
            traceback.print_exc()
        
        # 绘制检测结果
        if original_img is not None and len(detections) > 0 and save_path:
            print("\n" + "="*80)
            print("Drawing detection results on image...")
            print("="*80)
            draw_detections(original_img, detections, confidence_threshold, save_path)
        elif original_img is not None and len(detections) == 0:
            print("\nNo detections to draw")
        elif original_img is None:
            print("\nNo original image provided for drawing")
        elif not save_path:
            print("\nNo save path provided for drawing")
    
    else:
        print("Warning: Expected 3 outputs, but got", len(outputs_data))

if __name__ == '__main__':
    model_path, platform, do_quant, output_path, image_path, output_image_path = parse_arg()
    
    # 可选：指定模型输入尺寸（根据你的实际模型修改）
    input_size = None  # 设为None则不做resize
    
    # Create RKNN object
    rknn = RKNN(verbose=False)
    
    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[123.675, 116.28, 103.53]], 
                std_values=[[58.395, 57.12, 57.375]], 
                target_platform=platform,
                quantized_algorithm=quantized_algorithm)
    print('done')
    
    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')
    
    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')
    
    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
    
    # 加载模型进行推理
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')
    
    # 执行推理
    outputs, original_img, original_size = inference_with_rknn(rknn, image_path, input_size)
    
    if outputs is not None:
        # 打印推理结果并保存绘制图片
        print_inference_results(outputs, original_img, output_image_path)
    
    # Release
    rknn.release()