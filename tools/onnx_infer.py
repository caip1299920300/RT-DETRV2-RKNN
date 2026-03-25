"""
RT-DETR ONNX推理代码
模型输入: [1, 3, input_size, input_size] 图像
模型输出: labels, boxes, scores
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple
import time

class RTDETR_ONNX:
    def __init__(self, model_path: str, input_size: int = 640, 
                 original_size: Tuple[int, int] = (1920, 1080)):
        """
        初始化RT-DETR ONNX模型
        
        Args:
            model_path: ONNX模型路径
            input_size: 模型输入尺寸（默认640）
            original_size: 原始图像尺寸（宽, 高），默认1920x1080
        """
        self.input_size = input_size
        self.original_width, self.original_height = original_size
        
        # 创建ONNX Runtime会话
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"模型输入: {self.input_name}")
        print(f"模型输出: {self.output_names}")
        print(f"原始尺寸: {self.original_width}x{self.original_height}")
        print(f"模型输入尺寸: {self.input_size}x{self.input_size}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 原始图像 (H, W, C) BGR格式
            
        Returns:
            预处理后的图像 (1, 3, H, W)
        """
        # 调整尺寸到模型输入大小
        resized = cv2.resize(image, (self.input_size, self.input_size))
        
        # BGR转RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 归一化（根据RT-DETR的训练配置）
        # 使用ImageNet的均值和标准差
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (rgb / 255.0 - mean) / std
        
        # 转换维度 (H, W, C) -> (C, H, W) -> (1, C, H, W)
        input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
        
        return input_tensor
    
    def postprocess(self, outputs: List[np.ndarray], 
                   conf_threshold: float = 0.5, 
                   nms_threshold: float = 0.5,
                   num_flag: bool = False) -> List[dict]:
        """
        后处理
        
        Args:
            outputs: 模型输出 [labels, boxes, scores]
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
            
        Returns:
            检测结果列表，每个元素为 {'label': int, 'score': float, 'bbox': [x1, y1, x2, y2]}
        """
        labels, boxes, scores = outputs
        
        # 转换数据类型
        labels = labels.flatten().astype(np.int32)
        boxes = boxes.reshape(-1, 4)  # [N, 4]
        scores = scores.flatten()
        
        # 过滤低置信度检测
        keep = scores > conf_threshold
        labels = labels[keep]
        boxes = boxes[keep]
        scores = scores[keep]
        
        if len(boxes) == 0:
            return []
        
        # 坐标转换：模型输出的是 [x1, y1, x2, y2] 格式
        # 如果模型输出的是归一化坐标，需要转换
        # 这里假设输出已经是绝对坐标（因为固定了原始尺寸）
        bboxes = boxes.copy()
        
        # 如果需要从归一化坐标转换（根据实际情况选择）
        # bboxes[:, 0] = boxes[:, 0] * self.original_width   # x1
        # bboxes[:, 1] = boxes[:, 1] * self.original_height  # y1
        # bboxes[:, 2] = boxes[:, 2] * self.original_width   # x2
        # bboxes[:, 3] = boxes[:, 3] * self.original_height  # y2
        
        # 不使用nms
        if not num_flag:
            results = []
            for idx in range(len(labels)):
                results.append({
                    'label': int(labels[idx]),
                    'score': float(scores[idx]),
                    'bbox': bboxes[idx].tolist()
                })
            return results

        # 执行NMS
        indices = self.nms(bboxes, scores, nms_threshold)
        
        # 整理结果
        results = []
        for idx in indices:
            results.append({
                'label': int(labels[idx]),
                'score': float(scores[idx]),
                'bbox': bboxes[idx].tolist()
            })
        
        return results
    
    def nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        非极大值抑制
        
        Args:
            boxes: 边界框数组 [N, 4]
            scores: 置信度数组 [N]
            iou_threshold: IOU阈值
            
        Returns:
            保留的索引列表
        """
        if len(boxes) == 0:
            return []
        
        # 按置信度降序排序
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # 计算当前框与其他框的IOU
            ious = self.compute_iou(boxes[i], boxes[order[1:]])
            
            # 保留IOU小于阈值的框
            inds = np.where(ious <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    @staticmethod
    def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        计算IOU
        
        Args:
            box: 单个边界框 [4]
            boxes: 多个边界框 [N, 4]
            
        Returns:
            IOU数组 [N]
        """
        # 计算交集
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # 计算并集
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - intersection
        
        return intersection / (union + 1e-6)
    
    def inference(self, image: np.ndarray, conf_threshold: float = 0.5, 
                  nms_threshold: float = 0.5) -> List[dict]:
        """
        完整推理流程
        
        Args:
            image: 原始图像 (H, W, C)
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
            
        Returns:
            检测结果列表
        """
        # 预处理
        input_tensor = self.preprocess(image)
        
        # 推理
        start_time = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        inference_time = (time.time() - start_time) * 1000
        
        print(f"推理时间: {inference_time:.2f}ms")
        
        # 后处理
        results = self.postprocess(outputs, conf_threshold, nms_threshold)
        
        return results
    
    def visualize(self, image: np.ndarray, results: List[dict], 
                  class_names: List[str] = None, save_path: str = None):
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            results: 检测结果列表
            class_names: 类别名称列表
            save_path: 保存路径，如果为None则显示图像
        """
        vis_image = image.copy()
        
        for det in results:
            bbox = det['bbox']
            label = det['label']
            score = det['score']
            
            # 获取类别名称
            if class_names and label < len(class_names):
                label_name = class_names[label]
            else:
                label_name = f'Class_{label}'
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            text = f'{label_name}: {score:.2f}'
            cv2.putText(vis_image, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"结果已保存到: {save_path}")
        else:
            cv2.imshow('RT-DETR Detection', vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    """示例使用"""
    # 模型配置
    model_path = 'model.onnx'
    input_size = 640
    original_size = (1920, 1080)  # 与导出时保持一致
    
    # 初始化模型
    model = RTDETR_ONNX(model_path, input_size, original_size)
    
    # 读取图像
    image_path = 'datas/1920_1080.jpeg'
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    print(f"原始图像尺寸: {image.shape}")
    
    # 推理
    results = model.inference(image, conf_threshold=0.5, nms_threshold=0.5)
    
    # 打印结果
    print(f"\n检测到 {len(results)} 个目标:")
    for i, det in enumerate(results):
        print(f"{i+1}. 类别: {det['label']}, 置信度: {det['score']:.3f}, "
              f"边界框: {det['bbox']}")
    
    # 可视化结果
    # 如果有类别名称，可以传入
    class_names = [
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
    model.visualize(image, results, class_names, save_path='result.jpg')


if __name__ == '__main__':
    main()