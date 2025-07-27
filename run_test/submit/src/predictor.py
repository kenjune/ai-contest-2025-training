import os
import platform


import torch
import numpy as np
from ultralytics import YOLO


def get_device(force_cpu: bool = False, use_multi_gpus: bool = True):
    """Finds the best device to run the training."""
    if force_cpu:
        return 'cpu'

    if torch.cuda.is_available():
        if use_multi_gpus:
            device = list(range(torch.cuda.device_count()))
        else:
            device = [0,]
    else:
        if platform.processor() == 'arm':
            device = 'mps'
        else:
            device = 'cpu'

    return device


class Predictor(object):
    @classmethod
    def get_model(cls, model_path):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            bool: The return value. True for success, False otherwise.
        """
        # load some model(s)
        full_model_path = os.path.join(model_path, "best.pt")
        cls.model = YOLO(full_model_path)
        cls.device = get_device()
        return True

    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input (np.ndarray): image data from cv2.imread

        Returns:
            dict: Inference for the given input.
        """
        results = cls.model.predict(
            source=input, 
            imgsz=640,
            max_det=5,  # Limit to 5 detections.
            save=False,
            device=cls.device,
            verbose=False
        )
        

        if len(results) == 0:
            print(f"[WARNING] Nothing detected.")
            return []

        if len(results) != 1:
            print(f"[WARNING] Expected one image only, but got {len(results)}. Using the first one.")

        result = results[0]
        boxes = result.cpu().numpy().boxes.xywh
        confs = result.cpu().numpy().boxes.conf
        categories = result.cpu().numpy().boxes.cls

        has_max_5_elements = (len(boxes) <= 5) and \
            (len(boxes) == len(confs)) and \
            (len(boxes) == len(categories))

        if not has_max_5_elements:
            print(f"[WARNING] Expected at most 5 predictions, but got {len(boxes)}. Using top 5 ones.")
            sorting_idxes = np.argsort(confs)[::-1][:5]
            boxes = boxes[sorting_idxes]
            confs = confs[sorting_idxes]
            categories = categories[sorting_idxes]

        predict_list = []
        for category_id, box, score in zip(categories, boxes, confs):
            # xywh is given as center_x, center_y -> transform to top-left.
            box[0] -= box[2] * 0.5
            box[1] -= box[3] * 0.5
            box = box.round(0).astype(int)
            predict_list.append({
                "category_id": int(category_id + 1),  # Class IDs begin at index 1.
                "bbox": box.tolist(),
                "score": float(score)
            })

        return predict_list
