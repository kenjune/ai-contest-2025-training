import time
import cv2

class Runner():
    def __init__(self, predictor, generator) -> None:
        self.predictor = predictor
        self.generator = generator
        self.pred_result = {}
        self.speed_result = {}

    def run(self) -> None:
        for image_id, image_path in self.generator():
            img = cv2.imread(image_path)
            s = time.perf_counter()
            pred = self.predictor.predict(img)
            runtime = time.perf_counter() - s
            
            self.pred_result[image_id] = pred
            self.speed_result[image_id] = runtime

    def get_result(self) -> tuple:
        return self.pred_result, self.speed_result
