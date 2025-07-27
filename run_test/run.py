import os
import sys
import argparse
import json
from src.generator import ImageGenerator
from src.runner import Runner
from src.validator import DictValidator

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exec-dir', default = './submit/src')
    parser.add_argument('--input-data-dir', default = './dataset')
    parser.add_argument('--input-name', default = 'annotations/custom_val.json')
    parser.add_argument('--tmp-eval-name', default = 'tmp_val.json')
    parser.add_argument('--result-dir', default = './results')
    parser.add_argument('--result-name', default = 'result.json')
    args = parser.parse_args()

    return args

def calc_score_speed(input_coco_gt, input_predict_dict, evaluate_maxdets=[1, 3, 5]):
    cocoDt = input_coco_gt.loadRes(input_predict_dict["predicts"])
    cocoEval = COCOeval(input_coco_gt, cocoDt, iouType='bbox')
    cocoEval.params.maxDets = evaluate_maxdets
    cocoEval.evaluate()
    cocoEval.accumulate()
    
    precision = cocoEval.eval['precision']
    area_idx = cocoEval.params.areaRngLbl.index('all')
    max_det_idx = cocoEval.params.maxDets.index(evaluate_maxdets[-1])
    valid_mask = precision[..., area_idx, max_det_idx] > -1
    
    output_score = np.mean(precision[..., area_idx, max_det_idx][valid_mask])
    
    output_speed = float(np.mean(input_predict_dict["speed"]))
    output_speed = round(output_speed, 9)
    return output_score, output_speed

def main():
    # parse the arguments
    args = parse_args()

    # set the input data and instanciate the generator
    input_path = os.path.join(args.input_data_dir, args.input_name)
    params = {'root_dir': os.path.abspath(os.path.join(args.input_data_dir, 'images'))} # 推論対象の画像データが存在するルートディレクトリ
    data_format = {'keys': {"category_id", "bbox", "score"}, 
                   'dtype': {"category_id": int, "bbox": list, "score": float}, 
                   "categories": [i for i in range(1,92)],
                   "k": 5}
    
    with open(input_path) as f:
        input_data = json.load(f)
    
    # pycocotoolsによる評価のために、iscrowdとareaを導入 
    runtime_test_json = []
    for json_ in input_data['annotations']:
        json_["iscrowd"] = 0
        json_["area"] = int(json_["bbox"][2]*json_["bbox"][3])
        runtime_test_json.append(json_)
    input_data['annotations'] = runtime_test_json
    
    with open(args.tmp_eval_name, mode="w") as f:
        json.dump(input_data, f, indent=2)
    
    cocoGt = COCO(args.tmp_eval_name)
    
    generator = ImageGenerator(input_data=input_data, params=params)
    result_dir = os.path.abspath(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)

    # 予測器のインスタンス化
    exec_dir = os.path.abspath(args.exec_dir)
    sys.path.append(exec_dir)
    from predictor import Predictor
    os.chdir(args.exec_dir)
    model_path = '../model'
    model_flag = Predictor.get_model(model_path)
    if not model_flag:
        raise Exception('Could not load the model.')

    # run the inference
    runner = Runner(predictor=Predictor, generator=generator)
    runner.run()
    pred_result, speed_result = runner.get_result()
    
    # 予測チェッカー
    predict_checker = DictValidator(data_format=data_format)
    predict_checker.validate(pred_result)
    
    predicts_dict = {"predicts": [], "speed": []}
    id_num = 1
    for image_json in input_data["images"]:
        image_id = image_json["id"]
        pred_per_image, per_speed = pred_result[image_id], speed_result[image_id]

        for pred in pred_per_image:
            pred.update(
                id = id_num,  # 自動採番
                image_id  = image_id
                )
            predicts_dict["predicts"].append(pred)
            id_num += 1
        predicts_dict["speed"].append(per_speed)
    
    score, speed = calc_score_speed(cocoGt, predicts_dict, evaluate_maxdets=[1, 3, 5])
    
    results = {
        "score": score,
        "prediction_second" : speed
        }
    
    with open(os.path.join(result_dir, args.result_name), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
        