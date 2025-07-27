from argparse import ArgumentParser
import json
from sklearn.model_selection import train_test_split

def split_coco_annotations(
    load_anno_path: str,
    save_anno_path: str,
    val_ratio: float,
    seed: int = 42):
    with open(load_anno_path) as f:
        train_json = json.load(f)
    
    images = train_json['images']
    annotations = train_json['annotations']
    categories = train_json['categories']
    
    # データ分割
    _, val_images = train_test_split(
        images,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )

    # 画像 ID の抽出
    val_ids = {img['id'] for img in val_images}

    # アノテーションの振り分け
    annotations_val = [ann for ann in annotations if ann['image_id'] in val_ids]

    print(f"val images: {len(val_images)}, annotations: {len(annotations_val)}")
    
    output_coco = {
            'images': val_images,
            'annotations': annotations_val,
            'categories': categories}
    with open(save_anno_path, 'w', encoding='utf‑8') as f:
        json.dump(output_coco, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--load-anno-path", default = './dataset/annotations/train.json', type=str)
    parser.add_argument(
        "--save-anno-path", default = './dataset/annotations/custom_val.json', type=str)
    parser.add_argument(
        "--val-ratio", default = 0.25, type=float)
    parser.add_argument(
        "--seed", default = 42, type=int)
    args = parser.parse_args()
    
    split_coco_annotations(
        load_anno_path = args.load_anno_path,
        save_anno_path = args.save_anno_path,
        val_ratio = args.val_ratio,
        seed = args.seed)