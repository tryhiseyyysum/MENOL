from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector
import json
import numpy as np
import torch


def xyxy_to_xywh(bbox):
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1, y2 - y1]
    elif len(bbox) == 5:  
        x1, y1, x2, y2, score = bbox
        return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
    else:
        raise ValueError("Invalid bounding box format.")


def convert_to_serializable(data):
    if isinstance(data, torch.Tensor):
        return data.tolist()
    elif isinstance(data, np.int64):  
        return int(data)
    elif isinstance(data, np.float32):
        return float(data)
    elif isinstance(data, (list, tuple)):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, dict):
        return {convert_to_serializable(key): convert_to_serializable(value) for key, value in data.items()}
    else:
        return data



def main():

    config_file = './mmdetection/configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py'
    checkpoint_file = './checkpoints/model_final.pth'  

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_dir = './dataset/CODA/images/'
    out_dir = './output'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    test_path = './CODA2022-val.txt'
    all_detections = [] 
    with open(test_path, 'r') as fp:
        test_list = fp.readlines()

    for idx, test in enumerate(test_list):
        print(f"idx=   {idx}")
        test = test.replace('\n', '')
        name = os.path.join(img_dir, test + '.jpg')

        print('Model is processing image {}/{}.'.format(idx + 1, len(test_list)))
        detections_for_image = []
        result = inference_detector(model, name)
        
        bboxes = np.vstack(result.pred_instances.bboxes.cpu().numpy())
        scores= result.pred_instances.scores
        labels=result.pred_instances.labels

        for label, box,score in zip(labels, bboxes,scores):
            if label+1 in [1, 2, 3, 4, 5, 6, 7]:  # Map the labels to appropriate category_id
                category_id = label+1
            else:
                category_id = 8  # Set category_id to 8 for labels other than 1-7

            x, y, w, h = xyxy_to_xywh(box)
            
            detection = {
                "image_id": idx + 1,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "score": convert_to_serializable(score)
            }

            existing_detections = detections_for_image
            existing_detection = next((existing for existing in existing_detections if existing["bbox"] == detection["bbox"] and existing["score"] >= detection["score"]), None)

            if existing_detection is None:
                existing_detections.append(detection)

        all_detections.extend(detections_for_image)  # Store detections for the current image
    # Save all detections to a single JSON file
    output_file = './Final_test/my_results.json'
    serializable_detections = convert_to_serializable(all_detections)
    with open(output_file, 'w') as f:
        json.dump(serializable_detections, f)

if __name__ == '__main__':
    main()
