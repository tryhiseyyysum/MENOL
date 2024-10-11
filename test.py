# Eval the detection metrics on CODA dataset
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

# To place class-aware detection results
results = './Final_test/my_results.json'  ##模型预测结果
anno = './Final_test/gt.json'  #ground truth

print('='*20 + 'class-aware results' + '='*20)
coco_anno = coco.COCO(anno)
coco_dets = coco_anno.loadRes(results)
coco_eval = COCOeval(coco_anno, coco_dets, "bbox")
coco_eval.params.catIds = [1,2,3,4,5,6,7]
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

AP_common = coco_eval.stats[0]

print('='*20 + 'class-agnostic results' + '='*20)
# To place class-agnostic detection results
results = './Final_test/my_results_agnostic.json'  
anno = './Final_test/gt_agnostic.json'  #ground truth

coco_anno = coco.COCO(anno)
coco_dets = coco_anno.loadRes(results)
coco_eval = COCOeval(coco_anno, coco_dets, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

AP_agnostic = coco_eval.stats[0]
AR_agnostic = coco_eval.stats[8]


print('='*20 + 'class-agnostic corner results' + '='*20)
# To place class-agnostic corner detection results
results = './Final_test/my_results_agnostic.json'  
anno = './Final_test/gt_agnostic_corner.json'  #ground truth

coco_anno = coco.COCO(anno)
coco_dets = coco_anno.loadRes(results)
coco_eval = COCOeval(coco_anno, coco_dets, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

AR_agnostic_corner = coco_eval.stats[8]

print('='*20 + 'Final results' + '='*20)
print('AR_agnostic_corner: ', AR_agnostic_corner)
print('AR_agnostic: ', AR_agnostic)
print('AP_agnostic: ', AP_agnostic)
print('AP_common: ', AP_common)