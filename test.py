# -*- coding: utf-8 -*-
#########################test##############################
import torch
import torchvision
import torch.nn.functional as F
import plot
import model
from plot import display
from util import box_corner_to_center,box_iou
from model import TinySSD

def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    anc = box_corner_to_center(anchors) # xy表示->中心表示
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) +       anc[:, :2] #+xy
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:] # ?????
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox



def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0: # 返回B中元素的个数
        i = B[0] # 位置
        keep.append(i)
        if B.numel() == 1: break # 加完最后一个元素
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4))   .reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1) # 交并比小于阈值的保留
        B = B[inds + 1]
    # while
    return torch.tensor(keep, device=boxes.device)

# 背景，猫，狗三种类别的预测概率[1,3,4]，偏移量[1,16],设为0，生成的四个锚框[1,4,4]，nms的阈值nms_threshold，正类的阈值pos_threshold
# 返回值为类别索引，置信度，预测边界框坐标
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    # nms_threshold??？？？？  pos_threshold??？？？？？？？
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0] # batch_size=1只有一张图片进来预测
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2] # 39840个框
    out = [] # ？？？？？？？？？？
    # 对每张图片
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0) #一共就两行，第0行不要了（大胆猜测是背景） 每列最大值（2，39840）
        a=class_id.numpy()
        predicted_bb = offset_inverse(anchors, offset_pred) # 根据锚框预测边界框
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        c=class_id.numpy()
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于 非背景 预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        d=class_id.numpy()
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    # for

    return torch.stack(out)


def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to('cpu'))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)  # 按列softmax （1，39840，2）->(1,2,39840)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    o=output[0]
    o=o[:,0]
    a=o.detach().numpy()
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]


if __name__ == '__main__':
    target_num='one' #目标个数：‘one’或‘two’或‘improve’

    # 定义网络并加载模型参数
    if target_num == 'one':
        net = TinySSD(num_classes=1)
        net = net.to('cpu')
        net.load_state_dict(torch.load('net/one_target/net_20.pkl', map_location=torch.device('cpu')))
        name = 'data/test/2.jpg'
    elif target_num == 'two':
        net = TinySSD(num_classes=2)
        net = net.to('cpu')
        net.load_state_dict(torch.load('net/two_target/net_50.pkl', map_location=torch.device('cpu')))
        name = 'data/test/4.jpg'
    else:
        net = TinySSD(num_classes=1)
        net = net.to('cpu')
        net.load_state_dict(torch.load('net/one_improve/net_30.pkl', map_location=torch.device('cpu')))
        name = 'data/test/2.jpg'

    X = torchvision.io.read_image(name).unsqueeze(0).float()
    img = X.squeeze(0).permute(1, 2, 0).long()

    output = predict(X)
    display(img, output.cpu(), threshold=0.5)