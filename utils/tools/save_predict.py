import os
from PIL import Image

from utils.tools.colorize_mask import camvid_colorize_mask, cityscapes_colorize_mask


def save_predict(output, gt, img_name, dataset, save_path, output_grey=False, output_color=True, gt_color=False):
    if output_grey:
        output_grey = Image.fromarray(output)
        output_grey.save(os.path.join(save_path, img_name + '.png'))

    if output_color:
        if dataset == 'cityscapes':
            output_color = cityscapes_colorize_mask(output)
        elif dataset == 'camvid':
            output_color = camvid_colorize_mask(output)

        output_color.save(os.path.join(save_path, img_name + '_color.png'))

    if gt_color:
        if dataset == 'cityscapes':
            gt_color = cityscapes_colorize_mask(gt)
        elif dataset == 'camvid':
            gt_color = camvid_colorize_mask(gt)

        gt_color.save(os.path.join(save_path, img_name + '_gt.png'))