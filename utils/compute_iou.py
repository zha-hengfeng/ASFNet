import time
import numpy as np
import torch
from torch.autograd import Variable

from utils.metrics import ComputeIoU
from utils.tools.save_predict import *


def eval_one_model(args, test_loader, model):
    """
    args:
      test_loader: loaded for test dataset
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()
    total_batches = len(test_loader)
    iouEvalVal = ComputeIoU(args.classes+1)  # cityscapes
    # iouEvalVal = iouEval(args.classes+1, ignoreIndex=11)    #camvid
    data_list = []
    for i, (input, label, size, name) in enumerate(test_loader):
        with torch.no_grad():
            if args.cuda:
                input_var = Variable(input).cuda()
            else:
                input_var = Variable(input)
            start_time = time.time()

            output = model(input_var)

            torch.cuda.synchronize()
            time_taken = time.time() - start_time
        print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
        # print(output.max(1)[1].unsqueeze(1).dtype)
        # print(label.dtype)
        iouEvalVal.addBatch(output.max(1)[1].unsqueeze(1).data, label.unsqueeze(1))
        output = output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype=np.uint8)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        # data_list.append([gt.flatten(), output.flatten()])

        # save the predicted image
        if args.save:
            save_predict(output, gt, name[0], args.dataset, args.save_seg_dir,
                         output_grey=False, output_color=True, gt_color=False)

    meanIoU, per_class_iu = iouEvalVal.getIoU()
    return meanIoU, per_class_iu