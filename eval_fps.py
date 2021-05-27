import time
import torch
import torch.backends.cudnn as cudnn

from argparse import ArgumentParser
from builders.model_builder import build_model


def compute_speed(model, input_size, device, iteration=100):
    torch.cuda.set_device(device)
    cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size, device=device)

    for _ in range(50):
        model(input)
        # model(input, only_encode=True)

    print('=========Speed Testing=========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
        # model(input, only_encode=True)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
    return speed_time, fps


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--size", type=str, default="1024, 2048", help="input size(512,1024) of model")
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--classes', type=int, default=11)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--model', type=str, default="apfnet_r34c32_bi-2-stage", help="[FCN-ResNet-18-C32-3x3, FCN34-c32]"
                                                                       "APFNet_CAM_r34, FCN-ResNet-18-C32")
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 0)")
    args = parser.parse_args()

    h, w = map(int, args.size.split(','))
    model = build_model(args.model, num_classes=args.classes)
    print(model)
    compute_speed(model, (args.batch_size, args.num_channels, h, w), int(args.gpus), iteration=args.iter)
