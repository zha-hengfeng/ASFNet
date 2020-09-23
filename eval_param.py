import timeit
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from utils.tools.utils import *

from thop import profile

def count_flops(args):
    pass

    # your rule
    #hereflops, params = profile(model, input_size=(1, 3, 224, 224),
    #                            custom_ops={YourModule: count_your_model})

def count_param(args):
    model = build_model(args.model, num_classes=args.classes)
        #print(model)

    init_weight(model, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-3, 0.1,
                mode='fan_in')
    print("=====> computing network parameters and FLOPs")
    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))

    input = torch.randn(1, 3, 480, 360)
    flops, params = profile(model, inputs=(input,))
    print("params:", params)
    print("flops:", flops)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="FCN-ResNet-18-C64",
                        help="[FCN-ResNet-18-C64]")
    parser.add_argument('--classes', type=int, default=19,
                        help="the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
    args = parser.parse_args()

    count_param(args)