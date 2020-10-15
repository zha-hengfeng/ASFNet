import importlib

# some SOTA methods
# from model.resnet18 import resnet18
# from model.ESPNet import ESPNet
# from model.ESPNetv2 import EESPNet_Seg
# from model.CGNet import Context_Guided_Network
# from model.RPNet import RPNet

# APFNet Experiment
from model.FCN_ResNet import FCN_ResNet


def build_model(model_name, num_classes):

    if model_name == 'FCN-ResNet-18-C64':
        print("=====> Build FCN-ResNet-18-C64 !")
        return FCN_ResNet(num_classes=num_classes, encoder_only=True, block_channel=64)

    if model_name == 'FCN-ResNet-18-C32':
        print("=====> Build FCN-ResNet-18-C32 !")
        return FCN_ResNet(num_classes=num_classes, encoder_only=True, block_channel=32)

    if model_name == 'FCN-ResNet-18-C32-3x3':
        print("=====> Build FCN-ResNet-18-C32 !")
        return FCN_ResNet(num_classes=num_classes, encoder_only=True, block_channel=32, use3x3=True)

    if model_name == 'FCN-ResNet-18-C64-3x3':
        print("=====> Build FCN-ResNet-18-C64 !")
        return FCN_ResNet(num_classes=num_classes, encoder_only=True, block_channel=64, use3x3=True)


    if model_name == 'APFNet':
        print("=====> Build APFNet !")
        return APFNet(num_classes=num_classes, block_channel=32)

    if model_name == 'APFNet_2':
        print("=====> Build APFNet !")
        return APFNet_2(num_classes=num_classes, block_channel=32)

    if model_name == 'APFNet_CAM':
        return APFNet_CAM(num_classes=num_classes, block_channel=32)


    if model_name == 'FCN34-c32':
        from model.FCN_ResNet import FCN_ResNet34
        return FCN_ResNet34(num_classes=num_classes, block_channel=32)

    if model_name == 'APFNet_CAM_r34':
        return APFNet_CAM(num_classes=num_classes, block_channel=32, backbone='res34', only34=True)

    if model_name == 'apfnet_cam_r34_numch':
        return APFNet_CAM_vnum(num_classes=num_classes, block_channel=64, backbone='res18')