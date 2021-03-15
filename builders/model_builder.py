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

    if model_name == 'res18c32u3x3':
        print("=====> Build FCN-ResNet-18-C32 !")
        return FCN_ResNet(num_classes=num_classes, backbone='res18', block_channel=32, use3x3=True)

    if model_name == 'res18c64u3x3':
        print("=====> Build FCN-ResNet-18-C64 !")
        return FCN_ResNet(num_classes=num_classes, backbone='res18', block_channel=64, use3x3=True)

    # base model of FCN use ResNet
    if model_name == 'res18c16':
        return FCN_ResNet(num_classes=num_classes, backbone='res18', block_channel=16)

    if model_name == 'res18c32':
        return FCN_ResNet(num_classes=num_classes, backbone='res18', block_channel=32)

    if model_name == 'res18c48':
        return FCN_ResNet(num_classes=num_classes, backbone='res18', block_channel=48)

    if model_name == 'res18c64':
        return FCN_ResNet(num_classes=num_classes, backbone='res18', block_channel=64)

    if model_name == 'res34c16':
        return FCN_ResNet(num_classes=num_classes, backbone='res34', block_channel=16)

    if model_name == 'res34c32':
        return FCN_ResNet(num_classes=num_classes, backbone='res34', block_channel=32)

    if model_name == 'res34c48':
        return FCN_ResNet(num_classes=num_classes, backbone='res34', block_channel=48)

    if model_name == 'res34c64':
        return FCN_ResNet(num_classes=num_classes, backbone='res34', block_channel=64)

    # APFNet use CAM
    if model_name == 'APFNet_CAM_r34':
        from model.APF_CAM import APFNet_CAM
        return APFNet_CAM(num_classes=num_classes, backbone='res34', block_channel=32, only34=True)

    # apfnet use sfnet
    # ResNet-18 based
    if model_name == 'apfnetv2r18c16':
        from model.APFNet_v2 import APFNetv2_sf
        return APFNetv2_sf(num_classes=num_classes, backbone='res18', block_channel=16)

    if model_name == 'apfnetv2r18c32':
        from model.APFNet_v2 import APFNetv2_sf
        return APFNetv2_sf(num_classes=num_classes, backbone='res18', block_channel=32)

    if model_name == 'apfnetv2r18c48':
        from model.APFNet_v2 import APFNetv2_sf
        return APFNetv2_sf(num_classes=num_classes, backbone='res18', block_channel=48)

    if model_name == 'apfnetv2r18c64':
        from model.APFNet_v2 import APFNetv2_sf
        return APFNetv2_sf(num_classes=num_classes, backbone='res18', block_channel=64)

    # ResNet-34 based
    if model_name == 'apfnetv2r34c16':
        from model.APFNet_v2 import APFNetv2_sf
        return APFNetv2_sf(num_classes=num_classes, backbone='res34', block_channel=16)

    if model_name == 'apfnetv2r34c32':
        from model.APFNet_v2 import APFNetv2_sf
        return APFNetv2_sf(num_classes=num_classes, backbone='res34', block_channel=32)

    if model_name == 'apfnetv2r34c32-gau':
        from model.APFNet_v2 import APFNetv2_sf
        return APFNetv2_sf(num_classes=num_classes, backbone='res34', block_channel=32, up_mode='GAU')

    if model_name == 'apfnetv2r34c48':
        from model.APFNet_v2 import APFNetv2_sf
        return APFNetv2_sf(num_classes=num_classes, backbone='res34', block_channel=48)

    if model_name == 'apfnetv2r34c64':
        from model.APFNet_v2 import APFNetv2_sf
        return APFNetv2_sf(num_classes=num_classes, backbone='res34', block_channel=64)


    if model_name == 'apfnetv2_2':
        from model.APFNet_v2 import APFNetv2_sf_2
        return APFNetv2_sf_2(num_classes=num_classes, backbone='res34', block_channel=32)


    if model_name == 'resnet_d':
        from model.resnet_d import resnet18
        return resnet18(pretrained=False)
