import importlib

# some SOTA methods
# from model.resnet18 import resnet18
# from model.ESPNet import ESPNet
# from model.ESPNetv2 import EESPNet_Seg
# from model.CGNet import Context_Guided_Network
# from model.RPNet import RPNet


from model.FCN_ResNet_18 import FCN_ResNet_18
from model.FCN_SKNet import FCN_SKNet
from model.APFNet import APFNet
from model.APFNet_2 import APFNet_2
from model.APF_CAM import APFNet_CAM


def build_model(model_name, num_classes):

    if model_name == 'FCN-ResNet-18-C64':
        print("=====> Build FCN-ResNet-18-C64 !")
        return FCN_ResNet_18(num_classes=num_classes, encoder_only=True, block_channel=64)

    if model_name == 'FCN-ResNet-18-C64-EACNet':
        print("=====> Build FCN-ResNet-18-C64-EACNet !")
        return FCN_ResNet_18(num_classes=num_classes, encoder_only=False, block_channel=64)

    if model_name == 'FCN-ResNet-18-C32':
        print("=====> Build FCN-ResNet-18-C32 !")
        return FCN_ResNet_18(num_classes=num_classes, encoder_only=True, block_channel=32)

    if model_name == 'FCN-ResNet-18-C32-EACNet':
        print("=====> Build FCN-ResNet-18-C32-EACNet !")
        return FCN_ResNet_18(num_classes=num_classes, encoder_only=False, block_channel=32)

    if model_name == 'FCN-ResNet-18-C32-3x3':
        print("=====> Build FCN-ResNet-18-C32 !")
        return FCN_ResNet_18(num_classes=num_classes, encoder_only=True, block_channel=32, use3x3=True)

    if model_name == 'FCN-ResNet-18-C64-3x3':
        print("=====> Build FCN-ResNet-18-C64 !")
        return FCN_ResNet_18(num_classes=num_classes, encoder_only=True, block_channel=64, use3x3=True)

    if model_name == 'FCN-SKNet':
        print("=====> Build FCN_SKNet !")
        return FCN_SKNet(num_classes=num_classes, block_channel=32, encoder_only=False)

    if model_name == 'FCN-Res18-C64-SK':
        print("=====> Build FCN_SKNet !")
        return FCN_SKNet(num_classes=num_classes, block_channel=64)

    if model_name == 'APFNet':
        print("=====> Build APFNet !")
        return APFNet(num_classes=num_classes, block_channel=32)

    if model_name == 'APFNet_2':
        print("=====> Build APFNet !")
        return APFNet_2(num_classes=num_classes, block_channel=32)

    if model_name == 'APFNet_CAM':
        return APFNet_CAM(num_classes=num_classes, block_channel=32)


    if model_name == 'FCN34-c32':
        from model.FCN_ResNet34 import FCN_ResNet34
        return FCN_ResNet34(num_classes=num_classes, block_channel=32)

    if model_name == 'APFNet_CAM_r34':
        return APFNet_CAM(num_classes=num_classes, block_channel=32, backbone='res34')