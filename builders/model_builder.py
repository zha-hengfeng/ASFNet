import importlib

# some SOTA methods
# from model.resnet18 import resnet18
# from model.ESPNet import ESPNet
# from model.ESPNetv2 import EESPNet_Seg
# from model.CGNet import Context_Guided_Network
# from model.RPNet import RPNet


from model.FCN_ResNet_18_C64 import FCN_ResNet_18


def build_model(model_name, num_classes):

    if model_name == 'FCN-ResNet-18-C64':
        print("=====> Build FCN-ResNet-18-C64 !")
        return FCN_ResNet_18(num_classes=num_classes, encoder_only=True, block_channel=64)

    if model_name == 'FCN-ResNet-18-C64-EACNet':
        print("=====> Build FCN-ResNet-18-C64-EACNet !")
        return FCN_ResNet_18(num_classes=num_classes, encoder_only=False, block_channel=64)