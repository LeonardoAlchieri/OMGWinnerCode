from keras.models import Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace

from codes.CNN_model import model as customCNN

def get_model(model_name: str, model_params: dict) -> Model:
    
    if model_name == 'cnn':
        return customCNN(**model_params)
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented")
    
def get_backbone_model(input_shape: tuple = (256, 256, 3)) -> Model:
    VGGFace_resnet50_model=VGGFace(model='vgg16', include_top=False, input_shape=(256, 256, 3), pooling='avg')
    for layer in VGGFace_resnet50_model.layers:
        layer.trainable=False
        
    input_tensor = Input(input_shape)
    outputs = VGGFace_resnet50_model(input_tensor)
    model = Model(input_tensor, outputs, name='vgg16')
    return model