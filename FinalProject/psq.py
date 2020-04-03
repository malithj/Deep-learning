import torch
import torch.nn as nn
import torch.quantization
import os
from encoder.model import SpeakerEncoder

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def main():
    saved_model_dir = './saved_model/'
    float_model_file = 'pretrained.pt'

    # create directory
    if not os.path.exists(saved_model_dir):
        try:
            os.makedirs(saved_model_dir)
        except OSError as e:
            raise Exception("Could not create directory {0:}. Please check file system permissions.".format(saved_model_dir))
    if not os.path.exists(saved_model_dir + float_model_file):
        raise Exception("Cannot perform static quantization without trained model. Please provide weights file.")

    num_calibration_batches = 10

    # set default device to cpu since pytorch only supports quantization on CPU
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _device = torch.device("cpu")

    myModel = SpeakerEncoder(_device, torch.device("cpu"))
    checkpoint = torch.load(saved_model_dir + float_model_file)
    myModel.load_state_dict(checkpoint["model_state"])
    myModel.to(_device)
    myModel.eval()

    # Fuse Conv, bn and relu
    # myModel.fuse_model()

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    # myModel.qconfig = torch.quantization.default_qconfig
    # print(myModel.qconfig)
    # torch.quantization.prepare(myModel, inplace=True)

    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')
    #print('\n Inverted Residual Block:After observer insertion \n\n', myModel.features[1].conv)

    # Calibrate with the training set
    print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    myModel = torch.quantization.quantize_dynamic(myModel, {nn.LSTM, nn.Linear}, dtype=torch.qint8)
    # torch.quantization.convert(myModel, inplace=True)
    print('Post Training Quantization: Convert done')
    #print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',myModel.features[1].conv)

    print("Size of model before quantization: ", end="")
    print('{0:} (MB):'.format(os.path.getsize(saved_model_dir + float_model_file)/1e6))
    print("Size of model after quantization: ", end="")
    step = checkpoint['step']
    store_file = {'step': step, 'model_state': myModel.state_dict(), 'quantized': True}
    torch.save(store_file, saved_model_dir + "quantized.pt")
    print('{0:} (MB):'.format(os.path.getsize(saved_model_dir + "quantized.pt")/1e6))



if __name__ == '__main__':
    main()
