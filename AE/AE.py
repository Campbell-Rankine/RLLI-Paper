import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

import ray
from tqdm import tqdm
from AE.network_utils import *

class Encoder(nn.Module):

    """
    Wrapper class for our VGG-16 Encoder Decoder structure. Based on the paper:
        DEEP STOCK REPRESENTATION LEARNING: FROM CANDLESTICK CHARTS TO
        INVESTMENT DECISIONS   
        url: https://arxiv.org/pdf/1709.03803.pdf

    Standard VGG-16 model with final linear layer changed to the avg-pooling layer

    We use 4 channel input, hence no pretrained weights, however we might need to use a 3 channel input of
        (open, diff, close)

    Other methods and variables define easy accessing of important network information
    """

    def __init__(self, batch_size, window_size, latent, dims):
        super(Encoder, self).__init__()
        self.features = dims

        self.dims = dims

        ### - Define VGG-16 Encoder Step - ###
        self.input_transform = input_transform(1, 3)
        self.encoder = models.vgg16(pretrained=False)
        print(self.encoder)

        ### - 512 Dimensional output representation vector - ###
        del self.encoder.classifier
        
        self.encoder.features[28].out_channels = 512
        self.encoder.features[30].kernel_size=4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        #self.encoder.Lin = nn.Linear(1024,128)

        self.encoder = self._encodify_(self.encoder)

        ### - Utils - ###
        self.batch_size = batch_size
        self.input_dims = (batch_size, self.features, window_size) #Picture creating a 4 channel input image
        self.in_channels = self.input_dims[-1]
        self.out_channels = latent
        self.activation = nn.Sigmoid()
        self.gradients = None
        self.latent = latent
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        pool_indices = []
        x_current = self.input_transform(x)
        for module_encode in self.encoder:
            #x_current = T.tensor(x_current, dtype=float)
            output = module_encode(x_current)

            # If the module is pooling, there are two outputs, the second the pool indices
            if isinstance(output, tuple) and len(output) == 2:
                x_current = output[0]
                pool_indices.append(output[1])
            else:
                x_current = output
        activation = self.activation(x_current)
        return activation
    
    def forward(self, x):
        """
        Run forward pass of representation vector through the VGG wrapper
        """
        pool_indices = []
        x_current = self.input_transform(x)
        transform_x = x_current
        for module_encode in self.encoder:
            #x_current = T.tensor(x_current, dtype=float)
            output = module_encode(x_current)
            # If the module is pooling, there are two outputs, the second the pool indices
            if isinstance(output, tuple) and len(output) == 2:
                x_current = output[0]
                ind = output[1]
                pool_indices.append(ind)
            else:
                x_current = output
        x_current = self.avgpool(x_current)
        activation = self.activation(x_current)
        h = activation.register_hook(self.activations_hook)
        return activation, pool_indices

    def get_shape(self):
        """
        Return Shape for ease of use
        """
        return {'Input': self.input_dims, 'Output': (512, 1)}

    def _encodify_(self, encoder):
        '''Create list of modules for encoder based on the architecture in VGG template model.
        In the encoder-decoder architecture, the unpooling operations in the decoder require pooling
        indices from the corresponding pooling operation in the encoder. In VGG template, these indices
        are not returned. Hence the need for this method to extent the pooling operations.
        Args:
            encoder : the template VGG model
        Returns:
            modules : the list of modules that define the encoder corresponding to the VGG model
        '''
        modules = nn.ModuleList()
        for module in encoder.features:
            if isinstance(module, nn.MaxPool2d):
                module_add = nn.MaxPool2d(kernel_size=module.kernel_size,
                                          stride=module.stride,
                                          padding=module.padding,
                                          return_indices=True)
                modules.append(module_add)
            else:
                modules.append(module)
        modules.append(self.avgpool)
        return modules

def invert_encoder(encoder):
    """
    return inverted network
    """
    decoder = []
    databar = tqdm(list(reversed(encoder)))
    for module in databar:
        databar.set_description('Reversing: %s' % (module))
        if isinstance(module, nn.Conv2d):
            args = {'in_channels': module.out_channels, 'out_channels': module.in_channels, 
                    'kernel_size' : module.kernel_size, 'stride' : module.stride,
                    'padding' : module.padding}
            module_transpose = nn.ConvTranspose2d(**args)
            module_norm = nn.BatchNorm2d(module.in_channels)
            module_act = nn.ReLU(inplace=True)
            modules_transpose = [module_transpose, module_norm, module_act]

            decoder += modules_transpose

        elif isinstance(module, nn.MaxPool2d):
                args = {'kernel_size' : module.kernel_size, 'stride' : module.stride,
                          'padding' : module.padding}
                module_transpose = nn.MaxUnpool2d(**args)
                decoder += [module_transpose]
    decoder = decoder[:-2]
    return nn.ModuleList(decoder)


class Decoder(nn.Module):
    def __init__(self, encoder):
        """
        Build network: 4 channel decoder
        Goal: Undo the operations learnt by the vgg16 encoder network
        """

        super(Decoder, self).__init__()
        self.decoder = invert_encoder(encoder.encoder)
        self.in_channels = encoder.out_channels
        self.out_channels = encoder.in_channels
        self.convert = output_transform(3, 1, (1,135))
        self.activation = nn.Sigmoid()

    def forward(self, x, pool_indices):
        x_current = x
        k_pool = 0
        reversed_pool_indices = list(reversed(pool_indices))
        for module_decode in self.decoder:

            # If the module is unpooling, collect the appropriate pooling indices
            if isinstance(module_decode, nn.MaxUnpool2d):
                x_current = module_decode(x_current, indices=reversed_pool_indices[k_pool])
                k_pool += 1
            else:
                x_current = module_decode(x_current)
        x_current = self.convert(self.activation(x_current))
        return x_current
    

class VGG16_AE(nn.Module):
    """
    Full Encoder-Decoder model
    """
    def __init__(self, encoder_args: dict, decoder_args: dict, device):
        super(VGG16_AE, self).__init__()

        self.encoder = Encoder(**encoder_args).to(device)
        self.decoder = Decoder(self.encoder, **decoder_args).to(device)
        print(self.encoder, self.decoder)
        self.in_channels = self.encoder.in_channels
        self.latent_dim = self.encoder.out_channels
        self.out_channels = self.decoder.out_channels

        self.freeze_encoder = False

    def forward(self, x):
        encoded, pool_indices = self.encoder(x)
        decoded = self.decoder(encoded, pool_indices)
        return decoded, encoded.shape

    def load_model(self, path, device, optim_args):
        print('... loading checkpoint ...')
        model = self.load_state_dict(T.load(path, map_location=device)['model'])
        optim = T.optim.Adam(self.parameters(), *optim_args)
        optimizer = optim.load_state_dict(T.load(path, map_location=device)['optimizer'])
        return model, optimizer


