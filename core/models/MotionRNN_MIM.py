import math

import torch
import torch.nn as nn
from core.layers.motionrnn_mim_cell import MotionRNN_cell


class MotionRNN_MIM(nn.Module):
    # def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
    def __init__(self, num_layers, num_hidden, configs):
        super().__init__()
        # self.n_layers = cfg.LSTM_layers
        self.configs = configs
        self.n_layers = num_layers
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_hidden = num_hidden
        width = configs.img_width // configs.patch_size // configs.sr_size
        height = configs.img_height // configs.patch_size // configs.sr_size
        b_h_w = [configs.batch_size, height, width]

        lstm = [MotionRNN_cell(num_hidden[l], num_hidden[l], b_h_w, configs.filter_size, configs.stride, self.n_layers)
                for l in range(self.n_layers)]
        self.lstm = nn.ModuleList(lstm)

        # Encoder
        n = int(math.log2(configs.sr_size))
        encoders = []
        encoder = nn.Sequential()
        encoder.add_module(name='encoder_t_conv{0}'.format(-1),
                           module=nn.Conv2d(in_channels=self.frame_channel,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder.add_module(name='relu_t_{0}'.format(-1),
                           module=nn.LeakyReLU(0.2))
        encoders.append(encoder)
        for i in range(n):
            encoder = nn.Sequential()
            encoder.add_module(name='encoder_t{0}'.format(i),
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))
            encoder.add_module(name='encoder_t_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # Decoder
        decoders = []

        for i in range(n - 1):
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoder.add_module(name='c_decoder_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            decoders.append(decoder)

        if n > 0:
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        self.srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        print('This is MotionRNN-MIM!')

    def forward(self, frames, mask_true):

        next_frames = []
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        x = None
        layer_hiddens = None
        m = None
        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                time_diff = t - self.configs.input_length
                net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x
            frames_feature = net
            frames_feature_encoded = []
            for i in range(len(self.encoders)):
                frames_feature = self.encoders[i](frames_feature)
                frames_feature_encoded.append(frames_feature)
            x = frames_feature
            next_layer_hiddens = []
            for l in range(self.n_layers):
                if layer_hiddens is not None:
                    hiddens = layer_hiddens[l]
                    if l != 0:
                        xt_1 = layer_hiddens[l - 1][0]
                    else:
                        xt_1 = None
                else:
                    hiddens = None
                    xt_1 = None
                x, m, next_hiddens = self.lstm[l](x, xt_1, m, hiddens, l)
                next_layer_hiddens.append(next_hiddens)
            layer_hiddens = next_layer_hiddens

            for i in range(len(self.decoders)):
                x = self.decoders[i](x)
            x = self.srcnn(x)
            next_frames.append(x)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()

        return next_frames
