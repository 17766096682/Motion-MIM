import torch.nn as nn
from core.layers.motionrnn_mim_cell import MotionRNN_cell



class MotionRNN_MIM(nn.Module):
    # def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
    def __init__(self, num_layers, num_hidden, configs):
        super().__init__()
        # self.n_layers = cfg.LSTM_layers
        lstm = [MotionRNN_cell(input_channel, output_channel, b_h_w, kernel_size, stride, padding) for l in
                range(self.n_layers)]
        self.lstm = nn.ModuleList(lstm)
        print('This is MotionRNN-MIM!')

    def forward(self, x, m, layer_hiddens, embed, fc):
        x = embed(x)
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
        x = fc(x)
        return x, m, next_layer_hiddens
