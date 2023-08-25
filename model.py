import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class interface(nn.Module):
    def __init__(
        self,
        num_pretrain_layers=12,
    ):
        super().__init__()

        self.layer_weights=nn.Parameter(torch.ones(num_pretrain_layers) /num_pretrain_layers)

    def forward(self, x):
        # print(self.layer_weights)
        norm_weights=F.softmax(self.layer_weights, dim=-1)
        # print(norm_weights)
        x=(x*norm_weights.view(1, 1, -1, 1)).sum(dim=2)
        return x

# Copied from transformers.models.wav2vec2.modeling_wav2vec2.TDNNLayer
class TDNNLayer(nn.Module):
    def __init__(self, tdnn_dim,tdnn_kernel,tdnn_dilation,layer_id=0):
        super().__init__()
        self.in_conv_dim = tdnn_dim[layer_id - 1] if layer_id > 0 else tdnn_dim[layer_id]
        self.out_conv_dim = tdnn_dim[layer_id]
        self.kernel_size = tdnn_kernel[layer_id]
        self.dilation = tdnn_dilation[layer_id]

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states = nn.functional.unfold(
            hidden_states,
            (self.kernel_size, self.in_conv_dim),
            stride=(1, self.in_conv_dim),
            dilation=(self.dilation, 1),
        )
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.kernel(hidden_states)

        hidden_states = self.activation(hidden_states)
        return hidden_states

# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForXVector with Wav2Vec2->WavLM, wav2vec2->wavlm, WAV_2_VEC_2->WAVLM
class XVector(nn.Module):
    def __init__(self, input_dim=768,
                        tdnn_dim=[512,512,512,512,1500],
                        tdnn_kernel=[5,3,3,1,1],
                        tdnn_dilation=[1,2,3,1,1],
                        xvector_output_dim=512,
                        inited=True
                        ):
        super().__init__()

        self.projector = nn.Linear(input_dim, tdnn_dim[0])

        tdnn_layers = [TDNNLayer(tdnn_dim=tdnn_dim,
                                tdnn_kernel=tdnn_kernel,
                                tdnn_dilation=tdnn_dilation, 
                                layer_id=i) for i in range(len(tdnn_dim))]
        self.tdnn = nn.ModuleList(tdnn_layers)

        self.feature_extractor = nn.Linear(tdnn_dim[-1] * 2, xvector_output_dim)

        if not inited:
            self.init_weights()


    def _get_tdnn_output_lengths(self, input_length):
        """
        Computes the output length of the TDNN layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights.
        """
        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights should be skipped when not initializing all weights
        # since from_pretrained(...) calls tie weights anyways
        self.tie_weights()

    def tie_weights(self):
        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    def _init_weights(self, module):
        """Initialize the weights"""
        # gumbel softmax requires special init
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def forward(self,hidden_states) :
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        hidden_states = self.projector(hidden_states)

        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # Statistic Pooling
        mean_features = hidden_states.mean(dim=1)
        std_features = hidden_states.std(dim=1)
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)

        output_embeddings = self.feature_extractor(statistic_pooling)
        return output_embeddings

class XVector_classifier(nn.Module):
    def __init__(self, xvector_output_dim=512, num_labels=8):
        super().__init__()
        self.classifier = nn.Linear(xvector_output_dim, num_labels)
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(self,embeddings):
        logits = self.classifier(embeddings)

        return logits

class RNN_enc(nn.Module):
    def __init__(
            self,
            input_dim=768,
            output_dim=32,
            # activation=torch.nn.LeakyReLU,
            rnn_blocks=2,
            rnn_neurons=512,
            dp=0.2,
            bidirectional=True,
            avg_pooling=False,
        ):
        super().__init__()

        self.RNN=nn.LSTM(input_size=input_dim,
                            hidden_size=rnn_neurons,
                            num_layers=rnn_blocks,
                            batch_first=True,
                            dropout=dp,
                            bidirectional=bidirectional,
                            )
        
        if bidirectional:
            self.FC=nn.Linear(rnn_neurons*2,output_dim)
        else:
            self.FC=nn.Linear(rnn_neurons,output_dim)
        self.avg = avg_pooling

    def forward(self, x):
        x = self.RNN(x)
        x=x[0]
        if self.avg:
            x = torch.mean(x,dim=1)
        x = self.FC(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self,input_dim=768,output_dim=3,d_model=256, nhead=4, num_encoder_layers=4,
                dim_feedforward=256,dp = 0.1,device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.fc_embed=nn.Linear(input_dim,d_model)

        self.pos_encoder = PositionalEncoding(d_model, dp)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dp)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.out_params = nn.Linear(d_model, output_dim)
        self.device = device


    def forward(self, src,src_key_padding_mask=None):
        src = src.permute(1,0,2)
        src = self.fc_embed(src)
        src = self.pos_encoder(src)    
        output = self.transformer_encoder(src,src_key_padding_mask=src_key_padding_mask)
        # mean pooling
        output = torch.mean(output,dim=0)
        params = self.out_params(output) 
        return params


    def create_pad_mask(self,matrix, pad_token):
        matrix=torch.where(torch.isnan(matrix),torch.tensor(pad_token,dtype=torch.float).to(self.device),matrix)
        return (matrix == pad_token)[:,:,0]


class PositionalEncoding(nn.Module):
    #https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout= 0.1, max_len = 1800):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2.0) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class VAD_linear_enc(nn.Module):
    def __init__(self, input_dim=768,output_dim=2,d_model=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc3 = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, output_dim)
        # self.out = nn.Linear(input_dim, output_dim)
        if output_dim==1:
            self.sigmoid = nn.Sigmoid()
        self.output_dim = output_dim
        

    def forward(self,x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.out(x)
        if self.output_dim==1:
            x = self.sigmoid(x)

        return x
