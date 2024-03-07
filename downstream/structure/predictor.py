from structure.resnet import renet_b16
from transformers.models.esm.modeling_esm import *
import pdb

class SSCNNPredictor(nn.Module):
    def __init__(self, args, extractor, esmconfig, is_freeze = False):
        super(SSCNNPredictor, self).__init__()
        self.extractor = extractor
        self.esmconfig = esmconfig
        self.cnn = renet_b16(myChannels=esmconfig.hidden_size, bbn=16)
        #print(self.cnn)
        self.args = args

        self.is_freeze = is_freeze
        if is_freeze:
            for param in self.extractor.parameters():
                param.detach_()
            self.extractor.eval()

    def forward(self, data_dict):
        input_ids, attention_mask = data_dict['input_ids'], data_dict['attention_mask']
        #print('input_ids',input_ids.shape)
        #print('attention_mask',attention_mask)
        #print(input_ids)
        if self.is_freeze:
            with torch.no_grad():
                output = self.extractor(input_ids=input_ids, attention_mask=attention_mask)
        else:
            output = self.extractor(input_ids=input_ids, attention_mask=attention_mask)
        #print(output[0].size())
        #pdb.set_trace()
        #print(dir(output))
        if self.args.model_type == 'esm-rna':
            hidden_states = output.last_hidden_state
            
        elif self.args.model_type == 'dnabert':
            hidden_states = output.last_hidden_state
        elif self.args.model_type == 'dnabert2': 
            hidden_states = output[0]
        elif self.args.model_type == 'rnabert':
            hidden_states = output[0]
        #print('hidden_states1',hidden_states.shape)
        ## L*ch-> LxL*ch
        batch_size = hidden_states.shape[0]
        weight_mask = data_dict['weight_mask'] #[bz,ori_max_len+2]
        ori_length = weight_mask.shape[1]
        cur_length = hidden_states.shape[1]
        padding_tensor = torch.zeros((batch_size, ori_length-cur_length, hidden_states.shape[-1]),dtype=hidden_states.dtype,device=hidden_states.device)
        mapping_final_input = torch.cat([padding_tensor, hidden_states],dim=1)
        mapping_final_input[:,0,:] = hidden_states[:,0,:] # [cls]
        if self.args.token_type == '6mer':
            for bz in range(batch_size):
                valid_length = torch.sum(attention_mask[bz,:]==1).item()
                for i in range(1, valid_length-1):
                    mapping_final_input[bz,i:i+6,:] += hidden_states[bz,i,:]
                mapping_final_input[bz,valid_length+5-1,:] = hidden_states[bz,valid_length-1,:] #[sep]
        hidden_states = mapping_final_input * weight_mask.unsqueeze(2)        
            
        matrix = torch.einsum('ijk,ilk->ijlk', hidden_states, hidden_states)
        #print('hidden_states2',matrix.shape)
        matrix = matrix.permute(0, 3, 1, 2)  # L*L*2d
        #print('hidden_states3',matrix.shape)
        x = self.cnn(matrix)
        #print('hidden_states4',matrix.shape)
        x = x.squeeze(-1)

        return x


    # def __init__(self, extractor, dropout=0.1, freeze=False):
    #     super(NAcls_predictor, self).__init__()

    #     self.freeze = freeze
    #     self.extractor = extractor
    #     feat_num = extractor.config.hidden_size

    #     if self.freeze:
    #         for param in self.extractor.parameters():
    #             param.requires_grad = False

    #     self.dropout = nn.Dropout(dropout)
    #     self.fc = nn.Linear(feat_num, 1)

    # def forward(self, input_ids, attention_mask):
    #     output = self.extractor(input_ids, attention_mask=attention_mask)
    #     pool_feat = output.pooler_output
    #     output = self.dropout(pool_feat)
    #     output = self.fc(output)
    #     return output