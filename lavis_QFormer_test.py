import inspect
vname = lambda v,nms: [ vn for vn in nms if id(v)==id(nms[vn])][0]

import torch
def save_param(ms_tensor: torch.Tensor, ms_tensor_name: str):
    print("torch_tensor_name is ", ms_tensor_name)
    save_path = "/mnt/d/compare/" + ms_tensor_name + ".pth"
    print("save path is ", save_path)
    torch.save(ms_tensor, save_path)

from lavis.models.blip2_models.Qformer import BertLMHeadModel, BertModel, BertOnlyMLMHead
from transformers.models.bert.configuration_bert import BertConfig
from transformers.configuration_utils import PretrainedConfig

pretrained_config = PretrainedConfig()
pretrained_config = pretrained_config.from_pretrained("/mnt/d/QFormer/config.json")

state_dict = torch.load("/mnt/d/QFormer/self_Qformer_state_dict_init.pth", map_location="cpu")

qFormer = BertLMHeadModel(pretrained_config)

qFormer.resize_token_embeddings(30523)

qFormer.load_state_dict(state_dict)

for name, param in qFormer.named_parameters():
    if "_query" in name:
        key_orig = name.replace("_query", "")
        param.data = state_dict[key_orig].data

from transformers.file_utils import ModelOutput
def SaveOutput(output: ModelOutput, name: str):
    global cnt
    cnt = 0
    def SaveTensorFromTuple(tup: tuple):
        for tt in tup:
            if isinstance(tt, torch.Tensor):
                global cnt
                save_param(tt, "{}_{}".format(name, cnt))
                cnt += 1
            elif isinstance(tt, ModelOutput):
                SaveTensorFromTuple(tt.to_tuple())
            elif isinstance(tt, tuple):
                SaveTensorFromTuple(tt)
    SaveTensorFromTuple(output.to_tuple())
    
import numpy as np 
def ModelOutputCompare(output: ModelOutput, output2: ModelOutput):
    def BasicCompare(M, N):
        if M is None:
            if N is not None:
                return "one is None the other is not."
            else:
                return "both None"
        elif type(M) != type(N):
            return "one is type {} but the other is of type {}".format(type(M), type(N))
        else:
            return True
                  
    def TensorCompare(M: torch.Tensor, N: torch.Tensor):
        return np.abs(M.detach().numpy() - N.detach().numpy()).max()
    
    def TupleCompare(M: tuple, N: tuple):
        compare_result = ()
        if len(M) != len(N):
            raise ValueError("tuple len not equal! first is {} and second is {}".format(len(M), len(N)))
        for i in range(len(M)):
            basic_ret = BasicCompare(M[i], N[i])
            if isinstance(basic_ret, str):
                compare_result += (basic_ret,)
                continue
            elif isinstance(M[i], torch.Tensor):
                compare_result += (TensorCompare(M[i], N[i]),)
            elif isinstance(M[i], ModelOutput):
                compare_result += (ModelOutputCompare(M[i], N[i]),)
            elif isinstance(M[i], tuple):
                compare_result += (TupleCompare(M[i], N[i]),)
        return compare_result
     
    for name in output.keys():
        value = output[name]
        if name not in output2.keys():
            raise ValueError("[{}] in output.keys() but not in output2.keys().".format(name))
        basic_ret = BasicCompare(value, output2[name])
        if isinstance(basic_ret, str):
            print("{}: {}".format(name, basic_ret))
            continue
        elif isinstance(value, torch.Tensor):
            print("{}: {}".format(name, TensorCompare(value, output2[name])))
        elif isinstance(value, tuple):
            print("{}: {}".format(name, TupleCompare(value, output2[name])))
        elif isinstance(value, ModelOutput):
            print("{}:".format(name))
            ModelOutputCompare(value, output2[name])
    
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ test cases begin @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# 1)
query_tokens = torch.load("/mnt/d/QFormer/query_tokens1.pth", map_location="cpu")
image_embeds = torch.load("/mnt/d/QFormer/image_embeds1.pth", map_location="cpu")
image_atts = torch.load("/mnt/d/QFormer/image_atts1.pth", map_location="cpu")

query_output = qFormer.bert(
           query_embeds=query_tokens,
           encoder_hidden_states=image_embeds,
           encoder_attention_mask=image_atts,
           use_cache=True,
           return_dict=True,
           )

query_output_real = torch.load("/mnt/d/QFormer/query_output1.pth", map_location="cpu")

ModelOutputCompare(query_output, query_output_real)
SaveOutput(query_output, "query_output")

hs1 = query_output.last_hidden_state
EqualOrNot = hs1.equal(query_output_real.last_hidden_state)
print("query_output.last_hidden_state == query_output_real.last_hidden_state ? ", EqualOrNot)


# 2)
input_ids = torch.load("/mnt/d/QFormer/input_ids.pth")
attention_mask = torch.load("/mnt/d/QFormer/attention_mask.pth")

text_output = qFormer.bert(
    input_ids,
    attention_mask,
    return_dict=True,
)

text_output_real = torch.load("/mnt/d/QFormer/text_output.pth")

ModelOutputCompare(text_output, text_output_real)    
SaveOutput(text_output, "text_output")

# 3)
text_ids_all = torch.load("/mnt/d/QFormer/text_ids_all.pth")
query_tokens_itm = torch.load("/mnt/d/QFormer/query_tokens_itm.pth")
attention_mask_all = torch.load("/mnt/d/QFormer/attention_mask_all.pth")
image_embeds_all = torch.load("/mnt/d/QFormer/image_embeds_all.pth")
image_atts_all = torch.load("/mnt/d/QFormer/image_atts_all.pth")

output_itm = qFormer.bert(
    text_ids_all,
    query_embeds=query_tokens_itm,
    attention_mask=attention_mask_all,
    encoder_hidden_states=image_embeds_all,
    encoder_attention_mask=image_atts_all,
    return_dict=True,
)
output_itm_real = torch.load("/mnt/d/QFormer/output_itm.pth")

ModelOutputCompare(output_itm, output_itm_real)    
SaveOutput(output_itm, "output_itm")

# 4)
decoder_input_ids = torch.load("/mnt/d/QFormer/decoder_input_ids.pth")
attention_mask = torch.load("/mnt/d/QFormer/attention_mask_lm.pth")
past_key_values = torch.load("/mnt/d/QFormer/past_key_values_lm.pth")
labels = torch.load("/mnt/d/QFormer/labels_lm.pth")
lm_output = qFormer(
    decoder_input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    return_dict=True,
    labels=labels,
)
lm_output_real = torch.load("/mnt/d/QFormer/lm_output.pth")

ModelOutputCompare(lm_output, lm_output_real)    
SaveOutput(lm_output, "lm_output")