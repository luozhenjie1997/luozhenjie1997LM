import torch
import h5py
import joblib
from torch.cuda import is_available
from model.utils.utils import set_seed, load_config
from transformers import BertModel, BertTokenizer
from mint.mint.helpers.extract import CSVDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda:0" if is_available() else "cpu")  # 获取可以使用的硬件资源
print(device)

if __name__ == '__main__':
    # 输入尺寸会变，因此设置为False
    torch.backends.cudnn.benchmark = False
    # 固定cuda的随机数种子，每次返回的卷积算法将是确定的
    torch.backends.cudnn.deterministic = True

    set_seed()

    config, config_name = load_config('./configs/base_pretrain.yaml')

    tokeniser = BertTokenizer.from_pretrained("/srv/storage/hdd/zzl/lzj/LLM/IgBert ", do_lower_case=False, trust_remote_code=True)
    model = BertModel.from_pretrained("/srv/storage/hdd/zzl/lzj/LLM/IgBert ", add_pooling_layer=False, trust_remote_code=True).to(device)
    model = model.eval()

    train_dataset = CSVDataset('/srv/storage/hdd/zzl/lzj/dataset/SAbDab/SAbDab_processed_only_sequences_onlyV.csv', 'H_seq', 'L_seq')
    train_loader = DataLoader(train_dataset, batch_size=1, pin_memory=True, num_workers=joblib.cpu_count())

    # 创建h5文件
    h5_file = h5py.File('./SAbDab_processed_sequences_IgBert_embedding.h5', 'w')
    pbar = tqdm(range(len(train_loader)))

    with torch.no_grad():
        for _, batches in zip(pbar, train_loader):
            data_id = batches[0][0]
            sequence = [' '.join(batches[1][0])+' [SEP] '+' '.join(batches[2][0])]
            tokens = tokeniser.batch_encode_plus(sequence, add_special_tokens=True, pad_to_max_length=True, return_tensors="pt", return_special_tokens_mask=True)
            result = model(input_ids=tokens['input_ids'].to(device), attention_mask=tokens['attention_mask'].to(device))
            embeddings = result.last_hidden_state
            masks = []
            for token, attention_mask in zip(tokens['input_ids'], tokens['attention_mask']):
                mask = attention_mask.bool()
                mask[0] = False
                mask[-1] = False
                # 查找链连接符号的索引
                con_index = torch.where(token == 3)[0][0]  # 最后一个SEP作为结束符
                mask[con_index] = False
                masks.append(mask)
            masks = torch.stack(masks).to(device)
            # 留下非特殊标记的token嵌入
            embeddings = embeddings[masks].clone()
            try:
                h5_file.create_dataset(data_id, data=embeddings.cpu().numpy())
            except:
                print(f"id{data_id} is exist!")
                continue
