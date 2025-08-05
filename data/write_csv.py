import torch
import pandas as pd
import lmdb
import pickle
from model.utils.utils import load_config
from model.utils.chemical import resindex_to_ressymb

if __name__ == '__main__':
    config, config_name = load_config('../configs/base_pretrain.yaml')

    db_conn = lmdb.open(config.dataset.processed_dir + "/structures.lmdb",
                        map_size=32*(1024*1024*1024),
                        create=False,
                        subdir=False,
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False,
                        )

    seqs_dict = {'id': [], 'H_seq': [], 'L_seq': []}
    with db_conn.begin() as txn:
        # 使用cursor遍历所有条目
        with txn.cursor() as cursor:
            for key, value in cursor:
                data = pickle.loads(value)
                seqs_dict['id'].append(data['id'])

                H_seq = data['aa'][data['is_heavy'].bool()]
                H_seq = ''.join([resindex_to_ressymb[token.item()] for token in H_seq])
                seqs_dict['H_seq'].append(H_seq)

                L_seq = data['aa'][~(data['is_heavy'].bool())]
                L_seq = ''.join([resindex_to_ressymb[token.item()] for token in L_seq])
                seqs_dict['L_seq'].append(L_seq)

    df = pd.DataFrame(seqs_dict)
    df['V_seq'] = df['H_seq'] + '|' + df['L_seq']
    df = df.drop_duplicates(subset='V_seq')
    pickle.dump(seqs_dict['id'], open(config.dataset.processed_dir + '/onlyV_drop_duplicates_id.pkl', 'rb'))
    df.to_csv(config.dataset.processed_dir + '/SAbDab_processed_only_sequences_onlyV_drop_duplicates.csv')
