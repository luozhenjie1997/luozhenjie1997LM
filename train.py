import random
import os
import time
import numpy as np
from torch.cuda import is_available
from model.utils.utils import load_config, set_seed, inf_iterator, recursive_to, add_weight_decay
from tqdm import tqdm
from model.utils.sabdab_onlyV import SAbDabDataset
from torch.utils.data import DataLoader
from model.utils.utils import PaddingCollate
from model.MultiFlow import FlowModel
from model.utils.chemical import CDR
from timm.scheduler.cosine_lr import CosineLRScheduler
from model.EMA import EMA
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda:0" if is_available() else "cpu")  # 获取可以使用的硬件资源
print(device)

if __name__ == '__main__':
    config, config_name = load_config('configs/base_pretrain.yaml')
    set_seed(config.train.seed)
    cdr_weight = config['train']['cdr_weight']
    # 输入尺寸会变，因此设置为False
    torch.backends.cudnn.benchmark = False
    # 固定cuda的随机数种子
    torch.backends.cudnn.deterministic = True

    # 设置填充工具选项，不在pad_values和no_padding上的会以0进行填充
    padding = PaddingCollate(
        pad_values={'aa': 21, 'chain_id': 4, 'mask_heavyatom': False, 'torsion_angle_mask': False},
        no_padding=['id', 'H1_seq', 'H2_seq', 'H3_seq', 'L1_seq', 'L2_seq', 'L3_seq'],
        eight=True)

    """加载数据集"""
    train_dataset = SAbDabDataset(summary_path=config.dataset.summary_path_dir, chothia_dir=config.dataset.structure_dir,
                                  processed_dir=config.dataset.processed_dir,
                                  embedding_h5_dir=config.dataset.embedding_dir,
                                  split='train', reset=config.dataset.reset)
    val_dataset = SAbDabDataset(summary_path=config.dataset.summary_path_dir, chothia_dir=config.dataset.structure_dir,
                                processed_dir=config.dataset.processed_dir,
                                embedding_h5_dir=config.dataset.embedding_dir,
                                split='val', reset=config.dataset.reset)
    print(len(train_dataset))
    print(len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, pin_memory=True, prefetch_factor=8,
                              collate_fn=padding, num_workers=8, drop_last=True)
    train_iterator = inf_iterator(train_loader)  # 无限循环的数据集迭代加载器
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, collate_fn=padding, num_workers=8)

    opti_num = config.train.pseudo_batch_size / config.train.batch_size

    """加载流生成模型"""
    model = FlowModel(config).to(device)
    param_group = add_weight_decay(model, l2_coeff=1e-2)
    optimizer = torch.optim.AdamW(param_group, lr=config.train.lr)
    # lr_scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 500, 10000, 0.95)
    lr_scheduler = CosineLRScheduler(optimizer, warmup_t=500, t_initial=config.train.max_iters, cycle_decay=0.1, cycle_limit=5, lr_min=1e-5)
    model.train()

    # ema = EMA(model, config.train.ema_decay)

    if os.path.exists('./save_model/checkpoint.ckpt'):
        checkpoint = torch.load('./save_model/checkpoint.ckpt')
        # ema.shadow.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['last_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        it = checkpoint['it']
        task_num = checkpoint['task_num']  # 记录进行的伪批次数，用于确定是否进行cdr训练
        normal_log_step = checkpoint['normal_log_step']
        cdr_log_step = checkpoint['cdr_log_step']
        val_log_step = checkpoint['val_log_step']
        log_name = checkpoint['log_name']
        best_loss = checkpoint['best_loss']
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['np_state'])
        torch.set_rng_state(checkpoint['rng_state'])  # 恢复CPU随机状态
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])  # 恢复所有GPU的随机状态
        del checkpoint
    else:
        # model.load_state_dict(torch.load('./save_model/model_from_multiflow.pth'))
        it = 0  # 初始化迭代次数，以优化次数为基准
        task_num = 0
        normal_log_step = 0
        cdr_log_step = 0
        val_log_step = 1
        best_loss = 999.
        log_name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    writer = SummaryWriter('./logs/MultiFlow/%s' % log_name)
    pbar = tqdm(range(config.train.max_iters), dynamic_ncols=True, leave=False, initial=it)
    pbar.set_description_str("trained it:0")

    batch_count = 0  # 记录使用的数据数量，用于确认是否优化
    batch_loss = 0
    batch_trans_loss = 0
    batch_rot_vf_loss = 0
    batch_bb_atom_loss = 0
    batch_pair_loss = 0
    batch_plm_emb_loss = 0
    batch_plm_cos_loss = 0
    batch_seqs_loss = 0

    loss_weights = config.train.loss_weights
    task_name = 'normal'

    while True:
        if it == config.train.max_iters:
            break
        batch = recursive_to(next(train_iterator), device)  # 一次性转移到device上
        b, _ = batch['aa'].shape

        # 若缺失主链原子，则不需要生成
        generate_mask = (batch['mask_heavyatom'][:, :, :3].sum(dim=-1) == 3.0)
        batch['generate_mask'] = torch.logical_and(generate_mask, batch['res_mask'])  # res_mask标记了哪里进行了填充

        # 用于控制重链和轻链损失的单独计算，“is_heavy”即表示重链位置
        batch['heavy_loss_mask'] = batch['is_heavy']
        batch['light_loss_mask'] = torch.logical_and(~(batch['is_heavy'].bool()), batch['res_mask'])

        use_checkpoint = False
        # if len(batch['aa'][0]) > 230:
        #     use_checkpoint = True
        # else:
        #     use_checkpoint = False

        # 时间步采样
        t = torch.rand((b, 1), device=batch['aa'].device)
        t = t * (1 - 2 * config.model.interpolant.min_t) + config.model.interpolant.min_t  # 避免时间步为0
        batch['t'] = t

        if random.random() > 0.5:
            with torch.no_grad():
                output = model(batch)
                # 获取自条件
                batch['pred_trans_1'] = output['pred_trans_1']
                batch['pred_bb_atom'] = output['pred_bb_atom']
        output = model(batch, use_checkpoint=use_checkpoint)
        loss = (loss_weights.trans_loss * output['trans_loss'] + loss_weights.rot_vf_loss * output['rot_vf_loss'] +
                loss_weights.bb_atom_loss * output['bb_atom_loss'] + loss_weights.pair_loss * output['pair_loss'] +
                loss_weights.plm_emb_loss * output['plm_emb_loss'] + loss_weights.plm_cos_loss * output['plm_cos_loss'] +
                loss_weights.seqs_loss * output['seqs_loss']) / opti_num
        loss.backward()

        batch_count += config.train.batch_size
        batch_loss += loss.item()
        batch_trans_loss += output['trans_loss'].item() / opti_num
        batch_rot_vf_loss += output['rot_vf_loss'].item() / opti_num
        batch_bb_atom_loss += output['bb_atom_loss'].item() / opti_num
        batch_pair_loss += output['pair_loss'].item() / opti_num
        batch_plm_emb_loss += output['plm_emb_loss'].item() / opti_num
        batch_plm_cos_loss += output['plm_cos_loss'].item() / opti_num
        batch_seqs_loss += output['seqs_loss'].item() / opti_num

        if batch_count % config.train.pseudo_batch_size == 0:
            # 梯度裁剪，并记录梯度的范数情况。max_norm=float('inf') 表示不进行裁剪，只计算范数。
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            # ema.update()
            if it % config.train.log_fq == 0:
                task_name = 'normal'
                log_step = normal_log_step
                normal_log_step += 1

                writer.add_scalar("%s_train_struc/trans_loss" % task_name, batch_trans_loss, log_step)
                writer.add_scalar("%s_train_struc/rot_vf_loss" % task_name, batch_rot_vf_loss, log_step)
                writer.add_scalar("%s_train_struc/bb_atom_loss" % task_name, batch_bb_atom_loss, log_step)
                writer.add_scalar("%s_train_struc/pair_loss" % task_name, batch_pair_loss, log_step)
                writer.add_scalar("%s_train_seq/plm_loss" % task_name, batch_plm_emb_loss, log_step)
                writer.add_scalar("%s_train_seq/plm_cos_loss" % task_name, batch_plm_cos_loss, log_step)
                writer.add_scalar("%s_train_seq/seqs_loss" % task_name, batch_seqs_loss, log_step)
                writer.add_scalar("%s_train_loss/loss" % task_name, batch_loss, log_step)
                writer.add_scalar("%s_train_loss/grad_norm" % task_name, grad_norm.item(), log_step)
                log_step += 1

            it += 1
            batch_count = 0
            task_num += 1

            lr_scheduler.step(it)

            """使用验证集验证模型性能"""
            if it % config.train.val_freq == 0 and batch_count == 0:
                val_batch_loss = torch.tensor(0., device=device)
                val_batch_trans_loss = 0
                val_batch_rot_vf_loss = 0
                val_batch_bb_atom_loss = 0
                val_batch_pair_loss = 0
                val_batch_plm_loss = 0
                val_batch_plm_cos_loss = 0
                val_batch_seqs_loss = 0

                # eval前，将影子权重应用到模型中
                # ema.eval()
                model.eval()
                val_batch_size = len(val_dataset)
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = recursive_to(val_batch, device)
                        b, _ = val_batch['aa'].shape

                        generate_mask = (val_batch['mask_heavyatom'][:, :, :3].sum(dim=-1) == 3.0)
                        val_batch['generate_mask'] = torch.logical_and(generate_mask, val_batch['res_mask'])

                        val_batch['heavy_loss_mask'] = val_batch['is_heavy']
                        val_batch['light_loss_mask'] = torch.logical_and(~(val_batch['is_heavy'].bool()), val_batch['res_mask'])

                        t = torch.rand((b, 1), device=val_batch['aa'].device)
                        t = t * (1 - 2 * config.model.interpolant.min_t) + config.model.interpolant.min_t  # 避免时间步为0
                        val_batch['t'] = t

                        val_output = model(val_batch)
                        val_batch['pred_trans_1'] = val_output['pred_trans_1']
                        val_batch['pred_bb_atom'] = val_output['pred_bb_atom']

                        val_output = model(val_batch, eval=True)

                        val_loss = (loss_weights.trans_loss * val_output['trans_loss'] + loss_weights.rot_vf_loss * val_output['rot_vf_loss'] +
                                    loss_weights.bb_atom_loss * val_output['bb_atom_loss'] + loss_weights.pair_loss * val_output['pair_loss'] +
                                    loss_weights.plm_emb_loss * val_output['plm_emb_loss'] + loss_weights.plm_cos_loss * output['plm_cos_loss'] +
                                    loss_weights.seqs_loss * val_output['seqs_loss']) / val_batch_size
                        val_batch_loss += val_loss
                        val_batch_trans_loss += val_output['trans_loss'].item() / val_batch_size
                        val_batch_rot_vf_loss += val_output['rot_vf_loss'].item() / val_batch_size
                        val_batch_bb_atom_loss += val_output['bb_atom_loss'].item() / val_batch_size
                        val_batch_pair_loss += val_output['pair_loss'].item() / val_batch_size
                        val_batch_plm_loss += val_output['plm_emb_loss'].item() / val_batch_size
                        val_batch_plm_cos_loss += val_output['plm_cos_loss'].item() / val_batch_size
                        val_batch_seqs_loss += val_output['seqs_loss'].item() / val_batch_size
                    writer.add_scalar("val/loss", val_batch_loss.item(), val_log_step)
                    writer.add_scalar("val/trans_loss", val_batch_trans_loss, val_log_step)
                    writer.add_scalar("val/rot_vf_loss", val_batch_rot_vf_loss, val_log_step)
                    writer.add_scalar("val/bb_atom_loss", val_batch_bb_atom_loss, val_log_step)
                    writer.add_scalar("val/pair_loss", val_batch_pair_loss, val_log_step)
                    writer.add_scalar("val/plm_loss", val_batch_plm_loss, val_log_step)
                    writer.add_scalar("val/plm_cos_loss", val_batch_plm_cos_loss, val_log_step)
                    writer.add_scalar("val/seqs_loss", val_batch_seqs_loss, val_log_step)
                    writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], val_log_step)

                # lr_scheduler.step(val_batch_loss)

                if val_batch_loss < best_loss:
                    # torch.save(ema.shadow.state_dict(), './save_model/model.pt')
                    torch.save(model.state_dict(), './save_model/model.pt')
                    best_loss = val_batch_loss
                checkpoint_dict = {
                    # 'model_state_dict': ema.shadow.state_dict(),
                    'last_model_state_dict': model.state_dict(),
                    'it': it,
                    'task_num': task_num, 
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'normal_log_step': normal_log_step,
                    'cdr_log_step': cdr_log_step,
                    'log_name': log_name,
                    'val_log_step': val_log_step,
                    'best_loss': best_loss,
                    'random_state': random.getstate(),
                    'np_state': np.random.get_state(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all()
                }
                torch.save(checkpoint_dict, './save_model/checkpoint.ckpt')
                # eval之后，恢复原来模型的参数
                # ema.train()
                model.train()

                val_log_step += 1

            pbar.update(1)
            pbar.set_description_str("trained it:%s, loss:%.4f" % (it, batch_loss))

            batch_loss = 0
            batch_trans_loss = 0
            batch_rot_vf_loss = 0
            batch_bb_atom_loss = 0
            batch_bb_fape_loss = 0
            batch_pair_loss = 0
            batch_plm_emb_loss = 0
            batch_plm_cos_loss = 0
            batch_seqs_loss = 0
            batch_acc = 0
            batch_contrast_loss = 0
