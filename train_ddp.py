import os
import time
import torch.distributed as dist
from tqdm import tqdm
from model.utils.utils import PaddingCollate
from model.utils.sabdab_onlyV import SAbDabDataset
from torch.utils.data import DataLoader
from model.MultiFlow import FlowModel
from model.utils.chemical import CDR
from timm.scheduler.cosine_lr import CosineLRScheduler
from model.EMA import EMA
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler  # 分布式采样器
from model.utils.utils import load_config, set_seed, recursive_to, add_weight_decay
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# 分布式训练参数
LOCAL_RANK = int(os.environ['LOCAL_RANK'])  # 当前GPU的本地rank
WORLD_SIZE = int(os.environ['WORLD_SIZE'])  # 总GPU数


# 初始化分布式训练环境
def setup_distributed():
    dist.init_process_group(
        backend="nccl",  # NVIDIA GPU推荐使用NCCL后端
        init_method="env://",
        rank=LOCAL_RANK,
        world_size=WORLD_SIZE
    )
    torch.cuda.set_device(LOCAL_RANK)  # 绑定当前进程到对应GPU


def train(local_rank):
    setup_distributed()  # 初始化分布式环境

    config, config_name = load_config('./configs/base_pretrain_ddp.yaml')
    set_seed(config.train.seed + local_rank)
    cdr_weight = config['train']['cdr_weight']

    # 设置填充工具选项，不在pad_values和no_padding上的会以0进行填充
    padding = PaddingCollate(
        pad_values={'aa': 21, 'chain_id': 4, 'mask_heavyatom': False, 'torsion_angle_mask': False},
        no_padding=['id', 'H1_seq', 'H2_seq', 'H3_seq', 'L1_seq', 'L2_seq', 'L3_seq'],
        eight=False
    )

    """加载数据集"""
    train_dataset = SAbDabDataset(summary_path=config.dataset.summary_path_dir, chothia_dir=config.dataset.structure_dir,
                                  processed_dir=config.dataset.processed_dir, embedding_h5_dir=config.dataset.embedding_dir,
                                  split='train', reset=config.dataset.reset)
    val_dataset = SAbDabDataset(summary_path=config.dataset.summary_path_dir, chothia_dir=config.dataset.structure_dir,
                                processed_dir=config.dataset.processed_dir, embedding_h5_dir=config.dataset.embedding_dir,
                                split='val', reset=config.dataset.reset)
    sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=WORLD_SIZE, rank=local_rank)  # 分布式采样器
    train_loader = DataLoader(train_dataset, batch_size=config.train.pseudo_batch_size, pin_memory=True, sampler=sampler,
                              collate_fn=padding, prefetch_factor=8, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=padding, pin_memory=True, num_workers=8)
    if local_rank == 0:
        print(len(train_dataset))
        print(len(val_dataset))

    """加载流生成模型"""
    model = FlowModel(config).to(local_rank)
    param_group = add_weight_decay(model, l2_coeff=1e-2)
    optimizer = torch.optim.AdamW(param_group, lr=config.train.lr)
    lr_scheduler = CosineLRScheduler(optimizer, warmup_t=500, t_initial=config.train.max_iters, cycle_decay=0.1, cycle_limit=5, lr_min=1e-5)
    model.train()

    # ema = EMA(model, config.train.ema_decay)

    if os.path.exists('./save_model/checkpoint.ckpt'):
        checkpoint = torch.load('./save_model/checkpoint.ckpt')
        # ema.shadow.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['last_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        try:
            epoch = checkpoint['epoch']
        except:
            epoch = 0
        it = checkpoint['it']
        task_num = checkpoint['task_num']
        normal_log_step = checkpoint['normal_log_step']
        cdr_log_step = checkpoint['cdr_log_step']
        val_log_step = checkpoint['val_log_step']
        log_name = checkpoint['log_name']
        best_loss = checkpoint['best_loss']
        del checkpoint
    else:
        # model.load_state_dict(torch.load('./save_model/model_from_frameflow.pth'))
        it = 0
        epoch = 0
        task_num = 0
        normal_log_step = 0
        cdr_log_step = 0
        val_log_step = 1
        best_loss = 999.
        log_name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True, static_graph=True)  # DDP包装
    
    if local_rank == 0:
        writer = SummaryWriter('./logs/MultiFlow/%s' % log_name)
        pbar = tqdm(range(config.train.max_iters), dynamic_ncols=True, leave=True, initial=it)
        pbar.set_description_str("trained it:0")
    else:
        writer = None
        pbar = None
        # 全部初始化为 0

    for e in range(epoch, 1000):
        if it == config.train.max_iters:
            break

        use_count = 0  # 记录使用的数据数量
        batch_loss = torch.tensor(0., device=local_rank)
        batch_trans_loss = torch.tensor(0., device=local_rank)
        batch_rot_vf_loss = torch.tensor(0., device=local_rank)
        batch_bb_atom_loss = torch.tensor(0., device=local_rank)
        batch_pair_loss = torch.tensor(0., device=local_rank)
        batch_plm_emb_loss = torch.tensor(0., device=local_rank)
        batch_plm_cos_loss = torch.tensor(0., device=local_rank)
        batch_seqs_loss = torch.tensor(0., device=local_rank)
        loss_weights = config.train.loss_weights

        sampler.set_epoch(e)
        for batch in train_loader:
            if it == config.train.max_iters:
                break
            # 训练数据
            batch = recursive_to(batch, local_rank)  # 一次性转移到device上
            b, _ = batch['aa'].shape

            # 若缺失主链原子，则不需要生成
            generate_mask = (batch['mask_heavyatom'][:, :, :3].sum(dim=-1) == 3.0)
            batch['generate_mask'] = torch.logical_and(generate_mask, batch['res_mask'])  # res_mask标记了哪里进行了填充

            # 用于控制重链和轻链损失的单独计算，“is_heavy”即表示重链位置
            batch['heavy_loss_mask'] = batch['is_heavy']
            batch['light_loss_mask'] = torch.logical_and(~(batch['is_heavy'].bool()), batch['res_mask'])

            # print('%s: %s' % (local_rank, len(batch['aa'][0])))
            use_checkpoint = True
            # if len(batch['aa'][0]) > 236:
            #     use_checkpoint = True
            # else:
            #     use_checkpoint = False

            # 时间步采样
            t = torch.rand((b, 1), device=batch['aa'].device)
            t = t * (1 - 2 * config.model.interpolant.min_t) + config.model.interpolant.min_t  # 避免时间步为0
            batch['t'] = t

            sc_flag = torch.rand((1, ), device=local_rank)
            dist.broadcast(sc_flag, src=0)
            if sc_flag > 0.5:
                with torch.no_grad():
                    with model.no_sync():
                        output = model(batch)
                        # 获取自条件
                        batch['pred_trans_1'] = output['pred_trans_1']
                        batch['pred_bb_atom'] = output['pred_bb_atom']

            output = model(batch, use_checkpoint=use_checkpoint)
            optimizer.zero_grad()
            loss = (loss_weights.trans_loss * output['trans_loss'] + loss_weights.rot_vf_loss * output['rot_vf_loss'] +
                    loss_weights.bb_atom_loss * output['bb_atom_loss'] + loss_weights.pair_loss * output['pair_loss'] +
                    loss_weights.plm_emb_loss * output['plm_emb_loss'] + loss_weights.plm_cos_loss * output['plm_cos_loss'] +
                    loss_weights.seqs_loss * output['seqs_loss'])
            loss.backward()

            use_count += config.train.pseudo_batch_size
            batch_loss += loss.item() / WORLD_SIZE
            batch_trans_loss += output['trans_loss'] / WORLD_SIZE
            batch_rot_vf_loss += output['rot_vf_loss'] / WORLD_SIZE
            batch_bb_atom_loss += output['bb_atom_loss'] / WORLD_SIZE
            batch_pair_loss += output['pair_loss'] / WORLD_SIZE
            batch_plm_emb_loss += output['plm_emb_loss'] / WORLD_SIZE
            batch_plm_cos_loss += output['plm_cos_loss'] / WORLD_SIZE
            batch_seqs_loss += output['seqs_loss'] / WORLD_SIZE

            # 梯度裁剪，并记录梯度的范数情况。max_norm=float('inf') 表示不进行裁剪，只计算范数。
            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()

            if it % config.train.log_fq == 0:
                # 定义需要同步的损失变量列表
                loss_vars = [
                    batch_loss, batch_trans_loss, batch_rot_vf_loss,
                    batch_bb_atom_loss, batch_pair_loss, batch_plm_emb_loss,
                    batch_plm_cos_loss, batch_seqs_loss,
                ]
                # 对所有损失变量进行跨进程求和
                for var in loss_vars:
                    dist.all_reduce(var, op=dist.ReduceOp.SUM)

                task_name = 'normal'
                log_step = normal_log_step
                normal_log_step += 1
                # 主进程记录日志
                if local_rank == 0:
                    writer.add_scalar("%s_train_struc/trans_loss" % task_name, batch_trans_loss, log_step)
                    writer.add_scalar("%s_train_struc/rot_vf_loss" % task_name, batch_rot_vf_loss, log_step)
                    writer.add_scalar("%s_train_struc/bb_atom_loss" % task_name, batch_bb_atom_loss, log_step)
                    writer.add_scalar("%s_train_struc/pair_loss" % task_name, batch_pair_loss, log_step)
                    writer.add_scalar("%s_train_seq/plm_loss" % task_name, batch_plm_emb_loss, log_step)
                    writer.add_scalar("%s_train_seq/plm_cos_loss" % task_name, batch_plm_cos_loss, log_step)
                    writer.add_scalar("%s_train_seq/seqs_loss" % task_name, batch_seqs_loss, log_step)
                    writer.add_scalar("%s_train/loss" % task_name, batch_loss, log_step)
                    writer.add_scalar("%s_train/grad_norm" % task_name, grad_norm.item(), log_step)
                    log_step += 1
            batch_loss = torch.tensor(0., device=local_rank)
            batch_trans_loss = torch.tensor(0., device=local_rank)
            batch_rot_vf_loss = torch.tensor(0., device=local_rank)
            batch_bb_atom_loss = torch.tensor(0., device=local_rank)
            batch_pair_loss = torch.tensor(0., device=local_rank)
            batch_plm_emb_loss = torch.tensor(0., device=local_rank)
            batch_plm_cos_loss = torch.tensor(0., device=local_rank)
            batch_seqs_loss = torch.tensor(0., device=local_rank)

            it += 1

            lr_scheduler.step(it)

            """使用验证集验证模型性能。"""
            if it % config.train.val_freq == 0:
                val_batch_loss = torch.tensor(0., device=local_rank)
                val_batch_trans_loss = 0
                val_batch_rot_vf_loss = 0
                val_batch_bb_atom_loss = 0
                val_batch_pair_loss = 0
                val_batch_plm_loss = 0
                val_batch_plm_cos_loss = 0
                val_batch_seqs_loss = 0

                model.eval()
                val_batch_size = len(val_dataset)
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = recursive_to(val_batch, local_rank)
                        b, _ = val_batch['aa'].shape

                        generate_mask = (val_batch['mask_heavyatom'][:, :, :3].sum(dim=-1) == 3.0)
                        val_batch['generate_mask'] = torch.logical_and(generate_mask, val_batch['res_mask'])

                        val_batch['heavy_loss_mask'] = val_batch['is_heavy']
                        val_batch['light_loss_mask'] = torch.logical_and(~(val_batch['is_heavy'].bool()), val_batch['res_mask'])

                        t = torch.rand((1, 1), device=val_batch['aa'].device)
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
                    if local_rank == 0:
                        writer.add_scalar("val/loss", val_batch_loss.item(), val_log_step)
                        writer.add_scalar("val/trans_loss", val_batch_trans_loss, val_log_step)
                        writer.add_scalar("val/rot_vf_loss", val_batch_rot_vf_loss, val_log_step)
                        writer.add_scalar("val/bb_atom_loss", val_batch_bb_atom_loss, val_log_step)
                        writer.add_scalar("val/pair_loss", val_batch_pair_loss, val_log_step)
                        writer.add_scalar("val/plm_loss", val_batch_plm_loss, val_log_step)
                        writer.add_scalar("val/plm_cos_loss", val_batch_plm_cos_loss, val_log_step)
                        writer.add_scalar("val/seqs_loss", val_batch_seqs_loss, val_log_step)
                        writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], val_log_step)
                        val_log_step += 1
                        
                        if val_batch_loss < best_loss:
                            best_loss = val_batch_loss
                            torch.save(model.state_dict(), './save_model/model.pt')
                        checkpoint_dict = {
                            # 'model_state_dict': ema.shadow.state_dict(),
                            'last_model_state_dict': model.module.state_dict(),
                            'epoch': e,
                            'it': it,
                            'task_num': task_num,
                            'optimizer': optimizer.state_dict(),
                            'scheduler': lr_scheduler.state_dict(),
                            'normal_log_step': normal_log_step,
                            'cdr_log_step': cdr_log_step,
                            'log_name': log_name,
                            'val_log_step': val_log_step,
                            'best_loss': best_loss,
                        }
                        torch.save(checkpoint_dict, './save_model/checkpoint.ckpt')

                model.train()
            if local_rank == 0:
                pbar.set_description_str("trained it:%s, epoch:%s" % (it, (e + 1)))
                pbar.update(1)


if __name__ == '__main__':
    # 解析命令行参数（通过torchrun启动时自动设置环境变量）
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # 输入尺寸会变，因此设置为False
    torch.backends.cudnn.benchmark = False
    # 固定cuda的随机数种子
    torch.backends.cudnn.deterministic = True

    # 启动训练函数
    train(local_rank)
    # 清理分布式进程
    dist.destroy_process_group()
