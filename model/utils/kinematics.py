import numpy as np
import torch
from .chemical import INIT_CRDS

PARAMS = {
    "DMIN"    : 2.0,
    "DMAX"    : 20.0,
    "DBINS"   : 36,
    "ABINS"   : 36,
}

# ============================================================
def get_pair_dist(a, b):
    """计算两组点之间的成对距离
    
    Parameters
    ----------
    a,b : pytorch tensors of shape [batch,nres,3]
          store Cartesian coordinates of two sets of atoms
    Returns
    -------
    dist : pytorch tensor of shape [batch,nres,nres]
           stores paitwise distances between atoms in a and b
    """

    dist = torch.cdist(a, b, p=2)
    return dist

# ============================================================
def get_ang(a, b, c):
    """计算所有连续三元组 (a[i], b[i], c[i]) 的平面角，这些三元组来自三组原子 a、b 和 c 的笛卡尔坐标。

    Parameters
    ----------
    a,b,c : pytorch tensors of shape [batch,nres,3]
            store Cartesian coordinates of three sets of atoms
    Returns
    -------
    ang : pytorch tensor of shape [batch,nres]
          stores resulting planar angles
    """
    v = a - b
    w = c - b
    v /= torch.norm(v, dim=-1, keepdim=True)
    w /= torch.norm(w, dim=-1, keepdim=True)
    vw = torch.sum(v*w, dim=-1)

    return torch.acos(vw)

# ============================================================
def get_dih(a, b, c, d):
    """calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d

    Parameters
    ----------
    a,b,c,d : pytorch tensors of shape [batch,nres,3]
              store Cartesian coordinates of four sets of atoms
    Returns
    -------
    dih : pytorch tensor of shape [batch,nres]
          stores resulting dihedrals
    """
    b0 = a - b
    b1 = c - b
    b2 = d - c

    b1 /= torch.norm(b1, dim=-1, keepdim=True)

    v = b0 - torch.sum(b0*b1, dim=-1, keepdim=True)*b1
    w = b2 - torch.sum(b2*b1, dim=-1, keepdim=True)*b1

    x = torch.sum(v*w, dim=-1)
    y = torch.sum(torch.cross(b1,v,dim=-1)*w, dim=-1)

    return torch.atan2(y, x)


# ============================================================
def xyz_to_c6d(xyz, params=PARAMS):
    """convert cartesian coordinates into 2d distance 
    and orientation maps
    
    Parameters
    ----------
    xyz : pytorch tensor of shape [batch,nres,3,3]
          stores Cartesian coordinates of backbone N,Ca,C atoms
    Returns
    -------
    c6d : pytorch tensor of shape [batch,nres,nres,4]
          stores stacked dist,omega,theta,phi 2D maps 
    """

    batch = xyz.shape[0]
    nres = xyz.shape[1]

    # three anchor atoms
    N = xyz[:, :, 0]
    Ca = xyz[:, :, 1]
    C = xyz[:, :, 2]

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

    # 6d coordinates order: (dist,omega,theta,phi)
    c6d = torch.zeros([batch, nres, nres, 4], dtype=xyz.dtype, device=xyz.device)

    dist = get_pair_dist(Cb, Cb)
    dist[torch.isnan(dist)] = 999.9
    c6d[..., 0] = dist + 999.9 * torch.eye(nres, device=xyz.device)[None, ...]
    b, i, j = torch.where(c6d[..., 0] < params['DMAX'])

    c6d[b, i, j, torch.full_like(b, 1)] = get_dih(Ca[b, i], Cb[b, i], Cb[b, j], Ca[b, j])
    c6d[b, i, j, torch.full_like(b, 2)] = get_dih(N[b, i], Ca[b, i], Cb[b, i], Cb[b, j])
    c6d[b, i, j, torch.full_like(b, 3)] = get_ang(Ca[b, i], Cb[b, i], Cb[b, j])

    # fix long-range distances
    c6d[..., 0][c6d[..., 0] >= params['DMAX']] = 999.9

    mask = torch.zeros((batch, nres, nres), dtype=xyz.dtype, device=xyz.device)
    mask[b, i, j] = 1.0
    return c6d, mask


def xyz_to_t2d(xyz_t, params=PARAMS):
    """将模板的笛卡尔坐标转换为二维距离和方向图
    
    Parameters
    ----------
    xyz_t : 形状为 [batch, templ, nres, 3, 3] 的 PyTorch 张量存储了模板主链 N、Ca 和 C 原子的笛卡尔坐标。

    Returns
    -------
    t2d : 形状为 [batch, nres, nres, 37 + 6 + 3] 的 PyTorch 张量存储了堆叠的距离（dist）、欧米伽角（omega）、theta角和phi角的二维图
    """
    B, T, L = xyz_t.shape[:3]
    c6d, mask = xyz_to_c6d(xyz_t[:, :, :, :3].view(B * T, L, 3, 3), params=params)
    c6d = c6d.view(B, T, L, L, 4)
    mask = mask.view(B, T, L, L, 1)
    #
    # 距离到one-hot编码
    dist = dist_to_onehot(c6d[..., 0], params)
    orien = torch.cat((torch.sin(c6d[..., 1:]), torch.cos(c6d[..., 1:])), dim=-1) * mask  # (B, T, L, L, 6)
    #
    mask = ~torch.isnan(c6d[:, :, :, :, 0])  # (B, T, L, L)
    t2d = torch.cat((dist, orien, mask.unsqueeze(-1)), dim=-1)
    t2d[torch.isnan(t2d)] = 0.0
    return t2d


def xyz_to_chi1(xyz_t):
    '''convert template cartesian coordinates into chi1 angles

    Parameters
    ----------
    xyz_t: pytorch tensor of shape [batch, templ, nres, 14, 3]
           stores Cartesian coordinates of template atoms. For missing atoms, it should be NaN

    Returns
    -------
    chi1 : pytorch tensor of shape [batch, templ, nres, 2]
           stores cos and sin chi1 angle
    '''
    B, T, L = xyz_t.shape[:3]
    xyz_t = xyz_t.reshape(B*T, L, 14, 3)
        
    # chi1 angle: N, CA, CB, CG
    chi1 = get_dih(xyz_t[:,:,0], xyz_t[:,:,1], xyz_t[:,:,4], xyz_t[:,:,5]) # (B*T, L)
    cos_chi1 = torch.cos(chi1)
    sin_chi1 = torch.sin(chi1)
    mask_chi1 = ~torch.isnan(chi1)
    chi1 = torch.stack((cos_chi1, sin_chi1, mask_chi1), dim=-1) # (B*T, L, 3)
    chi1[torch.isnan(chi1)] = 0.0
    chi1 = chi1.reshape(B, T, L, 3)
    return chi1


def xyz_to_bbtor(xyz, params=PARAMS):
    batch = xyz.shape[0]
    nres = xyz.shape[1]

    # three anchor atoms
    N  = xyz[:,:,0]
    Ca = xyz[:,:,1]
    C  = xyz[:,:,2]

    # recreate Cb given N,Ca,C
    next_N = torch.roll(N, -1, dims=1)
    prev_C = torch.roll(C, 1, dims=1)
    phi = get_dih(prev_C, N, Ca, C)
    psi = get_dih(N, Ca, C, next_N)
    #
    phi[:,0] = 0.0
    psi[:,-1] = 0.0
    #
    astep = 2.0*np.pi / params['ABINS']
    phi_bin = torch.round((phi+np.pi-astep/2)/astep)
    psi_bin = torch.round((psi+np.pi-astep/2)/astep)
    return torch.stack([phi_bin, psi_bin], axis=-1).long()

# ============================================================
def dist_to_onehot(dist, params=PARAMS):
    dist[torch.isnan(dist)] = 999.9
    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    dbins = torch.linspace(params['DMIN']+dstep, params['DMAX'], params['DBINS'],dtype=dist.dtype,device=dist.device)
    db = torch.bucketize(dist.contiguous(),dbins).long()
    dist = torch.nn.functional.one_hot(db, num_classes=params['DBINS']+1).float()
    return dist

def c6d_to_bins(c6d,params=PARAMS):
    """bin 2d distance and orientation maps
    """

    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    astep = 2.0 * np.pi / params['ABINS']

    dbins = torch.linspace(params['DMIN'] + dstep, params['DMAX'], params['DBINS'], dtype=c6d.dtype, device=c6d.device)
    ab360 = torch.linspace(-np.pi + astep, np.pi, params['ABINS'], dtype=c6d.dtype, device=c6d.device)
    ab180 = torch.linspace(astep, np.pi, params['ABINS'] // 2, dtype=c6d.dtype, device=c6d.device)

    db = torch.bucketize(c6d[..., 0].contiguous(), dbins)
    ob = torch.bucketize(c6d[..., 1].contiguous(), ab360)
    tb = torch.bucketize(c6d[..., 2].contiguous(), ab360)
    pb = torch.bucketize(c6d[..., 3].contiguous(), ab180)

    ob[db == params['DBINS']] = params['ABINS']
    tb[db == params['DBINS']] = params['ABINS']
    pb[db == params['DBINS']] = params['ABINS'] // 2

    return torch.stack([db, ob, tb, pb], axis=-1).to(torch.uint8)


# ============================================================
def dist_to_bins(dist,params=PARAMS):
    """bin 2d distance maps
    """

    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    db = torch.round((dist-params['DMIN']-dstep/2)/dstep)

    db[db<0] = 0
    db[db>params['DBINS']] = params['DBINS']
    
    return db.long()


# ============================================================
def c6d_to_bins2(c6d, same_chain, negative=False, params=PARAMS):
    """bin 2d distance and orientation maps
    """

    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    astep = 2.0*np.pi / params['ABINS']

    db = torch.round((c6d[...,0]-params['DMIN']-dstep/2)/dstep)
    ob = torch.round((c6d[...,1]+np.pi-astep/2)/astep)
    tb = torch.round((c6d[...,2]+np.pi-astep/2)/astep)
    pb = torch.round((c6d[...,3]-astep/2)/astep)

    # put all d<dmin into one bin
    db[db<0] = 0
    
    # synchronize no-contact bins
    db[db>params['DBINS']] = params['DBINS']
    ob[db==params['DBINS']] = params['ABINS']
    tb[db==params['DBINS']] = params['ABINS']
    pb[db==params['DBINS']] = params['ABINS']//2
    
    if negative:
        db = torch.where(same_chain.bool(), db.long(), params['DBINS'])
        ob = torch.where(same_chain.bool(), ob.long(), params['ABINS'])
        tb = torch.where(same_chain.bool(), tb.long(), params['ABINS'])
        pb = torch.where(same_chain.bool(), pb.long(), params['ABINS']//2)
    
    return torch.stack([db, ob, tb, pb], axis=-1).long()


def get_init_xyz(xyz_t):
    # input: xyz_t (B, T, L, 27, 3)。
    # ouput: xyz (B, T, L, 27, 3)。
    B, T, L = xyz_t.shape[:3]
    init = INIT_CRDS.to(xyz_t.device).reshape(1, 1, 1, 27, 3).repeat(B, T, L, 1, 1)  # 理想的 N、CA、C 初始坐标
    # 扩散的初始阶段直接返回理想坐标
    if torch.isnan(xyz_t).all():
        return init

    mask = torch.isnan(xyz_t[:, :, :, :3]).any(dim=-1).any(dim=-1)  # (B, T, L)
    # xyz_t[:, :, :, 1, :]表示CA原子的坐标。求出该蛋白质Cα的质心坐标
    center_CA = ((~mask[:, :, :, None]) * torch.nan_to_num(xyz_t[:, :, :, 1, :])).sum(dim=2) / ((~mask[:, :, :, None]).sum(dim=2)+1e-4)  # (B, T, 3)
    xyz_t = xyz_t - center_CA.view(B, T, 1, 1, 3)  # 平移
    #
    # idx_s = list()
    for i_b in range(B):
        for i_T in range(T):
            if mask[i_b, i_T].all():
                continue
            # 获取有效残基索引
            exist_in_templ = torch.where(~mask[i_b, i_T])[0]  # (L_sub)
            # 为每个缺失的残基位置找到最近的有效模板残基
            seqmap = (torch.arange(L, device=xyz_t.device)[:,None] - exist_in_templ[None,:]).abs() # (L, L_sub)
            seqmap = torch.argmin(seqmap, dim=-1)  # (L)
            # torch.gather(input, dim, index, out=None)，用于从源张量中按照指定的索引收集元素。
            idx = torch.gather(exist_in_templ, -1, seqmap)  # (L)
            # 对齐理想坐标
            offset_CA = torch.gather(xyz_t[i_b, i_T, :, 1, :], 0, idx.reshape(L, 1).expand(-1, 3))
            init[i_b, i_T] += offset_CA.reshape(L, 1, 3)
    #
    xyz = torch.where(mask.view(B, T, L, 1, 1), init, xyz_t)
    return xyz


def center_and_realign_missing(xyz, mask_t):
    # xyz: (L, 27, 3)
    # mask_t: (L, 27)
    L = xyz.shape[0]

    mask = mask_t[:, :3].all(dim=-1)  # True for valid atom (L)

    # center c.o.m at the origin
    center_CA = (mask[..., None] * xyz[:, 1]).sum(dim=0) / (mask[..., None].sum(dim=0) + 1e-5)  # (3)
    xyz = torch.where(mask.view(L, 1, 1), xyz - center_CA.view(1, 1, 3), xyz)

    # move missing residues to the closest valid residues
    exist_in_xyz = torch.where(mask)[0]  # L_sub
    seqmap = (torch.arange(L, device=xyz.device)[:, None] - exist_in_xyz[None, :]).abs()  # (L, Lsub)
    seqmap = torch.argmin(seqmap, dim=-1)  # L
    idx = torch.gather(exist_in_xyz, 0, seqmap)
    offset_CA = torch.gather(xyz[:, 1], 0, idx.reshape(L, 1).expand(-1, 3))
    xyz = torch.where(mask.view(L, 1, 1), xyz, xyz + offset_CA.reshape(L, 1, 3))

    return xyz
