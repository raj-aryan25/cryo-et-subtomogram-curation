"""M1 training on Modal A10G. One function call per CV fold.

Setup (once):  modal volume put subtomo data /data
Launch fold:  modal run research/modal_m1.py --fold 0 --epochs 30
Outputs in volume: /ckpt/fold{k}.pt, /ckpt/val_fold{k}.npz (val probs+idx), /ckpt/test_fold{k}.npz
"""
import modal

app = modal.App("subtomo-m1")
img = (modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
       .pip_install("numpy", "pandas", "scikit-learn", "scipy")
       .add_local_dir("research", remote_path="/opt/working"))
vol = modal.Volume.from_name("subtomo", create_if_missing=True)


@app.function(image=img, gpu="A10G", volumes={"/vol": vol},
              timeout=7200)
def train_fold(fold: int = 0, epochs: int = 60, seed: int = 42,
               lr: float = 2e-3, wd: float = 3e-4, drop: float = 0.3,
               ls: float = 0.10, distill_w: float = 0.0, distill_t: float = 2.0,
               radial_dim: int = 14, widths: str = "16,32,64,128", emb: int = 128,
               norm: str = "gn", sched: str = "cos", patience: int = 10,
               resid: int = 0):
    import sys
    sys.path.insert(0, "/opt/working")
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score
    from model_m1 import SEResNet3D, VolDataset, COARSE, SPECIES, CLS

    torch.manual_seed(seed)
    np.random.seed(seed)
    D = "/vol/data"
    train = pd.read_csv(f"{D}/train.csv")
    y = train["label"].map(CLS).values
    groups = train["tomogram_id"].values
    from sklearn.model_selection import GroupKFold
    splits = list(GroupKFold(n_splits=3).split(np.zeros(len(y)), y, groups))
    tri, vai = splits[fold]
    print(f"fold {fold}: train {len(tri)} val {len(vai)}", flush=True)

    tr_vols = np.load(f"{D}/train_volumes.npy", mmap_mode="r")
    te_vols = np.load(f"{D}/test_volumes.npy", mmap_mode="r")
    # radial side-input (gain-invariant rotation profile, first `radial_dim` cols)
    _z = np.load("/opt/working/cache_m2.npz")
    FE_TR = np.ascontiguousarray(_z["X_train"][:, :radial_dim], dtype=np.float32)
    FE_TE = np.ascontiguousarray(_z["X_test"][:, :radial_dim], dtype=np.float32)
    print("radial feats", FE_TR.shape, FE_TE.shape, flush=True)
    # residual mode: frozen M2 log-prob offsets (train/val use OOF rows = unbiased)
    OFF_TR, OFF_TE = None, None
    if resid:
        _o = np.load("/opt/working/oof_m2.npz")["oof"]
        OFF_TR = np.log(np.asarray(_o, dtype=np.float64) + 1e-6).astype(np.float32)
        _t = np.load("/opt/working/test_m2_probs.npy")
        OFF_TE = np.log(np.asarray(_t, dtype=np.float64) + 1e-6).astype(np.float32)
        print("residual offsets on", OFF_TR.shape, OFF_TE.shape, flush=True)

    # sqrt-inverse-frequency class weights from train fold
    cnt = np.bincount(y[tri], minlength=10).astype(np.float64)
    w = 1.0 / np.sqrt(cnt + 1.0)
    w = w / w.mean()
    w10 = torch.tensor(w, dtype=torch.float32).cuda()
    w8 = torch.tensor([w[0], w[1], (w[2] + w[3]) / 2, (w[4] + w[5]) / 2,
                       w[6], w[7], w[8], w[9]], dtype=torch.float32).cuda()

    ds_tr = VolDataset(tr_vols, y[tri], tri, train=True, seed=seed + fold * 1000,
                       wedge_p=0.5, erase_p=0.6, ret_pos=True, feats=FE_TR)
    ds_va = VolDataset(tr_vols, y[vai], vai, train=False, seed=seed, feats=FE_TR)
    n_te = len(te_vols)
    ds_te = VolDataset(te_vols, np.zeros(n_te, dtype=np.int64), np.arange(n_te),
                       train=False, seed=seed, feats=FE_TE)
    dl_tr = DataLoader(ds_tr, batch_size=96, shuffle=True, num_workers=2,
                       pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=128, shuffle=False, num_workers=2,
                       pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=128, shuffle=False, num_workers=2,
                       pin_memory=True)

    dev = torch.device("cuda")
    W = tuple(int(x) for x in widths.split(","))
    net = SEResNet3D(widths=W, emb=emb, drop=drop, radial_dim=radial_dim,
                     norm=norm).to(dev)
    print(f"arch widths={W} emb={emb} norm={norm} sched={sched}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    if sched == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                    total_steps=epochs * len(dl_tr))
        step_each_batch = True
    else:
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=5)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - 5))
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[5])
        step_each_batch = False
    scaler = torch.amp.GradScaler("cuda")
    coarse_map = COARSE.cuda()
    # M2 OOF soft targets for distillation (rows align with train.csv order)
    teacher = None
    if distill_w > 0:
        try:
            teacher = torch.tensor(np.load("/opt/working/oof_m2.npz")["oof"][tri],
                                   dtype=torch.float32)
            print("distillation teacher loaded", tuple(teacher.shape), flush=True)
        except Exception as e:
            print("no teacher:", e, flush=True)

    def run_eval(loader, off_full=None, off_idx=None, collect=False):
        net.eval()
        all_p, all_y = [], []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for batch in loader:
                xb, yb, fb = batch[0], batch[1], batch[2]
                o = net(xb.to(dev), fb.to(dev))["logits10"].float()
                if off_full is not None:
                    n0 = len(all_y) * loader.batch_size
                    sl = slice(n0, n0 + o.shape[0])
                    o = o + torch.tensor(off_full[off_idx[sl]],
                                         dtype=torch.float32, device=dev)
                all_p.append(o.cpu().numpy())
                all_y.append(yb.numpy())
        P = np.concatenate(all_p)
        Y = np.concatenate(all_y)
        pred = P.argmax(1)
        macro = f1_score(Y, pred, average="macro", zero_division=0)
        return (P, Y, macro) if collect else macro

    best_macro, best_state, wait = -1.0, None, 0
    for ep in range(epochs):
        net.train()
        tot, nstep = 0.0, 0
        for xb, yb, fb, pos in dl_tr:
            xb, yb, fb = xb.to(dev), yb.to(dev), fb.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                o = net(xb, fb)
                l10 = o["logits10"]
                if OFF_TR is not None:
                    l10 = l10 + torch.tensor(OFF_TR[tri[pos.cpu().numpy()]],
                                             dtype=torch.float32, device=dev)
                loss = F.cross_entropy(l10, yb, weight=w10, label_smoothing=ls)
                loss = loss + 0.3 * F.cross_entropy(o["logits8"], coarse_map[yb], weight=w8)
                mchap = (yb == 2) | (yb == 3)
                if mchap.any():
                    t = (yb[mchap] == 2).float()
                    loss = loss + 0.3 * F.binary_cross_entropy_with_logits(o["chap"][mchap], t)
                matp = (yb == 4) | (yb == 5)
                if matp.any():
                    t = (yb[matp] == 4).float()
                    loss = loss + 0.3 * F.binary_cross_entropy_with_logits(o["atp"][matp], t)
                if teacher is not None:
                    tb = teacher[pos].to(dev)
                    loss = loss + distill_w * distill_t * distill_t * F.kl_div(
                        F.log_softmax(o["logits10"] / distill_t, dim=1),
                        F.softmax(tb / distill_t, dim=1), reduction="batchmean")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step_each_batch:
                sched.step()
            tot += float(loss)
            nstep += 1
        if not step_each_batch:
            sched.step()
        if OFF_TR is not None:
            macro = run_eval(dl_va, OFF_TR, vai)
        else:
            macro = run_eval(dl_va)
        print(f"fold {fold} ep {ep}: loss {tot / nstep:.4f} val_macro {macro:.4f}", flush=True)
        if macro > best_macro:
            best_macro = macro
            best_state = {k: v.cpu() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"early stop at ep {ep}", flush=True)
                break

    net.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    if OFF_TR is not None:
        Pv, Yv, _ = run_eval(dl_va, OFF_TR, vai, collect=True)
    else:
        Pv, Yv, _ = run_eval(dl_va, collect=True)
    # test probs with this fold's model
    net.eval()
    Pt = []
    import math as _math
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for bi, batch in enumerate(dl_te):
            l10 = net(batch[0].to(dev),
                      batch[2].to(dev))["logits10"].float()
            if OFF_TE is not None:
                bs = dl_te.batch_size
                sl = slice(bi * bs, bi * bs + l10.shape[0])
                l10 = l10 + torch.tensor(OFF_TE[sl], dtype=torch.float32,
                                         device=dev)
            Pt.append(l10.cpu().numpy())
    Pt = np.concatenate(Pt)
    import os
    os.makedirs("/vol/ckpt", exist_ok=True)
    torch.save(best_state, f"/vol/ckpt/fold{fold}.pt")
    np.savez_compressed(f"/vol/ckpt/val_fold{fold}.npz", probs=Pv, idx=vai)
    np.savez_compressed(f"/vol/ckpt/test_fold{fold}.npz", logits=Pt)
    vol.commit()
    print(f"DONE fold {fold} best_macro {best_macro:.4f}", flush=True)
    return best_macro


@app.local_entrypoint()
def main(fold: int = 0, epochs: int = 60, lr: float = 2e-3, wd: float = 3e-4,
         drop: float = 0.3, ls: float = 0.10, widths: str = "16,32,64,128",
         emb: int = 128, norm: str = "gn", sched: str = "cos",
         patience: int = 10, seed: int = 42, resid: int = 0):
    print(train_fold.remote(fold, epochs, seed, lr, wd, drop, ls, 0.0, 2.0, 14,
                            widths, emb, norm, sched, patience, resid))
