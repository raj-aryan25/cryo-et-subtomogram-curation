"""Phase 3: tiny-GN CNN fold runner on Modal A10G.

Launch: modal run research/modal_m3.py --fold 0 --epochs 60
Saves: /ckpt/m3_fold{k}.pt, /ckpt/m3_val{k}.npz, /ckpt/m3_test{k}.npz,
       /ckpt/m3_report{k}.json (kill verdict + tomo probe + val metrics).
Kill by ~epoch 30 of fold 0 if: val macro < M2+0.03 with big train-val gap,
or tomo-probe high while val flat.
"""
import modal

app = modal.App("subtomo-m3")
img = (modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
       .pip_install("numpy", "pandas", "scikit-learn", "scipy")
       .add_local_dir("research", remote_path="/opt/working"))
vol = modal.Volume.from_name("subtomo", create_if_missing=True)


@app.function(image=img, gpu="A10G", volumes={"/vol": vol}, timeout=7200)
def train_fold(fold: int = 0, epochs: int = 60, seed: int = 42):
    import sys
    sys.path.insert(0, "/opt/working")
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from model_m3 import TinyNet, TinyDataset, count_params

    torch.manual_seed(seed)
    np.random.seed(seed)
    D = "/vol/data"
    train = pd.read_csv(f"{D}/train.csv")
    SPECIES = ["ribosome_60s", "proteasome_20s", "chaperonin_open",
               "chaperonin_closed", "atpase_open", "atpase_closed",
               "ferritin_shell", "filament_seg", "membrane_frag",
               "unassigned_density"]
    CLS = {c: i for i, c in enumerate(SPECIES)}
    y = train["label"].map(CLS).values
    groups = train["tomogram_id"].values
    splits = list(GroupKFold(n_splits=3).split(np.zeros(len(y)), y, groups))
    tri, vai = splits[fold]
    print(f"fold {fold}: train {len(tri)} val {len(vai)}", flush=True)

    # M2 reference on the SAME split (image-29 feats from shipped cache)
    z = np.load("/opt/working/cache_m2.npz")
    names = list(z["names"])
    tab = ["defocus_um", "tilt_max_deg", "az_sin", "az_cos", "ice_thickness_nm"]
    cc = [i for i, n in enumerate(names) if n not in tab]
    sc = StandardScaler().fit(z["X_train"][tri][:, cc])
    m2 = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                            random_state=42)
    m2.fit(sc.transform(z["X_train"][tri][:, cc]), y[tri])
    m2_macro = f1_score(y[vai], m2.predict(sc.transform(z["X_train"][vai][:, cc])),
                        average="macro", zero_division=0)
    print(f"M2 same-split val macro: {m2_macro:.4f}", flush=True)

    tr_vols = np.load(f"{D}/train_volumes.npy", mmap_mode="r")
    te_vols = np.load(f"{D}/test_volumes.npy", mmap_mode="r")
    ds_tr = TinyDataset(tr_vols, y[tri], tri, train=True, seed=seed + fold * 1000,
                        donor_idx=tri)
    ds_va = TinyDataset(tr_vols, y[vai], vai, train=False, seed=seed)
    ds_te = TinyDataset(te_vols, np.zeros(len(te_vols), dtype=np.int64),
                        np.arange(len(te_vols)), train=False, seed=seed)
    dl_tr = DataLoader(ds_tr, batch_size=32, shuffle=True, num_workers=2,
                       pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=64, shuffle=False, num_workers=2,
                       pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=64, shuffle=False, num_workers=2,
                       pin_memory=True)

    dev = torch.device("cuda")
    net = TinyNet().to(dev)
    print("params:", count_params(net), flush=True)
    cnt = np.bincount(y[tri], minlength=10).astype(np.float64)
    w = len(tri) / (10 * cnt)
    w10 = torch.tensor(w, dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    @torch.no_grad()
    def run_eval(loader, collect=False):
        net.eval()
        Ps, Ys, Gs = [], [], []
        with torch.amp.autocast("cuda"):
            for xb, yb in loader:
                logits, gap = net(xb.to(dev))
                Ps.append(logits.float().cpu().numpy())
                Gs.append(gap.float().cpu().numpy())
                Ys.append(yb.numpy())
        P, Y, G = np.concatenate(Ps), np.concatenate(Ys), np.concatenate(Gs)
        macro = f1_score(Y, P.argmax(1), average="macro", zero_division=0)
        return (P, Y, G, macro) if collect else macro

    best_macro, best_state, wait = -1.0, None, 0
    killed, kill_reason = False, ""
    for ep in range(epochs):
        net.train()
        tot, nstep, ncorr, ntot = 0.0, 0, 0, 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits, _ = net(xb)
                loss = F.cross_entropy(logits, yb, weight=w10, label_smoothing=0.05)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss)
            nstep += 1
            ncorr += (logits.argmax(1) == yb).sum().item()
            ntot += len(yb)
        sched.step()
        train_acc = ncorr / ntot
        macro = run_eval(dl_va)
        print(f"fold {fold} ep {ep}: loss {tot / nstep:.4f} train_acc "
              f"{train_acc:.3f} val_macro {macro:.4f}", flush=True)
        if macro > best_macro:
            best_macro, best_state, wait = macro, \
                {k: v.cpu() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
        # kill check at epoch 30
        if ep == 30:
            gap = train_acc - macro
            if best_macro < m2_macro + 0.03 and gap > 0.35:
                killed, kill_reason = True, \
                    f"val {best_macro:.3f} < M2 {m2_macro:.3f}+0.03 with gap {gap:.2f}"
                print("KILL: " + kill_reason, flush=True)
                break
            print(f"survive-check ep30: best {best_macro:.4f} vs M2 {m2_macro:.4f}, "
                  f"gap {gap:.2f} -> CONTINUE", flush=True)
        if wait >= 12:
            print(f"early stop at ep {ep}", flush=True)
            break

    report = {"fold": fold, "m2_macro": float(m2_macro),
              "best_macro": float(best_macro), "killed": killed,
              "kill_reason": kill_reason}
    if not killed and best_state is not None:
        net.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
        Pv, Yv, Gv, _ = run_eval(dl_va, collect=True)
        # tomo-ID probe on GAP embeddings (val tomos only)
        tomos_vai = groups[vai]
        uniq = np.unique(tomos_vai)
        if len(uniq) > 3:
            yt = np.array([list(uniq).index(t) for t in tomos_vai])
            probe = LogisticRegression(max_iter=1000).fit(Gv, yt)
            probe_acc = float(probe.score(Gv, yt))
            report["tomo_probe_acc"] = probe_acc
            report["tomo_probe_chance"] = float(1 / len(uniq))
            print(f"tomo probe acc {probe_acc:.3f} vs chance {1/len(uniq):.3f}",
                  flush=True)
        # full grader metric on val with maxprob selective
        import sys as _s
        _s.path.insert(0, "/opt/working")
        from metrics import evaluate
        m = evaluate(np.array([SPECIES[i] for i in Yv]),
                     np.array(SPECIES)[Pv.argmax(1)],
                     torch.softmax(torch.tensor(Pv), 1).max(1).values.numpy(),
                     np.array([f"p_{i:06d}" for i in vai]))
        report["val_total"] = m["total"]
        report["val_macro"] = m["macro_f1"]
        report["val_conf"] = m["conformer_acc"]
        report["val_minf1"] = m["min_class_f1"]
        report["val_sel"] = m["selective_accuracy"]
        print("val metrics: " + str({k: round(v, 4) for k, v in m.items()
                                     if isinstance(v, float)}), flush=True)
        # test logits
        net.eval()
        Pt = []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for xb, _ in dl_te:
                Pt.append(net(xb.to(dev))[0].float().cpu().numpy())
        Pt = np.concatenate(Pt)
        import os
        os.makedirs("/vol/ckpt", exist_ok=True)
        torch.save(best_state, f"/vol/ckpt/m3_fold{fold}.pt")
        np.savez_compressed(f"/vol/ckpt/m3_val{fold}.npz", logits=Pv, idx=vai)
        np.savez_compressed(f"/vol/ckpt/m3_test{fold}.npz", logits=Pt)
        vol.commit()
    import json
    with open(f"/vol/ckpt/m3_report{fold}.json", "w") as f:
        json.dump(report, f, indent=1)
    vol.commit()
    print("DONE " + str(report), flush=True)
    return report


@app.local_entrypoint()
def main(fold: int = 0, epochs: int = 60, seed: int = 42):
    print(train_fold.remote(fold, epochs, seed))
