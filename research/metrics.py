"""Grader-exact evaluation harness. Copy of the challenge metric — do not modify logic."""
import numpy as np

SPECIES = ["ribosome_60s", "proteasome_20s", "chaperonin_open",
           "chaperonin_closed", "atpase_open", "atpase_closed",
           "ferritin_shell", "filament_seg", "membrane_frag",
           "unassigned_density"]
CONFORMER_CLASSES = ["chaperonin_open", "chaperonin_closed",
                     "atpase_open", "atpase_closed"]
COVERAGE_LEVELS = (0.25, 0.50, 0.75, 1.00)


def evaluate(y_true, y_pred, confidence, particle_id):
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    f1s = []
    for cls in SPECIES:
        if (y_true == cls).sum() == 0:
            continue
        tp = ((y_pred == cls) & (y_true == cls)).sum()
        fp = ((y_pred == cls) & (y_true != cls)).sum()
        fn = ((y_pred != cls) & (y_true == cls)).sum()
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2.0 * tp / denom)
    macro_f1 = float(np.mean(f1s))
    min_class_f1 = float(np.min(f1s))
    m = np.isin(y_true, CONFORMER_CLASSES)
    conformer_acc = float((y_pred[m] == y_true[m]).mean()) if m.sum() else 0.0
    correct = (y_pred == y_true)
    order = np.lexsort((np.asarray(particle_id, dtype=str),
                        -np.asarray(confidence, dtype=float)))
    accs = [float(correct[order[:max(1, int(round(c * len(correct))))]].mean())
            for c in COVERAGE_LEVELS]
    selective_accuracy = float(np.mean(accs))
    total = (0.30 * macro_f1 + 0.20 * conformer_acc
             + 0.15 * min_class_f1 + 0.35 * selective_accuracy)
    return {"macro_f1": macro_f1, "conformer_acc": conformer_acc,
            "min_class_f1": min_class_f1, "selective_accuracy": selective_accuracy,
            "cov_accs": accs, "overall_acc": float(correct.mean()), "total": total,
            "per_class_f1": dict(zip([c for c in SPECIES
                                      if (y_true == c).sum() > 0], f1s))}


def selective_curve(y_true, y_pred, confidence, particle_id):
    """Full risk-coverage detail for diagnostics."""
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    correct = (y_pred == y_true)
    order = np.lexsort((np.asarray(particle_id, dtype=str),
                        -np.asarray(confidence, dtype=float)))
    out = []
    for pct in [0.10, 0.25, 0.50, 0.75, 1.00]:
        k = max(1, int(round(pct * len(correct))))
        out.append((pct, float(correct[order[:k]].mean()), k))
    return out
