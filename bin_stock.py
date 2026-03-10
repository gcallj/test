# -*- coding: utf-8 -*-
"""bin_Stock_modelos_individuais - Converted to importable module."""

import numpy as np
import pandas as pd
import warnings, os, json, contextlib, joblib, tempfile, time, subprocess
from pathlib import Path
from tqdm.auto import tqdm
from dataclasses import dataclass
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score, confusion_matrix, precision_recall_curve,
    matthews_corrcoef
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
)
import lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
import gc
import os, json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_curve
)
import os
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from IPython.display import display
except ImportError:
    display = print

warnings.filterwarnings("ignore")
np.random.seed(42)

def clean_binary_target(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    # keep only proper binary labels; anything else becomes NaN
    s = s.where(s.isin([0, 1]))
    return s

def report_to_dict(rep):
    if rep is None:
        return None
    if isinstance(rep, dict):
        return rep
    if hasattr(rep, '_asdict'):
        return rep._asdict()
    if hasattr(rep, '__dict__'):
        return vars(rep)
    try:
        return dict(rep)
    except:
        return {"value": str(rep)}
def get_joint_thresholds(pol_metrics):
    # supports dict or object-like
    if pol_metrics is None:
        return 0.5, 0.5
    if isinstance(pol_metrics, dict):
        return float(pol_metrics.get("thr_up20", 0.5)), float(pol_metrics.get("thr_dd5", 0.5))
    return float(getattr(pol_metrics, "thr_up20", 0.5)), float(getattr(pol_metrics, "thr_dd5", 0.5))

def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # --- CORREÇÃO AQUI ---
    if callable(obj):  # Se for uma função, converte para string
        return str(obj)
    # ---------------------
    return obj
def _combine_time_and_class_weights(y, w_time=None):
    """
    Combine optional time-decay weights with class-balanced weights.
    Returns float32 weights normalized to mean=1 (stability).
    """
    y_int = np.asarray(y).astype(int)
    w_class = compute_sample_weight(class_weight="balanced", y=y_int).astype(np.float32)

    if w_time is None:
        w = w_class
    else:
        w = np.asarray(w_time, dtype=np.float32) * w_class

    m = float(np.mean(w)) if len(w) else 1.0
    if not np.isfinite(m) or m <= 1e-12:
        return w.astype(np.float32)
    return (w / m).astype(np.float32)

def _trust_cols_from_proba(proba, thr, pred=None, eps=1e-6):
    """
    Retorna:
      pred (0/1),
      trust_class  = confiança na CLASSE predita (p se pred=1, 1-p se pred=0) em [0,1]
      trust_margin = distância normalizada até o limiar (abs(p-thr)/max(thr,1-thr)) em [0,1]
    """
    p = np.asarray(proba, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    p = np.clip(p, eps, 1.0 - eps)

    thr = float(thr)
    thr = float(np.clip(thr, eps, 1.0 - eps))

    if pred is None:
        pred = (p >= thr).astype(np.int8)
    else:
        pred = np.asarray(pred).astype(np.int8)

    trust_class = np.where(pred == 1, p, 1.0 - p).astype(np.float32)

    denom = max(thr, 1.0 - thr)
    trust_margin = (np.abs(p - thr) / max(1e-12, denom)).astype(np.float32)
    trust_margin = np.clip(trust_margin, 0.0, 1.0)

    return pred.astype(np.int8), trust_class, trust_margin

def _combine_time_and_class_weights(y, w_time=None):
    """
    Combine optional time-decay weights with class-balanced weights.
    Returns float32 weights normalized to mean=1 (stability).
    """
    y_int = np.asarray(y).astype(int)
    # Compute balanced weights: n_samples / (n_classes * np.bincount(y))
    w_class = compute_sample_weight(class_weight="balanced", y=y_int).astype(np.float32)

    if w_time is None:
        w = w_class
    else:
        w = np.asarray(w_time, dtype=np.float32) * w_class

    m = float(np.mean(w)) if len(w) > 0 else 1.0
    if not np.isfinite(m) or m <= 1e-12:
        return w.astype(np.float32)

    return (w / m).astype(np.float32)

def fit_final_lgbm(params, X_fit, y_fit, w_fit, num_boost_round=600):
    # LightGBM handles scale_pos_weight internally or via weight parameter.
    # If we use _combine_time_and_class_weights, we don't need scale_pos_weight in params usually.
    # But let's respect params if provided, though typically we remove it if using sample weights.

    # Recalculate weights
    w = _combine_time_and_class_weights(y_fit, w_fit) if USE_CLASS_BALANCE_WEIGHTS else w_fit

    p = params.copy()
    # Remove scale_pos_weight if we are using balanced sample weights to avoid double correction
    if USE_CLASS_BALANCE_WEIGHTS:
        p.pop("scale_pos_weight", None)
        p.pop("is_unbalance", None)

    used_gpu = False

    # Try GPU if available
    try_gpu = GPU_AVAILABLE and (p.get("device_type") == "gpu")

    d_fit = lgb.Dataset(X_fit, label=y_fit, weight=w, free_raw_data=False)

    if try_gpu:
        try:
            with suppress_output():
                booster = lgb.train(
                    p,
                    train_set=d_fit,
                    num_boost_round=int(num_boost_round),
                    valid_sets=None,
                    callbacks=[lgb.log_evaluation(period=0)]
                )
            used_gpu = True
            return booster, used_gpu
        except Exception:
            pass # Fallback to CPU

    # CPU Fallback
    p.pop("device_type", None)
    p.pop("gpu_device_id", None)

    with suppress_output():
        booster = lgb.train(
            p,
            train_set=d_fit,
            num_boost_round=int(num_boost_round),
            valid_sets=None,
            callbacks=[lgb.log_evaluation(period=0)]
        )

    return booster, used_gpu

def fit_final_hgb_params(params, X_fit, y_fit, w_fit):
    w = _combine_time_and_class_weights(y_fit, w_fit) if USE_CLASS_BALANCE_WEIGHTS else w_fit
    p = dict(params)
    # Avoid double balancing: we already do it via sample_weight
    p.pop("class_weight", None)
    clf = HistGradientBoostingClassifier(**p)
    return fit_with_weights(clf, X_fit, y_fit.astype(int), w)

def fit_final_rf_params(params, X_fit, y_fit, w_fit):
    w = _combine_time_and_class_weights(y_fit, w_fit) if USE_CLASS_BALANCE_WEIGHTS else w_fit
    p = dict(params)
    p.pop("class_weight", None) # Avoid double counting
    clf = RandomForestClassifier(**p)
    return fit_with_weights(clf, X_fit, y_fit.astype(int), w)

def fit_final_et_params(params, X_fit, y_fit, w_fit):
    w = _combine_time_and_class_weights(y_fit, w_fit) if USE_CLASS_BALANCE_WEIGHTS else w_fit
    p = dict(params)
    p.pop("class_weight", None)
    clf = ExtraTreesClassifier(**p)
    return fit_with_weights(clf, X_fit, y_fit.astype(int), w)

def _nvidia_smi_exists() -> bool:
    try:
        r = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except Exception:
        return False

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

def _meta_path():
    return os.path.join(MODEL_DIR, f"{MODEL_TAG}_meta.json")

def _model_path(model_key, target, kind="tv"):
    ext = ".txt" if model_key == "lgbm" else ".joblib"
    return os.path.join(MODEL_DIR, f"{MODEL_TAG}_{kind}_{model_key}_{target}{ext}")

def save_model_and_meta(kind, model_key, target, model_obj, meta):
    if not SAVE_MODELS:
        return
    path = _model_path(model_key, target, kind=kind)
    if model_key == "lgbm":
        _atomic_save_lgbm(model_obj, path)
    else:
        _atomic_save_joblib(model_obj, path)
    _atomic_save_json(meta, _meta_path())
    print(f"[saved] {kind} {model_key}/{target} → {path}")

def load_artifacts(run_models, targets, kind="tv"):
    meta = None
    if os.path.exists(_meta_path()):
        with open(_meta_path(), "r") as f:
            meta = json.load(f)

    loaded = {}
    for model_key in run_models:
        for target in targets:
            path = _model_path(model_key, target, kind=kind)
            obj = None
            if os.path.exists(path):
                if model_key == "lgbm":
                    with suppress_output():
                        obj = lgb.Booster(model_file=path)
                else:
                    obj = joblib.load(path)
            loaded[(model_key, target)] = obj

    ok = sum(1 for v in loaded.values() if v is not None)
    tot = len(run_models) * len(targets)
    print(f"[load] {kind} models loaded: {ok}/{tot}")
    return meta, loaded

def _atomic_save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(path)) as tf:
        json.dump(obj, tf, indent=2)
        tmp = tf.name
    os.replace(tmp, path)

def _atomic_save_lgbm(booster, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    booster.save_model(tmp)
    os.replace(tmp, path)

def _atomic_save_joblib(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    joblib.dump(model, tmp)
    os.replace(tmp, path)

def _safe_float(fn, default=np.nan):
    try:
        return float(fn())
    except Exception:
        return float(default)

def confusion_stats(y_true, pred):
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(1, (fp + tn))
    tnr = tn / max(1, (tn + fp))
    pos_rate = pred.mean() if len(pred) else 0.0
    return tn, fp, fn, tp, float(fpr), float(tnr), float(pos_rate)

def metrics_report(y_true, proba, thr):
    y_true = np.asarray(y_true, dtype=np.int8)
    proba  = np.asarray(proba, dtype=np.float64)
    proba  = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)

    if len(y_true) == 0:
        # Retorno seguro para arrays vazios
        return {
            "AP": np.nan, "ROC_AUC": np.nan, "MCC": np.nan,
            "F1": np.nan, "Precision": np.nan, "Recall": np.nan,
            "BalancedAcc": np.nan, "FPR": np.nan,
            "PosRate": 0.0, "Confusion": [0, 0, 0, 0],
        }

    pred = (proba >= float(thr)).astype(np.int8)
    tn, fp, fn, tp, fpr, tnr, pos_rate = confusion_stats(y_true, pred)

    # CÁLCULO DO MCC
    if len(np.unique(y_true)) > 1:
        mcc = matthews_corrcoef(y_true, pred)
        ap  = average_precision_score(y_true.astype(int), proba)
        auc = roc_auc_score(y_true.astype(int), proba)
    else:
        mcc = 0.0
        ap  = np.nan
        auc = np.nan

    return {
        "AP": float(ap),
        "ROC_AUC": float(auc),
        "MCC": float(mcc),  # <--- O print_report precisa disto aqui!
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "BalancedAcc": float(balanced_accuracy_score(y_true, pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "FPR": float(fpr),
        "PosRate": float(pos_rate),
        "Confusion": [int(tn), int(fp), int(fn), int(tp)],
    }

def predict_proba_model(model_key, model_obj, X, best_iter=None):
    """
    Prediz probabilidades para qualquer modelo (LGBM/RF/HGB/ET).
    Wrapper genérico para unificar chamadas.
    """
    if model_key == "lgbm":
        num_it = int(best_iter) if best_iter is not None else model_obj.best_iteration
        with suppress_output():
            return model_obj.predict(X, num_iteration=num_it)
    else:
        # RF, HGB, ET: predict_proba[:, 1]
        return model_obj.predict_proba(X)[:, 1]

def print_report(split_name, target, model_key, rep, thr):
    tn, fp, fn, tp = rep["Confusion"]
    print(f"\n[{split_name} | {target} | {model_key}] metrics")
    print(f"  AP:           {rep['AP']:.4f}")
    print(f"  ROC-AUC:      {rep['ROC_AUC']:.4f}")
    print(f"  MCC:          {rep['MCC']:.4f}")
    print(f"  F1:           {rep['F1']:.4f}")
    print(f"  Precision:    {rep['Precision']:.4f}")
    print(f"  Recall:       {rep['Recall']:.4f}")
    print(f"  BalancedAcc:  {rep['BalancedAcc']:.4f}")
    print(f"  FPR:          {rep['FPR']:.4f}   (lower is fewer false positives)")
    print(f"  PosRate:      {rep['PosRate']:.4f} (fraction predicted positive)")
    print(f"  Confusion:    tn={tn}  fp={fp}  fn={fn}  tp={tp}")
    print(f"  Threshold:    {thr:.4f}")

def precision_at_k_per_day(index, y_true, proba, k=20):
    # index: pandas DatetimeIndex aligned with y/proba (may repeat by ticker)
    df_tmp = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(proba)}, index=pd.to_datetime(index))
    vals = []
    for d, g in df_tmp.groupby(df_tmp.index):
        g2 = g.sort_values("p", ascending=False).head(k)
        if len(g2) == 0:
            continue
        vals.append(float(g2["y"].mean()))
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))

def choose_threshold_precision_focused(y_true, proba, beta=0.5,
                                       min_precision=0.50, min_recall=0.10, max_fpr=0.25):
    y_true = np.asarray(y_true).astype(int)
    proba  = np.asarray(proba)

    if len(np.unique(y_true)) < 2:
        return 0.5

    prec, rec, thr = precision_recall_curve(y_true, proba)
    if len(thr) == 0:
        return 0.5

    # for each threshold thr[i], precision=prec[i], recall=rec[i]
    prec_t = prec[:-1]
    rec_t  = rec[:-1]

    # F_beta
    b2 = beta * beta
    denom = np.clip(b2 * prec_t + rec_t, 1e-12, None)
    fbeta = (1 + b2) * (prec_t * rec_t) / denom

    # compute FPR per threshold (vectorized loop is hard; keep small loop)
    # Use sorted unique thresholds (still manageable)
    best_idx = None
    best_score = -1.0

    def _scan_with_constraints(req_prec, req_rec, req_fpr):
        nonlocal best_idx, best_score
        best_idx = None
        best_score = -1.0
        for i, t in enumerate(thr):
            p = (proba >= t).astype(int)
            tn, fp, fn, tp, fpr, _, _ = confusion_stats(y_true, p)

            if (prec_t[i] >= req_prec) and (rec_t[i] >= req_rec) and (fpr <= req_fpr):
                if np.isfinite(fbeta[i]) and fbeta[i] > best_score:
                    best_score = float(fbeta[i])
                    best_idx = i

    # 1) strict constraints
    _scan_with_constraints(min_precision, min_recall, max_fpr)
    if best_idx is not None:
        return float(thr[best_idx])

    # 2) relax FPR first
    _scan_with_constraints(min_precision, min_recall, 1.0)
    if best_idx is not None:
        return float(thr[best_idx])

    # 3) relax precision
    _scan_with_constraints(0.0, min_recall, 1.0)
    if best_idx is not None:
        return float(thr[best_idx])

    # 4) fallback: max F_beta ignoring constraints
    idx = int(np.nanargmax(fbeta))
    return float(thr[idx])

def _quantile_threshold_candidates(proba, q_low, q_high, n=25, include=None):
    proba = np.asarray(proba, dtype=float)
    if len(proba) == 0:
        return np.array([0.5], dtype=float)

    qs = np.linspace(q_low, q_high, n)
    c = np.quantile(proba, qs)
    c = np.unique(np.clip(c, 0.0, 1.0))

    if include is not None:
        c = np.unique(np.concatenate([c, np.asarray(include, dtype=float)]))

    c = np.sort(c)
    return c

def _select_topk_positions_per_day(index, score, eligible_mask, k):
    # Returns boolean mask aligned to the original arrays
    if (k is None) or (k <= 0):
        return np.asarray(eligible_mask, dtype=bool)

    idx = pd.to_datetime(index)
    score = np.asarray(score, dtype=float)
    eligible_mask = np.asarray(eligible_mask, dtype=bool)

    df = pd.DataFrame({
        "date": idx,
        "p": score,
        "elig": eligible_mask,
        "i": np.arange(len(score), dtype=int),
    })
    df = df[df["elig"]]
    if df.empty:
        return np.zeros(len(score), dtype=bool)

    df = df.sort_values(["date", "p"], ascending=[True, False])
    pick = df.groupby("date").head(int(k))["i"].to_numpy()

    out = np.zeros(len(score), dtype=bool)
    out[pick] = True
    return out

def policy_metrics_entry_exit(index, y_up, y_dd, p_up, p_dd, thr_up, thr_dd,
                             entry_topk=20, sell_topk=None):
    """
    BUY  elig = (p_up >= thr_up) & (p_dd < thr_dd)
    SELL elig = (p_up <  thr_up) & (p_dd >= thr_dd)

    Entry is capped by Top-K per day (rank by p_up).
    Sell can be optionally capped by Top-K per day (rank by p_dd).
    """
    y_up = np.asarray(y_up).astype(int)
    y_dd = np.asarray(y_dd).astype(int)
    p_up = np.asarray(p_up, dtype=float)
    p_dd = np.asarray(p_dd, dtype=float)

    pred_up = (p_up >= thr_up)
    pred_dd = (p_dd >= thr_dd)

    # dd5 base stats (important: dd5 drives avoid/sell)
    dd5_pred = pred_dd.astype(int)
    tn, fp, fn, tp, dd5_fpr, _, _ = confusion_stats(y_dd, dd5_pred)
    dd5_prec = float(precision_score(y_dd, dd5_pred, zero_division=0))
    dd5_rec  = float(recall_score(y_dd, dd5_pred, zero_division=0))

    # BUY
    entry_elig = pred_up & (~pred_dd)
    entry_sel  = _select_topk_positions_per_day(index, p_up, entry_elig, entry_topk)
    entry_n = int(entry_sel.sum())
    entry_prec_up = float(np.mean(y_up[entry_sel])) if entry_n else np.nan
    entry_dd5_rate = float(np.mean(y_dd[entry_sel])) if entry_n else np.nan

    # signals/day
    n_days = len(pd.to_datetime(index).unique())
    entry_per_day = float(entry_n / max(1, n_days))

    # SELL
    sell_elig = (~pred_up) & pred_dd
    sell_sel  = _select_topk_positions_per_day(index, p_dd, sell_elig, sell_topk)
    sell_n = int(sell_sel.sum())
    sell_dd5_rate = float(np.mean(y_dd[sell_sel])) if sell_n else np.nan  # should be high
    sell_up_rate  = float(np.mean(y_up[sell_sel])) if sell_n else np.nan  # ideally low (up20=0 zone)
    sell_per_day  = float(sell_n / max(1, n_days))

    return {
        "thr_up20": float(thr_up),
        "thr_dd5":  float(thr_dd),

        "dd5_precision": dd5_prec,
        "dd5_recall": dd5_rec,
        "dd5_fpr": float(dd5_fpr),

        "entry_n": entry_n,
        "entry_per_day": entry_per_day,
        "entry_precision_up20": entry_prec_up,
        "entry_dd5_rate": entry_dd5_rate,

        "sell_n": sell_n,
        "sell_per_day": sell_per_day,
        "sell_dd5_rate": sell_dd5_rate,
        "sell_up20_rate": sell_up_rate,
    }

def choose_joint_thresholds_policy(index, y_up, y_dd, p_up, p_dd,
                                  entry_topk=20, sell_topk=None,
                                  max_entry_dd5_rate=0.35,
                                  dd5_min_recall=0.60,
                                  dd5_max_fpr=0.20,
                                  up20_cands=None, dd5_cands=None):
    """
    Goal: maximize entry_precision_up20 (BUY quality),
    subject to constraints on:
      - entry_dd5_rate (avoid buying into drawdown risk)
      - dd5 recall and dd5 FPR (avoid false avoids/sells)
    Soft-relaxes constraints if needed.
    """
    if up20_cands is None:
        up20_cands = _quantile_threshold_candidates(p_up, POLICY_UP20_Q_LOW, POLICY_UP20_Q_HIGH, POLICY_N_CAND_UP20, include=[0.5])
    if dd5_cands is None:
        dd5_cands  = _quantile_threshold_candidates(p_dd, POLICY_DD5_Q_LOW,  POLICY_DD5_Q_HIGH,  POLICY_N_CAND_DD5,  include=[0.5])

    def _search(req_entry_dd5, req_dd5_rec, req_dd5_fpr):
        best = None
        for tu in up20_cands:
            for td in dd5_cands:
                m = policy_metrics_entry_exit(index, y_up, y_dd, p_up, p_dd, tu, td,
                                              entry_topk=entry_topk, sell_topk=sell_topk)
                if m["entry_n"] <= 0:
                    continue

                # constraints
                if np.isfinite(req_entry_dd5) and np.isfinite(m["entry_dd5_rate"]) and (m["entry_dd5_rate"] > req_entry_dd5):
                    continue
                if m["dd5_recall"] < req_dd5_rec:
                    continue
                if m["dd5_fpr"] > req_dd5_fpr:
                    continue

                # objective: prioritize entry precision, then more opportunities
                score = m["entry_precision_up20"]
                tie   = (m["entry_per_day"], -m["dd5_fpr"])
                if (best is None) or (score > best[0]) or (np.isclose(score, best[0]) and tie > best[2]):
                    best = (score, m, tie)
        return best

    # 1) strict
    best = _search(max_entry_dd5_rate, dd5_min_recall, dd5_max_fpr)
    if best is not None:
        return best[1]

    # 2) relax dd5_fpr
    best = _search(max_entry_dd5_rate, dd5_min_recall, 1.0)
    if best is not None:
        return best[1]

    # 3) relax entry dd5-rate
    best = _search(1.0, dd5_min_recall, 1.0)
    if best is not None:
        return best[1]

    # 4) relax dd5 recall
    best = _search(1.0, 0.0, 1.0)
    if best is not None:
        return best[1]

    # fallback
    return policy_metrics_entry_exit(index, y_up, y_dd, p_up, p_dd, 0.5, 0.5,
                                     entry_topk=entry_topk, sell_topk=sell_topk)

def undersample_indices(y, max_rows=None, max_pos=None, neg_pos_ratio=None, seed=42):
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y))
    y = np.asarray(y).astype(int)

    pos_idx = idx_all[y == 1]
    neg_idx = idx_all[y == 0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return idx_all

    if max_pos is not None and len(pos_idx) > max_pos:
        pos_idx = rng.choice(pos_idx, size=max_pos, replace=False)
    if neg_pos_ratio is not None and len(neg_idx) > neg_pos_ratio * max(1, len(pos_idx)):
        neg_keep = int(neg_pos_ratio * max(1, len(pos_idx)))
        neg_keep = max(1, min(neg_keep, len(neg_idx)))
        neg_idx  = rng.choice(neg_idx, size=neg_keep, replace=False)

    keep = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(keep)

    if max_rows is not None and len(keep) > max_rows:
        keep = keep[:max_rows]

    return np.sort(keep)

def undersample_for_tuning(X_train, y_train, X_valid, y_valid, w_train=None, w_valid=None):
    keep_tr = undersample_indices(
        y_train, max_rows=MAX_TRAIN_ROWS, max_pos=MAX_POS_SAMPLES,
        neg_pos_ratio=NEG_POS_RATIO, seed=42
    ) if (MAX_TRAIN_ROWS is not None and len(y_train) > MAX_TRAIN_ROWS) else np.arange(len(y_train))

    keep_va = undersample_indices(
        y_valid, max_rows=MAX_VALID_ROWS, max_pos=None,
        neg_pos_ratio=None, seed=123
    ) if (MAX_VALID_ROWS is not None and len(y_valid) > MAX_VALID_ROWS) else np.arange(len(y_valid))

    Xtr = X_train[keep_tr]; ytr = y_train[keep_tr]
    Xva = X_valid[keep_va]; yva = y_valid[keep_va]

    wtr = (w_train[keep_tr] if w_train is not None else None)
    wva = (w_valid[keep_va] if w_valid is not None else None)
    return Xtr, ytr, Xva, yva, wtr, wva

def _cap_rows_for_tree_models(X, y, cap, w=None):
    if len(y) <= cap:
        return X, y, w
    keep = undersample_indices(y, max_rows=cap, neg_pos_ratio=NEG_POS_RATIO, seed=101)
    return X[keep], y[keep], (w[keep] if w is not None else None)

def make_time_weights(index_dt, half_life_days=252):
    d = pd.to_datetime(index_dt)
    age = (d.max() - d).days if hasattr((d.max() - d), "days") else (d.max() - d).dt.days
    if not isinstance(age, np.ndarray):
        age = np.asarray(age)
    age = age.astype(np.float32)
    w = 0.5 ** (age / float(half_life_days))
    # normalize to mean=1 for stability
    w = w / max(1e-12, float(np.mean(w)))
    return w.astype(np.float32)

def infer_asset_class(ticker: str) -> str:
    t = str(ticker)
    if t.endswith(".SA"):
        return "equity_br"
    if t.startswith("^"):
        return "index"
    if t.endswith("-USD"):
        return "crypto"
    if "=X" in t:
        return "fx"
    if "=F" in t:
        return "futures"
    return "other"
def report_to_dict(rep):
    if rep is None:
        return None
    if isinstance(rep, dict):
        return rep
    if hasattr(rep, "_asdict"):
        return rep._asdict()
    if hasattr(rep, "__dict__"):
        return vars(rep)
    try:
        return dict(rep)
    except Exception:
        return {"value": str(rep)}

def _sample_train_indices(mask_labeled_split, y_up_i8, y_dd_i8, max_rows, neg_pos_ratio):
    """
    Mantém todos os positivos (up==1 ou dd==1) e amostra negativos até neg_pos_ratio * n_pos.
    """
    idx = np.flatnonzero(mask_labeled_split)
    if idx.size == 0:
        return idx

    yu = y_up_i8[idx]
    yd = y_dd_i8[idx]
    any_pos = (yu == 1) | (yd == 1)

    pos_idx = idx[any_pos]
    neg_idx = idx[~any_pos]

    if pos_idx.size == 0:
        # sem positivos: pega um subconjunto de negativos (para não virar vazio)
        if max_rows is None or neg_idx.size <= max_rows:
            return neg_idx
        take = rng.choice(neg_idx, size=max_rows, replace=False)
        return np.sort(take)

    max_neg = int(neg_pos_ratio * pos_idx.size) if neg_pos_ratio is not None else neg_idx.size
    max_neg = min(max_neg, neg_idx.size)

    if max_neg > 0:
        neg_take = rng.choice(neg_idx, size=max_neg, replace=False)
        keep = np.concatenate([pos_idx, neg_take])
    else:
        keep = pos_idx

    # cap total
    if (max_rows is not None) and (keep.size > max_rows):
        keep = rng.choice(keep, size=max_rows, replace=False)

    return np.sort(keep)

def _cap_indices(idx, max_rows):
    if (max_rows is None) or (idx.size <= max_rows):
        return idx
    take = rng.choice(idx, size=max_rows, replace=False)
    return np.sort(take)
def _normalize_feature_frame(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    # 1) Se vier MultiIndex (ex: (Open, ABEV3.SA)), reduz para colunas simples
    if isinstance(df.columns, pd.MultiIndex):
        # caso típico: nível do ticker está no último level
        try:
            if ticker is not None and ticker in df.columns.get_level_values(-1):
                df = df.xs(ticker, axis=1, level=-1, drop_level=True)
            else:
                # fallback: achata juntando os níveis
                df.columns = [
                    "|".join([str(x) for x in tup if x not in ("", "None")])
                    for tup in df.columns.to_list()
                ]
        except Exception:
            df.columns = [
                "|".join([str(x) for x in tup if x not in ("", "None")])
                for tup in df.columns.to_list()
            ]

    # 2) garante colunas como strings
    df.columns = [str(c) for c in df.columns]

    # 3) garante Adj Close (muitos fluxos não trazem mais essa coluna)
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    return df

def make_walk_forward_folds(train_dates_arr, n_folds=3, valid_len=60, purge_len=30):
    # Returns list of (train_end_idx, valid_start_idx, valid_end_idx) over date index positions
    # using contiguous segments from the end backwards.
    folds = []
    end = len(train_dates_arr)
    for _ in range(n_folds):
        v_end = end
        v_start = max(0, v_end - valid_len)
        t_end = max(0, v_start - purge_len)
        if t_end <= 0:
            break
        folds.append((t_end, v_start, v_end))
        end = t_end  # move backward for next fold
    folds = folds[::-1]  # chronological order
    return folds

def mask_from_date_range(idx_dt, start_dt, end_dt):
    # inclusive start, exclusive end
    idx_dt = pd.to_datetime(idx_dt)
    return (idx_dt >= start_dt) & (idx_dt < end_dt)

@dataclass
class TrialResult:
    params: dict
    best_iter: int | None
    score_mean: float
    score_std: float
    used_gpu: bool = False

def sample_lgbm_params(base_spw, try_gpu: bool):
    # allow scale_pos_weight < 1.0 (important when positives are frequent)
    p = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": float(10**np.random.uniform(-2.0, -0.7)),
        "num_leaves": int(np.random.randint(24, 128)),
        "max_depth": int(np.random.choice([-1, 5, 6, 7, 8])),
        "min_child_samples": int(np.random.randint(10, 60)),
        "subsample": float(np.random.choice(np.linspace(0.7, 1.0, 4))),
        "colsample_bytree": float(np.random.choice(np.linspace(0.6, 1.0, 5))),
        "reg_lambda": float(10**np.random.uniform(-3, 1)),
        "scale_pos_weight": float(np.random.choice([0.5*base_spw, 1.0*base_spw, 1.5*base_spw])),
        "n_estimators": 1800,
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
    }
    if try_gpu:
        p.update({"device_type": "gpu", "gpu_device_id": 0, "max_bin": 255})
    return p

def sample_hgb_params():
    if np.random.rand() < 0.5:
        max_depth = np.random.choice([None, 5, 7, 9])
        max_leaf_nodes = None
    else:
        max_depth = None
        max_leaf_nodes = int(np.random.choice([31, 63, 127, 255]))
    return {
        "loss": "log_loss",
        "learning_rate": float(10**np.random.uniform(-2.0, -0.7)),
        "max_depth": max_depth,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": int(np.random.choice([10, 20, 30, 50])),
        "l2_regularization": float(10**np.random.uniform(-4, 1)),
        "max_bins": int(np.random.choice([64, 128])),
        "class_weight": "balanced",
        "max_iter": int(np.random.choice([100, 150, 200])),  # tightened for speed
        "random_state": 42,
        "early_stopping": False,
        "verbose": 0
    }

def sample_rf_params():
    return {
        "n_estimators": int(np.random.choice([80, 120, 160])),
        "max_depth": int(np.random.choice([8, 12, 16])),
        "max_features": "sqrt",
        "min_samples_split": int(np.random.choice([2, 5, 10])),
        "min_samples_leaf": int(np.random.choice([1, 2, 4])),
        "bootstrap": True,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": 42,
        "verbose": 0,
        "warm_start": False,
        "max_samples": float(np.random.choice([0.4, 0.6])),
    }

def sample_et_params():
    return {
        "n_estimators": int(np.random.choice([200, 300, 400])),  # lighter in tuning
        "max_depth": int(np.random.choice([10, 14, 18])),
        "max_features": "sqrt",
        "min_samples_split": int(np.random.choice([2, 5, 10])),
        "min_samples_leaf": int(np.random.choice([1, 2, 4])),
        "bootstrap": False,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
        "verbose": 0,
        "warm_start": False,
    }

def lgb_train_with_gpu_fallback(params, X_tr, y_tr, w_tr, X_va, y_va, w_va):
    used_gpu = False
    try_params = params.copy()

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=dtrain, free_raw_data=False)

    # try GPU first if requested + available
    if GPU_AVAILABLE and try_params.get("device_type") == "gpu":
        try:
            with suppress_output():
                booster = lgb.train(
                    try_params,
                    train_set=dtrain,
                    valid_sets=[dvalid],
                    valid_names=["valid"],
                    num_boost_round=try_params["n_estimators"],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_LGBM, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )
            used_gpu = True
            return booster, used_gpu
        except Exception:
            pass

    # CPU fallback
    try_params.pop("device_type", None)
    try_params.pop("gpu_device_id", None)
    with suppress_output():
        booster = lgb.train(
            try_params,
            train_set=dtrain,
            valid_sets=[dvalid],
            valid_names=["valid"],
            num_boost_round=try_params["n_estimators"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_LGBM, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    return booster, used_gpu

def fit_with_weights(model, X, y, w):
    try:
        if w is None:
            model.fit(X, y)
        else:
            model.fit(X, y, sample_weight=w)
    except TypeError:
        # older versions / unsupported
        model.fit(X, y)
    return model

def _instantiate_rf(params):
    try:
        clf = RandomForestClassifier(**params)
    except TypeError:
        params = params.copy()
        params.pop("max_samples", None)
        clf = RandomForestClassifier(**params)
    return clf

def predict_proba_model(model_key, model_obj, X, best_iter=None):
    if model_key == "lgbm":
        num_it = int(best_iter) if best_iter else model_obj.current_iteration()
        with suppress_output():
            return model_obj.predict(X, num_iteration=num_it)
    else:
        return model_obj.predict_proba(X)[:, 1]

def quantile_threshold_candidates(proba, q_low, q_high, n=25, include=None):
    """
    Gera candidatos de threshold baseados em quantis das probabilidades.

    Args:
        proba: array de probabilidades
        q_low, q_high: faixa de quantis (ex: 0.80, 0.999)
        n: número de candidatos
        include: threshold adicional para incluir (ex: fallback)

    Returns:
        array sorted de thresholds únicos no intervalo [0, 1]
    """
    proba = np.asarray(proba, dtype=float)
    if len(proba) == 0:
        return np.array([0.5], dtype=float)

    # Quantis uniformemente espaçados
    qs = np.linspace(q_low, q_high, n)
    c = np.quantile(proba, qs)
    c = np.unique(np.clip(c, 0.0, 1.0))

    # Adiciona threshold fallback se fornecido
    if include is not None:
        c = np.unique(np.concatenate([c, np.asarray([include], dtype=float)]))

    c = np.sort(c)
    return c

def build_fold_arrays(fold_spec, target):
    """
    Usa SOMENTE o subconjunto TRAIN (X_train_full / y_train_full / w_train_full),
    evitando LONG slicing enorme em todo fold/trial.
    """
    t_end, v_start, v_end = fold_spec

    train_end_date  = train_unique_dates[t_end - 1]      # inclusive
    valid_start_date = train_unique_dates[v_start]       # inclusive
    valid_end_date   = train_unique_dates[v_end - 1]     # inclusive

    m_tr = (idx_train_dt <= train_end_date)
    m_va = (idx_train_dt >= valid_start_date) & (idx_train_dt <= valid_end_date)

    X_tr = X_train_full[m_tr]
    y_tr = y_train_full[target][m_tr].astype("int8")
    X_va = X_train_full[m_va]
    y_va = y_train_full[target][m_va].astype("int8")

    w_tr = (w_train_full[m_tr] if w_train_full is not None else None)
    w_va = (w_train_full[m_va] if w_train_full is not None else None)

    return X_tr, y_tr, w_tr, X_va, y_va, w_va

def tune_lgbm_cv(target, n_trials):
    # base scale_pos_weight from full TRAIN distribution
    y_tr_full = y_train_full[target].astype(int)
    pos = int(y_tr_full.sum()); neg = int(len(y_tr_full) - pos)
    if pos == 0 or neg == 0:
        return None
    base_spw = float(neg / max(1, pos))  # allow <1

    best = None
    pbar = tqdm(range(n_trials), desc="Tuning lgbm (CV)", leave=True)
    for _ in pbar:
        params = sample_lgbm_params(base_spw, try_gpu=GPU_AVAILABLE)

        fold_scores = []
        used_gpu_any = False
        for fs in CV_FOLDS_SPEC:
            X_tr, y_tr, w_tr, X_va, y_va, w_va = build_fold_arrays(fs, target)
            # tuning-only downsampling
            X_tr, y_tr, X_va, y_va, w_tr2, w_va2 = undersample_for_tuning(
                X_tr, y_tr, X_va, y_va, w_train=w_tr, w_valid=w_va
            )

            booster, used_gpu = lgb_train_with_gpu_fallback(params, X_tr, y_tr, w_tr2, X_va, y_va, w_va2)
            used_gpu_any = used_gpu_any or used_gpu

            with suppress_output():
                p = booster.predict(X_va, num_iteration=booster.best_iteration)

            ap = _safe_float(lambda: average_precision_score(y_va.astype(int), p))
            fold_scores.append(ap)

        m = float(np.mean(fold_scores))
        s = float(np.std(fold_scores))

        if (best is None) or (m > best.score_mean):
            # store best_iter as median of best_iterations across folds? here we just keep last; we'll refit later anyway.
            best = TrialResult(params=params, best_iter=None, score_mean=m, score_std=s, used_gpu=used_gpu_any)

        pbar.set_postfix(APm=f"{m:.3f}", APs=f"{s:.3f}", GPU=("Y" if used_gpu_any else "N"))
    return best

def tune_hgb_cv(target, n_trials):
    best = None
    pbar = tqdm(range(n_trials), desc="Tuning hgb (CV)", leave=True)
    for _ in pbar:
        params = sample_hgb_params()
        fold_scores = []
        for fs in CV_FOLDS_SPEC:
            X_tr, y_tr, w_tr, X_va, y_va, w_va = build_fold_arrays(fs, target)

            # smaller caps for hgb
            X_tr, y_tr, w_tr = _cap_rows_for_tree_models(X_tr, y_tr, 120_000, w_tr)

            clf = HistGradientBoostingClassifier(**params)
            clf = fit_with_weights(clf, X_tr, y_tr.astype(int), w_tr)
            p = clf.predict_proba(X_va)[:, 1]
            ap = _safe_float(lambda: average_precision_score(y_va.astype(int), p))
            fold_scores.append(ap)

        m = float(np.mean(fold_scores))
        s = float(np.std(fold_scores))
        if (best is None) or (m > best.score_mean):
            best = TrialResult(params=params, best_iter=params.get("max_iter"), score_mean=m, score_std=s)
        pbar.set_postfix(APm=f"{m:.3f}", APs=f"{s:.3f}")
    return best

def tune_rf_cv(target, n_trials):
    best = None
    pbar = tqdm(range(n_trials), desc="Tuning rf (CV)", leave=True)
    for _ in pbar:
        params = sample_rf_params()
        fold_scores = []
        t0 = time.time()
        for fs in CV_FOLDS_SPEC:
            X_tr, y_tr, w_tr, X_va, y_va, w_va = build_fold_arrays(fs, target)
            X_tr, y_tr, w_tr = _cap_rows_for_tree_models(X_tr, y_tr, RF_TUNE_MAX_ROWS, w_tr)
            clf = _instantiate_rf(params)
            clf = fit_with_weights(clf, X_tr, y_tr.astype(int), w_tr)
            p = clf.predict_proba(X_va)[:, 1]
            ap = _safe_float(lambda: average_precision_score(y_va.astype(int), p))
            fold_scores.append(ap)

        m = float(np.mean(fold_scores))
        s = float(np.std(fold_scores))
        if (best is None) or (m > best.score_mean):
            best = TrialResult(params=params, best_iter=None, score_mean=m, score_std=s)
        pbar.set_postfix(APm=f"{m:.3f}", APs=f"{s:.3f}", s=f"{(time.time()-t0):.1f}")
    return best

def tune_et_cv(target, n_trials):
    best = None
    pbar = tqdm(range(n_trials), desc="Tuning et (CV)", leave=True)
    for _ in pbar:
        params = sample_et_params()
        fold_scores = []
        t0 = time.time()
        for fs in CV_FOLDS_SPEC:
            X_tr, y_tr, w_tr, X_va, y_va, w_va = build_fold_arrays(fs, target)
            X_tr, y_tr, w_tr = _cap_rows_for_tree_models(X_tr, y_tr, ET_TUNE_MAX_ROWS, w_tr)
            clf = ExtraTreesClassifier(**params)
            clf = fit_with_weights(clf, X_tr, y_tr.astype(int), w_tr)
            p = clf.predict_proba(X_va)[:, 1]
            ap = _safe_float(lambda: average_precision_score(y_va.astype(int), p))
            fold_scores.append(ap)

        m = float(np.mean(fold_scores))
        s = float(np.std(fold_scores))
        if (best is None) or (m > best.score_mean):
            best = TrialResult(params=params, best_iter=None, score_mean=m, score_std=s)
        pbar.set_postfix(APm=f"{m:.3f}", APs=f"{s:.3f}", s=f"{(time.time()-t0):.1f}")
    return best

def fit_final_lgbm_simple(params, X_fit, y_fit, w_fit, num_boost_round=600):

    # recompute scale_pos_weight based on fit set (allow <1)
    y_fit_i = y_fit.astype(int)
    pos = int(y_fit_i.sum()); neg = int(len(y_fit_i) - pos)
    spw = float(neg / max(1, pos))
    p = params.copy()
    p["scale_pos_weight"] = spw

    used_gpu = False
    # use GPU if present in params and available
    try_gpu = GPU_AVAILABLE and p.get("device_type") == "gpu"

    dfit = lgb.Dataset(X_fit, label=y_fit_i, weight=w_fit, free_raw_data=False)

    if try_gpu:
        try:
            with suppress_output():
                booster = lgb.train(
                    p,
                    train_set=dfit,
                    num_boost_round=int(num_boost_round),
                    valid_sets=[],
                    callbacks=[lgb.log_evaluation(period=0)],
                )
            used_gpu = True
            return booster, used_gpu
        except Exception:
            pass

    # CPU fallback
    p.pop("device_type", None)
    p.pop("gpu_device_id", None)
    with suppress_output():
        booster = lgb.train(
            p,
            train_set=dfit,
            num_boost_round=int(num_boost_round),
            valid_sets=[],
            callbacks=[lgb.log_evaluation(period=0)],
        )
    return booster, used_gpu

def fit_final_hgb(params, X_fit, y_fit, w_fit):
    clf = HistGradientBoostingClassifier(**params)
    return fit_with_weights(clf, X_fit, y_fit.astype(int), w_fit)

def fit_final_rf(params, X_fit, y_fit, w_fit):
    clf = _instantiate_rf(params)
    return fit_with_weights(clf, X_fit, y_fit.astype(int), w_fit)

def fit_final_et(params, X_fit, y_fit, w_fit):
    # allow slightly more trees on final fit if you want:
    p = params.copy()
    p["n_estimators"] = max(int(p.get("n_estimators", 300)), 400)
    clf = ExtraTreesClassifier(**p)
    return fit_with_weights(clf, X_fit, y_fit.astype(int), w_fit)
def precisionatkperday(index, ytrue, proba, k=20):
    idx = pd.to_datetime(index)
    dftmp = pd.DataFrame(
        {'y': np.asarray(ytrue).astype(int), 'p': np.asarray(proba, dtype=float)},
        index=idx
    )
    vals = []
    for d, g in dftmp.groupby(dftmp.index.date):
        g2 = g.sort_values('p', ascending=False).head(k)
        if len(g2) > 0:
            vals.append(float(g2['y'].mean()))
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))

def align_columns(df_like, cols_expected):
    for c in cols_expected:
        if c not in df_like.columns:
            df_like[c] = 0.0
    extra = [c for c in df_like.columns if c not in cols_expected and c not in TARGETS and not c.startswith(("tk_", "ac_"))]
    if extra:
        df_like = df_like.drop(columns=extra)
    return df_like[cols_expected]

def top_per_day(df_in, prob_col, k=TOPK_PER_DAY):
    rows = []
    for d in apply_days:
        day = df_in.loc[df_in.index == d]
        rows.append(day.sort_values(prob_col, ascending=False).head(k).assign(date=d))
    return pd.concat(rows)

def _get_default_min_recall_select_up():
    return float(globals().get("MIN_RECALL", 0.10))

def _safe_float(x, default=np.nan):
    """Accepts a callable OR a value."""
    try:
        return float(x() if callable(x) else x)
    except Exception:
        return float(default)

def confusion_stats(y_true, pred):
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(1, (fp + tn))
    tnr = tn / max(1, (tn + fp))
    pos_rate = float(np.mean(pred)) if len(pred) else 0.0
    return int(tn), int(fp), int(fn), int(tp), float(fpr), float(tnr), float(pos_rate)

def metrics_report(y_true, proba, thr):
    y_true = np.asarray(y_true, dtype=np.int8)
    proba  = np.asarray(proba, dtype=np.float64)
    proba  = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)
    pred   = (proba >= float(thr)).astype(np.int8)

    mcc = matthews_corrcoef(y_true, pred) if len(np.unique(y_true)) > 1 else 0.0

    tn, fp, fn, tp, fpr, tnr, pos_rate = confusion_stats(y_true, pred)
    ap  = float(average_precision_score(y_true, proba)) if len(np.unique(y_true)) > 1 else np.nan
    auc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else np.nan

    return {
        "AP": ap,
        "ROC_AUC": auc,
        "MCC": float(mcc),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "BalancedAcc": float(balanced_accuracy_score(y_true, pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "FPR": float(fpr),
        "PosRate": float(pos_rate),
        "Confusion": [tn, fp, fn, tp],
    }

def choose_thr_for_target_up20(y_valid, proba_valid):
    """
    ENSEMBLE up20 threshold (volume-aware):
    - tenta aumentar o número de positivos mirando uma POSRATE alvo (fração de linhas positivas),
      mas com pisos mínimos de precision/recall.
    - fallback: threshold por quantil para bater o alvo de volume.
    """
    y = np.asarray(y_valid).astype(int)
    p = np.asarray(proba_valid, dtype=float)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)

    if len(y) == 0:
        return 0.5
    if len(np.unique(y)) < 2:
        # sem classes suficientes, pega algo razoável por quantil
        return float(np.quantile(p, 0.95))

    base_rate = float(np.mean(y))

    # --- knobs (ensemble only) ---
    # alvo de volume: se o evento é raro, força um mínimo (ex: 2%); se é menos raro, sobe com base_rate
    target_posrate = float(globals().get(
        "ENS_UP20_TARGET_POSRATE",
        max(0.02, min(0.20, base_rate * 3.0 + 0.02))
    ))
    min_posrate = float(globals().get(
        "ENS_UP20_MIN_POSRATE",
        max(0.01, min(0.15, base_rate * 1.4))
    ))
    min_precision = float(globals().get("ENS_UP20_MIN_PRECISION", 0.30))
    min_recall = float(globals().get("ENS_UP20_MIN_RECALL", float(globals().get("MIN_RECALL", 0.10))))
    beta = float(globals().get("ENS_UP20_BETA", 1.5))  # >1 favorece recall (mais sinais)
    posrate_penalty = float(globals().get("ENS_UP20_POSRATE_PENALTY", 0.25))

    target_posrate = float(np.clip(target_posrate, 0.005, 0.50))
    min_posrate    = float(np.clip(min_posrate,    0.001, 0.50))

    # candidatos: thresholds por quantis (estável e rápido)
    qs = np.linspace(0.01, 0.995, 301)
    thr_grid = np.unique(np.clip(np.quantile(p, qs), 0.0, 1.0))

    b2 = beta * beta
    best_thr, best_score = 0.5, -1e18

    for t in thr_grid:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp, fpr, _, posrate = confusion_stats(y, pred)

        prec = tp / max(1, (tp + fp))
        rec  = tp / max(1, (tp + fn))

        # pisos (para não virar “spam”)
        if (prec < min_precision) or (rec < min_recall) or (posrate < min_posrate):
            continue

        # F_beta com beta>1 + bônus de volume (via penalidade por desviar do target_posrate)
        denom = max(1e-12, (b2 * prec + rec))
        fbeta = (1.0 + b2) * prec * rec / denom

        # quanto mais perto do target_posrate, melhor (mas sem matar volume)
        vol_pen = abs(float(posrate) - target_posrate)
        score = float(fbeta) - posrate_penalty * vol_pen

        if score > best_score:
            best_score = score
            best_thr = float(t)

    if np.isfinite(best_score) and best_score > -1e17:
        return float(best_thr)

    # Fallback “garante volume”: threshold no quantil que dá ~target_posrate positivos
    # P(p >= thr) ~= target_posrate  => thr = quantile(p, 1 - target_posrate)
    q = float(np.clip(1.0 - target_posrate, 0.0, 1.0))
    return float(np.quantile(p, q))

def choose_thr_for_target_dd5_filter(y_valid, proba_valid, buy_base_mask=None):
    """
    ENSEMBLE dd5 threshold (risk filter) com opção de tunar para NÃO matar BUY:
    - mantém dd5 global "razoável" (recall/fpr/posrate caps)
    - e, se buy_base_mask fornecido, escolhe thr que maximiza BUY_pass
      garantindo que o dd5-rate verdadeiro dentro dos BUYs fique <= DD5_BUY_MAX_DD5RATE_SEL
    """
    y = np.asarray(y_valid).astype(int)
    p = np.asarray(proba_valid, dtype=float)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)

    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5

    base_rate = float(np.mean(y))

    # knobs já existentes no seu cell
    min_rec = float(globals().get("DD5_FILTER_MIN_RECALL", 0.80))
    max_fpr = float(globals().get("DD5_FILTER_MAX_FPR", 0.30))
    slack   = float(globals().get("DD5_POSRATE_SLACK", 0.20))
    hardcap = float(globals().get("DD5_POSRATE_HARD_CAP", 0.70))

    # tuning on buy-base
    tune_on_buy = bool(globals().get("TUNE_DD5_THR_ON_BUYBASE", True))
    min_buy_n   = int(globals().get("DD5_BUY_MIN_SIGNALS_SEL", 80))
    max_buy_dd5 = float(globals().get("DD5_BUY_MAX_DD5RATE_SEL", 0.15))
    grid_q      = int(globals().get("DD5_BUY_GRID_Q", 181))

    # grid de thresholds (quantis)
    qs = np.linspace(0.01, 0.99, grid_q)
    thr_grid = np.unique(np.clip(np.quantile(p, qs), 0.0, 1.0))

    best_thr = 0.5
    best_score = -1e18

    # helper: avalia constraints globais dd5
    def ok_global(thr):
        pred = (p >= thr).astype(int)
        tn, fp, fn, tp, fpr, _, posrate = confusion_stats(y, pred)
        rec = tp / max(1, (tp + fn))
        # caps de posrate (para não virar “dd5 em tudo”)
        pos_cap = min(hardcap, base_rate + slack)
        return (rec >= min_rec) and (fpr <= max_fpr) and (posrate <= pos_cap)

    # se não tiver buy_base_mask ou tuning desligado, escolhe "melhor" por recall-first/precision
    if (buy_base_mask is None) or (not tune_on_buy):
        # preferir thresholds mais ALTOS (menos dd5=1) mantendo constraints
        for thr in thr_grid[::-1]:
            if ok_global(thr):
                return float(thr)
        # fallback: relaxa constraints
        return float(np.quantile(p, 0.50))

    buy_base_mask = np.asarray(buy_base_mask, dtype=bool)
    if buy_base_mask.shape[0] != y.shape[0]:
        # se vier desalinhado, cai no modo simples
        for thr in thr_grid[::-1]:
            if ok_global(thr):
                return float(thr)
        return float(np.quantile(p, 0.50))

    # tuning: maximiza quantos BUY_base passam no filtro dd5 (p_dd < thr)
    # com limite de dd5-rate verdadeiro dentro desses BUYs
    for thr in thr_grid[::-1]:  # começa alto => mais BUY_pass
        if not ok_global(thr):
            continue

        buy_pass = buy_base_mask & (p < thr)
        n = int(buy_pass.sum())
        if n < min_buy_n:
            continue

        dd5_rate_true = float(np.mean(y[buy_pass]))  # dd5 verdadeiro entre BUYs
        if dd5_rate_true > max_buy_dd5:
            continue

        # score: mais sinais primeiro; empate: menor dd5_rate_true
        score = n - 1000.0 * dd5_rate_true
        if score > best_score:
            best_score = score
            best_thr = float(thr)

    if best_score > -1e17:
        return float(best_thr)

    # fallback: relaxa buy constraints mantendo global o máximo possível
    for thr in thr_grid[::-1]:
        if ok_global(thr):
            return float(thr)

    return float(np.quantile(p, 0.50))

def choose_thr_dd5_filter(
    y_true,
    proba,
    base_rate,
    min_recall=0.60,
    max_fpr=0.20,
    pos_slack=0.10,
    hard_cap=0.70,
):
    """
    Threshold para dd5 como filtro de risco:
    - tenta alcançar recall >= min_recall, mas SEM explodir FPR/PosRate
    - nunca aceita pos_rate > hard_cap e nunca aceita tn==0 (evita all-positive)
    - se inviável, relaxa recall em "ladder" e mantém caps
    - fallback final: escolhe threshold por posrate-alvo (quantil) + caps
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(proba, dtype=np.float64)
    if len(np.unique(y_true)) < 2 or len(p) == 0:
        return 0.5

    # caps por prevalência
    max_pos = min(float(hard_cap), float(base_rate) + float(pos_slack))
    max_pos = max(0.01, min(max_pos, float(hard_cap)))  # sanity

    prec, rec, thr = precision_recall_curve(y_true, p)
    if len(thr) == 0:
        # fallback quantil
        return float(np.quantile(p, 1.0 - max_pos))

    prec_t = prec[:-1]
    rec_t  = rec[:-1]

    def _eval_threshold(t):
        pred = (p >= float(t)).astype(np.int8)
        tn, fp, fn, tp, fpr, _, pos_rate = confusion_stats(y_true, pred)
        pr = float(prec_t[i])
        rr = float(rec_t[i])
        f1 = float(f1_score(y_true, pred, zero_division=0))
        return tn, fp, fn, tp, fpr, pos_rate, pr, rr, f1

    # ---- 1) ladder de recall: tenta min_recall, depois desce se inviável ----
    ladder = [min_recall, 0.55, 0.50, 0.45, 0.40, 0.35]
    ladder = [r for r in ladder if r <= 1.0]

    best = None
    for rmin in ladder:
        best = None
        for i, t in enumerate(thr):
            pred = (p >= float(t)).astype(np.int8)
            tn, fp, fn, tp, fpr, _, pos_rate = confusion_stats(y_true, pred)

            # HARD constraints SEMPRE
            if tn == 0:
                continue
            if pos_rate > max_pos:
                continue
            if fpr > max_fpr:
                continue
            if rec_t[i] < rmin:
                continue

            cand = (
                float(prec_t[i]),                 # maximize precision
                float(rec_t[i]),                  # then recall
                float(f1_score(y_true, pred, zero_division=0)),
                -float(fpr),                      # prefer lower fpr
                -float(pos_rate),                 # prefer lower pos rate
                int(tp),
                float(t)
            )
            if best is None or cand > best:
                best = cand

        if best is not None:
            return float(best[-1])

    # ---- 2) ainda inviável: minimize FPR sob hard caps (posrate + tn) ----
    best2 = None
    for i, t in enumerate(thr):
        pred = (p >= float(t)).astype(np.int8)
        tn, fp, fn, tp, fpr, _, pos_rate = confusion_stats(y_true, pred)
        if tn == 0:
            continue
        if pos_rate > max_pos:
            continue
        cand = (
            -float(fpr),                 # minimize fpr
            float(prec_t[i]),            # then precision
            -float(pos_rate),            # then lower posrate
            float(rec_t[i]),             # then recall
            int(tp),
            float(t)
        )
        if best2 is None or cand > best2:
            best2 = cand

    if best2 is not None:
        return float(best2[-1])

    # ---- 3) fallback final: threshold por quantil para bater posrate alvo ----
    return float(np.quantile(p, 1.0 - max_pos))
def tune_thr_dd5_for_buybase(
    p_up_sel: np.ndarray,
    p_dd_sel: np.ndarray,
    y_up_sel: np.ndarray,
    y_dd_sel: np.ndarray,
    thr_up: float,
    min_signals: int = 80,
    max_dd5_rate: float = 0.15,
    grid_q: int = 181,
):
    """
    Choose thr_dd to reduce dd5 inside BUY_BASE candidates on the selection split.

    BUY_BASE(thr_dd) := (p_up >= thr_up) & (p_dd < thr_dd)

    Objective (lexicographic):
      1) maximize Precision(GOOD) among BUY_BASE
      2) maximize Precision(up20) among BUY_BASE
      3) minimize dd5_rate among BUY_BASE
      4) maximize Recall(GOOD) (coverage)
      5) maximize number of BUYs (stability)

    With constraints ladder:
      - try max_dd5_rate and min_signals
      - if infeasible, relax dd5 cap gradually, then relax min_signals
    """
    p_up = np.asarray(p_up_sel, dtype=np.float64)
    p_dd = np.asarray(p_dd_sel, dtype=np.float64)
    y_up = np.asarray(y_up_sel, dtype=np.int8)
    y_dd = np.asarray(y_dd_sel, dtype=np.int8)

    y_good = ((y_up == 1) & (y_dd == 0)).astype(np.int8)
    n_good = int(y_good.sum())

    # candidate thresholds = quantiles (robust to calibration / scaling)
    qs = np.linspace(0.01, 0.99, max(21, int(grid_q)))
    thr_cand = np.unique(np.quantile(p_dd, qs))
    # add some fixed points for safety
    thr_cand = np.unique(np.concatenate([thr_cand, np.array([0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90])]))
    thr_cand = np.clip(thr_cand, 1e-6, 1 - 1e-6)

    buy_cand = (p_up >= float(thr_up))

    def _eval(thr_dd):
        buy = buy_cand & (p_dd < float(thr_dd))
        n_buy = int(buy.sum())
        if n_buy == 0:
            return None

        dd5_rate = float(np.mean(y_dd[buy] == 1))
        up20_prec = float(np.mean(y_up[buy] == 1))
        good_prec = float(np.mean(y_good[buy] == 1))
        good_rec = float((y_good[buy].sum()) / max(1, n_good))
        return {
            "thr_dd": float(thr_dd),
            "n_buy": n_buy,
            "buy_rate": float(n_buy / max(1, len(p_up))),
            "dd5_rate_in_buys": dd5_rate,
            "up20_prec_in_buys": up20_prec,
            "good_prec_in_buys": good_prec,
            "good_rec": good_rec,
        }

    # ladder of relaxation
    dd5_caps = [max_dd5_rate, 0.20, 0.25, 0.30, 0.40]
    min_buys = [min_signals, max(20, min_signals // 2), 20, 10]

    best = None
    best_info = None
    used = None

    for cap in dd5_caps:
        for mb in min_buys:
            cur_best = None
            cur_info = None
            for t in thr_cand:
                info = _eval(t)
                if info is None:
                    continue
                if info["n_buy"] < mb:
                    continue
                if info["dd5_rate_in_buys"] > cap:
                    continue

                cand = (
                    info["good_prec_in_buys"],
                    info["up20_prec_in_buys"],
                    -info["dd5_rate_in_buys"],
                    info["good_rec"],
                    info["n_buy"],
                    info["thr_dd"],
                )
                if (cur_best is None) or (cand > cur_best):
                    cur_best = cand
                    cur_info = info

            if cur_info is not None:
                best = cur_best
                best_info = cur_info
                used = {"max_dd5_rate": cap, "min_signals": mb}
                break
        if best_info is not None:
            break

    # absolute fallback: pick threshold that minimizes dd5 inside buys while keeping at least 10 buys
    if best_info is None:
        fallback_best = None
        fallback_info = None
        for t in thr_cand:
            info = _eval(t)
            if info is None or info["n_buy"] < 10:
                continue
            cand = (
                -info["dd5_rate_in_buys"],
                info["good_prec_in_buys"],
                info["up20_prec_in_buys"],
                info["n_buy"],
                info["thr_dd"],
            )
            if (fallback_best is None) or (cand > fallback_best):
                fallback_best = cand
                fallback_info = info
        if fallback_info is not None:
            best_info = fallback_info
            used = {"max_dd5_rate": None, "min_signals": 10}

    if best_info is None:
        # nothing worked; keep original behavior
        return float(np.quantile(p_dd, 0.70)), {
            "note": "dd5 buy-tuning failed; fallback quantile used",
            "thr_dd": float(np.quantile(p_dd, 0.70)),
        }

    best_info = dict(best_info)
    best_info["constraints_used"] = used
    return float(best_info["thr_dd"]), best_info

def make_purged_rolling_splits(unique_dates, test_len_days=15, step_days=3, purge_days=30):
    unique_dates = np.asarray(unique_dates)
    m = len(unique_dates)
    if m <= purge_days + 5:
        s = max(0, m - test_len_days)
        return [(unique_dates[:max(0, s - purge_days)], unique_dates[s:])]
    starts = list(range(purge_days, max(purge_days + 1, m - test_len_days + 1), step_days))
    splits = []
    for s in starts:
        e = min(m, s + test_len_days)
        tr = unique_dates[:max(0, s - purge_days)]
        te = unique_dates[s:e]
        if len(te) > 0:
            splits.append((tr, te))
    if not splits:
        s = max(purge_days, m - test_len_days)
        splits.append((unique_dates[:max(0, s - purge_days)], unique_dates[s:]))
    return splits

def make_cv_splits_with_relaxed_purge(valid_unique_dates, purge_days, test_len_days, step_days, min_splits=3):
    cand = [
        purge_days,
        max(0, purge_days // 2),
        max(0, purge_days // 3),
        30, 21, 14, 7, 0
    ]
    seen = set()
    for p in cand:
        if p in seen:
            continue
        seen.add(p)
        splits = make_purged_rolling_splits(valid_unique_dates, test_len_days=test_len_days, step_days=step_days, purge_days=p)
        if len(splits) >= min_splits or p == 0:
            return splits, p
    return make_purged_rolling_splits(valid_unique_dates, test_len_days=test_len_days, step_days=step_days, purge_days=purge_days), purge_days

def mask_rows_by_dates(row_dates, train_dates, test_dates):
    row_dates = pd.to_datetime(row_dates)
    tr_mask = row_dates.isin(train_dates)
    te_mask = row_dates.isin(test_dates)
    return np.where(tr_mask)[0], np.where(te_mask)[0]

def _logit(p, eps=1e-6):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1.0 - p))

def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))

def oof_isotonic(y_true, proba, row_dates, cv_splits, min_rows=2000, min_pos=20, min_neg=20):
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(proba, dtype=np.float64)
    oof = np.full_like(p, np.nan, dtype=float)
    covered = np.zeros(len(p), dtype=bool)

    for tr_dates, te_dates in cv_splits:
        tr_idx, te_idx = mask_rows_by_dates(row_dates, tr_dates, te_dates)
        if len(te_idx) == 0 or len(tr_idx) < min_rows:
            continue
        y_tr = y[tr_idx]
        pos = int(y_tr.sum()); neg = int(len(y_tr) - pos)
        if pos < min_pos or neg < min_neg:
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p[tr_idx], y_tr)
        oof[te_idx] = iso.transform(p[te_idx])
        covered[te_idx] = True

    cover_rate = float(np.mean(covered)) if len(covered) else 0.0
    oof[~covered] = p[~covered]

    if int(y.sum()) > 0 and int(y.sum()) < len(y):
        iso_full = IsotonicRegression(out_of_bounds="clip")
        iso_full.fit(p, y)
    else:
        iso_full = None

    return oof, iso_full, cover_rate

def oof_platt(y_true, proba, row_dates, cv_splits, min_rows=800, min_pos=10, min_neg=10):
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(proba, dtype=np.float64)
    x = _logit(p).reshape(-1, 1)

    oof = np.full(len(y), np.nan, dtype=float)
    covered = np.zeros(len(y), dtype=bool)

    for tr_dates, te_dates in cv_splits:
        tr_idx, te_idx = mask_rows_by_dates(row_dates, tr_dates, te_dates)
        if len(te_idx) == 0 or len(tr_idx) < min_rows:
            continue
        y_tr = y[tr_idx]
        pos = int(y_tr.sum()); neg = int(len(y_tr) - pos)
        if pos < min_pos or neg < min_neg:
            continue

        lr = LogisticRegression(solver="lbfgs", max_iter=300)
        lr.fit(x[tr_idx], y_tr)
        oof[te_idx] = lr.predict_proba(x[te_idx])[:, 1]
        covered[te_idx] = True

    cov = float(np.mean(covered)) if len(covered) else 0.0
    oof[~covered] = p[~covered]

    if int(y.sum()) > 0 and int(y.sum()) < len(y):
        lr_full = LogisticRegression(solver="lbfgs", max_iter=300)
        lr_full.fit(x, y)
    else:
        lr_full = None

    return oof, lr_full, cov

def apply_calibrator(cal, proba):
    if cal is None:
        return np.asarray(proba, dtype=np.float64)
    if hasattr(cal, "transform"):
        return cal.transform(np.asarray(proba, dtype=np.float64))
    x = _logit(np.asarray(proba, dtype=np.float64)).reshape(-1, 1)
    return cal.predict_proba(x)[:, 1]

def build_meta_matrix(preds_dict, keys_order=None):
    keys = list(preds_dict.keys()) if keys_order is None else list(keys_order)
    mat = np.vstack([preds_dict[k] for k in keys]).T  # (n, M)
    mean_ = mat.mean(axis=1, keepdims=True)
    std_  = mat.std(axis=1, keepdims=True)
    mn    = mat.min(axis=1, keepdims=True)
    mx    = mat.max(axis=1, keepdims=True)
    rng   = mx - mn
    disc  = np.sum((mat >= 0.5), axis=1, keepdims=True)
    centered = mat - mean_
    Xmeta = np.hstack([mat, mean_, std_, mn, mx, rng, disc, centered])
    return Xmeta, keys

def equal_weight_avg(preds_dict, keys_order=None):
    keys = list(preds_dict.keys()) if keys_order is None else list(keys_order)
    arr = np.vstack([preds_dict[k] for k in keys]).T
    return arr.mean(axis=1)

def recency_weights(row_dates, alpha=3.0):
    uniq = np.array(sorted(pd.Index(pd.to_datetime(row_dates)).unique()))
    pos = {d: i for i, d in enumerate(uniq)}
    idx = np.array([pos[d] for d in pd.to_datetime(row_dates)])
    t = idx / (len(uniq) - 1 + 1e-9)
    w = np.exp(alpha * t)
    return w.astype(np.float64)

def cv_weight_search(y_true, preds_dict, row_dates, cv_splits, trials=250, seed=42,
                     sample_weight=None, logit_space=False):
    keys = list(preds_dict.keys())
    base = np.vstack([preds_dict[k] for k in keys]).T  # (n, M)
    if logit_space:
        base = _logit(base)

    rng = np.random.default_rng(seed)
    best_w, best_score = None, -1.0

    for _ in tqdm(range(trials), desc="  weight search (purged CV)", leave=False):
        raw = rng.random(len(keys)) + 1e-8
        w = raw / raw.sum()

        ap_scores = []
        for tr_dates, te_dates in cv_splits:
            _, te_idx = mask_rows_by_dates(row_dates, tr_dates, te_dates)
            if len(te_idx) == 0:
                continue
            blend_lin = base[te_idx].dot(w)
            blend = _sigmoid(blend_lin) if logit_space else blend_lin

            if sample_weight is not None:
                ap = average_precision_score(y_true[te_idx], blend, sample_weight=sample_weight[te_idx])
            else:
                ap = average_precision_score(y_true[te_idx], blend)
            ap_scores.append(ap)

        if not ap_scores:
            continue
        score = float(np.mean(ap_scores))
        if score > best_score:
            best_score, best_w = score, w.copy()

    if best_w is None:
        best_w = np.ones(len(keys)) / max(1, len(keys))
    return best_w, keys, best_score

def logit_blend(preds_dict, weights, keys_order=None):
    keys = list(preds_dict.keys()) if keys_order is None else list(keys_order)
    W = np.asarray(weights, dtype=float)
    W = W / max(1e-12, W.sum())
    logits = np.vstack([_logit(preds_dict[k]) for k in keys]).T
    agg = logits.dot(W)
    return _sigmoid(agg)

def trimmed_mean_blend(preds_dict, trim_k=1, keys_order=None):
    keys = list(preds_dict.keys()) if keys_order is None else list(keys_order)
    mat = np.vstack([preds_dict[k] for k in keys])  # (M, n)
    M, n = mat.shape
    if M <= 2 * trim_k:
        return mat.mean(axis=0)
    sort_idx = np.argsort(mat, axis=0)
    keep_mask = np.ones_like(mat, dtype=bool)
    for j in range(n):
        keep_mask[sort_idx[:trim_k, j], j] = False
        keep_mask[sort_idx[-trim_k:, j], j] = False
    kept = np.where(keep_mask, mat, np.nan)
    return np.nanmean(kept, axis=0)

def ensemble_selection_forward(y_true, preds_dict, row_dates, cv_splits, steps=32, tol=1e-4):
    keys = list(preds_dict.keys())
    base = {k: preds_dict[k] for k in keys}
    n = len(y_true)
    current = np.zeros(n, dtype=float)
    counts = {k: 0 for k in keys}
    best_score = -1.0

    def score_of(pred):
        aps = []
        for tr_dates, te_dates in cv_splits:
            _, te_idx = mask_rows_by_dates(row_dates, tr_dates, te_dates)
            if len(te_idx) == 0:
                continue
            aps.append(average_precision_score(y_true[te_idx], pred[te_idx]))
        return float(np.mean(aps)) if aps else -1.0

    for step in range(steps):
        cand_best = None
        cand_key = None
        for k in keys:
            trial = (current * (step / (step + 1.0))) + (base[k] / (step + 1.0))
            sc = score_of(trial)
            if sc > best_score + tol:
                best_score = sc
                cand_best = trial
                cand_key = k
        if cand_key is None:
            break
        current = cand_best
        counts[cand_key] += 1

    total = sum(counts.values())
    if total == 0:
        w = np.ones(len(keys)) / max(1, len(keys))
    else:
        w = np.array([counts[k] for k in keys], dtype=float) / total
    return w, keys, best_score

def purged_stacked_lr(y_true, preds_dict, row_dates, cv_splits, min_rows=2000, min_pos=20, min_neg=20):
    y = np.asarray(y_true, dtype=np.int8)
    Xmeta_all, keys = build_meta_matrix(preds_dict)
    oof = np.full(len(y), np.nan, dtype=float)
    covered = np.zeros(len(y), dtype=bool)

    for tr_dates, te_dates in cv_splits:
        tr_idx, te_idx = mask_rows_by_dates(row_dates, tr_dates, te_dates)
        if len(te_idx) == 0 or len(tr_idx) < min_rows:
            continue
        y_tr = y[tr_idx]
        pos = int(y_tr.sum()); neg = int(len(y_tr) - pos)
        if pos < min_pos or neg < min_neg:
            continue

        sc = StandardScaler(with_mean=True, with_std=True)
        Xtr_s = sc.fit_transform(Xmeta_all[tr_idx])
        Xte_s = sc.transform(Xmeta_all[te_idx])

        clf = LogisticRegression(solver="lbfgs", max_iter=500, class_weight="balanced", C=1.0)
        clf.fit(Xtr_s, y_tr)
        oof[te_idx] = clf.predict_proba(Xte_s)[:, 1]
        covered[te_idx] = True

    fallback = equal_weight_avg(preds_dict, keys_order=keys)
    oof[~covered] = fallback[~covered]

    scF = StandardScaler(with_mean=True, with_std=True)
    XF = scF.fit_transform(Xmeta_all)
    clfF = LogisticRegression(solver="lbfgs", max_iter=500, class_weight="balanced", C=1.0)
    clfF.fit(XF, y)

    return oof, (scF, clfF), keys, float(np.mean(covered))

def align_columns(df_like, cols_expected):
    df_like = df_like.copy()
    for c in cols_expected:
        if c not in df_like.columns:
            df_like[c] = 0.0
    extra = [c for c in df_like.columns if c not in cols_expected and c not in TARGETS and not str(c).startswith(("tk_", "ac_"))]
    if extra:
        df_like = df_like.drop(columns=extra)
    return df_like[cols_expected]

def _to_bool_mask(m):
    arr = np.asarray(m)
    if arr.dtype == bool:
        if len(arr) != len(LONG):
            raise ValueError("Boolean mask length mismatch vs LONG.")
        return arr.copy()
    tmp = np.zeros(len(LONG), dtype=bool)
    tmp[arr.astype(int)] = True
    return tmp

def _mask_to_pos(mask, n):
    if isinstance(mask, (pd.Series, np.ndarray, list, tuple)):
        arr = np.asarray(mask)
        if arr.dtype == bool:
            return np.flatnonzero(arr)
        return arr.astype(int)
    return np.asarray(mask, dtype=int)

def recover_ticker_series_for_mask(mask):
    pos = _mask_to_pos(mask, len(LONG))
    if "ticker_series" in globals():
        ts = globals()["ticker_series"]
        try:
            if isinstance(ts, pd.Series):
                if len(ts) == len(LONG):
                    return ts.iloc[pos].astype(str).to_numpy()
                return ts[mask].astype(str).to_numpy()
            arr = np.asarray(ts)
            if len(arr) == len(LONG):
                return arr[pos].astype(str)
        except Exception:
            pass

    tk_cols = [c for c in LONG.columns if str(c).startswith("tk_")]
    if tk_cols:
        return (
            LONG.iloc[pos][tk_cols]
            .idxmax(axis=1)
            .astype(str)
            .str.replace("tk_", "", regex=False)
            .to_numpy()
        )

    # fallback
    n_out = len(pos)
    return np.array(["N/A"] * n_out, dtype=object)

def _find_train_only_model_dict():
    candidates = [
        "train_models", "models_train", "models_train_only", "train_only_models",
        "base_models_train", "models_by_target_train", "models_train_split"
    ]
    for k in candidates:
        if k in globals() and isinstance(globals()[k], dict) and globals()[k]:
            return k, globals()[k]
    if "models_by_split" in globals():
        mbs = globals()["models_by_split"]
        if isinstance(mbs, dict):
            for kk in ["train_only", "train", "train_models"]:
                if kk in mbs and isinstance(mbs[kk], dict) and mbs[kk]:
                    return f"models_by_split['{kk}']", mbs[kk]
    return None, None

def predict_proba_model_local(model_key, model_obj, X):
    X = np.asarray(X)
    if X.shape[0] == 0:
        return np.array([], dtype=np.float64)

    if "predict_proba_model" in globals():
        return np.asarray(globals()["predict_proba_model"](model_key, model_obj, X, best_iter=None), dtype=np.float64)

    if model_key == "lgbm":
        return np.asarray(model_obj.predict(X, num_iteration=getattr(model_obj, "current_iteration", lambda: None)()), dtype=np.float64)

    # sklearn-style
    return np.asarray(model_obj.predict_proba(X)[:, 1], dtype=np.float64)

def get_model_for_split(mk, target, split):
    # VALID: prefer TRAIN-only if available; else final_models
    if split == "valid" and TRAIN_ONLY_MODELS is not None:
        m = TRAIN_ONLY_MODELS.get((mk, target), None)
        if m is not None:
            return m
    return final_models.get((mk, target), None)
def report_to_dict(rep):
    """Converte report/namedtuple/dict para dict puro (JSON-safe)"""
    if rep is None:
        return None
    if isinstance(rep, dict):
        return rep
    if hasattr(rep, "_asdict"):
        return rep._asdict()
    if hasattr(rep, "__dict__"):
        return vars(rep)
    try:
        return dict(rep)
    except Exception:
        return {"value": str(rep)}

def _pick_best_ensemble_row(df_sel: pd.DataFrame, target: str, y_sel_for_caps=None):
    """
    Escolhe melhor ensemble na seleção (VALID ou TEST_EVAL):
      - up20: precision-first com recall >= MIN_RECALL_UP_SELECT (se possível)
      - dd5: recall-first com caps de FPR/PosRate (se possível)
    """
    df = df_sel.copy()
    if df.empty:
        return None

    if target == "target_up20":
        # tenta impor recall mínimo
        df1 = df[df["Recall"] >= float(MIN_RECALL_UP_SELECT)].copy()
        if not df1.empty:
            df = df1

        # ordenar: Precision, AP, F1, -FPR, -PosRate, tp
        df = df.sort_values(
            by=["Precision","AP","F1","FPR","PosRate","tp"],
            ascending=[False,  False, False, True,  True,    False],
        )
        return df.iloc[0]

    if target == "target_dd5":
        # caps dinâmicos de posrate (opcional)
        base_rate = float(np.mean(y_sel_for_caps)) if y_sel_for_caps is not None and len(y_sel_for_caps) else 0.0
        pos_cap = min(float(DD5_POSRATE_HARD_CAP), float(base_rate + DD5_POSRATE_SLACK))

        # tenta impor recall/fpr/posrate
        df1 = df.copy()
        df1 = df1[df1["Recall"] >= float(DD5_FILTER_MIN_RECALL)]
        df1 = df1[df1["FPR"]    <= float(DD5_FILTER_MAX_FPR)]
        df1 = df1[df1["PosRate"]<= float(pos_cap)]
        if not df1.empty:
            df = df1

        # ordenar: Recall, Precision, AP, -FPR, -PosRate, tp
        df = df.sort_values(
            by=["Recall","Precision","AP","FPR","PosRate","tp"],
            ascending=[False,  False,     False, True,  True,    False],
        )
        return df.iloc[0]

    # fallback
    df = df.sort_values(by=["AP","Precision","Recall"], ascending=[False,False,False])
    return df.iloc[0]

def _build_split_output(split_key: str, dates_index, tickers_arr,
                        y_up=None, y_dd=None,
                        p_up=None, p_dd=None,
                        thr_up=0.5, thr_dd=0.5):
    out = pd.DataFrame({
        "Date": pd.to_datetime(dates_index),
        "ticker": np.asarray(tickers_arr, dtype=object),
        "split": split_key.upper(),
    })

    out["p_up20"] = np.asarray(p_up, dtype=np.float64)
    out["p_dd5"]  = np.asarray(p_dd, dtype=np.float64)

    out["thr_up20"] = float(thr_up)
    out["thr_dd5"]  = float(thr_dd)

    out["pred_up20"] = (out["p_up20"].values >= float(thr_up)).astype(np.int8)
    out["pred_dd5"]  = (out["p_dd5"].values  >= float(thr_dd)).astype(np.int8)

    buy  = (out["pred_up20"].values == 1) & (out["pred_dd5"].values == 0)
    sell = (out["pred_up20"].values == 0) & (out["pred_dd5"].values == 1)

    out["buy"]  = buy.astype(np.int8)
    out["sell"] = sell.astype(np.int8)
    out["action"] = np.where(buy, "BUY", np.where(sell, "SELL", "HOLD"))

    p_upv = out["p_up20"].values.astype(np.float64)
    p_ddv = out["p_dd5"].values.astype(np.float64)

    out["buy_trust"]  = (p_upv * (1.0 - p_ddv)).astype(np.float32)
    out["sell_trust"] = ((1.0 - p_upv) * p_ddv).astype(np.float32)
    out["action_trust"] = np.where(buy, out["buy_trust"].values,
                            np.where(sell, out["sell_trust"].values, 0.0)).astype(np.float32)

    if y_up is not None:
        out["y_up20"] = np.asarray(y_up, dtype=np.float32)
    else:
        out["y_up20"] = np.nan

    if y_dd is not None:
        out["y_dd5"] = np.asarray(y_dd, dtype=np.float32)
    else:
        out["y_dd5"] = np.nan

    # confiança por target (classe e margem) — opcional
    _, tu, mu = _trust_cols_from_proba(out["p_up20"].values, float(thr_up), pred=out["pred_up20"].values)
    _, td, md = _trust_cols_from_proba(out["p_dd5"].values,  float(thr_dd), pred=out["pred_dd5"].values)
    out["trust_up20"]  = tu
    out["margin_up20"] = mu
    out["trust_dd5"]   = td
    out["margin_dd5"]  = md

    return out

def save_hist_out_minimal(hist_out: pd.DataFrame, out_path: str | Path):
    df = hist_out.copy()

    # 1) Garantir que Date é coluna (se estiver no index)
    if "Date" not in df.columns:
        # se o index se chama "Date" ou contém datas, traz para coluna
        df = df.reset_index()
        # se veio como "index" por padrão, tenta renomear para Date se fizer sentido
        if "Date" not in df.columns and "index" in df.columns:
            # se o antigo index era a data, vamos chamar de Date
            df = df.rename(columns={"index": "Date"})

    # 2) Ordenar para index sequencial ficar consistente (opcional, mas recomendado)
    sort_cols = [c for c in ["Date", "ticker"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort")

    # 3) Criar/forçar index sequencial como coluna "index"
    df = df.reset_index(drop=True)
    df.insert(0, "index", np.arange(len(df), dtype=np.int64))

    # 4) Garantir que todas colunas existem (se faltar alguma, cria NaN)
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # 5) Selecionar somente as colunas desejadas
    df_min = df[KEEP_COLS]

    # 6) Salvar (CSV ou Parquet dependendo da extensão)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet":
        df_min.to_parquet(out_path, index=False)
    else:
        df_min.to_csv(out_path, index=False)

    return df_min

def tickers_for(col0_name: str):
    m = (lvl0 == col0_name)
    return pd.Index(lvl1[m]).unique()


def run_bin_stock(output_dir="output"):
    """Run the bin_stock pipeline.

    Parameters
    ----------
    output_dir : str
        Directory for all input/output files (default "output").

    Returns
    -------
    dict
        Paths of all saved files.
    """
    global USE_CLASS_BALANCE_WEIGHTS, POLICY_USE_JOINT_THRESHOLDS, POLICY_ENTRY_TOPK_PER_DAY, POLICY_SELL_TOPK_PER_DAY, POLICY_MAX_ENTRY_DD5_RATE, POLICY_DD5_MIN_RECALL, POLICY_DD5_MAX_FPR, POLICY_UP20_Q_LOW
    global POLICY_UP20_Q_HIGH, POLICY_DD5_Q_LOW, POLICY_DD5_Q_HIGH, POLICY_N_CAND_UP20, POLICY_N_CAND_DD5, MODE, RUN_MODELS, SAVE_MODELS
    global MODEL_DIR, MODEL_TAG, ARTIFACTS_VERSION, PARQUET_PATH, TARGETS, APPLY_DAYS, VAL_DAYS, TEST_DAYS
    global HORIZON, TARGET_LOOKAHEAD_DAYS, FEATURE_LOOKAHEAD_DAYS, PURGE_DAYS, CV_FOLDS, CV_VALID_DAYS, CV_PURGE_DAYS, USE_RECENCY_WEIGHTS
    global HALF_LIFE_DAYS, N_TRIALS, EARLY_STOPPING_LGBM, MAX_TRAIN_ROWS, MAX_VALID_ROWS, MAX_POS_SAMPLES, NEG_POS_RATIO, RF_TUNE_MAX_ROWS
    global ET_TUNE_MAX_ROWS, ADD_TICKER_DUMMIES, ADD_ASSETCLASS_DUMMIES, TOPK_PER_DAY, THRESH_BETA, MIN_RECALL, MIN_PRECISION, MAX_FPR
    global NTRIALS, THRESH_POLICY_BY_TARGET, LOW_RAM_MODE, BUILD_MAX_TRAIN_ROWS_PER_TICKER, BUILD_MAX_VALID_ROWS_PER_TICKER, BUILD_MAX_TEST_ROWS_PER_TICKER, BUILD_NEG_POS_RATIO_TRAIN, TRAIN_MAX_UNIQUE_DATES
    global RNG_SEED, LONG, ticker_series, feature_cols_all, mask_labeled, mask_train, mask_valid, mask_test
    global mask_apply, X_train_full, X_valid_full, X_test_full, y_train_full, y_valid_full, y_test_full, idx_train
    global idx_valid, idx_test, w_train_full, w_valid_full, w_test_full, train_unique_dates, CV_FOLDS_SPEC, meta
    global final_models, models_loaded, models_loaded_tv, models_loaded_train, rng, idx_train_dt, train_dates, valid_dates
    global test_dates, apply_dates, dates_full, dates_lab, GPU_AVAILABLE, CALIBRATE_BASES, CALIBRATION_MODE, MIN_CV_SPLITS
    global MIN_CAL_OOF_COVERAGE, ENSEMBLE_WEIGHT_TRIALS, RECENCY_ALPHA, CARUANA_STEPS, SELECT_SPLIT, MIN_RECALL_UP_SELECT, DD5_FILTER_MIN_RECALL, DD5_FILTER_MAX_FPR
    global DD5_POSRATE_SLACK, DD5_POSRATE_HARD_CAP, TUNE_DD5_THR_ON_BUYBASE, DD5_BUY_MIN_SIGNALS_SEL, DD5_BUY_MAX_DD5RATE_SEL, DD5_BUY_GRID_Q, REQUIRE_TRAIN_ONLY_MODELS, GOOD_POLICY
    global LAMBDA_GRID, FINAL_DAYS, FINAL_MIN_DAYS, FINAL_FALLBACK_FRAC, HOLD_DAYS, ROUNDTRIP_COST_BPS, ALLOW_REENTRY_SAME_DAY, SAVE_ENSEMBLE_ARTIFACTS
    global ENSEMBLE_SAVE_DIR, TRAIN_ONLY_MODELS, TRAIN_MODELS_NAME, BEST_ON_SEL, BEST_UP, BEST_DD, THR_UP, THR_DD
    global SEL_SPLIT_LABEL, apply_df_display, HIST_SAVE, HIST_LABELED_ONLY, HIST_PATH, APPLY_SIGNALS_PATH, APPLY_DEBUG_PATH, CV_TEST_LEN_DAYS
    global CV_STEP_DAYS, trainmodels, finalmodels, date_mask_train, date_mask_valid, date_mask_test, date_mask_apply, mask_lab_dates

    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # Time-aware multi-model training (LGBM GPU if available, HGB, RF, ET)
    # Improvements applied:
    #  1) Guardrail: PURGE >= true lookahead (set TARGET_LOOKAHEAD_DAYS correctly!)
    #  2) Walk-forward CV inside TRAIN for hyperparam selection (reduces overfit to one VALID window)
    #  3) Recency sample-weights for ALL models (handles regime shift)
    #  4) Asset-class feature (to mitigate mixing equity/index/fx/crypto/futures)
    #  5) Threshold policy to REDUCE FALSE POSITIVES:
    #       - precision-focused F_beta (beta<1) + optional FPR constraint
    #  6) Extra reporting: FPR, signals/day, Precision@K per day (VALID/TEST)
    #
    # Keeps:
    #  - tqdm progress bars
    #  - printed metrics
    #  - display() of metrics table and apply outputs
    # ============================================================




    USE_CLASS_BALANCE_WEIGHTS = True


    # Configs de policy (se ausentes)
    POLICY_USE_JOINT_THRESHOLDS = True
    POLICY_ENTRY_TOPK_PER_DAY = 20
    POLICY_SELL_TOPK_PER_DAY = None
    POLICY_MAX_ENTRY_DD5_RATE = 0.35
    POLICY_DD5_MIN_RECALL = 0.60
    POLICY_DD5_MAX_FPR = 0.20
    POLICY_UP20_Q_LOW, POLICY_UP20_Q_HIGH = 0.80, 0.999
    POLICY_DD5_Q_LOW, POLICY_DD5_Q_HIGH = 0.20, 0.95
    POLICY_N_CAND_UP20 = 25
    POLICY_N_CAND_DD5 = 25
    # -------------------------
    # USER CONFIG
    # -------------------------
    MODE              = "load"   # "auto" (load if exists else train) | "train" | "load"
    RUN_MODELS        = ["lgbm", "hgb", "rf", "et"]
    SAVE_MODELS       = True
    MODEL_DIR         = os.path.join(output_dir, "models")
    MODEL_TAG         = "multi_model_timeaware_v11_cv_recency_asset_thr"   # bump tag
    ARTIFACTS_VERSION = 7

    PARQUET_PATH      = os.path.join(output_dir, "expanded_stock_reduced.parquet")
    TARGETS           = ["target_up20", "target_dd5"]

    # --- time split ---
    APPLY_DAYS        = 5
    VAL_DAYS          = 126
    TEST_DAYS         = 252
    HORIZON           = 90

    # !!! IMPORTANT !!! Put the REAL lookahead used when creating the targets.
    # If you used forward window of 60/90 days to label target_up20 / target_dd5, set it here.
    TARGET_LOOKAHEAD_DAYS  = 90   # <-- CHANGE if your target uses >30 days lookahead
    FEATURE_LOOKAHEAD_DAYS = 0    # set >0 if ANY feature inadvertently leaks future info
    PURGE_DAYS        = max(HORIZON, TARGET_LOOKAHEAD_DAYS, FEATURE_LOOKAHEAD_DAYS)

    # --- walk-forward CV inside TRAIN (for tuning) ---
    CV_FOLDS          = 3
    CV_VALID_DAYS     = 60  # length of each fold's validation window (in "unique date steps")
    CV_PURGE_DAYS     = PURGE_DAYS

    # --- recency weights (all models) ---
    USE_RECENCY_WEIGHTS = True
    HALF_LIFE_DAYS      = 252  # ~1 trading year

    # --- tuning trials ---
    N_TRIALS = {"lgbm": 12, "hgb": 10, "rf": 8, "et": 8}
    EARLY_STOPPING_LGBM = 100

    # --- tuning downsampling caps (speed only) ---
    MAX_TRAIN_ROWS    = 400_000
    MAX_VALID_ROWS    = None
    MAX_POS_SAMPLES   = None
    NEG_POS_RATIO     = None
    RF_TUNE_MAX_ROWS  = 100_000
    ET_TUNE_MAX_ROWS  = 80_000

    # --- features ---
    ADD_TICKER_DUMMIES   = True
    ADD_ASSETCLASS_DUMMIES = True

    # --- decision / reporting ---
    TOPK_PER_DAY      = 20

    # Precision-focused threshold policy (reduce FP):
    THRESH_BETA       = 0.5   # beta<1 favors precision (reduces false positives)
    MIN_RECALL        = 0.10  # won't accept thresholds that kill recall below this (soft constraint)
    MIN_PRECISION     = 0.50  # soft constraint; auto-relaxes if impossible
    MAX_FPR           = 0.25  # soft constraint; auto-relaxes if impossible
    # ============================
    # Configuração de Hiperparâmetros
    # ============================

    # Número de trials para o RandomizedSearch manual no CV
    # Ajuste conforme seu tempo disponível (ex: lgbm=20 se tiver GPU, rf=10 se tiver CPU rápida)
    NTRIALS = {
        "lgbm": 12,
        "hgb":  10,
        "rf":   8,
        "et":   8
    }

    # Configuração de Early Stopping para LightGBM
    EARLY_STOPPING_LGBM = 100

    # Se você não tiver definido estas variáveis globais anteriormente, defina-as aqui para garantir:
    if "RUN_MODELS" not in globals():
        RUN_MODELS = ["lgbm", "hgb", "rf", "et"]

    if "TARGETS" not in globals():
        TARGETS = ["target_up20", "target_dd5"]

    # ------------------------------------------------------------
    # TRADING POLICY (joint up20+dd5) — matches your real actions:
    #   BUY  when (up20==1) and (dd5==0)
    #   SELL when (up20==0) and (dd5==1)
    # We choose thresholds jointly on VALID to reduce false positives.
    # ------------------------------------------------------------
    POLICY_USE_JOINT_THRESHOLDS = True

    # On each day, we only consider the Top-K by up20 proba among eligible BUY candidates
    POLICY_ENTRY_TOPK_PER_DAY = TOPK_PER_DAY

    # Optional: also cap sells per day (None = no cap). Usually OK to keep None.
    POLICY_SELL_TOPK_PER_DAY = None

    # Constraints for the joint policy search (soft-relaxed automatically if impossible)
    POLICY_MAX_ENTRY_DD5_RATE = 0.35   # among BUY signals, fraction of true dd5 events must be <= this
    POLICY_DD5_MIN_RECALL     = 0.60   # dd5 model must still catch drawdowns reasonably
    POLICY_DD5_MAX_FPR        = 0.20   # dd5 false positives matter (avoid/sell wrongly)

    # Candidate thresholds (quantile ranges) for joint search
    POLICY_UP20_Q_LOW, POLICY_UP20_Q_HIGH = 0.80, 0.999
    POLICY_DD5_Q_LOW,  POLICY_DD5_Q_HIGH  = 0.20, 0.95

    POLICY_N_CAND_UP20 = 25
    POLICY_N_CAND_DD5  = 25

    # Optional: allow different per-target threshold constraints (still used as fallback)
    THRESH_POLICY_BY_TARGET = {
        "target_up20": dict(beta=THRESH_BETA, min_precision=MIN_PRECISION, min_recall=MIN_RECALL, max_fpr=MAX_FPR),
        # dd5 drives "avoid/sell", so reduce false positives harder than up20:
        "target_dd5":  dict(beta=0.7,        min_precision=0.75,         min_recall=0.60,     max_fpr=0.20),
    }






    # Helper para reports (fix #3)



    USE_CLASS_BALANCE_WEIGHTS = True













    # -------------------------
    # GPU detect for LightGBM (best-effort)
    # -------------------------

    GPU_AVAILABLE = _nvidia_smi_exists()

    # -------------------------
    # Suppress stdout/stderr (useful for LightGBM)
    # -------------------------

    # -------------------------
    # Atomic persistence
    # -------------------------








    # -------------------------
    # Metrics helpers
    # -------------------------



    # ============================
    # Função genérica para predições (corrige NameError)
    # ============================




    # -------------------------
    # Precision-focused threshold selection (reduces FP)
    #   - maximize F_beta with beta<1 (precision favored)
    #   - soft constraints: MIN_PRECISION, MIN_RECALL, MAX_FPR
    #   - auto-relax constraints if no threshold satisfies them
    # -------------------------


    # ============================================================
    # JOINT POLICY helpers (up20 + dd5)
    # ============================================================






    # -------------------------
    # Speedy undersampling (TUNING ONLY)
    # -------------------------



    # -------------------------
    # Recency weights
    # -------------------------

    # -------------------------
    # Asset class (heuristic)
    # -------------------------


    # ============================================================
    # LOAD PARQUET + BUILD LONG (LOW-RAM)
    # - evita explodir RAM com LONG full + ticker one-hot
    # - amostra TRAIN/VALID/TEST por ticker (mantém todos positivos)
    # - mantém APPLY completo (últimos APPLY_DAYS) para todos tickers
    # ============================================================

    # -------------------------
    # LOW-RAM knobs (AJUSTE AQUI)
    # -------------------------
    LOW_RAM_MODE = True

    # NUNCA faça one-hot de ticker em low-ram:
    ADD_TICKER_DUMMIES = False
    # Asset class one-hot é pequeno (ok)
    ADD_ASSETCLASS_DUMMIES = True

    # Limites por ticker (controle de RAM)
    BUILD_MAX_TRAIN_ROWS_PER_TICKER = 2500   # ~pos + neg amostrado
    BUILD_MAX_VALID_ROWS_PER_TICKER = 1200
    BUILD_MAX_TEST_ROWS_PER_TICKER  = 1200

    # Negativos por positivo (em TRAIN). Ex: 4 => ~80% neg, 20% pos (se evento raro)
    BUILD_NEG_POS_RATIO_TRAIN = 4

    # Limita quantos DIAS de treino entram (recency já existe, então ajuda muito)
    TRAIN_MAX_UNIQUE_DATES = 1800   # use None para não limitar

    RNG_SEED = 42

    # -------------------------
    # Load parquet
    # -------------------------
    df = pd.read_parquet(PARQUET_PATH)
    assert isinstance(df.columns, pd.MultiIndex), "Parquet needs MultiIndex columns (feature/target, ticker)."

    if not np.issubdtype(df.index.dtype, np.datetime64):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    lvl0 = df.columns.get_level_values(0).astype(str)
    lvl1 = df.columns.get_level_values(1).astype(str)
    assert all(t in set(lvl0) for t in TARGETS), "Required targets missing."

    is_target = lvl0.str.startswith("target_")
    all_feature_names = sorted(set(lvl0[~is_target]))

    # tickers que têm AMBOS targets
    tickers = sorted(set(lvl1[(lvl0 == TARGETS[0])]) & set(lvl1[(lvl0 == TARGETS[1])]))

    # -------------------------
    # Dates_full e dates_lab (SEM construir LONG)
    # -------------------------
    dates_full = np.array(sorted(df.index.unique()))

    # labeled dates = datas onde existe pelo menos 1 ticker com ambos targets não-NaN
    mask_lab_dates = np.zeros(len(df.index), dtype=bool)
    for tk in tqdm(tickers, desc="Scanning labeled dates (low-ram)"):
        yu = pd.to_numeric(df[("target_up20", tk)], errors="coerce").to_numpy()
        yd = pd.to_numeric(df[("target_dd5",  tk)], errors="coerce").to_numpy()
        mask_lab_dates |= (~np.isnan(yu)) & (~np.isnan(yd))

    dates_lab = np.array(sorted(df.index[mask_lab_dates].unique()))

    # -------------------------
    # Split por datas (igual sua lógica)
    # -------------------------
    apply_dates = dates_full[-APPLY_DAYS:]

    need_lab = TEST_DAYS + PURGE_DAYS + VAL_DAYS + PURGE_DAYS + 5
    assert len(dates_lab) >= need_lab, f"Labeled history too short. Need >= {need_lab} labeled dates, have {len(dates_lab)}."

    i_end = len(dates_lab)

    test_dates = dates_lab[i_end-TEST_DAYS:i_end]
    i_end -= TEST_DAYS
    i_end -= PURGE_DAYS

    valid_dates = dates_lab[i_end-VAL_DAYS:i_end]
    i_end -= VAL_DAYS
    i_end -= PURGE_DAYS

    train_dates = dates_lab[:i_end]
    assert len(train_dates) > 0

    # LOW-RAM: limita o histórico de treino (muito útil com recency weights)
    if LOW_RAM_MODE and (TRAIN_MAX_UNIQUE_DATES is not None) and (len(train_dates) > TRAIN_MAX_UNIQUE_DATES):
        train_dates = train_dates[-TRAIN_MAX_UNIQUE_DATES:]

    print(
        f"Max feature date: {pd.Timestamp(dates_full[-1]).date()} | "
        f"Max labeled date: {pd.Timestamp(dates_lab[-1]).date()}"
    )
    print(
        f"Dates — TRAIN: {pd.Timestamp(train_dates[0]).date()} → {pd.Timestamp(train_dates[-1]).date()} "
        f"| VALID: {pd.Timestamp(valid_dates[0]).date()} → {pd.Timestamp(valid_dates[-1]).date()} "
        f"| TEST: {pd.Timestamp(test_dates[0]).date()} → {pd.Timestamp(test_dates[-1]).date()} "
        f"| APPLY(last {APPLY_DAYS}): {pd.Timestamp(apply_dates[0]).date()} → {pd.Timestamp(apply_dates[-1]).date()} "
        f"| PURGE={PURGE_DAYS} (guard={max(TARGET_LOOKAHEAD_DAYS, FEATURE_LOOKAHEAD_DAYS)})"
    )

    date_mask_train = df.index.isin(train_dates)
    date_mask_valid = df.index.isin(valid_dates)
    date_mask_test  = df.index.isin(test_dates)
    date_mask_apply = df.index.isin(apply_dates)

    rng = np.random.default_rng(RNG_SEED)






    # -------------------------
    # Build sampled LONG
    # -------------------------
    rows = []
    for tk in tqdm(tickers, desc="Building sampled LONG (low-ram)"):
        X_tk = df.xs(tk, level=1, axis=1)
        # --- dentro do loop onde você monta X_tk ---
        X_tk = _normalize_feature_frame(X_tk, ticker=tk)

        # debug opcional (não quebra):
        missing_now = [c for c in all_feature_names if c not in X_tk.columns]
        if missing_now:
            print(f"[WARN] {tk}: {len(missing_now)} features ausentes (ex: {missing_now[:10]})")

        # ✅ troca o .loc por reindex (evita KeyError e padroniza schema)
        X_feat = (X_tk.reindex(columns=all_feature_names)
                      .apply(pd.to_numeric, errors="coerce")
                      .replace([np.inf, -np.inf], np.nan)
                      .fillna(0.0)
                      .astype(np.float32))

        # targets (float32 com NaN permitido)
        y_up_raw = pd.to_numeric(df[("target_up20", tk)], errors="coerce").replace([np.inf, -np.inf], np.nan)
        y_dd_raw = pd.to_numeric(df[("target_dd5",  tk)], errors="coerce").replace([np.inf, -np.inf], np.nan)

        y_up = clean_binary_target(y_up_raw).astype("float32")
        y_dd = clean_binary_target(y_dd_raw).astype("float32")

        # labeled mask por linha (data) para este ticker
        lab = (~y_up.isna().to_numpy()) & (~y_dd.isna().to_numpy())

        # split indices (por data)
        idx_tr = _sample_train_indices(lab & date_mask_train, y_up.fillna(0).astype("int8").to_numpy(),
                                       y_dd.fillna(0).astype("int8").to_numpy(),
                                       max_rows=BUILD_MAX_TRAIN_ROWS_PER_TICKER,
                                       neg_pos_ratio=BUILD_NEG_POS_RATIO_TRAIN)

        idx_va = _cap_indices(np.flatnonzero(lab & date_mask_valid), BUILD_MAX_VALID_ROWS_PER_TICKER)
        idx_te = _cap_indices(np.flatnonzero(lab & date_mask_test ), BUILD_MAX_TEST_ROWS_PER_TICKER)

        # APPLY: não exige labels (últimos dias)
        idx_ap = np.flatnonzero(date_mask_apply)

        idx_keep = np.unique(np.concatenate([idx_tr, idx_va, idx_te, idx_ap])).astype(int)
        if idx_keep.size == 0:
            continue

        block = X_feat.iloc[idx_keep].copy()
        block["target_up20"] = y_up.iloc[idx_keep].to_numpy()
        block["target_dd5"]  = y_dd.iloc[idx_keep].to_numpy()
        block["ticker"]      = tk
        block["asset_class"] = infer_asset_class(tk)
        rows.append(block)

        # libera memória por ticker
        del X_tk, X_feat, y_up_raw, y_dd_raw, y_up, y_dd, block
        gc.collect()

    LONG = pd.concat(rows, axis=0).sort_index()
    ticker_series = LONG["ticker"].copy()

    # one-hot APENAS asset_class (pequeno)
    if ADD_ASSETCLASS_DUMMIES:
        ac_dummies = pd.get_dummies(LONG["asset_class"], prefix="ac", dtype=np.uint8)
    else:
        ac_dummies = None

    to_drop = ["ticker", "asset_class"]
    base = LONG.drop(columns=to_drop)

    if ac_dummies is not None:
        base = pd.concat([base, ac_dummies], axis=1)

    LONG = base

    feature_cols_all = [c for c in LONG.columns if c not in TARGETS]

    # targets mantém NaN (float32)
    for t in TARGETS:
        LONG[t] = pd.to_numeric(LONG[t], errors="coerce").astype("float32")

    mask_labeled = LONG["target_up20"].notna() & LONG["target_dd5"].notna()
    print("Long shape (sampled):", LONG.shape, "| labeled rows:", int(mask_labeled.sum()))

    # libera o df wide (gigante) — MUITO importante para não estourar RAM mais tarde
    del df, rows
    gc.collect()



    # ============================================================
    # TIME SPLIT: TRAIN / VALID / TEST / APPLY with PURGE gaps
    # ============================================================
    # Guardrail (must hold, otherwise leakage risk)
    REQUIRED_PURGE = max(TARGET_LOOKAHEAD_DAYS, FEATURE_LOOKAHEAD_DAYS)
    assert PURGE_DAYS >= REQUIRED_PURGE, f"PURGE_DAYS={PURGE_DAYS} < REQUIRED_PURGE={REQUIRED_PURGE}"

    # --- FULL dates for APPLY (latest features) ---
    dates_full = np.array(sorted(LONG.index.unique()))

    # --- LABELED dates for TRAIN/VALID/TEST (targets exist) ---
    dates_lab = np.array(sorted(LONG.index[mask_labeled].unique()))

    # APPLY uses the truly latest dates (even if targets are NaN there)
    apply_dates = dates_full[-APPLY_DAYS:]

    # TRAIN/VALID/TEST use only labeled dates (so metrics work)
    need_lab = TEST_DAYS + PURGE_DAYS + VAL_DAYS + PURGE_DAYS + 5
    assert len(dates_lab) >= need_lab, f"Labeled history too short. Need >= {need_lab} labeled dates, have {len(dates_lab)}."

    i_end = len(dates_lab)

    test_dates = dates_lab[i_end-TEST_DAYS:i_end]
    i_end -= TEST_DAYS
    i_end -= PURGE_DAYS

    valid_dates = dates_lab[i_end-VAL_DAYS:i_end]
    i_end -= VAL_DAYS
    i_end -= PURGE_DAYS

    train_dates = dates_lab[:i_end]
    assert len(train_dates) > 0

    mask_apply = LONG.index.isin(apply_dates)                      # <-- NOT restricted to labeled
    mask_test  = mask_labeled & LONG.index.isin(test_dates)
    mask_valid = mask_labeled & LONG.index.isin(valid_dates)
    mask_train = mask_labeled & LONG.index.isin(train_dates)

    print(
        f"Max feature date: {pd.Timestamp(dates_full[-1]).date()} | "
        f"Max labeled date: {pd.Timestamp(dates_lab[-1]).date()}"
    )
    print(
        f"Dates — TRAIN: {pd.Timestamp(train_dates[0]).date()} → {pd.Timestamp(train_dates[-1]).date()} "
        f"| VALID: {pd.Timestamp(valid_dates[0]).date()} → {pd.Timestamp(valid_dates[-1]).date()} "
        f"| TEST: {pd.Timestamp(test_dates[0]).date()} → {pd.Timestamp(test_dates[-1]).date()} "
        f"| APPLY(last {APPLY_DAYS}): {pd.Timestamp(apply_dates[0]).date()} → {pd.Timestamp(apply_dates[-1]).date()} "
        f"| PURGE={PURGE_DAYS} (guard={max(TARGET_LOOKAHEAD_DAYS, FEATURE_LOOKAHEAD_DAYS)})"
    )


    # FULL matrices
    X_train_full = LONG.loc[mask_train, feature_cols_all].values
    X_valid_full = LONG.loc[mask_valid, feature_cols_all].values
    X_test_full  = LONG.loc[mask_test,  feature_cols_all].values

    y_train_full = {t: LONG.loc[mask_train, t].values.astype('int8') for t in TARGETS}
    y_valid_full = {t: LONG.loc[mask_valid, t].values.astype('int8') for t in TARGETS}
    y_test_full  = {t: LONG.loc[mask_test,  t].values.astype('int8') for t in TARGETS}

    # indices for recency weights
    idx_train = LONG.loc[mask_train].index
    idx_valid = LONG.loc[mask_valid].index
    idx_test  = LONG.loc[mask_test].index

    w_train_full = make_time_weights(idx_train, HALF_LIFE_DAYS) if USE_RECENCY_WEIGHTS else None
    w_valid_full = make_time_weights(idx_valid, HALF_LIFE_DAYS) if USE_RECENCY_WEIGHTS else None
    w_test_full  = make_time_weights(idx_test,  HALF_LIFE_DAYS) if USE_RECENCY_WEIGHTS else None  # not used in training; ok for analysis

    # ============================================================
    # WALK-FORWARD CV folds inside TRAIN
    # ============================================================
    train_unique_dates = np.array(sorted(pd.to_datetime(LONG.loc[mask_train].index.unique())))

    assert len(train_unique_dates) >= (CV_FOLDS * (CV_VALID_DAYS + CV_PURGE_DAYS) + 10)



    CV_FOLDS_SPEC = make_walk_forward_folds(train_unique_dates, CV_FOLDS, CV_VALID_DAYS, CV_PURGE_DAYS)

    MIN_FOLDS_REQUIRED = 1  # FAST_MODE allows 1 fold
    assert len(CV_FOLDS_SPEC) >= MIN_FOLDS_REQUIRED, (
        f"CV folds too few: got {len(CV_FOLDS_SPEC)}; need >= {MIN_FOLDS_REQUIRED}. "
        f"Try reducing CV_VALID_DAYS={CV_VALID_DAYS}, CV_PURGE_DAYS={CV_PURGE_DAYS}, or CV_FOLDS={CV_FOLDS}."
    )

    if len(CV_FOLDS_SPEC) < CV_FOLDS:
        print(f"[warn] Requested CV_FOLDS={CV_FOLDS} but only {len(CV_FOLDS_SPEC)} fold(s) fit in TRAIN.")



    # We'll build fold masks using the LONG index (repeated rows), based on date ranges.
    # For each fold: training = dates < train_end_date ; validation = [valid_start_date, valid_end_date)
    # NOTE: purge already applied via indices when choosing train_end_date.

    # ============================================================
    # TRAIN / LOAD
    # ============================================================
    meta_loaded, models_loaded_tv = (None, {})
    models_loaded_train = {}

    if MODE in ("auto", "load"):
        meta_loaded, models_loaded_tv = load_artifacts(RUN_MODELS, TARGETS, kind="tv")
        _,          models_loaded_train = load_artifacts(RUN_MODELS, TARGETS, kind="train")
    models_loaded = models_loaded_tv
    meta = meta_loaded if meta_loaded is not None else {}
    need_train = (MODE == "train")
    if MODE == "auto":
        for mk in RUN_MODELS:
            for t in TARGETS:
                # se faltar QUALQUER um dos dois, força treino
                if (models_loaded_tv.get((mk, t)) is None) or (models_loaded_train.get((mk, t)) is None):
                    need_train = True
                    break



    # -------------------------
    # Hyperparam samplers
    # -------------------------




    # -------------------------
    # LightGBM train with GPU fallback + sample weights
    # -------------------------

    # -------------------------
    # sklearn fit with sample_weight (robust)
    # -------------------------


    # -------------------------
    # predict proba
    # -------------------------

    # ============================================================
    # WALK-FORWARD TUNING (score = mean AP across folds)
    # ============================================================

    # Pré-calcula coisas do TRAIN uma única vez
    idx_train_dt = pd.to_datetime(idx_train)  # idx_train já existe no seu split







    # ============================================================
    # FINAL FITS (FULL TRAIN then FULL TRAIN+VALID)
    # ============================================================




    # ============================================================
    # ORCHESTRATION
    # ============================================================
    meta = {
        "artifacts_version": ARTIFACTS_VERSION,
        "model_tag": MODEL_TAG,
        "models": RUN_MODELS.copy(),
        "feature_cols": list(feature_cols_all),
        "add_ticker_dummies": bool(ADD_TICKER_DUMMIES),
        "add_assetclass_dummies": bool(ADD_ASSETCLASS_DUMMIES),
        "recency_weights": {"enabled": bool(USE_RECENCY_WEIGHTS), "half_life_days": int(HALF_LIFE_DAYS)},
        "threshold_policy": {
            "beta": float(THRESH_BETA),
            "min_recall": float(MIN_RECALL),
            "min_precision": float(MIN_PRECISION),
            "max_fpr": float(MAX_FPR),
            "by_target": THRESH_POLICY_BY_TARGET,
        },
        "trade_policy": {
            "enabled_joint_thresholds": bool(POLICY_USE_JOINT_THRESHOLDS),
            "buy_rule":  "BUY when (up20==1) and (dd5==0)",
            "sell_rule": "SELL when (up20==0) and (dd5==1)",
            "entry_topk_per_day": int(POLICY_ENTRY_TOPK_PER_DAY) if POLICY_ENTRY_TOPK_PER_DAY is not None else None,
            "sell_topk_per_day":  int(POLICY_SELL_TOPK_PER_DAY) if POLICY_SELL_TOPK_PER_DAY is not None else None,
            "max_entry_dd5_rate": float(POLICY_MAX_ENTRY_DD5_RATE),
            "dd5_min_recall": float(POLICY_DD5_MIN_RECALL),
            "dd5_max_fpr": float(POLICY_DD5_MAX_FPR),
        },
        "dates": {
            "train_start": str(pd.Timestamp(train_dates[0]).date()),
            "train_end":   str(pd.Timestamp(train_dates[-1]).date()),
            "valid_start": str(pd.Timestamp(valid_dates[0]).date()),
            "valid_end":   str(pd.Timestamp(valid_dates[-1]).date()),
            "test_start":  str(pd.Timestamp(test_dates[0]).date()),
            "test_end":    str(pd.Timestamp(test_dates[-1]).date()),
            "apply_start": str(pd.Timestamp(apply_dates[0]).date()),
            "apply_end":   str(pd.Timestamp(apply_dates[-1]).date()),
            "purge_days":  int(PURGE_DAYS),
            "horizon":     int(HORIZON),
            "target_lookahead_days": int(TARGET_LOOKAHEAD_DAYS),
            "feature_lookahead_days": int(FEATURE_LOOKAHEAD_DAYS),
            "cv_folds": int(len(CV_FOLDS_SPEC)),
            "cv_valid_days": int(CV_VALID_DAYS),
            "cv_purge_days": int(CV_PURGE_DAYS),
            "gpu_available_detected": bool(GPU_AVAILABLE),
        },
        "targets": {t: {} for t in TARGETS},
        "policy_per_model": {},   # NEW: stores joint thresholds + policy metrics for each model_key
    }

    final_models = {}
    metrics_list = []

    if (MODE in ("load", "auto")) and not need_train:
        meta = meta_loaded if meta_loaded is not None else meta

    if need_train:
        print(f"\nGPU available (nvidia-smi): {GPU_AVAILABLE}")
        print(f"Walk-forward CV folds: {len(CV_FOLDS_SPEC)} | valid_len={CV_VALID_DAYS} | purge={CV_PURGE_DAYS}")

        # ============================
        # ============================
        # Main Training Loop (Final - com predict_proba_model genérica)
        # ============================

        train_models = {}   # models fit on TRAIN only (used for VALID scoring/calibration)
        final_models = {}   # models fit on TRAIN+VALID (used for TEST/FINAL/APPLY)

        # Ensure full matrices are available (snake_case aliases se necessário)
        X_tv = np.vstack([X_train_full, X_valid_full])
        w_tv = None

        # Recency weights for TV set
        if USE_RECENCY_WEIGHTS:
            idx_tv = pd.to_datetime(LONG.loc[mask_train | mask_valid].index)
            w_tv = make_time_weights(idx_tv, HALF_LIFE_DAYS)

        print(f"Walking forward... TRAIN rows: {len(X_train_full)}, TRAIN+VALID rows: {len(X_tv)}")

        # ============================
        # Main Training Loop (REPLACE THIS WHOLE BLOCK)
        # ============================

        train_models = {}   # models fit on TRAIN only (used for VALID scoring/calibration)
        final_models = {}   # models fit on TRAIN+VALID (used for TEST/FINAL/APPLY)
        metrics_list = []

        # --- build TRAIN+VALID matrices once ---
        X_tv = np.vstack([X_train_full, X_valid_full])

        w_tv = None
        if USE_RECENCY_WEIGHTS:
            idx_tv = pd.to_datetime(LONG.loc[mask_train | mask_valid].index)
            w_tv = make_time_weights(idx_tv, HALF_LIFE_DAYS)

        print(f"Walking forward... TRAIN rows: {len(X_train_full)}, TRAIN+VALID rows: {len(X_tv)}")

        # ============================
        # Main Training Loop (PATCH B)
        # - Joint threshold selection ONCE per model_key
        # - Store meta consistently in meta["targets"][target][model_key]
        # ============================

        train_models = {}   # models fit on TRAIN only (for VALID predictions)
        final_models = {}   # models fit on TRAIN+VALID (for TEST/APPLY predictions)

        # TRAIN+VALID matrices
        X_tv = np.vstack([X_train_full, X_valid_full])

        w_tv = None
        if USE_RECENCY_WEIGHTS:
            idx_tv = pd.to_datetime(LONG.loc[mask_train | mask_valid].index)
            w_tv = make_time_weights(idx_tv, HALF_LIFE_DAYS)

        print(f"Walking forward... TRAIN rows: {len(X_train_full)}, TRAIN+VALID rows: {len(X_tv)}")

        # Ensure meta structure is consistent
        meta.setdefault("targets", {t: {} for t in TARGETS})
        meta.setdefault("policy_per_model", {})

        for model_key in RUN_MODELS:
            p_valid_by_t = {}
            p_test_by_t  = {}
            best_by_t    = {}

            # ---- 1) Train both targets first (no joint threshold here) ----
            for target in TARGETS:
                n_trials = int(NTRIALS.get(model_key, 0))

                # --- Tuning ---
                if n_trials <= 0:
                    best = None
                else:
                    if model_key == "lgbm":
                        best = tune_lgbm_cv(target, n_trials)
                    elif model_key == "hgb":
                        best = tune_hgb_cv(target, n_trials)
                    elif model_key == "rf":
                        best = tune_rf_cv(target, n_trials)
                    elif model_key == "et":
                        best = tune_et_cv(target, n_trials)
                    else:
                        raise ValueError(f"Unknown model {model_key}")

                if best is None:
                    print(f"[skip] {model_key}/{target}: no tuning result")
                    train_models[(model_key, target)] = None
                    final_models[(model_key, target)] = None
                    continue

                best_by_t[target] = best

                # --- Fit TRAIN-only model (for VALID predictions) ---
                y_tr = y_train_full[target].astype("int8")

                if model_key == "lgbm":
                    rounds = int(globals().get("FINAL_LGB_ROUNDS", 600))
                    model_train, _ = fit_final_lgbm(best.params, X_train_full, y_tr, w_train_full, num_boost_round=rounds)
                elif model_key == "hgb":
                    model_train = fit_final_hgb_params(best.params, X_train_full, y_tr, w_train_full)
                elif model_key == "rf":
                    model_train = fit_final_rf_params(best.params, X_train_full, y_tr, w_train_full)
                elif model_key == "et":
                    model_train = fit_final_et_params(best.params, X_train_full, y_tr, w_train_full)

                train_models[(model_key, target)] = model_train

                p_valid = predict_proba_model(model_key, model_train, X_valid_full)
                p_valid = np.nan_to_num(p_valid, nan=0.0, posinf=1.0, neginf=0.0)
                p_valid_by_t[target] = p_valid

                # --- Fit TRAIN+VALID model (for TEST/APPLY predictions) ---
                y_tv_target = np.concatenate([y_tr, y_valid_full[target].astype("int8")])

                if model_key == "lgbm":
                    rounds = int(globals().get("FINAL_LGB_ROUNDS", 600))
                    model_final, _ = fit_final_lgbm(best.params, X_tv, y_tv_target, w_tv, num_boost_round=rounds)
                elif model_key == "hgb":
                    model_final = fit_final_hgb_params(best.params, X_tv, y_tv_target, w_tv)
                elif model_key == "rf":
                    model_final = fit_final_rf_params(best.params, X_tv, y_tv_target, w_tv)
                elif model_key == "et":
                    model_final = fit_final_et_params(best.params, X_tv, y_tv_target, w_tv)

                final_models[(model_key, target)] = model_final

                p_test = predict_proba_model(model_key, model_final, X_test_full)
                p_test = np.nan_to_num(p_test, nan=0.0, posinf=1.0, neginf=0.0)
                p_test_by_t[target] = p_test

            # ---- 2) JOINT threshold selection ONCE per model_key ----
            have_both = ("target_up20" in p_valid_by_t) and ("target_dd5" in p_valid_by_t) \
                        and (train_models.get((model_key, "target_up20")) is not None) \
                        and (train_models.get((model_key, "target_dd5")) is not None)

            if POLICY_USE_JOINT_THRESHOLDS and have_both:
                y_va_up = y_valid_full["target_up20"].astype("int8")
                y_va_dd = y_valid_full["target_dd5"].astype("int8")
                p_va_up = p_valid_by_t["target_up20"]
                p_va_dd = p_valid_by_t["target_dd5"]

                thr_up_fallback = choose_threshold_precision_focused(
                    y_va_up, p_va_up,
                    beta=THRESH_BETA,
                    min_precision=MIN_PRECISION,
                    min_recall=MIN_RECALL,
                    max_fpr=MAX_FPR
                )
                thr_dd_fallback = choose_threshold_precision_focused(
                    y_va_dd, p_va_dd,
                    beta=THRESH_BETA,
                    min_precision=MIN_PRECISION,
                    min_recall=MIN_RECALL,
                    max_fpr=MAX_FPR
                )

                up_cands = quantile_threshold_candidates(p_va_up, POLICY_UP20_Q_LOW, POLICY_UP20_Q_HIGH,
                                                        POLICY_N_CAND_UP20, include=thr_up_fallback)
                dd_cands = quantile_threshold_candidates(p_va_dd, POLICY_DD5_Q_LOW, POLICY_DD5_Q_HIGH,
                                                        POLICY_N_CAND_DD5, include=thr_dd_fallback)

                pol_metrics = choose_joint_thresholds_policy(
                    idx_valid, y_va_up, y_va_dd, p_va_up, p_va_dd,
                    entry_topk=POLICY_ENTRY_TOPK_PER_DAY,
                    sell_topk=POLICY_SELL_TOPK_PER_DAY,
                    max_entry_dd5_rate=POLICY_MAX_ENTRY_DD5_RATE,
                    dd5_min_recall=POLICY_DD5_MIN_RECALL,
                    dd5_max_fpr=POLICY_DD5_MAX_FPR,
                    up20_cands=up_cands,
                    dd5_cands=dd_cands
                )

                thr_up, thr_dd = get_joint_thresholds(pol_metrics)

                print(f"POLICY {model_key} joint thresholds (VALID)")
                print(f"  thr_up20={thr_up:.4f} thr_dd5={thr_dd:.4f}")
                pm_dict = report_to_dict(pol_metrics)  # keep it as dict

                def pmget(key, default=np.nan):
                    if pm_dict is None:
                        return default
                    return pm_dict.get(key, default)

                print(f"POLICY {model_key} joint thresholds (VALID)")
                print(f"  thr_up20={thr_up:.4f} thr_dd5={thr_dd:.4f}")
                print(f"  entry_precision_up20={pmget('entry_precision_up20'):.4f} "
                      f"entry_dd5_rate={pmget('entry_dd5_rate'):.4f} "
                      f"entry_day={pmget('entry_per_day'):.2f}")
                print(f"  dd5_precision={pmget('dd5_precision'):.4f} "
                      f"dd5_recall={pmget('dd5_recall'):.4f} "
                      f"dd5_fpr={pmget('dd5_fpr'):.4f}")
                print(f"  sell_dd5_rate={pmget('sell_dd5_rate'):.4f} "
                      f"sell_up20_rate={pmget('sell_up20_rate'):.4f} "
                      f"sell_day={pmget('sell_per_day'):.2f}")

                meta["policy_per_model"][model_key] = {
                    "thr_up20": float(thr_up),
                    "thr_dd5":  float(thr_dd),
                    "valid_policy_metrics": pm_dict,
                }


            else:
                thr_up, thr_dd = 0.5, 0.5
                meta["policy_per_model"][model_key] = {
                    "thr_up20": float(thr_up),
                    "thr_dd5":  float(thr_dd),
                    "valid_policy_metrics": None,
                }

            # ---- 3) Evaluate + save meta per target using JOINT thresholds ----
            for target in TARGETS:
                model_final = final_models.get((model_key, target))
                if model_final is None:
                    continue

                best_t = best_by_t.get(target)
                if best_t is None:
                    continue

                thr_t = float(thr_up if target == "target_up20" else thr_dd)

                rep_valid = metrics_report(y_valid_full[target], p_valid_by_t[target], thr_t)
                print_report("VALID", target, model_key, rep_valid, thr_t)

                rep_test  = metrics_report(y_test_full[target], p_test_by_t[target], thr_t)
                print_report("TEST", target, model_key, rep_test, thr_t)

                # Store meta consistently
                meta["targets"].setdefault(target, {})
                params_safe = {}
                if isinstance(best_t.params, dict):
                    for k, v in best_t.params.items():
                        if isinstance(v, (np.floating,)):
                            params_safe[k] = float(v)
                        elif isinstance(v, (np.integer,)):
                            params_safe[k] = int(v)
                        else:
                            params_safe[k] = v

                meta["targets"][target][model_key] = {
                    "cv_ap_mean": float(getattr(best_t, "score_mean", np.nan)),
                    "cv_ap_std":  float(getattr(best_t, "score_std",  np.nan)),
                    "threshold":  float(thr_t),
                    "valid_report": report_to_dict(rep_valid),
                    "test_report":  report_to_dict(rep_test),
                    "lgbm_used_gpu": bool(getattr(best_t, "used_gpu", False)) if model_key == "lgbm" else False,
                    "params": params_safe,
                }

                # save artifacts (no-op if SAVE_MODELS=False)
                meta_safe = make_serializable(meta)
                model_train = train_models.get((model_key, target))
                if model_train is not None:
                    save_model_and_meta("train", model_key, target, model_train, meta_safe)

                save_model_and_meta("tv", model_key, target, model_final, meta_safe)


                # Metrics table row
                TOPKPERDAY = int(globals().get("TOPK_PER_DAY", 20))

                pK_m, pK_s   = precision_at_k_per_day(idx_valid, y_valid_full[target], p_valid_by_t[target], k=TOPKPERDAY)
                pK_mt, pK_st = precision_at_k_per_day(idx_test,  y_test_full[target],  p_test_by_t[target],  k=TOPKPERDAY)

                for split_name, rep, pkm, pks in [("VALID", rep_valid, pK_m, pK_s), ("TEST", rep_test, pK_mt, pK_st)]:
                    tn, fp, fn, tp = rep["Confusion"]
                    metrics_list.append({
                        "split": split_name,
                        "model": model_key,
                        "target": target,
                        "thr": float(thr_t),
                        "cv_ap_mean": float(getattr(best_t, "score_mean", np.nan)),
                        "cv_ap_std":  float(getattr(best_t, "score_std",  np.nan)),
                        "AP": rep["AP"],
                        "ROC_AUC": rep["ROC_AUC"],
                        "F1": rep["F1"],
                        "Precision": rep["Precision"],
                        "Recall": rep["Recall"],
                        "BalancedAcc": rep["BalancedAcc"],
                        "FPR": rep["FPR"],
                        "PosRate": rep["PosRate"],
                        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                        f"Prec@{TOPKPERDAY}_mean": pkm,
                        f"Prec@{TOPKPERDAY}_std":  pks,
                    })


    else:
        # LOAD path: align features, eval VALID/TEST, use saved threshold
        train_models = {k: v for k, v in models_loaded_train.items() if v is not None}
        final_models = {k: v for k, v in models_loaded_tv.items() if v is not None}

        meta = meta_loaded if meta_loaded is not None else meta
        saved_cols = meta["feature_cols"]

        def align_columns(df_like, cols_expected):
            for c in cols_expected:
                if c not in df_like.columns:
                    df_like[c] = 0.0
            extra = [c for c in df_like.columns if c not in cols_expected and c not in TARGETS and not c.startswith(("tk_", "ac_"))]
            if extra:
                df_like = df_like.drop(columns=extra)
            return df_like[cols_expected]



        HIST_SAVE = True
        HIST_PATH = os.path.join(output_dir, "ensemble_signals_history_with_trust.parquet")

        if HIST_SAVE:
            # 1) escolha quais linhas entram no histórico
            # - labeled-only: só onde tem target (bom pra auditoria de qualidade)
            # - all: inclui também datas “APPLY-like” (targets NaN)
            HIST_LABELED_ONLY = False
            mask_hist = mask_labeled if HIST_LABELED_ONLY else np.ones(len(LONG), dtype=bool)

            hist_df = LONG.loc[mask_hist].copy()
            hist_df_aln = align_columns(hist_df.copy(), saved_cols)

            hist_out = pd.DataFrame(index=hist_df.index)
            hist_out["ticker_rec"] = ticker_series.loc[mask_hist].values

            # 2) predições + trusts
            for target in TARGETS:
                # guarda y (se existir)
                hist_out[f"y_{target}"] = LONG.loc[mask_hist, target].astype("float64").values

                for model_key in RUN_MODELS:
                    tmeta = meta.get("targets", {}).get(target, {}).get(model_key, None)
                    model = final_models.get((model_key, target))
                    colp = f"proba_{target}__{model_key}"
                    coly = f"pred_{target}__{model_key}"

                    if (tmeta is None) or (model is None):
                        hist_out[colp] = 0.0
                        hist_out[coly] = 0
                        hist_out[f"trust_{target}__{model_key}"] = 0.0
                        hist_out[f"margin_{target}__{model_key}"] = 0.0
                        continue

                    thr = float(tmeta["threshold"])
                    p = predict_proba_model(model_key, model, hist_df_aln.values)

                    pred, trust_class, trust_margin = _trust_cols_from_proba(p, thr)

                    hist_out[colp] = p
                    hist_out[coly] = pred
                    hist_out[f"trust_{target}__{model_key}"]  = trust_class
                    hist_out[f"margin_{target}__{model_key}"] = trust_margin

            # 3) policy + trusts (opcional)
            for model_key in RUN_MODELS:
                col_up = f"pred_target_up20__{model_key}"
                col_dd = f"pred_target_dd5__{model_key}"
                if (col_up not in hist_out.columns) or (col_dd not in hist_out.columns):
                    continue

                buy  = (hist_out[col_up].astype(int).values == 1) & (hist_out[col_dd].astype(int).values == 0)
                sell = (hist_out[col_up].astype(int).values == 0) & (hist_out[col_dd].astype(int).values == 1)

                hist_out[f"buy__{model_key}"]  = buy.astype(int)
                hist_out[f"sell__{model_key}"] = sell.astype(int)
                hist_out[f"action__{model_key}"] = np.where(buy, "BUY", np.where(sell, "SELL", "HOLD"))

                p_up = pd.to_numeric(hist_out.get(f"proba_target_up20__{model_key}", 0.0), errors="coerce").fillna(0.0).clip(0, 1).to_numpy()
                p_dd = pd.to_numeric(hist_out.get(f"proba_target_dd5__{model_key}", 0.0), errors="coerce").fillna(0.0).clip(0, 1).to_numpy()

                buy_trust  = (p_up * (1.0 - p_dd)).astype(np.float32)
                sell_trust = ((1.0 - p_up) * p_dd).astype(np.float32)

                hist_out[f"buy_trust__{model_key}"]  = buy_trust
                hist_out[f"sell_trust__{model_key}"] = sell_trust
                hist_out[f"action_trust__{model_key}"] = np.where(buy, buy_trust, np.where(sell, sell_trust, 0.0)).astype(np.float32)

            # 4) salvar
            hist_out_reset = hist_out.reset_index().rename(columns={"index": "Date"})
            hist_out_reset.to_parquet(HIST_PATH, index=False)
            print(f"[OK] Saved history with trust → {HIST_PATH} | rows={len(hist_out_reset)}")



        valid_frame = align_columns(LONG.loc[mask_valid].copy(), saved_cols)
        test_frame  = align_columns(LONG.loc[mask_test].copy(),  saved_cols)

        X_valid_aln = valid_frame.values
        X_test_aln  = test_frame.values

        for target in TARGETS:
            y_va = LONG.loc[mask_valid, target].values.astype('int8')
            y_te = LONG.loc[mask_test,  target].values.astype('int8')

            for model_key in RUN_MODELS:
                model_obj = models_loaded.get((model_key, target))
                tmeta = meta.get("targets", {}).get(target, {}).get(model_key, None)
                if model_obj is None or tmeta is None:
                    print(f"[{model_key}/{target}] missing; train first.")
                    continue

                thr = float(tmeta["threshold"])
                p_va = predict_proba_model(model_key, model_obj, X_valid_aln)
                rep_va = metrics_report(y_va, p_va, thr)
                print_report("VALID", target, model_key, rep_va, thr)

                p_te = predict_proba_model(model_key, model_obj, X_test_aln)
                rep_te = metrics_report(y_te, p_te, thr)
                print_report("TEST", target, model_key, rep_te, thr)

                pK_m, pK_s = precision_at_k_per_day(idx_valid, y_va, p_va, k=TOPK_PER_DAY)
                pK_mt, pK_st = precision_at_k_per_day(idx_test, y_te, p_te, k=TOPK_PER_DAY)

                for split_name, rep, pkm, pks in [("VALID", rep_va, pK_m, pK_s), ("TEST", rep_te, pK_mt, pK_st)]:
                    tn, fp, fn, tp = rep["Confusion"]
                    metrics_list.append({
                        "split": split_name,
                        "model": model_key,
                        "target": target,
                        "thr": float(thr),
                        "cv_ap_mean": float(tmeta.get("cv_ap_mean", np.nan)),
                        "cv_ap_std": float(tmeta.get("cv_ap_std", np.nan)),
                        "AP": rep["AP"],
                        "ROC_AUC": rep["ROC_AUC"],
                        "F1": rep["F1"],
                        "Precision": rep["Precision"],
                        "Recall": rep["Recall"],
                        "BalancedAcc": rep["BalancedAcc"],
                        "FPR": rep["FPR"],
                        "PosRate": rep["PosRate"],
                        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                        f"Prec@{TOPK_PER_DAY}_mean": pkm,
                        f"Prec@{TOPK_PER_DAY}_std": pks,
                    })

        final_models = {k: v for k, v in models_loaded.items() if v is not None}


    # ============================================================
    # APPLY (last days) — aggregate outputs
    # ============================================================
    saved_cols = meta["feature_cols"]


    APPLY_INCLUDE_VALID_IN_OUTPUT = True

    mask_out = (mask_apply | mask_valid) if APPLY_INCLUDE_VALID_IN_OUTPUT else mask_apply
    apply_df = LONG.loc[mask_out].copy()
    apply_df_aln = align_columns(apply_df.copy(), saved_cols)

    apply_df_display = pd.DataFrame(index=apply_df.index)
    apply_df_display["ticker_rec"] = ticker_series.loc[mask_out].values
    apply_df_display["split"] = np.where(pd.to_datetime(apply_df.index).isin(pd.to_datetime(LONG.loc[mask_valid].index).unique()),
                                         "VALID", "APPLY")



    # predictions
    for target in TARGETS:
        ycol = f"y_{target}"
        apply_df_display[ycol] = LONG.loc[mask_out, target].astype("float64").values
        apply_df_display.loc[apply_df_display["split"] == "APPLY", ycol] = np.nan

        for model_key in RUN_MODELS:
            tmeta = meta.get("targets", {}).get(target, {}).get(model_key, None)
            model = final_models.get((model_key, target))
            colp = f"proba_{target}__{model_key}"
            coly = f"pred_{target}__{model_key}"

            if (tmeta is None) or (model is None):
                apply_df_display[colp] = 0.0
                apply_df_display[coly] = 0
                continue

            thr = float(tmeta["threshold"])
            p = predict_proba_model(model_key, model, apply_df_aln.values)

            # proba + pred
            apply_df_display[colp] = p
            apply_df_display[coly] = (np.asarray(p) >= thr).astype(int)

            # NEW: confiança no binário predito
            _, trust_class, trust_margin = _trust_cols_from_proba(
                p, thr, pred=apply_df_display[coly].values
            )
            apply_df_display[f"trust_{target}__{model_key}"]  = trust_class
            apply_df_display[f"margin_{target}__{model_key}"] = trust_margin

    # ------------------------------------------------------------
    # Derive policy actions per model: BUY / SELL / HOLD
    # BUY  when (up20==1) and (dd5==0)
    # SELL when (up20==0) and (dd5==1)
    # ------------------------------------------------------------
    for model_key in RUN_MODELS:
        col_up = f"pred_target_up20__{model_key}"
        col_dd = f"pred_target_dd5__{model_key}"

        if (col_up not in apply_df_display.columns) or (col_dd not in apply_df_display.columns):
            continue

        buy  = (apply_df_display[col_up].astype(int) == 1) & (apply_df_display[col_dd].astype(int) == 0)
        sell = (apply_df_display[col_up].astype(int) == 0) & (apply_df_display[col_dd].astype(int) == 1)

        apply_df_display[f"buy__{model_key}"]  = buy.astype(int)
        apply_df_display[f"sell__{model_key}"] = sell.astype(int)

        # action string (optional)
        act = np.where(buy, "BUY", np.where(sell, "SELL", "HOLD"))
        apply_df_display[f"action__{model_key}"] = act

    # build metrics DF
    METRICS_DF = pd.DataFrame(metrics_list).sort_values(["split", "target", "AP"], ascending=[True, True, False])

    print("\n=== Metrics comparison (VALID + TEST) ===")
    display(METRICS_DF)

    print(f"\n=== Aggregated APPLY outputs (last {APPLY_DAYS} days) — head ===")
    display(apply_df_display.head(50))

    # Optional: Top-K per day (example: LGBM on target_up20)
    apply_days = sorted(apply_df_display.index.unique())

    if "lgbm" in RUN_MODELS:
        print("\nTop-K BUY candidates by LGBM policy (up20 high AND dd5 low):")
        dfpol = apply_df_display.copy()
        dfpol = dfpol[dfpol["buy__lgbm"] == 1]
        if len(dfpol) == 0:
            display(dfpol.head(1))
        else:
            display(
                top_per_day(dfpol, "proba_target_up20__lgbm")[["date","ticker_rec","proba_target_up20__lgbm","proba_target_dd5__lgbm","buy__lgbm","action__lgbm"]]
            )

    # ============================================================
    # ENSEMBLES (execute AFTER your current training/load cell)
    #
    # Applied suggestions (1) to (4):
    #
    # 1) Fix leakage / selection discipline:
    #    - Selection + thresholding are ALWAYS done on VALID.
    #    - No more AUTO fallback to TEST for selection.
    #    - If TRAIN-only base models are missing, we still select on VALID,
    #      but we print a LOUD warning that VALID may be optimistic.
    #
    # 2) Add FINAL holdout (intocado) carved from the END of TEST:
    #    - We split your original mask_test into:
    #         TEST_EVAL (earlier part) + FINAL (last FINAL_DAYS unique dates)
    #    - We report metrics on VALID, TEST_EVAL, FINAL.
    #    - FINAL is the closest thing to a "clean" estimate here.
    #
    # 3) dd5 threshold + selection becomes recall-first (risk filter):
    #    - For target_dd5:
    #        * best ensemble is chosen by highest Recall on VALID (tie: Precision, F1, tp)
    #        * threshold is chosen with Recall-first constraint:
    #              Recall >= DD5_FILTER_MIN_RECALL (default 0.85)
    #          and among those, maximize Precision (tie: F1, tp).
    #
    # 4) Add trading-style evaluation (simple backtest) when possible:
    #    - If a price column exists (auto-detected), build a trade ledger per ticker:
    #        * enter on signal date, exit after HOLD_DAYS (default = HORIZON)
    #        * 1 open position per ticker at a time (no stacking)
    #        * net return subtracts roundtrip costs (bps)
    #    - If price not found, prints a clear warning and skips PnL.
    # ============================================================



    # -----------------------------
    # CONFIG
    # -----------------------------
    CALIBRATE_BASES = True
    CALIBRATION_MODE = "auto"  # "auto" | "isotonic" | "platt" | "none"

    # Purged CV inside VALID (for calibration + weight-search + stacking OOF)
    CV_TEST_LEN_DAYS = 15
    CV_STEP_DAYS     = 3
    CV_PURGE_DAYS    = int(globals().get("PURGE_DAYS", 30))
    MIN_CV_SPLITS    = 3

    # Calibration thresholds
    MIN_CAL_OOF_COVERAGE = 0.50

    # Ensemble weight search
    ENSEMBLE_WEIGHT_TRIALS = 250
    RECENCY_ALPHA = 3.0
    CARUANA_STEPS = 32

    # Selection: ALWAYS on VALID now (no TEST tuning)
    SELECT_SPLIT = "valid"
    trainmodels = {}   # models fit on TRAIN only (used for VALID scoring/calibration)
    finalmodels = {}   # models fit on TRAIN+VALID (used for TEST/FINAL/APPLY)

    # Up20 selection: Precision-focused with Recall constraint

    MIN_RECALL_UP_SELECT = _get_default_min_recall_select_up()

    # dd5 filter: Recall-first BUT with sane caps to avoid "predict dd5 everywhere"
    DD5_FILTER_MIN_RECALL  = float(globals().get("DD5_FILTER_MIN_RECALL", 0.80))
    DD5_FILTER_MAX_FPR     = float(globals().get("DD5_FILTER_MAX_FPR", 0.30))  # important: not 1.0
    DD5_POSRATE_SLACK      = float(globals().get("DD5_POSRATE_SLACK", 0.20))   # allow posrate up to base_rate + slack
    DD5_POSRATE_HARD_CAP   = float(globals().get("DD5_POSRATE_HARD_CAP", 0.70)) # never allow ~all-ones
    # -----------------------------
    # NEW: Tune dd5 threshold to reduce dd5 inside BUY candidates
    # -----------------------------
    TUNE_DD5_THR_ON_BUYBASE = bool(globals().get("TUNE_DD5_THR_ON_BUYBASE", True))

    # constraints for dd5 tuning on selection split (VALID or TEST_EVAL)
    DD5_BUY_MIN_SIGNALS_SEL = int(globals().get("DD5_BUY_MIN_SIGNALS_SEL", 80))      # minimum BUY_BASE signals on selection
    DD5_BUY_MAX_DD5RATE_SEL = float(globals().get("DD5_BUY_MAX_DD5RATE_SEL", 0.15))  # desired max dd5 rate within BUY_BASE on selection
    DD5_BUY_GRID_Q          = int(globals().get("DD5_BUY_GRID_Q", 181))              # how many quantile thresholds to scan

    # If True, stop the cell if TRAIN-only models are missing (strong anti-leak discipline)
    REQUIRE_TRAIN_ONLY_MODELS = bool(globals().get("REQUIRE_TRAIN_ONLY_MODELS", False))

    # GOOD event (for SCORE_BUY): up20==1 & dd5==0
    GOOD_POLICY = {
        "beta": 0.5,
        "min_precision": float(globals().get("GOOD_MIN_PRECISION", 0.40)),
        "min_recall":    float(globals().get("GOOD_MIN_RECALL", 0.01)),
        "max_fpr":       float(globals().get("GOOD_MAX_FPR", 1.00)),
    }
    LAMBDA_GRID = np.linspace(
        0.0,
        float(globals().get("GOOD_LAMBDA_MAX", 3.0)),
        int(globals().get("GOOD_LAMBDA_STEPS", 61))
    )

    # FINAL holdout carved from end of TEST
    FINAL_DAYS = int(globals().get("FINAL_DAYS", 180))  # number of UNIQUE dates from end of TEST
    FINAL_MIN_DAYS = int(globals().get("FINAL_MIN_DAYS", 60))  # ensure not too tiny
    FINAL_FALLBACK_FRAC = float(globals().get("FINAL_FALLBACK_FRAC", 0.33))  # if TEST too short

    # Trading-style evaluation
    HOLD_DAYS = int(globals().get("HORIZON", 90))  # default hold = horizon
    ROUNDTRIP_COST_BPS = float(globals().get("ROUNDTRIP_COST_BPS", 20.0))  # 20 bps roundtrip
    ALLOW_REENTRY_SAME_DAY = bool(globals().get("ALLOW_REENTRY_SAME_DAY", False))

    # Where to save artifacts (optional)
    SAVE_ENSEMBLE_ARTIFACTS = True
    MODEL_DIR = globals().get("MODEL_DIR", os.path.join(output_dir, "models"))
    MODEL_TAG = str(globals().get("MODEL_TAG", "model_tag"))
    ENSEMBLE_SAVE_DIR = os.path.join(MODEL_DIR, MODEL_TAG, "ensembles_valid_select_with_final_holdout")
    os.makedirs(ENSEMBLE_SAVE_DIR, exist_ok=True)

    # -----------------------------
    # Helpers
    # -----------------------------



    # If your training cell defined choose_threshold_precision_focused, reuse it.
    if "choose_threshold_precision_focused" in globals():
        choose_thr_pf = globals()["choose_threshold_precision_focused"]
    else:
        def choose_thr_pf(y_true, proba, beta=0.5, min_precision=0.50, min_recall=0.10, max_fpr=0.25):
          y = np.asarray(y_true).astype(np.int8)
          s = np.asarray(proba, dtype=np.float64)
          s = np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)

          if y.size == 0 or len(np.unique(y)) < 2:
              return 0.5

          P = int(y.sum())
          N = int(y.size - P)
          if P == 0:
              return 1.0  # no positives, pick high threshold
          if N == 0:
              return 0.0  # no negatives, pick low threshold

          # sort by score desc
          order = np.argsort(-s, kind="mergesort")
          s_sorted = s[order]
          y_sorted = y[order]

          tp_cum = np.cumsum(y_sorted)
          fp_cum = np.cumsum(1 - y_sorted)

          # evaluate only at the END of each unique score block
          change = np.where(np.diff(s_sorted) != 0)[0]
          end_idx = np.r_[change, y.size - 1]

          TP = tp_cum[end_idx].astype(np.float64)
          FP = fp_cum[end_idx].astype(np.float64)
          FN = P - TP
          TN = N - FP

          prec = TP / np.maximum(TP + FP, 1e-12)
          rec  = TP / np.maximum(P, 1e-12)
          fpr  = FP / np.maximum(FP + TN, 1e-12)

          b2 = beta * beta
          fbeta = (1 + b2) * (prec * rec) / np.maximum(b2 * prec + rec, 1e-12)

          thr = s_sorted[end_idx]

          def pick(mask):
              if not np.any(mask):
                  return None
              i = np.nanargmax(np.where(mask, fbeta, -np.inf))
              return float(thr[i])

          # same logic as your original: strict → relax fpr → relax precision → best fbeta
          t = pick((prec >= min_precision) & (rec >= min_recall) & (fpr <= max_fpr))
          if t is not None:
              return t
          t = pick((prec >= min_precision) & (rec >= min_recall) & (fpr <= 1.0))
          if t is not None:
              return t
          t = pick(rec >= min_recall)
          if t is not None:
              return t

          return float(thr[int(np.nanargmax(fbeta))])















    # -----------------------------
    # Ensemble mechanics
    # -----------------------------








    # -----------------------------
    # Recover / build aligned matrices
    # -----------------------------
    assert "LONG" in globals(), "Expected LONG from your training cell."
    assert "meta" in globals(), "Expected meta from your training cell."
    assert "final_models" in globals(), "Expected final_models from your training/load cell."
    assert "RUN_MODELS" in globals() and "TARGETS" in globals(), "Expected RUN_MODELS and TARGETS."

    saved_cols = list(meta["feature_cols"])


    mask_valid = globals().get("mask_valid", None)
    mask_test  = globals().get("mask_test", None)
    mask_apply = globals().get("mask_apply", None)
    assert mask_valid is not None and mask_test is not None and mask_apply is not None, "Missing masks (mask_valid/mask_test/mask_apply)."

    valid_rows_dates = pd.to_datetime(LONG.loc[mask_valid].index)
    test_rows_dates_all = pd.to_datetime(LONG.loc[mask_test].index)
    apply_rows_dates = pd.to_datetime(LONG.loc[mask_apply].index)

    # -------- FINAL holdout split from end of TEST (by unique dates) --------
    test_rows_dates_all = pd.to_datetime(LONG.loc[mask_test].index)
    test_unique = np.array(sorted(pd.Index(test_rows_dates_all).unique()))
    if len(test_unique) == 0:
        raise RuntimeError("mask_test produced 0 rows; cannot build FINAL holdout.")

    # NEW: ensure TEST_EVAL is not empty when possible
    MIN_TEST_EVAL_DAYS = int(globals().get("MIN_TEST_EVAL_DAYS", 20))
    MIN_FINAL_DAYS     = int(globals().get("MIN_FINAL_DAYS", 20))

    len_u = len(test_unique)

    # helper: convert any mask (bool or indices) to boolean mask over LONG

    mask_test_bool = _to_bool_mask(mask_test)

    if len_u <= 1:
        # cannot split
        final_days_eff = len_u
    elif len_u <= (MIN_TEST_EVAL_DAYS + MIN_FINAL_DAYS):
        # too short: split roughly half/half, but keep at least 1 day for TEST_EVAL
        final_days_eff = max(1, min(len_u - 1, len_u // 2))
    else:
        # prefer requested FINAL_DAYS but keep MIN_TEST_EVAL_DAYS available for TEST_EVAL
        final_days_eff = min(int(FINAL_DAYS), len_u - MIN_TEST_EVAL_DAYS)
        final_days_eff = max(MIN_FINAL_DAYS, final_days_eff)
        final_days_eff = min(final_days_eff, len_u - 1)  # ensure at least 1 day in TEST_EVAL

    final_dates = set(test_unique[-final_days_eff:])

    mask_final = mask_test_bool & pd.to_datetime(LONG.index).isin(final_dates)
    mask_test_eval = mask_test_bool & (~mask_final)

    test_rows_dates  = pd.to_datetime(LONG.loc[mask_test_eval].index)
    final_rows_dates = pd.to_datetime(LONG.loc[mask_final].index)

    print(f"[ens] TEST unique dates={len_u} | FINAL unique dates={len(pd.Index(final_rows_dates).unique())} "
          f"(requested {FINAL_DAYS}, eff {final_days_eff})")
    print(f"[ens] TEST_EVAL unique dates={len(pd.Index(test_rows_dates).unique())} (min wanted {MIN_TEST_EVAL_DAYS})")
    if len(test_rows_dates) == 0:
        print("[ens] ⚠️ TEST_EVAL ended up empty. Metrics/backtest for TEST will be skipped.")


    # Build X matrices
    valid_df_aln = align_columns(LONG.loc[mask_valid], saved_cols)
    test_df_aln  = align_columns(LONG.loc[mask_test_eval], saved_cols)
    final_df_aln = align_columns(LONG.loc[mask_final], saved_cols)
    apply_df_aln = align_columns(LONG.loc[mask_apply], saved_cols)

    X_valid = valid_df_aln.values
    X_test  = test_df_aln.values
    X_final = final_df_aln.values
    X_apply = apply_df_aln.values

    y_valid = {t: LONG.loc[mask_valid, t].values.astype("int8") for t in TARGETS}
    y_test  = {t: LONG.loc[mask_test_eval,  t].values.astype("int8") for t in TARGETS}
    y_final = {t: LONG.loc[mask_final, t].values.astype("int8") for t in TARGETS}

    valid_unique = np.array(sorted(pd.Index(valid_rows_dates).unique()))
    CV_SPLITS, CV_PURGE_USED = make_cv_splits_with_relaxed_purge(
        valid_unique,
        purge_days=CV_PURGE_DAYS,
        test_len_days=CV_TEST_LEN_DAYS,
        step_days=CV_STEP_DAYS,
        min_splits=MIN_CV_SPLITS
    )

    print(f"[ens] VALID rows={len(X_valid)} | TEST_EVAL rows={len(X_test)} | FINAL rows={len(X_final)} | APPLY rows={len(X_apply)}")
    print(f"[ens] TEST unique dates={len(test_unique)} | FINAL unique dates={len(pd.Index(final_rows_dates).unique())} (requested {FINAL_DAYS}, eff {final_days_eff})")
    print(f"[ens] Purged CV splits inside VALID: {len(CV_SPLITS)} | purge={CV_PURGE_USED} (requested {CV_PURGE_DAYS}) | test_len={CV_TEST_LEN_DAYS} | step={CV_STEP_DAYS}")
    if CV_PURGE_USED != CV_PURGE_DAYS:
        print("[ens] ⚠️ CV purge was auto-relaxed to improve split count (for more stable OOF/calibration/weight-search).")

    # -----------------------------
    # Ticker recovery (for outputs + trading)
    # -----------------------------


    ticker_valid = recover_ticker_series_for_mask(mask_valid)
    ticker_test  = recover_ticker_series_for_mask(mask_test_eval)
    ticker_final = recover_ticker_series_for_mask(mask_final)
    ticker_apply = recover_ticker_series_for_mask(mask_apply)

    # -----------------------------
    # Model source selection (avoid leakage on VALID when possible)
    # -----------------------------
    # -----------------------------
    # Model source selection (avoid leakage on VALID when possible)
    # -----------------------------

    TRAIN_MODELS_NAME, TRAIN_ONLY_MODELS = _find_train_only_model_dict()

    if TRAIN_ONLY_MODELS is None:
        print("[ens] ⚠️ TRAIN-only base models NOT found.")
        print("[ens]    VALID scoring/selection may be optimistic if final_models saw VALID.")
        print("[ens]    Strongly recommended: expose train_models[(mk, target)] = model_fit_on_train_only.")



        # Prefer TEST_EVAL for selection/thresholds if it exists; otherwise fallback to VALID.
        # Selection + thresholding ALWAYS on VALID (discipline).
        SELECT_SPLIT = "valid"
        print("[ens] ⚠️ Selection/thresholding kept on VALID (train-only missing -> VALID may be optimistic).")


        # IMPORTANT: if train-only is missing, VALID-based calibration is contaminated.
        # Disable calibration to avoid overconfident/saturated probabilities.
        print("[ens] ⚠️ Disabling calibration because TRAIN-only models are missing (avoid contaminated calibration).")
        CALIBRATE_BASES = False
        CALIBRATION_MODE = "none"

    else:
        print(f"[ens] Using TRAIN-only models for VALID scoring from: {TRAIN_MODELS_NAME}")
        SELECT_SPLIT = "valid"

    print("[ens] TRAIN_ONLY_MODELS:", "OK" if TRAIN_ONLY_MODELS else "MISSING")
    if TRAIN_ONLY_MODELS:
        print("[ens] train-only keys:", len(TRAIN_ONLY_MODELS))

    SEL_SPLIT_LABEL = "VALID" if SELECT_SPLIT == "valid" else "TEST"
    print(f"[ens] Selection split used for thresholds/selection: {SEL_SPLIT_LABEL}.")



    # -----------------------------
    # Collect base probas on VALID/TEST_EVAL/FINAL/APPLY
    # -----------------------------



    BASE = {split: {t: {} for t in TARGETS} for split in ["valid", "test", "final", "apply"]}

    print("\n[ens] Scoring base models on VALID/TEST_EVAL/FINAL/APPLY ...")
    for target in TARGETS:
        for mk in RUN_MODELS:
            mdl_va = get_model_for_split(mk, target, "valid")
            mdl_te = get_model_for_split(mk, target, "test")
            mdl_fi = get_model_for_split(mk, target, "final")
            mdl_ap = get_model_for_split(mk, target, "apply")

            if mdl_va is None or mdl_te is None or mdl_fi is None or mdl_ap is None:
                print(f"  - {mk}/{target}: missing -> skip")
                continue

            p_va = np.clip(predict_proba_model_local(mk, mdl_va, X_valid).astype("float64"), 1e-6, 1-1e-6)
            p_te = np.clip(predict_proba_model_local(mk, mdl_te, X_test ).astype("float64"), 1e-6, 1-1e-6)
            p_fi = np.clip(predict_proba_model_local(mk, mdl_fi, X_final).astype("float64"), 1e-6, 1-1e-6)
            p_ap = np.clip(predict_proba_model_local(mk, mdl_ap, X_apply).astype("float64"), 1e-6, 1-1e-6)

            BASE["valid"][target][mk] = p_va
            BASE["test"][target][mk]  = p_te
            BASE["final"][target][mk] = p_fi
            BASE["apply"][target][mk] = p_ap
            print(f"  - {mk}/{target}: ok")


    # -----------------------------
    # Calibration (auto isotonic -> platt -> disabled), trained using VALID only
    # -----------------------------
    CAL = {split: {t: {} for t in TARGETS} for split in ["valid", "test", "final", "apply"]}
    CALIBRATORS = {t: {} for t in TARGETS}
    CAL_MODE_USED = {t: {} for t in TARGETS}

    if (not CALIBRATE_BASES) or (CALIBRATION_MODE == "none"):
        CAL = BASE
        print("\n[ens] Calibration disabled; using raw base probas.")
    else:
        print(f"\n[ens] Calibrating base models (mode={CALIBRATION_MODE}) using VALID only ...")
        for target in TARGETS:
            y_va = y_valid[target]
            n_va = len(y_va)

            dyn_min_rows_iso = max(500, int(0.08 * n_va))
            dyn_min_pos_iso  = max(10,  int(0.005 * n_va))
            dyn_min_neg_iso  = max(10,  int(0.005 * n_va))

            dyn_min_rows_pl = max(300, int(0.04 * n_va))
            dyn_min_pos_pl  = max(5,  int(0.002 * n_va))
            dyn_min_neg_pl  = max(5,  int(0.002 * n_va))

            for mk, p_va in BASE["valid"][target].items():
                chosen_cal = None
                chosen_mode = "none"
                cov = 0.0
                oof = p_va

                want_iso = (CALIBRATION_MODE in ("auto", "isotonic"))
                want_pl  = (CALIBRATION_MODE in ("auto", "platt"))

                if want_iso:
                    oof_i, iso_full, cov_i = oof_isotonic(
                        y_va, p_va, valid_rows_dates, CV_SPLITS,
                        min_rows=dyn_min_rows_iso, min_pos=dyn_min_pos_iso, min_neg=dyn_min_neg_iso
                    )
                    if cov_i >= MIN_CAL_OOF_COVERAGE:
                        chosen_cal = iso_full
                        chosen_mode = "isotonic"
                        cov = cov_i
                        oof = oof_i

                if (chosen_mode == "none") and want_pl:
                    oof_p, pl_full, cov_p = oof_platt(
                        y_va, p_va, valid_rows_dates, CV_SPLITS,
                        min_rows=dyn_min_rows_pl, min_pos=dyn_min_pos_pl, min_neg=dyn_min_neg_pl
                    )
                    if cov_p >= MIN_CAL_OOF_COVERAGE:
                        chosen_cal = pl_full
                        chosen_mode = "platt"
                        cov = cov_p
                        oof = oof_p

                if chosen_mode == "none":
                    chosen_cal = None
                    chosen_mode = "disabled"
                    cov = float(np.nan)
                    oof = p_va

                p_te = apply_calibrator(chosen_cal, BASE["test"][target][mk])
                p_fi = apply_calibrator(chosen_cal, BASE["final"][target][mk])
                p_ap = apply_calibrator(chosen_cal, BASE["apply"][target][mk])

                CAL["valid"][target][mk] = np.clip(oof, 1e-6, 1-1e-6)
                CAL["test"][target][mk]  = np.clip(p_te, 1e-6, 1-1e-6)
                CAL["final"][target][mk] = np.clip(p_fi, 1e-6, 1-1e-6)
                CAL["apply"][target][mk] = np.clip(p_ap, 1e-6, 1-1e-6)

                CALIBRATORS[target][mk] = chosen_cal
                CAL_MODE_USED[target][mk] = {"mode": chosen_mode, "oof_coverage": float(cov) if np.isfinite(cov) else None}

                if chosen_mode in ("isotonic", "platt"):
                    print(f"  - {mk}/{target}: {chosen_mode} | OOF coverage={cov*100:.1f}%")
                else:
                    print(f"  - {mk}/{target}: calibration DISABLED (coverage too low)")

    # -----------------------------
    # Build ensembles per target
    # -----------------------------
    ENSEMBLES = {split: {t: {} for t in TARGETS} for split in ["valid", "test", "final", "apply"]}
    ENSEMBLE_INFO = {t: {} for t in TARGETS}
    ENSEMBLE_THRESHOLDS = {t: {} for t in TARGETS}   # thresholds from VALID
    ENSEMBLE_METRICS = []

    print("\n[ens] Building ensembles per target ...")
    for target in TARGETS:
        base_va = CAL["valid"][target]
        base_te = CAL["test"][target]
        base_fi = CAL["final"][target]
        base_ap = CAL["apply"][target]
        keys = list(base_va.keys())

        if len(keys) < 2:
            print(f"  - {target}: not enough base models for ensembling (have {len(keys)})")
            continue

        y_va = y_valid[target]
        y_te = y_test[target]
        y_fi = y_final[target]

        # 0) equal-weight baseline
        p_va_eq = equal_weight_avg(base_va)
        p_te_eq = equal_weight_avg(base_te, keys_order=keys)
        p_fi_eq = equal_weight_avg(base_fi, keys_order=keys)
        p_ap_eq = equal_weight_avg(base_ap, keys_order=keys)
        ENSEMBLES["valid"][target]["ens_equal"] = p_va_eq
        ENSEMBLES["test"][target]["ens_equal"]  = p_te_eq
        ENSEMBLES["final"][target]["ens_equal"] = p_fi_eq
        ENSEMBLES["apply"][target]["ens_equal"] = p_ap_eq
        ENSEMBLE_INFO[target]["ens_equal"] = {"type": "equal", "keys": keys}

        # 1) ens_weighted — AP optimized (purged CV)
        print(f"\n[{target}] 1) ens_weighted")
        w_lin, k_lin, cv_ap = cv_weight_search(
            y_va, base_va, valid_rows_dates, CV_SPLITS,
            trials=ENSEMBLE_WEIGHT_TRIALS, seed=42,
            sample_weight=None, logit_space=False
        )
        def _dot_for_split(base_dict, k_order, w):
            M = np.vstack([base_dict[k] for k in k_order]).T
            return M.dot(w)

        p_va_w = _dot_for_split(base_va, k_lin, w_lin)
        p_te_w = _dot_for_split(base_te, k_lin, w_lin)
        p_fi_w = _dot_for_split(base_fi, k_lin, w_lin)
        p_ap_w = _dot_for_split(base_ap, k_lin, w_lin)

        ENSEMBLES["valid"][target]["ens_weighted"] = p_va_w
        ENSEMBLES["test"][target]["ens_weighted"]  = p_te_w
        ENSEMBLES["final"][target]["ens_weighted"] = p_fi_w
        ENSEMBLES["apply"][target]["ens_weighted"] = p_ap_w
        ENSEMBLE_INFO[target]["ens_weighted"] = {"type": "lin_weighted", "keys": k_lin, "weights": w_lin.tolist(), "cv_ap": float(cv_ap)}

        # 2) ens_weighted_recency
        print(f"[{target}] 2) ens_weighted_recency")
        sw = recency_weights(valid_rows_dates, alpha=RECENCY_ALPHA)
        w_rec, k_rec, cv_ap_r = cv_weight_search(
            y_va, base_va, valid_rows_dates, CV_SPLITS,
            trials=ENSEMBLE_WEIGHT_TRIALS, seed=123,
            sample_weight=sw, logit_space=False
        )
        p_va_r = _dot_for_split(base_va, k_rec, w_rec)
        p_te_r = _dot_for_split(base_te, k_rec, w_rec)
        p_fi_r = _dot_for_split(base_fi, k_rec, w_rec)
        p_ap_r = _dot_for_split(base_ap, k_rec, w_rec)

        ENSEMBLES["valid"][target]["ens_weighted_recency"] = p_va_r
        ENSEMBLES["test"][target]["ens_weighted_recency"]  = p_te_r
        ENSEMBLES["final"][target]["ens_weighted_recency"] = p_fi_r
        ENSEMBLES["apply"][target]["ens_weighted_recency"] = p_ap_r
        ENSEMBLE_INFO[target]["ens_weighted_recency"] = {"type": "lin_weighted_recency", "keys": k_rec, "weights": w_rec.tolist(), "cv_ap": float(cv_ap_r)}

        # 3) ens_logit_weighted
        print(f"[{target}] 3) ens_logit_weighted")
        w_log, k_log, cv_ap_l = cv_weight_search(
            y_va, base_va, valid_rows_dates, CV_SPLITS,
            trials=ENSEMBLE_WEIGHT_TRIALS, seed=7,
            sample_weight=None, logit_space=True
        )
        base_va_sub = {k: base_va[k] for k in k_log}
        base_te_sub = {k: base_te[k] for k in k_log}
        base_fi_sub = {k: base_fi[k] for k in k_log}
        base_ap_sub = {k: base_ap[k] for k in k_log}

        p_va_log = logit_blend(base_va_sub, w_log, keys_order=k_log)
        p_te_log = logit_blend(base_te_sub, w_log, keys_order=k_log)
        p_fi_log = logit_blend(base_fi_sub, w_log, keys_order=k_log)
        p_ap_log = logit_blend(base_ap_sub, w_log, keys_order=k_log)

        ENSEMBLES["valid"][target]["ens_logit_weighted"] = p_va_log
        ENSEMBLES["test"][target]["ens_logit_weighted"]  = p_te_log
        ENSEMBLES["final"][target]["ens_logit_weighted"] = p_fi_log
        ENSEMBLES["apply"][target]["ens_logit_weighted"] = p_ap_log
        ENSEMBLE_INFO[target]["ens_logit_weighted"] = {"type": "logit_weighted", "keys": k_log, "weights": w_log.tolist(), "cv_ap": float(cv_ap_l)}

        # 4) ens_trimmed_mean
        print(f"[{target}] 4) ens_trimmed_mean")
        p_va_tm = trimmed_mean_blend(base_va, trim_k=1, keys_order=keys)
        p_te_tm = trimmed_mean_blend(base_te, trim_k=1, keys_order=keys)
        p_fi_tm = trimmed_mean_blend(base_fi, trim_k=1, keys_order=keys)
        p_ap_tm = trimmed_mean_blend(base_ap, trim_k=1, keys_order=keys)

        ENSEMBLES["valid"][target]["ens_trimmed_mean"] = p_va_tm
        ENSEMBLES["test"][target]["ens_trimmed_mean"]  = p_te_tm
        ENSEMBLES["final"][target]["ens_trimmed_mean"] = p_fi_tm
        ENSEMBLES["apply"][target]["ens_trimmed_mean"] = p_ap_tm
        ENSEMBLE_INFO[target]["ens_trimmed_mean"] = {"type": "trimmed_mean", "keys": keys, "trim_k": 1}

        # 5) ens_selection — Caruana
        print(f"[{target}] 5) ens_selection")
        w_sel, k_sel, cv_ap_s = ensemble_selection_forward(
            y_va, base_va, valid_rows_dates, CV_SPLITS,
            steps=CARUANA_STEPS, tol=1e-4
        )
        p_va_s = _dot_for_split(base_va, k_sel, w_sel)
        p_te_s = _dot_for_split(base_te, k_sel, w_sel)
        p_fi_s = _dot_for_split(base_fi, k_sel, w_sel)
        p_ap_s = _dot_for_split(base_ap, k_sel, w_sel)

        ENSEMBLES["valid"][target]["ens_selection"] = p_va_s
        ENSEMBLES["test"][target]["ens_selection"]  = p_te_s
        ENSEMBLES["final"][target]["ens_selection"] = p_fi_s
        ENSEMBLES["apply"][target]["ens_selection"] = p_ap_s
        ENSEMBLE_INFO[target]["ens_selection"] = {"type": "caruana_selection", "keys": k_sel, "weights": w_sel.tolist(), "cv_ap": float(cv_ap_s)}

        # 6) ens_stackLR — stacked logistic regression on meta-features
        print(f"[{target}] 6) ens_stackLR")
        n_va = len(y_va)
        dyn_min_rows = max(500, int(0.08 * n_va))
        dyn_min_pos  = max(10,  int(0.005 * n_va))
        dyn_min_neg  = max(10,  int(0.005 * n_va))

        oof_lr, (scLR, clfLR), k_lr, cov_lr = purged_stacked_lr(
            y_va, base_va, valid_rows_dates, CV_SPLITS,
            min_rows=dyn_min_rows, min_pos=dyn_min_pos, min_neg=dyn_min_neg
        )
        Xmeta_te, _ = build_meta_matrix(base_te, keys_order=k_lr)
        Xmeta_fi, _ = build_meta_matrix(base_fi, keys_order=k_lr)
        Xmeta_ap, _ = build_meta_matrix(base_ap, keys_order=k_lr)



        if Xmeta_te.shape[0] == 0:
            p_te_lr = np.array([], dtype=np.float64)
        else:
            p_te_lr = clfLR.predict_proba(scLR.transform(Xmeta_te))[:, 1]

        if Xmeta_fi.shape[0] == 0:
            p_fi_lr = np.array([], dtype=np.float64)
        else:
            p_fi_lr = clfLR.predict_proba(scLR.transform(Xmeta_fi))[:, 1]

        if Xmeta_ap.shape[0] == 0:
            p_ap_lr = np.array([], dtype=np.float64)
        else:
            p_ap_lr = clfLR.predict_proba(scLR.transform(Xmeta_ap))[:, 1]


        ENSEMBLES["valid"][target]["ens_stackLR"] = oof_lr
        ENSEMBLES["test"][target]["ens_stackLR"]  = p_te_lr
        ENSEMBLES["final"][target]["ens_stackLR"] = p_fi_lr
        ENSEMBLES["apply"][target]["ens_stackLR"] = p_ap_lr
        ENSEMBLE_INFO[target]["ens_stackLR"] = {"type": "stackLR", "keys": k_lr, "oof_coverage": float(cov_lr), "scaler": scLR, "clf": clfLR}

        # -----------------------------
        # Thresholds learned on VALID + metrics on VALID/TEST_EVAL/FINAL
        # -----------------------------
        for ens_name in list(ENSEMBLES["valid"][target].keys()):
            p_va = ENSEMBLES["valid"][target][ens_name]
            p_te = ENSEMBLES["test"][target][ens_name]
            p_fi = ENSEMBLES["final"][target][ens_name]

            # learn threshold on selected split (VALID if train-only exists; otherwise TEST_EVAL)
            p_sel = ENSEMBLES[SELECT_SPLIT][target][ens_name]
            y_sel = (y_valid[target] if SELECT_SPLIT == "valid" else y_test[target]).astype(np.int8)

            if target == "target_dd5":
                # dd5 threshold deve ser definido pelo próprio target_dd5 (sem depender de buy_base aqui)
                base_rate_sel = float(np.mean(y_sel)) if len(y_sel) else 0.0

                thr = choose_thr_dd5_filter(
                    y_true=y_sel,
                    proba=p_sel,
                    base_rate=base_rate_sel,
                    min_recall=DD5_FILTER_MIN_RECALL,
                    max_fpr=DD5_FILTER_MAX_FPR,
                    pos_slack=DD5_POSRATE_SLACK,
                    hard_cap=DD5_POSRATE_HARD_CAP,
                )

                pred_tmp = (p_sel >= thr).astype(np.int8)
                tn, fp, fn, tp, fpr, _, posr = confusion_stats(y_sel, pred_tmp)
                print(f"[dd5-thr] thr={thr:.6f} | base_rate={base_rate_sel:.3f} | posrate={posr:.3f} | fpr={fpr:.3f} | tn={tn}")

            elif target == "target_up20":
                # threshold do up20 é independente; NÃO use thr_up (global) aqui
                thr = choose_thr_for_target_up20(y_sel, p_sel)

            else:
                thr = choose_thr_pf(y_sel, p_sel, beta=0.5, min_precision=0.5, min_recall=0.1, max_fpr=0.25)
            ENSEMBLE_THRESHOLDS[target][ens_name] = float(thr)

            rep_va = metrics_report(y_va, p_va, thr)
            rep_te = metrics_report(y_te, p_te, thr) if len(y_te) else {k: np.nan for k in ["AP","ROC_AUC","Precision","Recall","F1","FPR","PosRate","Confusion"]}
            rep_fi = metrics_report(y_fi, p_fi, thr) if len(y_fi) else {k: np.nan for k in ["AP","ROC_AUC","Precision","Recall","F1","FPR","PosRate","Confusion"]}

            for split_name, rep in [("VALID", rep_va), ("TEST", rep_te), ("FINAL", rep_fi)]:
                ENSEMBLE_METRICS.append({
                    "target": target,
                    "model": ens_name,
                    "thr": float(thr),
                    "split": split_name,
                    "AP": rep["AP"],
                    "ROC_AUC": rep["ROC_AUC"],
                    "Precision": rep["Precision"],
                    "Recall": rep["Recall"],
                    "F1": rep["F1"],
                    "FPR": rep["FPR"],
                    "PosRate": rep["PosRate"],
                    "tn": rep["Confusion"][0], "fp": rep["Confusion"][1],
                    "fn": rep["Confusion"][2], "tp": rep["Confusion"][3],
                })

    # -----------------------------
    # Metrics table (VALID + TEST_EVAL + FINAL)
    # -----------------------------
    ENS_METRICS_DF = pd.DataFrame(ENSEMBLE_METRICS)
    ENS_METRICS_DF = ENS_METRICS_DF.sort_values(["split", "target", "Precision"], ascending=[True, True, False]).reset_index(drop=True)

    print(f"\n=== Ensemble metrics (thresholds learned on {SEL_SPLIT_LABEL}) ===")
    display(ENS_METRICS_DF)




    # =========================
    # SETTINGS (saídas)
    # =========================
    APPLY_SIGNALS_PATH = os.path.join(output_dir, "apply_ensemble_signals.csv")
    APPLY_DEBUG_PATH   = os.path.join(output_dir, "apply_ensemble_signals_debug.csv")
    HIST_PATH          = os.path.join(output_dir, "ensemble_signals_history.parquet")

    # Se True, salva histórico só onde existe target (labeled). Se False, salva tudo (bem maior).
    HIST_SAVE = True
    HIST_LABELED_ONLY = True

    # =========================
    # Helpers seleção best ensemble
    # =========================


    # =========================
    # 1) Escolher BEST ensembles no split de seleção
    # =========================
    BEST_ON_SEL = {}

    print(f"\n=== Best ensemble per target (selection on {SEL_SPLIT_LABEL}) ===")
    for target in TARGETS:
        df_sel = ENS_METRICS_DF[
            (ENS_METRICS_DF["split"] == SEL_SPLIT_LABEL) &
            (ENS_METRICS_DF["target"] == target)
        ].copy()

        y_sel_caps = (y_valid[target] if SELECT_SPLIT == "valid" else y_test[target])
        row = _pick_best_ensemble_row(df_sel, target, y_sel_for_caps=y_sel_caps)

        if row is None:
            print(f"[best] {target}: no candidates.")
            continue

        ens_name = str(row["model"])
        thr = float(ENSEMBLE_THRESHOLDS[target].get(ens_name, row["thr"]))

        BEST_ON_SEL[target] = {
            "ensemble": ens_name,
            "thr": thr,
            "sel_split": SEL_SPLIT_LABEL,
            "sel_metrics": {
                "Precision": float(row["Precision"]),
                "Recall": float(row["Recall"]),
                "F1": float(row["F1"]),
                "AP": float(row["AP"]),
                "FPR": float(row["FPR"]),
                "PosRate": float(row["PosRate"]),
                "tn": int(row["tn"]), "fp": int(row["fp"]), "fn": int(row["fn"]), "tp": int(row["tp"]),
            }
        }

        print(f"[best] {target}: {ens_name} | thr={thr:.6f} | "
              f"Prec={row['Precision']:.3f} Rec={row['Recall']:.3f} FPR={row['FPR']:.3f} PosRate={row['PosRate']:.3f}")


    assert "target_up20" in BEST_ON_SEL and "target_dd5" in BEST_ON_SEL, "BEST ensembles not found for both targets."

    BEST_UP = BEST_ON_SEL["target_up20"]["ensemble"]
    BEST_DD = BEST_ON_SEL["target_dd5"]["ensemble"]
    THR_UP  = float(BEST_ON_SEL["target_up20"]["thr"])
    THR_DD  = float(BEST_ON_SEL["target_dd5"]["thr"])

    print(f"\n[best-summary] up20={BEST_UP} thr_up20={THR_UP:.6f} | dd5={BEST_DD} thr_dd5={THR_DD:.6f}")


    # =========================
    # 2) Função para montar DataFrame de um split (VALID/TEST/FINAL/APPLY)
    # =========================


    # =========================
    # 3) Pegar probas do melhor ensemble por split
    # =========================
    p_valid_up = ENSEMBLES["valid"]["target_up20"][BEST_UP]
    p_valid_dd = ENSEMBLES["valid"]["target_dd5"][BEST_DD]

    p_test_up  = ENSEMBLES["test"]["target_up20"][BEST_UP]   if len(X_test)  else np.array([], dtype=np.float64)
    p_test_dd  = ENSEMBLES["test"]["target_dd5"][BEST_DD]    if len(X_test)  else np.array([], dtype=np.float64)

    p_final_up = ENSEMBLES["final"]["target_up20"][BEST_UP]  if len(X_final) else np.array([], dtype=np.float64)
    p_final_dd = ENSEMBLES["final"]["target_dd5"][BEST_DD]   if len(X_final) else np.array([], dtype=np.float64)

    p_apply_up = ENSEMBLES["apply"]["target_up20"][BEST_UP]
    p_apply_dd = ENSEMBLES["apply"]["target_dd5"][BEST_DD]

    # =========================
    # 4) Montar outputs por split
    # =========================
    out_valid = _build_split_output("valid", valid_rows_dates, ticker_valid,
                                   y_up=y_valid["target_up20"], y_dd=y_valid["target_dd5"],
                                   p_up=p_valid_up, p_dd=p_valid_dd,
                                   thr_up=THR_UP, thr_dd=THR_DD)

    out_test = _build_split_output("test", test_rows_dates, ticker_test,
                                  y_up=(y_test["target_up20"] if len(X_test) else None),
                                  y_dd=(y_test["target_dd5"]  if len(X_test) else None),
                                  p_up=p_test_up, p_dd=p_test_dd,
                                  thr_up=THR_UP, thr_dd=THR_DD) if len(X_test) else pd.DataFrame()

    out_final = _build_split_output("final", final_rows_dates, ticker_final,
                                   y_up=(y_final["target_up20"] if len(X_final) else None),
                                   y_dd=(y_final["target_dd5"]  if len(X_final) else None),
                                   p_up=p_final_up, p_dd=p_final_dd,
                                   thr_up=THR_UP, thr_dd=THR_DD) if len(X_final) else pd.DataFrame()

    out_apply = _build_split_output("apply", apply_rows_dates, ticker_apply,
                                   y_up=None, y_dd=None,
                                   p_up=p_apply_up, p_dd=p_apply_dd,
                                   thr_up=THR_UP, thr_dd=THR_DD)

    print("\n[ens] Output sizes:",
          f"VALID={len(out_valid)} TEST_EVAL={len(out_test)} FINAL={len(out_final)} APPLY={len(out_apply)}")


    # =========================
    # 5) Salvar APPLY (somente APPLY) + DEBUG (VALID+APPLY)
    # =========================
    os.makedirs(os.path.dirname(APPLY_SIGNALS_PATH), exist_ok=True)

    apply_only = out_apply.sort_values(["Date","ticker"]).reset_index(drop=True)
    apply_only.to_csv(APPLY_SIGNALS_PATH, index=False)
    print(f"[OK] Saved APPLY signals → {APPLY_SIGNALS_PATH} | rows={len(apply_only)}")

    debug = pd.concat([out_valid, out_apply], axis=0).sort_values(["Date","ticker"]).reset_index(drop=True)
    debug.to_csv(APPLY_DEBUG_PATH, index=False)
    print(f"[OK] Saved DEBUG (VALID+APPLY) → {APPLY_DEBUG_PATH} | rows={len(debug)}")


    KEEP_COLS = ["index", "Date", "ticker", "split", "p_up20", "p_dd5"]



    # =========================
    # 6) (Opcional) Salvar HISTÓRICO
    # =========================
    if HIST_SAVE:
        # ATENÇÃO: este "histórico" usa modelos finais (treinados com TRAIN+VALID),
        # portanto NÃO é um backtest walk-forward. Serve para auditoria/plots.
        if HIST_LABELED_ONLY:
            mask_hist = mask_labeled.copy()
        else:
            mask_hist = np.ones(len(LONG), dtype=bool)

        hist_idx = pd.to_datetime(LONG.loc[mask_hist].index)
        hist_tk  = recover_ticker_series_for_mask(mask_hist)

        # Para histórico, vamos usar diretamente as saídas já montadas quando possível:
        # - se você quer "todo histórico", a forma correta é recomputar ensembles em LONG (caro).
        # Aqui: salvamos pelo menos os splits disponíveis + APPLY; e, se HIST_LABELED_ONLY,
        # você já tem cobertura “boa” via VALID/TEST_EVAL/FINAL (labeled).
        pieces = [out_valid]
        if len(out_test):  pieces.append(out_test)
        if len(out_final): pieces.append(out_final)

        # NOTE: APPLY não é labeled, então só entra se HIST_LABELED_ONLY=False
        if not HIST_LABELED_ONLY:
            pieces.append(out_apply)

        hist_out = pd.concat(pieces, axis=0).sort_values(["Date","ticker"]).reset_index(drop=True)

        hist_out=save_hist_out_minimal(hist_out, HIST_PATH)
        hist_out.to_parquet(HIST_PATH, index=False)
        print(f"[OK] Saved ensemble history → {HIST_PATH} | rows={len(hist_out)}")


    # =========================================
    # 1) NO FINAL: quantos tickers únicos no output final (apply_df_display)
    # =========================================
    if "apply_df_display" in globals() and "ticker_rec" in apply_df_display.columns:
        n_tickers_apply_out = apply_df_display["ticker_rec"].astype(str).nunique()
        print(f"[FINAL] Unique tickers in apply_df_display: {n_tickers_apply_out}")
    else:
        print("[FINAL] apply_df_display or ticker_rec not found.")

    # =========================================
    # 2) (Opcional) Quantos tickers únicos no dataset LONG inteiro
    #    (preferindo ticker_series, que é o original)
    # =========================================
    if "ticker_series" in globals():
        n_tickers_long = pd.Series(ticker_series).astype(str).nunique()
        print(f"[FINAL] Unique tickers in LONG (ticker_series): {n_tickers_long}")
    else:
        # fallback: tenta achar colunas tk_ (dummies)
        tk_cols = [c for c in LONG.columns if str(c).startswith("tk_")]
        if tk_cols:
            tickers_from_dummies = LONG[tk_cols].idxmax(axis=1).astype(str).str.replace("tk_", "", regex=False)
            print(f"[FINAL] Unique tickers in LONG (from tk_ dummies): {tickers_from_dummies.nunique()}")
        else:
            print("[FINAL] ticker_series not found and no tk_ dummy columns found.")

    # =========================================
    # 3) (Opcional) Quantos tickers únicos por split (VALID/APPLY) dentro do apply_df_display
    # =========================================
    if "apply_df_display" in globals() and {"ticker_rec", "split"}.issubset(apply_df_display.columns):
        counts_by_split = apply_df_display.groupby("split")["ticker_rec"].nunique()
        print("[FINAL] Unique tickers by split:")
        print(counts_by_split.to_string())


    n_apply_only = apply_df_display.loc[apply_df_display["split"] == "APPLY", "ticker_rec"].nunique()
    print(f"[FINAL] Unique tickers in APPLY only: {n_apply_only}")


    PATH = os.path.join(output_dir, "expanded_stock_reduced.parquet")

    df = pd.read_parquet(PATH, engine="pyarrow")

    print("FILE:", PATH)
    print("Size (MB):", round(os.path.getsize(PATH)/1e6, 2))
    print("Shape:", df.shape)
    print("Index:", df.index.min(), "→", df.index.max())

    print("Columns is MultiIndex?", isinstance(df.columns, pd.MultiIndex), "nlevels=", getattr(df.columns, "nlevels", None))
    print("Column level names:", getattr(df.columns, "names", None))

    lvl0 = df.columns.get_level_values(0).astype(str)
    lvl1 = df.columns.get_level_values(1).astype(str)

    all_tickers = pd.Index(lvl1).unique()
    print("Unique tickers overall:", len(all_tickers))


    targets = ["target_up20", "target_dd5", "target_best_entry", "target_best_sale"]
    for t in targets:
        tks = tickers_for(t)
        print(f"{t}: {len(tks)} tickers")

    tku = set(tickers_for("target_up20"))
    tkd = set(tickers_for("target_dd5"))
    print("Up-only:", len(tku - tkd))
    print("DD-only:", len(tkd - tku))
    print("Example Up-only:", sorted(list(tku - tkd))[:20])
    print("Example DD-only:", sorted(list(tkd - tku))[:20])

    # Checa se tem tickers com espaços/variações esquisitas (causa clássica de interseção cair)
    raw = pd.Index(df.columns.get_level_values(1).astype(str))
    stripped = raw.str.strip()
    if (raw != stripped).any():
        bad = raw[raw != stripped].unique().tolist()
        print("Tickers com whitespace:", bad[:30])

    # Confirmação final via y "puro" (deveria dar 135)
    if "target_up20" in set(lvl0):
        y_up = df.loc[:, pd.IndexSlice["target_up20", :]]
        print("Tickers em y_up20:", len(pd.Index(y_up.columns.get_level_values(1).astype(str)).unique()))


    hist_out.head()

    # Collect paths of saved files
    saved_files = {
        "model_dir": MODEL_DIR,
        "parquet_path": PARQUET_PATH,
    }
    try:
        saved_files["apply_signals"] = APPLY_SIGNALS_PATH
    except NameError:
        pass
    try:
        saved_files["apply_debug"] = APPLY_DEBUG_PATH
    except NameError:
        pass
    try:
        saved_files["history"] = HIST_PATH
    except NameError:
        pass
    try:
        saved_files["ensemble_save_dir"] = ENSEMBLE_SAVE_DIR
    except NameError:
        pass
    return saved_files


if __name__ == "__main__":
    run_bin_stock()
