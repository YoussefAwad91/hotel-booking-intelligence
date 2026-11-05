#!/usr/bin/env python3
"""
Leakage-safe preprocessing for the hotel booking dataset (target-agnostic).

Every statistic (category lists, frequency maps, min/max) is learned on the
TRAIN file only and re-applied unchanged to validation and test.

Example:
    python preprocess.py --train data/train.csv --val data/val.csv --test data/test.csv \
        --out_dir processed

    # custom output locations / optional log transform / optional cyclical month+week
    python preprocess.py --train a.csv --val b.csv --test c.csv \
        --train_out out/tr.csv --val_out out/va.csv --test_out out/te.csv \
        --log_cols LeadTime ADR --cyclical --parquet

See preprocessing_summary.md for the full specification.
"""
import argparse
import calendar
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ONEHOT_COLS = [
    "Meal", "MarketSegment", "DistributionChannel", "ReservedRoomType",
    "AssignedRoomType", "DepositType", "CustomerType", "ReservationStatus",
]
FREQ_COLS = ["Country", "Agent", "Company"]      # frequency encoding only
MONTH_COL = "ArrivalDateMonth"                   # name -> 1..12
STATUS_DATE_COL = "ReservationStatusDate"        # dd/mm/yy -> days since 1970-01-01
CHILDREN_COL = "Children"                        # only column with true missing values -> 0
CYCLICAL_PERIODS = {"ArrivalDateMonth": 12, "ArrivalDateWeekNumber": 53}  # only used with --cyclical

UNDEFINED = "Undefined"                          # label for missing categorical values
# NOTE: "NA" is deliberately NOT here: it is the ISO code for Namibia in Country.
CAT_MISSING_TOKENS = {"", "nan", "NaN", "NULL", "null", "None"}
EPOCH = pd.Timestamp("1970-01-01")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _clean_cat(s: pd.Series) -> pd.Series:
    """Categorical -> clean string. '224.0' -> '224'. Missing tokens -> 'Undefined'."""
    s = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return s.where(~s.isin(CAT_MISSING_TOKENS), UNDEFINED)


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse dd/mm/yy (fallbacks: dd/mm/yyyy, yyyy-mm-dd). Raises if anything is unparsable."""
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        m = out.isna()
        if m.any():
            out[m] = pd.to_datetime(s[m], format=fmt, errors="coerce")
    if out.isna().any():
        raise ValueError(f"{int(out.isna().sum())} unparsable values in date column")
    return out


def _signed_log1p(x: pd.Series) -> pd.Series:
    """sign(x) * log(1 + |x|): monotonic, defined for zero and negatives (e.g. ADR)."""
    return np.sign(x) * np.log1p(np.abs(x))


# ----------------------------------------------------------------------------
# Preprocessor
# ----------------------------------------------------------------------------
class HotelPreprocessor:
    def __init__(self, log_cols=(), cyclical=False):
        self.log_cols = list(log_cols)    # numeric columns to signed-log1p BEFORE min-max (default: none)
        self.cyclical = cyclical          # add sin/cos of month & week number (default: off)
        self.fitted = False

    # ---- cleaning: raw frame -> dict of clean Series (strings or float64) ----
    def _clean(self, raw: pd.DataFrame) -> dict:
        raw = raw.copy()
        raw.columns = [c.strip() for c in raw.columns]
        cols = getattr(self, "input_cols_", list(raw.columns))
        missing = [c for c in cols if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")
        out, month_map = {}, {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
        for c in cols:
            s = raw[c].reset_index(drop=True)
            if c in ONEHOT_COLS or c in FREQ_COLS:
                out[c] = _clean_cat(s)
            elif c == MONTH_COL:
                n = pd.to_numeric(s, errors="coerce")
                n = n.fillna(s.astype(str).str.strip().str.lower().map(month_map))
                out[c] = n.astype("float64")
            elif c == STATUS_DATE_COL:
                out[c] = ((_parse_dates(s) - EPOCH).dt.days).astype("float64")
            else:
                n = pd.to_numeric(s, errors="coerce").astype("float64")
                if c == CHILDREN_COL:
                    n = n.fillna(0.0)             # EDA: only missing data; NA -> 0
                out[c] = n
        # Safety net: EDA says nothing else is missing, so fail loudly if that is wrong.
        bad = {c: int(v.isna().sum()) for c, v in out.items()
               if v.dtype == "float64" and v.isna().any()}
        if bad:
            raise ValueError(f"Unexpected missing/non-numeric values (column: count): {bad}")
        return out

    # ---- encoding: clean dict -> feature frame (original column order kept) ----
    def _build(self, clean: dict, fit: bool) -> pd.DataFrame:
        if fit:
            self.onehot_, self.freq_, self.scale_cols_ = {}, {}, []
        parts, known = {}, {}
        for col in self.input_cols_:
            s = clean[col]
            if col in ONEHOT_COLS:
                if fit:
                    self.onehot_[col] = sorted(s.unique())
                for k in self.onehot_[col]:
                    parts[f"{col}_{k}"] = (s == k).astype("float64")
                known[col] = set(self.onehot_[col])
            elif col in FREQ_COLS:
                if fit:
                    self.freq_[col] = s.value_counts(normalize=True).to_dict()
                    self.scale_cols_.append(f"{col}_freq")
                parts[f"{col}_freq"] = s.map(self.freq_[col]).fillna(0.0).astype("float64")
                known[col] = set(self.freq_[col])
            else:
                x = s
                if col in self.log_cols:
                    x = _signed_log1p(s)
                parts[col] = x.astype("float64")
                if fit:
                    self.scale_cols_.append(col)
                if self.cyclical and col in CYCLICAL_PERIODS:
                    ang = 2 * np.pi * s / CYCLICAL_PERIODS[col]
                    parts[f"{col}_sin"] = ((np.sin(ang) + 1) / 2).astype("float64")  # already [0,1]
                    parts[f"{col}_cos"] = ((np.cos(ang) + 1) / 2).astype("float64")
        if not fit:
            self.report_["unseen_rows"] = {c: int((~clean[c].isin(k)).sum())
                                           for c, k in known.items()
                                           if (~clean[c].isin(k)).any()}
        return pd.DataFrame(parts)

    # ---- min-max to [0, 1] with TRAIN statistics ----
    def _fit_scaler(self, X: pd.DataFrame):
        # frequency columns use lo = 0 so an unseen category (frequency 0) maps exactly to 0
        self.lo_ = {c: 0.0 if c.endswith("_freq") else float(X[c].min()) for c in self.scale_cols_}
        self.hi_ = {c: float(X[c].max()) for c in self.scale_cols_}

    def _scale(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = X.copy()
        oor = {}
        for c in self.scale_cols_:
            lo, hi = self.lo_[c], self.hi_[c]
            X[c] = 0.0 if hi == lo else (X[c] - lo) / (hi - lo)   # NO clipping
            if not fit:
                n = int(((X[c] < 0) | (X[c] > 1)).sum())
                if n:
                    oor[c] = (n, float(X[c].min()), float(X[c].max()))
        if not fit:
            self.report_["out_of_range"] = oor
        return X

    # ---- public API ----
    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        self.input_cols_ = [c.strip() for c in train.columns]
        unknown_log = [c for c in self.log_cols if c not in self.input_cols_]
        if unknown_log:
            raise ValueError(f"log_cols not in data: {unknown_log}")
        clean = self._clean(train)
        self.skew_ = {c: float(v.skew()) for c, v in clean.items()
                      if v.dtype == "float64" and v.nunique() > 2}
        X = self._build(clean, fit=True)
        self._fit_scaler(X)
        self.feature_names_ = list(X.columns)
        self.fitted = True
        return self._scale(X, fit=True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted, "call fit_transform on TRAIN first"
        self.report_ = {}
        clean = self._clean(df)
        X = self._build(clean, fit=False)[self.feature_names_]
        return self._scale(X, fit=False)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="input train csv")
    ap.add_argument("--val", required=True, help="input validation csv")
    ap.add_argument("--test", required=True, help="input test csv")
    ap.add_argument("--out_dir", default="processed", help="default folder for outputs")
    ap.add_argument("--train_out", help="override path of processed train csv")
    ap.add_argument("--val_out", help="override path of processed val csv")
    ap.add_argument("--test_out", help="override path of processed test csv")
    ap.add_argument("--preprocessor_out", help="override path of fitted preprocessor (.pkl)")
    ap.add_argument("--log_cols", nargs="*", default=[], help="numeric columns to signed-log1p before scaling")
    ap.add_argument("--parquet", action="store_true", help="also write a .parquet next to each csv (needs pyarrow)")
    ap.add_argument("--cyclical", action="store_true", help="also add sin/cos of ArrivalDateMonth and week number")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    paths = {
        "train": Path(a.train_out) if a.train_out else out_dir / "train_processed.csv",
        "val": Path(a.val_out) if a.val_out else out_dir / "val_processed.csv",
        "test": Path(a.test_out) if a.test_out else out_dir / "test_processed.csv",
    }
    pp_path = Path(a.preprocessor_out) if a.preprocessor_out else out_dir / "preprocessor.pkl"

    # read everything as text so nothing is silently converted to NaN
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    raw = {"train": read(a.train), "val": read(a.val), "test": read(a.test)}

    pp = HotelPreprocessor(log_cols=a.log_cols, cyclical=a.cyclical)
    out = {"train": pp.fit_transform(raw["train"])}          # fit on TRAIN only
    skew_report = dict(sorted(pp.skew_.items(), key=lambda kv: -abs(kv[1])))
    reports = {}
    for name in ("val", "test"):
        out[name] = pp.transform(raw[name])                   # reuse train statistics
        reports[name] = pp.report_

    # ---- write ----
    for name, X in out.items():
        assert not X.isna().any().any(), f"NaN in processed {name}"
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        X.to_csv(paths[name], index=False)                    # header included, float64 full precision
        if a.parquet:
            X.to_parquet(paths[name].with_suffix(".parquet"), index=False)
    pp.save(pp_path)

    # ---- report ----
    print("\n=== ROW COUNTS (no rows removed) ===")
    tot_in = tot_out = 0
    for name in ("train", "val", "test"):
        n_in, n_out = len(raw[name]), len(out[name])
        assert n_in == n_out, f"row count changed in {name}: {n_in} -> {n_out}"
        tot_in += n_in
        tot_out += n_out
        print(f"{name:5s} input={n_in:>8,}  output={n_out:>8,}  cols={out[name].shape[1]}  -> {paths[name]}")
    print(f"TOTAL input={tot_in:,}  output={tot_out:,}")

    print(f"\nFeatures per file: {len(pp.feature_names_)}  | preprocessor saved to {pp_path}")
    print("\n=== TRAIN SKEWNESS of numeric columns (|skew| > 1 -> log candidate; none applied unless --log_cols) ===")
    for c, v in skew_report.items():
        flag = "  <-- consider log" if abs(v) > 1 and c not in pp.log_cols else ""
        print(f"{c:32s} {v:8.2f}{flag}")
    for name in ("val", "test"):
        r = reports[name]
        print(f"\n=== {name.upper()} vs TRAIN statistics ===")
        print("rows with category unseen in train:", r["unseen_rows"] or "none")
        if r["out_of_range"]:
            for c, (n, lo, hi) in r["out_of_range"].items():
                print(f"outside [0,1] (not clipped): {c}: {n} rows, range [{lo:.4f}, {hi:.4f}]")
        else:
            print("all scaled values within [0,1]")


if __name__ == "__main__":
    main()
