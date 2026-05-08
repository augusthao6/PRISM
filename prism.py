from AlgorithmImports import *
import numpy as np
import pandas as pd
import pickle
from datetime import date, timedelta
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from scipy.stats import rankdata
from hmmlearn.hmm import GaussianHMM

"""
PRISM v7.7 
Starting Cash: $10,000,000
"""

BULL = 0
CHOP = 1
BEAR = 2
REGIME_NAMES = {BULL: "Bull", CHOP: "Chop", BEAR: "Bear"}


class Cfg:
    IS_START_DATE = (2016, 1, 1)
    IS_END_DATE   = (2022, 1, 1)

    OOS_PERIOD = "C"
    if OOS_PERIOD == "A":
        OOS_START = (2023, 5, 1)
        OOS_END   = (2023, 11, 1)
    elif OOS_PERIOD == "B":
        OOS_START = (2025, 1, 1)
        OOS_END   = (2025, 7, 1)
    elif OOS_PERIOD == "C":
        OOS_START = (2025, 10, 1)
        OOS_END   = (2026, 4, 1)
    elif OOS_PERIOD == "STRESS_COVID":
        OOS_START = (2020, 2, 1)
        OOS_END   = (2020, 5, 1)
    elif OOS_PERIOD == "STRESS_2008":
        OOS_START = (2008, 8, 1)
        OOS_END   = (2009, 2, 1)
    else:
        raise ValueError(f"Unknown OOS_PERIOD: {OOS_PERIOD}")

    START_DATE   = OOS_START
    END_DATE     = OOS_END
    INITIAL_CASH = 10_000_000  # Required by guidelines

    ETF_UNIVERSE = [
        "SPY", "QQQ", "IWM", "EFA", "EEM",
        "XLK", "XLF", "XLV", "XLE", "XLI",
        "XLP", "XLY", "XLU", "XLB", "XLRE",
        "TLT", "IEF", "HYG", "LQD", "AGG",
        "GLD", "SLV", "DBC", "VNQ", "USO",
    ]
    STOCK_UNIVERSE = [
        "AAPL", "MSFT", "GOOG", "META", "ADBE",
        "AMZN", "HD", "NKE", "MCD", "LOW",
        "JPM", "BAC", "V", "MA", "GS",
        "UNH", "JNJ", "PFE", "LLY", "ABBV",
        "XOM", "CVX", "COP", "EOG", "SLB",
        "CAT", "HON", "UPS", "LMT", "BA",
        "DIS", "CMCSA", "NFLX", "T", "VZ",
        "PG", "KO", "PEP", "WMT", "COST",
        "NEE", "DUK", "SO", "D", "AEP",
        "LIN", "SHW", "APD", "ECL", "DOW",
    ]

    @classmethod
    def full_universe(cls):
        return cls.ETF_UNIVERSE + cls.STOCK_UNIVERSE

    MOM_WINDOW     = 63
    REVERSAL_WIN   = 5
    QUALITY_VOL    = 60
    REL_STR_WIN    = 63
    SMA_SLOW       = 200
    SMA_FAST       = 50
    VOL_SHORT      = 20
    HISTORY_DAYS   = 265
    VOLPRICE_WIN   = 20
    WEEK52_WIN     = 252

    FACTORS = [
        "mom_63", "reversal_5", "inv_vol_60", "sma_spread",
        "rel_strength", "sector_rel_mom", "vol_price_div",
        "pos_52w", "state_cluster_fit",
    ]

    SECTOR_MAP = {
        "AAPL": "XLK", "MSFT": "XLK", "GOOG": "XLK", "META": "XLK", "ADBE": "XLK",
        "AMZN": "XLY", "HD": "XLY", "NKE": "XLY", "MCD": "XLY", "LOW": "XLY",
        "JPM": "XLF", "BAC": "XLF", "V": "XLF", "MA": "XLF", "GS": "XLF",
        "UNH": "XLV", "JNJ": "XLV", "PFE": "XLV", "LLY": "XLV", "ABBV": "XLV",
        "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "EOG": "XLE", "SLB": "XLE",
        "CAT": "XLI", "HON": "XLI", "UPS": "XLI", "LMT": "XLI", "BA": "XLI",
        "DIS": "XLC", "CMCSA": "XLC", "NFLX": "XLC", "T": "XLC", "VZ": "XLC",
        "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP", "COST": "XLP",
        "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "D": "XLU", "AEP": "XLU",
        "LIN": "XLB", "SHW": "XLB", "APD": "XLB", "ECL": "XLB", "DOW": "XLB",
    }

    N_NAME_CLUSTERS     = 6
    NAME_CLUSTER_FEATURES = [
        "lt_avg_ret", "lt_vol", "lt_spy_corr",
        "lt_abs_above_sma", "lt_mom_rank", "lt_drawdown_freq",
    ]

    UNIVERSE_PRUNE_FRACTION = 0.30
    UNIVERSE_MIN_RETAIN     = 40

    CLUSTER_MAX_WEIGHT = 0.40
    CORR_THRESHOLD  = 0.70
    CORR_SCALE_DOWN = 0.70

    N_REGIMES           = 3
    HMM_LOOKBACK        = 504
    REGIME_CONFIRM_DAYS = 3

    MODEL_A_PARAMS = {
        "n_estimators": 150, "max_depth": 5, "min_samples_leaf": 20,
        "max_features": "sqrt", "bootstrap": False, "n_jobs": 2, "random_state": 42,
    }
    MODEL_B_PARAMS = {
        "n_estimators": 150, "max_depth": 3, "learning_rate": 0.05,
        "subsample": 0.8, "random_state": 42,
    }
    FORWARD_DAYS = 10
    MIN_PANEL    = 2000

    HIGH_SCORE_PCTILE  = 0.55
    HIGH_PROB_THRESH   = 0.52
    LOOSE_SCORE_DELTA  = -0.10
    LOOSE_PROB_DELTA   = -0.01
    TIGHT_SCORE_DELTA  = +0.05
    TIGHT_PROB_DELTA   = +0.01
    HELD_NAME_SCORE_BONUS = 0.05
    HIGH_RELIABILITY   = 0.57
    LOW_RELIABILITY    = 0.48
    REL_SIZE_LOW       = 0.50
    REL_SIZE_MID       = 0.80
    REL_SIZE_HIGH      = 1.00
    REL_SIZE_UNKNOWN   = 0.70

    N_MARKET_STATES       = 5
    MARKET_STATE_FEATURES = [
        "breadth_above_sma200", "mean_ret_20d", "dispersion_ret_20d",
        "mean_vol_20d", "mean_correlation", "risk_on_tilt",
    ]

    TOP_K             = 6
    MAX_POSITION      = 0.15
    MIN_POSITION      = 0.05
    COV_LOOKBACK      = 60

    VOL_TARGET_BY_REGIME = {BULL: 0.12, CHOP: 0.07, BEAR: 0.04}
    RESCALE_MIN = 0.40
    RESCALE_MAX = 0.85

    REBAL_OVERLAP      = 0.80
    REBAL_WEIGHT_SHIFT = 0.02

    WARMUP_DAYS = 270

    USE_OBJECT_STORE = True
    FORCE_RETRAIN    = False
    MODEL_STORE_KEY  = "prism/v7opt2/trained_models.pkl"


class FeatureComputer:
    @staticmethod
    def compute(df, spy_df=None, spy_ret=None):
        if df is None or len(df) < Cfg.SMA_SLOW + 5:
            return None
        d = df.copy().sort_index()
        d["ret_1"]      = d["close"].pct_change(1)
        d["mom_63"]     = d["close"].pct_change(Cfg.MOM_WINDOW)
        d["reversal_5"] = -d["close"].pct_change(Cfg.REVERSAL_WIN)
        d["vol_60"]     = d["ret_1"].rolling(Cfg.QUALITY_VOL).std()
        d["inv_vol_60"] = -d["vol_60"]
        d["sma_fast"]   = d["close"].rolling(Cfg.SMA_FAST).mean()
        d["sma_slow"]   = d["close"].rolling(Cfg.SMA_SLOW).mean()
        d["sma_spread"] = (d["sma_fast"] / d["sma_slow"]) - 1
        d["above_sma200"] = (d["close"] > d["sma_slow"]).astype(float)
        d["vol_20"]     = d["ret_1"].rolling(Cfg.VOL_SHORT).std()
        d["vol_ratio"]  = -(d["vol_20"] / d["vol_60"].replace(0, np.nan))
        log_vol = np.log(d["volume"].replace(0, np.nan))
        d["vol_price_div"] = d["ret_1"].rolling(Cfg.VOLPRICE_WIN).corr(log_vol)
        high_52 = d["close"].rolling(Cfg.WEEK52_WIN, min_periods=60).max()
        low_52  = d["close"].rolling(Cfg.WEEK52_WIN, min_periods=60).min()
        rng = (high_52 - low_52).replace(0, np.nan)
        d["pos_52w"] = ((d["close"] - low_52) / rng).clip(0, 1)
        if spy_df is not None and not spy_df.empty:
            spy_mom = spy_df["close"].pct_change(Cfg.REL_STR_WIN).rename("spy_mom")
            d = d.join(spy_mom, how="left")
            d["rel_strength"] = d["mom_63"] - d["spy_mom"]
        else:
            d["rel_strength"] = 0.0
        d["sector_rel_mom"] = 0.0
        keep = ["close", "ret_1", "vol_60", "above_sma200"] + Cfg.FACTORS
        return d[[c for c in keep if c in d.columns]].replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def parse_ohlcv(symbol, hist):
        try:
            if hist is None or hist.empty: return None
            df = hist.copy()
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                df.columns = [str(c).lower() for c in df.columns]
                if "symbol" in df.columns:
                    df = df[df["symbol"] == symbol]
                if "time" not in df.columns: return None
                df["time"] = pd.to_datetime(df["time"])
                df = df.set_index("time")
            else:
                df.columns = [str(c).lower() for c in df.columns]
            needed = ["open", "high", "low", "close", "volume"]
            if any(c not in df.columns for c in needed): return None
            return df[needed].sort_index()
        except Exception:
            return None

    @staticmethod
    def parse_spy_series(hist):
        if hist is None or (hasattr(hist, "empty") and hist.empty):
            return pd.DataFrame()
        if isinstance(hist, pd.DataFrame):
            close = hist["close"] if "close" in hist.columns else hist.iloc[:, 0]
        else:
            close = hist
        if isinstance(close.index, pd.MultiIndex):
            close = close.droplevel(0)
        return close.to_frame(name="close")


class FactorEngine:
    BASE_FACTORS = Cfg.FACTORS

    @staticmethod
    def regime_feature_names():
        return [f"hmm_{REGIME_NAMES[r].lower()}" for r in [BULL, CHOP, BEAR]]

    @classmethod
    def all_factor_names(cls):
        return cls.BASE_FACTORS + cls.regime_feature_names()

    def compute_factors(self, feature_frames, hmm_regime,
                        name_cluster_fn=None, state_cluster_lookup=None,
                        market_state=-1):
        if not feature_frames: return pd.DataFrame()
        regime_onehot = {n: 0.0 for n in self.regime_feature_names()}
        regime_onehot[f"hmm_{REGIME_NAMES[hmm_regime].lower()}"] = 1.0
        state_fit = {}
        if name_cluster_fn and state_cluster_lookup and market_state >= 0:
            for sym in feature_frames:
                cid = name_cluster_fn(sym)
                state_fit[sym] = float(state_cluster_lookup.get((market_state, cid), 0.0)) \
                                  if cid >= 0 else 0.0
        rows = {}
        for sym, d in feature_frames.items():
            if d is None or d.empty: continue
            last = d.iloc[-1]
            row = {}
            for c in self.BASE_FACTORS:
                row[c] = state_fit.get(sym, 0.0) if c == "state_cluster_fit" else last.get(c, np.nan)
            row.update(regime_onehot)
            rows[sym] = row
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows).T.reindex(columns=self.all_factor_names())
        df = self._winsorize(df, self.BASE_FACTORS)
        df = self._zscore(df, self.BASE_FACTORS)
        return df

    def compute_factors_from_row_dict(self, rows_dict, hmm_regime, state_fit_by_ticker=None):
        if not rows_dict: return pd.DataFrame()
        regime_onehot = {n: 0.0 for n in self.regime_feature_names()}
        regime_onehot[f"hmm_{REGIME_NAMES[hmm_regime].lower()}"] = 1.0
        rows = {}
        for sym, vals in rows_dict.items():
            row = {}
            for c in self.BASE_FACTORS:
                row[c] = (state_fit_by_ticker or {}).get(sym, 0.0) if c == "state_cluster_fit" else vals.get(c, np.nan)
            row.update(regime_onehot)
            rows[sym] = row
        df = pd.DataFrame(rows).T.reindex(columns=self.all_factor_names())
        df = self._winsorize(df, self.BASE_FACTORS)
        df = self._zscore(df, self.BASE_FACTORS)
        return df

    @staticmethod
    def _winsorize(df, cols, sigma=3.0):
        out = df.copy()
        for col in cols:
            if col not in out.columns: continue
            s = out[col].dropna()
            if len(s) < 3: continue
            mu, sd = s.mean(), s.std()
            out[col] = out[col].clip(mu - sigma*sd, mu + sigma*sd)
        return out

    @staticmethod
    def _zscore(df, cols):
        out = df.copy()
        for col in cols:
            if col not in out.columns: continue
            s = out[col].dropna()
            if len(s) < 3: continue
            mu, sd = s.mean(), s.std()
            if sd > 1e-8: out[col] = (out[col] - mu) / sd
        return out


class RegimeDetector:
    def __init__(self, algorithm):
        self._algo = algorithm
        self._model = None
        self._state_map = {}
        self.current_regime = BULL
        self._pending_regime = None
        self._pending_count  = 0

    def fit(self, spy_prices):
        X = self._features(spy_prices)
        if X is None or len(X) < 120: return False
        X_train = X[-Cfg.HMM_LOOKBACK:]
        for seed in [42, 7, 123, 2024]:
            try:
                model = GaussianHMM(n_components=Cfg.N_REGIMES, covariance_type="diag",
                                    n_iter=200, tol=1e-4, min_covar=1e-3, random_state=seed)
                model.fit(X_train)
                if not self._validate_hmm(model):
                    self._algo.Log(f"[Regime] seed={seed} degenerate"); continue
                self._model = model
                self._state_map = self._build_state_map(model)
                self._algo.Log(f"[Regime] HMM trained seed={seed}"); return True
            except Exception as e:
                self._algo.Log(f"[Regime] seed={seed} failed: {e}")
        self._model = None
        self._algo.Log("[Regime] all seeds failed — vol-tertile proxy")
        return False

    @staticmethod
    def _validate_hmm(model):
        try:
            return (np.all(np.isfinite(model.startprob_)) and
                    np.all(np.isfinite(model.transmat_))  and
                    np.all(np.isfinite(model.means_))     and
                    np.all(np.isfinite(model.covars_))    and
                    np.isclose(model.startprob_.sum(), 1.0, atol=1e-3))
        except Exception: return False

    def predict_raw(self, spy_prices):
        if self._model is None: return self._vol_tertile_proxy(spy_prices)
        X = self._features(spy_prices)
        if X is None or len(X) < 10: return self.current_regime
        if not self._validate_hmm(self._model):
            self._model = None; return self._vol_tertile_proxy(spy_prices)
        try:
            raw = self._model.predict(X[-60:])
            return self._state_map.get(int(raw[-1]), BULL)
        except Exception:
            self._model = None; return self._vol_tertile_proxy(spy_prices)

    def predict_confirmed(self, spy_prices):
        raw = self.predict_raw(spy_prices)
        if raw == self.current_regime:
            self._pending_regime = None; self._pending_count = 0
            return self.current_regime
        if self._pending_regime == raw: self._pending_count += 1
        else: self._pending_regime = raw; self._pending_count = 1
        if self._pending_count >= Cfg.REGIME_CONFIRM_DAYS:
            self._algo.Log(f"[Regime] {REGIME_NAMES[self.current_regime]}→{REGIME_NAMES[raw]}")
            self.current_regime = raw; self._pending_regime = None; self._pending_count = 0
        return self.current_regime

    @staticmethod
    def _vol_tertile_proxy(spy_prices):
        if len(spy_prices) < 60: return BULL
        rv = spy_prices.pct_change().rolling(30).std().iloc[-1]
        if pd.isna(rv): return BULL
        return BULL if rv < 0.008 else (CHOP if rv < 0.014 else BEAR)

    @staticmethod
    def _features(spy_prices):
        p = spy_prices.dropna()
        if len(p) < 35: return None
        lr = np.log(p / p.shift(1)).dropna()
        rv = lr.rolling(30).std().dropna()
        if len(rv) < 5: return None
        rc = rv.diff().dropna(); idx = rc.index
        return np.column_stack([lr.reindex(idx).fillna(0).values, rc.values, rv.reindex(idx).ffill().values])

    @staticmethod
    def _build_state_map(model):
        order = np.argsort(model.means_[:, 2])
        return {int(r): [BULL, CHOP, BEAR][i] for i, r in enumerate(order)}


class DualModelStack:
    def __init__(self, algorithm):
        self._algo = algorithm
        self._model_a = None; self._model_b = None
        self._feature_names = list(FactorEngine.all_factor_names())
        self._n_feat = len(self._feature_names)
        self._chunk_size = 2048
        self._chunks_X = []; self._chunks_y = []
        self._buf_X = np.zeros((self._chunk_size, self._n_feat), dtype=np.float32)
        self._buf_y = np.zeros(self._chunk_size, dtype=np.float32)
        self._buf_fill = 0; self._total_rows = 0

    def record(self, factor_df, forward_returns):
        common = factor_df.index.intersection(forward_returns.index)
        if len(common) < 3: return
        try:
            mat = factor_df.loc[common, self._feature_names].values.astype(np.float32)
        except KeyError:
            cols = [c for c in self._feature_names if c in factor_df.columns]
            if not cols: return
            mat = np.full((len(common), self._n_feat), np.nan, dtype=np.float32)
            ci = {n: i for i, n in enumerate(self._feature_names)}
            for c in cols: mat[:, ci[c]] = factor_df.loc[common, c].values.astype(np.float32)
        fwd = forward_returns.reindex(common).values.astype(np.float32)
        for i in range(len(common)):
            if np.isnan(fwd[i]): continue
            self._buf_X[self._buf_fill] = mat[i]; self._buf_y[self._buf_fill] = fwd[i]
            self._buf_fill += 1; self._total_rows += 1
            if self._buf_fill >= self._chunk_size: self._flush()

    def _flush(self):
        if self._buf_fill == 0: return
        self._chunks_X.append(self._buf_X[:self._buf_fill].copy())
        self._chunks_y.append(self._buf_y[:self._buf_fill].copy())
        self._buf_fill = 0

    def train(self):
        self._flush()
        if not self._chunks_X: self._algo.Log("[DualModel] Panel empty"); return
        X = np.concatenate(self._chunks_X)
        y_raw = np.concatenate(self._chunks_y)
        if len(X) < Cfg.MIN_PANEL:
            self._algo.Log(f"[DualModel] {len(X)} < {Cfg.MIN_PANEL}"); return
        X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
        y_rank = (rankdata(y_raw) / len(y_raw)).astype(np.float32)
        self._model_a = ExtraTreesRegressor(**Cfg.MODEL_A_PARAMS)
        self._model_a.fit(X, y_rank)
        self._algo.Log(f"[Model A] trained n={len(X)}")
        y_pos = (y_raw > 0).astype(np.float32)
        self._model_b = GradientBoostingRegressor(**Cfg.MODEL_B_PARAMS)
        self._model_b.fit(X, y_pos)
        self._algo.Log(f"[Model B] trained n={len(X)} pos_rate={y_pos.mean():.3f}")
        if hasattr(self._model_a, "feature_importances_"):
            imps = sorted(zip(self._feature_names, self._model_a.feature_importances_), key=lambda kv: -kv[1])
            for name, imp in imps[:5]: self._algo.Log(f"  imp {name}: {imp:.4f}")
        self._chunks_X.clear(); self._chunks_y.clear(); self._buf_X = self._buf_y = None

    def predict(self, factor_df):
        if self._model_a is None or self._model_b is None:
            return pd.DataFrame({"score_a": 0.5, "prob_b": 0.5}, index=factor_df.index)
        avail = [f for f in self._feature_names if f in factor_df.columns]
        if not avail:
            return pd.DataFrame({"score_a": 0.5, "prob_b": 0.5}, index=factor_df.index)
        X = factor_df[avail].fillna(0.).values
        try:
            return pd.DataFrame({
                "score_a": self._model_a.predict(X),
                "prob_b":  np.clip(self._model_b.predict(X), 0., 1.),
            }, index=factor_df.index)
        except Exception as e:
            self._algo.Log(f"[DualModel] predict err: {e}")
            return pd.DataFrame({"score_a": 0.5, "prob_b": 0.5}, index=factor_df.index)

    def panel_size(self): return self._total_rows
    def has_models(self): return self._model_a is not None and self._model_b is not None
    def score_factor_df(self, fdf): return self.predict(fdf)


class NameClusterer:
    def __init__(self, algorithm):
        self._algo = algorithm
        self._scaler = None; self._kmeans = None
        self.labels = {}; self.ready = False

    @staticmethod
    def compute_name_features(feat_df, spy_ret_series):
        if feat_df is None or feat_df.empty or len(feat_df) < 100: return None
        try:
            ret = feat_df["ret_1"].dropna()
            if len(ret) < 100: return None
            avg_ret = float(ret.mean()); vol = float(ret.std())
            if spy_ret_series is not None and not spy_ret_series.empty:
                aligned = pd.concat([ret, spy_ret_series], axis=1, join="inner").dropna()
                corr = float(aligned.iloc[:,0].corr(aligned.iloc[:,1])) if len(aligned) >= 50 else 0.0
            else: corr = 0.0
            close = feat_df["close"].dropna()
            sma200 = close.rolling(200).mean()
            above = float((close > sma200).dropna().mean()) if len((close > sma200).dropna()) > 0 else 0.5
            mom_mean = float(feat_df["mom_63"].dropna().mean()) if "mom_63" in feat_df.columns else 0.0
            peak20 = close.rolling(20).max()
            dd_freq = float((close / peak20 < 0.95).dropna().mean())
            return {"lt_avg_ret": avg_ret, "lt_vol": vol, "lt_spy_corr": corr,
                    "lt_abs_above_sma": above, "lt_mom_rank": mom_mean, "lt_drawdown_freq": dd_freq}
        except Exception: return None

    def fit(self, name_features_dict):
        if not name_features_dict: return False
        df = pd.DataFrame(name_features_dict).T.replace([np.inf,-np.inf], np.nan).dropna()
        if len(df) < Cfg.N_NAME_CLUSTERS * 2: return False
        X = df[Cfg.NAME_CLUSTER_FEATURES].values
        self._scaler = RobustScaler(); Xs = self._scaler.fit_transform(X)
        self._kmeans = KMeans(n_clusters=Cfg.N_NAME_CLUSTERS, init="k-means++", n_init=20, random_state=42)
        labels = self._kmeans.fit_predict(Xs)
        self.labels = dict(zip(df.index.tolist(), labels.astype(int).tolist()))
        self.ready = True
        for cid in range(Cfg.N_NAME_CLUSTERS):
            m = [t for t,c in self.labels.items() if c==cid]
            self._algo.Log(f"[NameCluster] c{cid} n={len(m)} {m[:6]}")
        return True

    def get_cluster(self, ticker): return self.labels.get(ticker, -1)


class MarketStateCluster:
    def __init__(self, algorithm):
        self._algo = algorithm; self._scaler = None; self._kmeans = None; self.ready = False

    @staticmethod
    def compute_state_vector(feature_frames):
        if not feature_frames: return None
        above_sma=[]; ret_20=[]; vol_20=[]; ret_1_series={}
        for sym, feat in feature_frames.items():
            if feat is None or feat.empty: continue
            last = feat.iloc[-1]
            if not pd.isna(last.get("above_sma200")): above_sma.append(float(last["above_sma200"]))
            mom = last.get("mom_63")
            if mom is not None and not pd.isna(mom): ret_20.append(float(mom)*(20./63.))
            vol = last.get("vol_60")
            if vol is not None and not pd.isna(vol): vol_20.append(float(vol))
            r1 = feat["ret_1"].dropna().iloc[-20:]
            if len(r1) >= 18: ret_1_series[sym] = r1
        if len(above_sma) < 10 or len(ret_1_series) < 10: return None
        try:
            corr_df = pd.DataFrame(ret_1_series).corr(); n = corr_df.shape[0]
            mask = np.triu(np.ones(corr_df.shape, dtype=bool), k=1)
            mean_corr = float(corr_df.values[mask].mean()) if n >= 2 else 0.5
        except Exception: mean_corr = 0.5

        def _safe_mean(syms, col):
            vals = [float(feature_frames[s].iloc[-1].get(col))
                    for s in syms if s in feature_frames and feature_frames[s] is not None
                    and not feature_frames[s].empty
                    and not pd.isna(feature_frames[s].iloc[-1].get(col, np.nan))]
            return float(np.mean(vals)) if vals else 0.0

        return {
            "breadth_above_sma200": float(np.mean(above_sma)),
            "mean_ret_20d": float(np.mean(ret_20)) if ret_20 else 0.0,
            "dispersion_ret_20d": float(np.std(ret_20)) if len(ret_20) >= 2 else 0.0,
            "mean_vol_20d": float(np.mean(vol_20)) if vol_20 else 0.0,
            "mean_correlation": mean_corr,
            "risk_on_tilt": _safe_mean(["SPY","QQQ","XLK"],"mom_63") - _safe_mean(["TLT","GLD"],"mom_63"),
        }

    @staticmethod
    def compute_state_vector_from_panel_slice(feat_panels_at_date, ret_panel_tail):
        if not feat_panels_at_date: return None
        above_sma=[]; ret_20=[]; vol_20=[]
        for sym, last in feat_panels_at_date.items():
            if last is None or last.empty: continue
            if not pd.isna(last.get("above_sma200")): above_sma.append(float(last["above_sma200"]))
            mom = last.get("mom_63")
            if mom is not None and not pd.isna(mom): ret_20.append(float(mom)*(20./63.))
            vol = last.get("vol_60")
            if vol is not None and not pd.isna(vol): vol_20.append(float(vol))
        if len(above_sma) < 10: return None
        try:
            corr_df = ret_panel_tail.corr(); n = corr_df.shape[0]
            mask = np.triu(np.ones(corr_df.shape,dtype=bool),k=1)
            mean_corr = float(np.nanmean(corr_df.values[mask])) if n>=2 else 0.5
        except Exception: mean_corr = 0.5

        def _safe_mom(syms):
            vals=[float(feat_panels_at_date[s].get("mom_63")) for s in syms
                  if s in feat_panels_at_date and feat_panels_at_date[s] is not None
                  and not pd.isna(feat_panels_at_date[s].get("mom_63",np.nan))]
            return float(np.mean(vals)) if vals else 0.0

        return {
            "breadth_above_sma200": float(np.mean(above_sma)),
            "mean_ret_20d": float(np.mean(ret_20)) if ret_20 else 0.0,
            "dispersion_ret_20d": float(np.std(ret_20)) if len(ret_20)>=2 else 0.0,
            "mean_vol_20d": float(np.mean(vol_20)) if vol_20 else 0.0,
            "mean_correlation": mean_corr,
            "risk_on_tilt": _safe_mom(["SPY","QQQ","XLK"]) - _safe_mom(["TLT","GLD"]),
        }

    def fit(self, state_history):
        min_obs = Cfg.N_MARKET_STATES * 20
        if not state_history or len(state_history) < min_obs: return False
        df = pd.DataFrame(state_history).replace([np.inf,-np.inf], np.nan).dropna()
        if len(df) < min_obs: return False
        X = df[Cfg.MARKET_STATE_FEATURES].values
        self._scaler = RobustScaler(); Xs = self._scaler.fit_transform(X)
        self._kmeans = KMeans(n_clusters=Cfg.N_MARKET_STATES, init="k-means++", n_init=20, random_state=42)
        self._kmeans.fit(Xs); self.ready = True
        for cid in range(Cfg.N_MARKET_STATES):
            self._algo.Log(f"[MarketState] c{cid} n={int((self._kmeans.labels_==cid).sum())}")
        return True

    def classify(self, state_dict):
        if not self.ready or state_dict is None: return -1
        try:
            vec = np.array([[state_dict[k] for k in Cfg.MARKET_STATE_FEATURES]])
            if np.any(np.isnan(vec)): return -1
            return int(self._kmeans.predict(self._scaler.transform(vec))[0])
        except Exception: return -1

    def classify_batch(self, state_df):
        if not self.ready or state_df is None or state_df.empty:
            return np.full(len(state_df) if state_df is not None else 0, -1, dtype=int)
        try:
            X = state_df[Cfg.MARKET_STATE_FEATURES].values
            mask = ~np.isnan(X).any(axis=1)
            out = np.full(len(state_df), -1, dtype=int)
            if mask.any(): out[mask] = self._kmeans.predict(self._scaler.transform(X[mask]))
            return out
        except Exception: return np.full(len(state_df), -1, dtype=int)


class ReliabilityCalibrator:
    def __init__(self, algorithm):
        self._algo = algorithm
        self.cluster_reliability = {}; self.ready = False

    def calibrate(self, pass1_predictions, pass1_forward_returns, pass1_dates, pass1_state_labels):
        if pass1_predictions is None or pass1_forward_returns is None: return False
        try: df = pass1_predictions.join(pass1_forward_returns.rename("fwd"), how="inner")
        except Exception as e: self._algo.Log(f"[Reliability] {e}"); return False
        if df.empty: return False
        per_day = []
        for dt, grp in df.groupby(level=0):
            if len(grp) < 10: continue
            if "combined" not in grp.columns:
                grp = grp.copy(); grp["combined"] = grp["score_a"].rank(pct=True) * grp["prob_b"]
            sg = grp.sort_values("combined", ascending=False); n = len(sg); q = max(2, n//5)
            per_day.append({"date": dt, "hit": 1.0 if sg["fwd"].head(q).mean() > sg["fwd"].tail(q).mean() else 0.0})
        if not per_day: return False
        hr_df = pd.DataFrame(per_day)
        hr_df["cluster"] = hr_df["date"].map(pass1_state_labels).fillna(-1).astype(int)
        for cid in range(Cfg.N_MARKET_STATES):
            grp = hr_df[hr_df["cluster"]==cid]
            rel = float(grp["hit"].mean()) if len(grp) >= 10 else 0.50
            self.cluster_reliability[cid] = rel
            self._algo.Log(f"[Reliability] c{cid} n={len(grp)} hit={rel:.3f}")
        self.ready = True; return True

    def get_thresholds(self, cid):
        if not self.ready or cid < 0: return Cfg.HIGH_SCORE_PCTILE, Cfg.HIGH_PROB_THRESH
        rel = self.cluster_reliability.get(cid, 0.50)
        if rel < Cfg.LOW_RELIABILITY:
            return Cfg.HIGH_SCORE_PCTILE+Cfg.TIGHT_SCORE_DELTA, Cfg.HIGH_PROB_THRESH+Cfg.TIGHT_PROB_DELTA
        if rel > Cfg.HIGH_RELIABILITY:
            return Cfg.HIGH_SCORE_PCTILE+Cfg.LOOSE_SCORE_DELTA, Cfg.HIGH_PROB_THRESH+Cfg.LOOSE_PROB_DELTA
        return Cfg.HIGH_SCORE_PCTILE, Cfg.HIGH_PROB_THRESH

    def get_size_multiplier(self, cid):
        if not self.ready or cid < 0: return Cfg.REL_SIZE_UNKNOWN
        rel = self.cluster_reliability.get(cid, 0.50)
        return Cfg.REL_SIZE_LOW if rel < Cfg.LOW_RELIABILITY else \
               (Cfg.REL_SIZE_HIGH if rel >= Cfg.HIGH_RELIABILITY else Cfg.REL_SIZE_MID)


class ModelPersistence:
    @staticmethod
    def build_payload(algo):
        return {
            "version": 7,
            "saved_at_date": str(algo.Time.date()),
            "regime_model": algo._regime._model,
            "regime_state_map": algo._regime._state_map,
            "regime_current": algo._regime.current_regime,
            "ms_kmeans": algo._market_state._kmeans,
            "ms_scaler": algo._market_state._scaler,
            "ms_ready":  algo._market_state.ready,
            "nc_kmeans": algo._name_cluster._kmeans,
            "nc_scaler": algo._name_cluster._scaler,
            "nc_labels": algo._name_cluster.labels,
            "nc_ready":  algo._name_cluster.ready,
            "stack_model_a": algo._stack._model_a,
            "stack_model_b": algo._stack._model_b,
            "stack_feature_names": algo._stack._feature_names,
            "state_cluster_lookup": algo._state_cluster_lookup,
            "active_universe": algo._active_universe,
            "rel_cluster_reliability": algo._reliability.cluster_reliability,
            "rel_ready": algo._reliability.ready,
        }

    @staticmethod
    def save(algo):
        try:
            payload = ModelPersistence.build_payload(algo)
            blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            algo.ObjectStore.SaveBytes(Cfg.MODEL_STORE_KEY, blob)
            algo.Log(f"[Persist] saved {len(blob)/1e6:.2f}MB"); return True
        except Exception as e: algo.Log(f"[Persist] save FAILED: {e}"); return False

    @staticmethod
    def exists(algo):
        try: return algo.ObjectStore.ContainsKey(Cfg.MODEL_STORE_KEY)
        except: return False

    @staticmethod
    def load(algo):
        try:
            blob = algo.ObjectStore.ReadBytes(Cfg.MODEL_STORE_KEY)
            if not blob: return False
            p = pickle.loads(bytes(blob))
            if p.get("version",-1) != 7: algo.Log("[Persist] version mismatch"); return False
            algo.Log(f"[Persist] loading from {p.get('saved_at_date')}")
            algo._regime._model = p["regime_model"]; algo._regime._state_map = p["regime_state_map"]
            algo._regime.current_regime = p.get("regime_current", BULL)
            algo._market_state._kmeans = p["ms_kmeans"]; algo._market_state._scaler = p["ms_scaler"]
            algo._market_state.ready = p["ms_ready"]
            algo._name_cluster._kmeans = p["nc_kmeans"]; algo._name_cluster._scaler = p["nc_scaler"]
            algo._name_cluster.labels = p["nc_labels"]; algo._name_cluster.ready = p["nc_ready"]
            algo._stack._model_a = p["stack_model_a"]; algo._stack._model_b = p["stack_model_b"]
            algo._state_cluster_lookup = p["state_cluster_lookup"]
            algo._active_universe = p["active_universe"]
            algo._reliability.cluster_reliability = p["rel_cluster_reliability"]
            algo._reliability.ready = p["rel_ready"]
            algo.Log(f"[Persist] loaded OK"); return True
        except Exception as e: algo.Log(f"[Persist] load FAILED: {e}"); return False


class PrismQualityMomentum(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(*Cfg.START_DATE)
        self.SetEndDate(*Cfg.END_DATE)
        self.SetCash(Cfg.INITIAL_CASH)
        self.SetBenchmark("SPY")

        self._is_start = date(*Cfg.IS_START_DATE)
        self._is_end   = date(*Cfg.IS_END_DATE)
        self._trained  = False
        self._loaded_from_store = False

        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        self.SetSecurityInitializer(self._security_initializer)

        self._spy_symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
        self._symbols = [self._spy_symbol]
        for ticker in Cfg.full_universe():
            if ticker == "SPY": continue
            sym = self.AddEquity(ticker, Resolution.Daily).Symbol
            self._symbols.append(sym)
        self._ticker_to_sym = {sym.Value: sym for sym in self._symbols}

        self._feat_computer = FeatureComputer()
        self._factor_engine = FactorEngine()
        self._regime        = RegimeDetector(self)
        self._stack         = DualModelStack(self)
        self._market_state  = MarketStateCluster(self)
        self._reliability   = ReliabilityCalibrator(self)
        self._name_cluster  = NameClusterer(self)

        self._state_cluster_lookup = {}
        self._active_universe = None
        self._current_regime  = BULL
        self._current_ms      = -1
        self._target_weights  = {}
        self._prior_picks     = set()
        self._equity_peak     = Cfg.INITIAL_CASH
        self._spy_start_price = None

        self._frames_cache_date = None
        self._frames_cache = None

        if Cfg.USE_OBJECT_STORE and not Cfg.FORCE_RETRAIN:
            if ModelPersistence.exists(self):
                if ModelPersistence.load(self):
                    self._trained = True; self._loaded_from_store = True
                    self.Log("[init] ★ loaded pre-trained models")

        if not self._trained:
            self.Log("[init] training from IS history (2016-2022)")
            try:
                self._train_from_history()
                self._trained = True
                if Cfg.USE_OBJECT_STORE and self._stack.has_models():
                    ModelPersistence.save(self)
            except Exception as e:
                self.Log(f"[init] training FAILED: {e}")

        self.SetWarmUp(Cfg.WARMUP_DAYS, Resolution.Daily)

        self.Schedule.On(self.DateRules.Every(DayOfWeek.Monday),
                         self.TimeRules.AfterMarketOpen(self._spy_symbol, 30),
                         self._on_monday_rebalance)
        self.Schedule.On(self.DateRules.EveryDay(self._spy_symbol),
                         self.TimeRules.BeforeMarketClose(self._spy_symbol, 10),
                         self._end_of_day)
        self._setup_charts()
        self.Log(f"[init] v7.7 OOS={Cfg.OOS_PERIOD} cash=${Cfg.INITIAL_CASH:,} "
                 f"universe={len(self._symbols)} trained={self._trained}")

    def _security_initializer(self, security):
        security.SetSlippageModel(VolumeShareSlippageModel(volumeLimit=0.10, priceImpact=0.05))

    def _train_from_history(self):
        self.Log("[Train] ★ start")
        total_days = (self._is_end - self._is_start).days + 50
        lookback_trading = int(total_days * 0.72) + 100
        raw_panels = self._fetch_is_panels(lookback_trading)
        if not raw_panels or "SPY" not in raw_panels:
            self.Log("[Train] insufficient IS data"); return

        spy_df = FeatureComputer.parse_spy_series(raw_panels["SPY"])
        feat_panels = {}
        for ticker, raw_df in raw_panels.items():
            feat = FeatureComputer.compute(raw_df, spy_df)
            if feat is not None and not feat.empty:
                feat = feat[(feat.index.date >= self._is_start) & (feat.index.date < self._is_end)]
                if not feat.empty: feat_panels[ticker] = feat
        self._compute_sector_rel_mom_panel(feat_panels)
        self.Log(f"[Train] {len(feat_panels)} tickers")

        if "SPY" not in feat_panels: self.Log("[Train] no SPY — abort"); return
        all_dates = pd.DatetimeIndex(sorted(feat_panels["SPY"].index.unique()))
        self.Log(f"[Train] IS {len(all_dates)} days ({all_dates[0].date()}→{all_dates[-1].date()})")

        self._regime.fit(spy_df["close"])
        state_history = []; is_snaps = []

        for dt in all_dates:
            today_rows = {}; ret_recent = {}
            for ticker, panel in feat_panels.items():
                if dt not in panel.index: continue
                row = panel.loc[dt]
                if isinstance(row, pd.DataFrame): row = row.iloc[0]
                today_rows[ticker] = row
                pos = panel.index.get_indexer([dt])[0]
                recent = panel["ret_1"].iloc[max(0,pos-20):pos+1]
                if len(recent) >= 18: ret_recent[ticker] = recent.reset_index(drop=True)
            if len(today_rows) < 10: continue

            ret_df = pd.DataFrame(ret_recent) if ret_recent else pd.DataFrame()
            sv = MarketStateCluster.compute_state_vector_from_panel_slice(today_rows, ret_df)
            if sv is not None: state_history.append({"date": dt.date(), **sv})

            spy_slice = spy_df["close"][spy_df["close"].index <= dt]
            reg_proxy = RegimeDetector._vol_tertile_proxy(spy_slice)
            factor_df = self._factor_engine.compute_factors_from_row_dict(today_rows, reg_proxy)
            if factor_df.empty: continue

            closes = {t: float(row["close"]) for t, row in today_rows.items()
                      if "close" in row.index and not pd.isna(row["close"])}
            is_snaps.append({"date": dt.date(), "factor_df": factor_df,
                             "close_prices": closes, "regime_proxy": reg_proxy})

        self.Log(f"[Train] {len(state_history)} state vecs, {len(is_snaps)} snaps")
        self._state_history = state_history; self._is_factor_snapshots = is_snaps
        self._market_state.fit(state_history)

        spy_ret = spy_df["close"].pct_change().dropna()
        spy_ret_is = spy_ret[(spy_ret.index.date >= self._is_start) & (spy_ret.index.date < self._is_end)]
        name_feat = {}
        for t, fp in feat_panels.items():
            if "ret_1" not in fp.columns: continue
            nf = NameClusterer.compute_name_features(fp, spy_ret_is)
            if nf is not None: name_feat[t] = nf
        self._name_cluster.fit(name_feat)

        self._build_state_cluster_lookup()
        self._relabel_and_feed_stack()
        self.Log(f"[Train] panel={self._stack.panel_size()}")
        self._prune_universe()
        self._stack.train()
        if self._stack.has_models() and self._market_state.ready:
            self._calibrate_reliability()
        self._is_factor_snapshots = []; self._state_history = []
        self.Log("[Train] ★ done")

    def _fetch_is_panels(self, lookback):
        panels = {}
        for sym in self._symbols:
            try: raw = self.History(sym, self._is_start, self._is_end, Resolution.Daily)
            except:
                try: raw = self.History(sym, lookback, Resolution.Daily)
                except: continue
            df = FeatureComputer.parse_ohlcv(sym, raw)
            if df is None or len(df) < 100: continue
            df = df[(df.index.date >= self._is_start) & (df.index.date < self._is_end)]
            if len(df) >= 100: panels[sym.Value] = df
        return panels

    @staticmethod
    def _compute_sector_rel_mom_panel(feat_panels):
        for ticker, sector_etf in Cfg.SECTOR_MAP.items():
            if ticker not in feat_panels or sector_etf not in feat_panels: continue
            stock = feat_panels[ticker]; etf = feat_panels[sector_etf]
            if "mom_63" not in stock.columns or "mom_63" not in etf.columns: continue
            stock.loc[:, "sector_rel_mom"] = stock["mom_63"] - etf["mom_63"].reindex(stock.index).ffill()

    def _build_state_cluster_lookup(self):
        if not self._market_state.ready or not self._name_cluster.ready: return
        snaps = sorted(self._is_factor_snapshots, key=lambda s: s["date"])
        state_by_date = {}
        for e in self._state_history:
            sv = {k: e.get(k, 0.) for k in Cfg.MARKET_STATE_FEATURES}
            state_by_date[e["date"]] = self._market_state.classify(sv)
        by_cell = {}
        for i, snap in enumerate(snaps):
            fi = i + Cfg.FORWARD_DAYS
            if fi >= len(snaps): break
            ms = state_by_date.get(snap["date"], -1)
            if ms < 0: continue
            future = snaps[fi]["close_prices"]
            for t, p0 in snap["close_prices"].items():
                p1 = future.get(t)
                if p0 is None or p1 is None or p0 <= 0: continue
                cid = self._name_cluster.get_cluster(t)
                if cid < 0: continue
                by_cell.setdefault((ms, cid), []).append((p1/p0)-1.)
        for k, rets in by_cell.items():
            self._state_cluster_lookup[k] = float(np.mean(rets)) if len(rets) >= 10 else 0.

    def _relabel_and_feed_stack(self):
        snaps = sorted(self._is_factor_snapshots, key=lambda s: s["date"])
        state_by_date = {}
        for e in self._state_history:
            sv = {k: e.get(k, 0.) for k in Cfg.MARKET_STATE_FEATURES}
            state_by_date[e["date"]] = self._market_state.classify(sv)
        n_fed = 0
        for i, snap in enumerate(snaps):
            fi = i + Cfg.FORWARD_DAYS
            if fi >= len(snaps): break
            ms = state_by_date.get(snap["date"], -1)
            factor_df = snap["factor_df"].copy()
            if ms >= 0 and "state_cluster_fit" in factor_df.columns:
                for t in factor_df.index:
                    cid = self._name_cluster.get_cluster(t)
                    if cid >= 0:
                        factor_df.at[t, "state_cluster_fit"] = float(
                            self._state_cluster_lookup.get((ms, cid), 0.))
                col = factor_df["state_cluster_fit"].dropna()
                if len(col) >= 3:
                    mu, sd = col.mean(), col.std()
                    if sd > 1e-8: factor_df["state_cluster_fit"] = (factor_df["state_cluster_fit"]-mu)/sd
            future = snaps[fi]["close_prices"]
            fwd = {t: (p1/p0)-1. for t, p0 in snap["close_prices"].items()
                   if (p1 := future.get(t)) is not None and p0 > 0}
            if fwd: self._stack.record(factor_df, pd.Series(fwd)); n_fed += 1
        self.Log(f"[Train] fed {n_fed} snaps")

    def _prune_universe(self):
        snaps = sorted(self._is_factor_snapshots, key=lambda s: s["date"])
        per_name = {}
        for i, snap in enumerate(snaps):
            fi = i + Cfg.FORWARD_DAYS
            if fi >= len(snaps): break
            fdf = snap["factor_df"]
            if "mom_63" not in fdf.columns: continue
            mom_rank = fdf["mom_63"].rank(pct=True); future = snaps[fi]["close_prices"]
            for t, p0 in snap["close_prices"].items():
                p1 = future.get(t)
                if p0 is None or p1 is None or p0 <= 0 or t not in fdf.index: continue
                mr = mom_rank.get(t)
                if mr is None or pd.isna(mr): continue
                per_name.setdefault(t, []).append((float(mr), (p1/p0)-1.))
        per_name_ic = {}
        for t, pairs in per_name.items():
            if len(pairs) < 30: per_name_ic[t] = 0.; continue
            arr = np.array(pairs)
            try: corr = float(np.corrcoef(arr[:,0],arr[:,1])[0,1])
            except: corr = 0.
            per_name_ic[t] = 0. if np.isnan(corr) else corr
        if not per_name_ic: return
        ranked = sorted(per_name_ic.items(), key=lambda kv: kv[1])
        n = len(ranked); n_prune = int(n * Cfg.UNIVERSE_PRUNE_FRACTION)
        n_retain = max(n - n_prune, Cfg.UNIVERSE_MIN_RETAIN)
        self._active_universe = set(t for t, _ in ranked[n-n_retain:])
        self.Log(f"[Prune] retained {len(self._active_universe)}/{n}")

    def _calibrate_reliability(self):
        snaps = sorted(self._is_factor_snapshots, key=lambda s: s["date"])
        if len(snaps) < 20: return
        preds_rows = []; state_labels = {}
        if self._state_history:
            sh_df = pd.DataFrame(self._state_history)
            if "date" in sh_df.columns:
                labs = self._market_state.classify_batch(sh_df)
                for d, lab in zip(sh_df["date"].tolist(), labs):
                    if lab >= 0: state_labels[d] = int(lab)
        for i, snap in enumerate(snaps):
            fi = i + Cfg.FORWARD_DAYS
            if fi >= len(snaps): continue
            fwd_closes = snaps[fi]["close_prices"]
            scored = self._stack.score_factor_df(snap["factor_df"])
            scored["rank_pctile"] = scored["score_a"].rank(pct=True)
            scored["combined"] = scored["rank_pctile"] * scored["prob_b"]
            for sym in scored.index:
                p0 = snap["close_prices"].get(sym); p1 = fwd_closes.get(sym)
                if p0 is None or p1 is None or p0 <= 0: continue
                r = scored.loc[sym]
                preds_rows.append({"date": snap["date"], "symbol": sym,
                                   "score_a": float(r["score_a"]), "prob_b": float(r["prob_b"]),
                                   "combined": float(r["combined"]), "fwd": (p1/p0)-1.})
        if not preds_rows: return
        pred_df = pd.DataFrame(preds_rows).set_index(["date","symbol"])
        fwd_s   = pred_df["fwd"].copy()
        self._reliability.calibrate(pred_df[["score_a","prob_b","combined"]], fwd_s,
                                    list(pred_df.index.get_level_values(0).unique()), state_labels)

    def OnData(self, data):
        if self.IsWarmingUp or not self._trained: return
        cur = float(self.Portfolio.TotalPortfolioValue)
        if cur > self._equity_peak: self._equity_peak = cur

    def _on_monday_rebalance(self):
        if self.IsWarmingUp or not self._trained: return
        self._oos_rebalance()

    def _currently_held_tickers(self):
        return {sym.Value for sym in self.Portfolio.Keys
                if sym != self._spy_symbol and self.Portfolio[sym].Invested}

    def _oos_rebalance(self):
        feature_frames, spy_df = self._build_feature_frames()
        if not feature_frames: return

        if self._active_universe is not None:
            feature_frames = {t: f for t, f in feature_frames.items() if t in self._active_universe}
            if not feature_frames: return

        self._current_regime = self._regime.predict_confirmed(spy_df["close"])
        state_vec = MarketStateCluster.compute_state_vector(feature_frames)
        self._current_ms = self._market_state.classify(state_vec)
        score_th, prob_th = self._reliability.get_thresholds(self._current_ms)
        rel_size_mult = self._reliability.get_size_multiplier(self._current_ms)

        factor_df = self._factor_engine.compute_factors(
            feature_frames, self._current_regime,
            name_cluster_fn=self._name_cluster.get_cluster,
            state_cluster_lookup=self._state_cluster_lookup,
            market_state=self._current_ms,
        )
        if factor_df.empty: return

        pred = self._stack.predict(factor_df)
        pred["rank_pctile"] = pred["score_a"].rank(pct=True) if len(pred) >= 2 else 0.5

        held = self._currently_held_tickers()
        held_score_th = max(0., score_th - Cfg.HELD_NAME_SCORE_BONUS)
        is_held = pred.index.isin(held)
        score_ok = np.where(is_held, pred["rank_pctile"]>=held_score_th, pred["rank_pctile"]>=score_th)
        qualify   = pd.Series(score_ok & (pred["prob_b"]>=prob_th), index=pred.index)
        candidates = pred[qualify].copy()

        if len(candidates) < 3:
            self.Log(f"[RB] tight gate {len(candidates)} — fallback")
            candidates = pred.copy()
            if len(candidates) == 0:
                self._go_flat("no scored names"); return

        candidates["combined"] = candidates["rank_pctile"] * candidates["prob_b"]
        candidates = candidates.sort_values("combined", ascending=False)
        picks = self._select_with_cluster_diversification(candidates)
        if not picks: self._go_flat("diversification empty"); return

        current_picks = set(picks)
        overlap = len(current_picks & self._prior_picks) / max(1, len(current_picks)) if self._prior_picks else 0.
        picks_largely_same = overlap >= Cfg.REBAL_OVERLAP

        ret_matrix = self._return_matrix(feature_frames)
        target = self._build_weights(picks, candidates.loc[picks,"combined"],
                                     ret_matrix, self._current_regime, rel_size_mult)
        if not target: return
        target = self._apply_correlation_scale(target, ret_matrix)
        if picks_largely_same and self._weights_similar(target): return

        self._prior_picks = current_picks; self._target_weights = target
        self._execute(target)
        self._log_rebalance(target, candidates.loc[picks], score_th, prob_th, rel_size_mult)

    def _build_weights(self, picks, combined_scores, return_matrix, regime, rel_size_mult):
        if not picks: return {}
        raw = combined_scores.reindex(picks).fillna(0.5)
        w = raw / raw.sum() if raw.sum() > 0 else pd.Series(1./len(picks), index=picks)
        for _ in range(20):
            w = w.clip(0., Cfg.MAX_POSITION); w[w < Cfg.MIN_POSITION] = 0.
            s = w.sum()
            if s < 1e-8: break
            w /= s
            if w.max() <= Cfg.MAX_POSITION + 1e-6: break
        vt = Cfg.VOL_TARGET_BY_REGIME.get(regime, Cfg.VOL_TARGET_BY_REGIME[BULL])
        gross = self._compute_vol_target_gross(w, return_matrix, vt) * rel_size_mult
        gross = max(Cfg.RESCALE_MIN, min(gross, Cfg.RESCALE_MAX))
        return (w * gross).to_dict()

    @staticmethod
    def _compute_vol_target_gross(weights, return_matrix, vol_target):
        cols = [s for s in weights.index if s in return_matrix.columns]
        if len(cols) < 2: return Cfg.RESCALE_MAX
        sub = return_matrix[cols].dropna(how="all").fillna(0.)
        if sub.shape[0] < 20: return Cfg.RESCALE_MAX
        wv = weights.reindex(cols).fillna(0.).values
        var = float(wv @ sub.cov().values @ wv)
        if var <= 0: return Cfg.RESCALE_MAX
        return float(min(Cfg.RESCALE_MAX, vol_target / max(np.sqrt(var*252), 1e-6)))

    def _weights_similar(self, new_target):
        if not self._target_weights: return False
        all_keys = set(new_target) | set(self._target_weights)
        return max(abs(new_target.get(k,0.) - self._target_weights.get(k,0.)) for k in all_keys) <= Cfg.REBAL_WEIGHT_SHIFT

    def _select_with_cluster_diversification(self, candidates):
        if not self._name_cluster.ready:
            return list(candidates.head(Cfg.TOP_K).index)
        max_per = max(1, int(np.ceil(Cfg.TOP_K * Cfg.CLUSTER_MAX_WEIGHT)))
        picks = []; per_c = {}
        for t in candidates.index:
            if len(picks) >= Cfg.TOP_K: break
            cid = self._name_cluster.get_cluster(t)
            if cid < 0: picks.append(t); continue
            if per_c.get(cid, 0) >= max_per: continue
            picks.append(t); per_c[cid] = per_c.get(cid, 0) + 1
        return picks

    def _apply_correlation_scale(self, weights, return_matrix):
        if not weights or return_matrix.empty: return weights
        cols = [t for t in weights if t in return_matrix.columns]
        if len(cols) < 3: return weights
        sub = return_matrix[cols].dropna(how="all").fillna(0.)
        if sub.shape[0] < 20: return weights
        try:
            corr = sub.corr().values; n = corr.shape[0]
            if n < 2: return weights
            mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
            avg_corr = float(np.nanmean(corr[mask]))
        except Exception: return weights
        if avg_corr > Cfg.CORR_THRESHOLD:
            self.Log(f"[CorrMon] avg_corr={avg_corr:.3f} — scaling {Cfg.CORR_SCALE_DOWN:.2f}")
            return {k: v*Cfg.CORR_SCALE_DOWN for k,v in weights.items()}
        return weights

    def _build_feature_frames(self):
        today = self.Time.date()
        if self._frames_cache_date == today and self._frames_cache is not None:
            return self._frames_cache
        spy_raw = self.History(self._spy_symbol, Cfg.HISTORY_DAYS, Resolution.Daily)
        if spy_raw is None or spy_raw.empty:
            self._frames_cache_date = today; self._frames_cache = ({}, pd.DataFrame()); return self._frames_cache
        spy_df = FeatureComputer.parse_spy_series(spy_raw)
        frames = {}
        for sym in self._symbols:
            try: raw = self.History(sym, Cfg.HISTORY_DAYS, Resolution.Daily)
            except: continue
            df = FeatureComputer.parse_ohlcv(sym, raw)
            if df is None: continue
            feat = FeatureComputer.compute(df, spy_df)
            if feat is not None and not feat.empty: frames[sym.Value] = feat
        self._compute_sector_rel_mom(frames)
        self._frames_cache_date = today; self._frames_cache = (frames, spy_df); return self._frames_cache

    @staticmethod
    def _compute_sector_rel_mom(frames):
        for ticker, sector_etf in Cfg.SECTOR_MAP.items():
            if ticker not in frames or sector_etf not in frames: continue
            sf = frames[ticker]; ef = frames[sector_etf]
            if "mom_63" not in sf.columns or "mom_63" not in ef.columns: continue
            sf.loc[:, "sector_rel_mom"] = sf["mom_63"] - ef["mom_63"].reindex(sf.index).ffill()

    def _return_matrix(self, frames):
        rets = {sym: feat["ret_1"].dropna().iloc[-Cfg.COV_LOOKBACK:]
                for sym, feat in frames.items() if len(feat["ret_1"].dropna()) >= 20}
        return pd.DataFrame(rets) if rets else pd.DataFrame()

    def _execute(self, weights):
        for sym in list(self.Portfolio.Keys):
            if sym == self._spy_symbol: continue
            if self.Portfolio[sym].Invested and sym.Value not in weights:
                self.Liquidate(sym)
        for ticker, w in weights.items():
            if w < Cfg.MIN_POSITION: continue
            sym = self._ticker_to_sym.get(ticker)
            if sym is None: continue
            sec = self.Securities[sym]
            if not sec.HasData or sec.Price <= 0: continue
            self.SetHoldings(sym, w)

    def _go_flat(self, reason=""):
        for sym in list(self.Portfolio.Keys):
            if self.Portfolio[sym].Invested: self.Liquidate(sym)
        self._target_weights = {}; self._prior_picks = set()
        if reason: self.Log(f"{self.Time.date()} FLAT | {reason}")

    def _end_of_day(self):
        if self.IsWarmingUp: return
        cur = float(self.Portfolio.TotalPortfolioValue)
        if cur > self._equity_peak: self._equity_peak = cur
        dd = (cur/self._equity_peak)-1. if self._equity_peak > 0 else 0.
        self.Plot("Risk","DD",float(dd))

        spy_price = self.Securities[self._spy_symbol].Price
        if spy_price > 0:
            if self._spy_start_price is None:
                self._spy_start_price = spy_price
            spy_equity = (spy_price / self._spy_start_price) * Cfg.INITIAL_CASH
            self.Plot("Strategy Equity","SPY",float(spy_equity))

    def _log_rebalance(self, weights, picks_df, score_th, prob_th, rel_size_mult=1.0):
        r = REGIME_NAMES[self._current_regime]
        gross = sum(abs(w) for w in weights.values())
        lines = []
        for tk, w in sorted(weights.items(), key=lambda kv: -kv[1])[:8]:
            if tk in picks_df.index:
                row = picks_df.loc[tk]
                lines.append(f"{tk}:{w:.0%}(r{row['rank_pctile']:.2f},p{row['prob_b']:.2f})")
        rel = self._reliability.cluster_reliability.get(self._current_ms, float("nan"))
        vt  = Cfg.VOL_TARGET_BY_REGIME.get(self._current_regime, 0.12)
        self.Log(f"[RB] {self.Time.date()} {r}(vt={vt:.2%}) ms={self._current_ms} "
                 f"rel={rel:.2f} size={rel_size_mult:.2f} gross={gross:.1%} n={len(weights)} | "
                 + ", ".join(lines))
        self.Plot("Regime","State",float(self._current_regime))
        self.Plot("MarketState","Cluster",float(self._current_ms))
        self.Plot("Portfolio","N",float(len(weights)))
        self.Plot("Portfolio","Gross",float(gross))
        self.Plot("Risk","RelMult",float(rel_size_mult))

    def _setup_charts(self):
        for name, series in [("Regime",["State"]),("Portfolio",["N","Gross"]),
                              ("MarketState",["Cluster"]),("Risk",["DD","RelMult"])]:
            c = Chart(name)
            for s in series: c.AddSeries(Series(s, SeriesType.Line))
            self.AddChart(c)
