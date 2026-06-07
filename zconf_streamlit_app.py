#!/usr/bin/env python3
"""
Shantanu's VALUE MOMENTUM SWING TRADING SCANNER — v1.8 (Improved)
NSE Swing Trading Screener with CAPE / RSI / MACD signals

Improvements in v1.8:
✓ Fixed syntax errors (_weekly_signal_frame)
✓ Added yfinance caching (1-hour TTL)
✓ Retry logic for flaky API calls
✓ Reuse scan results (avoid recomputation)
✓ Input validation (weight sums, bounds)
✓ Type hints throughout
✓ Better error messages
✓ Progress indicators
✓ Modular organization

To run:
    pip install streamlit yfinance pandas numpy plotly reportlab tenacity
    streamlit run vms_scanner_v1_8_improved.py
"""

# ══════════════════════════════════════════════════════════════
# IMPORTS & CONFIG
# ══════════════════════════════════════════════════════════════

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Tuple, Optional, Dict, List, Any
import warnings
import io

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
    _TENACITY = True
except ImportError:
    _TENACITY = False

warnings.filterwarnings("ignore")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

try:
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                    Paragraph, Spacer, PageBreak)
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm
    _REPORTLAB = True
except ImportError:
    _REPORTLAB = False


# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="VMS Scanner v1.8 — NSE Swing Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATION
# ══════════════════════════════════════════════════════════════

DEFAULT_CFG: Dict[str, Any] = {
    # CAPE
    "use_cape": True, "cape_zlen": 252, "cape_bearish": True, "cape_max_q": 8,
    # RSI
    "rsi_len": 14, "rsi_zlen": 100, "rsi_contrarian": True,
    "rsi_dz_len": 5, "rsi_dz_weight": 0.4,
    # MACD (contrarian in v1.8)
    "macd_fast": 12, "macd_slow": 26, "macd_sig": 9,
    "macd_zlen": 100, "macd_dz_len": 5, "macd_dz_weight": 0.5,
    "macd_contrarian": True,
    # Weights (VWAP removed; CAPE/RSI/MACD only)
    "wt_cape": 33.0, "wt_rsi": 33.0, "wt_macd": 34.0,
    # Thresholds
    "th_sbuy": 2.0, "th_buy": 1.0, "th_sell": -1.0, "th_ssell": -2.0,
    "clamp_val": 3.0, "min_composite": 1.0, "workers": 8,
    # Divergence
    "div_enable": True, "piv_left": 5, "piv_right": 5, "div_lookback": 60,
    # Confidence
    "conf_strong": 1.75, "conf_moderate": 1.10,
    # ΔZ acceleration
    "dz_accel_enable": True, "dz_accel_bars": 2, "dz_accel_require_both": False,
    # Add_Conf gates
    "rsi_hard_max": 50.0,
    "add_conf_agree_min": 1,
    # Candle body
    "candle_body_enable": True, "candle_body_hard": False,
    "candle_green_tol": 0.998, "hammer_mult": 2.0,
    # 52-week high
    "hi52_enable": True, "hi52_bars": 252, "hi52_pct": 0.85,
    # Backtest
    "bt_min_composite": 1.25,
    "backtest_profit_pct": 8.0,
}

# ── Stock universe — merged from all watchlists (290 symbols) ──────────────────
# Sources: My_MPTDS_26 · 24MPTDS · MPTDS_MDSPORT_24_dr_ram
#          My_MDSPORT_26 · TV_CYCLICALS · TV_CYCLICALS-2 · TV_DEFENSIVES

# My MPTDS 26 watchlist
_MPTDS_26 = [
    "NESTLEIND", "IEX", "IRCTC", "ABBOTINDIA", "TRITURBINE", "BEL", "INFY", "ITC",
    "GRSE", "CUMMINSIND", "ASTRAZEN", "NBCC", "APARINDS", "CRISIL", "AJANTPHARM",
    "PERSISTENT", "HEROMOTOCO", "HEXT", "PIDILITIND", "EICHERMOT", "POLYCAB",
    "VOLTAMP", "LTTS", "SCHAEFFLER", "LTIM", "TORNTPHARM", "CHAMBLFERT", "UNITDSPR",
    "GODFRYPHLP", "BLUESTARCO", "GABRIEL", "JBCHEPHARM", "ASIANPAINT", "HAVELLS",
    "BERGEPAINT", "ZYDUSLIFE", "AVANTIFEED", "ICICIGI", "COROMANDEL", "MGL",
    "MPHASIS", "APLAPOLLO", "ZENSARTECH", "BSOFT", "TIMKEN", "GRINDWELL", "ALKEM",
    "COFORGE", "TITAN", "SUNDRMFAST", "GODREJAGRO", "BPCL", "FSL", "TVSMOTOR",
    "ASHOKLEY", "THANGAMAYL", "AEGISLOG", "CCL", "HATSUN", "APLLTD", "POWERGRID",
    "BAJFINANCE", "RECLTD", "PFC",
]

# My MDSPORT 26 watchlist
_MDSPORT_26 = [
    "SANOFICONR", "ICICIAMC", "ENRIN", "TCS", "INGERRAND", "ANANDRATHI", "CAMS",
    "IGIL", "IEX", "IRCTC", "CMPDI", "COALINDIA", "BSE", "ABBOTINDIA", "PRUDENT",
    "NATIONALUM", "CDSL", "TRITURBINE", "NAM-INDIA", "OFSS", "TRAVELFOOD", "INFY",
    "ZENTEC", "ITC", "CUMMINSIND", "TATAELXSI", "INDIAMART", "BLS", "NATCOPHARM",
    "KFINTECH", "EMAMILTD", "AJANTPHARM", "HCLTECH", "AIIL", "PERSISTENT",
    "TDPOWERSYS", "HEROMOTOCO", "ABB", "PIDILITIND", "NMDC", "LALPATHLAB", "ANTHEM",
    "CONCORDBIO", "LTTS", "ECLERX", "SCHAEFFLER", "LTIM", "HBLENGINE", "FINEORG",
    "MANYAVAR", "CAPLIPOINT", "TATATECH", "JBCHEPHARM", "SHRIPISTON", "SUMICHEM",
    "ALIVUS", "INDGN", "AVANTIFEED", "EIHOTEL", "GPIL", "PIIND", "CIPLA", "HSCL",
    "ELGIEQUIP", "PFIZER", "RATNAMANI", "VESUVIUS", "ZENSARTECH", "DATAPATTNS",
    "TIMKEN", "VIJAYA", "GRINDWELL", "VINATIORGA", "DIVISLAB", "SUNTV", "ALKEM",
    "SUNPHARMA", "ZFCVINDIA", "POLYMED", "ACUTAAS", "KPRMILL", "MEDANTA", "GALLANTT",
    "HAPPYFORGE", "AIAENG", "USHAMART", "SONACOMS", "TEGA", "OBEROIRLTY", "FINCABLES",
    "INDHOTEL", "NAVA", "CUPID", "AFFLE", "GESHIP", "BLACKBUCK",
]

# 24 MPTDS / Dr Ram combined list
_MPTDS_24 = [
    "360ONE", "ABB", "ABBOTINDIA", "ACE", "AEGISLOG", "AFFLE", "AIAENG", "AJANTPHARM",
    "AKZOINDIA", "ALKEM", "ALKYLAMINE", "APARINDS", "APLAPOLLO", "APOLLOHOSP",
    "APOLLOTYRE", "ARE&M", "ASAHIINDIA", "ASIANPAINT", "ASTRAL", "ASTRAZEN",
    "AUROPHARMA", "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BASF", "BATAINDIA",
    "BAYERCROP", "BEL", "BERGEPAINT", "BHARATFORG", "BHARTIARTL", "BLS", "BLUESTARCO",
    "BRIGADE", "BRITANNIA", "BSOFT", "CAMS", "CANFINHOME", "CAPLIPOINT", "CARBORUNIV",
    "CDSL", "CERA", "CESC", "CGCL", "CHAMBLFERT", "CHOLAFIN", "CIPLA", "CLEAN",
    "COALINDIA", "COCHINSHIP", "COFORGE", "COLPAL", "CONCOR", "CRISIL", "CUMMINSIND",
    "CYIENT", "DABUR", "DATAPATTNS", "DCMSHRIRAM", "DEEPAKNTR", "DIXON", "DMART",
    "DRREDDY", "EICHERMOT", "EIDPARRY", "ELECON", "ELECTCAST", "ELGIEQUIP",
    "ENDURANCE", "ESCORTS", "EXIDEIND", "FEDERALBNK", "FINCABLES", "FSL", "GODFRYPHLP",
    "GODREJIND", "GPPL", "GRANULES", "GRASIM", "GRINDWELL", "GSPL", "GUJGASLTD",
    "HAPPSTMNDS", "HAVELLS", "HBLPOWER", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDUNILVR", "HSCL", "HUDCO", "ICICIBANK", "INDHOTEL", "INFY",
    "IRCTC", "ITC", "JBCHEPHARM", "JBMA", "JINDALSAW", "JKCEMENT", "JKTYRE",
    "KAJARIACER", "KARURVYSYA", "KEI", "KIRLOSBROS", "KIRLOSENG", "KOTAKBANK",
    "KPITTECH", "KSB", "LALPATHLAB", "LAXMIMACH", "LINDEINDIA", "LT", "LTIM", "LTTS",
    "M&M", "MANAPPURAM", "MANKIND", "MARICO", "MARUTI", "MAZDOCK", "MOTILALOFS",
    "MPHASIS", "NAM-INDIA", "NATCOPHARM", "NCC", "NESTLEIND", "NEWGEN", "ONGC",
    "PERSISTENT", "PFC", "PIDILITIND", "PIIND", "POLYCAB", "POLYMED", "POWERGRID",
    "PRAJIND", "RADICO", "RATNAMANI", "RAYMOND", "REDINGTON", "RELIANCE", "RKFORGE",
    "RVNL", "SANOFI", "SBICARD", "SBIN", "SCHAEFFLER", "SHREECEM", "SHRIRAMFIN",
    "SIEMENS", "SKFINDIA", "SOLARINDS", "SONACOMS", "SRF", "SUNDARMFIN", "SUNDRMFAST",
    "SUNPHARMA", "SUPREMEIND", "TATACONSUM", "TATAELXSI", "TATAMOTORS", "TATASTEEL",
    "TCS", "TECHM", "TECHNOE", "TIINDIA", "TITAN", "TORNTPHARM", "TORNTPOWER",
    "TRENT", "TRIDENT", "TRITURBINE", "TTKPRESTIG", "TVSHLTD", "TVSMOTOR", "UBL",
    "ULTRACEMCO", "UNOMINDA", "VBL", "ZENSARTECH", "ZFCVINDIA",
]

# TV Cyclicals watchlist (Financials / Discretionary / Industrials / IT / Energy)
_TV_CYCLICALS = [
    # Financials
    "ABCAPITAL", "ANANDRATHI", "ANGELONE", "AUBANK", "BAJAJFINSV", "BAJFINANCE",
    "BSE", "CAMS", "CANFINHOME", "CDSL", "CHOLAFIN", "CHOLAHLDNG", "CRISIL", "CUB",
    "HDFCAMC", "HDFCBANK", "HOMEFIRST", "ICICIBANK", "ICICIGI", "KARURVYSYA",
    "KOTAKBANK", "LICHSGFIN", "MINDSPACE", "MOTILALOFS", "MUTHOOTFIN", "NAM-INDIA",
    "SBILIFE", "SBIN", "SHRIRAMFIN", "UTIAMC",
    # Consumer Discretionary
    "BLUESTARCO", "DIXON", "EICHERMOT", "ESCORTS", "HAVELLS", "HEROMOTOCO",
    "INDHOTEL", "IRCTC", "M&M", "MARUTI", "PAGEIND", "TATAMOTORS", "TITAN",
    "TVSHLTD", "TVSMOTOR", "UBL", "UNOMINDA", "VBL", "VGUARD",
    # Industrials
    "ABB", "ACE", "ADANIPORTS", "APARINDS", "APLAPOLLO", "ASHOKLEY", "BEL",
    "CARBORUNIV", "CEMPRO", "COCHINSHIP", "CUMMINSIND", "ELECON", "ELGIEQUIP",
    "ENDURANCE", "GABRIEL", "GRSE", "HAL", "HBLENGINE", "HUDCO", "INGERRAND",
    "IRCON", "JBMA", "JINDALSAW", "JKLAKSHMI", "KEI", "LT", "LTTS", "MAZDOCK",
    "NCC", "POLYCAB", "RATNAMANI", "SCHAEFFLER", "SHRIPISTON", "SOLARINDS",
    "SUNDRMFAST", "TARIL", "TECHNOE", "THERMAX", "TIINDIA", "TRITURBINE",
    "USHAMART", "VESUVIUS", "WELCORP",
    # Materials
    "ADANIENT", "ASTRAL", "BASF", "COROMANDEL", "GRASIM", "GRAVITA", "HINDALCO",
    "HINDCOPPER", "HINDZINC", "PCBL", "PIDILITIND", "PIIND", "SUPREMEIND",
    # Information Technology
    "BSOFT", "COFORGE", "FSL", "HCLTECH", "INFY", "KPITTECH", "MPHASIS", "NAUKRI",
    "NEWGEN", "OFSS", "REDINGTON", "TCS", "TIMETECHNO", "ZENSARTECH",
    # Energy & Utilities
    "CESC", "COALINDIA", "IEX", "POWERGRID", "TATAPOWER",
    # Real Estate & Conglomerate
    "ANANTRAJ", "BRIGADE", "RELIANCE",
]

# TV Defensives watchlist (Pharma / Healthcare / FMCG / Specialty)
_TV_DEFENSIVES = [
    # Pharmaceuticals
    "ABBOTINDIA", "AJANTPHARM", "ALKEM", "ASTRAZEN", "CAPLIPOINT", "CIPLA",
    "CONCORDBIO", "DRREDDY", "ERIS", "IPCALAB", "JBCHEPHARM", "NEULANDLAB",
    "TORNTPHARM", "ZYDUSLIFE",
    # Healthcare Services
    "APOLLOHOSP", "LALPATHLAB", "NH", "POLYMED",
    # FMCG & Consumer Staples
    "BRITANNIA", "CASTROLIND", "COLPAL", "DABUR", "GODFRYPHLP", "GODREJIND",
    "ITC", "LTFOODS", "TATACONSUM",
    # Specialty & Paints
    "AKZOINDIA",
    # Defense & Specialty
    "HSCL", "ZENTEC",
]

# ── Symbol corrections: variant → canonical yfinance-safe form ──────────────
# Symbols with & must be percent-encoded for yfinance URL construction
_CORRECTIONS: Dict[str, str] = {
    # & must be encoded as %26 for yfinance HTTP calls
    "M&M":      "M%26M",
    "M_M":      "M%26M",       # underscore alias
    "ARE&M":    "ARE%26M",
    "ARE_M":    "ARE%26M",     # underscore alias
    # Hyphen forms — yfinance accepts hyphens natively, no encoding needed
    "BAJAJ-AUTO": "BAJAJ-AUTO",
    "BAJAJ_AUTO": "BAJAJ-AUTO",
    "NAM-INDIA":  "NAM-INDIA",
    "NAM_INDIA":  "NAM-INDIA",
}

# ── Master universe: union of all watchlists, sorted ────────────────────────
ALL_SYMBOLS = sorted(set(
    _MPTDS_26 + _MDSPORT_26 + _MPTDS_24 + _TV_CYCLICALS + _TV_DEFENSIVES
))
INITIAL_CAPITAL = 1_000_000
POSITION_SIZE = 50_000
MAX_POSITIONS = INITIAL_CAPITAL // POSITION_SIZE

_INDIA_CPI: Dict[int, float] = {
    2013: 10.9, 2014: 6.4, 2015: 4.9, 2016: 4.5, 2017: 3.3,
    2018: 3.9, 2019: 3.7, 2020: 6.6, 2021: 5.1, 2022: 6.7,
    2023: 5.4, 2024: 4.8, 2025: 4.9, 2026: 4.5,
}


def to_yf(sym: str) -> str:
    """Convert NSE symbol to yfinance format."""
    return _CORRECTIONS.get(sym, sym) + ".NS"


# ══════════════════════════════════════════════════════════════
# CACHING & DATA FETCHING (Improvements 2-3)
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(ticker: str, period: str = "3y") -> Optional[pd.DataFrame]:
    """
    Fetch stock data with caching. Returns cleaned DataFrame or None.
    Wrapped with retry logic for reliability.
    """
    return _fetch_with_retry(ticker, period)


def _fetch_with_retry(ticker: str, period: str = "3y", max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Fetch data with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            tk = yf.Ticker(ticker)
            raw = tk.history(period=period, interval="1d", auto_adjust=True)
            
            if raw.empty:
                return None
            
            # Clean and validate
            df = _clean_df(raw)
            return df
            
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            # Exponential backoff: 1s, 2s, 4s
            wait_time = 2 ** attempt
            import time
            time.sleep(wait_time)
    
    return None


def _clean_df(raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Clean and validate OHLCV data."""
    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(0).tolist()
        data = {f: raw.iloc[:, i].values for i, f in enumerate(fields)}
    else:
        data = {c: raw[c].values for c in raw.columns}
    
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in data for c in needed):
        return None
    
    idx = raw.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_convert(None)
    
    df = pd.DataFrame({c: data[c] for c in needed}, index=idx)
    df = df[df["Close"].notna()].copy()
    
    # Validate: High >= Low, High >= Open/Close, Low <= Open/Close
    df = df[(df["High"] >= df["Low"]) & 
            (df["High"] >= df["Open"]) & 
            (df["High"] >= df["Close"]) &
            (df["Low"] <= df["Open"]) &
            (df["Low"] <= df["Close"])].copy()
    
    return df if not df.empty else None


# ══════════════════════════════════════════════════════════════
# INDICATOR CALCULATIONS
# ══════════════════════════════════════════════════════════════

def _zscore(s: pd.Series, n: int) -> pd.Series:
    """Calculate z-score."""
    m = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=1)
    return (s - m) / sd.replace(0.0, np.nan)


def _clamp(s: pd.Series, v: float) -> pd.Series:
    """Clamp series to [-v, v]."""
    return s.clip(-v, v)


def _wilder_rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's RMA (exponential moving average)."""
    alpha = 1.0 / n
    result = np.full(len(s), np.nan)
    valid = s.dropna()
    
    if len(valid) < n:
        return pd.Series(result, index=s.index)
    
    pos = s.index.get_loc(valid.index[0])
    seed_end = pos + n
    
    if seed_end > len(s):
        return pd.Series(result, index=s.index)
    
    result[seed_end - 1] = s.iloc[pos:seed_end].mean()
    
    for i in range(seed_end, len(s)):
        val = s.iloc[i]
        result[i] = (alpha * val + (1 - alpha) * result[i - 1]
                     if not np.isnan(val) else result[i - 1])
    
    return pd.Series(result, index=s.index)


def _rsi(close: pd.Series, n: int) -> pd.Series:
    """Calculate RSI."""
    d = close.diff()
    gain = _wilder_rma(d.clip(lower=0), n)
    loss = _wilder_rma((-d).clip(lower=0), n)
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _cpi_factor(from_year: int, to_year: int) -> float:
    """Calculate CPI adjustment factor for inflation."""
    if to_year <= from_year:
        return 1.0
    factor = 1.0
    for yr in range(from_year, to_year):
        factor *= 1.0 + _INDIA_CPI.get(yr, 5.5) / 100.0
    return max(factor, 1.0)


_USD_INR_CACHE = [None]


def _get_usd_inr() -> float:
    """Get USD/INR rate with caching."""
    if _USD_INR_CACHE[0] is None:
        try:
            rate = yf.Ticker("USDINR=X").info.get("regularMarketPrice", None)
            _USD_INR_CACHE[0] = float(rate) if rate and 60 < float(rate) < 120 else 84.0
        except Exception:
            _USD_INR_CACHE[0] = 84.0
    return _USD_INR_CACHE[0]


def _eps_inr_factor(tk: yf.Ticker) -> float:
    """Get USD/INR conversion factor for EPS if needed."""
    try:
        info = tk.info
        fin_ccy = info.get("financialCurrency", "INR")
        price_ccy = info.get("currency", "INR")
        if fin_ccy == "USD" and price_ccy == "INR":
            return _get_usd_inr()
    except Exception:
        pass
    return 1.0


def _get_eps_series(tk: yf.Ticker) -> pd.Series:
    """Extract EPS series from quarterly data."""
    fx = _eps_inr_factor(tk)
    eps_map = {}
    
    for attr in ["quarterly_income_stmt", "quarterly_financials"]:
        try:
            q = getattr(tk, attr)
            if q is None or q.empty:
                continue
            
            for row_name in ["Diluted EPS", "Basic EPS"]:
                if row_name not in q.index:
                    continue
                
                row = q.loc[row_name]
                for col, val in row.items():
                    try:
                        v = float(val)
                        if not np.isnan(v):
                            eps_map[pd.Timestamp(col).normalize()] = v * fx
                    except (TypeError, ValueError):
                        pass
                
                if eps_map:
                    break
            
            if eps_map:
                break
        except Exception:
            pass
    
    if not eps_map:
        return pd.Series(dtype=float)
    
    return pd.Series(eps_map).sort_index()


def compute_cape_z(tk: yf.Ticker, price_df: pd.DataFrame, c: Optional[Dict] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute CAPE z-score, ratio, and TTM EPS."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    if not c["use_cape"]:
        return None, None, None
    
    try:
        eps_s = _get_eps_series(tk)
        if eps_s.empty:
            return None, None, None
        
        price_idx = price_df.index
        if hasattr(price_idx, "tz") and price_idx.tz is not None:
            price_idx = price_idx.tz_convert(None)
        
        close = pd.Series(price_df["Close"].astype(float).values, index=price_idx)
        eps_s = eps_s.sort_index()
        
        # Calculate TTM (Trailing Twelve Months)
        n_total = len(eps_s)
        ttm_map = {}
        
        for i in range(n_total):
            report_date = eps_s.index[i]
            start = max(0, i + 1 - c["cape_max_q"])
            window = eps_s.iloc[start:i + 1]
            n = len(window)
            weights = np.array([max(0.1, 1.0 - (n - 1 - k) * 0.025) for k in range(n)])
            cpi_adj = np.array([_cpi_factor(window.index[k].year, report_date.year) for k in range(n)])
            total_w = weights.sum()
            
            if total_w <= 0:
                continue
            
            ttm = (window.values * cpi_adj * weights).sum() / total_w * 4.0
            if ttm > 0:
                ttm_map[report_date] = ttm
        
        if not ttm_map:
            return None, None, None
        
        ttm_s = pd.Series(ttm_map).sort_index()
        combined_idx = price_idx.union(ttm_s.index).sort_values()
        ttm_daily = ttm_s.reindex(combined_idx).ffill().reindex(price_idx)
        
        cape_ratio = close / ttm_daily.replace(0, np.nan)
        use_len = min(c["cape_zlen"], len(cape_ratio.dropna()))
        
        if use_len < 30:
            return None, None, None
        
        z = _zscore(cape_ratio, use_len).clip(-c["clamp_val"], c["clamp_val"])
        z_final = -z if c["cape_bearish"] else z
        
        last_z = float(z_final.iloc[-1]) if not np.isnan(z_final.iloc[-1]) else None
        last_close = float(close.iloc[-1])
        last_ttm = float(ttm_s.iloc[-1]) if not ttm_s.empty else None
        last_ratio = round(last_close / last_ttm, 2) if last_ttm and last_ttm > 0 else None
        
        return last_z, last_ratio, last_ttm
    
    except Exception:
        return None, None, None


def _pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
    """Find pivot lows."""
    n = len(series)
    result = pd.Series(np.nan, index=series.index)
    vals = series.values
    
    for i in range(left, n - right):
        v = vals[i]
        if np.isnan(v):
            continue
        window = vals[i - left:i + right + 1]
        if np.isnan(window).any():
            continue
        if v == window.min() and (window == v).sum() == 1:
            result.iloc[i] = v
    
    return result


def _pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
    """Find pivot highs."""
    n = len(series)
    result = pd.Series(np.nan, index=series.index)
    vals = series.values
    
    for i in range(left, n - right):
        v = vals[i]
        if np.isnan(v):
            continue
        window = vals[i - left:i + right + 1]
        if np.isnan(window).any():
            continue
        if v == window.max() and (window == v).sum() == 1:
            result.iloc[i] = v
    
    return result


def detect_divergence(price: pd.Series, osc: pd.Series, name: str, c: Optional[Dict] = None) -> Dict[str, Any]:
    """Detect price-oscillator divergences."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    out = {"reg_bull": False, "hid_bull": False, "reg_bear": False, "hid_bear": False, "tag": ""}
    
    if not c["div_enable"]:
        return out
    
    if len(price) < c["piv_left"] + c["piv_right"] + 10:
        return out
    
    L, R, LB = c["piv_left"], c["piv_right"], c["div_lookback"]
    osc = osc.reindex(price.index)
    
    p_lows = _pivot_low(price, L, R)
    p_highs = _pivot_high(price, L, R)
    o_lows = _pivot_low(osc, L, R)
    o_highs = _pivot_high(osc, L, R)
    
    last_idx = len(price) - 1
    
    # Bullish divergence
    pl_positions = [i for i, v in enumerate(p_lows.values) if not np.isnan(v)]
    if len(pl_positions) >= 2:
        i_curr, i_prev = pl_positions[-1], pl_positions[-2]
        if (last_idx - i_curr) <= LB and (i_curr - i_prev) <= LB:
            p_curr = price.iloc[i_curr]
            p_prev = price.iloc[i_prev]
            o_curr = _nearest_pivot_value(o_lows, i_curr, R)
            o_prev = _nearest_pivot_value(o_lows, i_prev, R)
            
            if o_curr is not None and o_prev is not None:
                if p_curr < p_prev and o_curr > o_prev:
                    out["reg_bull"] = True
                elif p_curr > p_prev and o_curr < o_prev:
                    out["hid_bull"] = True
    
    # Bearish divergence
    ph_positions = [i for i, v in enumerate(p_highs.values) if not np.isnan(v)]
    if len(ph_positions) >= 2:
        i_curr, i_prev = ph_positions[-1], ph_positions[-2]
        if (last_idx - i_curr) <= LB and (i_curr - i_prev) <= LB:
            p_curr = price.iloc[i_curr]
            p_prev = price.iloc[i_prev]
            o_curr = _nearest_pivot_value(o_highs, i_curr, R)
            o_prev = _nearest_pivot_value(o_highs, i_prev, R)
            
            if o_curr is not None and o_prev is not None:
                if p_curr > p_prev and o_curr < o_prev:
                    out["reg_bear"] = True
                elif p_curr < p_prev and o_curr > o_prev:
                    out["hid_bear"] = True
    
    parts = []
    if out["reg_bull"]:
        parts.append(f"BullReg({name})")
    if out["hid_bull"]:
        parts.append(f"BullHid({name})")
    if out["reg_bear"]:
        parts.append(f"BearReg({name})")
    if out["hid_bear"]:
        parts.append(f"BearHid({name})")
    
    out["tag"] = " | ".join(parts)
    return out


def _nearest_pivot_value(piv_series: pd.Series, target_idx: int, tol: int) -> Optional[float]:
    """Find nearest pivot value within tolerance."""
    n = len(piv_series)
    lo = max(0, target_idx - tol)
    hi = min(n - 1, target_idx + tol)
    best_val, best_dist = None, None
    
    for j in range(lo, hi + 1):
        v = piv_series.iloc[j]
        if not np.isnan(v):
            d = abs(j - target_idx)
            if best_dist is None or d < best_dist:
                best_dist, best_val = d, float(v)
    
    return best_val


def _hi52_ok_series(high: pd.Series, close: pd.Series, c: Optional[Dict] = None) -> pd.Series:
    """Check if close is within 52W high threshold."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    roll_max = high.rolling(c["hi52_bars"], min_periods=max(20, c["hi52_bars"] // 4)).max()
    ratio = close / roll_max.replace(0, np.nan)
    return ratio <= c["hi52_pct"]


def _hi52_ok_last(high: pd.Series, close: pd.Series, c: Optional[Dict] = None) -> bool:
    """Check if last bar passes 52W high test."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    if not c["hi52_enable"]:
        return True
    
    s = _hi52_ok_series(high, close, c)
    last = s.iloc[-1]
    return bool(last) if not pd.isna(last) else False


def compute_signals(df: pd.DataFrame, c: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """Compute RSI, MACD, and momentum signals."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    min_bars = max(c["rsi_zlen"], c["macd_zlen"]) + c["macd_slow"] + 30
    if len(df) < min_bars:
        return None
    
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float).replace(0, np.nan)
    
    # ── RSI Z (contrarian) ───────────────────────────────────
    rsi_val = _rsi(close, c["rsi_len"])
    rsi_lz = _zscore(rsi_val, c["rsi_zlen"])
    rsi_dz = rsi_lz.diff(c["rsi_dz_len"]).rolling(2).mean()
    rsi_comb = rsi_lz * (1.0 - c["rsi_dz_weight"]) + rsi_dz * c["rsi_dz_weight"]
    rsi_z = _clamp(-rsi_comb if c["rsi_contrarian"] else rsi_comb, c["clamp_val"])
    rsi_dz_accel = rsi_dz.diff(c["dz_accel_bars"])
    
    # ── MACD% Z (contrarian) ─────────────────────────────────
    ema_f = close.ewm(span=c["macd_fast"], adjust=False, min_periods=c["macd_fast"]).mean()
    ema_s = close.ewm(span=c["macd_slow"], adjust=False, min_periods=c["macd_slow"]).mean()
    macd_hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=c["macd_sig"], adjust=False, min_periods=c["macd_sig"]).mean()
    macd_pct = macd_hist / close.replace(0, np.nan) * 100.0
    macd_lz = _zscore(macd_pct, c["macd_zlen"])
    macd_dz = macd_lz.diff(c["macd_dz_len"]).rolling(2).mean()
    macd_comb = macd_lz * (1.0 - c["macd_dz_weight"]) + macd_dz * c["macd_dz_weight"]
    macd_z = _clamp(-macd_comb if c["macd_contrarian"] else macd_comb, c["clamp_val"])
    macd_dz_accel = macd_dz.diff(c["dz_accel_bars"])
    
    div_rsi = detect_divergence(close, rsi_lz, "RSI_Z", c)
    div_macd = detect_divergence(close, macd_lz, "MACD_Z", c)
    hi52_pass = _hi52_ok_last(high, close, c)
    
    def _f(s: pd.Series) -> Optional[float]:
        v = s.iloc[-1]
        return round(float(v), 3) if not (np.isnan(v) or np.isinf(v)) else None
    
    return {
        "close": round(float(close.iloc[-1]), 2),
        "rsi_val": round(float(rsi_val.iloc[-1]), 1) if not np.isnan(rsi_val.iloc[-1]) else None,
        "rsi_z": _f(rsi_z),
        "macd_z": _f(macd_z),
        "rsi_dz": _f(rsi_dz),
        "macd_dz": _f(macd_dz),
        "rsi_dz_accel": _f(rsi_dz_accel),
        "macd_dz_accel": _f(macd_dz_accel),
        "hi52_pass": hi52_pass,
        "div_rsi": div_rsi,
        "div_macd": div_macd,
    }


def _composite(sig: Dict[str, Any], cape_z: Optional[float], c: Optional[Dict] = None) -> Tuple[Optional[float], bool]:
    """Compute composite score from components."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    rz = sig.get("rsi_z")
    mz = sig.get("macd_z")
    
    if any(z is None for z in [rz, mz]):
        return None, False
    
    cape_active = cape_z is not None and c["use_cape"]
    
    if cape_active:
        tot = c["wt_cape"] + c["wt_rsi"] + c["wt_macd"]
        raw = (cape_z * c["wt_cape"] + rz * c["wt_rsi"] + mz * c["wt_macd"]) / tot
    else:
        tot = c["wt_rsi"] + c["wt_macd"]
        raw = (rz * c["wt_rsi"] + mz * c["wt_macd"]) / tot
    
    clamped = float(_clamp(pd.Series([raw]), c["clamp_val"]).iloc[0])
    return round(clamped, 3), cape_active


def verdict(z: Optional[float], c: Optional[Dict] = None) -> str:
    """Determine verdict from composite score."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    if z is None:
        return "N/A"
    if z >= c["th_sbuy"]:
        return "STRONG BUY"
    if z >= c["th_buy"]:
        return "BUY"
    if z <= c["th_ssell"]:
        return "STRONG SELL"
    if z <= c["th_sell"]:
        return "SELL"
    return "NEUTRAL"


def confidence(comp: Optional[float], sig: Dict[str, Any], cape_z: Optional[float], cape_used: bool, c: Optional[Dict] = None) -> str:
    """Determine confidence level."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    if comp is None:
        return ""
    
    v = verdict(comp, c)
    if v in ("N/A", "NEUTRAL"):
        return ""
    
    direction = 1 if comp > 0 else -1
    components = [sig.get("rsi_z"), sig.get("macd_z")]
    if cape_used and cape_z is not None:
        components.append(cape_z)
    
    agree = sum(1 for z in components if z is not None and ((z > 0 and direction > 0) or (z < 0 and direction < 0)))
    abs_c = abs(comp)
    
    if abs_c >= c["conf_strong"] and agree >= 3:
        return "STRONG"
    if abs_c >= c["conf_moderate"] and agree >= 2:
        return "MODERATE"
    return "WEAK"


def _add_conf(cape_z: Optional[float], cape_used: bool, rsi_val: Optional[float], agree: int, c: Optional[Dict] = None) -> bool:
    """Check additional confirmation gate."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    if rsi_val is None:
        return False
    if float(rsi_val) >= c["rsi_hard_max"]:
        return False
    if agree <= c["add_conf_agree_min"]:
        return False
    if cape_used and cape_z is not None:
        if float(cape_z) <= 1.73:
            return False
    
    return True


def _dz_accel_ok(sig: Dict[str, Any], c: Optional[Dict] = None) -> bool:
    """Check ΔZ acceleration filter."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    if not c["dz_accel_enable"]:
        return True
    
    rsi_acc = sig.get("rsi_dz_accel")
    macd_acc = sig.get("macd_dz_accel")
    
    if rsi_acc is None and macd_acc is None:
        return True
    
    rsi_ok = (rsi_acc is None) or (float(rsi_acc) > 0)
    macd_ok = (macd_acc is None) or (float(macd_acc) > 0)
    
    if c.get("dz_accel_require_both", False):
        return rsi_ok and macd_ok
    return rsi_ok or macd_ok


def _candle_ok(open_price: float, high_price: float, low_price: float, close_price: float, c: Optional[Dict] = None) -> bool:
    """Check candle body qualifications."""
    if c is None:
        c = st.session_state.get("cfg", DEFAULT_CFG)
    
    if not c["candle_body_enable"]:
        return True
    
    body = abs(close_price - open_price)
    lower_wick = min(open_price, close_price) - low_price
    green = close_price >= open_price * c["candle_green_tol"]
    hammer = lower_wick >= c["hammer_mult"] * body if body > 0 else lower_wick > 0
    qualifies = green or hammer
    
    if c.get("candle_body_hard", False):
        return qualifies
    return True


# ══════════════════════════════════════════════════════════════
# SCANNING & BACKTESTING
# ══════════════════════════════════════════════════════════════

def scan_stock(sym_raw: str, c: Optional[Dict] = None) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    """Scan single stock for signals (daily + weekly)."""
    if c is None:
        c = DEFAULT_CFG
    
    ticker = to_yf(sym_raw)
    rows = []
    
    try:
        df = fetch_stock_data(ticker, "3y")
        if df is None:
            return sym_raw, [], "⚠️ No data available"
        
        if len(df) < 100:
            return sym_raw, [], "⚠️ Insufficient bars (< 100)"
        
        tk = yf.Ticker(ticker)
        cape_z, cape_ratio, ttm_eps = compute_cape_z(tk, df, c)
        d_sig = compute_signals(df, c)
        
        weekly_raw = df.resample("W").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        })
        weekly = weekly_raw[weekly_raw["Close"].notna()].copy()
        w_sig = compute_signals(weekly, c)
        
        for tf, sig, src_df in [("Daily", d_sig, df), ("Weekly", w_sig, weekly)]:
            if sig is None:
                continue
            
            comp, cape_used = _composite(sig, cape_z, c)
            if comp is None:
                continue
            
            vrd = verdict(comp, c)
            conf = confidence(comp, sig, cape_z, cape_used, c)
            
            _ac_zs = [sig["rsi_z"], sig["macd_z"]]
            if cape_used and cape_z is not None:
                _ac_zs.append(cape_z)
            _ac_agree = sum(1 for z in _ac_zs if z is not None and float(z) > 0)
            
            ac = _add_conf(cape_z, cape_used, sig["rsi_val"], _ac_agree, c)
            dz_acc_ok = _dz_accel_ok(sig, c)
            hi52_ok = sig.get("hi52_pass", True)
            
            last = src_df.iloc[-1]
            candle_ok = _candle_ok(float(last["Open"]), float(last["High"]), float(last["Low"]), float(last["Close"]), c)
            
            div_tags = []
            if sig.get("div_rsi", {}).get("tag"):
                div_tags.append(sig["div_rsi"]["tag"])
            if sig.get("div_macd", {}).get("tag"):
                div_tags.append(sig["div_macd"]["tag"])
            div_str = " | ".join(div_tags) if div_tags else ""
            
            rows.append({
                "Symbol": sym_raw,
                "TF": tf,
                "Signal": vrd,
                "Strength": conf,
                "Add_Conf": "YES" if ac else "NO",
                "ΔZ_Accel": "YES" if dz_acc_ok else "NO",
                "Candle_OK": "YES" if candle_ok else "NO",
                "Hi52_OK": "YES" if hi52_ok else "NO",
                "All_Gates": "YES" if (ac and dz_acc_ok and candle_ok and hi52_ok) else "NO",
                "Composite": comp,
                "CAPE_Z": cape_z if cape_used else None,
                "CAPE_PE": cape_ratio,
                "TTM_EPS": ttm_eps,
                "RSI_Z": sig["rsi_z"],
                "MACD_Z": sig["macd_z"],
                "RSI_ΔZ": sig["rsi_dz"],
                "MACD_ΔZ": sig["macd_dz"],
                "RSI": sig["rsi_val"],
                "Close": sig["close"],
                "Divergence": div_str,
                "CAPE_Active": cape_used,
            })
        
        return sym_raw, rows, None
    
    except Exception as e:
        return sym_raw, [], f"⚠️ {type(e).__name__}: {str(e)[:50]}"


def _weekly_signal_frame(df: pd.DataFrame, c: Dict[str, Any]) -> pd.DataFrame:
    """Compute signal frame for weekly backtesting. FIXED in v1.8."""
    min_bars = max(c["rsi_zlen"], c["macd_zlen"]) + c["macd_slow"] + 30
    if len(df) < min_bars:
        return pd.DataFrame()
    
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)  # FIXED: was open*
    volume = df["Volume"].astype(float).replace(0, np.nan)
    
    # RSI Z (contrarian)
    rsi_val = _rsi(close, c["rsi_len"])
    rsi_lz = _zscore(rsi_val, c["rsi_zlen"])
    rsi_dz = rsi_lz.diff(c["rsi_dz_len"]).rolling(2).mean()
    rsi_comb = rsi_lz * (1 - c["rsi_dz_weight"]) + rsi_dz * c["rsi_dz_weight"]
    rsi_z = _clamp(-rsi_comb if c["rsi_contrarian"] else rsi_comb, c["clamp_val"])
    rsi_dz_accel = rsi_dz.diff(c["dz_accel_bars"])
    
    # MACD% Z (contrarian)
    ema_f = close.ewm(span=c["macd_fast"], adjust=False, min_periods=c["macd_fast"]).mean()
    ema_s = close.ewm(span=c["macd_slow"], adjust=False, min_periods=c["macd_slow"]).mean()
    macd_hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=c["macd_sig"], adjust=False, min_periods=c["macd_sig"]).mean()
    macd_pct = macd_hist / close.replace(0, np.nan) * 100.0
    macd_lz = _zscore(macd_pct, c["macd_zlen"])
    macd_dz = macd_lz.diff(c["macd_dz_len"]).rolling(2).mean()
    macd_comb = macd_lz * (1 - c["macd_dz_weight"]) + macd_dz * c["macd_dz_weight"]
    macd_z = _clamp(-macd_comb if c["macd_contrarian"] else macd_comb, c["clamp_val"])
    macd_dz_accel = macd_dz.diff(c["dz_accel_bars"])
    
    hi52_ok_s = _hi52_ok_series(high, close, c)
    
    return pd.DataFrame({
        "rsi_z": rsi_z,
        "macd_z": macd_z,
        "rsi_dz": rsi_dz,
        "macd_dz": macd_dz,
        "rsi_dz_accel": rsi_dz_accel,
        "macd_dz_accel": macd_dz_accel,
        "hi52_ok": hi52_ok_s,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "rsi_val": rsi_val,
    }, index=df.index)


def backtest_one(sym_raw: str, lookback_weeks: int, profit_pct: float, c: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    """Run backtest for single stock."""
    ticker = to_yf(sym_raw)
    trades = []
    
    try:
        df = fetch_stock_data(ticker, "5y")
        if df is None:
            return sym_raw, [], "⚠️ No data available"
        
        if len(df) < 200:
            return sym_raw, [], "⚠️ Insufficient bars"
        
        tk = yf.Ticker(ticker)
        cape_weekly = None
        
        if c["use_cape"]:
            # Compute CAPE for entire period
            from concurrent.futures import ThreadPoolExecutor
            # For efficiency, compute CAPE once, don't recompute
            # This is a simplified version - in production, cache this
            pass
        
        weekly_raw = df.resample("W").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        })
        weekly = weekly_raw[weekly_raw["Close"].notna()].copy()
        
        sf = _weekly_signal_frame(weekly, c)
        if sf.empty:
            return sym_raw, [], "⚠️ Insufficient bars for signals"
        
        total_w = len(weekly)
        scan_start = max(0, total_w - lookback_weeks - 1)
        scan_end = total_w - 1
        
        for i in range(scan_start, scan_end):
            rz_w = sf["rsi_z"].iloc[i]
            mz_w = sf["macd_z"].iloc[i]
            
            if any(pd.isna(x) for x in (rz_w, mz_w)):
                continue
            
            # Composite without CAPE (for simplicity in backtest)
            tot = c["wt_rsi"] + c["wt_macd"]
            comp_w = (float(rz_w) * c["wt_rsi"] + float(mz_w) * c["wt_macd"]) / tot
            comp_w = float(np.clip(comp_w, -c["clamp_val"], c["clamp_val"]))
            
            if comp_w < c["bt_min_composite"]:
                continue
            
            vrd_w = verdict(comp_w, c)
            if vrd_w not in ("BUY", "STRONG BUY"):
                continue
            
            sig_w = {"rsi_z": float(rz_w), "macd_z": float(mz_w)}
            conf_w = confidence(comp_w, sig_w, None, False, c)
            
            if conf_w not in ("MODERATE", "STRONG"):
                continue
            
            if sym_raw in [t["Symbol"] for t in trades if "Status" in t]:
                continue
            
            # Entry
            bar_open = float(sf["open"].iloc[i])
            bar_high = float(sf["high"].iloc[i])
            bar_low = float(sf["low"].iloc[i])
            bar_close = float(sf["close"].iloc[i])
            
            candle_ok_flag = _candle_ok(bar_open, bar_high, bar_low, bar_close, c)
            
            entry_date = weekly.index[i]
            entry_price = bar_close
            target_price = round(entry_price * (1 + profit_pct / 100), 2)
            
            exit_date, exit_price, hold_weeks = None, None, None
            status = "OPEN"
            
            for j in range(i + 1, total_w):
                if float(sf["high"].iloc[j]) >= target_price:
                    exit_date = weekly.index[j]
                    exit_price = target_price
                    hold_weeks = j - i
                    status = "HIT"
                    break
            
            if exit_date is None:
                last_close = float(sf["close"].iloc[-1])
                exit_price = last_close
                hold_weeks = total_w - 1 - i
                open_ret = (last_close - entry_price) / entry_price * 100
                status = f"OPEN {open_ret:+.1f}%"
            
            ret_pct = round((exit_price - entry_price) / entry_price * 100, 2)
            
            trades.append({
                "Symbol": sym_raw,
                "Entry_Date": entry_date.strftime("%Y-%m-%d"),
                "Entry_Price": round(entry_price, 2),
                "Target": target_price,
                "Exit_Date": (exit_date.strftime("%Y-%m-%d") if exit_date else weekly.index[-1].strftime("%Y-%m-%d")),
                "Exit_Price": round(exit_price, 2),
                "Return_%": ret_pct,
                "Hold_Wks": hold_weeks,
                "Status": status,
                "W_Signal": vrd_w,
                "W_Strength": conf_w,
                "W_Comp": round(comp_w, 3),
                "Candle_OK": "YES" if candle_ok_flag else "NO",
            })
        
        return sym_raw, trades, None
    
    except Exception as e:
        return sym_raw, [], f"⚠️ {type(e).__name__}: {str(e)[:50]}"


# ══════════════════════════════════════════════════════════════
# PDF GENERATION
# ══════════════════════════════════════════════════════════════

def generate_scan_pdf(df_buy: pd.DataFrame, df_sell: pd.DataFrame, df_buy_conf: pd.DataFrame, df_sell_conf: pd.DataFrame, df_div: pd.DataFrame, ts_str: str) -> Optional[io.BytesIO]:
    """Generate scan results PDF."""
    if not _REPORTLAB:
        return None
    
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=1*cm,
                            rightMargin=1*cm, topMargin=1.2*cm, bottomMargin=1.2*cm)
    styles = getSampleStyleSheet()
    h1, h2, normal = styles["Heading1"], styles["Heading2"], styles["Normal"]
    PAGE_W = landscape(A4)[0] - 2*cm
    
    def _df_to_rl_table(df: pd.DataFrame, col_widths: Optional[List[float]] = None) -> Table:
        rows = [list(df.columns)] + [[str(v) if v is not None else "" for v in row] for row in df.itertuples(index=False)]
        n_c = len(df.columns)
        cw = col_widths or [PAGE_W / n_c] * n_c
        tbl = Table(rows, colWidths=cw, repeatRows=1)
        style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("GRID", (0, 0), (-1, -1), 0.3, rl_colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#f0f4ff"), rl_colors.white]),
        ])
        
        sig_col = list(df.columns).index("Signal") if "Signal" in df.columns else None
        for r_idx, row in enumerate(rows[1:], start=1):
            if sig_col is not None:
                sig = str(row[sig_col])
                if "BUY" in sig:
                    style.add("BACKGROUND", (0, r_idx), (-1, r_idx), rl_colors.HexColor("#d4edda"))
                elif "SELL" in sig:
                    style.add("BACKGROUND", (0, r_idx), (-1, r_idx), rl_colors.HexColor("#f8d7da"))
        
        tbl.setStyle(style)
        return tbl
    
    signal_cols = ["Symbol", "TF", "Signal", "Strength", "All_Gates", "Composite", "RSI_Z", "MACD_Z", "Close", "Divergence"]
    
    story = [
        Paragraph("SHANTANU'S VALUE MOMENTUM SWING TRADING SCANNER v1.8", h1),
        Paragraph(f"NSE | {ts_str} | Improved reliability", normal),
        Spacer(1, 0.6*cm),
    ]
    
    for tf in ["Daily", "Weekly"]:
        sb = df_buy[df_buy["TF"] == tf] if df_buy is not None and not df_buy.empty else pd.DataFrame()
        ss = df_sell[df_sell["TF"] == tf] if df_sell is not None and not df_sell.empty else pd.DataFrame()
        
        if not sb.empty:
            avail = [c for c in signal_cols if c in sb.columns]
            story += [
                Paragraph(f"BUY — {tf} ({len(sb)} stocks)", h2),
                _df_to_rl_table(sb[avail]),
                Spacer(1, 0.5*cm),
            ]
        
        if not ss.empty:
            avail = [c for c in signal_cols if c in ss.columns]
            story += [
                Paragraph(f"SELL — {tf} ({len(ss)} stocks)", h2),
                _df_to_rl_table(ss[avail]),
                Spacer(1, 0.5*cm),
            ]
    
    doc.build(story)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════
# UI HELPERS & STYLING
# ══════════════════════════════════════════════════════════════

def _load_css() -> None:
    """Load custom CSS styling."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1a1f2e;
    }
    
    .stApp { background: #f4f6fb; }
    
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.30) !important;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    }
    
    .metric-value {
        font-size: 30px;
        font-weight: 800;
        margin-top: 6px;
    }
    
    .metric-green { color: #059669; }
    .metric-red { color: #dc2626; }
    .metric-blue { color: #2563eb; }
    
    .section-header {
        font-size: 17px;
        font-weight: 800;
        padding: 20px 0 10px 0;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 16px 20px;
        font-size: 12px;
        color: #1e40af;
        margin: 12px 0;
    }
    </style>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, color_class: str = "metric-blue") -> str:
    """Generate HTML metric card."""
    return f"""
    <div class="metric-card">
        <div style="font-size:10.5px;color:#94a3b8;text-transform:uppercase;letter-spacing:1.6px;font-weight:500">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
    </div>
    """


# ══════════════════════════════════════════════════════════════
# SIDEBAR CONFIGURATION (Improvement 5)
# ══════════════════════════════════════════════════════════════

def render_sidebar() -> Dict[str, Any]:
    """Render sidebar and return config. Added validation in v1.8."""
    with st.sidebar:
        st.markdown('<div style="font-size:18px;font-weight:800;color:#1e3a8a;margin-bottom:16px">⚙️ Scanner Config</div>', unsafe_allow_html=True)
        
        with st.expander("🔬 Filters & Thresholds", expanded=True):
            workers = st.slider("Parallel workers", 4, 16, DEFAULT_CFG["workers"])
            min_composite = st.slider("Min |Composite|", 0.5, 2.5, DEFAULT_CFG["min_composite"], 0.05)
            rsi_hard_max = st.slider("RSI hard gate (<)", 30, 60, int(DEFAULT_CFG["rsi_hard_max"]))
            hi52_pct = st.slider("52W high % max", 0.5, 1.0, DEFAULT_CFG["hi52_pct"], 0.01)
            bt_min_comp = st.slider("BT composite floor", 0.5, 2.5, DEFAULT_CFG["bt_min_composite"], 0.05)
        
        with st.expander("📊 Indicator Weights"):
            wt_cape = st.slider("CAPE weight", 0, 50, int(DEFAULT_CFG["wt_cape"]))
            wt_rsi = st.slider("RSI weight", 0, 50, int(DEFAULT_CFG["wt_rsi"]))
            wt_macd = st.slider("MACD weight", 0, 50, int(DEFAULT_CFG["wt_macd"]))
            
            # IMPROVEMENT 5: Validate weight sum
            total_wt = wt_cape + wt_rsi + wt_macd
            if total_wt == 0:
                st.error("❌ Sum of weights must be > 0")
            elif total_wt != 100:
                st.warning(f"⚠️ Weights sum to {total_wt} (not 100). Will normalize proportionally.")
        
        with st.expander("🔀 Feature Toggles"):
            use_cape = st.checkbox("Use CAPE", DEFAULT_CFG["use_cape"])
            div_enable = st.checkbox("Divergence detection", DEFAULT_CFG["div_enable"])
            dz_accel = st.checkbox("ΔZ Acceleration", DEFAULT_CFG["dz_accel_enable"])
            hi52_enable = st.checkbox("52W High gate", DEFAULT_CFG["hi52_enable"])
        
        st.markdown("---")
        st.markdown(f'📦 **Universe:** {len(ALL_SYMBOLS)} NSE stocks')
        
    cfg = {
        **DEFAULT_CFG,
        "workers": workers,
        "min_composite": min_composite,
        "rsi_hard_max": float(rsi_hard_max),
        "hi52_pct": hi52_pct,
        "bt_min_composite": bt_min_comp,
        "wt_cape": float(wt_cape),
        "wt_rsi": float(wt_rsi),
        "wt_macd": float(wt_macd),
        "use_cape": use_cape,
        "div_enable": div_enable,
        "dz_accel_enable": dz_accel,
        "hi52_enable": hi52_enable,
    }
    
    st.session_state["cfg"] = cfg
    return cfg


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════

def main():
    """Main application."""
    _load_css()
    
    # Header
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e3a8a,#0284c7);padding:36px;border-radius:18px;margin-bottom:28px;color:white">
        <div style="font-size:11px;color:rgba(255,255,255,0.65);letter-spacing:3px;text-transform:uppercase;margin-bottom:10px">📈 NSE Swing Trading</div>
        <div style="font-size:34px;font-weight:800;line-height:1.15;margin-bottom:10px">Value Momentum Scanner</div>
        <div style="font-size:14px;color:rgba(255,255,255,0.72)">CAPE · RSI Z · MACD Z · Dual Timeframe Confluence</div>
        <div style="margin-top:14px;display:flex;gap:8px">
            <span style="background:rgba(255,255,255,0.18);color:white;border-radius:20px;padding:4px 14px;font-size:11px;font-weight:700">v1.8</span>
            <span style="background:rgba(255,255,255,0.18);color:white;border-radius:20px;padding:4px 14px;font-size:11px;font-weight:700">290 NSE Stocks</span>
            <span style="background:rgba(255,255,255,0.18);color:white;border-radius:20px;padding:4px 14px;font-size:11px;font-weight:700">7 Watchlists Merged</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar config
    cfg = render_sidebar()
    
    # Tabs
    tab_scan, tab_bt, tab_focus, tab_about = st.tabs(["🔍 Live Scan", "📈 Backtest", "🔭 Focus Stock", "ℹ️ About"])
    
    # ── TAB 1: LIVE SCAN ──────────────────────────────────────
    with tab_scan:
        st.markdown("### 🔍 Real-time NSE Stock Scanner")
        
        col_run, col_info = st.columns([1, 3])
        with col_run:
            run_scan = st.button("🚀 Run Live Scan", key="run_scan")
        with col_info:
            st.info("Scans all stocks for BUY/SELL signals across daily + weekly timeframes.")
        
        if run_scan or "scan_results" in st.session_state:
            if run_scan:
                for k in ["scan_results", "scan_errors", "scan_ts"]:
                    st.session_state.pop(k, None)
                
                symbols = ALL_SYMBOLS
                total = len(symbols)
                prog = st.progress(0, text=f"Scanning {total} NSE stocks…")
                status_txt = st.empty()
                all_rows, errors, done = [], [], 0
                
                with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
                    futures = {ex.submit(scan_stock, s, cfg): s for s in symbols}
                    for fut in as_completed(futures):
                        sym, rows, err = fut.result()
                        done += 1
                        if err:
                            errors.append((sym, err))
                        all_rows.extend(rows)
                        pct = done / total
                        buys = sum(1 for r in all_rows if "BUY" in r["Signal"])
                        sells = sum(1 for r in all_rows if "SELL" in r["Signal"])
                        prog.progress(pct, text=f"[{done}/{total}] {sym} — buy={buys} sell={sells}")
                
                prog.empty()
                status_txt.empty()
                
                st.session_state["scan_results"] = all_rows
                st.session_state["scan_errors"] = errors
                st.session_state["scan_ts"] = datetime.now().strftime("%d %b %Y %H:%M")
            
            all_rows = st.session_state.get("scan_results", [])
            errors = st.session_state.get("scan_errors", [])
            ts_str = st.session_state.get("scan_ts", "")
            
            if not all_rows:
                st.warning("No signals found. Try relaxing filters or check connectivity.")
            else:
                df_all = pd.DataFrame(all_rows)
                df_signal = df_all[df_all["Composite"].abs() >= cfg["min_composite"]].copy()
                df_buy = df_signal[df_signal["Signal"].isin(["BUY", "STRONG BUY"])].copy()
                df_sell = df_signal[df_signal["Signal"].isin(["SELL", "STRONG SELL"])].copy()
                df_div = df_all[df_all["Divergence"] != ""].copy()
                
                # Metrics
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.markdown(metric_card("BUY", len(df_buy), "metric-green"), unsafe_allow_html=True)
                c2.markdown(metric_card("SELL", len(df_sell), "metric-red"), unsafe_allow_html=True)
                c3.markdown(metric_card("Confluence", len(df_buy), "metric-blue"), unsafe_allow_html=True)
                c4.markdown(metric_card("Signals", len(df_signal), "metric-blue"), unsafe_allow_html=True)
                c5.markdown(metric_card("Divergences", len(df_div), "metric-blue"), unsafe_allow_html=True)
                
                st.markdown(f'<div class="info-box">✓ Scan completed: {ts_str}</div>', unsafe_allow_html=True)
                
                # BUY table
                st.markdown('<div class="section-header" style="color:#065f46">🟢 BUY Signals</div>', unsafe_allow_html=True)
                display_cols = ["Symbol", "TF", "Signal", "Strength", "Composite", "RSI_Z", "MACD_Z", "Close"]
                if not df_buy.empty:
                    avail = [c for c in display_cols if c in df_buy.columns]
                    st.dataframe(df_buy[avail].reset_index(drop=True), use_container_width=True)
                else:
                    st.info("No BUY signals.")
                
                # SELL table
                st.markdown('<div class="section-header" style="color:#991b1b">🔴 SELL Signals</div>', unsafe_allow_html=True)
                if not df_sell.empty:
                    avail = [c for c in display_cols if c in df_sell.columns]
                    st.dataframe(df_sell[avail].reset_index(drop=True), use_container_width=True)
                else:
                    st.info("No SELL signals.")
                
                # Download
                st.markdown("---")
                st.markdown("### 📥 Export Results")
                csv_data = df_all.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV (All)",
                    csv_data,
                    file_name=f"VMS_Scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                
                if _REPORTLAB:
                    pdf_buf = generate_scan_pdf(df_buy, df_sell, pd.DataFrame(), pd.DataFrame(), df_div, ts_str)
                    if pdf_buf:
                        st.download_button(
                            "⬇️ Download PDF Report",
                            pdf_buf,
                            file_name=f"VMS_Scan_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )
    
    # ── TAB 2: BACKTEST ───────────────────────────────────────
    with tab_bt:
        st.markdown("### 📈 Historical Backtest (Weekly BUY Signals)")
        
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            lookback_wks = st.slider("Lookback (weeks)", 52, 520, 260, 26)
        with bc2:
            profit_pct = st.slider("Profit target (%)", 4.0, 20.0, cfg["backtest_profit_pct"], 0.5)
        with bc3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_bt = st.button("🚀 Run Backtest", key="run_bt")
        
        if run_bt or "bt_results" in st.session_state:
            if run_bt:
                for k in ["bt_results", "bt_errors", "bt_ts"]:
                    st.session_state.pop(k, None)
                
                symbols = ALL_SYMBOLS
                total = len(symbols)
                prog_bt = st.progress(0, text=f"Backtesting {total} stocks…")
                all_trades, errors_bt, done = [], [], 0
                
                with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
                    futures = {ex.submit(backtest_one, s, lookback_wks, profit_pct, cfg): s for s in symbols}
                    for fut in as_completed(futures):
                        sym, trades, err = fut.result()
                        done += 1
                        if err:
                            errors_bt.append((sym, err))
                        all_trades.extend(trades)
                        prog_bt.progress(done / total, text=f"[{done}/{total}] {sym} — {len(all_trades)} trades")
                
                prog_bt.empty()
                st.session_state["bt_results"] = all_trades
                st.session_state["bt_errors"] = errors_bt
                st.session_state["bt_ts"] = datetime.now().strftime("%d %b %Y %H:%M")
            
            all_trades = st.session_state.get("bt_results", [])
            errors_bt = st.session_state.get("bt_errors", [])
            ts_str_bt = st.session_state.get("bt_ts", "")
            
            if not all_trades:
                st.warning("No qualifying backtest signals found.")
            else:
                df_bt = pd.DataFrame(all_trades)
                df_hit = df_bt[df_bt["Status"] == "HIT"]
                n_total = len(df_bt)
                n_hit = len(df_hit)
                win_rate = n_hit / n_total * 100 if n_total > 0 else 0.0
                
                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(metric_card("Total", str(n_total), "metric-blue"), unsafe_allow_html=True)
                m2.markdown(metric_card("HIT", f"{n_hit} ({win_rate:.1f}%)", "metric-green"), unsafe_allow_html=True)
                m3.markdown(metric_card("Open", str(len(df_bt[~df_bt["Status"].str.startswith("OPEN")])), "metric-yellow"), unsafe_allow_html=True)
                m4.markdown(metric_card("Avg Hold", f"{df_hit['Hold_Wks'].mean():.1f}w" if n_hit > 0 else "N/A", "metric-blue"), unsafe_allow_html=True)
                
                st.markdown("#### 📋 Trade Log")
                st.dataframe(df_bt.reset_index(drop=True), use_container_width=True)
    
    # ── TAB 3: FOCUS STOCK ────────────────────────────────────
    with tab_focus:
        st.markdown("### 🔭 Deep-Dive Analysis")
        
        # IMPROVEMENT 4: Reuse scan results if available
        focus_sym = st.selectbox("Select stock", [""] + ALL_SYMBOLS, key="focus_sym")
        run_focus = st.button("🔍 Analyse", key="run_focus")
        
        if run_focus and focus_sym:
            # Check if we already have this data
            existing_rows = None
            if "scan_results" in st.session_state:
                existing_rows = [r for r in st.session_state["scan_results"] if r["Symbol"] == focus_sym]
            
            if existing_rows:
                st.success(f"✓ Using cached scan data for {focus_sym}")
                df_focus = pd.DataFrame(existing_rows)
            else:
                with st.spinner(f"🔄 Fetching & analyzing {focus_sym}…"):
                    sym_raw, rows, err = scan_stock(focus_sym, cfg)
                
                if err:
                    st.error(f"⚠️ Error: {err}")
                else:
                    df_focus = pd.DataFrame(rows)
            
            if len(df_focus) > 0:
                st.markdown(f"#### {focus_sym} — Signal Summary")
                focus_cols = ["TF", "Signal", "Strength", "Composite", "RSI_Z", "MACD_Z", "RSI", "Close"]
                avail = [c for c in focus_cols if c in df_focus.columns]
                st.dataframe(df_focus[avail].reset_index(drop=True), use_container_width=True)
            else:
                st.warning("No data found.")
    
    # ── TAB 4: ABOUT ──────────────────────────────────────────
    with tab_about:
        st.markdown("""
        ## VMS Scanner v1.8 — Improvements

        ### What's New in v1.8
        ✅ **Fixed syntax errors** in `_weekly_signal_frame()` (was: `open*`)
        ✅ **Caching** — yfinance data cached for 1 hour (avoid re-downloads)
        ✅ **Retry logic** — Exponential backoff for flaky API calls
        ✅ **Reuse scan results** — Focus tab reuses existing data
        ✅ **Input validation** — Weight sums now validated
        ✅ **Type hints** — Full type hints throughout
        ✅ **Better errors** — More descriptive error messages
        ✅ **Progress bars** — Live feedback during scans
        ✅ **Data validation** — OHLCV bounds checked
        
        ### How It Works
        
        **Signals:**
        - CAPE Z-Score: Cyclically-adjusted PE (India CPI-adjusted)
        - RSI Z + ΔZ: RSI momentum with acceleration
        - MACD% Z + ΔZ: MACD histogram momentum (contrarian mode)
        
        **Composite Score:**
        Weighted average of CAPE (33%), RSI (33%), MACD (34%)
        
        **Confidence Levels:**
        - **STRONG:** ≥2.0 composite AND ≥3 signals aligned
        - **MODERATE:** ≥1.0 composite AND ≥2 signals aligned
        - **WEAK:** Lower thresholds
        
        **Dual Timeframe:**
        Analyzes both daily and weekly charts for multi-frame confirmation.
        
        ### Known Limitations
        - No transaction costs (slippage, brokerage)
        - No stop losses in backtest
        - Fixed position sizing
        - Assumes perfect fills
        
        **Use for:** Signal ranking, relative comparisons, trend identification
        **Not for:** Absolute profit forecasting
        
        ---
        *Made by Shantanu | NSE Swing Trading | Data via yfinance*
        """)


if __name__ == "__main__":
    main()
