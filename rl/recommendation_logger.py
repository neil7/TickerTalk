"""
RecommendationLogger — appends (state, action, reward) records to JSONL.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_DIR = Path("logs")
DEFAULT_LOG = LOG_DIR / "recommendations.jsonl"


class RecommendationLogger:
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = Path(log_path) if log_path else DEFAULT_LOG
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def log(self, state: Dict[str, Any], signal_result: Dict[str, Any]) -> str:
        record = self._build_record(state, signal_result)
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(record, default=_json_serial) + "\n")
        return record["record_id"]

    def _build_record(self, state: Dict, sig: Dict) -> Dict:
        ticker = state.get("ticker", "UNKNOWN")
        ts = datetime.now(timezone.utc)
        unix_ts = int(ts.timestamp())

        fr = state.get("final_recommendation", {})
        action_data = fr.get("action", {})
        scores = fr.get("scores", {})
        price_targets = fr.get("price_targets", {})
        price = state.get("market_data", {}).get("current_price")

        vector = sig.get("vector")
        feature_list = vector.tolist() if hasattr(vector, "tolist") else list(vector or [])

        return {
            "record_id": f"{ticker}_{unix_ts}_{uuid.uuid4().hex[:6]}",
            "ticker": ticker,
            "timestamp_utc": ts.isoformat(),
            "unix_ts": unix_ts,
            "recommendation": action_data.get("recommendation", "HOLD"),
            "conviction": action_data.get("conviction"),
            "time_horizon": action_data.get("time_horizon"),
            "composite_score": scores.get("composite_score"),
            "price_at_decision": price,
            "feature_version": sig.get("version", "1.0"),
            "feature_vector": feature_list,
            "data_quality": sig.get("data_quality", 0.0),
            "missing_agents": sig.get("missing", []),
            "features": sig.get("features", {}),
            "risk_reward_at_entry": price_targets.get("risk_reward_ratio"),
            "atr_at_entry": price_targets.get("atr"),
            "stop_loss_pct": price_targets.get("stop_loss_pct"),
            "outcome_fetched": False,
            "price_at_1d": None,
            "price_at_5d": None,
            "price_at_10d": None,
            "return_5d": None,
            "return_10d": None,
            "reward_5d_sharpe": None,
            "reward_calmar": None,
            "reward_10d_sharpe": None,
            "reward_regime_adj": None,
            "reward_rr_weighted": None,
        }

    # ------------------------------------------------------------------
    # Labelling
    # ------------------------------------------------------------------
    def fetch_and_label_outcomes(self, min_days: int = 5) -> int:
        if not self.log_path.exists():
            return 0

        records = self._load_all()
        now_ts = int(datetime.now(timezone.utc).timestamp())
        labelled = 0

        for rec in records:
            if rec.get("outcome_fetched"):
                continue
            age_days = (now_ts - rec.get("unix_ts", now_ts)) / 86400
            if age_days < min_days:
                continue
            try:
                self._label_record(rec)
                labelled += 1
            except Exception:
                pass

        self._save_all(records)
        return labelled

    def _label_record(self, rec: Dict) -> None:
        import yfinance as yf

        ticker = rec["ticker"]
        start = datetime.fromtimestamp(rec["unix_ts"], tz=timezone.utc)

        hist = yf.download(ticker, start=start, period="15d", interval="1d",
                           progress=False, auto_adjust=True)
        if hist.empty or len(hist) < 2:
            return

        prices = hist["Close"].dropna()
        entry = rec.get("price_at_decision") or float(prices.iloc[0])

        if len(prices) >= 2:
            rec["price_at_1d"] = round(float(prices.iloc[1]), 4)
        if len(prices) >= 6:
            rec["price_at_5d"] = round(float(prices.iloc[5]), 4)
            ret5 = float((prices.iloc[5] - entry) / entry)
            rec["return_5d"] = round(ret5, 6)
        if len(prices) >= 11:
            rec["price_at_10d"] = round(float(prices.iloc[10]), 4)
            ret10 = float((prices.iloc[10] - entry) / entry)
            rec["return_10d"] = round(ret10, 6)

        recommendation = rec.get("recommendation", "HOLD")
        direction = 1 if "BUY" in recommendation else (-1 if "SELL" in recommendation else 0)

        ret5 = rec.get("return_5d")
        if ret5 is not None and direction != 0:
            rec["reward_5d_sharpe"] = round(direction * ret5, 6)

            lows = hist["Low"].dropna()
            if len(lows) >= 6:
                worst_low = float(lows.iloc[1:6].min())
                max_dd = (entry - worst_low) / entry
                rec["reward_calmar"] = round(direction * ret5 / max(max_dd, 0.005), 6)

            ret10 = rec.get("return_10d")
            if ret10 is not None:
                rec["reward_10d_sharpe"] = round(direction * ret10, 6)

            rr = rec.get("risk_reward_at_entry") or 1.0
            rr_weight = min(rr / 2.0, 1.5)
            rec["reward_rr_weighted"] = round(rec["reward_5d_sharpe"] * rr_weight, 6)
            rec["reward_regime_adj"] = rec["reward_5d_sharpe"]

        rec["outcome_fetched"] = True

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        if not self.log_path.exists():
            return {"total_records": 0, "log_path": str(self.log_path)}

        records = self._load_all()
        labelled = [r for r in records if r.get("outcome_fetched")]
        rewards = [r["reward_5d_sharpe"] for r in labelled
                   if r.get("reward_5d_sharpe") is not None]

        action_counts: Dict[str, int] = {}
        for r in records:
            a = r.get("recommendation", "UNKNOWN")
            action_counts[a] = action_counts.get(a, 0) + 1

        reward_stats: Dict[str, Any] = {}
        if rewards:
            import statistics
            reward_stats = {
                "mean": round(statistics.mean(rewards), 6),
                "std": round(statistics.stdev(rewards), 6) if len(rewards) > 1 else 0.0,
                "min": round(min(rewards), 6),
                "max": round(max(rewards), 6),
            }

        return {
            "total_records": len(records),
            "labelled_records": len(labelled),
            "unlabelled": len(records) - len(labelled),
            "action_counts": action_counts,
            "reward_stats": reward_stats,
            "log_path": str(self.log_path),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_all(self) -> List[Dict]:
        records = []
        with open(self.log_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _save_all(self, records: List[Dict]) -> None:
        with open(self.log_path, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec, default=_json_serial) + "\n")


def _json_serial(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Not serializable: {type(obj)}")
