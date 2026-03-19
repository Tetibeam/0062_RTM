from app.utils.dashboard_utility import (
    make_graph_template
)

from typing import Dict, Any
from app import cache

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np

def _build_summary():
    pass

#@cache.cached(timeout=300)  # 300秒間（5分間）キャッシュを保持する
def build_Market_Regime_Radar_payload(include_graphs=False, include_summary=False):

    result = {"ok":True, "summary": {}, "graphs": {}}

    make_graph_template()

    if include_summary:
        pass

    if include_graphs:
        result["graphs"] = {
            #"regime_transitions": _build_regime_transition(df_regime, df_regime_summarize),
            #"decision_confidence": _bulid_decision_confidence(df_regime, df_regime_summarize),
            #"early_warning": _build_early_warning(df_regime),
            #"rolling_driver_path": _build_rolling_driver_path(df_regime),
            #"regime_snapshot": _build_regime_snapshot(df_regime),
            #"regime_playbook": _build_regime_playbook(),
        }
    return result

if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
    # DBマネージャーの初期化
    from app.utils.db_manager import init_db
    init_db(base_dir)
    
    from app import create_app
    app = create_app()

    build_Market_Regime_Radar_payload(False, False)