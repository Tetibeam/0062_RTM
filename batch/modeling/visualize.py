########################################################
# グラフ化関数
########################################################
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from plotly.subplots import make_subplots


########################################################
# Market Navigator
########################################################

# Market Navigator統合版
def plot_market_navigator(df_features, regime_clf, driver_clf, start_date="2022-01-01", end_date="2023-01-01"):

    # 1. データの準備と期間抽出
    df_plot = df_features.loc[start_date:end_date].copy()
    price_index = (1 + df_plot['sp500_ret_1d']).cumprod()

    # 各モデルの学習済み特徴量リストを自動取得（順番と数を守るため）
    regime_feats = regime_clf.feature_name_
    driver_feats = driver_clf.feature_name_

    # 2. 推論 (各モデルの確率を取得)
    # --- 1-1. Regime Prism (19指標) ---
    r_labels = ['1: Golden Dip', '2: Flash Crash', '3: Slow Bleed', '4: Liquidity In', '5: Healthy']
    r_probs = regime_clf.predict_proba(df_plot[regime_feats])
    df_r = pd.DataFrame(r_probs, index=df_plot.index, columns=r_labels)
    df_r['dominant'] = df_r.idxmax(axis=1)

    # --- 1-2. Driver Profiler (21指標) ---
    d_labels = ['1: Credit', '2: Bond', '3: Equity', '4: Currency', '5: Neutral']
    d_probs = driver_clf.predict_proba(df_plot[driver_feats])
    df_d = pd.DataFrame(d_probs, index=df_plot.index, columns=d_labels)
    df_d['dominant'] = df_d.idxmax(axis=1)

    # 3. カラー定義 (RTM標準パレット)
    r_colors = ['rgba(255, 215, 0, 0.8)', 'rgba(255, 49, 49, 0.9)', 'rgba(178, 34, 34, 0.8)',
                'rgba(30, 144, 255, 0.8)', 'rgba(0, 250, 154, 0.7)']
    d_colors = ['rgba(147, 112, 219, 0.8)', 'rgba(255, 140, 0, 0.8)', 'rgba(50, 205, 50, 0.8)',
                'rgba(30, 144, 255, 0.8)', 'rgba(169, 169, 169, 0.7)']

    # 4. サブプロットの構築 (3段構成)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=("NAVIGATOR CONTEXT (Price & Regime Background)",
                        "REGIME PRISM (State Probability)",
                        "DRIVER PROFILER (Factor Probability)")
    )

    # --- Row 1: Price & Regime Background (Prismの色を背景に敷く) ---
    df_r['change'] = df_r['dominant'] != df_r['dominant'].shift()
    df_r['group'] = df_r['change'].cumsum()
    for _, group in df_r.groupby('group'):
        r_label = group['dominant'].iloc[0]
        idx = r_labels.index(r_label)
        bg_color = r_colors[idx].replace('0.8', '0.12').replace('0.9', '0.12').replace('0.7', '0.12')
        fig.add_shape(type="rect", xref="x", yref="paper", x0=group.index[0], x1=group.index[-1],
                      y0=0.62, y1=1, fillcolor=bg_color, line_width=0, layer="below", row=1, col=1)

    fig.add_trace(go.Scatter(x=price_index.index, y=price_index, name="S&P500",
                             line=dict(color='#FFFFFF', width=1.5)), row=1, col=1)

    # --- Row 2: Regime Prism Stacked Area ---
    for i, col in enumerate(r_labels):
        fig.add_trace(go.Scatter(x=df_r.index, y=df_r[col], name=col, stackgroup='regime',
                                 line=dict(width=0, color=r_colors[i]), fillcolor=r_colors[i],
                                 hovertemplate='%{y:.1%}'), row=2, col=1)

    # --- Row 3: Driver Profiler Stacked Area ---
    for i, col in enumerate(d_labels):
        fig.add_trace(go.Scatter(x=df_d.index, y=df_d[col], name=col, stackgroup='driver',
                                 line=dict(width=0, color=d_colors[i]), fillcolor=d_colors[i],
                                 hovertemplate='%{y:.1%}'), row=3, col=1)

    # --- レイアウトの最終磨き上げ ---
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>RTM: MARKET NAVIGATOR</b><br><span style='font-size:12px; color:#A0A0A0;'>Analysis Period: {start_date} to {end_date}</span>",
            x=0.02, y=0.97
        ),
        hovermode="x unified",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(size=10)),
        margin=dict(l=50, r=160, t=60, b=50),
        plot_bgcolor='#050505', paper_bgcolor='#050505',
    )

    fig.update_yaxes(title_text="Price Index", gridcolor='#222', row=1, col=1)
    fig.update_yaxes(title_text="Regime %", tickformat=".0%", range=[0, 1], gridcolor='#222', row=2, col=1)
    fig.update_yaxes(title_text="Driver %", tickformat=".0%", range=[0, 1], gridcolor='#222', row=3, col=1)
    fig.update_xaxes(gridcolor='#222', rangeslider_visible=False)

    fig.show(config=dict(displayModeBar=False))
    return fig

# Regime Prismの可視化
def plot_regime_trajectory(df_ready, sp500_ret, labels, start_date="2022-01-01", end_date="2023-01-01"):
    # 1. データの準備
    X = df_ready.drop(columns=['actual_regime'])
    X_latest = X.loc[start_date:end_date]

    # ★追加: 正解ラベルを取得
    y_latest = df_ready.loc[start_date:end_date, 'actual_regime']

    # 2. 予測確率の算出
    labels = labels
    probs = X_latest
    df_probs = pd.DataFrame(probs, index=X_latest.index, columns=labels)

    # 最も確率が高いレジュームを特定（強調表示用）
    df_probs['dominant_regime'] = df_probs.idxmax(axis=1)

    # ★追加: マッピング用の辞書を作成し、返り値のデータフレームにも正解を記録
    regime_dict = {float(i+1): label for i, label in enumerate(labels)}
    df_probs['actual_regime'] = y_latest.map(regime_dict)
    pd.set_option('display.max_rows', None)
    print(df_probs)

    # 3. カラーパレットの設定（プロ仕様）
    colors = ['rgba(255, 215, 0, 0.8)', 'rgba(255, 0, 255, 0.85)', 'rgba(178, 34, 34, 0.8)', 
              'rgba(30, 144, 255, 0.8)', 'rgba(0, 250, 154, 0.7)']

    # ★追加: 正解ラベルの色引き当て用辞書
    color_dict = {float(i+1): color for i, color in enumerate(colors)}

    # 4. 可視化
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.35, 0.65],
        # ★追加: タイトルを微修正
        subplot_titles=("MARKET CONTEXT (S&P500) & Actual Regime", "PROBABILISTIC TRAJECTORY")
    )

    # --- 上段：価格チャート & 支配的レジュームの背景強調 ---
    sp500_price = (1 + sp500_ret).cumprod()
    sp500_price = sp500_price.loc[start_date:end_date]
    min_sp500 = sp500_price.min() - 0.1
    max_sp500 = sp500_price.max() + 0.1

    df_probs['change'] = df_probs['dominant_regime'] != df_probs['dominant_regime'].shift()
    df_probs['group'] = df_probs['change'].cumsum()

    regime_groups = df_probs.groupby('group')

    for _, group in regime_groups:
        start_dt = group.index[0]
        end_dt = group.index[-1]
        regime_label = group['dominant_regime'].iloc[0]

        regime_idx = labels.index(regime_label)
        base_color = colors[regime_idx]
        bg_color = base_color.replace('0.8', '0.12').replace('0.9', '0.12').replace('0.7', '0.12')

        fig.add_shape(
            type="rect",
            xref="x", 
            yref="y", # ★修正: 元コードの"paper"だとデータ座標(min_sp500等)と噛み合わずズレるため"y"に修正
            x0=start_dt, x1=end_dt,
            y0=min_sp500, y1=max_sp500,
            fillcolor=bg_color,
            line_width=0,
            layer="below", 
            row=1, col=1
        )

    # ★追加: 正解ラベルの「カラーリボン」を価格チャートの下部に描画
    true_colors = y_latest.map(color_dict)
    true_texts = y_latest.map(regime_dict)

    fig.add_trace(go.Scatter(
        x=y_latest.index,
        y=[min_sp500 + 0.01] * len(y_latest), # 価格チャートの一番下（min_sp500の少し上）に固定
        mode='markers',
        marker=dict(color=true_colors, symbol='square', size=12),
        name='Actual (True)',
        text=true_texts,
        hovertemplate='<b>Actual: %{text}</b><extra></extra>',
        showlegend=False # 凡例が煩雑になるのを防ぐ
    ), row=1, col=1)

    # 価格線を重ねる
    fig.add_trace(go.Scatter(
        x=sp500_price.index, y=sp500_price, 
        name="S&P500", line=dict(color='#E0E0E0', width=1.5)
    ), row=1, col=1)

    # --- 下段：確率のスタックエリア（グラデーション） ---
    for i, col in enumerate(labels):
        fig.add_trace(go.Scatter(
            x=df_probs.index, y=df_probs[col],
            name=col,
            stackgroup='one',
            line=dict(width=0, color=colors[i]),
            fillcolor=colors[i],
            hovertemplate='%{y:.1%}'
        ), row=2, col=1)

    # --- レイアウトの磨き上げ ---
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>REGIME PRISM</b><br><span style='font-size:12px; color:#A0A0A0;'>Analysis Period: {start_date} to {end_date}</span>",
            x=0.05, y=0.95
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=120, b=50),
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
    )

    fig.update_yaxes(gridcolor='#222', zeroline=False)
    fig.update_yaxes(title_text="Probability", tickformat=".0%", range=[0, 1], row=2, col=1)
    fig.update_xaxes(gridcolor='#222', rangeslider_visible=False)

    fig.show(config=dict(displayModeBar=False))
    
    # 解析用に出力データフレームの列を整理して返す
    return df_probs[['actual_regime', 'dominant_regime'] + regime_labels]

# Driver Profilerの可視化
def plot_driver_trajectory(df_ready, sp500_ret, start_date="2022-01-01", end_date="2023-01-01"):
    # 1. データの準備
    X = df_ready.drop(columns=['actual_regime'])
    X_latest = X.loc[start_date:end_date]

    # ★追加: 正解ラベルを取得
    y_latest = df_ready.loc[start_date:end_date, 'actual_regime']

    # 2. 予測確率の算出
    regime_labels = ['1: Credit', '2: Bond', '3: Equity', '4: Neutral']
    probs = X_latest
    df_probs = pd.DataFrame(probs, index=X_latest.index, columns=regime_labels)

    # 最も確率が高いレジュームを特定（強調表示用）
    df_probs['dominant_regime'] = df_probs.idxmax(axis=1)

    # ★追加: マッピング用の辞書を作成し、返り値のデータフレームにも正解を記録
    regime_dict = {float(i+1): label for i, label in enumerate(regime_labels)}
    df_probs['actual_regime'] = y_latest.map(regime_dict)
    pd.set_option('display.max_rows', None)
    print(df_probs)

    # 3. カラーパレットの設定（プロ仕様）
    colors = ['rgba(255, 215, 0, 0.8)', 'rgba(255, 0, 255, 0.85)', 'rgba(178, 34, 34, 0.8)', 
              'rgba(30, 144, 255, 0.8)', 'rgba(0, 250, 154, 0.7)']

    # ★追加: 正解ラベルの色引き当て用辞書
    color_dict = {float(i+1): color for i, color in enumerate(colors)}

    # 4. 可視化
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.35, 0.65],
        # ★追加: タイトルを微修正
        subplot_titles=("MARKET CONTEXT (S&P500) & Actual Regime", "PROBABILISTIC TRAJECTORY")
    )

    # --- 上段：価格チャート & 支配的レジュームの背景強調 ---
    sp500_price = (1 + sp500_ret).cumprod()
    sp500_price = sp500_price.loc[start_date:end_date]
    min_sp500 = sp500_price.min() - 0.1
    max_sp500 = sp500_price.max() + 0.1

    df_probs['change'] = df_probs['dominant_regime'] != df_probs['dominant_regime'].shift()
    df_probs['group'] = df_probs['change'].cumsum()

    regime_groups = df_probs.groupby('group')

    for _, group in regime_groups:
        start_dt = group.index[0]
        end_dt = group.index[-1]
        regime_label = group['dominant_regime'].iloc[0]

        regime_idx = regime_labels.index(regime_label)
        base_color = colors[regime_idx]
        bg_color = base_color.replace('0.8', '0.12').replace('0.9', '0.12').replace('0.7', '0.12')

        fig.add_shape(
            type="rect",
            xref="x", 
            yref="y", # ★修正: 元コードの"paper"だとデータ座標(min_sp500等)と噛み合わずズレるため"y"に修正
            x0=start_dt, x1=end_dt,
            y0=min_sp500, y1=max_sp500,
            fillcolor=bg_color,
            line_width=0,
            layer="below", 
            row=1, col=1
        )

    # ★追加: 正解ラベルの「カラーリボン」を価格チャートの下部に描画
    true_colors = y_latest.map(color_dict)
    true_texts = y_latest.map(regime_dict)

    fig.add_trace(go.Scatter(
        x=y_latest.index,
        y=[min_sp500 + 0.01] * len(y_latest), # 価格チャートの一番下（min_sp500の少し上）に固定
        mode='markers',
        marker=dict(color=true_colors, symbol='square', size=12),
        name='Actual Regime (True)',
        text=true_texts,
        hovertemplate='<b>Actual: %{text}</b><extra></extra>',
        showlegend=False # 凡例が煩雑になるのを防ぐ
    ), row=1, col=1)

    # 価格線を重ねる
    fig.add_trace(go.Scatter(
        x=sp500_price.index, y=sp500_price, 
        name="S&P500", line=dict(color='#E0E0E0', width=1.5)
    ), row=1, col=1)

    # --- 下段：確率のスタックエリア（グラデーション） ---
    for i, col in enumerate(regime_labels):
        fig.add_trace(go.Scatter(
            x=df_probs.index, y=df_probs[col],
            name=col,
            stackgroup='one',
            line=dict(width=0, color=colors[i]),
            fillcolor=colors[i],
            hovertemplate='%{y:.1%}'
        ), row=2, col=1)

    # --- レイアウトの磨き上げ ---
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>REGIME PRISM</b><br><span style='font-size:12px; color:#A0A0A0;'>Analysis Period: {start_date} to {end_date}</span>",
            x=0.05, y=0.95
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=120, b=50),
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
    )

    fig.update_yaxes(gridcolor='#222', zeroline=False)
    fig.update_yaxes(title_text="Probability", tickformat=".0%", range=[0, 1], row=2, col=1)
    fig.update_xaxes(gridcolor='#222', rangeslider_visible=False)

    fig.show(config=dict(displayModeBar=False))
    
    # 解析用に出力データフレームの列を整理して返す
    return df_probs[['actual_regime', 'dominant_regime'] + regime_labels]

# Regional Biasの可視化
def plot_regional_bias_trajectory(df_features, compass_clf, nikkei_price, sp500_price, 
                                  start_date="2022-01-01", end_date="2026-03-01",
                                  threshold_c1=0.50, threshold_c2=0.35):
    # 1. データの準備
    df_plot = df_features.loc[start_date:end_date].copy()
    feature_cols = compass_clf.feature_name_

    # 2. 予測確率の算出
    bias_labels = ['1: JP Overdrive', '2: JP Fragile', '3: Synchronized']
    probs = compass_clf.predict_proba(df_plot[feature_cols])
    df_probs = pd.DataFrame(probs, index=df_plot.index, columns=bias_labels)

    # 判定ロジック（閾値適用）
    df_probs['final_signal'] = '3: Synchronized'
    df_probs.loc[df_probs['1: JP Overdrive'] >= threshold_c1, 'final_signal'] = '1: JP Overdrive'
    df_probs.loc[df_probs['2: JP Fragile'] >= threshold_c2, 'final_signal'] = '2: JP Fragile'

    # 3. カラーパレット（Regional Bias 用）
    # 青(Overdrive), 赤(Fragile), 灰(Sync)
    colors = [
        'rgba(30, 144, 255, 0.8)',  # Dodger Blue
        'rgba(255, 69, 0, 0.8)',    # Orange Red
        'rgba(169, 169, 169, 0.6)'   # Dark Gray
    ]

    # 4. 可視化の構築
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.6],
        subplot_titles=("RELATIVE PERFORMANCE (NIKKEI 225 / S&P 500)", "REGIONAL BIAS PROBABILITY TRAJECTORY")
    )

    # --- 上段：相対パフォーマンス & シグナル背景 ---
    # 日本株/米国株の相対レシオ（開始日を1.0として指数化）
    ratio = (nikkei_price / sp500_price).loc[start_date:end_date].dropna()
    relative_index = ratio / ratio.iloc[0]

    # 背景の矩形描画（final_signalに基づく）
    df_probs['change'] = df_probs['final_signal'] != df_probs['final_signal'].shift()
    df_probs['group'] = df_probs['change'].cumsum()
    signal_groups = df_probs.groupby('group')

    for _, group in signal_groups:
        start_dt = group.index[0]
        end_dt = group.index[-1]
        signal_label = group['final_signal'].iloc[0]

        idx = bias_labels.index(signal_label)
        bg_color = colors[idx].replace('0.8', '0.15').replace('0.6', '0.1')

        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=start_dt, x1=end_dt, y0=relative_index.min()-0.1, y1=relative_index.max()+0.1,
            fillcolor=bg_color, line_width=0, layer="below", row=1, col=1
        )

    # 相対ライン
    fig.add_trace(go.Scatter(
        x=relative_index.index, y=relative_index, 
        name="Nikkei/SP500 Ratio", line=dict(color="#EEE0E0", width=2)
    ), row=1, col=1)

    # --- 下段：確率のスタックエリア ---
    for i, col in enumerate(bias_labels):
        fig.add_trace(go.Scatter(
            x=df_probs.index, y=df_probs[col],
            name=col,
            stackgroup='one',
            line=dict(width=0, color=colors[i]),
            fillcolor=colors[i],
            hovertemplate='%{y:.1%}'
        ), row=2, col=1)

    # 閾値ライン（点線）の追加
    fig.add_shape(type="line", x0=df_probs.index[0], x1=df_probs.index[-1], 
                  y0=threshold_c1, y1=threshold_c1, line=dict(color=colors[0], dash="dot", width=1), row=2, col=1)
    fig.add_shape(type="line", x0=df_probs.index[0], x1=df_probs.index[-1], 
                  y0=threshold_c2, y1=threshold_c2, line=dict(color=colors[1], dash="dot", width=1), row=2, col=1)

    # --- レイアウト ---
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>REGIONAL BIAS COMPASS</b><br><span style='font-size:12px; color:#A0A0A0;'>Signal Thresholds: Overdrive >= {threshold_c1:.0%}, Fragile >= {threshold_c2:.0%}</span>",
            x=0.05, y=0.95
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=130, b=50),
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
    )

    fig.update_yaxes(gridcolor='#222', zeroline=False, range=[relative_index.min()-0.1, relative_index.max()+0.1])
    fig.update_yaxes(title_text="Probability", tickformat=".0%", range=[0, 1], row=2, col=1)
    fig.update_xaxes(gridcolor='#222', rangeslider_visible=False)

    fig.show(config=dict(displayModeBar=False))
    return df_probs

# Shapの可視化
def plot_shap_explanation(importance_df, pred_regime):
    # 貢献度の上位・下位を抽出
    top_pos = importance_df.head(5) # 確率を押し上げた要因
    top_neg = importance_df.tail(5) # 確率を押し下げた要因（ブレーキ役）

    plot_df = pd.concat([top_pos, top_neg]).sort_values('contribution')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=plot_df['feature'],
        x=plot_df['contribution'],
        orientation='h',
        marker_color=['#FF4500' if c > 0 else '#1E90FF' for c in plot_df['contribution']]
    ))

    fig.update_layout(
        template="plotly_dark",
        title=f"WHY REGIME {pred_regime}? (Top Contribution Factors)",
        xaxis_title="SHAP Value (Impact on Probability)",
        paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a'
    )
    fig.show()

########################################################
# Macro Altimeterの可視化
########################################################
def plot_gli_trajectory(df_trajectory, gli_series, price_series, start_date=None, end_date=None):
    # 1. データの準備
    start_date = start_date or df_trajectory.index[0]
    end_date = end_date or df_trajectory.index[-1]

    df_p = df_trajectory.loc[start_date:end_date].copy()
    gli_p = gli_series.loc[start_date:end_date]
    price_p = price_series.loc[start_date:end_date]

    # ラベルとカラー
    regime_labels = ['1: STALL', '2: CRUISE', '3: LIFT']
    colors = ['rgba(255, 82, 82, 0.8)', 'rgba(144, 164, 174, 0.6)', 'rgba(0, 230, 118, 0.8)']

    # 2. サブプロットの作成（上段のみ第2軸を有効化）
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.4, 0.6],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=("INDICATOR (GLI) vs PRICE ACTION", "PROBABILITY TRAJECTORY")
    )

    # --- 上段：GLI (左軸 / シアン) ---
    fig.add_trace(go.Scatter(
        x=gli_p.index, y=gli_p,
        name="GLI Indicator", line=dict(color='#00D4FF', width=2.5)
    ), row=1, col=1,secondary_y=False)

    # --- 上段：Price (右軸 / ゴールド) ---
    # ★ここがポイント：白や明るい色にして、第2軸を使うことで「見える」ようにします
    fig.add_trace(go.Scatter(
        x=price_p.index, y=price_p,
        name="Market Price", line=dict(color='#FFD700', width=2) # ゴールドで強調
    ), row=1, col=1, secondary_y=True)

    # 背景色の強調（支配的レジーム）
    df_p['dominant'] = df_p[regime_labels].idxmax(axis=1)
    df_p['group'] = (df_p['dominant'] != df_p['dominant'].shift()).cumsum()
    for _, group in df_p.groupby('group'):
        label = group['dominant'].iloc[0]
        bg_color = colors[regime_labels.index(label)].replace('0.8', '0.08').replace('0.6', '0.08')
        fig.add_vrect(x0=group.index[0], x1=group.index[-1], fillcolor=bg_color, line_width=0, layer="below", row=1, col=1)

    # --- 下段：確率のスタックエリア ---
    for i, label in enumerate(regime_labels):
        fig.add_trace(go.Scatter(
            x=df_p.index, y=df_p[label], name=label,
            stackgroup='one', line=dict(width=0, color=colors[i]), fillcolor=colors[i],
            hovertemplate='%{y:.1%}'
        ), row=2, col=1)

    # レイアウトの磨き上げ
    fig.update_layout(
        template="plotly_dark", height=800,
        title=dict(text=f"<b>GLI LIQUIDITY FLOW</b><br><span style='font-size:12px; color:#888;'>Target: {df_p.index[-1].date()} | Status: {df_p['dominant'].iloc[-1]}</span>", x=0.05, y=0.96),
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='#050505', paper_bgcolor='#050505'
    )

    # 軸の名称設定
    fig.update_yaxes(title_text="GLI", secondary_y=False, row=1, col=1, gridcolor='#222')
    fig.update_yaxes(title_text="Price", secondary_y=True, row=1, col=1, showgrid=False)
    fig.update_yaxes(title_text="Probability", tickformat=".0%", range=[0, 1], row=2, col=1, gridcolor='#222')
    fig.update_xaxes(gridcolor='#222')

    fig.show(config=dict(displayModeBar=False))


########################################################
# Regime Prismラベルの可視化
########################################################

# レジュームラベル
def plot_regime_label(df):
    fig = go.Figure()

    # 1. メインのS&P500チャート
    fig.add_trace(go.Scatter(
        x=df.index, y=df["sp500"],
        name="S&P 500",
        line=dict(color="rgba(200, 200, 200, 0.8)", width=1.5),
    ))

    # 2. 背景色（Shapes）のリスト作成
    shapes = []
    colors = {
        "1: Golden Dip": "rgba(255, 0, 0, 0.25)",
        "2: Flash Crash": "rgba(255, 255, 0, 0.25)",
        "3: Slow Bleed": "rgba(0, 255, 0, 0.25)",
        "4: Liquidity In": "rgba(0, 255, 2550, 0.25)",
        "5: Healthy/Neutral": "rgba(255, 255, 255, 0.25)",
    }

    for regime, color in colors.items():
        # --- 凡例（Legend）用のダミートレースを追加 ---
        solid_color = color.replace("0.25", "1.0") # 凡例は見やすいように透明度をなくす
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=solid_color, symbol='square'),
            name=regime
        ))

        # 特定のパターンの日を1、それ以外を0とする
        mask = (df["regime"] == regime).astype(int)
        # 0から1、1から0への変化点を特定して「期間」を作る
        diff = mask.diff()
        starts = df.index[diff == 1].tolist()
        ends = df.index[diff == -1].tolist()

        # データの最後がパターンのまま終わる場合の処理
        if mask.iloc[0] == 1:
            starts.insert(0, df.index[0])
        if mask.iloc[-1] == 1:
            ends.append(df.index[-1])

        # shapeオブジェクトを作成
        for start, end in zip(starts, ends):
            shapes.append(dict(
                type="rect",
                xref="x", yref="paper", # X軸は日付、Y軸はチャート全体
                x0=start, x1=end,
                y0=0, y1=1,
                fillcolor=color,
                opacity=1, # 既にRGBAで透明度指定済み
                layer="below",
                line_width=0,
            ))

    # 3. レイアウト更新
    fig.update_layout(
        shapes=shapes,
        title="Step 1 Verification: Background Shading with Shapes",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=100, b=50)
    )

    fig.show(config=dict(displayModeBar=False))

# 主導要因ラベル
def plot_factor_label(df):
    """
    regime_detailed (方向 + 主導要因) に基づいて多色で背景を塗り分ける
    """
    fig = go.Figure()

    # 1. メインのS&P500チャート
    fig.add_trace(go.Scatter(
        x=df.index, y=df["sp500"], 
        name="S&P 500",
        line=dict(color="rgba(200, 200, 200, 0.8)", width=1.5),
    ))

    # 2. 詳細パターンと色のマッピング定義
    # 押し目(Green系) と 崖(Red系) をベースに、犯人ごとに色相をずらす
    colors = {
        #"1: Golden Dip (Equity)": "rgba(0, 255, 0, 0.3)",      # ピュアグリーン (通常の押し目)
        #"1: Golden Dip (Bond)": "rgba(0, 255, 255, 0.3)",    # シアン (金利低下による反発)
        #"1: Golden Dip (Credit)": "rgba(173, 255, 47, 0.3)", # イエローグリーン (信用回復)
        #"1: Golden Dip (Currency)": "rgba(100, 149, 237, 0.3)",# ライトブルー (為替主導)

        #"2: Flash Crash (Equity)": "rgba(255, 0, 0, 0.3)",     # ピュアレッド (通常のパニック)
        #"2: Flash Crash (Bond)": "rgba(255, 165, 0, 0.3)",   # オレンジ (金利上昇ショック)
        #"2: Flash Crash (Credit)": "rgba(255, 0, 255, 0.3)", # マゼンタ/紫 (信用・流動性危機)
        #"2: Flash Crash (Currency)": "rgba(255, 105, 180, 0.3)",# ピンク (為替ショック)

        "3: Slow Bleed (Equity)": "rgba(139, 0, 0, 0.3)",    # ダークレッド
        "3: Slow Bleed (Bond)": "rgba(210, 105, 30, 0.4)",   # チョコレート/濃いオレンジ (★2022年の主役)
        "3: Slow Bleed (Credit)": "rgba(75, 0, 130, 0.3)",   # インディゴ/暗い紫
        "3: Slow Bleed (Currency)": "rgba(70, 130, 180, 0.3)", # スチールブルー
    }

    shapes = []

    for regime, color in colors.items():
        # --- 凡例（Legend）用のダミートレースを追加 ---
        # 背景色と同じ色の四角形アイコンを凡例に表示させる
        solid_color = color.replace("0.3", "1.0") # 凡例は見やすいように透明度をなくす
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=solid_color, symbol='square'),
            name=regime
        ))

        # --- 背景色（Shapes）の作成 ---
        # 特定の詳細パターンの日を1、それ以外を0とする
        mask = (df["regime_detailed"] == regime).astype(int)
        diff = mask.diff()
        starts = df.index[diff == 1].tolist()
        ends = df.index[diff == -1].tolist()

        if len(mask) > 0:
            if mask.iloc[0] == 1:
                starts.insert(0, df.index[0])
            if mask.iloc[-1] == 1:
                ends.append(df.index[-1])

        for start, end in zip(starts, ends):
            shapes.append(dict(
                type="rect",
                xref="x", yref="paper",
                x0=start, x1=end,
                y0=0, y1=1,
                fillcolor=color,
                opacity=1,
                layer="below",
                line_width=0,
            ))

    # 3. レイアウト更新
    fig.update_layout(
        shapes=shapes,
        title="Step 2 Verification: Market Regimes by Driver (Equity/Bond/Credit/Currency)",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="S&P 500 Price",
        # 凡例が多いため、少し折り返して表示しやすくする
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            traceorder="normal"
        ),
        margin=dict(l=50, r=50, t=120, b=50)
    )

    fig.show(config=dict(displayModeBar=False))


########################################################
# 個別指標等の単純な可視化
########################################################
def plot_index(df, x_label="Date"):

    fig = go.Figure()

    for col in df.columns:
        df_sub = df[col].dropna()
        fig.add_trace(go.Scatter(
            x=df_sub.index,
            y=df_sub,
            mode='lines',
            name=col,
        ))
    # レイアウト設定
    fig.update_layout(
        title=f'Original Data',
        xaxis=dict(
            title=dict(
                text=x_label,
                font=dict(size=16),
            ),
            tickfont=dict(size=16),
        ),
        yaxis=dict(
            title=dict(
                text='Value',
                font=dict(size=16),
            ),
            tickfont=dict(size=16),
        ),
        legend=dict(
            title="Indicators",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=50,r=50,t=50,b=50),
    )

    fig.show(config=dict(displayModeBar=False))

def _plot_lag_correlation(df_lag_corrs):

    fig = go.Figure()

    for col in df_lag_corrs.columns:
        fig.add_trace(go.Scatter(
            x=df_lag_corrs.index,
            y=df_lag_corrs[col],
            mode='lines+markers',
            name=col,
            marker=dict(size=20)
        ))

    # レイアウト設定
    fig.update_layout(
        title=f'Lagged Correlation Analysis',
        xaxis=dict(
            title='Quarters Leading (Lag)',
            dtick=1,
        ),
        yaxis=dict(
            title='Correlation Coefficient',
            zeroline=True,
            zerolinewidth=2,
        ),
        legend=dict(
            title="Indicators",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=50,r=50,t=50,b=50),
    )

    fig.show()




#-------------------------------------------------------------
def _plot_graphs(cal_out, origin):

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cal_out.index,
        y=cal_out,
        mode='lines',
        name=cal_out.name,
        #yaxis="y1"
        
    ))
    fig.add_trace(go.Scatter(
        x=origin.index,
        y=origin,
        mode='lines',
        name=origin.name,
        #yaxis='y2'
    ))
    fig.update_layout(
        xaxis=dict(
            title='Date',
            showgrid=False
        ),
        #yaxis=dict(side='left', showgrid=True),
        #yaxis2=dict(side='right', overlaying='y', showgrid=False),
    )
    fig.show()

def _plot_contribution(contrib_rolling):

    # 4. Plotlyで積層エリアチャートを作成
    fig = go.Figure()

    for col in contrib_rolling.columns:
        fig.add_trace(go.Scatter(
            x=contrib_rolling.index,
            y=contrib_rolling[col],
            mode='lines',
            stackgroup='one', # 積層設定
            name=col,
            hoverinfo='x+y+name'
        ))

    # レイアウト設定
    fig.update_layout(
        title='GLI Factor Contribution Analysis (50-week Rolling)',
        xaxis_title='Date',
        yaxis_title='Contribution Level',
        template='plotly_dark', # ダークモードで見やすく
        legend_title='Variables',
        hovermode='x unified'
    )
    
    # 0のラインを追加
    fig.add_shape(type="line", x0=contrib_rolling.index[0], y0=0, x1=contrib_rolling.index[-1], y1=0,
                  line=dict(color="Gray", width=2, dash="dash"))

    fig.show()




def _plot_original_series(series:pd.Series):

    import plotly.graph_objects as go
    fig = go.Figure()

    series = series.dropna()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series,
        mode='lines',
        name=series.name,
    ))
    """
    series_rolling12 = series.dropna().rolling(window=12).mean()
    fig.add_trace(go.Scatter(
        x=series_rolling12.index,
        y=series_rolling12,
        mode='lines',
        name=series.name+"_rolling12",
        ))
    """
    # レイアウト設定
    fig.update_layout(
        title=f'Original Data',
        xaxis=dict(
            title='Date',
        ),
        yaxis=dict(
            title='Value',
        ),
        legend=dict(
            title="Indicators",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=50,r=50,t=50,b=50),
    )

    fig.show()

def _plot_corr_matrix(df):
    corr = df.dropna().corr()

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation Matrix (DFA Input Features)"
    )

    fig.update_layout(
        width=900,
        height=900,
        xaxis_title="Features",
        yaxis_title="Features",
        margin=dict(l=50,r=50,t=50,b=50),
    )

    fig.show()

def _plot_mid_label(df):

    # 20日後の累積リターン
    df['next_20d_ret'] = df['sp500_ret_1m'].shift(-20)
    # 未来20日間の最大下落率
    future_min_price = df['sp500'].rolling(window=20).min().shift(-20)
    df['next_20d_mae'] = (future_min_price / df['sp500']) - 1
    df['next_20d_mae'] = df['next_20d_mae'].clip(upper=0)   #下落なしの処理

    # サブプロットの作成
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.25, 0.25, 0.25, 0.25],
        subplot_titles=(
            "S&P 500 & Regimes", "Future return(20d)", "Stress counter", "MAE(20d)"
            )
        )

    # 1. S&P 500 & Regimes
    fig.add_trace(go.Scatter(
        x=df.index, y=df['sp500'], name='sp500',
        line=dict(width=1.5)
        ),row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['sp500_ma20'], name='sp500_ma20',
        line=dict(width=1.5)
        ),row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['sp500_ma200'], name='sp500_ma200',
        line=dict(width=1.5)
        ),row=1, col=1)
    # 2. Future return(20d)
    fig.add_trace(go.Bar(
        x=df.index, y=df['next_20d_ret'], name='next_20d_ret',
        marker=dict(color="white")
        ),row=2, col=1)
    # 3. Stress counter
    fig.add_trace(go.Scatter(
        x=df.index, y=df['r3_stress_count'], name='r3_stress_count',
        line=dict(width=1.5),
        ),row=3, col=1)
    # 4. MAE(20d)
    fig.add_trace(go.Bar(
        x=df.index, y=df['next_20d_mae'], name='next_20d_mae',
        marker=dict(color="white")
        ),row=4, col=1)

    # 区間IDを作成
    df["regime_change"] = df["final_regime"] != df["final_regime"].shift(1)
    df["regime_block"] = df["regime_change"].cumsum()
    df_regime = df.groupby("regime_block").agg(
        Regime=("final_regime", "first"),
        start=("final_regime", lambda x: x.index.min()),
        end=("final_regime", lambda x: x.index.max()),
        ).reset_index()
    #print(df_regime)

    # Regime背景
    # レジームごとの色設定
    colors = {1: 'rgb(0,176,80)', 2: 'rgb(255,192,0)', 3: 'rgb(237,125,49)', 4: 'rgb(192,0,0)'}
    names = {1: 'Regime 1: Quiet', 2: 'Regime 2: Overheat',
             3: 'Regime 3: Stress', 4: 'Regime 4: Panic'}
    for _, row in df_regime.iterrows():
        fig.add_shape(
            type="rect",
            x0=row['start'],
            x1=row['end'],
            y0=0,
            y1=6000,
            fillcolor=colors[row['Regime']],
            opacity=0.3,
            line_width=0,
            layer="below",
            row=1,col=1
        )

    # レイアウト設定
    fig.update_layout(
        height =1800,
        #xaxis_rangeslider_visible=True,
        hovermode="x unified",
        margin=dict(l=10,r=10,t=10,b=60),
    )

    fig.show(config=dict(displayModeBar=False))
