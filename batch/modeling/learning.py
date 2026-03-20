import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.linear_model import Ridge, Lasso

from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
import statsmodels.api as sm

import lightgbm as lgb
import shap

# 開発用のテストで使うのみ本番はダッシュボードはplotly
import matplotlib.pyplot as plt 
import seaborn as sns

############################################################
# DFA - 複数の特徴量から共通因子を導き出す
############################################################
# DFA フレームワーク（テスト、本番一体）
def learning_dfa(df, factors=1, factor_orders=1, target_col=None, target_is_positive=True, output_name="output"):
    # 1. 標準化
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df)
    X_scaled = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)

    # 2. DFA実行
    model = DynamicFactorMQ(X_scaled, factors=factors, factor_orders=factor_orders)
    # toleranceを 1e-4 に緩和して収束エラーを回避
    results = model.fit(maxiter=1000, tolerance=1e-4, disp=False)

    # 3. 因子の抽出
    factor_values = results.states.filtered.iloc[:, 0].values

    # 4. 符号の調整（賢い向き合わせロジック）
    is_inverted = False
    if target_col is not None:
        loading_params = results.params[
            (results.params.index.str.contains('loading')) & 
            (results.params.index.str.contains(target_col))
        ]

        if not loading_params.empty:
            loading_value = loading_params.iloc[0]

            # パターンA：ターゲットが「上がると景気に良い」指標の場合（CP, 雇用など）
            if target_is_positive:
                # 因子もターゲットと同じ方向（正の相関）を向いてほしい
                if loading_value < 0:
                    factor_values = -factor_values
                    is_inverted = True
                    print(f"INFO: Factor inverted. Made to move WITH {target_col} (Original loading: {loading_value:.4f})")
                else:
                    print(f"INFO: Factor remains. Already moving WITH {target_col} (Loading: {loading_value:.4f})")

            # パターンB：ターゲットが「上がると景気に悪い」指標の場合（VIX, 失業率など）
            else:
                # 因子はターゲットと逆の方向（負の相関）を向いてほしい（ストレス低下＝因子プラス）
                if loading_value > 0:
                    factor_values = -factor_values
                    is_inverted = True
                    print(f"INFO: Factor inverted. Made to move OPPOSITE to {target_col} (Original loading: {loading_value:.4f})")
                else:
                    print(f"INFO: Factor remains. Already moving OPPOSITE to {target_col} (Loading: {loading_value:.4f})")
        else:
            print(f"WARNING: Loading for {target_col} not found.")

    # 5. 結果の格納
    res_series = pd.Series(factor_values, index=df.index, name=output_name)

    return res_series, results, is_inverted

# フォーキャスト
def learning_forecast(results, forecast_steps=12):
    forecast_res = results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_res.predicted_mean
    forecast_ci = forecast_res.conf_int()
    return forecast_mean, forecast_ci

# ラグ相関分析
def lag_analysis(df, target_col="", max_lag=8):
    lag_results = {}
    features = [c for c in df.columns if c != target_col]

    for col in features:
        corrs = []
        for l in range(max_lag + 1):
            # 指標をl期分「過去」にずらしてDSRとの相関を見る
            # feature(t-l) と DSR(t) の相関
            c = df[target_col].corr(df[col].shift(l))
            corrs.append(c)
        lag_results[col] = corrs
    df_result = pd.DataFrame(lag_results)

    return df_result

############################################################
# Light GBM for dev.-　AIが複数の特徴量から多くのルールを見つけ答えを予測する
############################################################
"""
LGBM 開発用の関数（TimeSeriesSplit）

【目的」
    ・学習パラメータの最終決定
    ・特徴量の最終選定
    ・シグナル・フィルターの値の決定
【機能】
    ・Accuracyの表示
    ・混合行列の表示
    ・特徴量重要度の表示
    ・学習曲線機能（過学習検証）
    ・カスタム閾値の探索（シグナル・フィルター）
"""

def learning_lgbm_test(
    # 特徴量と目的変数
    df_ready, target_col, labels,
    # TimeSeriesSplitの設定値
    n_splits=3, gap=3,
    # 学習パラメータの設定
    n_estimators=200, learning_rate=0.03, num_leaves=7, min_data_in_leaf=5,
    class_weight="balanced", reg_alpha=0.5, reg_lambda=0.5, importance_type='gain',
    sample_weight=None,
    # 学習曲線の表示
    learning_curve=False,
    # カスタム閾値の探索
    study_signal_filter=False,
    ):
    # 1. 前処理：データの準備
    df_ready = df_ready.dropna(subset=target_col)
    X = df_ready.drop(columns=target_col)
    y = df_ready[target_col]

    # 2. TimeSeriesSplitの設定
    # gap を指定することで、TrainとTestの間に空白を作り、未来リーク（カンニング）を完全に防ぐ
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    # 3. 学習過程を保持するリスト
    all_importances = [] # 各フォールドの特徴量寄与度(Gain)を保存
    all_y_probs = []     # 後の閾値調整用に「予測確率」を保存
    all_y_test = []      # 精度検証用の「正解ラベル」を集約
    all_y_pred = []      # 精度検証用の「予測ラベル」を集約
    evals_result = {}    # 学習曲線描画用のスコア履歴(Loss)を記録
    oof_probs_list = []  # OOF

    # クラスごとにSHAP値のリストを格納する辞書
    # 構造: {"1:Credit": [df_fold1, df_fold2...], "2:Bond": [...], ...}
    oof_shap_dict = {label: [] for label in labels}

    # 4. 学習ループ開始
    print(f"\n全サンプル数: {len(X)}")
    print("\n=== TimeSeriesSplit ===\n")

    fold = 1
    for train_index, test_index in tscv.split(X):
        # データ分割
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # モデル設定
        clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            #max_depth=4,
            num_leaves=num_leaves,
            min_data_in_leaf=min_data_in_leaf,
            class_weight=class_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            importance_type=importance_type,
            random_state=42,
            verbose=-1
        )

        # 学習
        clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_names=['train', 'valid'], # グラフのラベル名
            eval_metric='multi_logloss',   # 評価指標を明示
            # Early Stoppingで過学習を防ぐ
            # evals_result に学習の軌跡（スコアの履歴）を記録
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.record_evaluation(evals_result)
            ]
        )
        # 予測と確率
        # A. 予測確率(OOF)の保存
        y_prob = clf.predict_proba(X_test)
        df_fold_probs = pd.DataFrame(y_prob, index=X_test.index, columns=labels)
        df_fold_probs['actual_regime'] = y_test
        oof_probs_list.append(df_fold_probs)

        # B. SHAP値(OOF-SHAP)の計算と全クラス保存
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)

        # shap_valuesは多クラスの場合、リスト形式 [class0_array, class1_array, ...] で返る
        for i, label in enumerate(labels):
            # 各クラスのSHAP値をDataFrame化して辞書に格納
            df_fold_shap = pd.DataFrame(shap_values[i], index=X_test.index, columns=X.columns)
            oof_shap_dict[label].append(df_fold_shap)

        # C. Fold情報の表示
        y_pred = clf.predict(X_test)
        acc = balanced_accuracy_score(y_test, y_pred)
        print(f"Fold {fold} | Test: {X_test.index[0].date()} ~ {X_test.index[-1].date()} | Acc: {acc:.4f}")
        print(f" => Balanced Acc: {acc:.4f} (Best Iter: {best_iter})")

        # D. 重要度(Gain)の記録
        imp = pd.Series(clf.feature_importances_, index=X.columns)
        all_importances.append(imp)

        # extendを使うことで、各Foldの検証結果が順番
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_probs.extend(y_prob)

        fold += 1

    # 5. 総合結果レポートの表示
    report_lgbm_total_result(all_y_test, all_y_pred, all_importances)

    # 6．学習曲線の表示
    if learning_curve:
        from batch.modeling.visualize import plot_index
        train_loss = evals_result['train']['multi_logloss']
        valid_loss = evals_result['valid']['multi_logloss']
        plot_df = pd.DataFrame({
            'Train Loss (Learning)': train_loss,
            'Valid Loss (Generalization)': valid_loss
        })
        plot_index(plot_df, x_label="counts")

    # 7. 閾値探索
    if study_signal_filter:
        print(f"Total Test Samples: {len(all_y_test)}")
        print(classification_report(all_y_test, all_y_pred))
        search_optimal_thresholds(np.array(all_y_test), np.array(all_y_probs))

    # 8. 全テストフォールドの予測を結合
    df_oof_all = pd.concat(oof_probs_list).sort_index()

    # クラス別SHAP DataFrameの統合
    final_shap_dfs = {
        label: pd.concat(list_df).sort_index() 
        for label, list_df in oof_shap_dict.items()
    }

    return df_oof_all, final_shap_dfs

def report_lgbm_total_result(all_y_test, all_y_pred, all_importances):
    print("\n=======================================================")
    print("=== 学習結果 (Overall CV Performance) ===")
    print(classification_report(all_y_test, all_y_pred))

    print("\n=== 全フォールド総合の混同行列 ===")
    print(confusion_matrix(all_y_test, all_y_pred))

    # --- 特徴量重要度の集計と表示（修正ポイント） ---
    # DataFrameとして結合して平均を取る
    feat_imp = pd.concat(all_importances, axis=1).mean(axis=1).sort_values(ascending=False)

    print("\n===  Top 50 Features (CV Average Gain) ===")
    print(feat_imp.head(50))

# シグナル・フィルター (カスタム閾値の探索)
def search_optimal_thresholds(y_true, y_probs):
    print("\n" + "="*50)
    print("=== 閾値（しきい値）チューニング・シミュレーション ===")
    print("="*50)

    # テストする閾値の候補 (AIの自信度: 35% 〜 50%)
    thresholds_c1 = [0.40, 0.45, 0.50] # Class 1を発火させる最低ライン
    thresholds_c2 = [0.30, 0.35, 0.40] # Class 2を発火させる最低ライン

    for th1 in thresholds_c1:
        for th2 in thresholds_c2:
            y_pred_custom = []

            for probs in y_probs:
                prob_c1 = probs[0] # Class 1 の確率
                prob_c2 = probs[1] # Class 2 の確率

                # ルール1: Class 1 の確率が閾値を超えたら「1」
                if prob_c1 >= th1:
                    y_pred_custom.append(1.0)
                # ルール2: Class 2 の確率が閾値を超えたら「2」
                elif prob_c2 >= th2:
                    y_pred_custom.append(2.0)
                # ルール3: どちらも自信がない場合はすべて「3 (Synchronized)」で静観
                else:
                    y_pred_custom.append(3.0)

            y_pred_custom = np.array(y_pred_custom)

            # 結果の集計 (Precisionの確認)
            report = classification_report(y_true, y_pred_custom, output_dict=True, zero_division=0)
            prec_c1 = report['1.0']['precision']
            prec_c2 = report['2.0']['precision']
            acc = report['accuracy']

            print(f"閾値設定 [C1: {th1:.2f}, C2: {th2:.2f}] => Accuracy: {acc:.3f} | C1 Precision: {prec_c1:.3f} | C2 Precision: {prec_c2:.3f}")

            # もし詳細な混同行列が見たい場合は以下をコメントアウト解除
            # print(confusion_matrix(y_true, y_pred_custom))
            # print("-" * 40)


############################################################
# Light GBM -　AIが複数の特徴量から多くのルールを見つけ答えを予測する
############################################################
"""
LGBM 本番用の関数

【オプション機能】
    ・TimeSeriesSplitによる精度再検証
    ・特徴量重要度の表示
"""
def learning_lgbm_final(
    df_ready, target_col, model_name, label_name_list:list,
    # 学習パラメータ
    n_estimators=200, learning_rate=0.03, num_leaves=7, min_data_in_leaf=5,
    class_weight="balanced", reg_alpha=0.5, reg_lambda=0.5, importance_type='gain',
    # TimeSeriesSplitによる精度オプション
    option_tscv=False, n_splits=3, gap=3,
    # 特徴量重要度の表示オプション
    option_feat_imp=False
    ):

    print(f"\n--- {model_name} Master Model Training ---")

    # 1. データの分離
    df_trainable = df_ready.dropna(subset=target_col)
    df_live = df_ready[df_ready[target_col].isna()]   # ラベル未確定の直近期間

    X_train = df_trainable.drop(columns=[target_col])
    y_train = df_trainable[target_col]
    X_all = df_ready.drop(columns=[target_col]) # 可視化用（全期間）

    print(f" - Trainable Samples: {len(X_train)} (Up to {X_train.index[-1].date()})")
    print(f" - Live Prediction Samples: {len(df_live)} (Since {df_live.index[0].date() if len(df_live)>0 else 'N/A'})")

    # 2. 精度の再検証 (TimeSeriesSplit)
    if option_tscv:
        print("Evaluating Cross-Validation Performance...")
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        all_y_test, all_y_pred = [], []

        # モデル設定
        clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            #max_depth=4,
            num_leaves=num_leaves,
            min_data_in_leaf=min_data_in_leaf,
            class_weight=class_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            importance_type=importance_type,
            random_state=42,
            verbose=-1
        )
        for train_idx, test_idx in tscv.split(X_train):
            clf.fit(
                X_train.iloc[train_idx], y_train.iloc[train_idx],
                eval_set=[(X_train.iloc[test_idx], y_train.iloc[test_idx])],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
            all_y_test.extend(y_train.iloc[test_idx])
            all_y_pred.extend(clf.predict(X_train.iloc[test_idx]))

        # CV結果の表示
        print("\n[Overall CV Performance]")
        print(classification_report(all_y_test, all_y_pred))
        print("\n[Confusion Matrix]")
        print(confusion_matrix(all_y_test, all_y_pred))

    # 3. 本番学習
    master_clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        #max_depth=4,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        class_weight=class_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        importance_type=importance_type,
        random_state=42,
        verbose=-1
        )

    master_clf.fit(X_train, y_train)

    # 4. 全期間の予測確率を算出
    probs = master_clf.predict_proba(X_all)
    df_trajectory = pd.DataFrame(probs, index=X_all.index, columns=label_name_list)

    # 特徴量重要度の表示
    if option_feat_imp:
        feat_imp = pd.Series(master_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        print("\n[Top Features (Final Model)]")
        print(feat_imp.head(10))

    return master_clf, df_trajectory

def generate_oof_predictions(X, y, model_params):
    # TimeSeriesSplitの設定（既存の設定に合わせて調整してください）
    tscv = TimeSeriesSplit(n_splits=5, gap=20)

    oof_probs_list = []

    print("--- 真実の検証（OOF）を開始します ---")

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        # 1. データの分割
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 2. モデルの学習（その時点の過去データのみを使用）

        clf = lgb.LGBMClassifier(**model_params)
        clf.fit(X_train, y_train)

        # 3. 未知の期間（テストフォールド）に対する予測確率の算出
        probs = clf.predict_proba(X_test)

        # 4. 結果をデータフレームに保存
        regime_labels = [f"{float(c)}" for c in clf.classes_] # クラス名を動的に取得
        df_fold_probs = pd.DataFrame(probs, index=X_test.index, columns=regime_labels)

        # 正解ラベルも横に並べておく
        df_fold_probs['actual_regime'] = y_test

        oof_probs_list.append(df_fold_probs)
        print(f"Fold {i+1} の検証完了: {X_test.index[0].date()} ~ {X_test.index[-1].date()}")

    # 5. 全テストフォールドの予測を結合
    df_oof_all = pd.concat(oof_probs_list).sort_index()

    print("\n--- 検証完了 ---")
    return df_oof_all

############################################################
# SHAP -　LightBGMの予測に対し、その理由となる特徴量を示す（AIの理由）
############################################################

def learning_get_shap_df(model, X, start=None, end=None, class_idx=None, rolling=None, use_abs=False):

    # --- SHAP計算 ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # --- 多クラス対応（重要修正） ---
    if isinstance(shap_values, list):
        if class_idx is None:
            raise ValueError("多クラスモデルです。class_idxを指定してください。")
        shap_array = shap_values[class_idx]

    elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
        if class_idx is None:
            raise ValueError("多クラスモデルです。class_idxを指定してください。")
        shap_array = shap_values[:, :, class_idx]

    else:
        shap_array = shap_values

    # --- DataFrame化 ---
    shap_df = pd.DataFrame(
        shap_array,
        index=X.index,
        columns=X.columns
    )

    # --- 期間スライス ---
    if start is not None or end is not None:
        shap_df = shap_df.loc[start:end]

    # --- 絶対値オプション ---
    if use_abs:
        shap_df = shap_df.abs()

    # --- rolling平均 ---
    if rolling is not None:
        shap_df = shap_df.rolling(rolling).mean()

    return shap_df

def learning_get_shap_date(clf, df_ready, target_date=None, label_cols=['regime', 'driver']):
    """
    指定した日（デフォルトは最新）の予測根拠をSHAPで解剖する汎用関数
    """
    # 1. 特徴量とラベルの分離（リストにあるラベル列をすべて除外）
    X_all = df_ready.drop(columns=[col for col in label_cols if col in df_ready.columns])

    # モデルが学習した際の特徴量リストと順序を強制的に合わせる
    # (これを行わないと、列の並び順が違う場合にSHAPが誤作動します)
    model_features = clf.feature_name_
    X = X_all[model_features]

    # 2. 対象データの抽出
    if target_date is None:
        latest_x = X.tail(1)
        display_date = X.index[-1]
    else:
        # 指定日以前で最新の1件を取得
        latest_x = X[:target_date].tail(1)
        display_date = latest_x.index[0]

    # 3. SHAP Explainerの初期化
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(latest_x)

    # 4. 予測されたクラスの取得とインデックス変換
    pred_class = clf.predict(latest_x)[0]
    # クラスリストの中から、予測された値が何番目にあるかを探す
    # (1-5 などの数値だけでなく、文字列ラベルでも対応可能にするため)
    pred_idx = list(clf.classes_).index(pred_class)

    # 5. SHAPの戻り値形式の自動判別 (ユーザー様のロジックを継承・強化)
    if isinstance(shap_values, list):
        # 形式A: [class_idx][sample_idx, feature_idx]
        current_shap = shap_values[pred_idx][0]
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            # 形式B: [sample_idx, feature_idx, class_idx]
            current_shap = shap_values[0, :, pred_idx]
        else:
            # 形式C: 2次元の場合（バイナリ分類など）
            current_shap = shap_values[0]
    else:
        # 形式D: Explanationオブジェクト
        current_shap = shap_values.values[0, :, pred_idx]

    # 6. 特徴量名と貢献度を紐付け
    importance_df = pd.DataFrame({
        'feature': model_features,
        'contribution': current_shap
    }).sort_values(by='contribution', ascending=False)

    # --- 結果の確認 ---
    print(f"Predicted Class: {pred_class}")
    print(f"Model Classes: {clf.classes_}")
    probs = clf.predict_proba(latest_x)[0]
    for i, p in enumerate(probs):
        print(f"Class {clf.classes_[i]} Probability: {p:.2%}")

    return importance_df, pred_class, display_date



############################################################
# Ridge回帰、Lasso回帰 -　複数の説明変数からフィッティングで答えを導き出す
############################################################
def learning_regressions(X, y, cv_splits=5, purge_gap=4):

    # スケーリング（ペナルティ付き回帰の必須処理）
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # ★パージ付き交差検証 (gap=4 が「3ヶ月予測＋1ヶ月バッファのパージ」に相当)
    tscv = TimeSeriesSplit(n_splits=cv_splits, gap=purge_gap)

    ridge_model = Ridge(alpha=1.0) # alphaはペナルティの強さ
    lasso_model = Lasso(alpha=0.1)

    ridge_r2_scores, lasso_r2_scores = [], []
    ridge_coefs, lasso_coefs = [], []

    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, y_train = X_scaled.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X_scaled.iloc[test_idx], y.iloc[test_idx]

        # モデルの学習
        ridge_model.fit(X_train, y_train)
        lasso_model.fit(X_train, y_train)

        # 精度の記録 (R2スコア)
        ridge_r2_scores.append(r2_score(y_test, ridge_model.predict(X_test)))
        lasso_r2_scores.append(r2_score(y_test, lasso_model.predict(X_test)))

        # 係数の記録
        ridge_coefs.append(ridge_model.coef_)
        lasso_coefs.append(lasso_model.coef_)

    print(f"--- Purged CV ({cv_splits} Folds, Gap={purge_gap} months) ---")
    print(f"Ridge Mean R2: {np.mean(ridge_r2_scores):.3f}")
    print(f"Lasso Mean R2: {np.mean(lasso_r2_scores):.3f}")

    # 最終的な全データでの学習と係数の可視化
    ridge_model.fit(X_scaled, y)
    lasso_model.fit(X_scaled, y)

    return ridge_model, lasso_model

############################################################
# 異常検知 - Isolation Forest
############################################################
def learning_anomaly_detection(df, rule="final_regime"):
    # 1. 特徴量の選択（スケーリングが必要）
    X = df.dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Isolation Forest の実行
    # contamination: 異常データの割合（仮に全体の5%とする）
    model = IsolationForest(contamination=0.05, random_state=42)
    labels = model.fit_predict(X_scaled) # -1 が異常、1 が正常
    scores = model.decision_function(X_scaled) # スコアが低いほど異常

    # 3. 結果の統合
    result = X.copy()
    result["is_anomaly"] = labels
    result["anomaly_score"] = scores
    result["rule_regime"] = df[rule]

    # 4. 「不一致」の抽出
    # ルールでは平和（1）なのに、統計的に異常（-1）な日
    discrepancy = result[(result["rule_regime"] == 1) & (result["is_anomaly"] == -1)]

    return discrepancy, result

############################################################
# PCA
############################################################
# フレームワーク
def learning_pca(df, n_components=1, output_name="output", sign_name=""):
    # 1. 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # 2. PCA実行
    pca = PCA(n_components=1)
    factor = pca.fit_transform(X_scaled)

    # 3. 符号の調整
    if sign_name:
        feature_names = df.columns.tolist()
        sign_idx = feature_names.index(sign_name)
        loadings = pca.components_[0]
        if loadings[sign_idx] < 0:
            factor = -factor
            print(f"INFO: Factor inverted. Correlation with {sign_name}")

    df[output_name] = factor

    return df, pca
