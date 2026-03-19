# cspell:ignore tickformat
import plotly.io as pio
import plotly.graph_objects as go

# ----- ベクトル表示 -----
def make_vector(current, previous):
    if previous == 0:
        return 0
    rate = current / previous
    if rate > 1.005:
        return 1
    elif rate < 0.995:
        return -1
    else:
        return 0

# ----- グラフ設定 -----
FINANCE_DARK_LUXE = {
    "bg": "#0E1117",
    "grid": "#2A2F3A",
    "text": "#E6E6E6",
    "text2": "#A9AFBB",
    "muted": "#A9AFBB",
    
    # line colors
    "soft_gold": "#D4B96A",
    "Steel Blue": "#4C78A8",
    "Muted Teal": "#3FA7A3",
    "Muted Purple": "#7A6C9D",
    "Ash Cyan": "#8FB9C8",
    # bar colors
    "Gold Gray": "#C7B77A",
    "Neutral Blue": "#4C78A8",
    "Neutral Gray": "#7C828D",
    "Muted Teal": "#3FA7A3",
    "Muted Purple": "#7A6C9D",
    # bubble colors
    "accent_gold": "#D4B96A",
    "deep blue-gray": "#1B2636",
    "Muted Purple": "#344B6C",
    # treemap colors
    "dark_slate": "#2A2F3A",
    "Gold Gray": "#6B6F7A",
    "soft_gold": "#D4B96A",

    "positive": "#4CAF7D",
    "negative": "#D06C6C",
}

def make_graph_template():
    theme = go.layout.Template(
        layout=go.Layout(
            autosize=True, margin=dict(l=0,r=0,t=0,b=0,pad=0),
            #paper_bgcolor="#1f1f1f",
            paper_bgcolor=FINANCE_DARK_LUXE["bg"],
            #plot_bgcolor="#1f1f1f",
            plot_bgcolor=FINANCE_DARK_LUXE["bg"],
            font=dict(family="Inter, Roboto", size=14, color="#DDDDDD"),

            xaxis=dict(
                title=dict(font_size=12),
                title_standoff=16,
                tickfont=dict(size=10),
                showgrid=True,
                #gridcolor="#444444",
                gridcolor=FINANCE_DARK_LUXE["grid"],
                zeroline=False,
                #color="#cccccc"
                color=FINANCE_DARK_LUXE["text"]
            ),
            
            yaxis=dict(
                title=dict(font_size=12),
                title_standoff=16,
                separatethousands=False,
                tickfont=dict(size=10),
                showgrid=True,
                #gridcolor="#444444",
                gridcolor=FINANCE_DARK_LUXE["grid"],
                zeroline=False,
                #color="#cccccc"
                color=FINANCE_DARK_LUXE["text"]
            ),
            
            legend=dict(
                visible=True,
                orientation="h",
                yanchor="top",
                y=1.2,
                xanchor="right",
                x=1,               
            ),
        )
    )
    pio.templates["dark_dashboard"] = theme
    pio.templates.default = "plotly_dark+dark_dashboard"

def graph_individual_setting(fig, x_title, x_tickformat, y_title, y_tickprefix, y_tickformat):
    fig.update_xaxes(
        title = dict(text = x_title),
        tickformat=x_tickformat
    )
    fig.update_yaxes(
        title = dict(text = y_title),
        tickprefix=y_tickprefix,
        tickformat=y_tickformat,
    )
    return fig

# ----- マッピング ----- 
def get_map_jp_to_en_sub_type(df_item_attribute):
    return dict(zip(
        df_item_attribute["項目"],
        df_item_attribute["英語名"]
    ))

def get_map_en_to_jp_sub_type(df_item_attribute):
    return dict(zip(
        df_item_attribute["英語名"],
        df_item_attribute["項目"]
    ))

def get_map_subtype_to_purpose(df_item_attribute):
    return dict(zip(
        df_item_attribute["項目"],
        df_item_attribute["資産目的"]
    ))

def get_map_asset_name_to_asset_class(df_asset_attribute):
    df = df_asset_attribute[df_asset_attribute["アセットクラス"].notna()]
    return dict(zip(
        df["資産名"],
        df["アセットクラス"]
    ))

def get_map_index_name_to_asset_class():
    return {
        "nikkei": "日本株式",
        "sp500": "米国株式",
        "acwi_world": "全世界株式",
        "efa": "先進国株式",
        "eem": "新興国株式",
        "agg": "米国債券",
        "agbp": "全世界債券",
        "bndx": "先進国債券",
        "jp_bond": "日本債券",
        "emb": "新興国債券",
        "wgbi": "全世界国債",
        "iyr": "米国不動産",
        "usdjpy": "ドル",
        "eurjpy": "ユーロ",
        "audjpy": "オーストラリアドル",
        "chfjpy": "フラン",
        "btc-usd": "ビットコイン",
        "eth-usd": "イーサリアム",
        "dot-usd": "ポルカドット",
        "enj-usd": "エンジンコイン",
        "tlt": "米国国債長期債",
        "hyg": "ハイイールド債スプレッド",
        "uup": "ドル流動性"
    }

# ----- リターン集計 -----
def get_sub_type_by_return_method(method:str, df_item_attribute) -> list:
    return df_item_attribute[
        df_item_attribute["リターン計算方法"] == method
    ]["項目"].tolist()

def get_agg_data_by_return_agg_unit(df_asset_market_all, df_item_attribute, df_asset_attribute):
    df = df_asset_market_all.copy()
    # 集計先の資産名
    account_list = (
        df_asset_attribute[
            df_asset_attribute["資産名"].isin(df["資産名"].unique())
        ]["リターン集計単位"].dropna().unique()
    )

    df.set_index("date", inplace=True)
    for account in account_list:
        asset_list = df_asset_attribute.loc[
            df_asset_attribute["リターン集計単位"] == account, "資産名"
        ].unique()
        
        df.loc[df["資産名"] == account, "資産額"] += (
            df[df["資産名"].isin(asset_list)].groupby("date")["資産額"].sum()
        )
        df = df[~df["資産名"].isin(asset_list)]

    return df.reset_index()

