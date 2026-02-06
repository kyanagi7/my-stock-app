import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime, timedelta

# --- 1. 銘柄と目標設定 ---
TICKERS_CONFIG = {
    '5970.T': [2070, '売却'],
    '7272.T': [1225, '売却'],
    '8306.T': [3050, '売却'],
    '8316.T': [5700, '売却'],
    '9101.T': [4950, '購入'],
}

st.set_page_config(page_title="Stock Trading Advisor", layout="centered")
st.title("⚖️ テクニカル判定 & 同期予測")

# --- 表示と予測の連動設定 ---
PERIOD_OPTIONS = {
    "6か月": {"days": 180, "interval": "1d", "pred_len": 14, "pred_freq": "D", "label": "2週間先まで"},
    "3か月": {"days": 90, "interval": "1d", "pred_len": 10, "pred_freq": "D", "label": "10日先まで"},
    "1か月": {"days": 30, "interval": "1d", "pred_len": 7, "pred_freq": "D", "label": "1週間先まで"},
    "1週間": {"days": 7, "interval": "30m", "pred_len": 16, "pred_freq": "30min", "label": "数営業日先まで"},
    "1日": {"days": 1, "interval": "5m", "pred_len": 24, "pred_freq": "5min", "label": "数時間先まで"}
}

selected_label = st.segmented_control(
    "表示期間を選択", 
    options=list(PERIOD_OPTIONS.keys()), 
    default="1か月"
)
v = PERIOD_OPTIONS[selected_label]

@st.cache_data(ttl=600)
def get_stock_data(ticker, interval):
    tk = yf.Ticker(ticker)
    # 予測精度のため、短期間表示でも多めにデータを取得
    period_map = {"5m": "5d", "30m": "15d", "1d": "2y"}
    df = tk.history(period=period_map[interval], interval=interval)
    if not df.empty:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def get_advice(current_price, rsi, upper, lower):
    if rsi >= 70 or current_price >= upper:
        return "⚠️ 売り検討", "過熱気味です。利益確定を検討するか、新規購入は控えましょう。", "error"
    elif rsi <= 30 or current_price <= lower:
        return "💎 買い検討", "売られすぎです。反発のチャンスかもしれません。", "success"
    else:
        return "😐 様子見", "過熱感はありません。トレンドに沿った運用を継続しましょう。", "info"

# --- メイン処理 ---
for ticker, config in TICKERS_CONFIG.items():
    target_price, target_type = config[0], config[1]
    
    with st.spinner(f'{ticker} を読み込み中...'):
        df = get_stock_data(ticker, v["interval"])
        tk = yf.Ticker(ticker)
        name = tk.info.get('longName', ticker)
    
    if df is None or df.empty: continue

    with st.expander(f"📌 {name} ({ticker})", expanded=True):
        try:
            # 表示用にフィルタリング
            hist_display = df.tail(v["days"] if v["interval"] == "1d" else 100) # 分足の場合は直近100件程度
            current_price = float(hist_display['Close'].iloc[-1])
            
            # テクニカル計算（表示範囲のデータを使用）
            close = df['Close']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi_series = 100 - (100 / (1 + (gain / loss)))
            ma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            upper_s, lower_s = ma20 + (std20 * 2), ma20 - (std20 * 2)

            # 判定とメトリクス
            status, msg, style = get_advice(current_price, rsi_series.iloc[-1], upper_s.iloc[-1], lower_s.iloc[-1])
            st.subheader(f"判定: {status}")
            if style == "success": st.success(msg)
            elif style == "error": st.error(msg)
            else: st.info(msg)

            c1, c2, c3 = st.columns(3)
            c1.metric("現在値", f"¥{current_price:,.1f}")
            c2.metric(f"{target_type}目標", f"¥{target_price:,.0f}")
            c3.metric("RSI", f"{rsi_series.iloc[-1]:.1f}")

            # --- AI予測（表示解像度と同期） ---
            df_p = df['Close'].reset_index()
            df_p.columns = ['ds', 'y']
            df_p['ds'] = pd.to_datetime(df_p['ds']).dt.tz_localize(None)
            
            # 期間が短い場合は、その分足データで直近のトレンドを予測
            model = Prophet(daily_seasonality=True, weekly_seasonality=True).fit(df_p)
            future = model.make_future_dataframe(periods=v["pred_len"], freq=v["pred_freq"])
            forecast = model.predict(future)

            # --- グラフ描画 ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
            
            # 実績
            fig.add_trace(go.Scatter(x=hist_display.index, y=hist_display['Close'], name='実績', 
                                     line=dict(color='#0055FF', width=3)), row=1, col=1)
            
            # ボリンジャーバンド
            fig.add_trace(go.Scatter(x=hist_display.index, y=upper_s.loc[hist_display.index], name='BB上', line=dict(width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist_display.index, y=lower_s.loc[hist_display.index], name='BB下', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,150,255,0.1)', showlegend=False), row=1, col=1)
            
            fig.add_hline(y=target_price, line_dash="dash", line_color=("#28a745" if target_type == '購入' else "#dc3545"), row=1, col=1)
            
            # 予測線（解像度と期間を同期）
            fore_plot = forecast[forecast['ds'] >= hist_display.index[-1]].head(v["pred_len"] + 1)
            pred_color = "#FF0000" if fore_plot['yhat'].iloc[-1] >= current_price else "#0000FF"
            fig.add_trace(go.Scatter(x=fore_plot['ds'], y=fore_plot['yhat'], name='予測', 
                                     line=dict(color=pred_color, dash='dot', width=3)), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=hist_display.index, y=rsi_series.loc[hist_display.index], name='RSI', line=dict(color='#8A2BE2')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="#FF4B4B", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="#4B4BFF", row=2, col=1)
            
            fig.update_layout(height=480, margin=dict(l=0,r=0,b=0,t=10), showlegend=False, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # 予測メッセージの最適化
            pred_price = fore_plot['yhat'].iloc[-1]
            st.write(f"🔮 **AI予測 ({v['label']}):** 約 ¥{pred_price:,.1f} ({'上昇' if pred_price >= current_price else '下落'}傾向)")

        except Exception as e:
            st.error(f"分析失敗: {e}")
