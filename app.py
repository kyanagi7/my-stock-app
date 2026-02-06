import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime, time, timedelta

# --- 1. 銘柄と目標設定 ---
TICKERS_CONFIG = {
    '5970.T': [1970, '売却'],
    '7272.T': [1082, '売却'],
    '7731.T': [1800, '売却'],
    '8306.T': [2950, '売却'],
    '3245.T': [1085, '購入'],
    '9101.T': [5000, '購入'],
}

st.set_page_config(page_title="Stock Trading Advisor", layout="centered")
st.title("⚖️ 戦略的株価分析ボード")

# --- 表示設定 ---
PERIOD_OPTIONS = {
    "6か月": {"days": 180, "interval": "1d", "pred_len": 14, "pred_freq": "D", "label": "2週間先"},
    "3か月": {"days": 90, "interval": "1d", "pred_len": 10, "pred_freq": "D", "label": "10日先"},
    "1か月": {"days": 30, "interval": "1d", "pred_len": 7, "pred_freq": "D", "label": "1週間先"},
    "1週間": {"days": 7, "interval": "30m", "pred_len": 16, "pred_freq": "30min", "label": "数日先"},
    "1日": {"days": 1, "interval": "5m", "pred_len": 24, "pred_freq": "5min", "label": "今日の大引け"}
}

selected_label = st.segmented_control("表示期間", options=list(PERIOD_OPTIONS.keys()), default="1か月")
v = PERIOD_OPTIONS[selected_label]

@st.cache_data(ttl=600)
def get_stock_data(ticker, interval):
    tk = yf.Ticker(ticker)
    period_map = {"5m": "5d", "30m": "15d", "1d": "2y"}
    df = tk.history(period=period_map[interval], interval=interval)
    if not df.empty:
        df.index = df.index.tz_convert('Asia/Tokyo').tz_localize(None)
    # 前日終値取得用
    hist_daily = tk.history(period="5d", interval="1d")
    prev_close = hist_daily['Close'].iloc[-2] if len(hist_daily) > 1 else df['Close'].iloc[0]
    return df, prev_close

def get_advice(current_price, rsi, upper, lower):
    if rsi >= 70 or current_price >= upper:
        return "⚠️ 売り検討", "過熱気味です。利益確定を優先してください。", "error"
    elif rsi <= 30 or current_price <= lower:
        return "💎 買い検討", "売られすぎです。反発のサインが出ています。", "success"
    else:
        return "😐 様子見", "トレンドは安定しています。継続保有でOKです。", "info"

# --- メイン処理 ---
for ticker, config in TICKERS_CONFIG.items():
    target_price, target_type = config[0], config[1]
    
    with st.spinner(f'{ticker}...'):
        df, prev_close = get_stock_data(ticker, v["interval"])
        tk = yf.Ticker(ticker)
        name = tk.info.get('longName', ticker)
    
    if df is None or df.empty: continue

    with st.expander(f"📌 {name} ({ticker})", expanded=True):
        try:
            last_dt = df.index[-1]
            if selected_label == "1日":
                day_start = last_dt.replace(hour=9, minute=0, second=0)
                day_end = last_dt.replace(hour=15, minute=30, second=0)
                hist_display = df.loc[day_start:day_end]
            else:
                hist_display = df.tail(v["days"] if v["interval"] == "1d" else 100)

            current_price = float(hist_display['Close'].iloc[-1])
            
            # --- 数値計算 ---
            is_achieved = (current_price <= target_price) if target_type == '購入' else (current_price >= target_price)
            color = "#FF4B4B" if is_achieved else "#1F77B4" # 達成=赤(Hot), 未達成=青(Cool)
            
            # 前日比
            price_diff = current_price - prev_close
            price_pct = (price_diff / prev_close) * 100
            
            # 目標比
            target_diff = current_price - target_price
            target_pct = (target_diff / target_price) * 100

            # テクニカル
            rsi_val = 100 - (100 / (1 + (df['Close'].diff().where(lambda x: x > 0, 0).rolling(14).mean() / 
                                        (-df['Close'].diff().where(lambda x: x < 0, 0)).rolling(14).mean()))).iloc[-1]

            # --- UI表示 ---
            # 1. アドバイス
            status, msg, style = get_advice(current_price, rsi_val, 0, 0) # 簡易判定
            if style == "success": st.success(f"**{status}**: {msg}")
            elif style == "error": st.error(f"**{status}**: {msg}")
            else: st.info(f"**{status}**: {msg}")

            # 2. メトリクス表示 (カスタムHTML)
            c1, c2 = st.columns([1.2, 1])
            
            with c1:
                st.markdown(f"""
                    <div style="line-height:1;">
                        <p style="margin:0; font-size:0.9rem; color:gray;">現在値</p>
                        <p style="margin:0; font-size:1.8rem; font-weight:bold;">¥{current_price:,.1f}</p>
                        <p style="margin:0; font-size:1.0rem; color:{color}; font-weight:bold;">
                            前日比: {price_diff:+,.1f} ({price_pct:+.2f}%)
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                    <div style="line-height:1;">
                        <p style="margin:0; font-size:0.8rem; color:gray;">{target_type}目標</p>
                        <p style="margin:0; font-size:1.2rem; font-weight:bold;">¥{target_price:,.0f}</p>
                        <p style="margin:0; font-size:0.9rem; color:{color}; font-weight:bold;">
                            目標差: {target_diff:+,.1f} ({target_pct:+.2f}%)
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            # 3. グラフ
            df_p = df['Close'].reset_index()
            df_p.columns = ['ds', 'y']
            model = Prophet(daily_seasonality=True, weekly_seasonality=True).fit(df_p)
            forecast = model.predict(model.make_future_dataframe(periods=v["pred_len"], freq=v["pred_freq"]))

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=hist_display.index, y=hist_display['Close'], name='実績', line=dict(color='#0055FF', width=3)), row=1, col=1)
            
            line_color = "#28a745" if target_type == '購入' else "#dc3545"
            fig.add_hline(y=target_price, line_dash="dash", line_color=line_color, row=1, col=1)
            
            fore_plot = forecast[forecast['ds'] >= hist_display.index[-1]].head(v["pred_len"] + 1)
            if selected_label == "1日": fore_plot = fore_plot[fore_plot['ds'] <= day_end]
            if not fore_plot.empty:
                pred_c = "#FF0000" if fore_plot['yhat'].iloc[-1] >= current_price else "#0000FF"
                fig.add_trace(go.Scatter(x=fore_plot['ds'], y=fore_plot['yhat'], name='予測', line=dict(color=pred_c, dash='dot', width=3)), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=hist_display.index, y=(100 - (100 / (1 + (df['Close'].diff().where(lambda x: x > 0, 0).rolling(14).mean() / (-df['Close'].diff().where(lambda x: x < 0, 0)).rolling(14).mean())))).loc[hist_display.index], name='RSI', line=dict(color='#8A2BE2')), row=2, col=1)
            
            if selected_label == "1日":
                fig.update_xaxes(range=[day_start, day_end], row=1, col=1)
                fig.update_xaxes(range=[day_start, day_end], row=2, col=1)

            fig.update_layout(height=420, margin=dict(l=0,r=0,b=0,t=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"分析失敗: {e}")

