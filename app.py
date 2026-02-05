import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta

# --- 1. ポートフォリオ設定 ---
MY_PORTFOLIO = {
    '7203.T': 100,  # トヨタ
    'AAPL': 10,     # Apple
    '7974.T': 50,   # 任天堂
}

st.set_page_config(page_title="Stock Expert Pro", layout="centered")
st.title("📊 My Stock Dashboard")

@st.cache_data(ttl=600) # 10分ごとに更新（お昼休み用）
def get_data(ticker):
    df = yf.download(ticker, period="2y", interval="1d")
    if df.empty:
        return None
    if 'Close' in df.columns:
        close_data = df['Close']
        if isinstance(close_data, pd.DataFrame):
            close_data = close_data.iloc[:, 0]
        return close_data
    return None

# --- メイン処理 ---
for ticker, shares in MY_PORTFOLIO.items():
    # 通貨単位の判定
    unit = "¥" if ticker.endswith(".T") else "$"
    
    with st.expander(f"📌 {ticker} (保有: {shares}株)", expanded=True):
        try:
            prices = get_data(ticker)
            if prices is None or len(prices) < 25:
                st.warning(f"{ticker}: データ取得失敗")
                continue
                
            current_price = float(prices.iloc[-1])
            prev_price = float(prices.iloc[-2])
            diff = current_price - prev_price
            
            # --- 【追加】現在の値を目立たせる ---
            st.markdown(f"### 現在値: {unit}{current_price:,.1f}")
            
            # 変動率計算
            change_today = (current_price / prev_price - 1) * 100
            change_week = (current_price / prices.iloc[-5] - 1) * 100 if len(prices) > 5 else 0
            change_month = (current_price / prices.iloc[-21] - 1) * 100 if len(prices) > 21 else 0

            # メトリクス表示
            m1, m2, m3 = st.columns(3)
            m1.metric("本日比", f"{unit}{diff:+,.1f}", f"{change_today:+.2f}%")
            m2.metric("今週", f"{change_week:+.2f}%")
            m3.metric("1ヶ月", f"{change_month:+.2f}%")

            # --- AI予測 (Prophet) ---
            df_p = prices.reset_index()
            df_p.columns = ['ds', 'y']
            df_p['ds'] = pd.to_datetime(df_p['ds']).dt.tz_localize(None)
            df_p['y'] = pd.to_numeric(df_p['y'], errors='coerce')

            model = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
            model.fit(df_p)
            future = model.make_future_dataframe(periods=14)
            forecast = model.predict(future)

            pred_tonight = forecast.iloc[len(df_p)]['yhat']
            pred_tomorrow = forecast.iloc[len(df_p)+1]['yhat']
            pred_next_week = forecast.iloc[len(df_p)+6]['yhat']

            st.write("🔮 **AI予測価格**")
            p1, p2, p3 = st.columns(3)
            p1.caption("本日夜")
            p1.write(f"**{unit}{pred_tonight:,.1f}**")
            p2.caption("明日")
            p2.write(f"**{unit}{pred_tomorrow:,.1f}**")
            p3.caption("来週")
            p3.write(f"**{unit}{pred_next_week:,.1f}**")

            # --- グラフ描画 ---
            fig = go.Figure()
            hist_plot = df_p.tail(30)
            fig.add_trace(go.Scatter(x=hist_plot['ds'], y=hist_plot['y'], name='実績', line=dict(color='#333')))
            fore_plot = forecast[forecast['ds'] >= hist_plot['ds'].iloc[-1]].head(8)
            fig.add_trace(go.Scatter(x=fore_plot['ds'], y=fore_plot['yhat'], name='予測', line=dict(color='#0066ff', dash='dot')))
            fig.add_trace(go.Scatter(x=fore_plot['ds'], y=fore_plot['yhat_upper'], fill='tonexty', mode='none', fillcolor='rgba(0,102,255,0.1)', showlegend=False))
            fig.add_trace(go.Scatter(x=fore_plot['ds'], y=fore_plot['yhat_lower'], fill='tonexty', mode='none', fillcolor='rgba(0,102,255,0.1)', showlegend=False))
            fig.update_layout(height=200, margin=dict(l=0,r=0,b=0,t=10), hovermode="x unified", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"{ticker} エラー: {e}")
