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

st.set_page_config(page_title="Stock Expert", layout="centered")
st.title("📊 Individual Stock Analysis")

@st.cache_data(ttl=3600)
def get_data(ticker):
    df = yf.download(ticker, period="2y", interval="1d")
    return df['Close']

# --- メイン処理 ---
for ticker, shares in MY_PORTFOLIO.items():
    with st.expander(f"📌 {ticker} (保有: {shares}株)", expanded=True):
        try:
            # データ取得
            prices = get_data(ticker)
            current_price = prices.iloc[-1]
            
            # --- 変動率の計算 ---
            change_today = (current_price / prices.iloc[-2] - 1) * 100
            change_week = (current_price / prices.iloc[-5] - 1) * 100
            change_month = (current_price / prices.iloc[-21] - 1) * 100

            # 実績メトリクス表示
            m1, m2, m3 = st.columns(3)
            m1.metric("本日", f"{change_today:+.2f}%")
            m2.metric("今週", f"{change_week:+.2f}%")
            m3.metric("1ヶ月", f"{change_month:+.2f}%")

            # --- AI予測 (Prophet) ---
            df_p = prices.reset_index()
            df_p.columns = ['ds', 'y']
            df_p['ds'] = df_p['ds'].dt.tz_localize(None)

            model = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
            model.fit(df_p)
            future = model.make_future_dataframe(periods=14) # 2週間分予測
            forecast = model.predict(future)

            # 予測値の抽出
            # yhatが予測の中央値
            pred_tonight = forecast.iloc[-14]['yhat'] # 本日（最新データから1日後相当）
            pred_tomorrow = forecast.iloc[-13]['yhat'] # 明日
            pred_next_week = forecast.iloc[-7]['yhat'] # 1週間後

            # 予測メトリクス表示
            st.write("🔮 **AI予測価格**")
            p1, p2, p3 = st.columns(3)
            p1.caption("本日夜")
            p1.write(f"**{pred_tonight:,.1f}**")
            p2.caption("明日")
            p2.write(f"**{pred_tomorrow:,.1f}**")
            p3.caption("来週")
            p3.write(f"**{pred_next_week:,.1f}**")

            # --- グラフ描画 ---
            fig = go.Figure()
            # 実績（直近30日）
            hist_30 = df_p.tail(30)
            fig.add_trace(go.Scatter(x=hist_30['ds'], y=hist_30['y'], name='実績', line=dict(color='#333')))
            # 予測（未来7日）
            fore_7 = forecast.tail(14).head(8)
            fig.add_trace(go.Scatter(x=fore_7['ds'], y=fore_7['yhat'], name='予測', line=dict(color='#0066ff', dash='dot')))
            # 予測の幅
            fig.add_trace(go.Scatter(x=fore_7['ds'], y=fore_7['yhat_upper'], fill='tonexty', mode='none', fillcolor='rgba(0,102,255,0.1)', showlegend=False))
            fig.add_trace(go.Scatter(x=fore_7['ds'], y=fore_7['yhat_lower'], fill='tonexty', mode='none', fillcolor='rgba(0,102,255,0.1)', showlegend=False))

            fig.update_layout(height=250, margin=dict(l=0,r=0,b=0,t=20), hovermode="x unified", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"{ticker} の解析中にエラーが発生しました。")
