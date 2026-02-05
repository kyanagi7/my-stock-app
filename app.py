import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta

# --- 1. 銘柄と目標単価の設定 ---
# '銘柄コード': 目標金額
TICKERS_CONFIG = {
    '5970.T': 1970,
    '7272.T': 1075,
    '8306.T': 2950,
    '8316.T': 5470,
    '9101.T': 4950,
}

st.set_page_config(page_title="Stock Target Tracker", layout="centered")
st.title("📈 銘柄別・目標株価管理")

@st.cache_data(ttl=600)
def get_stock_info(ticker):
    tk = yf.Ticker(ticker)
    # 銘柄名を取得
    long_name = tk.info.get('longName', ticker)
    df = tk.history(period="2y")
    if df.empty:
        return None, None
    return long_name, df['Close']

# --- メイン処理 ---
for ticker, target_price in TICKERS_CONFIG.items():
    
    with st.spinner(f'{ticker} を読み込み中...'):
        name, prices = get_stock_info(ticker)
    
    if prices is None:
        st.error(f"{ticker}: データ取得失敗")
        continue

    unit = "¥"
    
    with st.expander(f"📌 {name} ({ticker})", expanded=True):
        try:
            current_price = float(prices.iloc[-1])
            prev_price = float(prices.iloc[-2])
            diff = current_price - prev_price
            
            # 目標値との比較計算
            dist_to_target = current_price - target_price
            dist_percent = (dist_to_target / target_price) * 100

            # --- 現在値と目標値の表示 ---
            c1, c2 = st.columns(2)
            c1.metric("現在値", f"{unit}{current_price:,.1f}", f"{diff:+,.1f}")
            c2.metric("目標単価", f"{unit}{target_price:,.0f}")
            
            # 目標までの進捗をメッセージ表示
            if dist_to_target >= 0:
                st.success(f"🎉 目標達成中！ (目標比: {dist_percent:+.2f}%)")
            else:
                st.info(f"🚀 目標まであと **{unit}{abs(dist_to_target):,.1f}** ({abs(dist_percent):.2f}%)")

            # --- AI予測 (Prophet) ---
            df_p = prices.reset_index()
            df_p.columns = ['ds', 'y']
            df_p['ds'] = pd.to_datetime(df_p['ds']).dt.tz_localize(None)
            
            model = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
            model.fit(df_p)
            future = model.make_future_dataframe(periods=14)
            forecast = model.predict(future)

            st.write("🔮 **株価予想**")
            p1, p2, p3 = st.columns(3)
            p1.caption("本日夜")
            p1.write(f"**{unit}{forecast.iloc[len(df_p)]['yhat']:,.1f}**")
            p2.caption("明日")
            p2.write(f"**{unit}{forecast.iloc[len(df_p)+1]['yhat']:,.1f}**")
            p3.caption("来週")
            p3.write(f"**{unit}{forecast.iloc[len(df_p)+6]['yhat']:,.1f}**")

            # --- グラフ描画（目標線を赤色で表示） ---
            fig = go.Figure()
            hist_plot = df_p.tail(30)
            fig.add_trace(go.Scatter(x=hist_plot['ds'], y=hist_plot['y'], name='実績', line=dict(color='#333')))
            
            # 目標価格の横線
            fig.add_hline(y=target_price, line_dash="dash", line_color="#FF4B4B", 
                          annotation_text="目標", annotation_position="top left")
            
            fore_plot = forecast[forecast['ds'] >= hist_plot['ds'].iloc[-1]].head(8)
            fig.add_trace(go.Scatter(x=fore_plot['ds'], y=fore_plot['yhat'], name='予測', line=dict(color='#0066ff', dash='dot')))
            
            fig.update_layout(height=180, margin=dict(l=0,r=0,b=0,t=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"{ticker} エラー: {e}")
