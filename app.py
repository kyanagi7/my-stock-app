import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta

# --- 1. 銘柄と目標の設定 ---
# '銘柄コード': [目標金額, '購入' または '売却']
TICKERS_CONFIG = {
    '5970.T': [2070, '売却'],
    '7272.T': [1225, '売却'],
    '8306.T': [3050, '売却'],
    '8316.T': [5700, '売却'],
    '9101.T': [4950, '購入'],
}

st.set_page_config(page_title="Stock Target Tracker", layout="centered")
st.title("📈 銘柄別・売買目標管理")

@st.cache_data(ttl=600)
def get_stock_info(ticker):
    tk = yf.Ticker(ticker)
    long_name = tk.info.get('longName', ticker)
    df = tk.history(period="2y")
    if df.empty:
        return None, None
    return long_name, df['Close']

# --- メイン処理 ---
for ticker, config in TICKERS_CONFIG.items():
    target_price = config[0]
    target_type = config[1] # '購入' or '売却'
    
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
            
            # --- 目標判定ロジック ---
            dist_to_target = current_price - target_price
            dist_percent = (abs(dist_to_target) / target_price) * 100

            # 達成条件の判定
            is_achieved = False
            if target_type == '購入':
                if current_price <= target_price:
                    is_achieved = True
            else: # 売却
                if current_price >= target_price:
                    is_achieved = True

            # 表示
            c1, c2 = st.columns(2)
            c1.metric("現在値", f"{unit}{current_price:,.1f}", f"{diff:+,.1f}")
            c2.metric(f"{target_type}目標", f"{unit}{target_price:,.0f}")
            
            # 達成状況に応じたメッセージと色の出し分け
            if is_achieved:
                st.success(f"✨ 【{target_type}判定】目標を達成しています！")
            else:
                if target_type == '購入':
                    st.warning(f"⏳ 【購入待ち】目標まで あと **{unit}{dist_to_target:,.1f}** ({dist_percent:.2f}%) 安くなるのを待機中")
                else:
                    st.info(f"🚀 【売却待ち】目標まで あと **{unit}{abs(dist_to_target):,.1f}** ({dist_percent:.2f}%) の上昇が必要です")

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

            # --- グラフ描画 ---
            fig = go.Figure()
            hist_plot = df_p.tail(30)
            fig.add_trace(go.Scatter(x=hist_plot['ds'], y=hist_plot['y'], name='実績', line=dict(color='#333')))
            
            # 目標価格の横線
            line_color = "#28a745" if target_type == '購入' else "#dc3545" # 購入なら緑、売却なら赤
            fig.add_hline(y=target_price, line_dash="dash", line_color=line_color, 
                          annotation_text=f"{target_type}目標", annotation_position="top left")
            
            fore_plot = forecast[forecast['ds'] >= hist_plot['ds'].iloc[-1]].head(8)
            fig.add_trace(go.Scatter(x=fore_plot['ds'], y=fore_plot['yhat'], name='予測', line=dict(color='#0066ff', dash='dot')))
            
            fig.update_layout(height=180, margin=dict(l=0,r=0,b=0,t=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"{ticker} エラー: {e}")
