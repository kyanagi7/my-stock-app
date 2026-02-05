import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime, timedelta

# --- 1. 銘柄と目標の最新設定 ---
TICKERS_CONFIG = {
    '5970.T': [2070, '売却'],
    '7272.T': [1225, '売却'],
    '8306.T': [3050, '売却'],
    '8316.T': [5700, '売却'],
    '9101.T': [4950, '購入'],
}

st.set_page_config(page_title="Stock Expert Pro+", layout="centered")
st.title("📊 テクニカル分析 & 株価予測")

@st.cache_data(ttl=600)
def get_stock_data(ticker):
    tk = yf.Ticker(ticker)
    name = tk.info.get('longName', ticker)
    df = tk.history(period="2y")
    if df.empty:
        return None, None
    
    # --- テクニカル指標の計算 ---
    close = df['Close']
    
    # 1. RSI (14日間)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. ボリンジャーバンド (20日間)
    df['MA20'] = close.rolling(window=20).mean()
    df['STD20'] = close.rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['STD20'] * 2)
    df['Lower'] = df['MA20'] - (df['STD20'] * 2)
    
    return name, df

# --- メイン処理 ---
for ticker, config in TICKERS_CONFIG.items():
    target_price, target_type = config[0], config[1]
    
    with st.spinner(f'{ticker} を解析中...'):
        name, df = get_stock_data(ticker)
    
    if df is None:
        continue

    unit = "¥"
    with st.expander(f"📌 {name} ({ticker})", expanded=True):
        try:
            current_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            
            # 判定ロジック
            is_achieved = (current_price <= target_price) if target_type == '購入' else (current_price >= target_price)

            # メトリクス表示
            c1, c2 = st.columns(2)
            c1.metric("現在値", f"{unit}{current_price:,.1f}", f"{current_price-prev_price:+,.1f}")
            c2.metric(f"{target_type}目標", f"{unit}{target_price:,.0f}")
            
            if is_achieved:
                st.success(f"✨ 【{target_type}判定】目標を達成しています！")
            else:
                dist = abs(current_price - target_price)
                if target_type == '購入':
                    st.warning(f"⏳ あと {unit}{dist:,.1f} の下落で購入目標到達")
                else:
                    st.info(f"🚀 あと {unit}{dist:,.1f} の上昇で売却目標到達")

            # --- AI予測 (Prophet) ---
            df_p = df['Close'].reset_index()
            df_p.columns = ['ds', 'y']
            df_p['ds'] = df_p['ds'].dt.tz_localize(None)
            model = Prophet(daily_seasonality=True).fit(df_p)
            future = model.make_future_dataframe(periods=14)
            forecast = model.predict(future)

            # --- グラフ描画 (株価 + RSI) ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.1, row_heights=[0.7, 0.3])

            hist_plot = df.tail(40) # 直近40日を表示
            
            # メインチャート: 株価 + ボリンジャーバンド
            fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['Close'], name='実績', line=dict(color='black')), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['Upper'], name='BB上', line=dict(width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['Lower'], name='BB下', line=dict(width=0), fill='tonexty', fillcolor='rgba(173,216,230,0.2)', showlegend=False), row=1, col=1)
            
            # 目標線 (購入:緑 / 売却:赤)
            line_color = "#28a745" if target_type == '購入' else "#dc3545"
            fig.add_hline(y=target_price, line_dash="dash", line_color=line_color, row=1, col=1)

            # AI予測線 (ドット表示)
            fore_plot = forecast[forecast['ds'] >= hist_plot.index[-1]].head(8)
            fig.add_trace(go.Scatter(x=fore_plot['ds'], y=fore_plot['yhat'], name='予測', line=dict(color='#0066ff', dash='dot')), row=1, col=1)

            # サブチャート: RSI
            fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="blue", row=2, col=1)

            fig.update_layout(height=450, margin=dict(l=0,r=0,b=0,t=10), showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)

            # 予測数値のサマリー
            st.write(f"🔮 **AI予想:** 今晩 {unit}{forecast.iloc[len(df_p)]['yhat']:,.1f} / 来週 {unit}{forecast.iloc[len(df_p)+6]['yhat']:,.1f}")

        except Exception as e:
            st.error(f"{ticker} 分析失敗: {e}")
