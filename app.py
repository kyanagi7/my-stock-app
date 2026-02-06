import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime

# --- 1. 銘柄と目標設定 ---
TICKERS_CONFIG = {
    '5970.T': [2070, '売却'],
    '7272.T': [1225, '売却'],
    '8306.T': [3050, '売却'],
    '8316.T': [5700, '売却'],
    '9101.T': [4950, '購入'],
}

st.set_page_config(page_title="Stock Trading Advisor", layout="centered")
st.title("⚖️ テクニカル自動判定 & 株価予測")

# --- 期間切り替え用設定 ---
PERIOD_OPTIONS = {
    "6か月": 180,
    "3か月": 90,
    "1か月": 30,
    "1週間": 7,
    "1日": 2
}

# 画面上部に期間選択ボタンを配置
selected_label = st.segmented_control(
    "表示期間を選択", 
    options=list(PERIOD_OPTIONS.keys()), 
    default="1か月"
)
view_days = PERIOD_OPTIONS[selected_label]

@st.cache_data(ttl=600)
def get_stock_data(ticker):
    tk = yf.Ticker(ticker)
    name = tk.info.get('longName', ticker)
    # 2年分取得（予測モデルの精度維持のため）
    df = tk.history(period="2y")
    if df.empty:
        return None, None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    # 指標計算
    close = df['Close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    df['MA20'] = close.rolling(window=20).mean()
    df['STD20'] = close.rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['STD20'] * 2)
    df['Lower'] = df['MA20'] - (df['STD20'] * 2)
    return name, df

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
    name, df = get_stock_data(ticker)
    if df is None: continue

    with st.expander(f"📌 {name} ({ticker})", expanded=True):
        try:
            current_price = float(df['Close'].iloc[-1])
            rsi_val = float(df['RSI'].iloc[-1])
            upper_val = float(df['Upper'].iloc[-1])
            lower_val = float(df['Lower'].iloc[-1])

            # 判定アドバイス
            status, message, type_style = get_advice(current_price, rsi_val, upper_val, lower_val)
            st.subheader(f"判定: {status}")
            if type_style == "success": st.success(message)
            elif type_style == "error": st.error(message)
            else: st.info(message)

            c1, c2, c3 = st.columns(3)
            c1.metric("現在値", f"¥{current_price:,.1f}")
            c2.metric(f"{target_type}目標", f"¥{target_price:,.0f}")
            c3.metric("RSI", f"{rsi_val:.1f}")

            # AI予測
            df_p = df['Close'].reset_index()
            df_p.columns = ['ds', 'y']
            df_p['ds'] = pd.to_datetime(df_p['ds']).dt.tz_localize(None)
            model = Prophet(daily_seasonality=True).fit(df_p)
            forecast = model.predict(model.make_future_dataframe(periods=14))
            
            # --- グラフ描画（期間切り替え対応） ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
            
            # 選択された期間でフィルタリング
            hist_plot = df.tail(view_days)
            
            # 実績線
            fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['Close'], name='実績', 
                                     line=dict(color='#0055FF', width=3)), row=1, col=1)
            
            # ボリンジャーバンド
            fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['Upper'], name='BB上', line=dict(width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['Lower'], name='BB下', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,150,255,0.1)', showlegend=False), row=1, col=1)
            
            # 目標線
            line_color = "#28a745" if target_type == '購入' else "#dc3545"
            fig.add_hline(y=target_price, line_dash="dash", line_color=line_color, row=1, col=1)
            
            # 予測線の色判定
            fore_plot = forecast[forecast['ds'] >= hist_plot.index[-1]].head(8)
            prediction_end_price = fore_plot['yhat'].iloc[-1]
            pred_line_color = "#FF0000" if prediction_end_price >= current_price else "#0000FF"
            
            fig.add_trace(go.Scatter(x=fore_plot['ds'], y=fore_plot['yhat'], name='予測', 
                                     line=dict(color=pred_line_color, dash='dot', width=3)), row=1, col=1)
            
            # RSIチャート
            fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot['RSI'], name='RSI', line=dict(color='#8A2BE2')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="#FF4B4B", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="#4B4BFF", row=2, col=1)
            
            fig.update_layout(height=480, margin=dict(l=0,r=0,b=0,t=10), showlegend=False, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            trend_icon = "📈" if prediction_end_price >= current_price else "📉"
            st.write(f"🔮 **AI予想 {trend_icon}:** 今晩 ¥{forecast.iloc[len(df_p)]['yhat']:,.1f} / 来週 ¥{forecast.iloc[len(df_p)+6]['yhat']:,.1f}")

        except Exception as e:
            st.error(f"分析失敗: {e}")
