import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime, timedelta
import time
import requests

# --- 1. 銘柄・目標・名称の事前設定 (通信削減のため) ---
TICKERS_CONFIG = {
    '5970.T': {'target': 2070, 'type': '売却', 'name': 'ジーテクト'},
    '7272.T': {'target': 1225, 'type': '売却', 'name': 'ヤマハ発動機'},
    '8306.T': {'target': 3050, 'type': '売却', 'name': '三菱UFJ FG'},
    '8316.T': {'target': 5700, 'type': '売却', 'name': '三井住友 FG'},
    '9101.T': {'target': 4950, 'type': '購入', 'name': '日本郵船'},
}

st.set_page_config(page_title="Stock Advisor", layout="centered")

# サイドバー設定
st.sidebar.title("📊 設定")
PERIOD_OPTIONS = {
    "6か月": {"days": 180, "interval": "1d", "pred_len": 14, "pred_freq": "D", "label": "2週間先"},
    "3か月": {"days": 90, "interval": "1d", "pred_len": 10, "pred_freq": "D", "label": "10日先"},
    "1か月": {"days": 30, "interval": "1d", "pred_len": 7, "pred_freq": "D", "label": "1週間先"},
    "1週間": {"days": 7, "interval": "30m", "pred_len": 16, "pred_freq": "30min", "label": "数営業日先"},
    "1日": {"days": 1, "interval": "5m", "pred_len": 24, "pred_freq": "5min", "label": "今日の大引け"}
}
selected_label = st.sidebar.radio("表示期間", options=list(PERIOD_OPTIONS.keys()), index=2)
v = PERIOD_OPTIONS[selected_label]

st.title("⚖️ 高度分析 & 戦略ボード")

# セッションを作成（レート制限対策）
@st.cache_resource
def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

@st.cache_data(ttl=900) # キャッシュ時間を15分に延長
def get_stock_data(ticker_symbol, interval):
    session = get_session()
    tk = yf.Ticker(ticker_symbol, session=session)
    
    period_map = {"5m": "5d", "30m": "15d", "1d": "2y"}
    
    # データ取得 (リトライ処理)
    try:
        df = tk.history(period=period_map[interval], interval=interval)
        if df.empty: return None, None
        df.index = df.index.tz_convert('Asia/Tokyo').tz_localize(None)
        
        # 前日終値の取得
        hist_daily = tk.history(period="5d", interval="1d")
        prev_close = hist_daily['Close'].iloc[-2] if len(hist_daily) > 1 else df['Close'].iloc[0]
        return df, prev_close
    except Exception:
        return None, None

def get_advice(current_price, rsi, upper, lower):
    if rsi >= 70 or current_price >= upper:
        return "⚠️ 売り検討", "過熱気味です。利益確定を検討してください。", "error"
    elif rsi <= 30 or current_price <= lower:
        return "💎 買い検討", "売られすぎです。リバウンドの好機かもしれません。", "success"
    return "😐 様子見", "過熱感はなく、安定した推移です。", "info"

# --- メイン処理 ---
for ticker, info in TICKERS_CONFIG.items():
    target_price, target_type, name = info['target'], info['type'], info['name']
    
    with st.spinner(f'{name} ({ticker}) を取得中...'):
        df, prev_close = get_stock_data(ticker, v["interval"])
        time.sleep(1.0) # 銘柄間に1秒の待機を入れて制限を回避
    
    if df is None or df.empty:
        st.warning(f"{ticker} のデータ取得に失敗しました。時間をおいて試してください。")
        continue

    with st.expander(f"📌 {name} ({ticker})", expanded=True):
        try:
            # --- データ抽出・計算 ---
            last_dt = df.index[-1]
            if selected_label == "1日":
                day_start = last_dt.replace(hour=9, minute=0, second=0)
                day_end = last_dt.replace(hour=15, minute=30, second=0)
                hist_display = df.loc[day_start:day_end]
            else:
                hist_display = df.tail(v["days"] if v["interval"] == "1d" else 100)
            
            if hist_display.empty: hist_display = df.tail(50)
            current_price = float(hist_display['Close'].iloc[-1])
            
            # テクニカル計算
            close_full = df['Close']
            delta = close_full.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi_series = 100 - (100 / (1 + (gain / loss)))
            ma20 = close_full.rolling(20).mean()
            std20 = close_full.rolling(20).std()
            upper_s, lower_s = ma20 + (std20 * 2), ma20 - (std20 * 2)

            # 判定表示
            current_rsi = rsi_series.iloc[-1]
            status, advice_msg, style = get_advice(current_price, current_rsi, upper_s.iloc[-1], lower_s.iloc[-1])
            if style == "success": st.success(f"**判定: {status}** \n{advice_msg}")
            elif style == "error": st.error(f"**判定: {status}** \n{advice_msg}")
            else: st.info(f"**判定: {status}** \n{advice_msg}")

            # メトリクス
            is_achieved = (current_price <= target_price) if target_type == '購入' else (current_price >= target_price)
            metric_color = "#FF4B4B" if is_achieved else "#1F77B4"
            rsi_color = "#FF4B4B" if current_rsi >= 70 else ("#1F77B4" if current_rsi <= 30 else "#333333")
            p_diff = current_price - prev_close
            p_pct = (p_diff / prev_close) * 100
            t_diff = current_price - target_price
            t_pct = (t_diff / target_price) * 100

            c1, c2, c3 = st.columns([1.2, 1, 0.8])
            with c1: st.markdown(f'<div style="line-height:1.2;"><p style="font-size:0.8rem; color:gray; margin:0;">現在値 (前日比)</p><p style="font-size:1.6rem; font-weight:bold; margin:0;">¥{current_price:,.1f}</p><p style="font-size:0.9rem; color:{metric_color}; font-weight:bold; margin:0;">{p_diff:+,.1f} ({p_pct:+.2f}%)</p></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div style="line-height:1.2;"><p style="font-size:0.8rem; color:gray; margin:0;">{target_type}目標</p><p style="font-size:1.2rem; font-weight:bold; margin:0;">¥{target_price:,.0f}</p><p style="font-size:0.9rem; color:{metric_color}; font-weight:bold; margin:0;">{t_diff:+,.1f} ({t_pct:+.2f}%)</p></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div style="line-height:1.2;"><p style="font-size:0.8rem; color:gray; margin:0;">RSI</p><p style="font-size:1.6rem; font-weight:bold; color:{rsi_color}; margin:0;">{current_rsi:.1f}</p></div>', unsafe_allow_html=True)

            # AI予測
            df_p = df['Close'].reset_index()
            df_p.columns = ['ds', 'y']
            model = Prophet(daily_seasonality=True, weekly_seasonality=True).fit(df_p)
            forecast = model.predict(model.make_future_dataframe(periods=v["pred_len"], freq=v["pred_freq"]))

            # グラフ作成
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=hist_display.index, y=upper_s.loc[hist_display.index], name='BB上', line=dict(width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist_display.index, y=lower_s.loc[hist_display.index], name='BB下', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,150,255,0.1)', showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist_display.index, y=hist_display['Close'], name='実績', line=dict(color='#0055FF', width=3)), row=1, col=1)
            fig.add_hline(y=target_price, line_dash="dash", line_color=("#28a745" if target_type == '購入' else "#dc3545"), row=1, col=1)
            
            fore_plot = forecast[forecast['ds'] >= hist_display.index[-1]].head(v["pred_len"] + 1)
            if selected_label == "1日": fore_plot = fore_plot[fore_plot['ds'] <= day_end]
            if not fore_plot.empty:
                pred_c = "#FF0000" if fore_plot['yhat'].iloc[-1] >= current_price else "#0000FF"
                fig.add_trace(go.Scatter(x=fore_plot['ds'], y=fore_plot['yhat'], name='予測', line=dict(color=pred_c, dash='dot', width=3)), row=1, col=1)

            fig.add_trace(go.Scatter(x=hist_display.index, y=rsi_series.loc[hist_display.index], name='RSI', line=dict(color='#8A2BE2', width=2)), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="#FF4B4B", opacity=0.5, row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="#4B4BFF", opacity=0.5, row=2, col=1)

            if selected_label == "1日":
                fig.update_xaxes(range=[day_start, day_end], row=1, col=1)

            fig.update_layout(height=450, margin=dict(l=0,r=0,b=0,t=10), showlegend=False, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"🔮 **AI予測 ({v['label']}):** 約 ¥{fore_plot['yhat'].iloc[-1]:,.1f}")

        except Exception as e:
            st.error(f"表示エラー: {e}")
