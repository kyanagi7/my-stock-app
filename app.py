import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime, time, timedelta

# --- 1. 銘柄と目標設定 ---
TICKERS_CONFIG = {
    '5970.T': [2070, '売却'],
    '7272.T': [1225, '売却'],
    '8306.T': [3050, '売却'],
    '8316.T': [5700, '売却'],
    '9101.T': [4950, '購入'],
}

st.set_page_config(page_title="Stock Advisor", layout="centered")

# --- CSS: 期間選択ボタンをiPhoneでも確実に追従させる設定 ---
st.markdown("""
    <style>
    /* 親要素のスクロール制限を解除してstickyを有効にする */
    [data-testid="stMain"] {
        overflow: visible !important;
    }
    .main .block-container {
        overflow: visible !important;
    }
    
    /* 期間選択コントロールを最上部に固定 */
    div[data-testid="stSegmentedControl"] {
        position: -webkit-sticky; /* Safari対応 */
        position: sticky;
        top: 3.5rem; /* Streamlitのヘッダー直下に配置 */
        z-index: 999;
        background-color: rgba(255, 255, 255, 0.95);
        padding: 10px 0 !important;
        border-bottom: 1px solid #eee;
    }
    
    /* モバイルでの見た目調整 */
    .stExpander { border: none !important; margin-top: 10px !important; }
    </style>
""", unsafe_allow_html=True)

st.title("⚖️ 高度分析 & 戦略ボード")

# --- 期間選択設定 ---
PERIOD_OPTIONS = {
    "6か月": {"days": 180, "interval": "1d", "pred_len": 14, "pred_freq": "D", "label": "2週間先"},
    "3か月": {"days": 90, "interval": "1d", "pred_len": 10, "pred_freq": "D", "label": "10日先"},
    "1か月": {"days": 30, "interval": "1d", "pred_len": 7, "pred_freq": "D", "label": "1週間先"},
    "1週間": {"days": 7, "interval": "30m", "pred_len": 16, "pred_freq": "30min", "label": "数日先"},
    "1日": {"days": 1, "interval": "5m", "pred_len": 24, "pred_freq": "5min", "label": "今日の大引け"}
}

# 追従するボタン
selected_label = st.segmented_control("表示期間を切り替え", options=list(PERIOD_OPTIONS.keys()), default="1か月")
v = PERIOD_OPTIONS[selected_label]

@st.cache_data(ttl=600)
def get_stock_data(ticker, interval):
    tk = yf.Ticker(ticker)
    period_map = {"5m": "5d", "30m": "15d", "1d": "2y"}
    df = tk.history(period=period_map[interval], interval=interval)
    if not df.empty:
        df.index = df.index.tz_convert('Asia/Tokyo').tz_localize(None)
    hist_daily = tk.history(period="5d", interval="1d")
    prev_close = hist_daily['Close'].iloc[-2] if len(hist_daily) > 1 else df['Close'].iloc[0]
    return df, prev_close

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
            
            # 1. テクニカル指標の計算
            close_full = df['Close']
            delta = close_full.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi_series = 100 - (100 / (1 + (gain / loss)))
            
            ma20 = close_full.rolling(window=20).mean()
            std20 = close_full.rolling(window=20).std()
            upper_s, lower_s = ma20 + (std20 * 2), ma20 - (std20 * 2)

            # 2. 達成度と色の決定
            is_achieved = (current_price <= target_price) if target_type == '購入' else (current_price >= target_price)
            metric_color = "#FF4B4B" if is_achieved else "#1F77B4"
            
            # 3. メトリクス表示
            price_diff = current_price - prev_close
            price_pct = (price_diff / prev_close) * 100
            target_diff = current_price - target_price
            target_pct = (target_diff / target_price) * 100

            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.markdown(f"""
                    <div style="line-height:1.2;">
                        <p style="margin:0; font-size:0.9rem; color:gray;">現在値 (前日比)</p>
                        <p style="margin:0; font-size:1.8rem; font-weight:bold;">¥{current_price:,.1f}</p>
                        <p style="margin:0; font-size:1.0rem; color:{metric_color}; font-weight:bold;">
                            {price_diff:+,.1f} ({price_pct:+.2f}%)
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                    <div style="line-height:1.2;">
                        <p style="margin:0; font-size:0.8rem; color:gray;">{target_type}目標値 (目標差)</p>
                        <p style="margin:0; font-size:1.2rem; font-weight:bold;">¥{target_price:,.0f}</p>
                        <p style="margin:0; font-size:0.9rem; color:{metric_color}; font-weight:bold;">
                            {target_diff:+,.1f} ({target_pct:+.2f}%)
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            # 4. AI予測
            df_p = df['Close'].reset_index()
            df_p.columns = ['ds', 'y']
            model = Prophet(daily_seasonality=True, weekly_seasonality=True).fit(df_p)
            forecast = model.predict(model.make_future_dataframe(periods=v["pred_len"], freq=v["pred_freq"]))

            # 5. グラフ作成
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
            
            # 実績・ボリンジャーバンド
            fig.add_trace(go.Scatter(x=hist_display.index, y=hist_display['Close'], name='実績', line=dict(color='#0055FF', width=3)), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist_display.index, y=upper_s.loc[hist_display.index], name='BB上', line=dict(width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist_display.index, y=lower_s.loc[hist_display.index], name='BB下', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,150,255,0.12)', showlegend=False), row=1, col=1)
            
            fig.add_hline(y=target_price, line_dash="dash", line_color=("#28a745" if target_type == '購入' else "#dc3545"), row=1, col=1)
            
            # 予測線
            fore_plot = forecast[forecast['ds'] >= hist_display.index[-1]].head(v["pred_len"] + 1)
            if selected_label == "1日": fore_plot = fore_plot[fore_plot['ds'] <= day_end]
            if not fore_plot.empty:
                pred_line_c = "#FF0000" if fore_plot['yhat'].iloc[-1] >= current_price else "#0000FF"
                fig.add_trace(go.Scatter(x=fore_plot['ds'], y=fore_plot['yhat'], name='予測', line=dict(color=pred_line_c, dash='dot', width=3)), row=1, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=hist_display.index, y=rsi_series.loc[hist_display.index], name='RSI', line=dict(color='#8A2BE2', width=2)), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="#FF4B4B", opacity=0.5, row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="#4B4BFF", opacity=0.5, row=2, col=1)

            if selected_label == "1日":
                fig.update_xaxes(range=[day_start, day_end], row=1, col=1)
                fig.update_xaxes(range=[day_start, day_end], row=2, col=1)

            fig.update_layout(height=450, margin=dict(l=0,r=0,b=0,t=10), showlegend=False, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            pred_p = fore_plot['yhat'].iloc[-1] if not fore_plot.empty else current_price
            st.write(f"🔮 **AI予測 ({v['label']}):** 約 ¥{pred_p:,.1f}")

        except Exception as e:
            st.error(f"分析失敗: {e}")
