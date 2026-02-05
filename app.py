import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta

# --- 設定：自分のポートフォリオ ---
# 銘柄コード: 保有株数 (日本株は .T をつける)
MY_PORTFOLIO = {
    '7203.T': 100,  # トヨタ自動車
    'AAPL': 10,     # Apple
    '7974.T': 50,   # 任天堂
}

st.set_page_config(page_title="My Stock Dash", layout="centered")

st.title("?? Stock Portfolio & Predict")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

@st.cache_data(ttl=3600)  # 1時間はキャッシュを保持
def load_data(tickers):
    data = yf.download(list(tickers.keys()), period="1y", interval="1d")
    return data['Close']

try:
    prices = load_data(MY_PORTFOLIO)
    
    # 資産総額の計算
    portfolio_val = (prices * pd.Series(MY_PORTFOLIO)).sum(axis=1)
    current_val = portfolio_val.iloc[-1]
    prev_val = portfolio_val.iloc[-2]
    change = current_val - prev_val

    # --- メトリクス表示 (iPhoneで見やすい横並び) ---
    col1, col2 = st.columns(2)
    col1.metric("総資産額", f"\{current_val:,.0f}")
    col2.metric("前日比", f"{change:+,.0f}", f"{(change/prev_val)*100:.2f}%")

    # --- 過去の推移グラフ ---
    st.subheader("Asset History")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=portfolio_val.index, y=portfolio_val, mode='lines', name='Total Value'))
    fig_hist.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300, hovermode="x unified")
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- AI予測 (Prophet) ---
    st.subheader("Forecast (1 Week)")
    
    # 予測用データの整形
    df_p = portfolio_val.reset_index()
    df_p.columns = ['ds', 'y']
    df_p['ds'] = df_p['ds'].dt.tz_localize(None) # タイムゾーン解除

    model = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
    model.fit(df_p)
    
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    # 予測グラフの作成
    fig_fore = go.Figure()
    # 実績
    fig_fore.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name='Actual', line=dict(color='gray')))
    # 予測
    fig_fore.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='blue')))
    # 予測の幅 (信頼区間)
    fig_fore.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='Upper', fillcolor='rgba(0,0,255,0.1)'))
    fig_fore.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='Lower', fillcolor='rgba(0,0,255,0.1)'))
    
    fig_fore.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300, showlegend=False)
    # 予測範囲（直近30日＋未来7日）にズーム
    fig_fore.update_xaxes(range=[datetime.now() - timedelta(days=30), datetime.now() + timedelta(days=7)])
    st.plotly_chart(fig_fore, use_container_width=True)

    # 本日の予想と今週末の予想
    today_pred = forecast.iloc[-7]['yhat']
    weekend_pred = forecast.iloc[-1]['yhat']
    st.info(f"?? **本日夜の予測値:** \{today_pred:,.0f}\n\n?? **今週末の予測値:** \{weekend_pred:,.0f}")

except Exception as e:
    st.error(f"データの取得に失敗しました。銘柄コードを確認してください。: {e}")