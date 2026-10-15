import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
BASE_YEAR = 2025
MIN_OBSERVATIONS = 300

st.title("🌡️ 서울 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고, "
    "선택한 연도의 회귀 예상 평균기온을 확인해 보세요."
)
st.info(
    "분석 기준: 2025년까지 · 연간 관측일이 300일 미만인 해는 제외"
)


@st.cache_data(ttl=3600)
def load_data(url):
    df = pd.read_csv(url, encoding="utf-8")

    required_columns = ["날짜", "지점", "평균기온", "최저기온", "최고기온"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError("CSV에 필요한 열이 없습니다: " + ", ".join(missing))

    df = df[["날짜", "평균기온"]].copy()
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df = df[df["날짜"].dt.year <= BASE_YEAR]

    # 중복 날짜는 하루 한 건으로 계산
    df = df.drop_duplicates(subset=["날짜"], keep="last")
    df["연도"] = df["날짜"].dt.year

    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique"),
        )
        .reset_index()
    )
    annual = annual[annual["관측일수"] >= MIN_OBSERVATIONS]
    return annual.sort_values("연도").reset_index(drop=True)


try:
    with st.spinner("서울 기온 데이터를 불러오는 중입니다..."):
        annual = load_data(DATA_URL)
except Exception as error:
    st.error(f"데이터를 불러오거나 처리하지 못했습니다: {error}")
    st.stop()

if len(annual) < 2:
    st.error("조건을 만족하는 연도가 2개 미만이라 회귀분석을 할 수 없습니다.")
    st.stop()

X = annual[["연도"]].to_numpy()
y = annual["연평균기온"].to_numpy()

model = LinearRegression()
model.fit(X, y)
correlation = np.corrcoef(annual["연도"], annual["연평균기온"])[0, 1]

start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
year_count = len(annual)

st.subheader("📊 회귀분석 정보")
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석한 연도 수", f"{year_count:,}개")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수", f"{correlation:.3f}")

st.caption(
    "상관계수는 연도와 연평균기온 사이의 선형 관계를 나타냅니다. "
    "양수이면 대체로 시간이 지날수록 기온이 높아지는 관계입니다."
)

st.subheader("📈 연도별 평균기온과 회귀 직선")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7, opacity=0.75),
        customdata=annual[["관측일수"]],
        hovertemplate=(
            "연도: %{x}년<br>연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일<extra></extra>"
        ),
    )
)

line_years = np.linspace(start_year, end_year, 300)
line_temps = model.predict(line_years.reshape(-1, 1))
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="선형회귀 직선",
        line=dict(width=3),
        hovertemplate="연도: %{x:.0f}년<br>회귀 예상 평균기온: %{y:.2f}℃<extra></extra>",
    )
)
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"회귀 직선은 {start_year}년부터 {end_year}년까지 "
    f"{year_count}개 연평균기온 자료로 만들었습니다."
)

st.divider()
st.subheader("🔮 연도별 예상 평균기온")
selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)
selected_temp = float(model.predict(np.array([[selected_year]]))[0])

st.markdown(
    f"""
    <div style="background:#FFF4D6;border-radius:18px;padding:28px 18px;
                text-align:center;margin:12px 0 20px 0;">
      <div style="font-size:22px;color:#72551B;">
        {selected_year}년 예상 연평균기온
      </div>
      <div style="font-size:52px;font-weight:800;color:#B76E00;margin-top:8px;">
        {selected_temp:.2f}℃
      </div>
      <div style="font-size:14px;color:#72551B;">
        과거 연평균기온의 선형회귀 추정값
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

prediction_fig = go.Figure()
prediction_fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=6, opacity=0.65),
    )
)
prediction_fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="학습 구간 회귀 직선",
        line=dict(width=3),
    )
)
prediction_fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[selected_temp],
        mode="markers",
        name="선택한 연도의 예상값",
        marker=dict(size=14, symbol="diamond"),
        hovertemplate=f"{selected_year}년 예상 평균기온: %{{y:.2f}}℃<extra></extra>",
    )
)
prediction_fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(prediction_fig, use_container_width=True)

with st.expander("분석에 사용된 연도별 데이터 보기"):
    display_df = annual.copy()
    display_df["연평균기온"] = display_df["연평균기온"].round(2)
    display_df.columns = ["연도", "연평균기온 (℃)", "관측일수"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.warning(
    "주의: 이 결과는 과거 연도와 연평균기온 사이의 선형 관계를 "
    "1900~2100년까지 연장한 통계적 추정값입니다. 실제 미래 기온을 "
    "예보하는 기후모형이 아니며, 관측 기간 밖의 예측은 불확실성이 큽니다."
)
