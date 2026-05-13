import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from supabase import create_client, Client
from google import genai
from google.genai import types
import json
import re
from difflib import SequenceMatcher

st.set_page_config(
    page_title="Consumer Pulse AI",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Pretendard', sans-serif;
}

.main {
    background-color: #F8FAFC;
}

.block-container {
    padding-top: 2rem;
}

.stButton > button {
    background-color: #4F46E5;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.65rem 1.2rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #4338CA;
    color: white;
    border: none;
}

.card {
    background-color: white;
    padding: 1.4rem;
    border-radius: 16px;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
    margin-bottom: 1rem;
    border: 1px solid #E5E7EB;
}

.kpi-card {
    background-color: white;
    padding: 1.1rem;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
    border: 1px solid #E5E7EB;
}

.kpi-card small {
    color: #64748B;
    font-weight: 600;
}

.kpi-card h3 {
    margin-top: 0.4rem;
    margin-bottom: 0;
}

.badge-positive {
    background-color: #DCFCE7;
    color: #166534;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
}

.badge-neutral {
    background-color: #F1F5F9;
    color: #334155;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
}

.badge-negative {
    background-color: #FEE2E2;
    color: #991B1B;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_supabase() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


@st.cache_resource
def init_gemini():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


try:
    supabase = init_supabase()
    client = init_gemini()
except Exception:
    st.error("Secrets 설정을 확인하세요. GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY가 필요합니다.")
    st.stop()


def extract_json_array(text: str):
    if not text:
        return []

    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, list) else []
    except Exception:
        pass

    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    return []


def normalize_sentiment(value):
    value = str(value).lower()
    if "positive" in value or "긍정" in value:
        return "Positive"
    if "negative" in value or "부정" in value:
        return "Negative"
    return "Neutral"


def normalize_risk(value):
    value = str(value).lower()
    if "high" in value or "높" in value:
        return "High"
    if "medium" in value or "중" in value:
        return "Medium"
    return "Low"


def sentiment_ko(value):
    return {
        "Positive": "긍정",
        "Neutral": "중립",
        "Negative": "부정"
    }.get(value, "중립")


def risk_ko(value):
    return {
        "Low": "낮음",
        "Medium": "보통",
        "High": "높음"
    }.get(value, "낮음")


def title_similarity(a, b):
    a = re.sub(r"\s+", "", str(a).lower())
    b = re.sub(r"\s+", "", str(b).lower())
    if not a or not b:
        return 0
    return SequenceMatcher(None, a, b).ratio()


def replace_with_grounded_urls(items, response):
    try:
        chunks = response.candidates[0].grounding_metadata.grounding_chunks
    except Exception:
        chunks = []

    grounded_sources = []

    for chunk in chunks:
        try:
            title = chunk.web.title
            uri = chunk.web.uri
            if uri and uri.startswith("http") and "grounding-api-redirect" not in uri:
                grounded_sources.append({"title": title, "url": uri})
        except Exception:
            continue

    for item in items:
        best_url = item.get("url", "")
        best_score = 0

        for source in grounded_sources:
            score = title_similarity(item.get("title", ""), source.get("title", ""))
            if score > best_score:
                best_score = score
                best_url = source["url"]

        if best_score >= 0.25 and best_url.startswith("http") and "grounding-api-redirect" not in best_url:
            item["url"] = best_url

    return items


with st.sidebar:
    st.title("Consumer Pulse AI")
    selected = option_menu(
        menu_title=None,
        options=["소비자 반응 검색", "저장된 인사이트", "분석 대시보드"],
        icons=["search", "archive", "bar-chart-line"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#FAFAFA"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#EEF2FF"
            },
            "nav-link-selected": {"background-color": "#4F46E5"},
        }
    )


if selected == "소비자 반응 검색":
    st.subheader("소비자 반응 검색")
    st.caption("브랜드나 서비스를 입력하면 최신 소비자 반응, 불만 요소, 리스크 수준을 자동으로 분석합니다.")

    col1, col2 = st.columns([4, 1])

    with col1:
        keyword = st.text_input(
            "브랜드 또는 서비스명",
            placeholder="예: 쿠팡, 스타벅스, 무신사, 올리브영"
        )

    with col2:
        st.write("")
        search_btn = st.button("분석 시작", use_container_width=True)

    if search_btn:
        if not keyword.strip():
            st.warning("검색할 브랜드 또는 서비스명을 입력하세요.")
            st.stop()

        with st.spinner("최신 소비자 반응을 검색하고 분석하는 중입니다..."):
            prompt = f"""
다음 브랜드 또는 서비스에 대한 최신 소비자 반응과 불만 관련 뉴스 딱 2건만 검색해줘.

검색어: {keyword}

반드시 아래 JSON 배열 형식으로만 답변해.
마크다운 설명, 코드블록, 추가 문장은 절대 넣지 마.

[
  {{
    "title": "기사 제목",
    "source": "출처",
    "news_date": "YYYY-MM-DD",
    "url": "실제 기사 URL",
    "summary": "소비자 반응과 불만 요소 중심의 2~3문장 요약",
    "sentiment": "Positive 또는 Neutral 또는 Negative",
    "risk_level": "Low 또는 Medium 또는 High"
  }}
]

규칙:
1. Google Search로 확인한 최신 정보만 사용해.
2. URL은 절대 지어내지 마.
3. 소비자 불만, 서비스 문제, 가격, 품질, 배송, 고객 응대, 브랜드 평판 이슈가 있으면 요약에 반영해.
4. sentiment와 risk_level은 반드시 영어 값으로 작성해.
"""

            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                        temperature=0.0,
                    ),
                )
            except Exception as e:
                st.error("Gemini API 호출 중 문제가 발생했습니다. API 키, 사용량 제한, 모델 접근 권한을 확인하세요.")
                st.caption(str(e))
                st.stop()

            items = extract_json_array(response.text)
            if not items:
                st.error("AI 응답을 JSON으로 해석하지 못했습니다. 잠시 후 다시 검색해 보세요.")
                st.stop()

            items = replace_with_grounded_urls(items, response)

            valid_results = []
            for item in items[:2]:
                url = str(item.get("url", "")).strip()

                if not url.startswith("http") or "grounding-api-redirect" in url:
                    continue

                record = {
                    "keyword": keyword.strip(),
                    "title": str(item.get("title", "제목 없음")).strip(),
                    "source": str(item.get("source", "출처 미상")).strip(),
                    "news_date": str(item.get("news_date", "")).strip(),
                    "url": url,
                    "summary": str(item.get("summary", "")).strip(),
                    "sentiment": normalize_sentiment(item.get("sentiment", "Neutral")),
                    "risk_level": normalize_risk(item.get("risk_level", "Low")),
                }
                valid_results.append(record)

            if not valid_results:
                st.warning("사용 가능한 실제 기사 URL을 찾지 못했습니다. 다른 키워드로 다시 시도해 보세요.")
                st.stop()

            saved_count = 0
            duplicate_count = 0

            for res in valid_results:
                sentiment_class = {
                    "Positive": "badge-positive",
                    "Neutral": "badge-neutral",
                    "Negative": "badge-negative"
                }.get(res["sentiment"], "badge-neutral")

                risk_color = {
                    "Low": "#2563EB",
                    "Medium": "#D97706",
                    "High": "#DC2626"
                }.get(res["risk_level"], "#2563EB")

                st.markdown(f"""
                <div class="card">
                    <div style="display:flex; justify-content:space-between; gap:1rem;">
                        <h4 style="margin:0; color:#111827;">{res["title"]}</h4>
                        <span style="color:#64748B; font-size:0.9rem; white-space:nowrap;">
                            {res["source"]} · {res["news_date"]}
                        </span>
                    </div>
                    <p style="margin:1rem 0; color:#334155; line-height:1.6;">
                        {res["summary"]}
                    </p>
                    <div>
                        <span class="{sentiment_class}">감성: {sentiment_ko(res["sentiment"])}</span>
                        <span style="color:{risk_color}; font-weight:700; margin-left:10px;">
                            리스크: {risk_ko(res["risk_level"])}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.link_button("원문 기사 보기", res["url"])

                try:
                    supabase.table("consumer_insights").insert(res).execute()
                    saved_count += 1
                except Exception as e:
                    if "23505" in str(e) or "duplicate" in str(e).lower():
                        duplicate_count += 1
                    else:
                        st.warning(f"일부 데이터 저장 중 문제가 발생했습니다: {e}")

            st.toast(f"저장 완료: 신규 {saved_count}건, 중복 생략 {duplicate_count}건")


elif selected == "저장된 인사이트":
    st.subheader("저장된 인사이트")
    st.caption("Supabase에 저장된 소비자 반응 분석 결과를 조회하고 필터링할 수 있습니다.")

    try:
        data_res = supabase.table("consumer_insights").select("*").order("created_at", desc=True).execute()
        df = pd.DataFrame(data_res.data)
    except Exception as e:
        st.error("데이터베이스 조회 중 문제가 발생했습니다.")
        st.caption(str(e))
        st.stop()

    if df.empty:
        st.info("아직 저장된 데이터가 없습니다.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        keyword_filter = st.text_input("키워드/제목 검색", placeholder="예: 쿠팡, 배송, 가격")

    with col2:
        sentiment_filter_ko = st.multiselect("감성 필터", ["긍정", "중립", "부정"])

    with col3:
        risk_filter_ko = st.multiselect("리스크 필터", ["낮음", "보통", "높음"])

    filtered_df = df.copy()

    if keyword_filter:
        keyword_filter = keyword_filter.lower()
        filtered_df = filtered_df[
            filtered_df["keyword"].astype(str).str.lower().str.contains(keyword_filter, na=False)
            | filtered_df["title"].astype(str).str.lower().str.contains(keyword_filter, na=False)
            | filtered_df["summary"].astype(str).str.lower().str.contains(keyword_filter, na=False)
        ]

    sentiment_reverse = {"긍정": "Positive", "중립": "Neutral", "부정": "Negative"}
    risk_reverse = {"낮음": "Low", "보통": "Medium", "높음": "High"}

    if sentiment_filter_ko:
        sentiment_values = [sentiment_reverse[x] for x in sentiment_filter_ko]
        filtered_df = filtered_df[filtered_df["sentiment"].isin(sentiment_values)]

    if risk_filter_ko:
        risk_values = [risk_reverse[x] for x in risk_filter_ko]
        filtered_df = filtered_df[filtered_df["risk_level"].isin(risk_values)]

    view_df = filtered_df.copy()
    view_df["감성"] = view_df["sentiment"].apply(sentiment_ko)
    view_df["리스크"] = view_df["risk_level"].apply(risk_ko)

    columns = {
        "keyword": "검색어",
        "title": "제목",
        "source": "출처",
        "news_date": "기사 날짜",
        "summary": "요약",
        "url": "URL",
        "created_at": "저장일시"
    }

    display_cols = ["keyword", "title", "source", "news_date", "감성", "리스크", "summary", "url", "created_at"]
    display_cols = [c for c in display_cols if c in view_df.columns]

    st.dataframe(
        view_df[display_cols].rename(columns=columns),
        use_container_width=True,
        hide_index=True
    )

    csv = view_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "CSV 다운로드",
        data=csv,
        file_name="consumer_pulse_insights.csv",
        mime="text/csv"
    )


elif selected == "분석 대시보드":
    st.subheader("분석 대시보드")
    st.caption("저장된 데이터를 바탕으로 브랜드별 언급량, 감성, 리스크 흐름을 확인합니다.")

    try:
        data_res = supabase.table("consumer_insights").select("*").execute()
        df = pd.DataFrame(data_res.data)
    except Exception as e:
        st.error("데이터베이스 조회 중 문제가 발생했습니다.")
        st.caption(str(e))
        st.stop()

    if df.empty:
        st.info("분석할 데이터가 아직 없습니다.")
        st.stop()

    total = len(df)
    negative_count = len(df[df["sentiment"] == "Negative"])
    negative_rate = (negative_count / total) * 100 if total else 0
    high_risk_count = len(df[df["risk_level"] == "High"])
    top_keyword = df["keyword"].mode()[0] if "keyword" in df.columns and not df["keyword"].empty else "-"

    k1, k2, k3, k4 = st.columns(4)

    k1.markdown(f"<div class='kpi-card'><small>총 저장 건수</small><h3>{total}</h3></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi-card'><small>부정 반응 비율</small><h3 style='color:#DC2626;'>{negative_rate:.1f}%</h3></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi-card'><small>고위험 이슈</small><h3 style='color:#DC2626;'>{high_risk_count}</h3></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi-card'><small>최다 검색 키워드</small><h3>{top_keyword}</h3></div>", unsafe_allow_html=True)

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 키워드별 누적 언급량")
        keyword_df = df["keyword"].value_counts().reset_index()
        keyword_df.columns = ["키워드", "건수"]

        fig1 = px.bar(
            keyword_df,
            x="키워드",
            y="건수",
            text="건수",
            color_discrete_sequence=["#4F46E5"]
        )
        fig1.update_layout(
            xaxis_title="키워드",
            yaxis_title="저장 건수",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.markdown("#### 일자별 저장 추이")
        df["날짜"] = pd.to_datetime(df["created_at"], errors="coerce").dt.date
        trend_df = df.dropna(subset=["날짜"]).groupby("날짜").size().reset_index(name="건수")

        fig2 = px.line(
            trend_df,
            x="날짜",
            y="건수",
            markers=True,
            color_discrete_sequence=["#4F46E5"]
        )
        fig2.update_layout(
            xaxis_title="날짜",
            yaxis_title="저장 건수",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### 감성 분석 비율")
        sentiment_df = df["sentiment"].value_counts().reset_index()
        sentiment_df.columns = ["감성", "건수"]
        sentiment_df["감성"] = sentiment_df["감성"].apply(sentiment_ko)

        fig3 = px.pie(
            sentiment_df,
            names="감성",
            values="건수",
            color="감성",
            color_discrete_map={
                "긍정": "#10B981",
                "중립": "#9CA3AF",
                "부정": "#EF4444"
            }
        )
        fig3.update_layout(paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown("#### 리스크 수준 분포")
        risk_df = df["risk_level"].value_counts().reset_index()
        risk_df.columns = ["리스크", "건수"]
        risk_df["리스크"] = risk_df["리스크"].apply(risk_ko)

        fig4 = px.bar(
            risk_df,
            x="리스크",
            y="건수",
            text="건수",
            color="리스크",
            color_discrete_map={
                "낮음": "#2563EB",
                "보통": "#F59E0B",
                "높음": "#EF4444"
            }
        )
        fig4.update_layout(
            xaxis_title="리스크 수준",
            yaxis_title="건수",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig4, use_container_width=True)
