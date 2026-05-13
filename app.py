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
from urllib.parse import quote_plus

st.set_page_config(page_title="Consumer Pulse AI", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Pretendard', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 100%);
}

.block-container {
    padding-top: 2.2rem;
}

section[data-testid="stSidebar"] {
    background: #111827;
}

section[data-testid="stSidebar"] * {
    color: #F9FAFB;
}

.main-title {
    font-size: 2.15rem;
    font-weight: 800;
    color: #111827;
    margin-bottom: 0.35rem;
}

.sub-text {
    color: #64748B;
    font-size: 1rem;
    margin-bottom: 1.4rem;
}

.insight-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 20px;
    padding: 1.45rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
}

.insight-title {
    font-size: 1.1rem;
    font-weight: 750;
    color: #111827;
    margin-bottom: 0.5rem;
}

.meta {
    color: #64748B;
    font-size: 0.88rem;
    margin-bottom: 0.8rem;
}

.summary {
    color: #334155;
    line-height: 1.65;
    font-size: 0.95rem;
    margin-bottom: 1rem;
}

.badge {
    display: inline-block;
    padding: 5px 11px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 0.4rem;
}

.badge-positive { background: #DCFCE7; color: #166534; }
.badge-neutral { background: #F1F5F9; color: #334155; }
.badge-negative { background: #FEE2E2; color: #991B1B; }
.badge-low { background: #DBEAFE; color: #1D4ED8; }
.badge-medium { background: #FEF3C7; color: #B45309; }
.badge-high { background: #FEE2E2; color: #B91C1C; }

.kpi-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 20px;
    padding: 1.2rem;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.07);
}

.kpi-label {
    color: #64748B;
    font-size: 0.86rem;
    font-weight: 700;
}

.kpi-value {
    color: #111827;
    font-size: 1.7rem;
    font-weight: 800;
    margin-top: 0.35rem;
}

.stButton > button {
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    color: white;
    border: none;
    border-radius: 13px;
    padding: 0.78rem 1rem;
    font-weight: 700;
    height: 46px;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #4338CA, #6D28D9);
    color: white;
    border: none;
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


def extract_json_array(text):
    if not text:
        return []

    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
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
    if "medium" in value or "보통" in value or "중" in value:
        return "Medium"
    return "Low"


def sentiment_ko(value):
    return {"Positive": "긍정", "Neutral": "중립", "Negative": "부정"}.get(value, "중립")


def risk_ko(value):
    return {"Low": "낮음", "Medium": "보통", "High": "높음"}.get(value, "낮음")


def sentiment_badge_class(value):
    return {
        "Positive": "badge-positive",
        "Neutral": "badge-neutral",
        "Negative": "badge-negative"
    }.get(value, "badge-neutral")


def risk_badge_class(value):
    return {
        "Low": "badge-low",
        "Medium": "badge-medium",
        "High": "badge-high"
    }.get(value, "badge-low")


def title_similarity(a, b):
    a = re.sub(r"\s+", "", str(a).lower())
    b = re.sub(r"\s+", "", str(b).lower())
    if not a or not b:
        return 0
    return SequenceMatcher(None, a, b).ratio()


def safe_text(value, default=""):
    value = str(value).strip()
    return value if value else default


def is_valid_url(url):
    url = str(url).strip()
    return url.startswith("http") and "grounding-api-redirect" not in url


def google_news_fallback(keyword, title):
    query = quote_plus(f"{keyword} {title}")
    return f"https://news.google.com/search?q={query}"


def collect_grounding_sources(search_response):
    sources = []

    try:
        chunks = search_response.candidates[0].grounding_metadata.grounding_chunks
    except Exception:
        return sources

    for chunk in chunks:
        try:
            title = safe_text(chunk.web.title)
            uri = safe_text(chunk.web.uri)
            if is_valid_url(uri):
                sources.append({"title": title, "url": uri})
        except Exception:
            continue

    return sources


def replace_with_best_url(items, search_response, keyword):
    sources = collect_grounding_sources(search_response)

    for item in items:
        current_url = safe_text(item.get("url", ""))
        title = safe_text(item.get("title", ""))

        if is_valid_url(current_url):
            item["url"] = current_url
            continue

        best_url = ""
        best_score = 0

        for source in sources:
            score = title_similarity(title, source["title"])
            if score > best_score:
                best_score = score
                best_url = source["url"]

        if is_valid_url(best_url):
            item["url"] = best_url
        else:
            item["url"] = google_news_fallback(keyword, title)

    return items


with st.sidebar:
    st.markdown("<h2 style='font-weight:800; margin-bottom:0;'>Consumer<br>Pulse AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#CBD5E1; font-size:0.9rem;'>브랜드 반응 분석 플랫폼</p>", unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["소비자 반응 검색", "인사이트 저장소", "분석 대시보드"],
        icons=["search", "folder2-open", "bar-chart"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#111827"},
            "icon": {"color": "#E0E7FF", "font-size": "16px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "6px 0",
                "color": "#E5E7EB",
                "--hover-color": "#1F2937",
                "border-radius": "12px"
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg, #4F46E5, #7C3AED)",
                "color": "white",
                "font-weight": "700"
            },
        }
    )


if selected == "소비자 반응 검색":
    st.markdown("<div class='main-title'>소비자 반응 검색</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-text'>브랜드나 서비스를 입력하면 최근 기사와 소비자 반응을 바탕으로 긍정·중립·부정 흐름을 분석합니다.</div>",
        unsafe_allow_html=True
    )

    article_count = st.slider("가져올 기사 수", min_value=3, max_value=10, value=6)

    st.markdown("#### 브랜드 또는 서비스명")
    col1, col2 = st.columns([5, 1.2])

    with col1:
        keyword = st.text_input(
            "브랜드 또는 서비스명",
            placeholder="예: 쿠팡, 스타벅스, 무신사, 올리브영",
            label_visibility="collapsed"
        )

    with col2:
        search_btn = st.button("분석 시작", use_container_width=True)

    st.caption("검색 결과는 Supabase 데이터베이스에 자동 저장됩니다. 동일 URL은 중복 저장되지 않습니다.")

    if search_btn:
        if not keyword.strip():
            st.warning("검색할 브랜드 또는 서비스명을 입력하세요.")
            st.stop()

        with st.spinner("최신 소비자 반응을 검색하고 분석하는 중입니다..."):
            search_prompt = f"""
너는 소비자 인사이트 분석가야.

다음 브랜드 또는 서비스와 관련된 최신 기사와 소비자 반응을 검색해줘.

검색어: {keyword}
가져올 개수: {article_count}건

중요한 방향:
1. 부정 이슈만 찾지 말고, 긍정/중립/부정 반응이 섞이도록 최근 자료를 균형 있게 찾아.
2. 신제품, 서비스 개선, 매출 성장, 고객 만족, 브랜드 호감 등 긍정 반응도 포함해.
3. 논란, 불만, 가격, 배송, 품질, 고객응대, 서비스 장애 같은 부정 반응도 포함해.
4. 단순 기업 공시보다 소비자 반응이나 브랜드 이미지와 연결되는 자료를 우선해.
5. 뉴스, 기사, 공식 발표, 신뢰 가능한 웹 문서를 중심으로 찾아.
6. 각 항목에 제목, 출처, 날짜, 실제 URL, 핵심 내용을 포함해.
7. URL은 실제 검색 결과에 있는 링크만 사용해.
"""

            try:
                search_response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=search_prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                        temperature=0.0,
                    ),
                )
            except Exception as e:
                st.error("Gemini 검색 호출 중 문제가 발생했습니다.")
                st.caption(str(e))
                st.stop()

            json_prompt = f"""
아래 검색 결과를 바탕으로 소비자 반응 분석 결과를 JSON 배열로만 변환해줘.

검색어: {keyword}
목표 개수: {article_count}건

검색 결과:
{search_response.text}

반드시 아래 형식의 JSON 배열만 출력해.
설명문, 마크다운, 코드블록은 절대 넣지 마.

[
  {{
    "title": "기사 제목",
    "source": "출처",
    "news_date": "YYYY-MM-DD",
    "url": "기사 URL",
    "summary": "소비자 반응 또는 브랜드 이미지와 연결한 2~3문장 요약",
    "sentiment": "Positive 또는 Neutral 또는 Negative",
    "risk_level": "Low 또는 Medium 또는 High"
  }}
]

분류 기준:
- 소비자 반응이 긍정적이거나 브랜드 이미지에 도움이 되면 Positive
- 단순 정보 전달이거나 영향이 명확하지 않으면 Neutral
- 불만, 논란, 신뢰 하락, 고객 이탈 가능성이 있으면 Negative
- risk_level은 평판 리스크가 작으면 Low, 일부 우려면 Medium, 확산 가능성이 크면 High

주의:
- 반드시 {article_count}건에 가깝게 만들어.
- 전부 Negative로 만들지 말고 실제 내용에 따라 Positive, Neutral, Negative를 구분해.
"""

            try:
                json_response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=json_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                    ),
                )
            except Exception as e:
                st.error("AI 응답을 JSON으로 변환하는 중 문제가 발생했습니다.")
                st.caption(str(e))
                st.stop()

            items = extract_json_array(json_response.text)

            if not items:
                st.error("AI 응답을 JSON으로 해석하지 못했습니다. 잠시 후 다시 검색해 보세요.")
                with st.expander("AI 원본 응답 확인"):
                    st.write(search_response.text)
                    st.write(json_response.text)
                st.stop()

            items = replace_with_best_url(items, search_response, keyword)

            valid_results = []

            for item in items[:article_count]:
                record = {
                    "keyword": keyword.strip(),
                    "title": safe_text(item.get("title"), "제목 없음"),
                    "source": safe_text(item.get("source"), "출처 미상"),
                    "news_date": safe_text(item.get("news_date"), ""),
                    "url": safe_text(item.get("url"), google_news_fallback(keyword, item.get("title", ""))),
                    "summary": safe_text(item.get("summary"), "요약 없음"),
                    "sentiment": normalize_sentiment(item.get("sentiment", "Neutral")),
                    "risk_level": normalize_risk(item.get("risk_level", "Low")),
                }

                if not is_valid_url(record["url"]):
                    record["url"] = google_news_fallback(keyword, record["title"])

                valid_results.append(record)

            saved_count = 0
            duplicate_count = 0

            for res in valid_results:
                st.markdown(f"""
                <div class="insight-card">
                    <div class="insight-title">{res["title"]}</div>
                    <div class="meta">{res["source"]} · {res["news_date"]}</div>
                    <div class="summary">{res["summary"]}</div>
                    <span class="badge {sentiment_badge_class(res["sentiment"])}">감성: {sentiment_ko(res["sentiment"])}</span>
                    <span class="badge {risk_badge_class(res["risk_level"])}">리스크: {risk_ko(res["risk_level"])}</span>
                </div>
                """, unsafe_allow_html=True)

                st.link_button("원문 또는 관련 뉴스 보기", res["url"])

                try:
                    supabase.table("consumer_insights").insert(res).execute()
                    saved_count += 1
                except Exception as e:
                    if "23505" in str(e) or "duplicate" in str(e).lower():
                        duplicate_count += 1
                    else:
                        st.warning(f"일부 데이터 저장 중 문제가 발생했습니다: {e}")

            st.toast(f"분석 완료: 신규 저장 {saved_count}건, 중복 생략 {duplicate_count}건")


elif selected == "인사이트 저장소":
    st.markdown("<div class='main-title'>인사이트 저장소</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-text'>저장된 소비자 반응 분석 결과를 검색하고 필터링할 수 있습니다.</div>",
        unsafe_allow_html=True
    )

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
        keyword_filter = st.text_input("키워드/제목/요약 검색", placeholder="예: 쿠팡, 배송, 가격")

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

    display_cols = ["keyword", "title", "source", "news_date", "감성", "리스크", "summary", "url", "created_at"]
    display_cols = [c for c in display_cols if c in view_df.columns]

    rename_cols = {
        "keyword": "검색어",
        "title": "제목",
        "source": "출처",
        "news_date": "기사 날짜",
        "summary": "요약",
        "url": "URL",
        "created_at": "저장일시"
    }

    st.dataframe(
        view_df[display_cols].rename(columns=rename_cols),
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
    st.markdown("<div class='main-title'>분석 대시보드</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-text'>저장된 데이터를 바탕으로 브랜드별 언급량, 감성 비율, 리스크 수준을 확인합니다.</div>",
        unsafe_allow_html=True
    )

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
    positive_count = len(df[df["sentiment"] == "Positive"])
    neutral_count = len(df[df["sentiment"] == "Neutral"])
    negative_count = len(df[df["sentiment"] == "Negative"])
    negative_rate = (negative_count / total) * 100 if total else 0
    high_risk_count = len(df[df["risk_level"] == "High"])
    top_keyword = df["keyword"].mode()[0] if "keyword" in df.columns and not df["keyword"].empty else "-"

    k1, k2, k3, k4 = st.columns(4)

    k1.markdown(f"<div class='kpi-card'><div class='kpi-label'>총 저장 건수</div><div class='kpi-value'>{total}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi-card'><div class='kpi-label'>긍정 / 중립 / 부정</div><div class='kpi-value'>{positive_count} / {neutral_count} / {negative_count}</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi-card'><div class='kpi-label'>부정 반응 비율</div><div class='kpi-value' style='color:#DC2626;'>{negative_rate:.1f}%</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi-card'><div class='kpi-label'>최다 검색 키워드</div><div class='kpi-value'>{top_keyword}</div></div>", unsafe_allow_html=True)

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 키워드별 누적 언급량")
        keyword_df = df["keyword"].value_counts().reset_index()
        keyword_df.columns = ["키워드", "건수"]

        fig1 = px.bar(keyword_df, x="키워드", y="건수", text="건수", color_discrete_sequence=["#4F46E5"])
        fig1.update_traces(textposition="outside")
        fig1.update_layout(xaxis_title="키워드", yaxis_title="저장 건수", plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.markdown("#### 일자별 저장 추이")
        df["날짜"] = pd.to_datetime(df["created_at"], errors="coerce").dt.date
        trend_df = df.dropna(subset=["날짜"]).groupby("날짜").size().reset_index(name="건수")

        fig2 = px.line(trend_df, x="날짜", y="건수", markers=True, color_discrete_sequence=["#7C3AED"])
        fig2.update_layout(xaxis_title="날짜", yaxis_title="저장 건수", plot_bgcolor="white", paper_bgcolor="white")
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
            hole=0.45,
            color="감성",
            color_discrete_map={"긍정": "#10B981", "중립": "#94A3B8", "부정": "#EF4444"}
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
            color_discrete_map={"낮음": "#2563EB", "보통": "#F59E0B", "높음": "#EF4444"}
        )
        fig4.update_traces(textposition="outside")
        fig4.update_layout(xaxis_title="리스크 수준", yaxis_title="건수", plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig4, use_container_width=True)
