import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from supabase import create_client, Client
from google import genai
from google.genai import types
import json
import re

# 1. Page Configuration
st.set_page_config(page_title="Consumer Pulse AI", layout="wide")

# CSS: Modern Minimalist Style (Indigo & White)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #F9FAFB; }
    .stButton>button { background-color: #4F46E5; color: white; border-radius: 8px; border: none; padding: 0.5rem 1rem; }
    .stButton>button:hover { background-color: #4338CA; border: none; }
    .card { background-color: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1rem; border-left: 5px solid #4F46E5; }
    .kpi-card { background-color: white; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .badge-pos { background-color: #DCFCE7; color: #166534; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
    .badge-neu { background-color: #F3F4F6; color: #374151; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
    .badge-neg { background-color: #FEE2E2; color: #991B1B; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
    </style>
""", unsafe_allow_html=True)

# 2. API & DB Client Setup
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error("API 키 설정이 필요합니다. st.secrets를 확인하세요.")
    st.stop()

# 3. Sidebar Navigation
with st.sidebar:
    st.title("Pulse AI")
    selected = option_menu(
        menu_title=None,
        options=["Consumer Search", "Insight Archive", "Analytics Dashboard"],
        icons=["search", "archive", "bar-chart-line"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"5px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#4F46E5"},
        }
    )

# --- Logic: Search & AI Analysis ---
if selected == "Consumer Search":
    st.subheader("Consumer Reaction Search")
    col1, col2 = st.columns([3, 1])
    with col1:
        keyword = st.text_input("Brand or Service Keyword", placeholder="e.g. Starbucks, Coupang, Musinsa")
    with col2:
        st.write(" ")
        search_btn = st.button("Analyze Now")

    if search_btn and keyword:
        with st.spinner("AI가 최신 정보를 분석 중입니다..."):
            prompt = f"""
            Search for the 2 latest consumer reactions and news regarding '{keyword}', specifically focusing on potential dissatisfaction or issues.
            Return the results as a JSON array of objects with these keys: 
            'title', 'source', 'news_date', 'url', 'summary', 'sentiment', 'risk_level'.
            - sentiment: Positive, Neutral, Negative
            - risk_level: Low, Medium, High
            Use the actual news URL found in the search grounding.
            """
            
            # Gemini 2.0 Flash with Google Search Grounding
           response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.0,
    ),
)
            
            # URL Hallucination Prevention Logic
            raw_json = json.loads(response.text)
            grounding_chunks = response.candidates[0].grounding_metadata.grounding_chunks
            
            valid_results = []
            for item in raw_json:
                # Match title with grounding chunks to get real URL
                for chunk in grounding_chunks:
                    if chunk.web and (item['title'] in chunk.web.title or chunk.web.title in item['title']):
                        real_url = chunk.web.uri
                        if real_url.startswith("http") and "grounding-api-redirect" not in real_url:
                            item['url'] = real_url
                            break
                item['keyword'] = keyword
                valid_results.append(item)

            # DB Save & UI Output
            saved_count = 0
            for res in valid_results:
                # Card UI
                with st.container():
                    st.markdown(f"""
                    <div class="card">
                        <div style="display:flex; justify-content:space-between;">
                            <h4 style="margin:0;">{res['title']}</h4>
                            <span>{res['source']} | {res['news_date']}</span>
                        </div>
                        <p style="margin:10px 0;">{res['summary']}</p>
                        <div>
                            <span class="badge-{'pos' if res['sentiment']=='Positive' else 'neu' if res['sentiment']=='Neutral' else 'neg'}">{res['sentiment']}</span>
                            <span style="color:{'#3B82F6' if res['risk_level']=='Low' else '#F59E0B' if res['risk_level']=='Medium' else '#EF4444'}; font-weight:bold; margin-left:10px;">Risk: {res['risk_level']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.link_button("View Original Article", res['url'])

                # Supabase Insert
                try:
                    supabase.table("consumer_insights").insert(res).execute()
                    saved_count += 1
                except Exception as e:
                    if "23505" in str(e): # Duplicate URL
                        pass
            
            if saved_count > 0:
                st.toast(f"{saved_count}건의 새로운 데이터가 저장되었습니다.")
            else:
                st.toast("이미 존재하는 데이터입니다.")

# --- Logic: Archive ---
elif selected == "Insight Archive":
    st.subheader("Insight Archive")
    
    # Filters
    data_res = supabase.table("consumer_insights").select("*").order("created_at", desc=True).execute()
    df = pd.DataFrame(data_res.data)

    if not df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            f_keyword = st.multiselect("Keyword", options=df['keyword'].unique())
        with col2:
            f_sentiment = st.multiselect("Sentiment", options=["Positive", "Neutral", "Negative"])
        with col3:
            f_risk = st.multiselect("Risk Level", options=["Low", "Medium", "High"])

        filtered_df = df.copy()
        if f_keyword: filtered_df = filtered_df[filtered_df['keyword'].isin(f_keyword)]
        if f_sentiment: filtered_df = filtered_df[filtered_df['sentiment'].isin(f_sentiment)]
        if f_risk: filtered_df = filtered_df[filtered_df['risk_level'].isin(f_risk)]

        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        
        csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("Download CSV", data=csv, file_name="consumer_insights.csv", mime="text/csv")
    else:
        st.info("저장된 데이터가 없습니다.")

# --- Logic: Dashboard ---
elif selected == "Analytics Dashboard":
    st.subheader("Strategic Dashboard")
    
    data_res = supabase.table("consumer_insights").select("*").execute()
    df = pd.DataFrame(data_res.data)

    if not df.empty:
        # KPI Metrics
        total = len(df)
        neg_pct = (len(df[df['sentiment'] == 'Negative']) / total) * 100
        high_risk = len(df[df['risk_level'] == 'High'])
        top_keyword = df['keyword'].mode()[0] if not df['keyword'].empty else "N/A"

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f"<div class='kpi-card'><small>Total Insights</small><h3>{total}</h3></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi-card'><small>Negative Rate</small><h3 style='color:#EF4444;'>{neg_pct:.1f}%</h3></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi-card'><small>High Risk Cases</small><h3 style='color:#EF4444;'>{high_risk}</h3></div>", unsafe_allow_html=True)
        k4.markdown(f"<div class='kpi-card'><small>Top Keyword</small><h3>{top_keyword}</h3></div>", unsafe_allow_html=True)

        st.write("---")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Mentions by Keyword")
            fig1 = px.bar(df['keyword'].value_counts().reset_index(), x='keyword', y='count', color_discrete_sequence=['#4F46E5'])
            st.plotly_chart(fig1, use_container_width=True)
        
        with c2:
            st.markdown("#### Insight Collection Trend")
            df['date'] = pd.to_datetime(df['created_at']).dt.date
            trend_df = df.groupby('date').size().reset_index(name='count')
            fig2 = px.line(trend_df, x='date', y='count', markers=True, color_discrete_sequence=['#4F46E5'])
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### Sentiment Distribution")
            fig3 = px.pie(df, names='sentiment', color='sentiment', color_discrete_map={'Positive':'#10B981', 'Neutral':'#9CA3AF', 'Negative':'#EF4444'})
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            st.markdown("#### Risk Level Distribution")
            fig4 = px.bar(df['risk_level'].value_counts().reset_index(), x='risk_level', y='count', color='risk_level', 
                          color_discrete_map={'Low':'#3B82F6', 'Medium':'#F59E0B', 'High':'#EF4444'})
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("데이터가 충분하지 않습니다.")
