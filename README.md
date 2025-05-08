# 🌍 비체계적 위험 분석 Streamlit 앱 (고도화 버전)
import streamlit as st
import requests
import re
import pandas as pd
import datetime
import plotly.express as px
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# 기본 위험 키워드 정의
BASE_RISK_KEYWORDS = {
    "침공": 1.5, "전쟁": 1.4, "비판": 1.2, "논쟁": 1.3, "제3차 세계대전": 1.6,
    "분열": 1.5, "위협": 1.4, "훼손": 1.6, "실수": 1.2, "불만": 1.3,
    "위기": 1.4, "생활비": 1.1, "범죄": 1.2, "불안감": 1.3, "관세": 1.3,
    "합병": 1.5,
    "recession": 1.5, "misleading": 1.3, "contracted": 1.4, "discontent": 1.3,
    "blamed": 1.1, "disrupted": 1.5, "scorn": 1.2, "failed": 1.6, "shortages": 1.3,
    "war": 1.7, "tariffs": 1.4, "levies": 1.2,
}

POSITIVE_KEYWORDS = {
    "평화": -0.8, "번영": -0.6, "재건": -0.5, "성장": -0.4, "협정": -0.3,
    "파트너십": -0.3, "유치": -0.4, "회복": -0.5, "기회": -0.4
}

@st.cache_resource
def load_sentiment_model():
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    model = BertForSequenceClassification.from_pretrained('beomi/kcbert-base', num_labels=3)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def calculate_risk_score(text, keywords):
    score = 0
    for word, weight in keywords.items():
        count = len(re.findall(re.escape(word), text))
        score += weight * count
    return score

def kobert_sentiment(pipeline, text):
    result = pipeline(text[:512])[0]
    label = result['label']
    score = result['score']
    if label == 'LABEL_0': return -score
    elif label == 'LABEL_2': return score
    return 0

def extract_risk_keywords_tfidf(texts, top_n=20):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(texts)
    tfidf_scores = X.sum(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame({'word': feature_names, 'score': tfidf_scores})
    top_keywords = tfidf_df.sort_values('score', ascending=False).head(top_n)
    return {row['word']: round(row['score'], 2) for _, row in top_keywords.iterrows()}

def visualize_keyword_contributions(text, keywords):
    contrib = []
    for k, w in keywords.items():
        count = text.lower().count(k)
        if count > 0:
            contrib.append({"키워드": k, "빈도": count, "기여도": w * count})
    df = pd.DataFrame(contrib).sort_values("기여도", ascending=False)
    st.plotly_chart(px.bar(df, x="키워드", y="기여도", text="빈도", title="문서 내 키워드별 위험 기여도"))

# UI 시작
st.set_page_config(page_title="Unsystematic Risk Analyzer", layout="wide")
st.title("📉 뉴스 기반 비체계적 위험 분석기")

keyword = st.text_input("🔍 뉴스 키워드", "아르헨티나 정치 불안")
api_key = st.text_input("🔑 GNews API Key", type="password")
custom_article = st.text_area("✍️ 분석할 기사 텍스트 직접 입력 (선택)", height=300)

# 키워드 설정
st.sidebar.title("⚙️ 키워드 설정")
keyword_mode = st.sidebar.radio("키워드 설정 방식", ["기본 키워드 사용", "직접 추가", "TF-IDF 기반 자동 추출"])
custom_risks = {}

if keyword_mode == "직접 추가":
    with st.sidebar.expander("위험 키워드 추가"):
        new_kw = st.text_input("키워드 입력")
        new_weight = st.number_input("가중치", 0.1, 2.0, step=0.1)
        if st.button("➕ 추가"):
            custom_risks[new_kw] = new_weight
            st.success(f"추가됨: {new_kw} ({new_weight})")
    keywords = BASE_RISK_KEYWORDS.copy()
    keywords.update(custom_risks)
else:
    keywords = BASE_RISK_KEYWORDS.copy()

# 분석 시작
if st.button("🚀 분석 시작"):
    with st.spinner("모델 로딩 중..."):
        nlp = load_sentiment_model()

    if custom_article.strip():
        articles = [custom_article.strip()]
    else:
        url = f"https://gnews.io/api/v4/search?q={keyword}&lang=ko&token={api_key}"
        r = requests.get(url)
        articles = [a['title'] + ' ' + (a['description'] or '') for a in r.json().get('articles', [])]

    if not articles:
        st.warning("뉴스가 없습니다.")
    else:
        if keyword_mode == "TF-IDF 기반 자동 추출":
            keywords = extract_risk_keywords_tfidf(articles)
            st.sidebar.success("TF-IDF 키워드 자동 추출 완료!")

        rows = []
        for t in articles:
            base_score = calculate_risk_score(t, keywords)
            bonus = calculate_risk_score(t, POSITIVE_KEYWORDS)
            final_score = base_score + bonus
            polarity = kobert_sentiment(nlp, t)
            score = round(final_score * (1 - polarity), 2)
            rows.append({"텍스트": t, "위험 점수": final_score, "감성 점수": round(polarity, 2), "최종 위험 지수": score,
                         "시간": datetime.datetime.now()})

        df = pd.DataFrame(rows)
        st.success(f"총 {len(df)}개 문서 분석 완료!")
        st.dataframe(df.sort_values("최종 위험 지수", ascending=False))

        st.subheader("📊 시각화 결과")
        st.plotly_chart(px.line(df, x="시간", y="최종 위험 지수", title="시간별 위험 추세"))
        st.plotly_chart(px.histogram(df, x="최종 위험 지수", nbins=15, title="위험 분포"))
        st.plotly_chart(px.scatter(df, x="감성 점수", y="위험 점수", color="최종 위험 지수", size="최종 위험 지수",
                                   hover_data=["텍스트"], title="감성 vs 위험 점수 산점도"))

        top5 = df.sort_values("최종 위험 지수", ascending=False).head(5)
        st.subheader("🔥 고위험 기사 Top 5")
        for i, row in top5.iterrows():
            st.markdown(f"**{i+1}. 위험도: {row['최종 위험 지수']} / 감성: {row['감성 점수']}**")
            st.caption(row['텍스트'][:400] + ("..." if len(row['텍스트']) > 400 else ""))

        if st.checkbox("문서별 키워드 기여도 보기"):
            for i, row in df.iterrows():
                st.markdown(f"**문서 {i+1}**")
                st.caption(row['텍스트'][:300] + "...")
                visualize_keyword_contributions(row['텍스트'], keywords)

        st.download_button("📥 CSV 다운로드", df.to_csv(index=False).encode(), "risk_analysis.csv", "text/csv")
