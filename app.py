import streamlit as st
import pandas as pd
import re
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from snownlp import SnowNLP

# ==========================
# 0. 页面全局配置
# ==========================
st.set_page_config(page_title="竞品舆情分析罗盘", page_icon="🧭", layout="wide")
st.title("🧭 竞品舆情自动化分析罗盘")
st.markdown("上传应用商店评论数据，一键提取核心槽点与情感健康度。")

# ==========================
# 1. 核心处理函数 (使用 st.cache_data 缓存计算，让网页飞起来)
# ==========================
@st.cache_data
def process_data(df):
    # 停用词库
    stop_words = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '小红书', '软件', '非常', '可以', '不错', '不能'}
    
    # 清洗与分词
    def advanced_clean_and_cut(text):
        text = str(text)
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        words = pseg.lcut(text)
        useful_words = [w for w, f in words if len(w) > 1 and w not in stop_words and (f.startswith('n') or f.startswith('v') or f.startswith('a'))]
        return " ".join(useful_words)
        
    df['分词内容'] = df['内容'].apply(advanced_clean_and_cut)
    
    # 情感打分
    def get_sentiment(text):
        try:
            return round(SnowNLP(str(text)).sentiments * 100, 2)
        except:
            return 50.0
            
    df['情绪指数'] = df['内容'].apply(get_sentiment)
    return df

@st.cache_data
def extract_keywords(text_series, top_n=10):
    if text_series.empty or text_series.str.strip().eq('').all():
        return pd.DataFrame()
    vectorizer = TfidfVectorizer(max_features=500)
    tfidf_matrix = vectorizer.fit_transform(text_series)
    words = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    df_words = pd.DataFrame({'特征词': words, '权重': scores}).sort_values(by='权重', ascending=False).head(top_n)
    return df_words.set_index('特征词')

# ==========================
# 2. 前端交互界面
# ==========================
# 侧边栏：上传文件
with st.sidebar:
    st.header("📂 数据接入")
    uploaded_file = st.file_uploader("请上传评论数据集 (CSV格式)", type=['csv'])
    st.info("数据需包含 [评分] 和 [内容] 两列。")

# 主界面逻辑
if uploaded_file is not None:
    # 读取数据
    raw_df = pd.read_csv(uploaded_file)
    
    with st.spinner('AI 引擎正在进行自然语言处理与情感计算，请稍候...'):
        df = process_data(raw_df)
        
    st.success("数据处理完成！")
    
    # 顶层数据看板 (KPI Metrics)
    st.header("📊 全局健康度看板")
    col1, col2, col3 = st.columns(3)
    col1.metric("总评论数", len(df))
    col2.metric("平均星级", round(df['评分'].mean(), 2))
    
    avg_sentiment = df['情绪指数'].mean()
    # 动态颜色提示
    sentiment_delta = "健康" if avg_sentiment > 60 else "需警惕" if avg_sentiment > 40 else "极其危险"
    col3.metric("大盘情绪指数", f"{round(avg_sentiment, 1)} 分", sentiment_delta, delta_color="normal" if avg_sentiment > 50 else "inverse")
    
    st.divider()

    # 业务下钻区 (使用标签页)
    tab1, tab2, tab3 = st.tabs(["🚨 核心槽点分析 (1-3星)", "✨ 核心爽点分析 (4-5星)", "🕵️‍♂️ 异常用户抓取"])
    
    with tab1:
        st.subheader("导致用户流失的核心因素")
        bad_df = df[df['评分'] <= 3]
        if not bad_df.empty:
            bad_keywords = extract_keywords(bad_df['分词内容'], 15)
            # 使用 Streamlit 自带的炫酷横向柱状图
            st.bar_chart(bad_keywords)
            
            st.markdown("**高频差评原声：**")
            st.dataframe(bad_df.sort_values(by='情绪指数').head(5)[['评分', '情绪指数', '内容']], use_container_width=True)

    with tab2:
        st.subheader("驱动用户好评的核心要素")
        good_df = df[df['评分'] >= 4]
        if not good_df.empty:
            good_keywords = extract_keywords(good_df['分词内容'], 15)
            st.bar_chart(good_keywords)

    with tab3:
        st.subheader("情绪严重错位预警 (高星级但情绪极度负面)")
        fake_good = df[(df['评分'] >= 4) & (df['情绪指数'] < 30)]
        if not fake_good.empty:
            st.dataframe(fake_good[['评分', '情绪指数', '内容']], use_container_width=True)
        else:
            st.write("目前未发现明显的阴阳怪气评论。")
else:
    # 默认引导提示
    st.info("👈 请在左侧上传你在上一步爬取的 `xiaohongshu_reviews.csv` 文件以生成分析报告。")