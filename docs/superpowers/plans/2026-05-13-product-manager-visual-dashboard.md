# Product Manager Visual Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个产品经理视角的自动可视化看板，把评论数据转换成健康指标、问题优先级、诊断图表和可行动评论池。

**Architecture:** 新增 `visual_analysis.py` 负责纯数据分析逻辑，包括指标计算、问题分类、风险识别、优先级评分、图表数据准备和筛选。`app.py` 继续负责 Streamlit 页面布局、筛选控件、图表展示和 AI 按钮，并在需要时把 `ai_analysis.py` 的 AI 痛点严重度融合进本地优先级评分。

**Tech Stack:** Python, pandas, scikit-learn, Streamlit, unittest, SnowNLP, DeepSeek optional AI insights.

---

## 文件结构

- Create: `visual_analysis.py`
  - 负责产品经理看板的本地数据逻辑。
  - 不导入 Streamlit。
  - 不调用 DeepSeek。
  - 对外提供稳定函数给 `app.py` 使用。

- Create: `tests/test_visual_analysis.py`
  - 使用项目现有的 `unittest` 风格。
  - 只测试本地逻辑，不启动 Streamlit，不访问网络。

- Modify: `app.py`
  - 导入 `visual_analysis.py` 中的指标、筛选、图表数据函数。
  - 在上传 CSV 并完成 `process_data` 后生成产品经理看板。
  - 保留现有 AI 洞察按钮和渲染逻辑。

- Unchanged: `ai_analysis.py`
  - 保持 AI 配置、调用、解析职责。
  - 只把已经标准化后的 `insights` 传给 `visual_analysis.calculate_priority_table()`。

- Unchanged: `requirenments.txt`
  - 当前依赖已经包含 `streamlit`、`pandas`、`scikit-learn`、`snownlp` 和 `requests`。
  - 第一版看板使用 Streamlit 内置图表，不新增依赖。

当前目录不是 git 仓库，所有任务中的提交步骤都记录为跳过。

---

### Task 1: 为本地可视化分析写失败测试

**Files:**
- Create: `tests/test_visual_analysis.py`
- Create later: `visual_analysis.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_visual_analysis.py`:

```python
import unittest

import pandas as pd

import visual_analysis


class VisualAnalysisTests(unittest.TestCase):
    def make_reviews(self):
        return pd.DataFrame(
            [
                {"评分": 1, "内容": "无故封号，申诉没人处理，客服也找不到", "情绪指数": 8},
                {"评分": 1, "内容": "账号被封，人工客服一直没有回复", "情绪指数": 12},
                {"评分": 2, "内容": "审核太严格，笔记莫名违规", "情绪指数": 25},
                {"评分": 3, "内容": "广告太多，推荐质量下降", "情绪指数": 45},
                {"评分": 5, "内容": "内容很多，种草体验不错", "情绪指数": 86},
                {"评分": 4, "内容": "高分但是广告真的太多了", "情绪指数": 22},
            ]
        )

    def test_prepare_dashboard_data_adds_category_and_risk_label(self):
        prepared = visual_analysis.prepare_dashboard_data(self.make_reviews())

        self.assertIn("问题类型", prepared.columns)
        self.assertIn("风险标签", prepared.columns)
        self.assertEqual(prepared.iloc[0]["问题类型"], "账号类")
        self.assertIn("极端负面", prepared.iloc[0]["风险标签"])
        self.assertIn("高星低情绪", prepared.iloc[5]["风险标签"])

    def test_calculate_health_metrics_counts_core_values(self):
        prepared = visual_analysis.prepare_dashboard_data(self.make_reviews())

        metrics = visual_analysis.calculate_health_metrics(prepared)

        self.assertEqual(metrics["total_reviews"], 6)
        self.assertEqual(metrics["average_rating"], 2.67)
        self.assertEqual(metrics["negative_ratio"], 66.67)
        self.assertEqual(metrics["average_sentiment"], 33.0)
        self.assertEqual(metrics["high_risk_count"], 4)

    def test_filter_reviews_by_rating_sentiment_category_keyword_and_risk(self):
        prepared = visual_analysis.prepare_dashboard_data(self.make_reviews())

        filtered = visual_analysis.filter_reviews(
            prepared,
            rating_range=(1, 2),
            sentiment_range=(0, 30),
            categories=["账号类"],
            keyword="客服",
            high_risk_only=True,
        )

        self.assertEqual(len(filtered), 2)
        self.assertTrue((filtered["问题类型"] == "账号类").all())
        self.assertTrue(filtered["内容"].str.contains("客服").all())

    def test_empty_data_returns_safe_metrics(self):
        empty = pd.DataFrame(columns=["评分", "内容", "情绪指数"])
        prepared = visual_analysis.prepare_dashboard_data(empty)

        metrics = visual_analysis.calculate_health_metrics(prepared)

        self.assertEqual(metrics["total_reviews"], 0)
        self.assertEqual(metrics["average_rating"], 0.0)
        self.assertEqual(metrics["negative_ratio"], 0.0)
        self.assertEqual(metrics["average_sentiment"], 0.0)
        self.assertEqual(metrics["high_risk_count"], 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
python -m unittest tests.test_visual_analysis -v
```

Expected:

```text
ModuleNotFoundError: No module named 'visual_analysis'
```

- [ ] **Step 3: 记录提交限制**

Run:

```powershell
Test-Path .git
```

Expected:

```text
False
```

Commit is skipped because this workspace is not a git repository.

---

### Task 2: 实现数据准备、问题分类、风险识别和健康指标

**Files:**
- Create: `visual_analysis.py`
- Test: `tests/test_visual_analysis.py`

- [ ] **Step 1: 创建最小实现**

Create `visual_analysis.py`:

```python
import re

import pandas as pd


RATING_COLUMN = "评分"
CONTENT_COLUMN = "内容"
SENTIMENT_COLUMN = "情绪指数"
TOKEN_COLUMN = "分词内容"
CATEGORY_COLUMN = "问题类型"
RISK_LABEL_COLUMN = "风险标签"

ISSUE_KEYWORDS = {
    "账号类": ["封号", "禁言", "限流", "实名", "申诉", "账号", "登录", "注销"],
    "审核类": ["审核", "违规", "举报", "笔记", "内容规则", "误判"],
    "客服类": ["客服", "人工", "反馈", "没人处理", "无人处理", "回复"],
    "体验类": ["卡顿", "闪退", "加载", "广告", "推荐", "崩溃", "发热"],
    "内容类": ["搜索", "内容", "社区", "种草", "质量", "博主"],
}

HIGH_SEVERITY_SENTIMENT = 20
LOW_SENTIMENT = 30
NEGATIVE_RATING_MAX = 3


def classify_issue(text):
    content = str(text or "")
    for category, keywords in ISSUE_KEYWORDS.items():
        if any(keyword in content for keyword in keywords):
            return category
    return "其他类"


def build_risk_label(row):
    labels = []
    rating = float(row.get(RATING_COLUMN, 0) or 0)
    sentiment = float(row.get(SENTIMENT_COLUMN, 50) or 50)

    if rating <= 2:
        labels.append("低星差评")
    if sentiment <= HIGH_SEVERITY_SENTIMENT:
        labels.append("极端负面")
    if rating >= 4 and sentiment < LOW_SENTIMENT:
        labels.append("高星低情绪")

    return "、".join(labels) if labels else "正常"


def is_high_risk_label(label):
    return str(label or "") != "正常"


def prepare_dashboard_data(df):
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                RATING_COLUMN,
                CONTENT_COLUMN,
                SENTIMENT_COLUMN,
                CATEGORY_COLUMN,
                RISK_LABEL_COLUMN,
            ]
        )

    required_columns = {RATING_COLUMN, CONTENT_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = "、".join(sorted(missing_columns))
        raise ValueError(f"缺少必要列：{missing}")

    prepared = df.copy()
    prepared[CONTENT_COLUMN] = prepared[CONTENT_COLUMN].fillna("").astype(str).str.strip()
    prepared = prepared[prepared[CONTENT_COLUMN] != ""].copy()
    prepared[RATING_COLUMN] = pd.to_numeric(prepared[RATING_COLUMN], errors="coerce")
    prepared = prepared.dropna(subset=[RATING_COLUMN]).copy()

    if SENTIMENT_COLUMN not in prepared.columns:
        prepared[SENTIMENT_COLUMN] = 50.0

    prepared[SENTIMENT_COLUMN] = (
        pd.to_numeric(prepared[SENTIMENT_COLUMN], errors="coerce")
        .fillna(50.0)
        .clip(lower=0, upper=100)
    )

    prepared[CATEGORY_COLUMN] = prepared[CONTENT_COLUMN].apply(classify_issue)
    prepared[RISK_LABEL_COLUMN] = prepared.apply(build_risk_label, axis=1)
    return prepared


def calculate_health_metrics(df):
    if df is None or df.empty:
        return {
            "total_reviews": 0,
            "average_rating": 0.0,
            "negative_ratio": 0.0,
            "average_sentiment": 0.0,
            "high_risk_count": 0,
        }

    total_reviews = int(len(df))
    negative_count = int((df[RATING_COLUMN] <= NEGATIVE_RATING_MAX).sum())
    high_risk_count = int(df[RISK_LABEL_COLUMN].apply(is_high_risk_label).sum())

    return {
        "total_reviews": total_reviews,
        "average_rating": round(float(df[RATING_COLUMN].mean()), 2),
        "negative_ratio": round(negative_count / total_reviews * 100, 2),
        "average_sentiment": round(float(df[SENTIMENT_COLUMN].mean()), 2),
        "high_risk_count": high_risk_count,
    }


def filter_reviews(
    df,
    rating_range=(1, 5),
    sentiment_range=(0, 100),
    categories=None,
    keyword="",
    high_risk_only=False,
):
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])

    filtered = df.copy()
    min_rating, max_rating = rating_range
    min_sentiment, max_sentiment = sentiment_range

    filtered = filtered[
        filtered[RATING_COLUMN].between(min_rating, max_rating)
        & filtered[SENTIMENT_COLUMN].between(min_sentiment, max_sentiment)
    ].copy()

    if categories:
        filtered = filtered[filtered[CATEGORY_COLUMN].isin(categories)].copy()

    keyword = str(keyword or "").strip()
    if keyword:
        filtered = filtered[
            filtered[CONTENT_COLUMN].str.contains(re.escape(keyword), case=False, na=False)
        ].copy()

    if high_risk_only:
        filtered = filtered[filtered[RISK_LABEL_COLUMN].apply(is_high_risk_label)].copy()

    return filtered
```

- [ ] **Step 2: 运行测试确认通过**

Run:

```powershell
python -m unittest tests.test_visual_analysis -v
```

Expected:

```text
Ran 4 tests

OK
```

- [ ] **Step 3: 运行语法检查**

Run:

```powershell
python -m compileall visual_analysis.py tests
```

Expected:

```text
Compiling 'visual_analysis.py'...
```

The command exits with code 0.

- [ ] **Step 4: 记录提交限制**

Run:

```powershell
Test-Path .git
```

Expected:

```text
False
```

Commit is skipped because this workspace is not a git repository.

---

### Task 3: 测试并实现问题优先级评分和 AI 严重度融合

**Files:**
- Modify: `tests/test_visual_analysis.py`
- Modify: `visual_analysis.py`

- [ ] **Step 1: 追加失败测试**

Append these methods inside `VisualAnalysisTests` in `tests/test_visual_analysis.py`:

```python
    def test_calculate_priority_table_ranks_severe_account_problem_first(self):
        prepared = visual_analysis.prepare_dashboard_data(self.make_reviews())

        priority = visual_analysis.calculate_priority_table(prepared, top_n=3)

        self.assertEqual(priority.iloc[0]["问题类型"], "账号类")
        self.assertGreater(priority.iloc[0]["优先级分数"], priority.iloc[1]["优先级分数"])
        self.assertEqual(priority.iloc[0]["严重程度"], "high")
        self.assertIn("封号", priority.iloc[0]["代表评论"])

    def test_calculate_priority_table_merges_ai_severity_and_suggestion(self):
        prepared = visual_analysis.prepare_dashboard_data(self.make_reviews())
        insights = {
            "pain_points": [
                {
                    "name": "广告过多",
                    "severity": "high",
                    "suggestion": "降低信息流广告密度，优先检查推荐页体验。",
                }
            ]
        }

        priority = visual_analysis.calculate_priority_table(prepared, ai_insights=insights, top_n=5)
        experience = priority[priority["问题类型"] == "体验类"].iloc[0]

        self.assertGreaterEqual(experience["AI严重度"], 100)
        self.assertEqual(experience["AI建议"], "降低信息流广告密度，优先检查推荐页体验。")
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
python -m unittest tests.test_visual_analysis -v
```

Expected:

```text
AttributeError: module 'visual_analysis' has no attribute 'calculate_priority_table'
```

- [ ] **Step 3: 添加优先级评分实现**

Append to `visual_analysis.py`:

```python
AI_SEVERITY_SCORES = {
    "high": 100.0,
    "medium": 60.0,
    "low": 25.0,
}

PRIORITY_WEIGHTS = {
    "volume": 0.35,
    "sentiment": 0.25,
    "rating": 0.20,
    "ai": 0.15,
    "risk": 0.05,
}


def severity_label(score):
    if score >= 75:
        return "high"
    if score >= 45:
        return "medium"
    return "low"


def build_ai_category_map(ai_insights):
    category_map = {}
    if not ai_insights:
        return category_map

    pain_points = ai_insights.get("pain_points", [])
    if not isinstance(pain_points, list):
        return category_map

    for item in pain_points:
        if not isinstance(item, dict):
            continue

        text = " ".join(
            str(item.get(key, ""))
            for key in ("name", "explanation", "suggestion")
            if item.get(key)
        )
        category = classify_issue(text)
        severity = AI_SEVERITY_SCORES.get(str(item.get("severity", "")).lower(), 50.0)
        suggestion = str(item.get("suggestion") or "")

        current = category_map.get(category)
        if current is None or severity > current["severity_score"]:
            category_map[category] = {
                "severity_score": severity,
                "suggestion": suggestion,
            }

    return category_map


def representative_review(group):
    if group.empty:
        return ""

    sorted_group = group.sort_values([RATING_COLUMN, SENTIMENT_COLUMN], ascending=[True, True])
    return str(sorted_group.iloc[0][CONTENT_COLUMN])


def calculate_priority_table(df, ai_insights=None, top_n=5):
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                CATEGORY_COLUMN,
                "优先级分数",
                "严重程度",
                "相关评论数",
                "平均评分",
                "平均情绪指数",
                "AI严重度",
                "AI建议",
                "代表评论",
            ]
        )

    ai_category_map = build_ai_category_map(ai_insights)
    total_negative = max(int((df[RATING_COLUMN] <= NEGATIVE_RATING_MAX).sum()), 1)
    rows = []

    for category, group in df.groupby(CATEGORY_COLUMN):
        negative_count = int((group[RATING_COLUMN] <= NEGATIVE_RATING_MAX).sum())
        high_risk_count = int(group[RISK_LABEL_COLUMN].apply(is_high_risk_label).sum())
        average_rating = float(group[RATING_COLUMN].mean())
        average_sentiment = float(group[SENTIMENT_COLUMN].mean())

        volume_score = min(negative_count / total_negative, 1.0) * 100
        sentiment_score = max(0.0, min(100.0, 100 - average_sentiment))
        rating_score = max(0.0, min(100.0, (5 - average_rating) / 4 * 100))
        ai_data = ai_category_map.get(category, {"severity_score": 50.0, "suggestion": ""})
        ai_score = ai_data["severity_score"]
        risk_score = high_risk_count / max(len(group), 1) * 100

        priority_score = (
            volume_score * PRIORITY_WEIGHTS["volume"]
            + sentiment_score * PRIORITY_WEIGHTS["sentiment"]
            + rating_score * PRIORITY_WEIGHTS["rating"]
            + ai_score * PRIORITY_WEIGHTS["ai"]
            + risk_score * PRIORITY_WEIGHTS["risk"]
        )

        rows.append(
            {
                CATEGORY_COLUMN: category,
                "优先级分数": round(priority_score, 2),
                "严重程度": severity_label(priority_score),
                "相关评论数": int(len(group)),
                "平均评分": round(average_rating, 2),
                "平均情绪指数": round(average_sentiment, 2),
                "AI严重度": round(ai_score, 2),
                "AI建议": ai_data["suggestion"],
                "代表评论": representative_review(group),
            }
        )

    priority = pd.DataFrame(rows)
    if priority.empty:
        return priority

    return priority.sort_values("优先级分数", ascending=False).head(top_n).reset_index(drop=True)
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```powershell
python -m unittest tests.test_visual_analysis -v
```

Expected:

```text
Ran 6 tests

OK
```

- [ ] **Step 5: 运行语法检查**

Run:

```powershell
python -m compileall visual_analysis.py tests
```

Expected: command exits with code 0.

- [ ] **Step 6: 记录提交限制**

Run:

```powershell
Test-Path .git
```

Expected:

```text
False
```

Commit is skipped because this workspace is not a git repository.

---

### Task 4: 测试并实现图表数据辅助函数

**Files:**
- Modify: `tests/test_visual_analysis.py`
- Modify: `visual_analysis.py`

- [ ] **Step 1: 追加失败测试**

Append these methods inside `VisualAnalysisTests` in `tests/test_visual_analysis.py`:

```python
    def test_rating_and_sentiment_distribution_are_chart_ready(self):
        prepared = visual_analysis.prepare_dashboard_data(self.make_reviews())

        rating = visual_analysis.rating_distribution(prepared)
        sentiment = visual_analysis.sentiment_distribution(prepared)

        self.assertEqual(list(rating.columns), ["评分", "评论数"])
        self.assertEqual(int(rating["评论数"].sum()), 6)
        self.assertEqual(list(sentiment.columns), ["情绪区间", "评论数"])
        self.assertEqual(int(sentiment["评论数"].sum()), 6)

    def test_sentiment_scatter_data_keeps_required_columns(self):
        prepared = visual_analysis.prepare_dashboard_data(self.make_reviews())

        scatter = visual_analysis.sentiment_scatter_data(prepared)

        self.assertEqual(
            list(scatter.columns),
            ["评分", "情绪指数", "问题类型", "风险标签", "内容"],
        )

    def test_extract_keyword_scores_returns_top_words(self):
        prepared = visual_analysis.prepare_dashboard_data(self.make_reviews())
        prepared["分词内容"] = [
            "封号 申诉 客服",
            "账号 封号 客服",
            "审核 违规 笔记",
            "广告 推荐 质量",
            "内容 种草 体验",
            "广告 推荐",
        ]

        keywords = visual_analysis.extract_keyword_scores(prepared["分词内容"], top_n=3)

        self.assertEqual(list(keywords.columns), ["关键词", "权重"])
        self.assertEqual(len(keywords), 3)

    def test_sentiment_trend_skips_when_time_column_missing(self):
        prepared = visual_analysis.prepare_dashboard_data(self.make_reviews())

        trend = visual_analysis.sentiment_trend(prepared)

        self.assertTrue(trend.empty)
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
python -m unittest tests.test_visual_analysis -v
```

Expected:

```text
AttributeError
```

The first missing helper should be `rating_distribution`.

- [ ] **Step 3: 添加图表数据辅助函数**

Append to `visual_analysis.py`:

```python
from sklearn.feature_extraction.text import TfidfVectorizer


def rating_distribution(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=[RATING_COLUMN, "评论数"])

    distribution = (
        df[RATING_COLUMN]
        .round()
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis(RATING_COLUMN)
        .reset_index(name="评论数")
    )
    return distribution


def sentiment_distribution(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["情绪区间", "评论数"])

    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-20", "21-40", "41-60", "61-80", "81-100"]
    bucket = pd.cut(
        df[SENTIMENT_COLUMN],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    distribution = bucket.value_counts(sort=False).rename_axis("情绪区间").reset_index(name="评论数")
    distribution["情绪区间"] = distribution["情绪区间"].astype(str)
    return distribution


def sentiment_scatter_data(df):
    columns = [RATING_COLUMN, SENTIMENT_COLUMN, CATEGORY_COLUMN, RISK_LABEL_COLUMN, CONTENT_COLUMN]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    return df[columns].copy()


def extract_keyword_scores(text_series, top_n=15):
    if text_series is None or text_series.empty:
        return pd.DataFrame(columns=["关键词", "权重"])

    clean_series = text_series.fillna("").astype(str)
    clean_series = clean_series[clean_series.str.strip() != ""]
    if clean_series.empty:
        return pd.DataFrame(columns=["关键词", "权重"])

    vectorizer = TfidfVectorizer(max_features=500)
    tfidf_matrix = vectorizer.fit_transform(clean_series)
    words = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1

    keyword_df = (
        pd.DataFrame({"关键词": words, "权重": scores})
        .sort_values("权重", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    keyword_df["权重"] = keyword_df["权重"].round(4)
    return keyword_df


def sentiment_trend(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["日期", "平均情绪指数", "平均评分", "评论数"])

    time_column = None
    for candidate in ["时间", "日期", "评论时间", "发布时间"]:
        if candidate in df.columns:
            time_column = candidate
            break

    if time_column is None:
        return pd.DataFrame(columns=["日期", "平均情绪指数", "平均评分", "评论数"])

    trend_df = df.copy()
    trend_df["日期"] = pd.to_datetime(trend_df[time_column], errors="coerce").dt.date
    trend_df = trend_df.dropna(subset=["日期"])
    if trend_df.empty:
        return pd.DataFrame(columns=["日期", "平均情绪指数", "平均评分", "评论数"])

    return (
        trend_df.groupby("日期")
        .agg(
            平均情绪指数=(SENTIMENT_COLUMN, "mean"),
            平均评分=(RATING_COLUMN, "mean"),
            评论数=(CONTENT_COLUMN, "count"),
        )
        .reset_index()
        .round({"平均情绪指数": 2, "平均评分": 2})
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```powershell
python -m unittest tests.test_visual_analysis -v
```

Expected:

```text
Ran 10 tests

OK
```

- [ ] **Step 5: 运行全部现有单元测试**

Run:

```powershell
python -m unittest discover -v
```

Expected:

```text
OK
```

The exact test count includes the existing `tests.test_ai_analysis` tests and the new `tests.test_visual_analysis` tests.

- [ ] **Step 6: 记录提交限制**

Run:

```powershell
Test-Path .git
```

Expected:

```text
False
```

Commit is skipped because this workspace is not a git repository.

---

### Task 5: 把产品经理看板接入 Streamlit 页面

**Files:**
- Modify: `app.py`
- Test: `visual_analysis.py`, `tests/test_visual_analysis.py`

- [ ] **Step 1: 修改导入**

In `app.py`, replace the import block:

```python
import streamlit as st
import pandas as pd
import re
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from snownlp import SnowNLP
from ai_analysis import AiAnalysisError, analyze_reviews, load_ai_config
```

with:

```python
import streamlit as st
import pandas as pd
import re
import jieba.posseg as pseg
from snownlp import SnowNLP
from ai_analysis import AiAnalysisError, analyze_reviews, load_ai_config
from visual_analysis import (
    CATEGORY_COLUMN,
    CONTENT_COLUMN,
    RATING_COLUMN,
    RISK_LABEL_COLUMN,
    SENTIMENT_COLUMN,
    TOKEN_COLUMN,
    calculate_health_metrics,
    calculate_priority_table,
    extract_keyword_scores,
    filter_reviews,
    prepare_dashboard_data,
    rating_distribution,
    sentiment_distribution,
    sentiment_scatter_data,
    sentiment_trend,
)
```

The existing `extract_keywords` helper in `app.py` can stay during this task, but the new dashboard should use `extract_keyword_scores()`.

- [ ] **Step 2: 在数据处理后创建看板数据**

Find this block in `app.py`:

```python
    with st.spinner('AI 引擎正在进行自然语言处理与情感计算，请稍候...'):
        df = process_data(raw_df)
        
    st.success("数据处理完成！")
```

Replace it with:

```python
    with st.spinner('AI 引擎正在进行自然语言处理与情感计算，请稍候...'):
        df = process_data(raw_df)

    try:
        dashboard_df = prepare_dashboard_data(df)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.success("数据处理完成！")
```

- [ ] **Step 3: 在侧边栏增加产品看板筛选器**

Add this block immediately after `dashboard_df = prepare_dashboard_data(df)` and before `st.success("数据处理完成！")`:

```python
    with st.sidebar:
        st.header("🧭 产品看板筛选")
        rating_range = st.slider("评分范围", 1, 5, (1, 5))
        sentiment_range = st.slider("情绪指数范围", 0, 100, (0, 100))
        category_options = sorted(dashboard_df[CATEGORY_COLUMN].dropna().unique().tolist())
        selected_categories = st.multiselect("问题类型", category_options, default=category_options)
        keyword_query = st.text_input("关键词搜索", placeholder="例如：封号、广告、客服")
        high_risk_only = st.checkbox("只看高风险评论")

    filtered_df = filter_reviews(
        dashboard_df,
        rating_range=rating_range,
        sentiment_range=sentiment_range,
        categories=selected_categories,
        keyword=keyword_query,
        high_risk_only=high_risk_only,
    )
```

- [ ] **Step 4: 用产品健康概览替换顶部 KPI 数据源**

Find the current KPI block:

```python
    st.header("📊 全局健康度看板")
    col1, col2, col3 = st.columns(3)
    col1.metric("总评论数", len(df))
    col2.metric("平均星级", round(df['评分'].mean(), 2))
    
    avg_sentiment = df['情绪指数'].mean()
    sentiment_delta = "健康" if avg_sentiment > 60 else "需警惕" if avg_sentiment > 40 else "极其危险"
    col3.metric("大盘情绪指数", f"{round(avg_sentiment, 1)} 分", sentiment_delta, delta_color="normal" if avg_sentiment > 50 else "inverse")
```

Replace it with:

```python
    if filtered_df.empty:
        st.warning("当前筛选条件下没有评论，请放宽筛选条件。")
        st.stop()

    metrics = calculate_health_metrics(filtered_df)

    st.header("📊 产品健康概览")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("评论数", metrics["total_reviews"])
    col2.metric("平均星级", metrics["average_rating"])
    col3.metric("差评占比", f"{metrics['negative_ratio']}%")
    sentiment_delta = "健康" if metrics["average_sentiment"] > 60 else "需警惕" if metrics["average_sentiment"] > 40 else "高风险"
    col4.metric("平均情绪", f"{metrics['average_sentiment']} 分", sentiment_delta)
    col5.metric("高风险评论", metrics["high_risk_count"])
```

- [ ] **Step 5: 新增问题优先级区域**

Add this block after the health overview and before the existing tab section:

```python
    st.subheader("🎯 问题优先级 Top 5")
    current_insights = st.session_state.get("ai_insights")
    priority_df = calculate_priority_table(filtered_df, ai_insights=current_insights, top_n=5)

    if priority_df.empty:
        st.info("当前筛选范围内没有足够数据生成问题优先级。")
    else:
        priority_chart = priority_df[[CATEGORY_COLUMN, "优先级分数"]].set_index(CATEGORY_COLUMN)
        st.bar_chart(priority_chart)
        st.dataframe(priority_df, use_container_width=True, hide_index=True)
```

- [ ] **Step 6: 新增评分与情绪诊断图表**

Add this block after the priority section:

```python
    st.subheader("🔎 评分与情绪诊断")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        rating_chart = rating_distribution(filtered_df)
        if rating_chart.empty:
            st.info("暂无评分分布数据。")
        else:
            st.bar_chart(rating_chart.set_index(RATING_COLUMN))

    with chart_col2:
        sentiment_chart = sentiment_distribution(filtered_df)
        if sentiment_chart.empty:
            st.info("暂无情绪分布数据。")
        else:
            st.bar_chart(sentiment_chart.set_index("情绪区间"))

    scatter_df = sentiment_scatter_data(filtered_df)
    if scatter_df.empty:
        st.info("暂无评分与情绪散点数据。")
    else:
        st.scatter_chart(scatter_df, x=RATING_COLUMN, y=SENTIMENT_COLUMN, color=CATEGORY_COLUMN)
```

- [ ] **Step 7: 新增关键词对比、趋势和可行动评论池**

Add this block after the diagnostics section:

```python
    st.subheader("🔤 关键词对比")
    keyword_col1, keyword_col2 = st.columns(2)
    negative_keywords = extract_keyword_scores(
        filtered_df[filtered_df[RATING_COLUMN] <= 3].get(TOKEN_COLUMN, pd.Series(dtype=str)),
        top_n=15,
    )
    positive_keywords = extract_keyword_scores(
        filtered_df[filtered_df[RATING_COLUMN] >= 4].get(TOKEN_COLUMN, pd.Series(dtype=str)),
        top_n=15,
    )

    with keyword_col1:
        st.markdown("**差评关键词**")
        if negative_keywords.empty:
            st.info("当前筛选范围内没有差评关键词。")
        else:
            st.bar_chart(negative_keywords.set_index("关键词"))

    with keyword_col2:
        st.markdown("**好评关键词**")
        if positive_keywords.empty:
            st.info("当前筛选范围内没有好评关键词。")
        else:
            st.bar_chart(positive_keywords.set_index("关键词"))

    trend_df = sentiment_trend(filtered_df)
    if not trend_df.empty:
        st.subheader("📈 舆情趋势")
        st.line_chart(trend_df.set_index("日期")[["平均情绪指数", "平均评分"]])

    st.subheader("🧾 可行动评论池")
    display_columns = [
        RATING_COLUMN,
        SENTIMENT_COLUMN,
        CATEGORY_COLUMN,
        RISK_LABEL_COLUMN,
        CONTENT_COLUMN,
    ]
    st.dataframe(filtered_df[display_columns], use_container_width=True, hide_index=True)
```

- [ ] **Step 8: 运行语法检查**

Run:

```powershell
python -m compileall app.py visual_analysis.py tests
```

Expected: command exits with code 0.

- [ ] **Step 9: 运行单元测试**

Run:

```powershell
python -m unittest discover -v
```

Expected:

```text
OK
```

- [ ] **Step 10: 记录提交限制**

Run:

```powershell
Test-Path .git
```

Expected:

```text
False
```

Commit is skipped because this workspace is not a git repository.

---

### Task 6: 调整 AI 区域与最终验证

**Files:**
- Modify: `app.py`
- Verify: `visual_analysis.py`, `tests/test_visual_analysis.py`, `tests/test_ai_analysis.py`

- [ ] **Step 1: 保证 AI 结果会刷新优先级**

Find the AI button block in `app.py`:

```python
    if st.button("生成 AI 舆情洞察", type="primary", disabled=not bool(config["api_key"])):
        try:
            with st.spinner("DeepSeek 正在阅读评论并生成结构化洞察..."):
                st.session_state["ai_insights"] = analyze_reviews(df)
        except AiAnalysisError as exc:
            st.error(str(exc))
```

Replace it with:

```python
    if st.button("生成 AI 舆情洞察", type="primary", disabled=not bool(config["api_key"])):
        try:
            with st.spinner("DeepSeek 正在阅读评论并生成结构化洞察..."):
                st.session_state["ai_insights"] = analyze_reviews(filtered_df)
                st.rerun()
        except AiAnalysisError as exc:
            st.error(str(exc))
```

This makes the priority section recompute with AI severity after the AI response is stored.

- [ ] **Step 2: 确认 AI 区域读取当前状态**

Keep this existing pattern in `app.py`:

```python
    insights = st.session_state.get("ai_insights")
    if insights:
        st.subheader("舆情总览")
        st.write(insights.get("summary", "暂无 AI 总览。"))
```

If the AI section is currently below the new dashboard, leave it there. The priority table already reads `st.session_state["ai_insights"]`, so AI 严重度会体现在 Top 5 里。

- [ ] **Step 3: 运行全部测试**

Run:

```powershell
python -m unittest discover -v
```

Expected:

```text
OK
```

- [ ] **Step 4: 运行语法检查**

Run:

```powershell
python -m compileall app.py ai_analysis.py visual_analysis.py nlp_analysis.py spider.py tests
```

Expected: command exits with code 0.

- [ ] **Step 5: 启动 Streamlit 手动检查**

Run:

```powershell
streamlit run app.py --server.port 8503
```

Expected:

```text
Local URL: http://localhost:8503
```

Manual checks:

- 上传 `xiaohongshu_reviews.csv`。
- 页面显示产品健康概览。
- 问题优先级 Top 5 有柱状图和表格。
- 评分分布、情绪分布、评分 × 情绪散点图正常显示。
- 关键词对比区能显示差评关键词和好评关键词。
- 可行动评论池能跟随筛选条件变化。
- 未配置 `DEEPSEEK_API_KEY` 时，AI 按钮禁用或显示友好提示，本地看板仍可用。

- [ ] **Step 6: 停止 Streamlit**

In the terminal running Streamlit, press:

```text
Ctrl+C
```

Expected:

```text
Streamlit server stops and returns to the PowerShell prompt.
```

- [ ] **Step 7: 记录提交限制**

Run:

```powershell
Test-Path .git
```

Expected:

```text
False
```

Commit is skipped because this workspace is not a git repository.

---

## 最终验收标准

- `python -m unittest discover -v` 通过。
- `python -m compileall app.py ai_analysis.py visual_analysis.py nlp_analysis.py spider.py tests` 通过。
- `streamlit run app.py --server.port 8503` 能启动。
- 上传 `xiaohongshu_reviews.csv` 后，本地产品经理看板可用。
- 不配置 DeepSeek API Key 时，本地可视化仍然可用。
- 配置 DeepSeek API Key 并生成 AI 洞察后，问题优先级能融合 AI 严重度。
- 当前目录不是 git 仓库，因此没有 commit。
