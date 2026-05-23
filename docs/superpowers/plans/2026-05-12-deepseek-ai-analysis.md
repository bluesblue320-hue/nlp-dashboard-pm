# DeepSeek AI Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight configurable DeepSeek AI analysis layer to the existing Streamlit review dashboard.

**Architecture:** Keep local NLP statistics in `app.py`, add `ai_analysis.py` for configuration, review packet preparation, prompt construction, DeepSeek HTTP calls, and JSON parsing. Use `tests/test_ai_analysis.py` to cover the new non-UI behavior before implementation.

**Tech Stack:** Python, Streamlit, pandas, requests, unittest, DeepSeek OpenAI-compatible chat completions API.

---

## File Structure

- Create: `ai_analysis.py`
  - Owns DeepSeek configuration, `.env` loading, representative review selection, prompt construction, API call, and response normalization.
- Create: `tests/test_ai_analysis.py`
  - Uses Python `unittest` and fake HTTP responders to verify behavior without calling the network.
- Modify: `app.py`
  - Imports `ai_analysis`, validates required CSV columns, adds AI insight button, and renders returned sections.
- Modify: `requirenments.txt`
  - Add `requests` explicitly because both the spider and AI analysis module call HTTP APIs directly.

Git commits are skipped because `C:\Users\blues\Desktop\xhs` is not a git repository.

---

### Task 1: Test AI Configuration And Parsing

**Files:**
- Create: `tests/test_ai_analysis.py`
- Create later: `ai_analysis.py`

- [ ] **Step 1: Write the failing test**

```python
import json
import os
import unittest
from unittest.mock import patch

import pandas as pd

import ai_analysis


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload, ensure_ascii=False)

    def json(self):
        return self._payload


class AiAnalysisTests(unittest.TestCase):
    def test_load_config_defaults_to_deepseek_flash(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=True):
            config = ai_analysis.load_ai_config()

        self.assertEqual(config["provider"], "deepseek")
        self.assertEqual(config["model"], "deepseek-v4-flash")
        self.assertEqual(config["api_key"], "sk-test")

    def test_parse_json_content_tolerates_markdown_fences(self):
        parsed = ai_analysis.parse_json_content('```json\n{"summary": "稳定"}\n```')

        self.assertEqual(parsed["summary"], "稳定")

    def test_normalize_insights_fills_missing_sections(self):
        normalized = ai_analysis.normalize_insights({"summary": "整体偏负面"})

        self.assertEqual(normalized["summary"], "整体偏负面")
        self.assertEqual(normalized["pain_points"], [])
        self.assertEqual(normalized["recommendations"], [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_ai_analysis -v`

Expected: FAIL or ERROR because `ai_analysis` does not exist yet.

- [ ] **Step 3: Implement minimal configuration and parsing functions**

Create `ai_analysis.py` with `load_env_file`, `load_ai_config`, `parse_json_content`, `normalize_insights`, and the `AiAnalysisError` exception.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_ai_analysis -v`

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

Skip because the workspace is not a git repository.

---

### Task 2: Test Review Packet Preparation And API Request

**Files:**
- Modify: `tests/test_ai_analysis.py`
- Modify: `ai_analysis.py`

- [ ] **Step 1: Add failing tests**

Add these methods inside `AiAnalysisTests`:

```python
    def test_build_review_packet_keeps_representative_reviews(self):
        df = pd.DataFrame(
            [
                {"评分": 1, "内容": "总是无故封号，客服也找不到"},
                {"评分": 5, "内容": "内容很多，种草体验不错"},
                {"评分": 4, "内容": "高分但是广告太多"},
            ]
        )

        packet = ai_analysis.build_review_packet(df, max_reviews=2)

        self.assertEqual(packet["metrics"]["total_reviews"], 3)
        self.assertEqual(packet["metrics"]["average_rating"], 3.33)
        self.assertLessEqual(len(packet["reviews"]), 2)
        self.assertIn("评分", packet["reviews"][0])
        self.assertIn("内容", packet["reviews"][0])

    def test_analyze_reviews_posts_deepseek_payload_and_normalizes_response(self):
        df = pd.DataFrame([{"评分": 1, "内容": "无故封号，申诉没人理"}])
        payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "用户主要抱怨账号治理。",
                                "pain_points": [{"name": "账号封禁"}],
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

        calls = []

        def fake_post(url, headers, json, timeout):
            calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
            return FakeResponse(payload=payload)

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=True):
            result = ai_analysis.analyze_reviews(df, post_func=fake_post)

        self.assertEqual(result["summary"], "用户主要抱怨账号治理。")
        self.assertEqual(result["pain_points"][0]["name"], "账号封禁")
        self.assertEqual(calls[0]["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(calls[0]["json"]["model"], "deepseek-v4-flash")
        self.assertEqual(calls[0]["json"]["response_format"], {"type": "json_object"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_ai_analysis -v`

Expected: FAIL because `build_review_packet` and `analyze_reviews` are missing.

- [ ] **Step 3: Implement minimal packet and API behavior**

Add `build_review_packet`, `build_messages`, `call_deepseek`, and `analyze_reviews` to `ai_analysis.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_ai_analysis -v`

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

Skip because the workspace is not a git repository.

---

### Task 3: Integrate Streamlit AI Analysis

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Modify imports**

Add:

```python
from ai_analysis import AiAnalysisError, analyze_reviews, load_ai_config
```

- [ ] **Step 2: Validate columns before processing**

After reading the CSV, check:

```python
required_columns = {"评分", "内容"}
missing_columns = required_columns - set(raw_df.columns)
if missing_columns:
    st.error(f"CSV 缺少必要列：{', '.join(sorted(missing_columns))}")
    st.stop()
```

- [ ] **Step 3: Add AI insight section after existing tabs**

Add a divider, header, API key status, and a button:

```python
st.divider()
st.header("🧠 AI 舆情洞察")
config = load_ai_config()
if not config["api_key"]:
    st.warning("未检测到 DEEPSEEK_API_KEY。请在系统环境变量或 .env 中配置后再生成 AI 洞察。")

if st.button("生成 AI 舆情洞察", type="primary", disabled=not bool(config["api_key"])):
    try:
        with st.spinner("DeepSeek 正在阅读评论并生成结构化洞察..."):
            insights = analyze_reviews(df)
    except AiAnalysisError as exc:
        st.error(str(exc))
    else:
        st.subheader("舆情总览")
        st.write(insights["summary"])
```

- [ ] **Step 4: Render pain points, delighters, high-risk reviews, recommendations, and report copy**

Render returned lists with Streamlit expanders, tables, and text areas. Use `.get()` and defaults so missing AI fields do not crash the app.

- [ ] **Step 5: Run syntax verification**

Run: `python -m compileall app.py ai_analysis.py tests`

Expected: exit code 0.

- [ ] **Step 6: Commit**

Skip because the workspace is not a git repository.

---

### Task 4: Final Verification

**Files:**
- All changed files.

- [ ] **Step 1: Run unit tests**

Run: `python -m unittest tests.test_ai_analysis -v`

Expected: 5 tests pass.

- [ ] **Step 2: Run syntax verification**

Run: `python -m compileall app.py ai_analysis.py nlp_analysis.py spider.py tests`

Expected: exit code 0.

- [ ] **Step 3: Verify app import dependencies**

Run: `python -c "import ai_analysis; print(ai_analysis.load_ai_config()['model'])"`

Expected: `deepseek-v4-flash`.

- [ ] **Step 4: Report**

Summarize changed files, verification output, and note that live DeepSeek API calls were not run unless a real key was configured.
