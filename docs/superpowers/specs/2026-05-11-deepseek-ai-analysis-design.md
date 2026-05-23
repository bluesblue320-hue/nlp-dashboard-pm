# DeepSeek AI Analysis Design

## Context

The current project collects App Store reviews for Xiaohongshu and analyzes them in a Streamlit dashboard. The existing analysis relies on traditional NLP:

- `jieba` and part-of-speech filtering for segmentation.
- TF-IDF for positive and negative keyword extraction.
- SnowNLP for sentiment scoring.
- Streamlit charts and tables for the dashboard.

This is useful for quick statistics, but it struggles with short Chinese reviews, sarcasm, product-specific complaints, and business-level recommendations. The goal is to add an AI insight layer using DeepSeek while keeping the current NLP layer for fast local statistics.

## Goal

Build a lightweight configurable AI analysis engine that uses DeepSeek first, but does not hard-code the application around DeepSeek. The dashboard should turn uploaded review data into structured product insights instead of only showing keywords and sentiment scores.

## Non-Goals

- Do not replace all existing NLP logic in the first version.
- Do not build a complex multi-provider framework.
- Do not require a database or user account system.
- Do not send every raw review blindly when the dataset is large.
- Do not store API keys in source code.

## Proposed Architecture

Keep the current dashboard flow and add one focused module:

```text
CSV reviews
-> local NLP statistics
-> representative review selection
-> ai_analysis.py
-> DeepSeek API
-> structured JSON insights
-> Streamlit report sections
```

### Files

- `app.py`: Keeps the Streamlit UI, file upload, local metrics, and chart display. Adds controls for running AI analysis and rendering AI insight sections.
- `nlp_analysis.py`: Can remain as the local script for standalone NLP exploration. Shared local analysis logic may later be extracted if duplication becomes painful.
- `ai_analysis.py`: New module for AI provider configuration, prompt construction, DeepSeek request handling, JSON parsing, and graceful fallback.
- `.env`: Local environment file for secrets and model configuration.
- `requirenments.txt`: Add the small dependencies needed for environment loading and HTTP/API calls if not already present.

## Configuration

The first version should support this local configuration:

```text
DEEPSEEK_API_KEY=
AI_PROVIDER=deepseek
AI_MODEL=deepseek-v4-flash
```

`DEEPSEEK_API_KEY` must be set in the local shell or an uncommitted `.env` file. `AI_PROVIDER` defaults to `deepseek`. `AI_MODEL` defaults to `deepseek-v4-flash`, which is the lower-cost DeepSeek V4 model suitable for this analysis workflow. The app must show a clear Streamlit warning if `DEEPSEEK_API_KEY` is missing.

## AI Output Contract

The AI layer should return a Python dictionary shaped like this:

```json
{
  "summary": "overall public-opinion summary",
  "pain_points": [
    {
      "name": "issue category",
      "severity": "high | medium | low",
      "evidence": ["representative review quote"],
      "explanation": "why users are unhappy",
      "suggestion": "product or operations action"
    }
  ],
  "delighters": [
    {
      "name": "positive category",
      "evidence": ["representative review quote"],
      "explanation": "why users like it"
    }
  ],
  "sentiment_drivers": ["main reasons behind user sentiment"],
  "high_risk_reviews": [
    {
      "rating": 1,
      "content": "review text",
      "risk_reason": "why this review matters"
    }
  ],
  "recommendations": ["specific product or operations recommendations"],
  "report_copy": "a polished paragraph that can be copied into a report"
}
```

The UI should tolerate missing fields and display a fallback message rather than crashing.

## Data Selection

To control cost and latency, the first version should not send all comments by default. It should send a compact review packet:

- Basic metrics: total review count, average rating, rating distribution, local average sentiment if available.
- Top low-rating reviews, especially 1-2 star comments.
- High-rating reviews for positive drivers.
- Reviews with rating-sentiment mismatch if local sentiment is available.
- Local top keywords from negative and positive review pools.

Initial limit: about 80 to 120 representative reviews. If the dataset is smaller, send all valid reviews.

## Prompt Strategy

The model should be asked to act as a Chinese product analyst for App Store review mining. The prompt should require:

- Output in simplified Chinese.
- Return strict JSON only.
- Focus on product, community governance, account, customer service, content review, performance, and user experience issues.
- Separate facts from recommendations.
- Use representative comments as evidence.
- Avoid inventing statistics that were not provided.

## Error Handling

The AI module should handle:

- Missing API key.
- Network failure or non-200 API response.
- Model timeout.
- Invalid JSON response.
- Empty uploaded CSV.
- Missing required columns: `评分` and `内容`.

For user-facing errors, Streamlit should show short, actionable messages.

## Streamlit Experience

Add an AI analysis section after the existing local NLP dashboard:

- A button such as `生成 AI 舆情洞察`.
- A spinner while DeepSeek is running.
- A summary card or text block for `summary`.
- Expandable sections for pain points and delighters.
- A table for high-risk reviews.
- A recommendations list.
- A final copy-ready report paragraph.

The existing keyword charts and dataframes should stay, because they make the AI analysis more transparent.

## Testing And Verification

Minimum verification for the first implementation:

- Run the local analysis path on `xiaohongshu_reviews.csv`.
- Verify the app still starts with Streamlit.
- Verify missing API key produces a friendly warning.
- Unit-test or manually test JSON parsing fallback with malformed AI output.
- Verify the dashboard still works when only local NLP analysis is used.

## Open Constraint

This workspace is not currently a git repository, so the design document cannot be committed unless git is initialized or the project is moved into an existing repository.
