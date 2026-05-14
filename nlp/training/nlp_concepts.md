# NLP Concepts — Simple Explanation Guide
### News Headline Sentiment Dashboard Project

---

![News Sentiment Pipeline](news_sentiment_pipeline_flowchart.svg)

## 1. Basic NLP (Natural Language Processing)

**What it is in plain English:**
NLP is teaching a computer to read and understand human language — the same way you understand a sentence, a computer can be programmed to do it too, just in a more structured, rule-based way.

**Key ideas:**
- Computers cannot understand raw text directly. We break text into smaller, manageable pieces first.
- We remove noise (punctuation, numbers, junk words) so only meaningful words remain.
- We count, compare, and analyse those meaningful words.

**In our project:**
We take a raw news headline like `"Apple Inc. hits record high! Stock up 12% Tuesday."` and transform it into a clean list of meaningful words: `["apple", "hits", "record", "high", "stock", "tuesday"]`.

**Core NLP sub-steps:**

| Step | What it does | Example |
|---|---|---|
| Tokenization | Splits text into individual words | `"good morning"` → `["good", "morning"]` |
| Stopword removal | Removes common words with no meaning | removes "the", "is", "at", "a" |
| Frequency analysis | Counts how often each word appears | `"economy"` appeared 45 times |
| POS Tagging | Labels each word as noun/verb/adjective | `"rising"` → verb, `"market"` → noun |

---

## 2. Sentiment Analysis with Classical NLP

**What it is in plain English:**
Sentiment analysis is figuring out the *emotion* or *opinion* hidden inside a piece of text — is it positive, negative, or neutral? Classical means we use rules and word lists (not deep learning/AI models).

**How it works:**
- A sentiment **lexicon** (dictionary) assigns a score to thousands of words.
  - "excellent" → +0.9, "crash" → −0.8, "the" → 0.0
- We add up scores for all words in a sentence to get an overall tone.

**Two tools we use:**

**VADER** (Valence Aware Dictionary and sEntiment Reasoner)
- Designed specifically for short social-media style text and headlines.
- Understands things like CAPS ("GREAT" is stronger than "great") and punctuation ("great!!!" scores higher than "great").
- Returns a `compound` score from -1.0 (most negative) to +1.0 (most positive).

**TextBlob**
- Returns `polarity` (-1 to +1) and `subjectivity` (0 = factual, 1 = opinion-based).
- Good for slightly longer, more formal text.

**In our project:**
`"Markets crash as inflation fears grow"` → VADER compound: −0.72 → **Negative**
`"Scientists discover breakthrough cancer treatment"` → VADER compound: +0.81 → **Positive**

---

## 3. Regex (Regular Expressions) for Text Processing

**What it is in plain English:**
Regex is a special search language that lets you find and extract patterns in text — like a super-powered Ctrl+F. Instead of searching for one exact word, you describe a *pattern* and it finds everything that matches.

**Why we need it in NLP:**
Raw news text is messy — it has URLs, HTML tags, numbers, special characters, and formatting junk. We use regex to clean it before any NLP can happen.

**Common patterns used in our project:**

| Pattern | Meaning | Finds |
|---|---|---|
| `https?://\S+` | Any URL | `https://bbc.com/news/...` |
| `<[^>]+>` | Any HTML tag | `<p>`, `<div class="x">` |
| `[^a-zA-Z\s]` | Anything NOT a letter or space | Numbers, punctuation, symbols |
| `\b[A-Z]{2,5}\b` | 2–5 uppercase letters | `NASA`, `BBC`, `AAPL` (tickers) |
| `\b[A-Z][a-z]+ [A-Z][a-z]+\b` | Two capitalised words together | `Elon Musk`, `New York` |

**In our project:**
Input: `"Apple Inc. (AAPL) shares up 3% — see full story at https://t.co/abc123 <br/>"`
After regex cleaning: `"apple inc shares up see full story"`

---

## 4. Calling REST APIs and Handling JSON

**What is a REST API in plain English:**
An API (Application Programming Interface) is a way for two programs to talk to each other over the internet. A REST API works like a menu at a restaurant — you place an order (send a request), the kitchen processes it (server handles it), and you get your food (receive a response).

**How it works step by step:**
1. You send an HTTP GET request to a URL (the endpoint).
2. You include parameters (filters) like category, language, date.
3. The server returns data in **JSON format**.
4. You parse the JSON into Python dictionaries and lists.

**What is JSON:**
JSON (JavaScript Object Notation) is a text format for storing structured data. Python reads it as a dictionary.

```json
{
  "title": "Oil prices fall sharply",
  "source": { "name": "Reuters" },
  "publishedAt": "2025-03-30T08:00:00Z",
  "description": "Crude oil dropped 4% amid demand concerns."
}
```

**NewsAPI endpoints we use:**

| Endpoint | Purpose | Key parameters |
|---|---|---|
| `/v2/top-headlines` | Breaking news by category | `category`, `country`, `pageSize` |
| `/v2/everything` | Search all articles by keyword | `q` (query), `language`, `sortBy` |

**In our project:**
We send a request → get back up to 20 headlines per category → parse JSON → store in a Pandas DataFrame for analysis.

---

## 5. Web Scraping with BeautifulSoup / Requests

**What it is in plain English:**
Web scraping is automatically reading a webpage and extracting specific information from it — the same way you manually read and copy text from a website, but done by a program at scale.

**Two libraries we use:**

**`requests`** — fetches the raw HTML of a webpage (like your browser does, but without rendering it visually).

**`BeautifulSoup`** — parses that raw HTML and lets you navigate the structure to find specific elements (headings, paragraphs, links, metadata).

**How a webpage is structured (HTML):**
```html
<article>
  <h1>Markets fall on recession fears</h1>
  <p>Stocks dropped sharply on Thursday...</p>
  <p>Analysts warn that the trend may continue...</p>
</article>
```
BeautifulSoup lets us say: *"find the `<article>` tag, then get all `<p>` tags inside it"* — and we get the full article text.

**Why we need it here:**
NewsAPI only returns the first ~200 characters of article content. To get the full text, we scrape the actual article page.

**Responsible scraping rules we follow:**
- Send a `User-Agent` header so the server knows who we are.
- Add `time.sleep(1)` between requests — don't hammer the server.
- Handle errors gracefully — some sites block scraping, that's fine, we skip them.

---

## 6. Collecting and Analyzing Web Data for NLP Tasks

**What it is in plain English:**
This is the full pipeline — combining all the above steps into one coherent workflow that goes from *raw internet data* to *meaningful insights*. In real NLP projects, this is called building a **data collection and analysis pipeline**.

**The full pipeline in our project:**

```
Internet (NewsAPI + Web pages)
        ↓
  Collect raw data (REST API + Web Scraping)
        ↓
  Clean & structure (Regex + Pandas)
        ↓
  Process language (Tokenize + Stopwords + Freq. analysis)
        ↓
  Analyse sentiment (VADER + TextBlob)
        ↓
  Visualise insights (Matplotlib + Plotly + WordCloud)
        ↓
  Export results (CSV + Dashboard)
```

**Why this matters:**
Every real-world NLP application — spam filters, review analysers, chatbots, news monitors — follows this exact same pattern. The specific tools change, but the pipeline structure stays the same. Mastering this pipeline gives you the foundation to build any text analytics system.

**Key insight about web data for NLP:**
Web data is always noisy (HTML tags, ads, broken encoding, truncated text). The cleaning and structuring steps (regex, stopwords) are not optional — they are what make the difference between getting junk output and getting useful insights.

---

## Quick Reference: Concept → Tool → Purpose

| Concept | Python Tool | What it does in our project |
|---|---|---|
| Basic NLP | `nltk` | Tokenize headlines, remove stopwords, count word frequencies |
| Sentiment Analysis | `vaderSentiment`, `textblob` | Score each headline as Positive / Neutral / Negative |
| Regex | `re` (built-in) | Clean raw text, extract entities, remove URLs and HTML |
| REST API + JSON | `requests` | Fetch 100+ live headlines from NewsAPI |
| Web Scraping | `requests` + `beautifulsoup4` | Extract full article text from news websites |
| Web Data for NLP | `pandas`, `matplotlib`, `plotly` | Store, analyse, and visualise the complete dataset |

---

Why Two Score Types — Polarity and Subjectivity?

They measure completely different things.

Polarity answers: "Is this positive or negative?"
Subjectivity answers: "Is this a fact or an opinion?"

Think of them as two separate axes:

Headline	Polarity	Subjectivity	Meaning
"GDP grew 4.2% last quarter"	Neutral (0.0)	Objective (0.1)	Pure fact, no emotion
"Scientists confirm vaccine works"	Positive (+0.6)	Objective (0.2)	Good news, stated as fact
"This government is absolutely terrible"	Negative (−0.8)	Subjective (0.9)	Strong negative opinion
"What a brilliant and inspiring speech!"	Positive (+0.9)	Subjective (0.95)	Strong positive opinion
Why Polarity Alone Is Not Enough

Consider these two headlines:

"Unemployment rose to 8%"
Polarity: slightly negative
Subjectivity: 0.05 (factual)
"Unemployment is a devastating crisis destroying families"
Polarity: negative
Subjectivity: 0.85 (opinion)

Both are negative, but very different:

First → journalism (fact-based)
Second → opinion (emotion-driven)

👉 Subjectivity helps determine how much to trust the sentiment.

High subjectivity → possible bias/opinion
Low subjectivity → likely real-world event

This is why dashboards often plot both (e.g., bubble charts) to distinguish:

Opinionated sources
Fact-based reporting
Why Two Libraries — VADER and TextBlob?

They are designed for different use cases.

Using both improves reliability instead of relying on one.

VADER (Best for Short, Informal Text)
Designed for social media / headlines
Understands:
"GREAT" > "great"
"great!!!" > "great"
Negations → "not bad" = positive
Handles slang & emoticons
Outputs:
pos, neg, neu, compound
Fast and no training needed
TextBlob (Best for Formal Text)
Designed for general English prose
Uses adjective/adverb lexicon
Outputs:
Polarity
Subjectivity (important advantage)
Better for:
Long sentences
Descriptive language
Key Differences in Practice
Example 1

"Markets not doing terribly today"

VADER → +0.34 ✅ (understands "not terribly")
TextBlob → -0.35 ❌ (misreads "terribly")
Example 2

"A truly magnificent and groundbreaking scientific achievement"

VADER → +0.72 (good)
TextBlob → +0.87 ✅ (better with rich adjectives)
Why Use Both?

Neither is always correct.

👉 Best approach:

If both agree → trust the result
If they disagree → mark as Neutral

This improves reliability significantly.

Summary
Use VADER → for short, informal, headline-style text
Use TextBlob → for subjectivity & formal language
Use both together → for more accurate, trustworthy results


*This guide is your reference sheet. Keep it open while going through the notebook code.*
