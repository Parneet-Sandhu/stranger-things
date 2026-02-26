# 📺 Stranger Things: Fan Theories, Sentiment & Narrative Loopholes

> **How does narrative ambiguity fuel fan theories, sentiment, and online discourse?**

An end-to-end Data Science + NLP project exploring how fans reacted to the *Stranger Things* finale — through theories, emotions, and debates across Reddit and beyond.

---

## 🧭 Table of Contents

- [Overview](#-overview)
- [Key Questions](#-key-questions)
- [Dataset](#-dataset)
- [Detected Fan Theory Themes](#-detected-fan-theory-themes)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Streamlit App](#-streamlit-app)
- [Getting Started](#-getting-started)
- [Limitations](#️-limitations)
- [Key Takeaway](#-key-takeaway)

---

## 🔭 Overview

Instead of asking *"Was the ending good or bad?"*, this project asks something more interesting:

This project blends **NLP, unsupervised learning, and sentiment analysis** to uncover:
- What fans were *actually* talking about
- Which narrative threads ignited the strongest emotional reactions
- How open endings drive engagement rather than kill it

It's built to feel **exploratory and story-driven**, not just a collection of charts.

---

## 🧠 Key Questions

- What fan theories dominated discussion after the finale?
- Which themes triggered the strongest emotional reactions?
- How polarized was audience sentiment across theory clusters?
- Do open endings increase engagement and online debate?
- Which narrative "loopholes" were discussed the most?

---

## 📂 Dataset

The dataset is a **combined and cleaned master CSV** built from multiple sources:

| Source | Description |
|--------|-------------|
| Reddit | Scraped fan discussions and public datasets |
| Kaggle | Curated Stranger Things datasets |
| BrowseAI | Social commentary extracted via web automation |

### Core Schema

| Column | Description |
|--------|-------------|
| `text` | Raw fan comment or discussion |
| `theme` | NLP-clustered theory category |
| `sentiment` | Positive / Neutral / Negative |
| `sentiment_score` | Sentiment intensity score (where available) |
| `source` | Reddit / Kaggle / BrowseAI |

**~700+ rows** across all sources. Themes are discovered automatically using unsupervised learning — no manual labeling.

---

## 🎭 Detected Fan Theory Themes

Themes emerge organically from fan language using TF-IDF + clustering. Examples include:

| Theme | Description |
|-------|-------------|
| 🧬 Eleven Survival Theories | Speculation around Eleven's fate and powers |
| 😡 Ending Dissatisfaction | Negative reactions to narrative closure |
| ✍️ Writing Criticism | Fans critiquing plot decisions and character arcs |
| 🌀 Upside Down Lore | Deep dives into the mythology of the alternate dimension |
| 💖 Nostalgia & Emotional Closure | Fans processing the emotional weight of the finale |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data wrangling | `pandas`, `numpy` |
| NLP & Clustering | `scikit-learn` (TF-IDF, KMeans) |
| Sentiment Analysis | `NLTK` (VADER) |
| Visualization | `matplotlib`, `seaborn`, `wordcloud` |
| Web App | `Streamlit` |
| Language | Python 3.9+ |

---

## 📁 Project Structure

```
stranger-things/
├── data/
│   └── browse_st.csv
    └── kaggle_all_discussions.csv       
├── stranger_things_sentiment.ipynb
├── stranger_things_final.csv     #final dataset
├── app.py                        # Streamlit application
├── requirements.txt
└── README.md
```

---

## 🎨 Streamlit App

The interactive app is designed for exploration, not just reporting.

### 🏠 Overview
- Project summary and dataset at a glance
- Theme distribution chart

### 🧠 Theory Explorer
- Select any detected theme
- Browse real fan comments behind each cluster
- Discover dominant keywords per theme

### 💬 Sentiment Analysis
- Overall audience sentiment breakdown
- Sentiment distribution per theme
- Identify which theories were most polarizing

### 🔥 Fun Insights
- Word clouds per theme
- Theme vs. sentiment heatmaps
- "Most emotional" fan discussions ranked by sentiment score

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Parneet-Sandhu/stranger-things.git
cd stranger-things

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## ⚠️ Limitations

- **Sampling bias** — social media data overrepresents vocal, engaged fans
- **Sarcasm & memes** — sentiment models may misclassify ironic or humorous content
- **Statistical themes** — clusters are inferred from language patterns, not confirmed canon
- **Data freshness** — dataset reflects discussions at a specific point in time

These are expected tradeoffs in real-world NLP on social data.

---

## 💡 Key Takeaway

**Ambiguity drives engagement.** Open endings don't kill fandoms — they ignite them.

This project demonstrates how data science can be applied to culture, storytelling, and collective human behavior — not just structured business data. If you can measure sentiment, you can measure meaning.

---

## 🤝 Contributing

Pull requests and dataset contributions are welcome. If you have additional Reddit threads, Kaggle datasets, or scraped fan forums to add, feel free to open an issue.

---

## 📄 License

MIT License. See `LICENSE` for details.
