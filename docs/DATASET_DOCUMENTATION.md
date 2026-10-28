# Dataset Documentation: AG News Classification Benchmark

## 1. Overview and Provenance
The **AG News** dataset is an established academic benchmark for natural language processing and text classification research. It was compiled by Xiang Zhang, Junbo Zhao, and Yann LeCun in 2015 for evaluating deep learning and linear text classification models.

- **Primary Academic Citation:**
  > Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. *Advances in Neural Information Processing Systems (NeurIPS 2015)*, 28, 649–657.
- **Corpus Origin:** Derived from Antonio Gulli's AG Corpus of news articles indexed from over 2,000 news sources between 2004 and 2005.
- **Upstream Repository & Distribution:** Distributed openly on Hugging Face Datasets (`ag_news`) and academic archives.
- **License / Terms:** Open academic research and educational use.

## 2. Category Taxonomy and Semantics
The AG News corpus defines four mutually exclusive, balanced news domains:

| Category ID | Class Label | Description and Typical Subjects |
| :---: | :---: | :--- |
| **0** | **World** | International diplomacy, treaties, United Nations resolutions, global conflicts, bilateral summits, and foreign governance. |
| **1** | **Sports** | Athletic tournaments, team sports (football, basketball, cricket, baseball), Olympic events, player transfers, and match scores. |
| **2** | **Business** | Stock market indices (Wall Street, FTSE, Nikkei), corporate earnings, mergers and acquisitions, central bank interest rates, and trade data. |
| **3** | **Sci/Tech** | Space exploration (NASA, ESA), semiconductor manufacturing, consumer electronics, internet software, artificial intelligence, and cybersecurity. |

## 3. Dataset Splits and Sampling

### Official Benchmark Dimensions
- **Full Training Split:** 120,000 articles (exactly 30,000 articles per category).
- **Official Held-Out Test Split:** 7,600 articles (exactly 1,900 articles per category).
- **Total Articles:** 127,600.

### Project Experimental Split (NewsLens AI)
To provide CPU-friendly training and reproduction within seconds on standard student hardware while ensuring strict statistical validity:
- **Training Subset:** 24,000 articles sampled using reproducible stratified random sampling (`stratify=label`, `random_state=42`), allocating exactly **6,000 articles per class**.
- **Held-Out Test Set:** Full 7,600 test articles from the official benchmark, ensuring results are directly comparable to published literature.

## 4. Schema and Field Definitions
Each record contains:
- `text` *(string)*: Concatenation of the news headline (title) and article summary description.
- `label` *(int64)*: Target category index ranging from 0 to 3.

## 5. Exploratory Data Quality Audit

| Metric | Training Sample (N = 24,000) | Held-Out Test Set (N = 7,600) |
| :--- | :---: | :---: |
| **Missing / Null Texts** | 0 (0.00%) | 0 (0.00%) |
| **Missing / Null Labels** | 0 (0.00%) | 0 (0.00%) |
| **Duplicate Articles** | 0 (0.00%) | 0 (0.00%) |
| **Min Character Length** | 100 characters | 100 characters |
| **Mean Character Length** | 236.48 characters | 235.30 characters |
| **Max Character Length** | 1,012 characters | 892 characters |
| **Min Word Count** | 17 words | 16 words |
| **Mean Word Count** | 38.6 words | 38.4 words |
| **Max Word Count** | 175 words | 158 words |

## 6. Preprocessing Observations & Text Artifacts
Raw news articles extracted from wire syndication feeds (Reuters, AP, PR Newswire) exhibit specific syntactic artifacts that must be handled transparently:
1. **HTML Character Entities:** Widespread occurrences of `&amp;`, `&quot;`, `&lt;`, `&gt;`, and numeric references.
2. **Control Characters & Escapes:** Occurrences of backslash-escaped characters such as `\b`, `\n`, `\t`, and carriage returns.
3. **Syndication Attribution Headers:** Datelines such as `(Reuters)`, `(AP)`, `NEW YORK -`, and trailing URLs.
4. **Acronyms and Domain Terminology:** Stock tickers (`NASDAQ`, `DJIA`), technical abbreviations (`GHz`, `TCP/IP`, `AI`), and sports abbreviations (`UEFA`, `NBA`) must be preserved without destructive stemming.
