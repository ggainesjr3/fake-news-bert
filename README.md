# 🛡️ Trust Guard AI: BERT-Based Disinformation Analysis

Trust Guard AI is a high-performance audit terminal designed to detect disinformation patterns in news headlines. By leveraging a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model, this application provides real-time analysis, confidence scoring, and explainable AI metrics via a Streamlit dashboard.

## 🚀 Project Overview
The core goal of this project was to move beyond simple keyword matching and train a model to recognize the linguistic "fingerprints" of misinformation. The model was trained on the **ISOT Dataset**, which includes thousands of real news articles from Reuters and flagged disinformation from various unreliable sources.

## 🖼️ Dashboard Preview
| Verified Pattern | Baseline Detection | Technical Detail Audit |
| :---: | :---: | :---: |
| ![Verified News](screenshots/verified_success.png) | ![Disinfo Catch](screenshots/disinfo_catch.png) | ![Tech Details](screenshots/tech_audit.png) |

## 🛠️ Technical Audit & Explainability
One of the key features of this project is the **Technical Audit Expander**. It allows users to see:
* **Softmax Probability:** The mathematical confidence level of the prediction.
* **Model Architecture:** Details on the `bert-base-uncased` transformer used.
* **Logic Constraints:** How the model handles sensationalist language vs. factual reporting.

## 📊 Performance Analysis
During testing, the model proved robust against "adversarial" headlines—factual reports that use sensationalist words like "Leaked" or "Clandestine." However, the audit also revealed areas for improvement regarding highly technical or scientific jargon.

### 🚀 Future Improvements & Model Robustness
During the audit, the model demonstrated a high sensitivity to technical jargon, occasionally misclassifying dense scientific reporting as disinformation. 

**Engineering Insight:**
> "The ISOT dataset is highly effective for detecting political sensationalism, but it is heavily biased toward 2016-2017 political rhetoric. To make Trust Guard AI production-ready, the next phase would involve augmenting the training set with technical journals and local news. This would help the model better distinguish between 'complex technical jargon' and 'actual misinformation' patterns."

## ⚙️ Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ggainesjr3/fake-news-bert.git](https://github.com/ggainesjr3/fake-news-bert.git)