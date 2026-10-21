# Forensic Text-Topic Prediction

A Python research prototype for predicting topics in digital forensic text documents using semantic embeddings, clustering, and zero-shot classification.

## About the Project

Digital forensic investigations can involve large volumes of text, including messages, emails, and other documents. Manually reviewing all this information can be time-consuming.

This project explores how Natural Language Processing (NLP) and pretrained language models can help identify documents related to candidate investigation topics.

The project is inspired by the research paper:

**Evaluating and Extending Techniques for Fine-Grained Text-Topic Prediction for Digital Forensic Data**

## Features

- Load text documents from CSV files.
- Generate semantic embeddings using Sentence Transformers.
- Cluster semantically similar documents.
- Cluster candidate topic labels.
- Perform zero-shot text classification.
- Rank candidate topics for each document.
- Combine document-level and cluster-level classification scores.
- Export predictions to a CSV file.
- Analyze additional documents without hardcoding their contents.

## Technologies Used here

- Python
- Pandas
- NumPy
- Scikit-learn
- Sentence Transformers
- Hugging Face Transformers
- PyTorch

## Project Structure

```text
forensic_topic_prediction/
├── main.py
├── document.csv
├── requirements.txt
├── README.md
├── .gitignore
└── predictions.csv    # Generated output
```

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd forensic-text-topic-prediction
```

Replace the placeholder with your GitHub repository URL.

### 2. Create a virtual environment

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

The pretrained models may need to be downloaded on the first run, requiring an internet connection.

## Input Dataset

The application reads a CSV file containing a `text` column and optionally a `document_id` column.

Example `document.csv`:

```csv
document_id,text
D001,"Money was transferred between several accounts."
D002,"The invoice was altered before payment."
D003,"Someone accessed the account without permission."
```

If `document_id` is missing, the program generates document identifiers automatically.

## Running the Project

Run the application with the default dataset:

```powershell
python main.py
```

By default, the application reads `document.csv` and saves its predictions to `predictions.csv`.

### Analyze another CSV file

```powershell
python main.py --input "new_documents.csv" --output "new_predictions.csv"
```

You can add new documents to your CSV file or provide another dataset without changing the document-processing code.

## Methodology

The current prototype follows this workflow:

1. Load and validate text documents.
2. Generate semantic embeddings for documents and candidate topics.
3. Cluster candidate topics using agglomerative clustering.
4. Cluster documents based on semantic similarity.
5. Apply zero-shot classification to candidate topic labels.
6. Refine predictions using topic-cluster representatives.
7. Combine individual document scores with representative-document scores.
8. Rank the predicted topics and export the results.

The current default models are:

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Zero-shot classification model:** `facebook/bart-large-mnli`

These are practical model choices and have not been verified as the exact models used in the original paper.

## Output

The generated `predictions.csv` includes:

| Column | Description |
|---|---|
| `document_id` | Document identifier |
| `text` | Input document text |
| `document_cluster` | Assigned document cluster |
| `predicted_topics` | Ranked candidate topics |
| `topic_scores` | Scores for the ranked topics |
| `top_topic` | Highest-ranked topic |
| `top_topic_score` | Score of the highest-ranked topic |

The scores are model outputs, not calibrated probabilities that a document belongs to a topic or proves criminal activity.

## Limitations

This project is a research prototype and is not a production-ready digital forensic tool.

- The complete experiments from the original research paper have not been reproduced.
- The current clustering procedure has not been verified as an exact reproduction of the paper's algorithms.
- The original datasets and supervised baseline comparisons have not been fully reproduced.
- Predictions require evaluation against labelled data.
- Model outputs should be reviewed by a human investigator.
- Real forensic evidence must be handled with appropriate authorization, privacy safeguards, and integrity controls.

## Future Work

- Reproduce the original paper's algorithms and experiments.
- Evaluate predictions using appropriate classification metrics.
- Implement the supervised baseline methods from the paper.
- Add document upload and manual text-entry interfaces.
- Improve evidence retrieval and result explainability.
- Investigate secure, fully offline deployment.

## Reference Paper

Al Farabi, K. M., Agrawal, G., Dorai, G., Bardhan, R., Maleki, H., and Kadri, T. E. (2025).

*Evaluating and Extending Techniques for Fine-Grained Text-Topic Prediction for Digital Forensic Data.*

- [University of Georgia research record](https://digitalcommons.law.uga.edu/fac_artchop/1767/)
- [Springer publisher page](https://link.springer.com/chapter/10.1007/978-3-031-89360-5_16)

## License

No license has been selected for this project yet.
