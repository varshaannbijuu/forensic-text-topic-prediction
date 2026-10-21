
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from transformers import pipeline


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "document.csv"
OUTPUT_FILE = BASE_DIR / "predictions.csv"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLASSIFICATION_MODEL = "facebook/bart-large-mnli"

TOPICS = [
    "financial fraud",
    "money laundering",
    "credential theft",
    "unauthorized computer access",
    "personal conversations",
]

DOCUMENT_DISTANCE_THRESHOLD = 0.65
TOPIC_DISTANCE_THRESHOLD = 0.55

TOP_TOPIC_CLUSTERS = 2
TOP_K = 3

INDIVIDUAL_WEIGHT = 0.70
CLUSTER_WEIGHT = 0.30

DEVICE = 0 if torch.cuda.is_available() else -1


# =========================================================
# 2. LOAD DOCUMENTS
# =========================================================

def load_documents(path):
    path = Path(path)

    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        available = list(BASE_DIR.glob("*.csv"))

        print(f"\nERROR: CSV file not found:\n{path}")
        print("\nCSV files in your project folder:")

        for file in available:
            print(" -", file.name)

        raise FileNotFoundError(
            f"Place document.csv in {BASE_DIR}"
        )

    df = pd.read_csv(path, encoding="utf-8-sig")

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    if "text" not in df.columns:
        raise ValueError(
            "CSV must contain a column named 'text'. "
            f"Found columns: {list(df.columns)}"
        )

    if "document_id" not in df.columns:
        df.insert(
            0,
            "document_id",
            [f"D{i + 1:04d}" for i in range(len(df))]
        )

    df = df[["document_id", "text"]].copy()

    df["document_id"] = df["document_id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    df = df[df["text"] != ""].reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable documents found in CSV.")

    if df["document_id"].duplicated().any():
        raise ValueError("Duplicate document IDs found.")

    print(f"\nLoaded {len(df)} documents from {path}")

    return df


# =========================================================
# 3. LOAD PRETRAINED MODELS
# =========================================================

def load_models():
    print("\nLoading embedding model...")

    encoder = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading zero-shot classification model...")

    classifier = pipeline(
        "zero-shot-classification",
        model=CLASSIFICATION_MODEL,
        device=DEVICE,
    )

    print(
        "Models loaded successfully.",
        "Device:",
        "GPU" if DEVICE >= 0 else "CPU",
    )

    return encoder, classifier


# =========================================================
# 4. GENERATE TEXT EMBEDDINGS
# =========================================================

def generate_embeddings(texts, encoder):
    embeddings = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return np.asarray(embeddings, dtype=np.float32)


# =========================================================
# 5. CLUSTER TEXT EMBEDDINGS
# =========================================================

def cluster_embeddings(embeddings, distance_threshold):
    count = len(embeddings)

    if count == 0:
        return np.array([], dtype=int)

    if count == 1:
        return np.zeros(1, dtype=int)

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )

    return model.fit_predict(embeddings)


# =========================================================
# 6. ZERO-SHOT CLASSIFICATION
# =========================================================

def classify_texts(classifier, texts, candidate_topics):
    if not candidate_topics:
        return [{} for _ in texts]

    results = []

    for text in texts:
        prediction = classifier(
            text,
            candidate_labels=list(candidate_topics),
            multi_label=True,
        )

        scores = dict(
            zip(
                prediction["labels"],
                prediction["scores"],
            )
        )

        results.append(scores)

    return results


# =========================================================
# 7. CLUSTER CANDIDATE TOPICS
# =========================================================

def build_topic_clusters(topic_embeddings):
    labels = cluster_embeddings(
        topic_embeddings,
        TOPIC_DISTANCE_THRESHOLD,
    )

    topic_to_cluster = {
        topic: int(cluster_id)
        for topic, cluster_id in zip(TOPICS, labels)
    }

    topic_groups = {}

    for topic, cluster_id in topic_to_cluster.items():
        topic_groups.setdefault(cluster_id, []).append(topic)

    representatives = {}

    for cluster_id, topics in topic_groups.items():
        indices = [
            TOPICS.index(topic)
            for topic in topics
        ]

        vectors = topic_embeddings[indices]

        centroid = vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)

        if norm > 0:
            centroid = centroid / norm

        similarities = vectors @ centroid

        representatives[cluster_id] = topics[
            int(np.argmax(similarities))
        ]

    return topic_to_cluster, topic_groups, representatives


# =========================================================
# 8. PREDICT TOPICS USING CLUSTERING + ZERO-SHOT
# =========================================================

def predict_topics(
    df,
    document_embeddings,
    topic_embeddings,
    classifier,
):
    print("\nClustering candidate topics...")

    (
        topic_to_cluster,
        topic_groups,
        representatives,
    ) = build_topic_clusters(topic_embeddings)

    print("Topic clusters:", len(topic_groups))

    print("\nClustering documents...")

    document_cluster_ids = cluster_embeddings(
        document_embeddings,
        DOCUMENT_DISTANCE_THRESHOLD,
    )

    df = df.copy()
    df["document_cluster"] = document_cluster_ids

    print(
        "Document clusters:",
        len(set(document_cluster_ids.tolist())),
    )

    # -----------------------------------------------------
    # A. Coarse classification against topic representatives
    # -----------------------------------------------------

    representative_topics = [
        representatives[c]
        for c in sorted(representatives)
    ]

    print("\nClassifying documents against topic clusters...")

    coarse_scores = classify_texts(
        classifier,
        df["text"].tolist(),
        representative_topics,
    )

    # -----------------------------------------------------
    # B. Refine candidate topics within top clusters
    # -----------------------------------------------------

    print("\nRefining topic predictions...")

    individual_scores = []

    for index, text in enumerate(df["text"].tolist()):
        ranked_representatives = sorted(
            coarse_scores[index],
            key=coarse_scores[index].get,
            reverse=True,
        )

        selected_topics = []

        for representative in ranked_representatives[
            :TOP_TOPIC_CLUSTERS
        ]:
            cluster_id = topic_to_cluster[representative]

            selected_topics.extend(
                topic_groups[cluster_id]
            )

        selected_topics = list(dict.fromkeys(selected_topics))

        if not selected_topics:
            selected_topics = list(TOPICS)

        refined = classify_texts(
            classifier,
            [text],
            selected_topics,
        )[0]

        # Keep all topics in a consistent output schema.
        full_scores = {
            topic: np.nan for topic in TOPICS
        }

        full_scores.update(refined)

        individual_scores.append(full_scores)

    # -----------------------------------------------------
    # C. Get a representative document for each cluster
    # -----------------------------------------------------

    print("\nComputing cluster-level topic scores...")

    cluster_scores = {}

    for cluster_id in sorted(set(document_cluster_ids.tolist())):
        indices = np.where(
            document_cluster_ids == cluster_id
        )[0]

        vectors = document_embeddings[indices]

        centroid = vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)

        if norm > 0:
            centroid = centroid / norm

        similarities = vectors @ centroid

        representative_index = int(
            indices[np.argmax(similarities)]
        )

        representative_text = df.iloc[
            representative_index
        ]["text"]

        cluster_scores[int(cluster_id)] = classify_texts(
            classifier,
            [representative_text],
            TOPICS,
        )[0]

    # -----------------------------------------------------
    # D. Combine scores and rank topics
    # -----------------------------------------------------

    output_rows = []

    for index, row in df.iterrows():
        cluster_id = int(row["document_cluster"])

        own_scores = individual_scores[index]
        shared_scores = cluster_scores[cluster_id]

        combined = {}

        for topic in TOPICS:
            own = own_scores.get(topic, np.nan)
            shared = shared_scores[topic]

            if pd.isna(own):
                combined[topic] = float(shared)
            else:
                combined[topic] = (
                    INDIVIDUAL_WEIGHT * float(own)
                    + CLUSTER_WEIGHT * float(shared)
                )

        ranked = sorted(
            combined,
            key=combined.get,
            reverse=True,
        )[:TOP_K]

        output_rows.append({
            "document_id": row["document_id"],
            "text": row["text"],
            "document_cluster": cluster_id,
            "predicted_topics": "; ".join(ranked),
            "topic_scores": "; ".join(
                f"{topic}={combined[topic]:.4f}"
                for topic in ranked
            ),
            "top_topic": ranked[0],
            "top_topic_score": round(
                combined[ranked[0]], 4
            ),
        })

    return pd.DataFrame(output_rows)


# =========================================================
# 9. MAIN EXECUTION
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Forensic fine-grained text-topic prediction"
    )

    parser.add_argument(
        "--input",
        default=str(INPUT_FILE),
        help="Input CSV path",
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_FILE),
        help="Output CSV path",
    )

    args = parser.parse_args()

    output_path = Path(args.output).expanduser()

    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    try:
        # Load documents.
        df = load_documents(args.input)

        # Load models.
        encoder, classifier = load_models()

        # Embed documents.
        print("\nGenerating document embeddings...")

        document_embeddings = generate_embeddings(
            df["text"].tolist(),
            encoder,
        )

        # Embed candidate topics.
        print("\nGenerating topic embeddings...")

        topic_embeddings = generate_embeddings(
            TOPICS,
            encoder,
        )

        # Predict topics.
        results = predict_topics(
            df=df,
            document_embeddings=document_embeddings,
            topic_embeddings=topic_embeddings,
            classifier=classifier,
        )

        # Save predictions.
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        results.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print("\n" + "=" * 60)
        print("FORENSIC TOPIC PREDICTION COMPLETED")
        print("=" * 60)

        print(
            results[
                [
                    "document_id",
                    "top_topic",
                    "top_topic_score",
                ]
            ].to_string(index=False)
        )

        print(f"\nResults saved to:\n{output_path}")

    except (FileNotFoundError, ValueError) as error:
        print(f"\nERROR: {error}")
        sys.exit(1)

    except Exception as error:
        print(f"\nUnexpected error: {error}")
        raise


if __name__ == "__main__":
    main()
