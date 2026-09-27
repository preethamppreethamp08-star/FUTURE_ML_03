"""
Resume / Candidate Screening System
------------------------------------
Works with the Kaggle "Resume Dataset" (snehaanbhawal/resume-dataset):
    ID, Resume_str, Resume_html, Category

Reads resumes + a job description, cleans the text, extracts skills,
scores each resume against the job description using TF-IDF + cosine
similarity, ranks candidates, and reports missing skills for each one.

Requires: pandas, scikit-learn, matplotlib
Also requires skills_dictionary.py in the same folder.

Usage:
    python resume_screening.py --category "Information-Technology"
Outputs:
    results/ranked_candidates.csv
    results/ranking_chart.png
"""

import os
import re
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skills_dictionary import extract_skills

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESUME_CSV = os.path.join(BASE_DIR, "data", "resumes.csv")       # place the Kaggle CSV here
JOB_DESC_TXT = os.path.join(BASE_DIR, "data", "job_description.txt")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ---------------------------------------------------------------------------
# 1. Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/extra whitespace. Scraped resume text
    from the Kaggle dataset also has repeated whitespace/newlines, which
    this collapses too."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\+\.\s]", " ", text)  # keep + and . for C++, Node.js
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 2. Load data — adapted to the Kaggle "Resume Dataset" column names
# ---------------------------------------------------------------------------
def load_resumes(path: str, category: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize the Kaggle schema (ID, Resume_str, Resume_html, Category)
    # to what the rest of the pipeline expects, while staying tolerant of
    # a custom CSV with different column names.
    rename_map = {}
    if "ID" in df.columns:
        rename_map["ID"] = "candidate_id"
    if "Resume_str" in df.columns:
        rename_map["Resume_str"] = "resume_text"
    if "Category" in df.columns:
        rename_map["Category"] = "category"
    df = df.rename(columns=rename_map)

    if "resume_text" not in df.columns:
        raise ValueError(
            "Couldn't find a resume text column. Expected 'Resume_str' "
            "(Kaggle dataset) or 'resume_text'."
        )
    if "candidate_id" not in df.columns:
        df["candidate_id"] = [f"C{i:04d}" for i in range(1, len(df) + 1)]
        df["candidate_id"] = df["candidate_id"].astype(str)
    if "name" not in df.columns:
    # The Kaggle dataset has no candidate names — use the ID as a stand-in,
    # forced to string so matplotlib treats it as a category label, not a number.
        df["name"] = "ID " + df["candidate_id"].astype(str)
    if "Resume_html" in df.columns:
        df = df.drop(columns=["Resume_html"])

    if category and "category" in df.columns:
        before = len(df)
        df = df[df["category"].str.lower() == category.lower()].reset_index(drop=True)
        print(f"Filtered to category '{category}': {len(df)}/{before} resumes")

    df["clean_text"] = df["resume_text"].apply(clean_text)
    return df


def load_job_description(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 3. Similarity scoring (TF-IDF + cosine similarity)
# ---------------------------------------------------------------------------
def score_resumes(df: pd.DataFrame, job_desc_clean: str) -> pd.Series:
    """
    TF-IDF + cosine similarity: explainable (score = weighted term overlap,
    not a black-box embedding) and needs no pretrained model download.
    Scaled to 0-100.
    """
    corpus = [job_desc_clean] + df["clean_text"].tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    job_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(job_vector, resume_vectors).flatten()
    return pd.Series((similarities * 100).round(1), index=df.index)


# ---------------------------------------------------------------------------
# 4. Skill extraction & gap analysis
# ---------------------------------------------------------------------------
def analyze_skills(df: pd.DataFrame, job_desc_raw: str):
    required_skills = extract_skills(job_desc_raw)
    df["matched_skills"] = df["resume_text"].apply(
        lambda t: sorted(extract_skills(t) & required_skills)
    )
    df["missing_skills"] = df["resume_text"].apply(
        lambda t: sorted(required_skills - extract_skills(t))
    )
    df["skill_match_pct"] = df["matched_skills"].apply(
        lambda matched: round(100 * len(matched) / max(len(required_skills), 1), 1)
    )
    return df, required_skills


# ---------------------------------------------------------------------------
# 5. Ranking + explanation
# ---------------------------------------------------------------------------
def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df["final_score"] = (0.6 * df["similarity_score"] + 0.4 * df["skill_match_pct"]).round(1)
    ranked = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked


def explain_ranking(ranked: pd.DataFrame, top_n: int = 15) -> None:
    print("\n=== Candidate Ranking ===\n")
    for _, row in ranked.head(top_n).iterrows():
        print(f"Rank {row['rank']}: {row['name']}  (Final score: {row['final_score']}/100)")
        print(f"   Text similarity to job description: {row['similarity_score']}/100")
        print(f"   Required-skill coverage: {row['skill_match_pct']}%  "
              f"-> matched: {', '.join(row['matched_skills']) if row['matched_skills'] else 'none'}")
        if row["missing_skills"]:
            print(f"   Missing skills: {', '.join(row['missing_skills'])}")
        else:
            print("   Missing skills: none — meets all listed requirements")
        print()


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------
def plot_ranking(ranked: pd.DataFrame, out_path: str, top_n: int = 15) -> None:
    top = ranked.head(top_n)
    plt.figure(figsize=(8, max(4, 0.4 * len(top))))
    colors = plt.cm.viridis([i / len(top) for i in range(len(top))])
    bars = plt.barh(top["name"][::-1], top["final_score"][::-1], color=colors[::-1])
    plt.xlabel("Final Score (0-100)")
    plt.title("Candidate Ranking (top results)")
    plt.xlim(0, 100)
    for bar, score in zip(bars, top["final_score"][::-1]):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                  f"{score}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Resume screening & ranking system")
    parser.add_argument("--category", type=str, default=None,
                         help="Filter resumes to one Kaggle dataset category, "
                              "e.g. 'Information-Technology', 'Engineering', 'Sales'")
    parser.add_argument("--top", type=int, default=15,
                         help="How many top candidates to print/plot")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_resumes(RESUME_CSV, category=args.category)
    job_desc_raw = load_job_description(JOB_DESC_TXT)
    job_desc_clean = clean_text(job_desc_raw)

    df["similarity_score"] = score_resumes(df, job_desc_clean)
    df, required_skills = analyze_skills(df, job_desc_raw)
    ranked = rank_candidates(df)

    print(f"Required skills identified from job description ({len(required_skills)}):")
    print(", ".join(sorted(required_skills)))

    explain_ranking(ranked, top_n=args.top)

    out_csv = os.path.join(RESULTS_DIR, "ranked_candidates.csv")
    ranked[[
        "rank", "candidate_id", "name", "final_score", "similarity_score",
        "skill_match_pct", "matched_skills", "missing_skills"
    ]].to_csv(out_csv, index=False)
    print(f"Saved ranked results to {out_csv}")

    chart_path = os.path.join(RESULTS_DIR, "ranking_chart.png")
    plot_ranking(ranked, chart_path, top_n=args.top)
    print(f"Saved ranking chart to {chart_path}")


if __name__ == "__main__":
    main()