import json
import csv
import math
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CANDIDATES_FILE = "candidates.jsonl"
OUTPUT_FILE = "submission.csv"

JD_TEXT = """
Senior AI Engineer founding team. Needs production ML systems, embeddings,
retrieval, ranking, vector search, recommendation systems, LLMs, fine tuning,
Python, evaluation frameworks, NDCG, MAP, MRR, A/B testing, hybrid search,
FAISS, Milvus, Elasticsearch, OpenSearch, Qdrant, Pinecone, Weaviate.
Prefer product company experience, startup mindset, shipped real systems.
Location Pune Noida India hybrid. Experience 5 to 9 years.
"""

POSITIVE_KEYWORDS = [
    "embedding", "embeddings", "retrieval", "ranking", "ranker",
    "recommendation", "recommender", "search", "semantic search",
    "vector", "vector database", "faiss", "milvus", "qdrant",
    "pinecone", "weaviate", "elasticsearch", "opensearch",
    "nlp", "llm", "fine-tuning", "finetuning", "lora", "qlora",
    "rag", "machine learning", "ml", "python", "evaluation",
    "ndcg", "map", "mrr", "a/b", "ab testing", "production",
    "deployed", "scale", "real users"
]

NEGATIVE_KEYWORDS = [
    "marketing manager", "accountant", "hr manager", "civil engineer",
    "mechanical engineer", "customer support", "operations manager",
    "photoshop", "seo", "sales", "accounting"
]

SERVICE_COMPANIES = [
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "mindtree"
]

PRODUCT_COMPANIES = [
    "product", "saas", "software", "platform", "startup",
    "marketplace", "ai-native"
]

INDIA_LOCATIONS = [
    "pune", "noida", "mumbai", "delhi", "gurgaon",
    "hyderabad", "bengaluru", "bangalore", "india"
]


def safe_lower(x):
    return str(x or "").lower()


def normalize(value, max_value):
    try:
        value = float(value)
        return max(0, min(value / max_value, 1))
    except:
        return 0


def candidate_text(c):
    p = c.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
        p.get("current_industry", ""),
        p.get("location", ""),
        p.get("country", "")
    ]

    for job in c.get("career_history", []):
        parts.extend([
            job.get("title", ""),
            job.get("industry", ""),
            job.get("description", ""),
            job.get("company", "")
        ])

    for s in c.get("skills", []):
        parts.append(s.get("name", ""))

    for e in c.get("education", []):
        parts.extend([
            e.get("degree", ""),
            e.get("field_of_study", ""),
            e.get("tier", "")
        ])

    return " ".join(parts)


def keyword_score(text, keywords):
    text = safe_lower(text)
    score = 0
    for kw in keywords:
        if kw in text:
            score += 1
    return score / max(len(keywords), 1)


def experience_score(years):
    try:
        years = float(years)
    except:
        return 0

    if 5 <= years <= 9:
        return 1.0
    if 4 <= years < 5:
        return 0.75
    if 9 < years <= 11:
        return 0.65
    if 3 <= years < 4:
        return 0.45
    return 0.25


def location_score(c):
    p = c.get("profile", {})
    text = safe_lower(p.get("location", "") + " " + p.get("country", ""))
    signals = c.get("redrob_signals", {})

    if any(loc in text for loc in INDIA_LOCATIONS):
        return 1.0

    if signals.get("willing_to_relocate", False):
        return 0.65

    return 0.25


def behavior_score(c):
    s = c.get("redrob_signals", {})

    score = 0
    score += 0.18 if s.get("open_to_work_flag") else 0
    score += 0.15 * float(s.get("recruiter_response_rate", 0) or 0)
    score += 0.15 * float(s.get("interview_completion_rate", 0) or 0)

    avg_resp = float(s.get("avg_response_time_hours", 999) or 999)
    score += 0.10 * (1 - min(avg_resp, 168) / 168)

    notice = float(s.get("notice_period_days", 180) or 180)
    score += 0.12 * (1 - min(notice, 180) / 180)

    github = float(s.get("github_activity_score", -1) or -1)
    if github >= 0:
        score += 0.10 * min(github / 100, 1)

    score += 0.08 * normalize(s.get("saved_by_recruiters_30d", 0), 20)
    score += 0.05 * normalize(s.get("profile_completeness_score", 0), 100)
    score += 0.04 if s.get("verified_email") else 0
    score += 0.03 if s.get("verified_phone") else 0

    return min(score, 1)


def career_score(c):
    text = safe_lower(candidate_text(c))
    score = 0

    if any(x in text for x in ["recommendation", "recommender", "ranking", "search", "retrieval"]):
        score += 0.35

    if any(x in text for x in ["production", "deployed", "scale", "real users", "platform"]):
        score += 0.25

    if any(x in text for x in PRODUCT_COMPANIES):
        score += 0.20

    if any(x in text for x in ["python", "ml", "machine learning", "nlp", "llm"]):
        score += 0.20

    return min(score, 1)


def penalty_score(c):
    text = safe_lower(candidate_text(c))
    p = c.get("profile", {})

    penalty = 0

    current_title = safe_lower(p.get("current_title", ""))
    if any(x in current_title for x in NEGATIVE_KEYWORDS):
        penalty += 0.25

    neg_count = sum(1 for x in NEGATIVE_KEYWORDS if x in text)
    penalty += min(neg_count * 0.03, 0.20)

    companies = [
        safe_lower(j.get("company", ""))
        for j in c.get("career_history", [])
    ]

    if companies and all(any(sc in comp for sc in SERVICE_COMPANIES) for comp in companies):
        penalty += 0.15

    if "langchain" in text and not any(x in text for x in ["production", "retrieval", "ranking", "search"]):
        penalty += 0.15

    return min(penalty, 0.6)


def reasoning(c, score):
    p = c.get("profile", {})
    s = c.get("redrob_signals", {})
    text = safe_lower(candidate_text(c))

    reasons = []

    exp = p.get("years_of_experience", 0)
    title = p.get("current_title", "Candidate")

    reasons.append(f"{title} with {exp} years experience")

    if any(x in text for x in ["retrieval", "ranking", "search", "recommendation", "vector", "embedding"]):
        reasons.append("shows relevant retrieval/ranking or ML system exposure")

    if any(x in text for x in ["python", "nlp", "llm", "machine learning", "fine-tuning", "lora"]):
        reasons.append("has AI/ML skills aligned with the JD")

    if s.get("open_to_work_flag"):
        reasons.append("is open to work")

    if s.get("notice_period_days", 180) <= 30:
        reasons.append("has a short notice period")

    if not reasons:
        reasons.append("included based on overall profile strength")

    return "; ".join(reasons[:3]) + "."


def main():
    candidates = []

    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                candidates.append(c)

    texts = [JD_TEXT] + [candidate_text(c) for c in candidates]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2)
    )

    tfidf = vectorizer.fit_transform(texts)
    jd_vec = tfidf[0]
    cand_vecs = tfidf[1:]

    semantic_scores = cosine_similarity(jd_vec, cand_vecs).flatten()

    scored = []

    for i, c in enumerate(candidates):
        text = candidate_text(c)
        p = c.get("profile", {})

        semantic = float(semantic_scores[i])
        keyword = keyword_score(text, POSITIVE_KEYWORDS)
        exp = experience_score(p.get("years_of_experience", 0))
        behavior = behavior_score(c)
        career = career_score(c)
        location = location_score(c)
        penalty = penalty_score(c)

        final = (
            semantic * 0.30 +
            keyword * 0.18 +
            career * 0.22 +
            behavior * 0.15 +
            exp * 0.10 +
            location * 0.05
        )

        final = max(0, final - penalty)
        final = round(final, 6)

        scored.append({
            "candidate_id": c["candidate_id"],
            "score": final,
            "candidate": c
        })

    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    top100 = scored[:100]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, item in enumerate(top100, start=1):
            writer.writerow([
                item["candidate_id"],
                rank,
                item["score"],
                reasoning(item["candidate"], item["score"])
            ])

    print("Advanced submission.csv generated successfully")
    print("Top candidate:", top100[0]["candidate_id"], top100[0]["score"])


if __name__ == "__main__":
    main()