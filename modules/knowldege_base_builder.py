# modules/knowledge_base_builder.py
# Builds the FAISS knowledge base from trusted sources.
# This runs ONCE — creates the vector store saved to disk.
#
# Why build once and save?
# Embedding 1000s of documents takes minutes.
# Loading a saved FAISS index takes seconds.

import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
import faiss
import wikipedia

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Hardcoded Trusted Documents ───────────────────────────────────────
TRUSTED_DOCUMENTS = [
    {
        "text"   : "COVID-19 vaccines have been approved by health authorities worldwide including the FDA, EMA, and WHO. Clinical trials involving tens of thousands of participants demonstrated safety and efficacy. Vaccines are not banned in Europe. The European Medicines Agency has approved multiple COVID-19 vaccines for use across EU member states.",
        "source" : "who",
        "date"   : "2023-01-01"
    },
    {
        "text"   : "There is no scientific evidence that COVID-19 vaccines cause infertility. Studies involving hundreds of thousands of vaccinated individuals show no effect on fertility rates. Major health organizations including WHO, CDC, and NHS have confirmed vaccines are safe for people of reproductive age.",
        "source" : "who",
        "date"   : "2023-06-01"
    },
    {
        "text"   : "The United States unemployment rate reached a 50-year low of 3.5% in September 2019 under the Trump administration, matching a rate last seen in 1969. The Bureau of Labor Statistics confirmed this figure.",
        "source" : "reuters",
        "date"   : "2019-10-04"
    },
    {
        "text"   : "Barack Obama signed the Affordable Care Act into law in March 2010. The law expanded Medicaid coverage and created health insurance marketplaces. Over 20 million Americans gained health insurance coverage as a result.",
        "source" : "reuters",
        "date"   : "2020-03-23"
    },
    {
        "text"   : "Climate change is supported by overwhelming scientific consensus. NASA and NOAA data show global temperatures have risen approximately 1.1 degrees Celsius since the pre-industrial period. Human activities, primarily burning fossil fuels, are the dominant cause.",
        "source" : "reuters",
        "date"   : "2023-03-01"
    },
    {
        "text"   : "The United States national debt exceeded 30 trillion dollars in February 2022. The debt has grown under both Republican and Democratic administrations. The Congressional Budget Office tracks and projects federal debt levels.",
        "source" : "reuters",
        "date"   : "2022-02-01"
    },
    {
        "text"   : "Donald Trump was impeached twice by the House of Representatives — first in December 2019 on charges of abuse of power and obstruction of Congress, and second in January 2021 on charges of incitement of insurrection. He was acquitted by the Senate both times.",
        "source" : "reuters",
        "date"   : "2021-02-13"
    },
    {
        "text"   : "Hillary Clinton served as United States Secretary of State from 2009 to 2013 under President Barack Obama. She previously served as a US Senator from New York from 2001 to 2009. She was the Democratic presidential nominee in 2016.",
        "source" : "wikipedia",
        "date"   : "2023-01-01"
    },
    {
        "text"   : "The Affordable Care Act, also known as Obamacare, was upheld by the Supreme Court in 2012 and again in 2021. The individual mandate requiring Americans to purchase insurance was effectively repealed in 2017 when Congress set the penalty to zero.",
        "source" : "reuters",
        "date"   : "2021-06-17"
    },
    {
        "text"   : "Gun violence in the United States results in over 40,000 deaths annually according to CDC data. This includes homicides, suicides, and accidents. The US has the highest rate of gun ownership among developed nations.",
        "source" : "reuters",
        "date"   : "2022-01-01"
    },
    {
        "text"   : "The US federal minimum wage has been $7.25 per hour since 2009, making it one of the longest periods without an increase. Many states and cities have set higher minimum wages. Economists debate the employment effects of minimum wage increases.",
        "source" : "reuters",
        "date"   : "2023-01-01"
    },
    {
        "text"   : "Social Security is a federal program providing retirement, disability, and survivor benefits. It is funded through payroll taxes. The Social Security Administration projects the trust fund could be depleted by 2033 without legislative changes.",
        "source" : "reuters",
        "date"   : "2023-06-01"
    },
    {
        "text"   : "The United States spends more on healthcare per capita than any other developed nation. According to CMS data, US healthcare spending reached $4.3 trillion in 2021, representing 18.3% of GDP.",
        "source" : "reuters",
        "date"   : "2022-12-01"
    },
    {
        "text"   : "Immigration to the United States is regulated by federal law. The US admits approximately 1 million legal permanent residents annually. Unauthorized immigration has been a contentious political issue, with estimates of undocumented population ranging from 10 to 12 million.",
        "source" : "reuters",
        "date"   : "2023-01-01"
    },
    {
        "text"   : "The Centers for Disease Control and Prevention (CDC) is a federal public health agency headquartered in Atlanta, Georgia. It does not have self-destruct mechanisms. The CDC monitors and responds to public health threats including disease outbreaks.",
        "source" : "wikipedia",
        "date"   : "2023-01-01"
    },
    {
        "text"   : "Coal production in the United States began declining around 2008, primarily due to competition from natural gas which became cheaper following the shale gas revolution. Renewable energy growth has further reduced coal's share of electricity generation.",
        "source" : "reuters",
        "date"   : "2022-01-01"
    },
    {
        "text"   : "The US federal tax code underwent major changes with the Tax Cuts and Jobs Act of 2017, which lowered the corporate tax rate from 35% to 21% and reduced individual income tax rates. The law added approximately $1.9 trillion to the national debt over 10 years.",
        "source" : "reuters",
        "date"   : "2018-01-01"
    },
    {
        "text"   : "Planned Parenthood provides reproductive health services including contraception, STI testing, and cancer screenings. Abortion services represent approximately 3% of total services provided. The organization receives federal funding through Medicaid reimbursements.",
        "source" : "reuters",
        "date"   : "2022-01-01"
    },
    {
        "text"   : "The North American Free Trade Agreement (NAFTA) was replaced by the United States-Mexico-Canada Agreement (USMCA) in 2020. Trade between the three countries totals over $1.5 trillion annually. The agreements govern tariffs and trade rules.",
        "source" : "reuters",
        "date"   : "2020-07-01"
    },
    {
        "text"   : "Voter fraud in US elections is extremely rare according to multiple studies and court decisions. The Heritage Foundation database documents approximately 1,300 proven instances over several decades across billions of votes cast.",
        "source" : "reuters",
        "date"   : "2022-11-01"
    }
]


# ── Wikipedia Fetcher ─────────────────────────────────────────────────
def fetch_wikipedia_docs():
    """
    Fetches Wikipedia summaries for topics covering LIAR claim domains.
    Why Wikipedia? Free, broad, reliable, covers politics/health/economy.
    """

    topics = [
        "Barack Obama presidency",
        "Donald Trump presidency",
        "Hillary Clinton political career",
        "United States unemployment rate history",
        "Affordable Care Act",
        "Medicare United States",
        "United States national debt",
        "federal minimum wage United States",
        "gun violence United States statistics",
        "United States Border Patrol",
        "coal industry United States decline",
        "natural gas production United States",
        "climate change scientific consensus",
        "voter fraud United States",
        "Social Security United States",
        "United States immigration statistics",
        "NATO alliance history",
        "Iran human rights",
        "Wisconsin state government",
        "Texas economy jobs",
        "Neville Chamberlain appeasement policy",
        "United States GDP growth history",
        "childhood obesity United States",
        "Latino poverty United States",
        "United States Congress budget process",
        "Scott Walker Wisconsin governor",
        "Mitt Romney political career",
        "US healthcare spending",
        "United States federal budget",
        "American Recovery Reinvestment Act"
    ]

    documents = []

    for topic in topics:
        try:
            results = wikipedia.search(topic, results=1)
            if not results:
                continue

            page    = wikipedia.page(results[0], auto_suggest=False)
            summary = page.summary[:800]

            if len(summary) < 100:
                continue

            documents.append({
                "text"        : summary,
                "source"      : "wikipedia",
                "date"        : "2024-01-01",
                "origin_split": "external",
                "origin_dataset": "wikipedia",
                "is_external" : True,
                "kb_profile"  : "external_reference"
            })
            print(f"  ✓ {page.title}")
            time.sleep(0.3)

        except Exception as e:
            print(f"  ✗ {topic} — {e}")
            continue

    return documents


# ── Recency Weight ────────────────────────────────────────────────────
def compute_recency_weight(date_str):
    """
    Computes recency weight for a document based on its date.
    More recent = higher weight. Older than MAX_AGE_DAYS = zero.
    """
    try:
        doc_date = datetime.strptime(date_str, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        now      = datetime.now(timezone.utc)
        days_old = (now - doc_date).days
        weight   = config.RECENCY_WEIGHT * max(
            0, 1 - days_old / config.MAX_AGE_DAYS
        )
        return round(weight, 4)
    except Exception:
        return 0.0


def _truncate_words(text, max_words=300):
    """Returns the first ``max_words`` words of a text string."""

    words = str(text).split()
    return " ".join(words[:max_words])


def _parse_isot_date(date_value):
    """
    Converts ISOT date strings like 'January 26, 2016 ' to ISO format.
    Falls back to an empty string when parsing fails.
    """

    raw = str(date_value or "").strip()
    if not raw:
        return ""

    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    return ""


def load_isot_trusted_documents(max_articles=None):
    """
    Builds a trusted ISOT corpus from Reuters articles in the train split.

    Why train split only?
    It gives the retriever broad coverage while avoiding direct leakage
    from validation/test rows into the evidence store.
    """

    path = config.get_processed_split_path("train", "isot")
    print(f"Loading ISOT trusted articles from {path}")

    df = pd.read_csv(path)
    df = df.dropna(subset=["binary_label", "title", "text"])
    df = df[df["binary_label"].astype(int) == 1].reset_index(drop=True)

    if max_articles:
        df = df.head(max_articles).copy()

    documents = []
    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        body = _truncate_words(row.get("text", ""), max_words=300)
        text = f"Title: {title}\nBody: {body}".strip()

        if len(text) < 80:
            continue

        documents.append({
            "text"          : text,
            "source"        : "reuters",
            "date"          : _parse_isot_date(row.get("date", "")) or "2017-01-01",
            "origin_split"  : "train",
            "origin_dataset": "isot",
            "is_external"   : False,
            "kb_profile"    : "dataset_train_reuters"
        })

    print(f"Loaded {len(documents)} trusted Reuters articles for ISOT")
    return documents


def load_liar_trusted_documents():
    """Builds the original LIAR-oriented trusted corpus."""

    print("Fetching Wikipedia articles...")
    print("="*50)
    wiki_docs = fetch_wikipedia_docs()
    print(f"\nFetched {len(wiki_docs)} Wikipedia articles")

    all_docs = TRUSTED_DOCUMENTS + wiki_docs
    print(f"Total documents: {len(all_docs)}")
    print(f"  Hardcoded : {len(TRUSTED_DOCUMENTS)}")
    print(f"  Wikipedia : {len(wiki_docs)}")
    return all_docs


def build_documents_for_active_dataset(max_articles=None):
    """Returns the trusted document list for the active dataset."""

    dataset = config._normalize_dataset_name()
    if dataset == "isot":
        return load_isot_trusted_documents(max_articles=max_articles)
    return load_liar_trusted_documents()


# ── Build Index ───────────────────────────────────────────────────────
def build_knowledge_base(documents, output_dir):
    """
    Encodes documents and saves FAISS index + metadata to disk.

    Args:
        documents : list of dicts with text, source, date
        output_dir: directory to save index + metadata
    """

    print(f"\nBuilding knowledge base with {len(documents)} documents...")

    # Load embedding model
    print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    embedder   = SentenceTransformer(config.EMBEDDING_MODEL)

    # Encode all documents
    texts      = [doc["text"] for doc in documents]
    print("Computing embeddings...")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    index.add(embeddings)
    print(f"  Index size: {index.ntotal} vectors")

    # Build metadata
    metadata = []
    for doc in documents:
        source = doc.get("source", "unknown").lower()
        metadata.append({
            "text"           : doc["text"],
            "source"         : source,
            "date"           : doc.get("date", "unknown"),
            "source_weight"  : config.SOURCE_WEIGHTS.get(
                source, config.SOURCE_WEIGHTS["unknown"]
            ),
            "recency_weight" : compute_recency_weight(
                doc.get("date", "")
            ),
            "origin_split"   : doc.get("origin_split", "external"),
            "origin_dataset" : doc.get("origin_dataset", "external"),
            "is_external"    : bool(doc.get("is_external", True)),
            "kb_profile"     : doc.get("kb_profile", "external_reference"),
        })

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)

    faiss.write_index(
        index,
        os.path.join(output_dir, "faiss_index.bin")
    )
    with open(
        os.path.join(output_dir, "metadata.pkl"), "wb"
    ) as f:
        pickle.dump(metadata, f)

    print(f"\nKnowledge base saved to {output_dir}")
    print(f"  faiss_index.bin — {index.ntotal} vectors")
    print(f"  metadata.pkl    — {len(metadata)} documents")

    return index, metadata


if __name__ == "__main__":
    dataset = config._normalize_dataset_name()
    output_dir = os.path.join(config.KNOWLEDGE_BASE, dataset)
    print(f"Building knowledge base for dataset: {dataset}")

    documents = build_documents_for_active_dataset()

    # Build and save index
    index, metadata = build_knowledge_base(documents, output_dir)

    print("\nKnowledge base ready.")
    print(f"Final index size: {index.ntotal} vectors")
