
# Pipeline entry point for the NLP Research Analyzer.


import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml


# load config

CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


#imports from package

from nlp_research.ingestion import ingest_files
from nlp_research.preprocessing import preprocess, split_into_paragraphs
from nlp_research.features import build_tfidf_matrix, get_top_keywords
from nlp_research.modeling import build_corpus, train_lda, compute_coherence, sweep_num_topics
from nlp_research.summarization import summarize
from nlp_research.visualization import (
    plot_top_keywords,
    plot_lda_topics,
    plot_coherence_curve,
    plot_wordcloud,
)


def _make_run_folder_name(file_paths):
    stems = [p.stem for p in file_paths]
    label = "+".join(stems)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{label}_{timestamp}"


def run_pipeline(file_paths,num_topics,top_n,save_plots,no_plots):
    if save_plots:
        run_name = _make_run_folder_name(file_paths)
        figures_dir = Path("outputs/figures") / run_name
        figures_dir.mkdir(parents=True, exist_ok=True)
    else:
        figures_dir = None

    print(f"\n[1/6] Ingesting {len(file_paths)} file(s):")
    for fp in file_paths:
        print(f"      • {fp}")
    raw_text = ingest_files(file_paths)
    print(f"      Combined: {len(raw_text):,} characters extracted.\n")

    print("[2/6] Preprocessing text …")
    tokens = preprocess(raw_text, min_word_len=CFG["preprocessing"]["min_word_len"])
    print(f"      {len(tokens):,} tokens after cleaning. Sample: {tokens[:10]}\n")

    print("[3/6] Computing TF-IDF …")
    processed_texts = [" ".join(tokens)]
    tfidf_matrix, tfidf_vectorizer = build_tfidf_matrix(
        processed_texts,
        max_df=CFG["tfidf"]["max_df"],
        min_df=CFG["tfidf"]["min_df"],
    )
    top_keywords = get_top_keywords(
        tfidf_matrix, tfidf_vectorizer, top_n=CFG["tfidf"]["top_n_keywords"]
    )
    print(f"      TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"      Top keywords: {[w for w, _ in top_keywords]}\n")

    print("[4/6] Training LDA model …")
    paragraphs = split_into_paragraphs(raw_text, min_length=CFG["preprocessing"]["min_paragraph_len"])
    cleaned_docs = [preprocess(p) for p in paragraphs]
    dictionary, corpus = build_corpus(cleaned_docs)
    lda_model = train_lda(
        corpus,
        dictionary,
        num_topics=num_topics,
        passes=CFG["lda"]["passes"],
        random_state=CFG["lda"]["random_state"],
    )
    for topic in lda_model.print_topics(num_words=8):
        print("     ", topic)

    coherence = compute_coherence(lda_model, cleaned_docs, dictionary)
    print(f"\n      Coherence Score (c_v): {coherence:.4f}\n")

    print(f"[5/6] Generating extractive summary (top {top_n} sentences) …")
    summary = summarize(raw_text, top_n=top_n)
    print("\n🔹 SUMMARY:\n")
    for s in summary:
        print(f"  - {s}")
    print()

    if no_plots:
        print("[6/6] Skipping visualizations (--no-plots).")
        return

    print("[6/6] Generating visualizations …")
    words, scores = zip(*top_keywords)

    plot_top_keywords(
        list(words), list(scores),
        save_path=(figures_dir / "top_keywords.png") if figures_dir else None,
        show=not save_plots,
    )
    plot_lda_topics(
        lda_model,
        save_dir=figures_dir,
        show=not save_plots,
    )

    sweep = sweep_num_topics(
        corpus, dictionary, cleaned_docs,
        topic_range=range(*CFG["lda"]["sweep_range"]),
        passes=CFG["lda"]["passes"],
    )
    nums, coh_scores = zip(*sweep)
    plot_coherence_curve(
        list(nums), list(coh_scores),
        save_path=(figures_dir / "coherence_curve.png") if figures_dir else None,
        show=not save_plots,
    )

    plot_wordcloud(
        " ".join(processed_texts),
        save_path=(figures_dir / "wordcloud.png") if figures_dir else None,
        show=not save_plots,
    )

    if save_plots:
        print(f"\n      Figures saved to: {figures_dir.resolve()}")



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NLP Research Analyzer — Milestone 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single PDF
  python3 main.py --files data/raw/paper.pdf --save-plots

  # Single TXT
  python3 main.py --files data/raw/notes.txt --save-plots

  # Multiple documents combined
  python3 main.py --files data/raw/paper1.pdf data/raw/paper2.pdf data/raw/notes.txt --save-plots
""",
    )
    parser.add_argument(
        "--files", required=True, nargs="+", type=Path,
        metavar="FILE",
        help="One or more input files (.pdf or .txt) to analyse together",
    )
    parser.add_argument(
        "--num-topics", type=int, default=CFG["lda"]["num_topics"],
        help=f"Number of LDA topics (default: {CFG['lda']['num_topics']})",
    )
    parser.add_argument(
        "--top-n", type=int, default=CFG["summarization"]["top_n"],
        help=f"Summary sentences to extract (default: {CFG['summarization']['top_n']})",
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save figures to outputs/figures/<run_name>/ instead of displaying",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip all visualizations",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    missing = [fp for fp in args.files if not fp.exists()]
    if missing:
        for fp in missing:
            print(f"ERROR: File not found: {fp}", file=sys.stderr)
        sys.exit(1)

    run_pipeline(
        file_paths=args.files,
        num_topics=args.num_topics,
        top_n=args.top_n,
        save_plots=args.save_plots,
        no_plots=args.no_plots,
    )