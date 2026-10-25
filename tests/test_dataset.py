"""Optional MultiHopRAG downloader; imports never perform network work."""


def main():
    from pathlib import Path

    from datasets import load_dataset

    output_dir = Path("./data/MultiHop-RAG")
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_dataset(
        "yixuantt/MultiHopRAG",
        "MultiHopRAG",
        split="train",
    )

    corpus = load_dataset(
        "yixuantt/MultiHopRAG",
        "corpus",
        split="train",
    )

    questions.to_json(
        str(output_dir / "questions.jsonl"),
        force_ascii=False,
    )
    corpus.to_json(
        str(output_dir / "corpus.jsonl"),
        force_ascii=False,
    )

    print(f"Saved {len(questions)} questions and {len(corpus)} documents")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
