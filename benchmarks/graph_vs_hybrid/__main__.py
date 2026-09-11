"""CLI: ``python -m benchmarks.graph_vs_hybrid <stage> -c bench.yaml``.

Stages are separate commands on purpose. Indexing is expensive and rarely needs redoing;
retrieval evaluation is free and gets re-run while tuning graph parameters; answer
evaluation costs API calls.
"""

import argparse
import json
import logging
from dataclasses import asdict

from dotenv import load_dotenv

from .config import BenchConfig, write_stage_configs

logger = logging.getLogger("bench")

DEFAULT_CONFIG = "benchmarks/graph_vs_hybrid/bench_pubmedqa.yaml"


def _prepare(config: BenchConfig, args: argparse.Namespace) -> None:
    from .tasks import load_task

    task = load_task(config.task)

    if not args.qa_only:
        stats = task.prepare_corpus(config)
        path = config.results_path("corpus.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")

    if not args.corpus_only:
        task.prepare_questions(config)

    write_stage_configs(config)


def _index(config: BenchConfig, args: argparse.Namespace) -> None:
    from . import index_bench

    index_bench.run(
        config,
        keep_existing=args.keep_existing,
        graph_only=getattr(args, "graph_only", False),
        tag=getattr(args, "tag", None),
    )


def _retrieval(config: BenchConfig, args: argparse.Namespace) -> None:
    from . import retrieval_bench

    retrieval_bench.run(config, args.strategies, getattr(args, "tag", None))


def _answer(config: BenchConfig, args: argparse.Namespace) -> None:
    from . import answer_bench

    # `all` may carry `lexical` through for the retrieval stage; there is no lexical RAG
    # pipeline, so it is dropped here rather than making `all` reject the flag.
    strategies = [s for s in args.strategies if s != "lexical"]
    if strategies != list(args.strategies):
        logger.info("Skipping the lexical baseline: it is a retrieval-only strategy.")
    answer_bench.run(config, strategies)


def _report(config: BenchConfig, _: argparse.Namespace) -> None:
    from . import report

    path = report.run(config)
    logger.info(f"Report written to {path}")
    print(path.read_text(encoding="utf-8"))


def _all(config: BenchConfig, args: argparse.Namespace) -> None:
    _prepare(config, args)
    _index(config, args)
    _retrieval(config, args)
    if not args.skip_answer:
        _answer(config, args)
    _report(config, args)


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(prog="graph_vs_hybrid", description=__doc__)
    parser.add_argument("-c", "--config-file", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--run-dir", default=None, help="Override run_dir from the config."
    )

    # Repeated on every subcommand so the options work on either side of the stage name.
    # SUPPRESS keeps a value given before the stage from being overwritten by the default.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config-file", default=argparse.SUPPRESS)
    common.add_argument("--run-dir", default=argparse.SUPPRESS)
    common.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override a config field, e.g. --set graph.passage_ratio=1.5. Query-time "
        "graph settings need no re-indexing, so a sweep is `retrieval` runs.",
    )

    subparsers = parser.add_subparsers(dest="stage", required=True)

    def add_stage(name: str, **kwargs) -> argparse.ArgumentParser:
        return subparsers.add_parser(name, parents=[common], **kwargs)

    prepare = add_stage("prepare", help="Build the corpus and question set.")
    prepare.add_argument("--corpus-only", action="store_true")
    prepare.add_argument("--qa-only", action="store_true")
    prepare.set_defaults(handler=_prepare)

    index = add_stage("index", help="Time `index` then `graph-index`.")
    index.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not wipe the previous index first (timings then measure an update).",
    )
    index.add_argument(
        "--graph-only",
        action="store_true",
        help="Reuse the existing Milvus collection and rebuild only the graph. Pair with "
        "--set graph.artifacts_name=... to keep several graphs side by side.",
    )
    index.add_argument(
        "--tag", default=None, help="Suffix for this graph's log and result files."
    )
    index.set_defaults(handler=_index)

    for name, handler, help_text in (
        ("retrieval", _retrieval, "Score retrieval for the selected strategies."),
        ("answer", _answer, "Score end-to-end answers for both strategies."),
    ):
        sub = add_stage(name, help=help_text)
        # Only `hybrid` and `graph` are mmore RAG pipelines. The baselines and the fusions
        # exist to be *retrieved* with and have no generator behind them, so they are
        # offered on `retrieval` only.
        from .retrieval_bench import DEFAULT_STRATEGIES, STRATEGIES

        available = list(STRATEGIES) if name == "retrieval" else ["hybrid", "graph"]
        sub.add_argument(
            "--strategies",
            nargs="+",
            default=list(DEFAULT_STRATEGIES) if name == "retrieval" else available,
            choices=available,
        )
        sub.set_defaults(handler=handler)
        if name == "retrieval":
            sub.add_argument(
                "--tag",
                default=None,
                help="Suffix for the result files, so a parameter sweep does not "
                "overwrite its own baseline.",
            )

    add_stage("report", help="Render the comparison.").set_defaults(handler=_report)

    everything = add_stage("all", help="Every stage, in order.")
    everything.add_argument("--corpus-only", action="store_true")
    everything.add_argument("--qa-only", action="store_true")
    everything.add_argument("--keep-existing", action="store_true")
    everything.add_argument("--skip-answer", action="store_true")
    everything.add_argument(
        "--strategies",
        nargs="+",
        default=["hybrid", "graph", "lexical"],
        choices=["hybrid", "graph", "lexical"],
    )
    everything.set_defaults(handler=_all)

    args = parser.parse_args()
    config = BenchConfig.load(getattr(args, "config_file", DEFAULT_CONFIG))
    run_dir = getattr(args, "run_dir", None)
    if run_dir:
        config.run_dir = run_dir
    if getattr(args, "overrides", None):
        config.override(args.overrides)
        logger.info(f"Config overrides: {', '.join(args.overrides)}")
    config.root.mkdir(parents=True, exist_ok=True)

    args.handler(config, args)


if __name__ == "__main__":
    main()
