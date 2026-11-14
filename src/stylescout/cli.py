"""Command-line entry points for StyleScout."""

from __future__ import annotations

import argparse
import logging
import sys


def build_catalog_cmd(args: argparse.Namespace) -> None:
    from stylescout.catalog.loader import build_catalog

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = build_catalog(sample_path=args.sample, db_path=args.db)
    print(
        f"Done — loaded: {summary['loaded']}, "
        f"dropped: {summary['dropped']}, "
        f"written: {summary['written']}"
    )


def build_index_cmd(args: argparse.Namespace) -> None:
    from stylescout.catalog.embeddings import build_index

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = build_index(db_path=args.db, chroma_path=args.chroma, collection_name=args.collection)
    print(f"Done — indexed: {summary['indexed']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="stylescout")
    sub = parser.add_subparsers(dest="command", required=True)

    bc = sub.add_parser("build-catalog", help="Load sample CSV and write catalog.db")
    bc.add_argument("--sample", default="data/sample_styles.csv")
    bc.add_argument("--db", default="data/catalog.db")
    bc.set_defaults(func=build_catalog_cmd)

    bi = sub.add_parser("build-index", help="Build Chroma vector index from catalog.db")
    bi.add_argument("--db", default="data/catalog.db")
    bi.add_argument("--chroma", default="data/chroma")
    bi.add_argument("--collection", default="products")
    bi.set_defaults(func=build_index_cmd)

    args = parser.parse_args()
    args.func(args)
    sys.exit(0)


if __name__ == "__main__":
    main()
