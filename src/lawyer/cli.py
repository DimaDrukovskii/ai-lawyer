from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

from .bootstrap import build_deps
from .config import Settings
from .doctor import run_doctor
from .llm.base import LLMError
from .research import ResearchContext, load_zones, run_research, select_zones
from .review import ReviewPaths, run_review

CHECKLIST = Path("skills/usn-marketplace-review/checklist.md")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lawyer", description=__doc__)
    p.add_argument(
        "--mock", action="store_true", help="прогон без ключей и сети (LLM_PROVIDER=mock)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("zones", help="показать зоны исследования")
    sub.add_parser("doctor", help="проверить ключи, tool calling, поиск и fetch")

    r = sub.add_parser("research", help="мультиагентный research → база знаний")
    r.add_argument("--tax-year", type=int, default=date.today().year)
    r.add_argument("--zones", help="id зон через запятую (по умолчанию все)")
    r.add_argument("--rounds", type=int, default=1, help="раундов критики и дозакрытия")
    r.add_argument("--resume", metavar="RUN_ID", help="продолжить прогон с кэша")

    v = sub.add_parser("review", help="проверить документы кейса (декларация УСН + КУДиР)")
    v.add_argument("case", type=Path, help="папка кейса: profile.yaml, context.md, docs/")
    v.add_argument("--knowledge", metavar="RUN_ID", help="прогон базы знаний (по умолчанию LATEST)")
    return p


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings.from_env()
    return replace(s, provider="mock") if args.mock else s


def _knowledge_root(settings: Settings) -> Path:
    """Мок-прогоны пишут фейковые «доверенные» утверждения — им нельзя попадать в knowledge/,
    иначе настоящий review потом опрётся на них."""
    sub = Path("out") / "mock-knowledge" if settings.provider == "mock" else Path("knowledge")
    return settings.root / sub


async def _research(args: argparse.Namespace, settings: Settings) -> int:
    deps = build_deps(settings)
    zones = select_zones(
        load_zones(settings.root / "research" / "zones.yaml"),
        args.zones.split(",") if args.zones else None,
    )
    run_id = args.resume or datetime.now().strftime("%Y%m%d-%H%M")
    result = await run_research(
        deps,
        zones,
        ResearchContext(tax_year=args.tax_year, as_of=date.today()),
        run_id=run_id,
        out_root=settings.root / "out" / "research",
        knowledge_root=_knowledge_root(settings),
        rounds=args.rounds,
        resume=bool(args.resume),
        emit=print,
    )
    print(
        f"Зон: {len(result.findings)} ок, {len(result.failed)} упало. Открытых пробелов: {len(result.open_gaps)}"
    )
    return 1 if result.failed and not result.findings else 0


async def _review(args: argparse.Namespace, settings: Settings) -> int:
    deps = build_deps(settings)
    paths = ReviewPaths(
        knowledge_root=_knowledge_root(settings),
        rules_file=settings.root / "rules" / "usn.yaml",
        checklist_file=settings.root / CHECKLIST,
        out_dir=settings.root / "out" / "review",
    )
    result = await run_review(
        deps,
        args.case,
        paths,
        ask=input if sys.stdin.isatty() else None,
        knowledge_run=args.knowledge,
    )
    print(result.report_md)
    print(f"→ {result.report_path}")
    return 0


async def _doctor(settings: Settings) -> int:
    checks = await run_doctor(build_deps(settings))
    for c in checks:
        print(f"{'OK  ' if c.ok else 'FAIL'} {c.name}: {c.detail}")
    return 0 if all(c.ok for c in checks) else 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parser().parse_args(argv)
    settings = _settings(args)
    try:
        if args.cmd == "zones":
            for z in load_zones(settings.root / "research" / "zones.yaml"):
                print(f"{z.id:44} {z.title}")
            return 0
        if args.cmd == "doctor":
            return asyncio.run(_doctor(settings))
        if args.cmd == "research":
            return asyncio.run(_research(args, settings))
        return asyncio.run(_review(args, settings))
    except (LLMError, FileNotFoundError, KeyError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
