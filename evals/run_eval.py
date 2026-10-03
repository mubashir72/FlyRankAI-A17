import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


CASES_PATH = Path(__file__).with_name("cases.json")


def load_cases() -> list[dict[str, str]]:
    with CASES_PATH.open(encoding="utf-8") as cases_file:
        cases: Any = json.load(cases_file)
    if not isinstance(cases, list) or len(cases) != 8:
        raise ValueError("The evaluation file must contain exactly eight cases.")
    for case in cases:
        if not isinstance(case, dict) or not {
            "id",
            "text",
            "expected_category",
        }.issubset(case):
            raise ValueError("Each case needs id, text, and expected_category.")
    return cases


def evaluate(endpoint: str, cases: list[dict[str, str]]) -> int:
    matched = 0
    failures: list[tuple[str, str, str]] = []
    opener = build_opener(ProxyHandler({}))

    for case in cases:
        request = Request(
            endpoint,
            data=json.dumps({"text": case["text"]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with opener.open(request, timeout=300) as response:
                output: Any = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            failures.append((case["id"], case["expected_category"], str(error)))
            continue

        actual = output.get("category") if isinstance(output, dict) else None
        if actual == case["expected_category"]:
            matched += 1
        else:
            failures.append(
                (
                    case["id"],
                    case["expected_category"],
                    str(actual) if actual is not None else "missing category",
                )
            )

    percentage = matched / len(cases) * 100
    print(f"Category accuracy: {matched}/{len(cases)} ({percentage:.1f}%)")
    if failures:
        print("Failures:")
        for case_id, expected, actual in failures:
            print(f"- {case_id}: expected {expected}; got {actual}")
    else:
        print("Failures: none")
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate category accuracy using the eight local cases."
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/triage",
        help="Triage endpoint URL (default: http://127.0.0.1:8000/triage)",
    )
    args = parser.parse_args()
    return evaluate(args.endpoint, load_cases())


if __name__ == "__main__":
    sys.exit(main())
