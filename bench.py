#!/usr/bin/env python3
"""BFCL v3 tool-name routing benchmark via OpenAI GPT-6 Luna.

Downloads the public BFCL dataset into --data-dir when missing, then runs the
comparison using the OpenAI SDK. This is NOT official BFCL scoring: arguments,
real tools, and environmental state are not executed. No Exact Match metric.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

BFCL_REPO = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/"
DOCS = ("gorilla_file_system", "math_api", "message_api", "posting_api",
        "ticket_api", "trading_bot", "travel_booking", "vehicle_control")
SOURCES = ("multiple", "parallel_multiple", "multi_turn_base",
           "multi_turn_composite", "multi_turn_long_context",
           "multi_turn_miss_func", "multi_turn_miss_param")
PROMPT_VERSION = "gpt6-luna-v4-auto-bfcl-no-em"
HF_REPO_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


def required_files():
    for stem in SOURCES:
        yield f"BFCL_v3_{stem}.json"
        yield f"possible_answer/BFCL_v3_{stem}.json"
    for stem in DOCS:
        yield f"multi_turn_func_doc/{stem}.json"


def resolve_local_file(root: Path, rel: str) -> Path | None:
    direct = root / rel
    if direct.is_file():
        return direct
    # Also accept a manually downloaded flat directory; do NOT resolve outside root.
    hits = [p for p in root.rglob(Path(rel).name) if p.is_file()] if root.is_dir() else []
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError(f"Ambiguous local file '{rel}': {hits}. Keep BFCL folder structure.")
    return None


def check_data_dir(root: Path) -> list[str]:
    return [rel for rel in required_files() if resolve_local_file(root, rel) is None]



def ensure_dataset(root: Path, revision: str = "main", download: bool = True, force: bool = False,
                   workers: int = 6) -> None:
    """Download only required public dataset files, preserving BFCL folder structure.

    Files already present remain untouched unless force=True. Hugging Face stores
    local metadata to allow repeat runs without full downloads.
    """
    missing = check_data_dir(root)
    if not missing and not force:
        print(f"BFCL files ready locally: {root.resolve()}", flush=True)
        return
    if not download:
        raise FileNotFoundError(f"BFCL missing {len(missing)} files under {root}. "
                                "Remove --no-download or run --download-only.")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install dataset dependency: pip install huggingface_hub") from exc
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading BFCL V3 ({len(missing)} missing / {len(tuple(required_files()))} total) "
          f"from {HF_REPO_ID}@{revision} ...", flush=True)
    try:
        snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset", revision=revision,
                          allow_patterns=list(required_files()), local_dir=str(root),
                          max_workers=workers, force_download=force)
    except Exception as exc:
        raise RuntimeError(f"BFCL download failed: {exc}. "
                           "Check network/HF_TOKEN, or prepare --data-dir locally.") from exc
    still_missing = check_data_dir(root)
    if still_missing:
        raise FileNotFoundError(f"BFCL incomplete: {still_missing}")
    print("BFCL download complete.", flush=True)

def read_jsonl(root: Path, rel: str) -> list[dict]:
    path = resolve_local_file(root, rel)
    if path is None:
        raise FileNotFoundError(f"Missing BFCL file: {root / rel}\nDataset URL: {BFCL_REPO + rel}")
    rows = []
    for num, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSONL {path}:{num}: {e}") from e
        if not isinstance(item, dict):
            raise ValueError(f"Expected JSON object at {path}:{num}")
        rows.append(item)
    return rows


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def digest(obj):
    return hashlib.sha256(dump(obj).encode("utf8")).hexdigest()[:20]


def stable_seed(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def bare(name):
    return str(name).split(".")[-1]


def call_name(expr):
    m = re.match(r"^\s*([\w.]+)\s*\(", expr)
    return m.group(1) if m else None


def gold_names(ground_truth):
    """Get names only. Ground truth argument values are intentionally not scored."""
    result = []
    for item in (ground_truth if isinstance(ground_truth, list) else []):
        if isinstance(item, dict):
            result.extend(item.keys())
        elif isinstance(item, str):
            name = call_name(item)
            if name:
                result.append(name)
    return result


def user_turns(question):
    turns = []
    for item in (question if isinstance(question, list) else []):
        if isinstance(item, list):
            msgs = item
        elif isinstance(item, dict):
            msgs = [item]
        else:
            msgs = []
        contents = [m.get("content", "") for m in msgs if m.get("role") == "user"]
        if contents:
            turns.append("\n".join(str(s) for s in contents))
    return turns


@dataclass
class Case:
    id: str
    dialog: list[str]
    gold: list[str]
    native: list[dict]
    scope: str
    family: str

    def query(self):
        # Prior user messages only, with no replay of tool outputs/state.
        return "\n".join(f"User turn {i+1}: {text}" for i, text in enumerate(self.dialog))


def map_gold(case: Case, candidates: list[dict]) -> list[str] | None:
    available = {d["name"] for d in candidates}
    mapped = []
    for name in case.gold:
        if name in available:
            mapped.append(name)
            continue
        names = [x for x in available if bare(x) == bare(name)]
        if len(names) != 1:
            return None
        mapped.append(names[0])
    return mapped


def load_cases(data_dir: Path):
    questions = {split: read_jsonl(data_dir, f"BFCL_v3_{split}.json") for split in SOURCES}
    answers = {split: {x["id"]: x["ground_truth"] for x in
               read_jsonl(data_dir, f"possible_answer/BFCL_v3_{split}.json")}
               for split in SOURCES}
    # BFCL multiturn docs use unqualified names (e.g., "ls"), whereas test
    # paths are qualified (e.g., "GorillaFileSystem.ls"). Preserve namespace
    # to avoid collisions such as same-named methods in different services.
    class_names = {
        "gorilla_file_system": "GorillaFileSystem",
        "math_api": "MathAPI",
        "message_api": "MessageAPI",
        "posting_api": "TwitterAPI",
        "ticket_api": "TicketAPI",
        "trading_bot": "TradingBot",
        "travel_booking": "TravelBooking",
        "vehicle_control": "VehicleControl",
    }
    docs_by_class = {}
    by_bare = {}
    for stem in DOCS:
        for doc in read_jsonl(data_dir, f"multi_turn_func_doc/{stem}.json"):
            docs_by_class[(class_names[stem], bare(doc["name"]))] = doc
            by_bare.setdefault(bare(doc["name"]), []).append(doc)
    registry = {}
    for split in ("multiple", "parallel_multiple"):
        for row in questions[split]:
            for d in row.get("function", []):
                registry.setdefault(d["name"], d)
    for (cls, method), doc in docs_by_class.items():
        qualified = {**doc, "name": f"{cls}.{method}"}
        registry.setdefault(qualified["name"], qualified)

    groups = {name: [] for name in ("route_single", "route_multi", "parallel_single",
                                    "parallel_multi", "sequence_single", "sequence_multi")}
    seen = Counter()
    for split, rows in questions.items():
        for row in rows:
            gt = answers[split].get(row["id"])
            if gt is None:
                continue
            dialog = user_turns(row.get("question", []))
            if not dialog:
                continue
            if not split.startswith("multi_turn"):
                names = gold_names(gt)
                native = row.get("function", [])
                if split == "multiple" and len(names) == 1 and len(native) >= 2:
                    groups["route_single"].append(Case(row["id"], dialog, names, native, "single", split))
                elif split == "parallel_multiple" and len(set(names)) >= 2:
                    groups["parallel_single"].append(Case(row["id"], dialog, names, native, "single", split))
                continue
            paths = row.get("path", [])
            native = []
            for path in paths:
                cls, _, method = path.rpartition(".")
                doc = docs_by_class.get((cls, method))
                if doc is None and len(by_bare.get(method, [])) == 1:
                    doc = by_bare[method][0]
                doc = doc or {}
                native.append({"name": path, "description": doc.get("description",
                               f"Call {path} to perform the {bare(path).replace('_', ' ')} operation"),
                               "parameters": doc.get("parameters", {"properties": {}})})
            if not isinstance(gt, list):
                continue
            for turn_index, per_turn in enumerate(gt):
                if turn_index >= len(dialog):
                    break
                names = gold_names(per_turn)
                if not names:
                    continue
                case = Case(f"{row['id']}_t{turn_index}", dialog[:turn_index+1], names,
                            native, "multi", split)
                if turn_index >= 1 and len(names) == 1:
                    groups["route_multi"].append(case)
                if len(set(names)) >= 2:
                    if turn_index >= 1:
                        groups["parallel_multi"].append(case)
                        groups["sequence_multi"].append(case)
                    else:
                        groups["sequence_single"].append(
                            Case(case.id, case.dialog, names, native, "single", split))

    for name in groups:
        before = len(groups[name])
        groups[name] = [c for c in groups[name] if map_gold(c, c.native) is not None]
        seen[name] = (before, len(groups[name]))
    return groups, registry, dict(seen)


def original_id(case):
    return case.id.rsplit("_t", 1)[0] if "_t" in case.id else case.id


def sampling(groups, val_n, test_n, seed):
    result = {}
    # Split consistently by conversation ID, across all categories.
    # Within each pool, select at most one turn of a conversation.
    for group, pool in groups.items():
        rng = random.Random(seed + stable_seed(group))
        by_conv = {}
        for c in pool:
            by_conv.setdefault(original_id(c), []).append(c)
        chosen = [rng.choice(x) for x in by_conv.values()]
        validation = [x for x in chosen if stable_seed(f"{seed}|{original_id(x)}") % 10 < 2]
        test = [x for x in chosen if stable_seed(f"{seed}|{original_id(x)}") % 10 >= 2]
        rng.shuffle(validation)
        rng.shuffle(test)
        if len(validation) < val_n or len(test) < test_n:
            raise ValueError(f"{group}: validation {len(validation)}, test {len(test)} available; "
                             f"need --val {val_n}, --test {test_n}. "
                             "Try lower sizes or use more data; no synthetic padding is performed.")
        result[group] = (validation[:val_n], test[:test_n])
    return result


def tool_document(d):
    props = d.get("parameters", {}).get("properties", {}) or {}
    if not isinstance(props, dict):
        props = {}
    args = ", ".join(f"{k}: {v.get('description', '')[:70]}" for k, v in props.items()
                     if isinstance(v, dict))
    return (f"{d['name'].replace('_', ' ').replace('.', ' ')}. "
            f"{d.get('description', '')[:360]} Arguments: {args[:240]}")


def create_catalog(case, registry, distractors):
    native = {d["name"]: d for d in case.native}
    basename = {bare(name) for name in native}
    others = [d for name, d in registry.items() if name not in native and bare(name) not in basename]
    rng = random.Random(stable_seed(case.id + ":candidate-tools"))
    rng.shuffle(others)
    for d in others[:distractors]:
        native[d["name"]] = d
    result = list(native.values())
    rng.shuffle(result)
    return result


TOKEN_PATTERN = re.compile(r"[a-z]+|[0-9]+", re.I)


def tokens(value):
    return TOKEN_PATTERN.findall(re.sub(r"([a-z])([A-Z])", r"\1 \2", value).lower())


class BM25:
    def __init__(self, docs):
        self.docs = [tokens(s) for s in docs]
        self.df = Counter(w for d in self.docs for w in set(d))
        self.avg_len = sum(map(len, self.docs)) / max(1, len(self.docs))
        self.term_counts = [Counter(d) for d in self.docs]

    def scores(self, query):
        n = len(self.docs)
        results = []
        for doc, freq in zip(self.docs, self.term_counts):
            score = 0.
            for w in set(tokens(query)):
                if w not in freq:
                    continue
                idf = math.log(1 + (n-self.df[w]+.5)/(self.df[w]+.5))
                score += idf * freq[w] * 2.2 / (
                    freq[w] + 1.2 * (.25 + .75 * len(doc) / max(1, self.avg_len)))
            results.append(score)
        return results


def retrieve(case, catalog, k):
    scores = BM25([tool_document(d) for d in catalog]).scores(case.query())
    order = sorted(range(len(catalog)), key=lambda i: (-scores[i], i))[:k]
    return [catalog[i] for i in order]


def clean_response(raw):
    # Some local reasoning models return hidden/visible thinking tags.
    return re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()


def parse_ids(raw, style, kind):
    raw = clean_response(raw)
    if style == "num" and kind == "one":
        match = re.fullmatch(r"(\d+)", raw)
        return ([int(match.group(1))] if match else []), ""
    try:
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        match = re.search(r"\{.*\}|\[.*\]", raw, flags=re.S)
        obj = json.loads(match.group() if match else raw)
        if isinstance(obj, dict):
            ids = obj.get("steps", obj.get("ids", obj.get("id", [])))
            query = obj.get("query", "")
        else:
            ids, query = obj, ""
        if not isinstance(ids, list):
            ids = [ids]
        result = [int(x) for x in ids if isinstance(x, int) or (isinstance(x, str) and x.isdigit())]
        return result, str(query or "")
    except (ValueError, TypeError, AttributeError):
        return [], ""


def p95(values):
    if not values:
        return 0.
    sorted_values = sorted(values)
    return sorted_values[min(len(values)-1, math.ceil(.95 * len(values))-1)]


def f1_of_sets(pred, truth):
    intersection = len(set(pred) & set(truth))
    precision = intersection/len(set(pred)) if pred else 0.
    recall = intersection/len(set(truth)) if truth else 0.
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.
    return precision, recall, f1


def ordered_lcs_recall(pred, truth):
    dp = [0] * (len(truth)+1)
    for x in pred:
        nxt = [0]
        for j, y in enumerate(truth, 1):
            nxt.append(dp[j-1]+1 if x == y else max(nxt[-1], dp[j]))
        dp = nxt
    return dp[-1]/len(truth) if truth else 0.


class Runner:
    def __init__(self, args, registry):
        self.args = args
        self.registry = registry
        self.local = threading.local()
        self.lock = threading.Lock()
        self.catalogs = {}
        self.pred_file = Path(args.out) / "predictions.jsonl"
        self.saved = {}
        if self.pred_file.exists():
            for line in self.pred_file.read_text(encoding="utf8").splitlines():
                try:
                    row = json.loads(line)
                    if row.get("key"):
                        self.saved[row["key"]] = row
                except (ValueError, KeyError):
                    continue

    def client(self):
        if not hasattr(self.local, "client"):
            from openai import OpenAI
            self.local.client = OpenAI(
                api_key=self.args.api_key or os.getenv("OPENAI_API_KEY"),
                timeout=self.args.timeout, max_retries=self.args.api_retries)
        return self.local.client

    def call(self, system, prompt, max_tokens):
        begin = time.perf_counter()
        try:
            kwargs = {"model": self.args.model,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": prompt}],
                      "reasoning_effort": self.args.effort,
                      "max_completion_tokens": max_tokens}
            if self.args.temperature is not None:
                kwargs["temperature"] = self.args.temperature
            resp = self.client().chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            raw = msg.content or ""
            usage = resp.usage
            return (raw, time.perf_counter()-begin,
                    getattr(usage, "prompt_tokens", 0) or 0,
                    getattr(usage, "completion_tokens", 0) or 0, None)
        except Exception as e:
            return "", time.perf_counter()-begin, 0, 0, f"{type(e).__name__}: {str(e)[:200]}"

    def catalog_for(self, case):
        with self.lock:
            if case.id not in self.catalogs:
                self.catalogs[case.id] = create_catalog(case, self.registry, self.args.distractors)
            return self.catalogs[case.id]

    def eval_one(self, case, mode, k, style, stage):
        started = time.perf_counter()
        catalog = self.catalog_for(case)
        truth = map_gold(case, catalog)
        if truth is None:
            raise RuntimeError(f"{case.id}: ground truth tool not found in candidate catalogue")
        kind = "one" if stage in ("route", "react") else ("set" if stage == "parallel" else "sequence")
        selected = retrieve(case, catalog, k) if mode == "retrieval" else catalog
        signature = {"ver": PROMPT_VERSION, "id": case.id, "query": case.query(),
                     "truth": truth, "catalog": [tool_document(d) for d in catalog],
                     "mode": mode, "k": k, "style": style, "stage": stage,
                     "model": self.args.model, "effort": self.args.effort,
                     "temperature": self.args.temperature,
                     "max_output_tokens": self.args.max_output_tokens,
                     "recovery_ratio": self.args.recovery_ratio}
        key = digest(signature)
        with self.lock:
            if key in self.saved:
                return self.saved[key]

        docs = "\n".join(f"{i+1}. {tool_document(doc)}" for i, doc in enumerate(selected))
        if kind == "one":
            output = ("Output ONLY one integer index, such as 2." if style == "num" else
                      'Output ONLY JSON: {"id":2,"query":"short tool-specific query or arguments intent"}.')
            rule = "Select exactly one tool."
        else:
            field = "ids" if kind == "set" else "steps"
            output = (f'Output ONLY JSON: {{"{field}":[1,3]}}.' if style == "num" else
                      f'Output ONLY JSON: {{"{field}":[1,3],"query":"concise intent"}}.')
            rule = ("Select all distinct required tools; ignore order." if kind == "set" else
                    "Select all tool invocations in EXECUTION ORDER; duplicates are permitted.")
        sys = ("You are a precise tool router. Tool indices must refer ONLY to the supplied list. "
               "Use previous user turns as context; route the MOST RECENT user request. "
               "No explanations. " + rule + " " + output)
        prompt = f"{case.query()}\n\nAVAILABLE TOOLS:\n{docs}"
        raw, llm_sec, in_tokens, out_tokens, error = self.call(
            sys, prompt, 16 if style == "num" and kind == "one" else self.args.max_output_tokens)
        ids, generated_query = parse_ids(raw, style, kind)
        invalid_ids = any(i < 1 or i > len(selected) for i in ids)
        pred = [selected[i-1]["name"] for i in ids if 1 <= i <= len(selected)]
        if kind == "one":
            pred = pred[:1]
        if kind == "set":
            pred = list(dict.fromkeys(pred))
        first_pred = pred.copy()
        initial_correct = int(len(first_pred) == 1 and len(truth) == 1 and first_pred[0] == truth[0]) if kind == "one" else None
        react_triggered = False

        # Simulated observation (tool *metadata* relevance, not real tool execution).
        if stage == "react":
            scores = BM25([tool_document(x) for x in catalog]).scores(case.query())
            by_name = dict(zip([x["name"] for x in catalog], scores))
            peak = max(scores, default=0.)
            low_relevance = peak > 0 and (
                by_name.get(pred[0], 0.) < self.args.recovery_ratio * peak if pred else True)
            if not pred or invalid_ids or low_relevance:
                react_triggered = True
                expanded = (retrieve(case, catalog, min(len(catalog), max(k*2, k+4)))
                            if mode == "retrieval" else catalog)
                expanded_text = "\n".join(f"{i+1}. {tool_document(x)}" for i, x in enumerate(expanded))
                observation = (f"OBSERVATION: previous selection {first_pred or 'NONE'} is invalid "
                               "or has low BM25 relevance. Rethink selected tool.\n")
                r2, t2, i2, o2, e2 = self.call(
                    sys, f"{case.query()}\n\n{observation}\nAVAILABLE TOOLS:\n{expanded_text}",
                    16 if style == "num" else self.args.max_output_tokens)
                ids2, query2 = parse_ids(r2, style, "one")
                if ids2 and 1 <= ids2[0] <= len(expanded):
                    pred = [expanded[ids2[0]-1]["name"]]
                    generated_query = query2 or generated_query
                llm_sec += t2
                in_tokens += i2
                out_tokens += o2
                error = error or e2

        accuracy = int(len(pred) == 1 and len(truth) == 1 and pred[0] == truth[0]) if kind == "one" else None
        p, recall, f1 = f1_of_sets(pred, truth)
        candidate_recall = f1_of_sets([x["name"] for x in selected], truth)[1]
        record = {"key": key, "id": case.id, "scope": case.scope, "family": case.family,
                  "stage": stage, "style": style, "mode": mode, "k": k,
                  "kind": kind, "gold": truth, "pred": pred,
                  "accuracy": accuracy,
                  "precision": p, "recall": recall, "instance_f1": f1,
                  "lcs_recall": ordered_lcs_recall(pred, truth) if kind == "sequence" else None,
                  "candidate_recall": candidate_recall,
                  "latency_sec": time.perf_counter()-started, "llm_sec": llm_sec,
                  "input_tokens": in_tokens, "output_tokens": out_tokens,
                  "query_generated": generated_query, "query_nonempty": int(bool(generated_query.strip())),
                  "react_triggered": int(react_triggered), "initial_correct": initial_correct if stage == "react" else None,
                  "recovered": int(stage == "react" and initial_correct == 0 and accuracy == 1),
                  "regressed": int(stage == "react" and initial_correct == 1 and accuracy == 0),
                  "llm_calls": 1+int(react_triggered), "n_candidates": len(selected),
                  "invalid_ids": int(invalid_ids), "raw_output": raw[:400], "error": error}
        with self.lock:
            if key not in self.saved:
                with self.pred_file.open("a", encoding="utf8") as f:
                    f.write(dump(record)+"\n")
                self.saved[key] = record
        return record

    def evaluate(self, cases, mode, k, style, stage):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.concurrency) as pool:
            return list(pool.map(lambda c: self.eval_one(c, mode, k, style, stage), cases))


def summarize(rows):
    n = len(rows)
    if not n:
        raise ValueError("Cannot summarize an empty evaluation set")
    mean = lambda key: sum((row[key] or 0) for row in rows)/n
    bad_initial = sum(1 for row in rows if row["stage"] == "react" and row["initial_correct"] == 0)
    return {"n": n, "api_ok": sum(row["error"] is None for row in rows),
            "accuracy": mean("accuracy") if rows[0]["kind"] == "one" else None,
            "mean_instance_precision": mean("precision"),
            "mean_instance_recall": mean("recall"),
            "mean_instance_f1": mean("instance_f1"),
            "candidate_recall": mean("candidate_recall"),
            "latency_mean_s": mean("latency_sec"),
            "latency_median_s": statistics.median(r["latency_sec"] for r in rows),
            "latency_p95_s": p95([r["latency_sec"] for r in rows]),
            "input_tokens_mean": mean("input_tokens"),
            "output_tokens_mean": mean("output_tokens"),
            "query_nonempty_rate": mean("query_nonempty"),
            "react_trigger_rate": mean("react_triggered"),
            "react_recovery_rate": sum(r["recovered"] for r in rows)/bad_initial if bad_initial else 0.,
            "react_regression_rate": mean("regressed"),
            "llm_calls_mean": mean("llm_calls"),
            "sequence_lcs_recall": mean("lcs_recall")}


def main():
    p = argparse.ArgumentParser(description="OpenAI GPT-6 Luna + automatically downloaded BFCL V3 router benchmark")
    p.add_argument("--data-dir", type=Path, default=Path("./bfcl_data"))
    p.add_argument("--list-urls", action="store_true", help="Print dataset source URLs")
    p.add_argument("--check-data", action="store_true", help="Check local files without downloading")
    p.add_argument("--download-only", action="store_true", help="Download dataset and exit; no API call")
    p.add_argument("--no-download", action="store_true", help="Use existing dataset only")
    p.add_argument("--force-download", action="store_true", help="Refresh files even if present")
    p.add_argument("--revision", default="main", help="Hugging Face dataset revision (commit SHA or branch)")
    p.add_argument("--hf-workers", type=int, default=6)
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--effort", choices=["none", "low", "medium", "high", "xhigh", "max"],
                   default="none", help="GPT-6 Luna reasoning_effort")
    p.add_argument("--temperature", type=float, default=None,
                   help="Optional, only use with --effort none")
    p.add_argument("--api-retries", type=int, default=2)
    p.add_argument("--test", type=int, default=100)
    p.add_argument("--val", type=int, default=25)
    p.add_argument("--topks", default="3,5,8,12")
    p.add_argument("--distractors", type=int, default=24)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--max-output-tokens", type=int, default=128)
    p.add_argument("--recovery-ratio", type=float, default=.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_luna")
    args = p.parse_args()

    if args.list_urls:
        for rel in required_files():
            print(f"{rel}\n  {BFCL_REPO}{rel}")
        return
    if args.check_data:
        missing = check_data_dir(args.data_dir)
        print(f"Data directory: {args.data_dir.resolve()}")
        print(f"Required files: {len(tuple(required_files()))}; missing: {len(missing)}")
        for rel in missing:
            print(f"MISSING {rel}\n  {BFCL_REPO}{rel}")
        return
    if args.effort != "none" and args.temperature is not None:
        p.error("--temperature can only be set with --effort none")
    if args.hf_workers < 1:
        p.error("--hf-workers must be >=1")
    ensure_dataset(args.data_dir, revision=args.revision,
                   download=not args.no_download, force=args.force_download,
                   workers=args.hf_workers)
    if args.download_only:
        return
    if not args.api_key:
        p.error("Set OPENAI_API_KEY (or pass --api-key) to run GPT-6 Luna")
    if args.test < 1 or args.val < 1 or args.concurrency < 1:
        p.error("--test, --val, and --concurrency must all be >=1")
    try:
        topks = sorted({int(k) for k in args.topks.split(",") if k.strip()})
    except ValueError:
        p.error("--topks must be comma-separated positive integers")
    if not topks or any(k <= 0 for k in topks):
        p.error("Specify at least one positive --topks value")

    # Unlike original code, no API/model call before local files are validated.
    groups, registry, pool_stats = load_cases(args.data_dir)
    samples = sampling(groups, args.val, args.test, args.seed)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    runner = Runner(args, registry)
    configs = [("llm_only", 0)] + [("retrieval", k) for k in topks]
    records = []

    def run(phase, stage, scope, cases, mode, k, style):
        results = runner.evaluate(cases, mode, k, style, stage)
        record = {"phase": phase, "stage": stage, "scope": scope, "mode": mode,
                  "k": k, "style": style, **summarize(results)}
        records.append(record)
        acc = f"{record['accuracy']:.3f}" if record['accuracy'] is not None else "n/a"
        print(f"{phase:10} {scope:6} {stage:8} {style:5} {mode:10} k={k:<3} "
              f"Accuracy={acc} F1={record['mean_instance_f1']:.3f} "
              f"p95={record['latency_p95_s']:.2f}s "
              f"API_OK={record['api_ok']}/{record['n']}", flush=True)
        return record

    # Tune on held-out validation, never on the final test sets.
    validation = []
    for mode, k in configs:
        for style in ("num", "joint"):
            for scope in ("single", "multi"):
                validation.append(run("validation", "route", scope,
                                      samples["route_"+scope][0], mode, k, style))
    scored = []
    for mode, k in configs:
        for style in ("num", "joint"):
            rows = [r for r in validation if (r["mode"], r["k"], r["style"]) == (mode, k, style)]
            if any(r["api_ok"] < r["n"] for r in rows):
                continue
            accuracy = statistics.mean(r["accuracy"] for r in rows)
            latency = statistics.mean(r["latency_mean_s"] for r in rows)
            scored.append((accuracy, -latency, style, mode, k))
    if not scored:
        raise RuntimeError("Every validation setting had an API error; see predictions.jsonl")
    _, _, winning_style, winning_mode, winning_k = max(scored)
    print(f"\nVALIDATION WINNER: {winning_style} / {winning_mode} / K={winning_k}\n", flush=True)

    # Test 1/2 independently using each style's own best validation config.
    style_winners = {}
    for style in ("num", "joint"):
        choice = max(row for row in scored if row[2] == style)
        mode, k = choice[3], choice[4]
        style_winners[style] = {"mode": mode, "k": k}
        for scope in ("single", "multi"):
            run("test", "route", scope, samples["route_"+scope][1], mode, k, style)

    # Stages 3/4/5: use validation winner; do not re-tune on test metrics.
    for stage, prefix in (("react", "route"), ("parallel", "parallel"), ("sequence", "sequence")):
        for scope in ("single", "multi"):
            run("test", stage, scope, samples[prefix+"_"+scope][1],
                winning_mode, winning_k, winning_style)

    out = Path(args.out)
    report = {"model": args.model, "effort": args.effort,
              "dataset_repo": HF_REPO_ID, "dataset_revision": args.revision,
              "data_dir": str(args.data_dir.resolve()), "pool_sizes_before_after_filter": pool_stats,
              "selection_on_validation": {"style": winning_style, "mode": winning_mode,
                                          "k": winning_k}, "per_style_best": style_winners,
              "metrics": {"route/react": "single-tool selection accuracy, precision, recall, F1",
                          "parallel": "mean instance precision/recall/F1",
                          "sequence": "mean instance precision/recall/F1 and sequence LCS recall",
                          "retrieval": "mean candidate recall"},
              "limitations": "NOT BFCL official: no argument value scoring, stateful tool execution or query gold; "
                             "ReAct feedback is metadata-based and multiturn is user-history-only.",
              "records": records}
    (out/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    with (out/"summary.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    lines = ["# GPT-6 Luna Tool Router Benchmark", "",
             f"Model: `{args.model}` | Reasoning effort: `{args.effort}`",
             f"Validation winner: `{winning_style}` / `{winning_mode}` / K={winning_k}", "",
             "|Phase|Stage|Scope|Style|Mode|K|N|Accuracy|Precision|Recall|Instance F1|Candidate Recall|LCS Recall|p95 sec|",
             "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        accuracy_text = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "–"
        lines.append(f"|{r['phase']}|{r['stage']}|{r['scope']}|{r['style']}|{r['mode']}|"
                     f"{r['k']}|{r['n']}|{accuracy_text}|{r['mean_instance_precision']:.3f}|"
                     f"{r['mean_instance_recall']:.3f}|{r['mean_instance_f1']:.3f}|"
                     f"{r['candidate_recall']:.3f}|{r['sequence_lcs_recall']:.3f}|{r['latency_p95_s']:.3f}|")
    lines.extend(["", "**Validation choice**: highest mean single-tool routing accuracy across "
                  "single-turn and multi-turn; tie-break by mean latency.",
                  "**Metric scope**: Accuracy is shown only for single-tool route/recovery. "
                  "Multi-tool quality uses per-instance precision/recall/F1, "
                  "and combination additionally reports ordered LCS recall.",
                  "**Limitations**: Does not score arguments, live execution, query semantic accuracy, "
                  "or full-state BFCL multiturn. ReAct is metadata-observation-based."])
    (out/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    print(f"\nWrote {out/'summary.md'}, {out/'summary.csv'}, {out/'predictions.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
