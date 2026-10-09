"""Offline unit tests. Mock OpenAI API and Hugging Face downloads; no network required.
Run: python test_smoke.py
"""
import argparse
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from bench import (BFCL_REPO, BM25, Case, Runner, check_data_dir, ensure_dataset,
                   f1_of_sets, gold_names, load_cases, map_gold,
                   ordered_lcs_recall, parse_ids, read_jsonl, required_files,
                   retrieve, sampling, summarize, user_turns)


def spec(name, desc="a useful function"):
    return {"name": name, "description": desc, "parameters": {"properties": {}}}


class FakeCompletion:
    def __init__(self, content):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
        self.usage = types.SimpleNamespace(prompt_tokens=23, completion_tokens=2)


class FakeChat:
    def __init__(self, model_result):
        self.model_result = model_result
        self.calls = []
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeCompletion(self.model_result)


class FakeClient:
    def __init__(self, model_result="1"):
        self.chat = FakeChat(model_result)


class ToolRouterTests(unittest.TestCase):
    def test_parser_and_partial_credit_metrics(self):
        self.assertEqual(parse_ids("2", "num", "one"), ([2], ""))
        self.assertEqual(parse_ids('<think>thinking</think>{"id":2,"query":"find csv"}',
                                   "joint", "one"), ([2], "find csv"))
        self.assertEqual(parse_ids('{"ids":[2,3]}', "num", "set"), ([2, 3], ""))
        self.assertEqual(parse_ids('{"steps":[2,1,2]}', "num", "sequence"), ([2,1,2], ""))
        self.assertEqual(f1_of_sets(["B","A"], ["A","B"]), (1., 1., 1.))
        self.assertAlmostEqual(f1_of_sets(["B"], ["A","B"])[1], .5)
        self.assertEqual(f1_of_sets(["A","C"], ["A","B"])[2], .5)
        self.assertAlmostEqual(ordered_lcs_recall(["B","A"], ["A","B"]), .5)
        self.assertEqual(gold_names(['mkdir(dir_name="x")', 'ls(a=True)']), ["mkdir", "ls"])
        self.assertEqual(user_turns([[{"role":"user", "content":"one"}],
                                     [{"role":"user", "content":"two"}]]), ["one", "two"])

    def test_dataset_auto_download_logic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(FileNotFoundError):
                ensure_dataset(root, download=False)
            # Patch the HF module: no actual network request.
            import sys
            fake_module = types.SimpleNamespace()
            invocations = []
            def fake_download(**kwargs):
                invocations.append(kwargs)
                for rel in required_files():
                    dest = root / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text('{"id":"dummy"}\n', encoding="utf8")
            fake_module.snapshot_download = fake_download
            with patch.dict(sys.modules, {"huggingface_hub": fake_module}):
                ensure_dataset(root, revision="mycommit", download=True)
                ensure_dataset(root, revision="mycommit", download=True)
            self.assertEqual(len(invocations), 1)
            self.assertEqual(invocations[0]["repo_type"], "dataset")
            self.assertEqual(invocations[0]["revision"], "mycommit")
            self.assertEqual(invocations[0]["repo_id"], "gorilla-llm/Berkeley-Function-Calling-Leaderboard")
            self.assertEqual(len(check_data_dir(root)), 0)

    def test_local_data_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(len(check_data_dir(root)), len(tuple(required_files())))
            path = root / "possible_answer" / "BFCL_v3_multiple.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"id":"a","ground_truth":[]}', encoding="utf8")
            self.assertEqual(read_jsonl(root, "possible_answer/BFCL_v3_multiple.json")[0]["id"], "a")
            self.assertNotIn("possible_answer/BFCL_v3_multiple.json", check_data_dir(root))
            self.assertIn("resolve/main/", BFCL_REPO)

    def test_load_cases_from_local_bfcl_layout(self):
        # Build the BFCL hierarchy exclusively on disk and load it offline.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for rel in required_files():
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text("", encoding="utf8")
            one = spec("Calculator.add", "add numbers")
            two = spec("Calculator.subtract", "subtract numbers")
            multiple = {"id":"multiple_0", "question":[[{"role":"user", "content":"add 1 and 2"}]],
                        "function":[one, two]}
            m_answer = {"id":"multiple_0", "ground_truth":[{"Calculator.add":{"x":[1]}}]}
            parallel = {"id":"parallel_multiple_0", "question":[[{"role":"user", "content":"add and subtract"}]],
                        "function":[one, two]}
            p_answer = {"id":"parallel_multiple_0", "ground_truth":[{"Calculator.add":{}}, {"Calculator.subtract":{}}]}
            mt = {"id":"multi_turn_base_0", "question":[[{"role":"user", "content":"list"}],
                  [{"role":"user", "content":"copy and move"}], [{"role":"user", "content":"list again"}]],
                  "path":["GorillaFileSystem.ls", "GorillaFileSystem.cp", "GorillaFileSystem.mv"]}
            mt_answer = {"id":"multi_turn_base_0", "ground_truth":[["ls(a=True)", "cp(source='a',destination='b')"],
                         ["cp(source='a',destination='b')", "mv(source='b',destination='c')"], ["ls(a=True)"]]}
            payloads = {
                "BFCL_v3_multiple.json":multiple,
                "possible_answer/BFCL_v3_multiple.json":m_answer,
                "BFCL_v3_parallel_multiple.json":parallel,
                "possible_answer/BFCL_v3_parallel_multiple.json":p_answer,
                "BFCL_v3_multi_turn_base.json":mt,
                "possible_answer/BFCL_v3_multi_turn_base.json":mt_answer,
            }
            for rel, obj in payloads.items():
                (root/rel).write_text(json.dumps(obj)+"\n", encoding="utf8")
            docs = [spec("ls", "list directory"), spec("cp", "copy file"), spec("mv", "move file")]
            (root/"multi_turn_func_doc"/"gorilla_file_system.json").write_text(
                "\n".join(json.dumps(x) for x in docs)+"\n", encoding="utf8")
            pools, registry, counts = load_cases(root)
            self.assertEqual(len(pools["route_single"]), 1)
            self.assertEqual(len(pools["route_multi"]), 1)
            self.assertEqual(len(pools["parallel_single"]), 1)
            self.assertEqual(len(pools["parallel_multi"]), 1)
            self.assertEqual(len(pools["sequence_single"]), 1)
            self.assertEqual(len(pools["sequence_multi"]), 1)
            self.assertIn("Calculator.add", registry)

    def test_bm25_and_map(self):
        documents = ["search financial documents", "book hotel ticket"]
        self.assertGreater(BM25(documents).scores("search documents")[0], 0)
        docs = [spec("GorillaFileSystem.ls", "list all files"), spec("Hotel.book", "book hotel")]
        case = Case("case1", ["list files"], ["ls"], docs, "single", "test")
        self.assertEqual(map_gold(case, docs), ["GorillaFileSystem.ls"])
        self.assertEqual(retrieve(case, docs, 1)[0]["name"], "GorillaFileSystem.ls")

    def test_runner_luna_chat_completions_and_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(out=temp, api_key="fake-secret", model="gpt-6-luna", timeout=2,
                                      temperature=None, effort="none", api_retries=0, distractors=0,
                                      max_output_tokens=80, recovery_ratio=.15, concurrency=1)
            doc = spec("Docs.search", "search documents and find documents")
            case = Case("sample01", ["search documents"], ["Docs.search"], [doc], "single", "synthetic")
            runner = Runner(args, {"Docs.search": doc})
            fake = FakeClient("1")
            runner.local.client = fake
            res = runner.eval_one(case, "llm_only", 0, "num", "route")
            self.assertEqual(res["accuracy"], 1)
            self.assertEqual(res["instance_f1"], 1)
            self.assertEqual(res["candidate_recall"], 1)
            self.assertNotIn("em", res)
            self.assertEqual(res["llm_calls"], 1)
            self.assertEqual(res["input_tokens"], 23)
            metrics = summarize([res])
            self.assertEqual(metrics["accuracy"], 1)
            self.assertEqual(metrics["mean_instance_f1"], 1)
            self.assertNotIn("em", metrics)
            req = fake.chat.calls[0]
            self.assertEqual(req["model"], "gpt-6-luna")
            self.assertNotIn("temperature", req)
            self.assertEqual(req["max_completion_tokens"], 16)
            self.assertEqual(req["reasoning_effort"], "none")
            self.assertNotIn("base_url", req)
            # Cached result survives a new Runner (no re-inference).
            second = Runner(args, {"Docs.search": doc})
            self.assertEqual(second.eval_one(case, "llm_only", 0, "num", "route")["accuracy"], 1)

    def test_openai_sdk_factory_without_local_vllm_url(self):
        import sys
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(out=temp, api_key="sk-test", model="gpt-6-luna",
                                      timeout=20, api_retries=2, distractors=0)
            runner = Runner(args, {})
            observed = {}
            def fake_factory(**kwargs):
                observed.update(kwargs)
                return FakeClient()
            fake_openai = types.SimpleNamespace(OpenAI=fake_factory)
            with patch.dict(sys.modules, {"openai": fake_openai}):
                runner.client()
            self.assertEqual(observed["api_key"], "sk-test")
            self.assertEqual(observed["timeout"], 20)
            self.assertEqual(observed["max_retries"], 2)
            self.assertNotIn("base_url", observed)

    def test_multitool_scores_without_exact_match(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(out=temp, api_key="fake-secret", model="gpt-6-luna", timeout=2,
                                      temperature=None, effort="none", api_retries=0, distractors=0,
                                      max_output_tokens=80, recovery_ratio=.15, concurrency=1)
            a, b = spec("Tool.a", "get a"), spec("Tool.b", "get b")
            case = Case("multi", ["do a and b"], ["Tool.a", "Tool.b"],
                        [a,b], "single", "synthetic")
            runner = Runner(args, {a["name"]:a,b["name"]:b})
            runner.local.client = FakeClient('{"ids":[1]}')
            row = runner.eval_one(case, "llm_only", 0, "num", "parallel")
            self.assertIsNone(row["accuracy"])
            self.assertAlmostEqual(row["instance_f1"], 2/3)
            self.assertAlmostEqual(row["recall"], .5)
            self.assertIsNone(summarize([row])["accuracy"])
            self.assertFalse(any("_em" in key.lower() for key in row))

    def test_conversation_split_global(self):
        docs = [spec("tool")]
        group1, group2 = [], []
        for i in range(160):
            c = Case(f"conv{i}_t1", ["hello", "tool"], ["tool"], docs, "multi", "test")
            group1.append(c)
            group2.append(c)
        samples = sampling({"route_multi": group1, "parallel_multi": group2}, 5, 5, 42)
        # Same original dialogs are always on the same side across groups.
        val = {c.id for name in samples for c in samples[name][0]}
        test = {c.id for name in samples for c in samples[name][1]}
        self.assertFalse(val & test)

    def test_reject_incorrect_gold(self):
        case = Case("z", ["hi"], ["Missing"], [spec("Available")], "single", "test")
        self.assertIsNone(map_gold(case, case.native))


if __name__ == "__main__":
    unittest.main(verbosity=2)
