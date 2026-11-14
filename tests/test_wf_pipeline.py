"""Pipeline runs keep their configuration, references and outputs together."""
from __future__ import annotations

from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAMP = "20261114_123456_789012"


@pytest.fixture
def prepare_module(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "prepare_wf_run", ROOT / "scripts" / "prepare_wf_run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            assert str(tz) == "Asia/Shanghai"
            return cls(2026, 11, 14, 12, 34, 56, 789012, tzinfo=tz)

    monkeypatch.setattr(module, "datetime", FixedDatetime)
    return module


def write_config(directory):
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "output": str(directory / "training output"),
        "resume": "/previous/ckpt_latest.pt",
        "warm_start": "/previous/warm_start.pt",
        "warm_start_actor": True,
        "demonstrations": {
            "directory": "old/reference/library",
            "pretrain_updates": 123,
        },
        "policy": {"hidden": [128, 64]},
    }
    path = directory / "source config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path, config


def test_prepare_run_saves_synchronized_paths_and_preserves_source(prepare_module, tmp_path):
    source, original = write_config(tmp_path)
    original_bytes = source.read_bytes()

    snapshot = prepare_module.prepare_run(source, tmp_path / "reference data")

    assert isinstance(snapshot, Path)
    config = json.loads(snapshot.read_text(encoding="utf-8"))
    output = tmp_path / f"training output_{STAMP}"
    references = tmp_path / f"reference data_{STAMP}"
    assert snapshot == output / "pipeline_config.json"
    assert output.is_dir() and references.is_dir()
    assert config["output"] == str(output)
    assert config["demonstrations"]["directory"] == str(references / "library")
    expected = json.loads(json.dumps(original))
    expected["output"] = str(output)
    expected["demonstrations"]["directory"] = str(references / "library")
    expected["resume"] = None
    assert config == expected
    assert source.read_bytes() == original_bytes


def test_prepare_run_same_timestamp_never_overwrites(prepare_module, tmp_path):
    source, _ = write_config(tmp_path)
    first = prepare_module.prepare_run(source, tmp_path / "refs")
    first_content = first.read_bytes()
    marker = first.parent / "keep.txt"
    marker.write_text("first run", encoding="utf-8")

    second = prepare_module.prepare_run(source, tmp_path / "refs")

    assert second != first
    assert first.read_bytes() == first_content
    assert marker.read_text(encoding="utf-8") == "first run"
    config = json.loads(second.read_text(encoding="utf-8"))
    assert second.parent.name == f"training output_{STAMP}_1"
    assert Path(config["demonstrations"]["directory"]).parent.name == f"refs_{STAMP}_1"


def test_prepare_run_retries_reference_collision_without_touching_data(prepare_module, tmp_path):
    source, _ = write_config(tmp_path)
    existing_refs = tmp_path / f"refs_{STAMP}"
    existing_refs.mkdir()
    marker = existing_refs / "keep.txt"
    marker.write_text("existing references", encoding="utf-8")

    snapshot = prepare_module.prepare_run(source, tmp_path / "refs")

    config = json.loads(snapshot.read_text(encoding="utf-8"))
    assert snapshot.parent.name == f"training output_{STAMP}_1"
    assert Path(config["demonstrations"]["directory"]).parent.name == f"refs_{STAMP}_1"
    assert marker.read_text(encoding="utf-8") == "existing references"
    assert list(existing_refs.iterdir()) == [marker]
    assert not (tmp_path / f"training output_{STAMP}").exists()


@pytest.fixture
def fake_python(tmp_path):
    """Run the real shell orchestration without importing simulation or training."""
    directory = tmp_path / "bin"
    directory.mkdir()
    wrapper = directory / "python"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """\
            import json
            import os
            from pathlib import Path
            import sys

            args = sys.argv[1:]
            if not args or args[0] != "-m":
                os.execv(os.environ["WF_TEST_REAL_PYTHON"],
                         [os.environ["WF_TEST_REAL_PYTHON"], *args])

            module, args = args[1], args[2:]

            def option(name):
                return args[args.index(name) + 1]

            record = {"module": module, "args": args}
            if "--config" in args:
                record["config_path"] = option("--config")
                record["config"] = json.loads(Path(option("--config")).read_text())
            with open(os.environ["WF_TEST_LOG"], "a") as stream:
                stream.write(json.dumps(record) + "\\n")

            if module == "supermimic.wf.command_reference":
                config = record["config"]
                library = Path(option("--output")) / "library"
                assert Path(config["demonstrations"]["directory"]) == library
                library.mkdir(parents=True)
                (library / "manifest.json").write_text("{}")
            elif module == "supermimic.wf.goal_trajectory":
                output = Path(option("--output"))
                assert Path(record["config"]["trajectory"]["library"]) == output
                output.mkdir(parents=True)
                (output / "manifest.json").write_text("{}")
            elif module == "supermimic.wf.train":
                config = record["config"]
                assert (Path(config["demonstrations"]["directory"]) / "manifest.json").is_file()
                output = Path(config["output"])
                output.mkdir(parents=True, exist_ok=True)
                (output / "ckpt_latest.pt").write_text("fake checkpoint")
            elif module == "supermimic.wf.evaluate":
                assert Path(option("--checkpoint")).is_file()
                if "--reference-library" in args:
                    assert (Path(option("--reference-library")) / "manifest.json").is_file()
                Path(option("--output")).write_text("{}")
            elif module == "supermimic.wf.export":
                assert Path(option("--checkpoint")).is_file()
                Path(option("--output")).write_text("fake policy")
            else:
                raise AssertionError("Unexpected module: " + module)
            """
        ),
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = dict(os.environ)
    env.update(
        PATH=str(directory) + os.pathsep + env.get("PATH", ""),
        WF_TEST_REAL_PYTHON=sys.executable,
        WF_TEST_LOG=str(log),
        REF_COUNT="2",
        WORKERS="1",
    )
    return env, log


def run_pipeline(first, last, env):
    return subprocess.run(
        ["bash", "scripts/wf_cmd_pipeline.sh", str(first), str(last)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=True,
    )


def read_calls(log):
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def argument(record, name):
    args = record["args"]
    return args[args.index(name) + 1]


def test_shell_pipeline_new_runs_keep_all_stages_on_their_own_paths(fake_python, tmp_path):
    env, log = fake_python
    source, _ = write_config(tmp_path / "runs")
    original_bytes = source.read_bytes()
    env.update(CONFIG=str(source), REFS=str(tmp_path / "references with spaces"))

    run_pipeline(1, 4, env)
    first_calls = read_calls(log)
    run_pipeline(1, 4, env)
    second_calls = read_calls(log)[len(first_calls):]

    configs = []
    for calls in (first_calls, second_calls):
        references = [call for call in calls if call["module"].endswith(".command_reference")]
        training = [call for call in calls if call["module"].endswith(".train")]
        evaluation = [call for call in calls if call["module"].endswith(".evaluate")]
        export = [call for call in calls if call["module"].endswith(".export")]
        assert len(references) == len(training) == len(export) == 1
        assert len(evaluation) == 8
        reference, train = references[0], training[0]
        assert reference["config_path"] == train["config_path"]
        assert reference["config"] == train["config"]
        assert "--resume" not in train["args"]
        config = train["config"]
        configs.append(config)
        output = Path(config["output"])
        library = Path(argument(reference, "--output")) / "library"
        assert Path(config["demonstrations"]["directory"]) == library
        assert Path(train["config_path"]) == output / "pipeline_config.json"
        assert config["resume"] is None
        assert config["warm_start"] == "/previous/warm_start.pt"
        for call in [*evaluation, *export]:
            assert Path(argument(call, "--checkpoint")) == output / "ckpt_latest.pt"
            assert Path(argument(call, "--output")).parent == output
            if "--reference-library" in call["args"]:
                assert Path(argument(call, "--reference-library")) == library
        assert (output / "train.log").is_file()
        assert (output / "policy.pt").is_file()

    assert configs[0]["output"] != configs[1]["output"]
    assert configs[0]["demonstrations"]["directory"] != configs[1]["demonstrations"]["directory"]
    assert source.read_bytes() == original_bytes


def test_shell_pipeline_later_stages_reuse_selected_config(fake_python, tmp_path):
    env, log = fake_python
    source, _ = write_config(tmp_path / "runs")
    env.update(CONFIG=str(source), REFS=str(tmp_path / "refs"))
    run_pipeline(1, 2, env)
    calls = read_calls(log)
    train = next(call for call in calls if call["module"].endswith(".train"))
    snapshot = Path(train["config_path"])
    before = snapshot.read_bytes()
    env["CONFIG"] = str(snapshot)

    run_pipeline(2, 4, env)

    later_calls = read_calls(log)[len(calls):]
    assert not any(call["module"].endswith(".command_reference") for call in later_calls)
    later_train = next(call for call in later_calls if call["module"].endswith(".train"))
    assert later_train["config_path"] == str(snapshot)
    checkpoint = snapshot.parent / "ckpt_latest.pt"
    assert Path(argument(later_train, "--resume")) == checkpoint
    for call in later_calls:
        if "--checkpoint" in call["args"]:
            assert Path(argument(call, "--checkpoint")) == checkpoint
    assert snapshot.read_bytes() == before


def test_shell_pipeline_generates_trajectory_library_next_to_references(fake_python, tmp_path):
    env, log = fake_python
    source, config = write_config(tmp_path / "runs")
    config["trajectory"] = {"library": "old/trajectories", "segment_steps": 250}
    source.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    env.update(CONFIG=str(source), REFS=str(tmp_path / "refs"))
    run_pipeline(1, 2, env)
    calls = read_calls(log)
    reference = next(call for call in calls if call["module"].endswith(".command_reference"))
    trajectory = next(call for call in calls if call["module"].endswith(".goal_trajectory"))
    train = next(call for call in calls if call["module"].endswith(".train"))
    references = Path(argument(reference, "--output"))
    assert Path(argument(trajectory, "--output")) == references / "trajectories"
    assert Path(train["config"]["trajectory"]["library"]) == references / "trajectories"
    assert train["config"]["trajectory"]["segment_steps"] == 250
