import csv
import json

from qchunk.tracking import MetricLogger, record_manifest_event


def test_metric_logger_writes_jsonl_and_long_form_csv(tmp_path):
    logger = MetricLogger(tmp_path)
    logger.log("offline", 1, {"loss": 2.5, "q": 1.0})
    logger.log("online", 3, {"return": 4.0})

    records = logger.read_records()
    assert [(record["phase"], record["step"]) for record in records] == [
        ("offline", 1),
        ("online", 3),
    ]
    assert records[0]["metrics"] == {"loss": 2.5, "q": 1.0}

    with logger.csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [(row["phase"], row["step"], row["metric"]) for row in rows] == [
        ("offline", "1", "loss"),
        ("offline", "1", "q"),
        ("online", "3", "return"),
    ]


def test_prepare_phase_truncates_only_records_after_resume_step(tmp_path):
    logger = MetricLogger(tmp_path)
    logger.log("offline", 1, {"loss": 3.0})
    logger.log("online", 1, {"loss": 2.0})
    logger.log("offline", 2, {"loss": 1.0})
    logger.log("offline", 3, {"loss": 0.5})

    logger.prepare_phase("offline", resume_step=2)

    records = logger.read_records()
    assert [(record["phase"], record["step"]) for record in records] == [
        ("offline", 1),
        ("online", 1),
        ("offline", 2),
    ]
    with logger.csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [(row["phase"], row["step"]) for row in rows] == [
        ("offline", "1"),
        ("online", "1"),
        ("offline", "2"),
    ]


def test_manifest_records_runtime_config_and_resume_event(tmp_path, monkeypatch):
    runtime = {"git": {"commit": "abc123", "dirty": False}}
    monkeypatch.setattr("qchunk.tracking.collect_runtime_metadata", lambda _cwd: runtime)
    monkeypatch.setattr("sys.argv", ["train", "--config", "config.yaml"])
    config = {"seed": 0, "offline": {"steps": 10}, "output": {"checkpoint": "x"}}

    path = record_manifest_event(tmp_path, config, "offline")
    record_manifest_event(
        tmp_path,
        config,
        "offline",
        resume_checkpoint="step_5.msgpack",
        resume_step=5,
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["runtime"] == runtime
    assert len(manifest["events"]) == 2
    assert manifest["events"][0]["config"] == config
    assert manifest["events"][1]["resume_step"] == 5
    assert manifest["events"][1]["resume_checkpoint"] == "step_5.msgpack"
