import json

from symbiosis_edge.cli import main


def test_info_command_returns_zero(capsys):
    rc = main(["info"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "symbiosis-edge" in out
    assert "Symbiosis-Edge" in out


def test_quick_run_writes_artifacts(tmp_path):
    out = tmp_path / "run"
    rc = main(["run", "--quick", "--out", str(out)])
    assert rc == 0

    manifest = out / "manifest.json"
    summary = out / "summary.csv"
    assert manifest.exists()
    assert summary.exists()

    data = json.loads(manifest.read_text())
    assert data["schema"] == "symbiosis-edge/manifest@1"
    assert data["symbiosis_edge_version"]
    assert set(data["config"]["datasets"]) == {"SECOM", "APS", "SYNTHETIC"}
    # Every recorded output carries a checksum for traceability.
    assert data["outputs"]
    assert all(o["sha256"] for o in data["outputs"])


def test_run_single_dataset_subset(tmp_path):
    out = tmp_path / "secom"
    rc = main(["run", "--datasets", "SECOM", "--seeds", "2", "--n", "300",
               "--no-figures", "--out", str(out)])
    assert rc == 0
    assert (out / "summary.csv").exists()
    assert (out / "tables" / "table_cost_secom.tex").exists()
