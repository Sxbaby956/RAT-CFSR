from pathlib import Path

from rat_cfsr.snapshot import copy_current_snapshot


def test_copy_current_snapshot_excludes_large_and_generated_dirs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (repo / "rat_cfsr").mkdir()
    (repo / "rat_cfsr" / "train.py").write_text("# train\n", encoding="utf-8")
    (repo / "GlobecomPOWDER").mkdir()
    (repo / "GlobecomPOWDER" / "data.bin").write_text("data", encoding="utf-8")
    (repo / "outputs").mkdir()
    output_root = repo / "outputs" / "rat_cfsr"
    output_root.mkdir()
    (output_root / "metrics.json").write_text("{}", encoding="utf-8")

    snapshot = copy_current_snapshot(repo, output_root)

    assert (snapshot / "code" / "main.py").exists()
    assert (snapshot / "outputs" / "rat_cfsr" / "metrics.json").exists()
    assert not (snapshot / "code" / "GlobecomPOWDER").exists()
    assert not (snapshot / "code" / "outputs").exists()
    assert (snapshot / "SNAPSHOT.txt").exists()
