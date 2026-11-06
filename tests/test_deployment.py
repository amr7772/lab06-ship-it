"""The files a Space reads must agree with what is installed and with each other."""

from pathlib import Path

import gradio
import sklearn

import predict

ROOT = Path(__file__).resolve().parent.parent


def readme_settings() -> dict:
    """Read the Space settings block at the very top of README.md."""
    lines = (ROOT / "README.md").read_text().splitlines()
    assert lines[0] == "---", "README.md must begin with the Space settings block"
    end = lines.index("---", 1)
    settings = {}
    for line in lines[1:end]:
        key, _, value = line.partition(":")
        settings[key.strip()] = value.strip()
    return settings


def requirement_names() -> list[str]:
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    return [
        line.split("==")[0].strip().lower()
        for line in lines
        if line.strip() and not line.startswith("#")
    ]


def test_scikit_learn_version_matches_model_info_and_requirements():
    saved_with = predict.INFO["scikit_learn_version"]
    assert sklearn.__version__ == saved_with
    assert f"scikit-learn=={saved_with}" in (ROOT / "requirements.txt").read_text()


def test_gradio_is_not_in_requirements_and_sdk_version_matches():
    assert "gradio" not in requirement_names()
    assert readme_settings()["sdk_version"] == gradio.__version__


def test_app_file_named_in_readme_exists():
    assert (ROOT / readme_settings()["app_file"]).is_file()


def test_space_settings_ask_for_gradio_on_python_3_12():
    settings = readme_settings()
    assert settings["sdk"] == "gradio"
    assert settings["python_version"] == "3.12"


def test_every_requirement_is_pinned_and_has_no_local_path():
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        assert "==" in line
        assert "/" not in line


def test_models_folder_is_not_ignored():
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert "models/" not in ignored


def test_page_has_one_input_per_feature_in_model_info():
    import app

    numeric, categorical = predict.feature_lists()
    assert len(app.INPUTS) == len(numeric) + len(categorical)
