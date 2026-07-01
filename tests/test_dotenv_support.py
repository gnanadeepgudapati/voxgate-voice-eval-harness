import os

from eval_system.judges.factory import load_dotenv_if_available


def test_loads_variables_from_a_dotenv_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("VOXGATE_TEST_DOTENV_VAR=hello\n")

    try:
        load_dotenv_if_available(env_file)
        assert os.environ.get("VOXGATE_TEST_DOTENV_VAR") == "hello"
    finally:
        os.environ.pop("VOXGATE_TEST_DOTENV_VAR", None)


def test_missing_dotenv_file_is_a_silent_noop(tmp_path):
    load_dotenv_if_available(tmp_path / "does_not_exist.env")  # must not raise


def test_existing_environment_variable_is_not_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXGATE_TEST_DOTENV_VAR", "from_shell")
    env_file = tmp_path / ".env"
    env_file.write_text("VOXGATE_TEST_DOTENV_VAR=from_dotenv\n")

    load_dotenv_if_available(env_file)

    assert os.environ["VOXGATE_TEST_DOTENV_VAR"] == "from_shell"
