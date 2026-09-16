"""Ordinary plans resolve one site locally; cross-site probing is opt-in."""

import json
from unittest.mock import patch

import pytest

from mlipipe.config import load_project
from mlipipe.errors import ConfigError
from mlipipe.services.commands import make_run_plan
from mlipipe.site import load_site_config
from .helpers import write_json
from .test_scheduled_dft import prepared_fixture, FakeTemplateLibrary


@pytest.mark.parametrize("selection", ["explicit", "default", "single", "ambiguous"])
def test_ordinary_planning_never_probes_other_clusters(tmp_path, selection):
    _, site = prepared_fixture(tmp_path)
    config = json.loads((tmp_path / "project.yaml").read_text())
    if selection != "explicit":
        config["workflow"]["nodes"][0].pop("backend_profile")
    write_json(tmp_path / "project.yaml", config)
    settings = json.loads(site.read_text())
    if selection != "single":
        settings["clusters"]["unused"] = {
            **settings["clusters"]["cluster-a"], "ssh_profile": "unused"
        }
    if selection == "default":
        settings["default_profile"] = "cluster-a"
    write_json(site, settings)
    with patch("mlipipe.backends.SshSlurmBackend.select_partition",
               side_effect=AssertionError("unexpected cross-cluster probe")):
        if selection == "ambiguous":
            with pytest.raises(ConfigError, match="default_profile.*backend_profile"):
                make_run_plan(load_project(tmp_path), "label-li", site, FakeTemplateLibrary())
        else:
            plan = make_run_plan(load_project(tmp_path), "label-li", site, FakeTemplateLibrary())
            assert plan["backend_profile"] == "cluster-a"


def test_default_profile_must_exist(tmp_path):
    _, site = prepared_fixture(tmp_path)
    settings = json.loads(site.read_text())
    settings["default_profile"] = "unknown"
    write_json(site, settings)
    with pytest.raises(ConfigError, match="default_profile.*unknown"):
        load_site_config(site)
