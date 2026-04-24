from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from naviertwin.gui.panels.model_panel import ModelPanel


def test_set_loaded_metadata_restores_nested_engine_surrogate_values(qtbot: object) -> None:
    panel = ModelPanel()
    qtbot.addWidget(panel)

    panel.set_loaded_metadata(
        {
            "engine": {
                "n_modes": 11,
                "surrogate": {
                    "n_params": 4,
                    "n_outputs": 9,
                    "n_samples": 33,
                },
            }
        }
    )

    assert panel._n_params_spin.value() == 4
    assert panel._n_outputs_spin.value() == 9
    assert panel._n_samples_spin.value() == 33


def test_set_loaded_metadata_prefers_top_level_over_nested(qtbot: object) -> None:
    panel = ModelPanel()
    qtbot.addWidget(panel)

    panel.set_loaded_metadata(
        {
            "n_modes": 18,
            "n_params": 6,
            "n_outputs": 7,
            "n_samples": 44,
            "engine": {
                "n_modes": 3,
                "surrogate": {
                    "n_params": 2,
                    "n_outputs": 5,
                    "n_samples": 12,
                },
            },
        }
    )

    assert panel._n_params_spin.value() == 6
    assert panel._n_outputs_spin.value() == 7
    assert panel._n_samples_spin.value() == 44


def test_set_loaded_metadata_invalid_or_non_positive_keeps_defaults(qtbot: object) -> None:
    panel = ModelPanel()
    qtbot.addWidget(panel)

    assert panel._n_samples_spin.value() == 20
    assert panel._n_params_spin.value() == 2
    assert panel._n_outputs_spin.value() == 5

    panel.set_loaded_metadata(
        {
            "n_modes": 0,
            "n_params": -1,
            "n_outputs": "bad",
            "n_samples": None,
            "engine": {
                "n_modes": -2,
                "surrogate": {
                    "n_params": 0,
                    "n_outputs": -3,
                    "n_samples": "NaN",
                },
            },
        }
    )

    assert panel._n_samples_spin.value() == 20
    assert panel._n_params_spin.value() == 2
    assert panel._n_outputs_spin.value() == 5
