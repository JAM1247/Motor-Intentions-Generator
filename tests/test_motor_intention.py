from __future__ import annotations

import unittest
import warnings

import numpy as np

from blocks.generators import BandGenerator
from core.types import BandSpec
from motor_intention import MotorIntentionArchitecture, MotorIntentionConfig
from motor_intention.configs import LayoutConfig, TrialConfig
from motor_intention.montages import get_layout
from motor_intention.projection import SourceToSensorProjector
from motor_intention.sources import source_names
from motor_intention.trials import build_trial_schedule
from motor_intention.ui_logic import (
    aggregate_electrode_importance,
    compare_electrode_importance,
    decoder_readiness_frame,
    default_region_for_trial,
    default_visible_channels,
)
from motor_intention.ui_plots import (
    build_decoder_readiness_table,
    plot_channel_delta_bars,
    plot_class_summary_bars,
    plot_class_topography,
    plot_electrode_cluster_map,
    plot_topography_difference,
)
from pipeline import Pipeline


class TestMotorIntention(unittest.TestCase):
    def test_label_schema_and_structured_metadata(self) -> None:
        trials = build_trial_schedule(TrialConfig(n_trials=5))
        self.assertEqual(
            [trial.flat_label for trial in trials],
            ["left_arm", "right_arm", "left_leg", "right_leg", "rest"],
        )
        self.assertEqual(trials[0].effector_family, "arm")
        self.assertEqual(trials[0].side, "left")
        self.assertIsNone(trials[0].joint_subset)
        self.assertEqual(trials[2].effector_family, "leg")
        self.assertEqual(trials[4].effector_family, "rest")
        self.assertIsNone(trials[4].side)

    def test_projection_is_deterministic(self) -> None:
        layout = get_layout("motor_21")
        names = source_names()
        projector_a = SourceToSensorProjector(layout, names, MotorIntentionConfig().projection)
        projector_b = SourceToSensorProjector(layout, names, MotorIntentionConfig().projection)
        np.testing.assert_allclose(
            projector_a.build_mixing_matrix(),
            projector_b.build_mixing_matrix(),
        )

    def test_arm_vs_leg_topography_differs(self) -> None:
        layout = get_layout("motor_21")
        names = source_names()
        matrix = SourceToSensorProjector(
            layout,
            names,
            MotorIntentionConfig().projection,
        ).build_mixing_matrix()
        name_to_idx = {name: idx for idx, name in enumerate(names)}

        left_upper = matrix[layout.index("C3"), name_to_idx["left_upper_limb_motor"]]
        left_upper_mid = matrix[layout.index("Cz"), name_to_idx["left_upper_limb_motor"]]
        self.assertGreater(left_upper, left_upper_mid)

        left_lower_mid = matrix[layout.index("Cz"), name_to_idx["left_lower_limb_motor"]]
        left_lower_lat = matrix[layout.index("C3"), name_to_idx["left_lower_limb_motor"]]
        self.assertGreater(left_lower_mid, left_lower_lat)

    def test_event_epoch_alignment(self) -> None:
        config = MotorIntentionConfig(
            include_eog=False,
            include_emg=False,
            include_line_noise=False,
        )
        result = MotorIntentionArchitecture(config).run()
        self.assertEqual(result.epochs.shape[0], len(result.trials))
        self.assertEqual(len(result.labels), len(result.trials))
        self.assertEqual(len(result.events), len(result.trials) * 3)
        self.assertEqual(result.trial_metadata[0]["flat_label"], result.labels[0])
        self.assertAlmostEqual(result.epoch_times[0], config.export.epoch_start_sec)
        self.assertAlmostEqual(result.epoch_times[-1], config.export.epoch_end_sec - 1.0 / config.sfreq)

    def test_reduced_layout_warns(self) -> None:
        config = MotorIntentionConfig(
            layout=LayoutConfig(montage_name="motor_14"),
            include_eog=False,
            include_emg=False,
            include_line_noise=False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            MotorIntentionArchitecture(config).run()
        self.assertTrue(any("motor_14" in str(item.message) for item in caught))

    def test_base_framework_is_unchanged(self) -> None:
        pipeline_a = Pipeline(backend="numpy", seed=7)
        pipeline_a.add(
            "alpha",
            BandGenerator(BandSpec("alpha", 8.0, 12.0, 10.0, 2)),
            accumulate="replace",
        )
        result_a = pipeline_a.run(duration=2.0, sfreq=100.0, n_channels=2)

        pipeline_b = Pipeline(backend="numpy", seed=7)
        pipeline_b.add(
            "alpha",
            BandGenerator(BandSpec("alpha", 8.0, 12.0, 10.0, 2)),
            accumulate="replace",
        )
        result_b = pipeline_b.run(duration=2.0, sfreq=100.0, n_channels=2)

        np.testing.assert_allclose(
            np.asarray(result_a["signal"]),
            np.asarray(result_b["signal"]),
        )

    def test_electrode_mapping_helpers_follow_limb_logic(self) -> None:
        result = MotorIntentionArchitecture(
            MotorIntentionConfig(
                include_eog=False,
                include_emg=False,
                include_line_noise=False,
            )
        ).run()
        left_arm_trial = next(trial for trial in result.trials if trial.flat_label == "left_arm")
        visible = default_visible_channels(result, left_arm_trial, "imagery", default_region_for_trial(left_arm_trial), limit=4)
        self.assertIn("C4", visible)

        left_arm_importance = aggregate_electrode_importance(result, "left_arm", "imagery")
        self.assertEqual(left_arm_importance.iloc[0]["flat_label"], "left_arm")

        arm_vs_leg = compare_electrode_importance(result, "left_arm", "left_leg", "imagery")
        self.assertGreater(arm_vs_leg["abs_delta"].iloc[0], 0.0)

    def test_decoder_readiness_and_plot_smoke_helpers(self) -> None:
        result = MotorIntentionArchitecture(
            MotorIntentionConfig(
                include_eog=False,
                include_emg=False,
                include_line_noise=False,
            )
        ).run()
        readiness = decoder_readiness_frame(result)
        self.assertTrue(readiness["ready"].all())
        self.assertIn("generator_version", result.metadata)
        self.assertEqual(len(build_decoder_readiness_table(result)), len(readiness))

        left_arm_trial = next(trial for trial in result.trials if trial.flat_label == "left_arm")
        highlight = default_visible_channels(result, left_arm_trial, "imagery", "left_arm", limit=4)
        self.assertIsNotNone(plot_electrode_cluster_map(result, highlight_channels=highlight))
        self.assertIsNotNone(plot_class_summary_bars(result, "imagery"))
        self.assertIsNotNone(plot_class_topography(result, "left_arm", "imagery"))
        self.assertIsNotNone(plot_topography_difference(result, "left_arm", "rest", "imagery"))
        self.assertIsNotNone(plot_channel_delta_bars(result, "left_arm", "rest", "imagery", top_n=6))


if __name__ == "__main__":
    unittest.main()
