"""Tests for aspect-ratio-safe display sizing and playback pacing."""

import unittest

import numpy as np

from src.video import PlaybackTiming, calculate_letterbox_layout, letterbox_frame
from src.utils.statistics import FrameStatistics
from src.utils.visualization import (
    calculate_statistics_panel_layout,
    draw_display_statistics_panel,
    draw_help_overlay,
    draw_statistics_panel,
)


class DisplayAndPlaybackTests(unittest.TestCase):
    """Validate display math without opening an OpenCV window."""

    def test_portrait_frame_is_centered_in_a_wide_viewport(self) -> None:
        layout = calculate_letterbox_layout((1080, 1450), (1600, 900))

        self.assertEqual((layout.image_width, layout.image_height), (670, 900))
        self.assertEqual((layout.offset_x, layout.offset_y), (465, 0))
        self.assertEqual(layout.source_to_display(0, 0), (465, 0))
        self.assertEqual(layout.source_to_display(1080, 1450), (1135, 900))

    def test_landscape_frame_gets_top_and_bottom_letterboxing(self) -> None:
        layout = calculate_letterbox_layout((1920, 1080), (1000, 1000))

        self.assertEqual((layout.image_width, layout.image_height), (1000, 562))
        self.assertEqual((layout.offset_x, layout.offset_y), (0, 219))

    def test_letterbox_frame_preserves_content_without_cropping(self) -> None:
        frame = np.full((100, 200, 3), 255, dtype=np.uint8)
        output, layout = letterbox_frame(frame, (300, 300))

        self.assertEqual(output.shape, (300, 300, 3))
        self.assertEqual((layout.image_width, layout.image_height), (300, 150))
        self.assertTrue(np.all(output[:75] == 0))
        self.assertTrue(np.all(output[75:225] == 255))
        self.assertTrue(np.all(output[225:] == 0))

    def test_playback_waits_only_when_processing_is_faster_than_source(self) -> None:
        timing = PlaybackTiming(25.0)

        self.assertEqual(timing.wait_delay_ms(0.01), 30)
        self.assertEqual(timing.wait_delay_ms(0.05), 1)
        self.assertEqual(PlaybackTiming(0).wait_delay_ms(0.01), 1)

    def test_statistics_and_help_overlays_keep_frame_dimensions(self) -> None:
        frame = np.zeros((120, 240, 3), dtype=np.uint8)
        statistics = FrameStatistics(24.6, True, 0.5, 2, {"person": 2}, 2)

        statistics_frame = draw_statistics_panel(frame, statistics)
        help_frame = draw_help_overlay(statistics_frame)

        self.assertEqual(statistics_frame.shape, frame.shape)
        self.assertEqual(help_frame.shape, frame.shape)
        self.assertTrue(np.any(statistics_frame != 0))
        self.assertTrue(np.any(help_frame != 0))

    def test_portrait_display_places_panel_in_left_letterbox(self) -> None:
        statistics = FrameStatistics(24.6, True, 0.5, 3, {"person": 2, "car": 1}, 3)
        bounds, _, _, _, _ = calculate_statistics_panel_layout(
            statistics,
            (1600, 900),
            (465, 0, 670, 900),
        )

        self.assertEqual(bounds.placement, "left_letterbox")
        self.assertGreaterEqual(bounds.x, 0)
        self.assertLessEqual(bounds.right, 465)
        self.assertGreaterEqual(bounds.y, 0)
        self.assertLessEqual(bounds.bottom, 900)

    def test_landscape_display_places_panel_inside_video_bounds(self) -> None:
        statistics = FrameStatistics(24.6, False, 0.5, 1, {"very long class name": 1}, None)
        bounds, _, _, _, _ = calculate_statistics_panel_layout(
            statistics,
            (1000, 1000),
            (0, 219, 1000, 562),
        )

        self.assertEqual(bounds.placement, "inside_video")
        self.assertGreaterEqual(bounds.x, 0)
        self.assertGreaterEqual(bounds.y, 219)
        self.assertLessEqual(bounds.right, 1000)
        self.assertLessEqual(bounds.bottom, 781)

    def test_small_display_panel_stays_within_canvas(self) -> None:
        statistics = FrameStatistics(12.0, True, 0.5, 1, {"person": 1}, 1)
        canvas = np.zeros((130, 200, 3), dtype=np.uint8)
        drawn, bounds = draw_display_statistics_panel(canvas, statistics, (0, 10, 200, 110))

        self.assertEqual(drawn.shape, canvas.shape)
        self.assertGreaterEqual(bounds.x, 0)
        self.assertGreaterEqual(bounds.y, 0)
        self.assertLessEqual(bounds.right, 200)
        self.assertLessEqual(bounds.bottom, 130)


if __name__ == "__main__":
    unittest.main()
