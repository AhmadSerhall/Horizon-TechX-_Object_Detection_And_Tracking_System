"""Tests for aspect-ratio-safe display sizing and playback pacing."""

import unittest

import numpy as np

from src.video import PlaybackTiming, calculate_letterbox_layout, letterbox_frame


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


if __name__ == "__main__":
    unittest.main()
