import importlib
from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from anysd import infer


class DownloadImageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "input.png"

    def test_local_non_square_image_is_resized_and_converted_to_rgb(self):
        Image.new("RGBA", (800, 600), (10, 20, 30, 128)).save(self.path)

        image = infer.download_image(self.path)

        self.assertEqual(image.size, (512, 512))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((256, 256)), (10, 20, 30))
        # The returned image must remain usable after the source has been closed.
        self.path.unlink()
        self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))

    def test_exif_orientation_rotates_pixels_before_resize(self):
        image = Image.new("RGB", (80, 40), "red")
        image.paste("blue", (40, 0, 80, 40))
        exif = image.getexif()
        exif[274] = 6  # Rotate 90 degrees clockwise: left red becomes top red.
        image.save(self.path, exif=exif)

        result = infer.download_image(self.path)

        self.assertEqual(result.size, (512, 512))
        self.assertEqual(result.getpixel((256, 64)), (255, 0, 0))
        self.assertEqual(result.getpixel((256, 448)), (0, 0, 255))
        self.assertIsNone(result.getexif().get(274))

    def test_http_and_https_images_use_timeout_and_close_response(self):
        encoded = BytesIO()
        Image.new("L", (90, 60), 123).save(encoded, format="PNG")
        for scheme in ("http", "https"):
            with self.subTest(scheme=scheme):
                response = BytesIO(encoded.getvalue())
                url = f"{scheme}://example.test/image.png"
                with patch("anysd.infer.urlopen", return_value=response) as request:
                    result = infer.download_image(url)

                request.assert_called_once_with(url, timeout=30)
                self.assertTrue(response.closed)
                self.assertEqual(result.size, (512, 512))
                self.assertEqual(result.mode, "RGB")
                self.assertEqual(result.getpixel((256, 256)), (123, 123, 123))

    def test_invalid_remote_image_still_closes_response(self):
        response = BytesIO(b"not an image")
        with patch("anysd.infer.urlopen", return_value=response):
            with self.assertRaises(OSError):
                infer.download_image("https://example.test/broken.png")
        self.assertTrue(response.closed)

    def test_missing_local_file_does_not_attempt_network(self):
        with patch("anysd.infer.urlopen") as request:
            with self.assertRaises(FileNotFoundError):
                infer.download_image(self.path)
        request.assert_not_called()


class MaxCasesTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.restore_environment)

    def restore_environment(self):
        self.environment.stop()
        importlib.reload(infer)

    def test_unset_limit_includes_more_than_99_cases(self):
        importlib.reload(infer)

        self.assertIsNone(infer.MAX_CASES)
        cases = list(range(120))
        self.assertEqual(cases[:infer.MAX_CASES], cases)

    def test_empty_limit_means_all_cases(self):
        os.environ["ANYSD_MAX_CASES"] = ""
        importlib.reload(infer)
        self.assertIsNone(infer.MAX_CASES)

    def test_explicit_limits(self):
        cases = list(range(7))
        for value, expected in (("0", []), ("2", [0, 1]), ("20", cases)):
            with self.subTest(value=value):
                os.environ["ANYSD_MAX_CASES"] = value
                importlib.reload(infer)
                self.assertEqual(cases[:infer.MAX_CASES], expected)

    def test_negative_limit_is_rejected(self):
        os.environ["ANYSD_MAX_CASES"] = "-1"
        with self.assertRaisesRegex(ValueError, "ANYSD_MAX_CASES"):
            importlib.reload(infer)

    def test_non_integer_limit_is_rejected(self):
        os.environ["ANYSD_MAX_CASES"] = "two"
        with self.assertRaises(ValueError):
            importlib.reload(infer)


if __name__ == "__main__":
    unittest.main()
