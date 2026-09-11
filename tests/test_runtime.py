from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from inference_service.config import Settings
from inference_service.runtime import OwnedBackend


class LifecycleTests(unittest.TestCase):
    def test_existing_listener_is_never_replaced(self):
        with patch("inference_service.runtime.importlib.metadata.version", return_value="0.23.0"), \
             patch("inference_service.runtime.occupied", return_value=True), \
             patch("inference_service.runtime.subprocess.Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                OwnedBackend(Settings()).start()
            popen.assert_not_called()

    def test_start_failure_cleans_owned_process_and_log(self):
        process = MagicMock(pid=314159)
        process.poll.return_value = 1
        with tempfile.TemporaryDirectory() as folder, \
             patch("inference_service.runtime.importlib.metadata.version", return_value="0.23.0"), \
             patch("inference_service.runtime.occupied", return_value=False), \
             patch("inference_service.runtime.subprocess.Popen", return_value=process), \
             patch("inference_service.runtime.os.killpg") as kill:
            backend = OwnedBackend(Settings(log_dir=Path(folder)))
            with self.assertRaisesRegex(RuntimeError, "Backend exited"):
                backend.start()
            kill.assert_called_once_with(314159, signal.SIGTERM)
            self.assertIsNone(backend.process)
            self.assertIsNone(backend.log)

    def test_stop_is_idempotent_and_signals_owned_group(self):
        backend = OwnedBackend(Settings())
        backend.process = MagicMock(pid=314159)
        with patch("inference_service.runtime.os.killpg") as kill:
            backend.stop()
            backend.stop()
        kill.assert_called_once_with(314159, signal.SIGTERM)


if __name__ == "__main__": unittest.main()
