"""No-GPU, no-cache tests for consumer.config.load_config.

From repository root after copying this to consumer/tests/test_config.py:
    python -m unittest discover -s consumer/tests -p 'test_config.py' -v
"""

import importlib.util
from pathlib import Path
import tempfile
import unittest

# Locate consumer.config using the original package when installed in its intended path;
# fall back to the adjacent standalone config.py for inspecting the supplied update.
try:
    from consumer.config import load_config
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location(
        "nanochatnt_config_validation", Path(__file__).with_name("config.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    load_config = module.load_config

# Use actual TOML parsing in every case. The original configuration is the fixture.
CONFIG_SOURCE = Path(__file__).with_name("config.toml")
if not CONFIG_SOURCE.exists():
    CONFIG_SOURCE = Path(__file__).resolve().parents[2] / "consumer" / "config.toml"


def read_config():
    return CONFIG_SOURCE.read_text(encoding="utf-8")


def replace(text, old, new):
    assert old in text, old
    return text.replace(old, new, 1)


class ConfigValidationTests(unittest.TestCase):
    def check(self, contents, *, error=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.toml"
            path.write_text(contents, encoding="utf-8")
            if error:
                with self.assertRaisesRegex(ValueError, error):
                    load_config(path)
            else:
                return load_config(path)

    def test_default_config(self):
        cfg = self.check(read_config())
        self.assertEqual(cfg["model"]["depth"], 4)
        self.assertEqual(cfg["training"]["device_batch_size"], 2)

    def test_missing_model_section(self):
        source = read_config()
        self.check(source.split("[model]")[0] + "[training]" + source.split("[training]", 1)[1], error=r"Missing \[model\]")

    def test_missing_training_key(self):
        self.check(replace(read_config(), 'save_every = 200', ''), error='Missing training.save_every')

    def test_invalid_depth_type(self):
        self.check(replace(read_config(), 'depth = 4', 'depth = true'), error='model.depth')

    def test_invalid_window_pattern(self):
        self.check(replace(read_config(), 'window_pattern = "L"', 'window_pattern = "LLX"'), error='model.window_pattern')

    def test_native_lowercase_window_pattern(self):
        cfg = self.check(replace(read_config(), 'window_pattern = "L"', 'window_pattern = "ssl"'))
        self.assertEqual(cfg['model']['window_pattern'], 'ssl')

    def test_context_over_packed_row(self):
        self.check(replace(read_config(), 'max_seq_len = 2048', 'max_seq_len = 4096'), error='model.max_seq_len')

    def test_shorter_context_is_native_valid(self):
        cfg = self.check(replace(read_config(), 'max_seq_len = 2048', 'max_seq_len = 512'))
        self.assertEqual(cfg['model']['max_seq_len'], 512)

    def test_native_disabled_eval_and_save(self):
        text = replace(read_config(), 'eval_every = 100', 'eval_every = -1')
        text = replace(text, 'save_every = 200', 'save_every = -1')
        self.check(text)

    def test_native_auto_device_batch(self):
        self.check(replace(read_config(), 'device_batch_size = 2', 'device_batch_size = -1'))

    def test_invalid_batch_zero(self):
        self.check(replace(read_config(), 'device_batch_size = 2', 'device_batch_size = 0'), error='training.device_batch_size')

    def test_bad_ratio(self):
        self.check(replace(read_config(), 'target_param_data_ratio = 12.0', 'target_param_data_ratio = nan'), error='training.target_param_data_ratio')

    def test_bad_device(self):
        self.check(replace(read_config(), 'device_type = "cuda"', 'device_type = "tpu"'), error='training.device_type')

    def test_warmup_zero_allowed(self):
        self.check(replace(read_config(), 'warmup_steps = 40', 'warmup_steps = 0'))

    def test_zero_eval_tokens_with_eval_enabled(self):
        self.check(replace(read_config(), 'eval_tokens = 262_144', 'eval_tokens = 0'), error='training.eval_tokens')

    def test_zero_eval_tokens_with_eval_disabled(self):
        text = replace(read_config(), 'eval_every = 100', 'eval_every = -1')
        self.check(replace(text, 'eval_tokens = 262_144', 'eval_tokens = 0'))


if __name__ == '__main__':
    unittest.main()
