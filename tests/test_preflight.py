import json
from pathlib import Path
import tempfile
import unittest
from inference_service.preflight import check_model


class ModelChecks(unittest.TestCase):
    def test_incomplete_wrong_and_complete_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Cannot read"):
                check_model(tmp)
            config = dict(model_type='qwen3', hidden_size=2560, num_hidden_layers=36,
                          num_attention_heads=32, num_key_value_heads=8, head_dim=128)
            (p/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                check_model(tmp)
            for name in ['model.safetensors', 'tokenizer.json', 'tokenizer_config.json']:
                (p/name).write_text('fixture')
            self.assertIn('identity is not verified', check_model(tmp))
            config['quantization_config'] = {'quant_method': 'awq'}
            (p/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "unquantized"):
                check_model(tmp)

    def test_other_remote_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "Qwen3-4B"):
            check_model('Qwen/Qwen3-8B')
