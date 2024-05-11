import unittest
from unittest.mock import mock_open, patch
# from your_script import load_config, override_config
import yaml
import argparse
import sys
from utils import override_config

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Run the application.')
    parser.add_argument('args', nargs='*',
                        help='Provide overrides as key=value pairs (e.g., username=newuser).')
    return parser.parse_args()


class TestConfigLoadingAndOverriding(unittest.TestCase):
    def test_load_config(self):
        mock_file_content = """
        database:
          host: localhost
          port: 5432
        username: admin
        password: secret
        """
        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            config = load_config('dummy_path.yaml')
            self.assertEqual(config['username'], 'admin')
            self.assertEqual(config['database']['host'], 'localhost')

    def test_override_config(self):
        config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'nested': {
                    'key': 'value'
                }
            },
            'username': 'admin',
            'password': 'secret'
        }
        overrides = ['username=newadmin', 'database.host=newhost', 'newkey=newvalue', 'database.nested.key=2']
        expected_config = {
            'database': {
                'host': 'newhost',
                'port': 5432,
                'nested': {
                    'key': 2
                }
            },
            'username': 'newadmin',
            'password': 'secret',
            'newkey': 'newvalue'
        }
        updated_config = override_config(config, overrides)
        self.assertEqual(updated_config, expected_config)

    def test_invalid_override_format(self):
        config = {'username': 'admin'}
        overrides = ['username-newadmin']  # Incorrect format
        with self.assertRaises(ValueError):
            updated_config = override_config(config, overrides)


if __name__ == '__main__':
    unittest.main()
