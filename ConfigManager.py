import json

class ConfigManager:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.config = self.load_config()

    def load_config(self):
        try:
            with open(self.config_file_path, 'r') as file:
                config_data = json.load(file)
            return config_data
        except (FileNotFoundError, json.JSONDecodeError):
            # Return default configuration if the file doesn't exist or is invalid
            return {}

    def save_config(self):
        with open(self.config_file_path, 'w') as file:
            json.dump(self.config, file, indent=4)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()
