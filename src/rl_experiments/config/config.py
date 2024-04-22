import yaml

class Config(dict):
    def __init__(self, config_file: str = None):
        super(Config, self).__init__()

        if config_file is not None:
            self.load(config_file)

    def load(self, config_file: str):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        self.update(config)

        for key, value in self.items():
            if isinstance(value, dict):
                value = Config.from_dict(value)
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, data: dict):
        instance = cls()
        for key, value in data.items():
            if isinstance(value, dict):
                value = cls.from_dict(value)
            setattr(instance, key, value)
        return instance
