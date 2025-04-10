import logging


class Setup:

    _logger: logging

    def __init__(self, model_name):
        self.setup_logger()
        self._logger = logging.getLogger(model_name)

    def setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("app.log", mode="w")
            ]
        )

    def get_logger(self):
        return self._logger
