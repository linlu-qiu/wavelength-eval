import json
import logging
import os

logger = logging.getLogger(__name__)


class SimpleCache:
    def __init__(self, cache_file=None):

        self.cache_file = cache_file
        logger.info(f"Loading cache from {self.cache_file}")
        self.cache = self.load_from_file(self.cache_file)
        logger.info(f"Loaded {len(self.cache)} items from {self.cache_file}")

    def load_from_file(self, filename):
        """Load cache from a JSONL file."""
        if not os.path.exists(filename):
            return {}

        cache = {}
        with open(filename, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    key = json.dumps(data["request"], sort_keys=True)
                    cache[key] = data["response"]
                except Exception as e:
                    logger.warning(f"Error loading cache from {filename}: {e}")
                    continue
        return cache

    def get(self, request):
        key = json.dumps(request, sort_keys=True)
        return self.cache.get(key)

    def set(self, request, response):
        key = json.dumps(request, sort_keys=True)
        self.cache[key] = response
        with open(self.cache_file, "a") as f:
            f.write(json.dumps({"request": request, "response": response}) + "\n")
