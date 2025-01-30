import requests

from netbalance.configs.disbiome import DISBIOME_RAW_DATA_DIR
from netbalance.utils import prj_logger
from netbalance.utils.io import json_dump

logger = prj_logger.getLogger(__name__)

collector_session = requests.Session()


class DisbiomeDataCollectorManager:
    ROOT_API_URL = "https://disbiome.ugent.be:8080/"

    def __init__(self):
        self.tables_data = list()

    def collect_all_data(self):
        self.update_tables(self.get_tables_list())

    def get_tables_list(self):
        return {
            "experiment": self.ROOT_API_URL + "experiment",
            "organism": self.ROOT_API_URL + "organism",
            "disease": self.ROOT_API_URL + "disease",
            "method": self.ROOT_API_URL + "method",
            "publication": self.ROOT_API_URL + "publication",
            "sample": self.ROOT_API_URL + "sample",
        }

    def update_tables(self, tables_data):
        for table_name, api_url in tables_data.items():
            self._update_table(table_name, api_url)
            logger.info(f"table with name {table_name} saved!")

    def _update_table(self, table_name, api_url):
        data = self.fetch_data(api_url)
        json_dump(f"{DISBIOME_RAW_DATA_DIR}/{table_name}.json", data)

    @staticmethod
    def fetch_data(url):
        logger.info(f"trying to fetch url {url}.")
        result = list()
        try:
            response = collector_session.get(url=url, timeout=10)
            if response.status_code == 200:
                result = response.json()
                logger.info(f"request with url {url} completed.")
            else:
                logger.warning(f"request with url {url} failed.")
        except Exception as e:
            print(f"exception in response of {url} : {e}")

        return result
