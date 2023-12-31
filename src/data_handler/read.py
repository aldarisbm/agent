import gzip
import json
from typing import List


def read_data(wikipedia_filepath: str) -> List[str]:
    """
    Read the data from the wikipedia file. Returns a list of passages
    :param: wikipedia_filepath
    :return: list of passages
    """
    passages = []
    with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
        for line in fIn:
            data = json.loads(line.strip())

            paragraphs = data['paragraphs']
            passages.extend(paragraphs)

    return passages
