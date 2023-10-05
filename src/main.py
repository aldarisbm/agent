import os
import json
from dotenv import load_dotenv

from src.helpers.state_handler import write_to_json
from src.data_handler.read import read_data
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

load_dotenv()


def main():
    local_map_loc = os.getenv("LOCAL_MAP_LOC")
    data_loc = os.getenv("DATA_LOC")

    print(f"Processing file: {data_loc}...")

    if not os.path.exists(local_map_loc):
        # if json doesn't exist, let's create it
        # this should probably an index db
        with open(local_map_loc, 'w+') as f:
            json.dump({}, f)

    with open(local_map_loc, ) as f:
        basename = os.path.basename(data_loc)
        dl_key = basename.lower().replace('.', '_').replace(' ', '_').replace('-', '_')
        print(f"Saving key: {dl_key} with value {data_loc}")

        local_map = json.load(f)
        print('Checking if file has already been processed...')
        if dl_key in local_map:
            print('Data has already been loaded.')
        else:
            print('Starting data processing...')
            data = read_data(data_loc)
            local_map[dl_key] = data_loc
            json.dump(local_map, f)

    # chroma_db = Chroma.from_texts(texts=data, embeddings=SentenceTransformerEmbeddings())
    # print(len(data))


if __name__ == "__main__":
    main()
