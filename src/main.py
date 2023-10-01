import os
from dotenv import load_dotenv

from src.data_handler.read import read_data

load_dotenv()


def main():
    data_loc = os.getenv("DATA_LOC")
    data = read_data(data_loc)
    print(len(data))


if __name__ == "__main__":
    main()
