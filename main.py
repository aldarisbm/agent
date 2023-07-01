import os
from dotenv import load_dotenv
from langchain.llms import OpenAI


def main():
    load_dotenv()
    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), temperature=0.5)
    prediction = llm.predict("What is the best dog breed?")
    print(prediction)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
