from bs4 import BeautifulSoup
import os


def parse_docs(path):
    documents = {}
    for subdir, dirs, files in os.walk(path):
        for file in files:
            print(os.path.join(subdir, file))
            with open(os.path.join(subdir, file), encoding='latin-1') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                text = soup.find('text').get_text()
                docno = soup.find('docno').get_text()
                documents[docno] = text
                # print(type(text))

    return documents
