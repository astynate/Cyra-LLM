import requests
from datasets import load_dataset
from __primary_transformations__ import preprocess_text

def load_wikitext(path: str) -> None:
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')

    with open(f'{path}/wikitext.txt', 'w', encoding='utf-8') as article_file:
        article_file.write(str(dataset['train']['text']))

def create_dataset_from_wikipedia(output_path: str, count_arcticles: int) -> None:
    
    session = requests.Session()
    wikipedia = "https://ru.wikipedia.org/w/api.php"

    search_params = {
        "action": "query",
        "format": "json",
        "list": "recentchanges",
        "rcnamespace": "0",
        "rclimit": str(count_arcticles),
        "rcprop": "title"
    }

    print('Getting articles...')

    request = session.get(url=wikipedia, params=search_params)
    articles = request.json()

    for i in range(count_arcticles):

        try:

            print(f'Reading article: {i}')
            page_title = articles["query"]["recentchanges"][i]["title"]

            acticle_params = {
                "action": "query",
                "prop": "extracts",
                "format": "json",
                "explaintext": True,
                "titles": page_title
            }

            article_request = session.get(url=wikipedia, params=acticle_params)
            article_text = article_request.json()

            page_id = list(article_text["query"]["pages"].keys())[0]
            page_content = article_text["query"]["pages"][page_id]["extract"]

            with open(f'{output_path}/{page_title}.txt', 'w', encoding='utf-8') as article_file:
                article_file.write(preprocess_text(page_content))

        except:

            print('Error')

if __name__ == '__main__':

    output_path = 'D:/Exider Company/Cyra/dataset_preparing/output_dataset/dataset-003'
    output_path_2 = 'D:/Exider Company/Cyra/dataset_preparing/output_dataset/dataset-004'

    # create_dataset_from_wikipedia(output_path, 5000)
    load_wikitext(output_path_2)
