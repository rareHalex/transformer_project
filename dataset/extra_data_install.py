from html.parser import HTMLParser
from bs4 import BeautifulSoup
import requests


list_url = ['https://arxiv.org/list/astro-ph/2301', 'https://arxiv.org/list/math/2301', 'https://arxiv.org/list/math.GN/pastweek?skip=0&show=25#item5',
            'https://arxiv.org/year/q-bio/23', 'https://arxiv.org/list/math.KT/recent', 'https://arxiv.org/list/cond-mat/2302',
            'https://arxiv.org/list/physics.ed-ph/pastweek?skip=0&show=25']


def to_ref(site_string):
    """
    Приписание url для создание ссылок
    """
    return 'https://arxiv.org/' + site_string


def get_html(site_link):
    """
    Подключение к сайту
    """
    requests_accept = requests.get(site_link)
    return requests_accept


def checker(text):
    """
    Проверка на корректность ссылок.
    """
    wrong_word = ['.php', 'http', '.ru', '//', '.com', 'https']
    for site_string in wrong_word:
        if site_string in text:
            return False
    return True


class MyHTMLparser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.link_list = []
        self.img_list = []

    def handle_starttag(self, tag: str, attrs) -> None:
        Find_tags = ['div', 'dl', 'dt', 'a', 'title']
        if tag in Find_tags:
            d = dict(attrs)
            if 'href' in d.keys():
                href_dict = d.get('href')
                if '/abs/' in href_dict:
                    if checker(href_dict):
                        self.link_list.append(to_ref(href_dict))


for url in list_url:
    parser = MyHTMLparser()
    html = get_html(url)
    parser.feed(html.text)
    extra_data_dict = {}

    for link in parser.link_list:
        page = requests.get(link)
        soup = BeautifulSoup(page.text, "html.parser")
        tittle_data = soup.findAll('h1', class_='title mathjax')
        abstract_data = soup.findAll('blockquote', class_='abstract mathjax')

        tittle_list = []
        for data in tittle_data:
            if data.find('span') is not None:
                tittle_list.append(data.text[6:])

        abstract_list = []
        for data in abstract_data:
            if data.find('span') is not None:
                data = data.text.replace("\n", "")
                abstract_list.append(data)

        label_list = soup.findAll('span', class_='arxivid')
        label_string = str(label_list[0])
        target = label_string[label_string.find('[')+1:label_string.find(']')]
        extra_data_dict[target] = [abstract_list, tittle_list]


