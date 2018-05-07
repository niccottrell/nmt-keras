import requests
import json
from functools import lru_cache

api_token = 'your_api_token'
api_url_base = 'https://api.sprawk.com/v2/'

# shared headers
headers = {'Content-Type': 'application/json',
           'Authorization': 'Bearer {0}'.format(api_token),
           'User-Agent': 'NeuMT'}


@lru_cache(maxsize=1024)
def is_proper(lang, text) -> bool:
    # do the call
    response = requests.get(api_url_base + '/is_proper.json?lc=' + lang + '&t=' + text, headers=headers)
    # unpack JSON
    if response.status_code == 200:
        loads = json.loads(response.content.decode('utf-8'))
        return loads.result  # true or false
    else:
        raise ValueError('Error contacting Sprawk')

