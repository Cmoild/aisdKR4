import requests
import json

def trng(n = 100, interval = [0, 1]):
    url = 'https://api.random.org/json-rpc/1/invoke'

    data = {'jsonrpc':'2.0','method':'generateIntegers','params': {'apiKey':'ad7e95b7-7a7d-4f68-b121-b9a55efdf6c1','n':n,'min':interval[0],'max':interval[1],'replacement':'true','base':10},'id':24565}

    params = json.dumps(data)

    response = requests.post(url,params)

    j = json.loads(response.text)
    #print(json.loads(response.text))

    return j["result"]["random"]["data"]