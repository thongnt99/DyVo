import requests

IP_ADDRESS = "http://localhost"
PORT = "1235"
while True:
    text_doc = input("Please enter your input")

    document = {
        "text": text_doc,
        "spans": [],  # in case of ED only, this can also be left out when using the API
    }

    API_result = requests.post("{}:{}".format(
        IP_ADDRESS, PORT), json=document).json()
    print(API_result)
