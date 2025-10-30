class context:
    def __init__(self):
        self.user = "default"
        self.info = "This is a context class."

        self.url_list = [
            "http://example.com/resource1",
            "http://example.com/resource2",
            "http://example.com/resource3"
        ]

        self.results = [
            {"url": "http://example.com/resource1", "status": "success", "data": {}},
        ]


'''
topic: Context class for sharing data between modules.



rule: 


format:
 {
    "http://example.com/resource2": 100, reason: "..." ,
    "http://example.com/resource3": 200

 }
'''

'''
URL0: http://example.com/resource2
context: ....

URL1: http://example.com/resource2
context: ....

URL2: http://example.com/resource2
context: ....
'''
