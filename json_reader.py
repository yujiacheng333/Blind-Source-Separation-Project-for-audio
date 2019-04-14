import json


class JsonReader:
    """This class is used to json file IO op
    """

    def __init__(self, path="../data.json"):
        self.path = path
        """
        :param path: file path
        """

    def store_json(self,data):
        """
        :param self:  None
        :param data: the dic to save
        :return: None
        """
        with open(self.path, 'w') as json_file:
            json_file.write(json.dumps(data))

    def load_json(self):
        """
        :param self: the class
        :return: the dic in json file
        """
        with open(self.path) as json_file:
            data = json.load(json_file)
            return data


if __name__ == '__main__':
    dic = {"file_path": "./buffers"}
    jr = JsonReader()
    jr.store_json(dic)
    a = jr.load_json()
    print(a)
