import os

class DictGenerator():
    def __init__(self, input_data: dict, params: dict = {}) -> None:
        self.input_data = input_data
        self.params = params


    def __len__(self):
        raise NotImplementedError


    def __call__(self):
        raise NotImplementedError


class ImageGenerator(DictGenerator):
    def __len__(self):
        return len(self.input_data["images"])

    def __call__(self):
        assert isinstance(self.params, dict) and 'root_dir' in self.params
        for image_json in self.input_data["images"]:
            yield image_json["id"], os.path.join(self.params['root_dir'], image_json["file_name"])
