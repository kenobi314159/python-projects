from time import gmtime
from glob import glob
from tensorflow.keras.models import load_model

class ModelStorage:
    def __init__(self, storage_dir="stored_models"):
        self.storage_dir = storage_dir

    def _getTimeString(self):
        t = gmtime()
        return f"{t.tm_year:04}-{t.tm_mon:02}-{t.tm_mday:02}_{t.tm_hour:02}-{t.tm_min:02}-{t.tm_sec:02}"

    def getStorageList(self):
        return glob(f"{self.storage_dir}/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*.keras")

    def storeModel(self, model, name):
        file_name = self.storage_dir + "/" + self._getTimeString() + "_" + name + ".keras"
        model.save(file_name)
        return file_name

    def loadModel(self, file_name):
        return load_model(file_name)

    def loadLatesModelContaining(self, substring):
        stored_models = self.getStorageList()
        models = [fn for fn in stored_models if substring in fn]
        model = models[-1]
        return self.loadModel(model), model
