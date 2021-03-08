from keras.models import Sequential, model_from_json
from keras.layers import Dense
from AI.actions import actions

class Model:
    def save_model(self, model):
        # serialize model to JSON
        model_json = model.to_json()
        with open("model_Tetris.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model_Tetris.h5")

    def loadModel(self, compil=True):
        try:   
            json_file = open('model_Tetris.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            if compil:
                model.compile(optimizer="adam", loss="mse")
        except:
            print("No model found")
            model = None
        try:
            model.load_weights("model_Tetris.h5")
        except:
            print("No weights found")
        return model

    def createModel(self, height, width, hidden_size=100, loadModel=False, compil = True):
        inputs = ["currentFigure", "nextFigure", "changeFigure"]
        model = None
        if loadModel:
            model = self.loadModel(compil)
        if model == None:
            model = Sequential()
            #Input-Layer -> Alle vorhandenen, unabhängigen Informationen
            model.add(Dense(hidden_size, input_shape=(height*width + len(inputs),), activation="relu"))
            #Hidden-Layer
            model.add(Dense(hidden_size, activation="relu"))
            #Output-Layer -> Beinhaltet die Anzahl der möglichen Aktionen als Anzahl von Neuronen -> Default-Activation: linear
            model.add(Dense(len(actions)))
            model.compile(optimizer = "adam", loss="mse")
            self.save_model(model)
        return model