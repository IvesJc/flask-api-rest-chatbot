# Fazer isso no Colab
# import pickle
# with open('meu_modelo_serializado.pickle', 'wb') as f:
#   pickle.dump(lda, f)

from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)


with open('meu_modelo_serializado.pickle', 'rb') as f:
    modelo = pickle.load(f)

@app.route('/prever', methods=['GET'])
def prever():
    parametro1 = float(request.args.get('comp_abd'))
    parametro2 = float(request.args.get('comp_ant'))    

    entrada = np.array([[parametro1, parametro2]])
    resultado = modelo.predict(entrada)

    return jsonify({'previsao': resultado.tolist()})



if __name__ == "__main__":
    app.run()


