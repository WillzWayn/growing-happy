import numpy as np
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource,fields# vamos usar daqui a pouco

app  = Flask(__name__)

import pickle
## carregando o modelo para o código
with open("./utils/model.pickle","rb") as f:
    #Carrega o arquivo model.pickle em modo read binary
    modelo_carregado = pickle.load(f)

app_infos = dict(version='1.0', title='Primeira API',
        description='Essa API faz o que eu quiser',
        contact_email='WillzWayn@gmail.com', doc='/documentacao',
        prefix='/eee')

rest_app = Api(app, **app_infos)



db_model = rest_app.model('Variáveis usadas no primeiro modelo', 
	{'array': fields.List(cls_or_instance= fields.Float ,required = True, 
					 description="String que contem um array: 1,9,8", 
					 help="Ex. 1,9,8"),
   'argumento_2': fields.Arbitrary(required = False,
           description = "Só para exemplificar que podemos ter mais entradas de dados." )})

## Vamos organizar os endpoints por aqui!
# link gerado será: http://127.0.0.1:5000/primeiro_endpoint_swagger
nome_do_endpoint = rest_app.namespace('primeiro_endpoint_swagger', description='Esse endpoint é responsável por fazer uma análise estatística.')
@nome_do_endpoint.route("/")
class Classe_que_contem_funcoes(Resource):
    @rest_app.expect(db_model)
    def post(self):
        array = request.json['array']
        #Usando o algoritmo de M.L
        array = np.array(array).reshape(-1,1)
        pred = modelo_carregado.predict(np.array(array))
        
        return {
                "status": "Array recebido",
                "Quantidade_de_numeros_recebidos_para_prever:": array.shape[0],
                "valores_requisicao" : array.T[0].tolist(),
                "valores_preditos": pred.T[0].tolist(),
            }


@app.route("/")
def primeiro_endpoint_get():
  return "Tudo Funcionando Corretamente !", 200

@app.route("/segundo_endpoint/<int:array_do_usuario>")
def segundo_endpoint(array_do_usuario):
  array_do_usuario = np.array([array_do_usuario])
  pred = modelo_carregado.predict(array_do_usuario.reshape(1,-1))
  return (f"sua solicitação foi predita como: {pred[0,0]}", 200)


if __name__ == "__main__":
  debug = True # com essa opção como True, ao salvar, o "site" recarrega automaticamente.
  app.run(host='0.0.0.0', port=5000, debug=debug)
