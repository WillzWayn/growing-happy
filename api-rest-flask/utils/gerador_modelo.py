## gerar dados
import numpy as np
import matplotlib.pyplot as plt

## ML
from sklearn.linear_model import LinearRegression
import pickle

# Vamos Criar aqui uma equação quase perfeitamente linear do tipo y=ax+b.
b = np.random.randint(20)
np.random.seed(7)
# Criando 50 valores aleatórios de x e y
x,y = [],[]
for i in range(60):
 x.append(i + np.random.normal(0,6,1) + b)
 y.append(i - np.random.normal(0,3,1) + b*2)


# Iniciando e fitando nosso modelo
model = LinearRegression()
model.fit(x,y)
# Analizando a resposta de saída
model.predict([[0],[100]])
#Respostas -> [[ 16.24413396], [102.13496669]]


## Salvando Modelo
with open("model.pickle","wb") as f: #cria um documento no seu pc chamado 'model.pickle' em modo de WB (write binary)
  pickle.dump(model, f) #dump significa despejar, descarregar
  
# Pronto. Agora você possuí o modelo já treinado salvo no seu pc !

## carregando o modelo para o código
with open("model.pickle","rb") as f: #Carrega o arquivo model.pickle em modo read binary
  modelo_carregado = pickle.load(f)
  
modelo_carregado.predict([[0],[100]])
#Respostas -> Respostas -> [[ 16.24413396], [102.13496669]]



# Figure plot
plt.style.use('seaborn-notebook')
plt.scatter(x,y, label='scatter plot com os valores de x e y')
plt.xlim(0,100)
plt.ylim(0,100)
plt.legend()
plt.title('Grafico com os valores artificiais gerados', fontdict=dict(fontsize = 20))
plt.savefig('grafico.png')
