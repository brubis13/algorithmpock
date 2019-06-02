import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('original.csv')
Xf = df.iloc[:, 1:].values
Yf = df.iloc[:, 0].values

for i in range(0,434):
    if Yf[i] == 'democrat':
        Yf[i] = -1
    elif Yf[i] == 'republican':
        Yf[i] = 1
Yf = Yf.astype(int)

for a in range(0, 434):
    for b in range(0,16):
        if 'y' in Xf[a][b]:
            Xf[a][b] = 1.0
        elif 'n' in Xf[a][b]:
            Xf[a][b] = -1.0
        elif '?' in Xf[a][b]:
            Xf[a][b] = 0.0
        #else:
            #print('erro')

learningRate = 0.01 #taxa de atualizacao de pesos
plotData = []
pesos = np.random.rand(16, 1)
minErroClassificacao = 1000 #minimo erro de classificacao

bolsoAccuracy = 0 ##melhor acuracia encontrada
bolso = [] #pesos da melhor acuracia encontrada

def predicao(x, pesos):
    pred = np.dot(x, pesos)[0][0] #multiplicacao de matriz
    return pred

#atualY = labels, erro nessa funcao por causa do python 2
def atualizar_pesos(label, prediction, atualX):
    if label == 1 and prediction < 0:
        return pesos + learningRate * np.transpose(atualX)
    elif label == -1 and prediction > 0:
        return pesos - learningRate * np.transpose(atualX)

def calcular_erro(label, prediction):
    return (label-prediction)**2

def atualizar_bolso(accuracy):
    global bolsoAccuracy
    global bolso
    if accuracy > bolsoAccuracy:
        bolsoAccuracy = accuracy
        bolso = pesos
        print("Bolso atualizado, melhor acuracia: {}".format(bolsoAccuracy))

def acuracia():
    aux = 0.0
    for i in range(0, len(Xf)):
        atualX = Xf[i].reshape(-1, Xf.shape[1])
        atualY = Yf[i]
        wTx = predicao(atualX, pesos)
        if (atualY == 1 and wTx > 0) or (atualY == -1 and wTx < 0):
            aux += 1
    return aux/len(Xf)

for iteracao in range(100):
    erroClassificacao = 0
    #for i in range(0, len(Xf)):  #tamanho da minha base de dados, 0 a 434
    #stocastico
    batch_size = 20
    a = []
    for i in range(batch_size):
        a.append(np.random.choice(range(0, len(Xf))))

    for i in a:
        atualX = Xf[i].reshape(-1, Xf.shape[1])
        atualY = Yf[i]
        wTx = predicao(atualX, pesos)

        ## atualizando pesos --
        if atualY == 1 and wTx < 0:
            pesos = pesos+learningRate * np.transpose(atualX)
        elif atualY == -1 and wTx > 0:
            pesos = pesos-learningRate * np.transpose(atualX)
        ## --
        erroClassificacao = erroClassificacao + calcular_erro(atualY, wTx)

    acc = acuracia()
    atualizar_bolso(acc)
    #print("acut", acc)

    #print("Erro Classificacao", erroClassificacao)
    plotData.append(erroClassificacao)
    if erroClassificacao<minErroClassificacao:
        minErroClassificacao = erroClassificacao

print("Acuraria do bolso: ", bolsoAccuracy)
print("Pesos do bolso:", bolso)
plt.plot(np.arange(0, 100),plotData)
plt.xlabel("Numero de interacoes")
plt.ylabel("Numero de Erro de classificacao")
plt.show()
