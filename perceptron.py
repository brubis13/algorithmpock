import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('house-votes-84.csv')
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
        if ('y' in Xf[a][b]):
            Xf[a][b] = 1
        elif ('n' in Xf[a][b]):
            Xf[a][b] = -1

learningRate = 0.01
plotData = []
pesos = np.random.rand(16, 1)
erroClassificacao = 1 #erro de classificacao
minErroClassificacao = 1000#minimo erro de classificacao
interacao = 0

while (erroClassificacao != 0 and (interacao<100)):
    interacao += 1
    erroClassificacao = 0
    for i in range(0, len(Xf)):  #tamanho da minha base de dados, 0 a 434
        atualX = Xf[i].reshape(-1, Xf.shape[1])
        atualY = Yf[i]
        wTx = np.dot(atualX, pesos)[0][0]
        if atualY == 1 and wTx < 0:
            erroClassificacao += 1
            pesos = pesos + learningRate * np.transpose(atualX)
        elif atualY == -1 and wTx > 0:
            erroClassificacao += 1
            pesos = pesos - learningRate * np.transpose(atualX)
    plotData.append(erroClassificacao)
    if erroClassificacao<minErroClassificacao:
        minErroClassificacao = erroClassificacao
    # if iteration%1==0:
    #print("Interacao {}, ErroClassificacao {}".format(interacao, erroClassificacao))

print ("Erro de Classificacao minimo : ",minErroClassificacao)
print("Pesos no bolso", pesos.transpose())

#acuracia = ((Xf.shape[0]-minErroClassificacao)/Xf.shape[0])*100
acuracia = Xf.shape[0]-minErroClassificacao
print("acuracia", acuracia)
print("tamanho", Xf.shape[0])
acur = (acuracia*100/Xf.shape[0]*100)
acur = acur/100
print ("Acuracia do Algoritmo do bolso:", float(acur),"%")
#print("Numero de Erro de Classificacao: ", erroClassificacao)
plt.plot(np.arange(0, 100),plotData)
plt.xlabel("Numero de interacoes")
plt.ylabel("Numero de Erro de classificacao")
#plt.show()
