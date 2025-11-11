# Clusterização do zero com Numpy

## Projeto de clusterização na base wine do scikit learn com numpy puro

### Objetivo

**Entender o funcionamento do kmeans**

## 1. O que é clusterização

* Faz parte do aprendizado não supervisionado de máquina, onde a intenção é entender o dataset e agrupar os dados de uma forma que faça sentido, uma vez que não temos variável de resposta.
* E através do algoritmo vamos descobrir a estrutura dos dados.

## 1.1 K-MEANS

*   O objetivo do k-means é atraves de "k" clusters, encontrar quais pontos temos a menor distancia entre os dados
*   Definimos nesse problema k=3, então de início o algoritmo escolhe aleatóriamente três pontos dado como centroides
*   E vamos fazer o cálculo de distancia para cada ponto do datasete e atribuir ao centroide com menor distancia
*   Depois vamos pegar a média desses centroides, daí que vem o nome "means" e calcular a distancia até a convergencia, ou seja, até que as médias não tenham movimento negativo ou atinjam a quantidade que definimos como máxima que é 300.

## 2. DataSet -> Usamos o dataset wine do scikitlearn

```python
dataset = load_wine().data
features=load_wine().feature_names
X=pd.DataFrame(dataset,columns=features)
X.head()
```
<img width="1206" height="193" alt="image" src="https://github.com/user-attachments/assets/a24a37d5-275f-44c6-927f-2fb4663c6b33" />



--------------------------------------------

* Chamamos o PCA para reduzir a dimensionalidade do dataset
  ```python
  #Chama o PCA para reduzir a dimensionalidade
  pca = PCA(n_components=2)
  dataset = pca.fit_transform(X)
  ```

---
##  Cálculo da Função de Custo (J)

A função `calcula_J()` é responsável por medir **o quão bem os pontos estão agrupados** em torno dos seus centróides.  
Matematicamente, ela calcula a **inércia total**, isto é, a soma das menores distâncias quadradas entre cada ponto e o centróide mais próximo.

$$
J = \sum_{i=1}^{n} \min_{k} \| x_i - c_k \|^2
$$


* Função que usamos para calcular a distancia entre os dados e os centróides
* Seguindo a fórmula fazemos a soma dos quadrados das diferenças
* E depois passsamos para `j` que é a nossa função de custo os minimos dessas diferenças

```python
class Kmeans:
    def __init__(self,k,iterations=300,tol=10e-4):
        self.k=k
        self.iterations=iterations
        self.tol=tol
    def centroides_iniciais(self):
        """Função para definicão inicial dos centróides, vamos utilizar o metodo aleatorio de inicio"
        """
        #escolha dos centróides
        idxs = np.random.choice(dataset.shape[0],self.k,replace=False)
        centroides = dataset[idxs, :]    
        return centroides
    def calcula_J(self,dataset,centroides):
        """Função para calcular a menor distancia entre os pontos e os centroides
        parametros de entrada : dataset e os centroides_iniciais
        Retorno: Menores distancias e menor J"""
        diff =dataset[:,None,:]-centroides[None,:,:]
        d=np.sum(diff**2,axis=2)
        j=np.sum(np.min(d,axis=1))
        menores=np.argmin(d,axis=1)
        return menores,j,d
    def atualiza_centroides(self,dataset,menores,k):
        "Função que irá fazer o cálculo dos novos centróides"
        "Parametros de entrada: dataset, menores e k"
        "Retorno: novos centroides"
        lists=[]
        for i in range(k):
            grupo=dataset[menores==i]
            grupo=np.mean(grupo,axis=0)
            lists.append(grupo)
            centroides=np.array(lists)
        return centroides

    def run_kmeans(self,X,centroides, tol, max_iter):
        """Função para calcular a função de custo
        Parametro de entrada: DATASET E OS CENTROIDES tolerancia e max iterações opcionais
        retorno: Valor de J(Função de custo)"""
        c=centroides
        labels,j,d= self.calcula_J(X,c)
        centroides=self.atualiza_centroides(X,labels,self.k)
        labels,j_final,d=self.calcula_J(X,centroides)
        it=0
        while abs(j-j_final)>tol and it<max_iter:    
            labels=np.argmin(d,axis=1)
            centroides=self.atualiza_centroides(X,labels,self.k)
            j=j_final
            labels,j_final,d=self.calcula_J(X,centroides)
            it+=1
        return j,labels,centroides
```
* Plotamos o grafico de cotovelo para nos auxiliar a identificar a quantidade ideal de clusters:
  <img width="1190" height="390" alt="image" src="https://github.com/user-attachments/assets/b774141a-2991-4d56-b61f-f798c4b3af68" />

---
* Chamamos nossa função principal
  ```python
  kmeans=Kmeans(k=4)
  centroides=kmeans.centroides_iniciais()
  j,labels,centroides=kmeans.run_kmeans(dataset,centroides,tol,max_iter)
  ```
* por fim plotamos nosso resultado:
<img width="1028" height="398" alt="image" src="https://github.com/user-attachments/assets/5da35aac-bf8b-4645-ac9f-e79d15cae3d1" />




