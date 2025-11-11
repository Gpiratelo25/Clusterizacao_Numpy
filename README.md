# Clusterização do zero com Numpy

## Projeto de clusterização na base wine do scikit learn com numpy puro

### Objetivo

** Entender o funcionamento do kmeans **

## 1. O que é clusterização

-> Faz parte do aprendizado não supervisionado de máquina, onde a intenção é entender o dataset e agrupar os dados de uma forma que faça sentido, uma vez que não temos variável de resposta.
-> E através do algoritmo vamos descobrir a estrutura dos dados.

## 1.1 K-MEANS

-> O objetivo do k-means é atraves de "k" clusters, encontrar quais pontos temos a menor distancia entre os dados
-> Definimos nesse problema k=3, então de início o algoritmo escolhe aleatóriamente três pontos dado como centroides
-> E vamos fazer o cálculo de distancia para cada ponto do datasete e atribuir ao centroide com menor distancia
-> Depois vamos pegar a média desses centroides, daí que vem o nome "means" e calcular a distancia até a convergencia, ou seja, até que as médias não tenham movimento negativo ou atinjam a quantidade que definimos como máxima que é 300.

- **Distância Euclidiana**  
  \[
  d(x, c) = \sqrt{(x_1 - c_1)^2 + (x_2 - c_2)^2}
  \]
- **Função de custo (inércia)**  
  \[
  J = \sum_i \min_k ||x_i - c_k||^2
  \]
  - **Atualização dos centróides**  
  \[
  c_k = \frac{1}{|S_k|} \sum_{x_i \in S_k} x_i
  \]


* Função que usamos para calcular a distancia entre os dados e os centróides
* Seguindo a fórmula fazemos a soma dos quadrados das diferenças
* E depois passsamos para `j` que é a nossa função de custo os minimos dessas diferenças

```python
    def calcula_J(self,dataset,centroides):
        """Função para calcular a menor distancia entre os pontos e os centroides
        parametros de entrada : dataset e os centroides_iniciais
        Retorno: Menores distancias e menor J"""
        diff =dataset[:,None,:]-centroides[None,:,:]
        d=np.sum(diff**2,axis=2)
        j=np.sum(np.min(d,axis=1))
        menores=np.argmin(d,axis=1)
        return menores,j,d
```


