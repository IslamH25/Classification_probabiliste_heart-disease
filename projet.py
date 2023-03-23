import math # for math
from utils import AbstractClassifier
import utils
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

def getPrior(df) :
    """
    Rend un dictionnaire contenant 3 clés 'estimation', 'min5pourcent', 'max5pourcent' 
  
    Arguments:
        df {pandas.dataframe} -- le pandas.dataframe contenant les données  
    """
    p=df['target'].mean()
    ecartType=df['target'].std()
    n=df.shape[0]
    interv_min=p-1.96*(ecartType/math.sqrt(n))
    interv_max=p+1.96*(ecartType/math.sqrt(n))
    dico={}
    dico['estimation']=p
    dico['min']=interv_min
    dico['max']=interv_max
    return dico

def nbParams(df,liste=None):
    """
    Rend la taille mémoire de ces tables 𝑃(𝑡𝑎𝑟𝑔𝑒𝑡|𝑎𝑡𝑡𝑟1,..,𝑎𝑡𝑡𝑟𝑘)
  
    Arguments:
        df {pandas.dataframe} -- le pandas.dataframe contenant les données
        liste [𝑡𝑎𝑟𝑔𝑒𝑡,𝑎𝑡𝑡𝑟1,...,𝑎𝑡𝑡𝑟𝑙] en supposant qu'un float est représenté sur 8octets.
    """
    if(liste==None):
        liste=df.columns
    n=len(liste)
    nb=1
    for i in range(0,n):
        nb*=len(df[liste[i]].unique())
        
    print(n,' variable(s):',nb*8,' octets ',tailleMemoire(nb*8))
    
def tailleMemoire(nbOctets):
    """
    Rend la conversion des octets en kiloOctet ou migaOctet ou gigaOctets

    Arguments:
        nbOctets
    """
    ko=nbOctets//math.pow(2,10)
    s=""
    if(ko!=0):
        o=nbOctets%math.pow(2,10)
        s="= "+str(ko)+"ko "+str(o)+"o"
        mo=ko//math.pow(2,10)
        if(mo!=0):
            ko=ko%math.pow(2,10)
            s="= "+str(mo)+"mo "+str(ko)+"ko "+str(o)+"o"
            go=mo//math.pow(2,10)
            if(go!=0):
                mo=mo%math.pow(2,10)
                s="= "+str(go)+"go "+str(mo)+"mo "+str(ko)+"ko "+str(o)+"o"
    
    return s

def nbParamsIndep(df,liste=None):
    """
    Rend la taille mémoire nécessaire pour représenter les tables de probabilité
  
    Arguments:
        df {pandas.dataframe} -- le pandas.dataframe contenant les données
        liste -- aucune liste
    """
    if(liste==None):
        liste=df.columns
    n=len(liste)
    nb=0
    for i in range(0,n):
        nb+=len(df[liste[i]].unique())
        
    print(n,' variable(s):',nb*8,' octets ',tailleMemoire(nb*8))

class APrioriClassifier (AbstractClassifier) :
    """
    Un classifier estime la classe d'un seul attribut
    et calcule les statistiques de reconnaissance à partir d'un pandas.dataframe.
    """
    def __init__(self):
        """
        Initialise un attribut contenant la valeur de l'estimation (sois 1 sois 0)
        """
        att=pd.read_csv("test.csv")
        prior=getPrior(att)
        if(prior['estimation']>0.5):
            self.class_found=1
        else:
            self.class_found=0
    
    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        Arguments:
            param attrs: le  dictionnaire nom-valeur des attributs
            return: la classe 0 ou 1 estimée
        """
        return self.class_found

    def statsOnDF(self, df):
        """
        Rend un dictionnaire contenant 6 clés VP,FP,VN,FN,précision et rappel

        Arguments:
            df:  le dataframe à tester
        """
        StatDico={'VP':0,'FP':0,'VN':0,'FN':0,'précision':0,'rappel':0}
        for t in df.itertuples():
            dic=t._asdict()
            dic.pop('Index',None)
            if dic['target']==1:
                if self.estimClass(dic)==1:
                    StatDico['VP']+=1
                else:
                    StatDico['FN']+=1
            else:
                if self.estimClass(dic)==1:
                    StatDico['FP']+=1
                else:
                    StatDico['VN']+=1
        StatDico['précision']= StatDico['VP']/(StatDico['VP']+StatDico['FP'])
        StatDico['rappel']=StatDico['VP']/ (StatDico['VP']+StatDico['FN'])
        return StatDico

def P2D_l(df, attr):
    '''
    Rend un dictionnaire asssociant à la valeur 𝑡 un dictionnaire associant à la valeur 𝑎 la probabilité 𝑃(𝑎𝑡𝑡𝑟=𝑎|𝑡𝑎𝑟𝑔𝑒𝑡=𝑡)    
    
    Arguments:
        df {pandas.dataframe} -- le pandas.dataframe contenant les données
        attribut
    '''
    dict_ret = dict()
    possible_target_values = df.target.unique()
    for target_val in possible_target_values:
        df_target = df[df.target == target_val]
        nb_same_target = df_target.shape[0]
        dict_ret[target_val] = dict()

        possible_attr_values = df[attr].unique()
        for attr_val in possible_attr_values:
            df_same_attr = df_target[df_target[attr]==attr_val]
            nb_same_attr_val = df_same_attr.shape[0]
            proba = nb_same_attr_val / nb_same_target
            dict_ret[target_val][attr_val] = proba

    return dict_ret 

def P2D_p(df, attr):
    '''
    Rend un dictionnaire associant à la valeur 𝑎 un dictionnaire asssociant à la valeur 𝑡 la probabilité 𝑃(𝑡𝑎𝑟𝑔𝑒𝑡=𝑡|𝑎𝑡𝑡𝑟=𝑎)

    Arguments:
        df {pandas.dataframe} -- le pandas.dataframe contenant les données
        attribut
    '''
    dict_ret = dict()
    possible_attr_values = df[attr].unique()
    for attr_val in possible_attr_values:
        df_same_attr = df[df[attr] == attr_val]
        nb_same_attr_val = df_same_attr.shape[0]
        dict_ret[attr_val] = dict()

        possible_target_values = df.target.unique()
        for target_val in possible_target_values:
            df_target = df_same_attr[df_same_attr.target==target_val]
            nb_same_target = df_target.shape[0]
            proba = nb_same_target/ nb_same_attr_val
            dict_ret[attr_val][target_val] = proba



    return dict_ret                     

class ML2DClassifier (APrioriClassifier):
    '''
    Classifie selon le max de vraisemblance
    '''
    def __init__(self, attrs, attr):
        self.probas = P2D_l(attrs, attr)
        self.attr = attr

    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        if self.probas[0][attrs[self.attr]] > self.probas[1][attrs[self.attr]]:
            return 0

        return 1

class MAP2DClassifier (APrioriClassifier):
    '''
    Classifie selon le maximum a posteriori (MAP)
    '''
    def __init__(self, attrs, attr):
        self.probas = P2D_p(attrs, attr)
        self.attr = attr

    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1
        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        if self.probas[attrs[self.attr]][0] > self.probas[attrs[self.attr]][1]:
            return 0

        return 1

def drawNaiveBayes(df,attr):
    '''
    Dessine le graphe a partir d'un dataframe et du nom de la colonne qui est la classe

    Arguments:
        df {pandas.dataframe} -- le pandas.dataframe contenant les données
        attribut
    '''
    liste=df.columns
    n=len(liste)
    s=""
    for i in range(0,n-1):
        if(liste[i]!=attr):
            s+=attr+"->"+liste[i]+";"
    if(liste[n-1]!=attr):        
        s+=attr+"->"+liste[n-1]
    
    return utils.drawGraph(s)

        
def nbParamsNaiveBayes (data,target,attr=None) :
    '''
    Rend la taille mémoire nécessaire pour représenter les tables de probabilité 
    Arguments:
        df {pandas.dataframe} -- le pandas.dataframe contenant les données
        target
        liste d'attributs -- (aucune)
    '''
    if attr == None :
        attr = data.columns
        
    if(len(attr)!=0): 
        valeur=2
        for col in attr:
            if(col!='target'):
                valeur += (len(data[col].unique()))*2
    else:
        valeur=2
    print(len(attr),"Variable(s) : ", valeur*8," octets ",tailleMemoire(valeur*8))

class MLNaiveBayesClassifier(APrioriClassifier):
    '''
    Utilise le maximum de vraisemblance (ML) et le maximum a posteriori (MAP) pour estimer la classe d'un individu en utilisant l'hypothèse du Naïve Bayes.
    '''
    def __init__(self,df):
        self.attrs=df.columns
        self.listeTableP2DL=[]
        
        for col in self.attrs:
            if(col!='target'):
                self.listeTableP2DL.append(P2D_l(df,col))
    
    
    def estimProbas(self,attrs):
        '''
        Rend le calcule de la vraisemblance 
        
        Arguments:
            df {pandas.dataframe} -- le pandas.dataframe contenant les données
            liste d'attributs
        '''
        dic=dict()
        a=1
        b=1
        i=0
        for col in self.attrs:
            if(col!='target'): 
                if(attrs[col] in self.listeTableP2DL[i][1] and attrs[col] in self.listeTableP2DL[i][0] ):
                    a*=self.listeTableP2DL[i][1][attrs[col]]
                    b*=self.listeTableP2DL[i][0][attrs[col]]
                    i+=1 
                else:
                    a=0
                    b=0
                    i+=1            
                
        dic[0]=b
        dic[1]=a
        return dic
        
        
    def estimClass(self,attrs):
        '''
        Rend la proba la plus elever de target grace a estimProba(attrs)

        Arguments:
            liste d'attributs
        '''
        dic=self.estimProbas(attrs)
        if(dic[0]>=dic[1]):
            return 0
        else:
            return 1

        
#--------------------------------------------------------------------------------------------------------------------------       
def probaAttribut(df):
    '''
    Rend le calcule de la probabilité de chaque attribut de chaque colonne du df(dataFrame)
    
    Arguments:
            df {pandas.dataframe} -- le pandas.dataframe contenant les données
            liste d'attributs
    '''
    valeurs=df.columns
    n=df.shape[0]
    dic={valeurs[i]: {} for i in range(0,len(valeurs))}
    for i in range(0,len(valeurs)):
        vals=df[valeurs[i]].unique()
        dic[valeurs[i]]={vals[j]: 0 for j in range(0,len(vals))}
        
    for t in df.itertuples():
        dic_t=t._asdict()
        for i in range(0,len(valeurs)):
            v=dic_t[valeurs[i]]
            dic[valeurs[i]][v]+=1/n
            
    return dic

class MAPNaiveBayesClassifier(APrioriClassifier):
    '''
    Utilise le maximum de vraisemblance (ML) et le maximum a posteriori (MAP) pour estimer la classe d'un individu en utilisant l'hypothèse du Naïve Bayes.
    '''
    def __init__(self,df):
        self.attrs=df.columns
        self.listeTableP2DL=[]
        self.df=df
        self.proba=getPrior(self.df)['estimation']
        self.dfProba=probaAttribut(df)
        for col in self.attrs:
            if(col!='target'):
                self.listeTableP2DL.append(P2D_l(df,col))
                                           
    def estimProbas(self,attrs):
        '''
        Rend le calcule de la vraisemblance 
        
        Arguments:
            df {pandas.dataframe} -- le pandas.dataframe contenant les données
            liste d'attributs
        '''
        dic=dict()
        a=1
        b=1
        i=0
        for col in self.attrs:
            if(col!='target'):
                if(attrs[col] in self.listeTableP2DL[i][0] and attrs[col] in self.listeTableP2DL[i][1]):
                    a*=self.listeTableP2DL[i][1][attrs[col]]/self.dfProba[col][attrs[col]]
                    b*=self.listeTableP2DL[i][0][attrs[col]]/self.dfProba[col][attrs[col]]
                    i+=1
                else:
                    a=0
                    b=0
                    i+=1
        
        proba=self.proba             
        dic[0]=b*(1-proba)
        dic[1]=a*proba
        return dic

    def estimClass(self,attrs):
        '''
        Rend la proba la plus elever de target grace a estimProba(attrs)

        Arguments:
            liste d'attributs
        '''
        dic=self.estimProbas(attrs)
        if(dic[0]>=dic[1]):
            return 0
        else:
            return 1

def isIndepFromTarget(df,attr,x):
    """
    Verifie si attr est indépendant de target au seuil de x%    
    df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
    attr: le nom d'une colonne du dataframe df.
    x: seuil de confiance.
    """
    listeVal=df[attr].unique()
    matContengence=np.zeros((2,listeVal.size), dtype=int)
    dictVal={listeVal[i]: i for i in range(len(listeVal))}
    for row in df.itertuples():
        rowDic=row._asdict()
        matContengence[rowDic['target'],dictVal[rowDic[attr]]]+=1
        
    _,p,_,_= scipy.stats.chi2_contingency(matContengence)
    
    return p>x

class ReducedMLNaiveBayesClassifier(APrioriClassifier):

    def __init__(self,df,x):
        self.attrs=df.columns
        self.listeTableP2DL=[]
        self.df=df
        self.df_=df.copy(deep=True)
        y=False
        
        for col in self.attrs:
            if(col!='target'):
                y=isIndepFromTarget(df,col,x)
                if(not y):
                    self.listeTableP2DL.append(P2D_l(df,col))
                else:
                    self.df_.drop([col],1,inplace=True)
                
        self.attrs=self.df_.columns
       
        
        
    def estimProbas(self,attrs):
        '''
        Rend le calcule de la vraisemblance 
        
        Arguments:
            df {pandas.dataframe} -- le pandas.dataframe contenant les données
            liste d'attributs
        '''
        dic=dict()
        a=1
        b=1
        i=0
        for col in self.attrs:
            if(col!='target'): 
                if(attrs[col] in self.listeTableP2DL[i][1] and attrs[col] in self.listeTableP2DL[i][0] ):
                    a*=self.listeTableP2DL[i][1][attrs[col]]
                    b*=self.listeTableP2DL[i][0][attrs[col]]
                    i+=1
                else:
                    a=0
                    b=0
                    i+=1
             
                
        dic[0]=b
        dic[1]=a
        return dic
        
        
    def estimClass(self,attrs):
        '''
        Rend la proba la plus elever de target grace a estimProba(attrs)

        Arguments:
            liste d'attributs
        '''
        
        dic=self.estimProbas(attrs)
        if(dic[0]>=dic[1]):
            return 0
        else:
            return 1

    def draw(self):
        return drawNaiveBayes(self.df_,'target')

class ReducedMAPNaiveBayesClassifier(APrioriClassifier):
    
    def __init__(self,df,x):
        self.attrs=df.columns
        self.listeTableP2DL=[]
        self.df=df
        self.df_=df.copy(deep=True)
        
        self.proba=getPrior(self.df)['estimation']
        self.dfProba=probaAttribut(df)
        
        
        y=False
        
        for col in self.attrs:
            if(col!='target'):
                y=isIndepFromTarget(df,col,x)
                if(not y):
                    self.listeTableP2DL.append(P2D_l(df,col))
                    
                else:
                    self.df_.drop([col],1,inplace=True)
                     
        self.attrs=self.df_.columns
        
    def estimProbas(self,attrs):
        '''
        Rend le calcule de la vraisemblance 
        
        Arguments:
            df {pandas.dataframe} -- le pandas.dataframe contenant les données
            liste d'attributs
        '''
        dic=dict()
        a=1
        b=1
        i=0
        for col in self.attrs:
            if(col!='target'):
                if(attrs[col] in self.listeTableP2DL[i][0] and attrs[col] in self.listeTableP2DL[i][1]):
                    a*=self.listeTableP2DL[i][1][attrs[col]]/self.dfProba[col][attrs[col]]
                    b*=self.listeTableP2DL[i][0][attrs[col]]/self.dfProba[col][attrs[col]]
                    i+=1
                else:
                    a=0
                    b=0
                    i+=1
        
                     
        dic[0]=b*(1-self.proba)
        dic[1]=a*self.proba
        return dic

    def estimClass(self,attrs):
        '''
        Rend la proba la plus elever de target grace a estimProba(attrs)

        Arguments:
            liste d'attributs
        '''
        dic=self.estimProbas(attrs)
        if(dic[0]>=dic[1]):
            return 0
        else:
            return 1
        
        
    def draw(self):
        return drawNaiveBayes(self.df_,'target')   

def mapClassifiers(dic,df):
    """
    Représente graphiquement ces classifiers dans l'espace 

    Argument:  
        dic: Dictionnaire de classifieur 
        df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
    """
    precision=[]
    rappel=[]
    
    for i in dic:
        precision.append(dic[i].statsOnDF(df)['précision'])
        rappel.append(dic[i].statsOnDF(df)['rappel'])
        
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("précision")
    ax.set_ylabel("rappel")
    ax.scatter(precision, rappel, marker = 'x', c = 'blue') 
    for i, nom in enumerate(dic):
        ax.annotate(nom, (precision[i], rappel[i]))
    
    plt.show()

def MutualInformation(df,x,y):
    """ 
    Calcule ces informations mutuelles
    
    Argument:  
        df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
        x: attribut target
        y: n'importe quel attribut sauf target 
    """
    n=df.shape[0]
    information=0.0
  
    
    for i in df[x].unique():
        px=0
        pyy=0
        pxy=0
        for j in df[y].unique():
            px=len(df[df[x]==i])/n
            py=len(df[df[y]==j])/n
            pxy=len(df[df[x]==i][df[y]==j])/n
            if(px!=0 and py!=0 and pxy!=0):
                information+=pxy*(math.log2(pxy)-math.log2(px*py))
    return information


def ConditionalMutualInformation(df,x,y,z):
    """ 
    Calcule ces informations mutuelles
    
    Argument:  
        df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
        x: n'importe quel attribut sauf target 
        y: n'importe quel attribut sauf target et attribut x
        z: attribut target
    """
    n=df.shape[0]
    information=0
    
    for i in df[x].unique():
        pxz=0
        pyz=0
        pz=0
        pxyz=0
        
        for j in df[y].unique():
            for k in df[z].unique():
                pxz=len(df[df[x]==i][df[z]==k])/n
                pyz=len(df[df[y]==j][df[z]==k])/n
                pz=len(df[df[z]==k])/n
                pxyz=len(df[df[x]==i][df[y]==j][df[z]==k])/n
                if(pxz!=0 and pyz!=0 and pz!=0 and pxyz!=0):
                    information+=pxyz*math.log2(pz*pxyz/(pxz*pyz))
                    
    return information

def MeanForSymetricWeights(mat):
    """ 
    Calcule la moyenne des poids pour une matrice
    
    Argument:  
        mat: matrice symétrique de diagonale nulle.
    """
    
    mean=0
    for i in range(0,len(mat)):
        for j in range(0,len(mat[0])):
            mean+=mat[i][j]
    mean/=(len(mat)*len(mat[0])-len(mat))  
    return mean

def SimplifyConditionalMutualInformationMatrix(matrice):
    """ 
    Annule toutes les valeurs plus petites que cette moyenne dans une matrice
    
    Argument:  
        mat: matrice symétrique de diagonale nulle.
    """
    mean=MeanForSymetricWeights(matrice)
    
    for i in range(len(matrice)):
        for j in range(len(matrice[0])):
            if(matrice[i][j]<mean):
                matrice[i][j]=0
        
    


        



