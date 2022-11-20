import nltk
import pandas as pd
import numpy as np
from collections import Counter
import random
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def tokenize(text):
    '''for simbol in ".,:;@?!%/0123456789#":
        text = text.replace(simbol, ' ')'''
    text = text.lower()   #textul este trecut in lower pentru ca poate exista un cuvant scris in doua moduri diferite (che/CHE)
    text = nltk.TweetTokenizer().tokenize(text) #se imparte propozitia in cuvinte si intoarce o lista
    return text   #se intoarce o lista de cuvinte


def get_representation(toate_cuvintele, how_many):

    most_comm = toate_cuvintele.most_common(how_many)  #selecteaza cele mai comune 100 de cuvinte
    wd2idx = {}  #dictionare goale
    idx2wd = {}
    for idx, itr in enumerate(most_comm):  #un element al enumeratiei are doua parti: indicele si continutul
        cuvant = itr[0]        #preia cuvantul
        wd2idx[cuvant] = idx   #indexul asociat cuvantului
        idx2wd[idx] = cuvant   #cuvantul asociat indexului
    return wd2idx, idx2wd      #intoarce dictionarele


def get_corpus_vocabulary(corpus):

    counter = Counter()            #container gol in care adaugam date prin update()
    #cnt = 1
    for text in corpus:
        tokens = tokenize(text)       #tokenizeaza fiecare text in parte
        #print(cnt)
        #cnt = cnt + 1
        #print(tokens)
        counter.update(tokens)
    return counter


def text_to_bow(text, wd2idx):

    features = np.zeros(len(wd2idx)) #populam cu 100 de zerouri

    for token in tokenize(text):       #iteratorul se plimba prin fiecare cuvant din lista
      if token not in ".,/0123456789|?!%$&;":  #se exclud semnele de punctuatie/cifrele/textul de tip link si cuvintele de o litera
        if token.find("https://")!=1:
          if len(token)>1:
            if token in wd2idx:        #daca se gaseste in dictionar
              features[wd2idx[token]] += 1   #creste nr. de aparitii
    #print(features)
    return features


def corpus_to_bow(corpus, wd2idx):


    all_features = []
    for text in corpus:
        all_features.append(text_to_bow(text, wd2idx))  #transforma fiecare text intr-un model bag-of-words
    all_features = np.array(all_features)   #transforma lista intr-un array
    return all_features


def write_prediction(out_file, predictions):

    with open(out_file, 'w') as fout:   #deschidem fisierul
        fout.write('id,label\n')
        start_id = 5001
        for i, pred in enumerate(predictions):
            linie = str(i + start_id) + ',' + str(pred) + '\n'   #se scriu predictiile in fisier
            fout.write(linie)
    fout.close()  # se inchide fisierul

'''
def split(date, labels, procentaj_valid=0.25):   #se imparte 75% cu 25%
    indici = np.arange(0, len(labels))  #indicii celor 5000 de texte
    random.shuffle(indici)      #se face random ca sa nu avem mereu acelasi set de date
    N=int((1-procentaj_valid)*len(labels))
    train = date[indici[:N]]        #datele de train
    y_train = labels[indici[:N]]
    valid = date[indici[N:]]         #datele de test
    y_valid = labels[indici[N:]]
    return train, valid, y_train, y_valid
'''

def cross_validate(k, data, labels):
    chunk_size=int(len(labels)/k)   #o parte are k elemente
    indici = np.arange(0, len(labels))   #indicii celor 5000 de texte
    random.shuffle(indici)       #se face random ca sa nu avem mereu acelasi set de date
    cnt = 1;         #contorul numara partea curenta
    for i in range(0, len(labels),chunk_size):   #trecem prin indici din k in k
        valid_indici = indici[i:i+chunk_size]    #indicii de pentri test
        right_side = indici[i+chunk_size:]      #indicii din dreapta
        left_side = indici[0:i]                 #indicii din stanga
        train_indici = np.concatenate([left_side, right_side])   #se concateneaza cei din stanga cu cei din dreapta
        train = data[train_indici]   #datele de train
        valid = data[valid_indici]
        y_train = labels[train_indici]  #datele de test
        y_valid= labels[valid_indici]


        clf.fit(train,y_train)     #se introduc datele de train (toate mai putin partea aleasa pentru test)
        vprd=clf.predict(valid)    #calculam predictiile
        acc= (vprd== y_valid).mean()  #calculam acuratetea
        print("Partea ",cnt,":")  #afisam partea
        print("Acuratete: ",acc)  #afisam acuratetea
        matrice=confusion_matrix(y_valid,vprd)  #calculam matricea de confuzie(true pozitive/true negative/false pozitive/false negative)
        print("Matrice: ")
        print(matrice)  #afisam matricea
        print(" ")
        cnt=cnt+1;    #contorul se mareste cu 1
        
        


train_df = pd.read_csv('train.csv')          #se citesc datele
test_df = pd.read_csv('test.csv')
corpus = train_df['text']                    #aducem toate textele intr-un singur loc
text = train_df['text'][2]
toate_cuvintele = get_corpus_vocabulary(corpus) #selectam toate cuvintele din corpus
#print(get_corpus_vocabulary(corpus))
wd2idx, idx2wd = get_representation(toate_cuvintele, 100)  #populam cele doua dictionare


data = corpus_to_bow(corpus, wd2idx)     #transformam multimea de cuvinte intr-un bag of words (lista)
labels = train_df['label']               #etichetele textelor

test_data = corpus_to_bow(test_df['text'], wd2idx)  #se apeleaza aceeasi metoda pentru datele de test
#print(test_data.shape)

#predictii = np.ones(len(test_data))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

vec=[x for x in range (1,70) if x%2!=0]   #vom incerca sa gasim un k optim pentru KNN (doar nr. impare)

rez=[]
for k in vec:
  clf= KNeighborsClassifier(k)   #pentru fiecare valoare din vec se apeleaza metoda KNN
  #clf.fit(data,labels)
  #predictii = clf.predict(test_data)
  #write_prediction('submission.csv', predictii)  
  scor = cross_val_score(clf, data, labels, cv=10 , scoring='accuracy')  #se efectueaza 10fold cross-validation pentru fiecare k iar rezultatele sunt trecute in scor
  rez.append(scor.mean())  #in vectorul cu rezultate se introduce media scorurilor

err=[1-x for x in rez]       #erorile corespunzatoare rezultatelor
opt_k_i=err.index(min(err))  #se alege k-ul corespunzator celei mai mici erori
opt_k=vec[opt_k_i]
print(opt_k)              #afisam k optim

plt.plot(vec,err)        #se afiseaza si graficul lui k in functie de eroare
plt.xlabel('Vecini')
plt.ylabel('Eroare')
plt.show()

clf= KNeighborsClassifier(opt_k)   #se apeleaza KNN pentru acel k optim
clf.fit(data,labels)              #se introduc datele in model
print("Data fitted")
predictii = clf.predict(test_data)  #se calculeaza predictiile
write_prediction('submission.csv', predictii)  #se scriu predictiile in fisier

cross_validate(10,data,labels)   #apelam cross-validation pentru datele de train; testam performanta

'''
#train, valid, y_train, y_valid = split(data, labels, 0.25)   #testam clasificatorul si pe multimea de train
#clf.fit(train, y_train)       #se introduc datele in model

vprd = clf.predict(valid)     #se calculeaza predictiile
acc= (vprd== y_valid).mean()  #se calculeaza acuratetea
print("Split: ",acc)         #se afiseaza acuratetea
print(" ")

matrice=confusion_matrix(y_valid,vprd)     #se calculeaza matricea de confuzie(true pozitive/true negative/false pozitive/false negative)
print(matrice) #se afiseaza matricea
'''


