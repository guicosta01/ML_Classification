import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np

def main():

    uri = 'https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv'
    data = pd.read_csv(uri) 

    change = { 
        0:1,
        1:0
    }
    data['finished'] = data.unfinished.map(change)

    sns.scatterplot(x = 'expected_hours', y="price", hue= "finished", data= data)

    sns.relplot(x = 'expected_hours', y="price", col= "finished", data= data)

    plt.show()
    
    x_raw = data[["expected_hours","price"]] 
    y_raw = data[["finished"]]

    SEED = 5
    #fix
    np.random.seed(SEED)
    train_x, test_x, train_y, test_y = train_test_split(x_raw, y_raw, random_state= SEED, test_size= 0.25)


    model = SVC(gamma='auto')
    model.fit(train_x, train_y)
    p = model.predict(test_x)

    #accuracy 
    hit_rate = accuracy_score(test_y, p)

    print(hit_rate)



if __name__ == "__main__":
    main()