import pandas as pd 
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    uri = 'https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv'
    data = pd.read_csv(uri)

    x_raw = data[["home","how_it_works","contact"]] 
    y_raw = data[["bought"]]

    SEED = 20 

    #slip in train and test -> 75% to train
    #len_rows = x_raw.shape[0]
    #len_rows_75 = int(len_rows*0.75)
    #train_x = x_raw[:len_rows_75]
    #train_y = y_raw[:len_rows_75]
    #test_x = x_raw[len_rows_75:]
    #test_y = y_raw[len_rows_75:]

    train_x, test_x, train_y, test_y = train_test_split(x_raw, y_raw, random_state= SEED, test_size= 0.25)


    model = LinearSVC()
    model.fit(train_x, train_y)
    p = model.predict(test_x)

    #accuracy 
    hit_rate = accuracy_score(test_y, p)

    print(hit_rate)


    

if __name__ == '__main__':
    main()