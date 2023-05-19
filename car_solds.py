import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler


def main():
    uri = "https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv"
    data = pd.read_csv(uri) 
    

    change = { 
        "no":0,
        "yes":1
    }
    data['sold'] = data.sold.map(change)

    current_year = datetime.today().year
    data['model_year'] = current_year - data.model_year

    data['km_per_year'] = data.mileage_per_year * 1.60934

    x_raw = data[["price", "model_year", "km_per_year"]]  
    y_raw = data[["sold"]]  

    SEED = 5
    #fix
    np.random.seed(SEED)
    train_x, test_x, train_y, test_y = train_test_split(x_raw, y_raw, random_state= SEED, test_size= 0.25)

    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    model = SVC(gamma='auto')
    model.fit(train_x, train_y)
    p = model.predict(test_x)

    #accuracy 
    hit_rate = accuracy_score(test_y, p)


    #base line
    dummy = DummyClassifier()
    dummy.fit(train_x, train_y)
    p_d = dummy.predict(test_x)
    #accuracy 
    hit_rate_dummy = accuracy_score(test_y, p_d)


    print(hit_rate)




if __name__ == "__main__":
    main()