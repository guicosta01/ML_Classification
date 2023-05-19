from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# classification 0 or 1

# 0 = dog // 1 = pig

# guess by features 

#dog or pig
#long hair 
#short leg 
#song "Auau"

def main():
    #my code
    pig1 = [0,1,0]
    pig2 = [0,1,1]
    pig3 = [1,1,0]

    dog1 = [0,1,1]
    dog2 = [1,0,1]
    dog3 = [0,1,1]

    train_x  = [pig1,pig2,pig3,dog1,dog2,dog3]
    train_y = [1,1,1,0,0,0]

    # f(x) = y 
    model = LinearSVC()
    model.fit(train_x, train_y)

    mistery_animal1 = [1,1,1]
    mistery_animal2 = [1,1,0]
    mistery_animal3 = [0,1,1]

    test_x = [mistery_animal1 , mistery_animal2 , mistery_animal3]
    test_y = [0 , 1 ,1]
    
    p = model.predict(test_x)

    #accuracy 
    hit_rate = accuracy_score(test_y, p)

    print(hit_rate)




if __name__ == '__main__':
    main()