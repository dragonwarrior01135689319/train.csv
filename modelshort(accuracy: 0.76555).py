from sklearn.linear_model import LogisticRegression
import pandas as pd
import math

# processed train data
family_content = []
df = pd.read_csv(r'C:\Users\User-PC\Downloads\titanic_machine_learning\train.csv')
df = df.drop(['Name', 'Ticket', 'Cabin'], axis='columns')

for i in range(len(df.PassengerId)):
    family_content.append(df.SibSp[i] + df.Parch[i])

dummies = pd.get_dummies(df.Embarked)
dummiesSex = pd.get_dummies(df.Sex)

df = pd.concat(
    [df.drop(['SibSp', 'Parch', 'Embarked', 'Sex'], axis='columns'), pd.DataFrame({'Family': family_content}), dummies,
     dummiesSex],
    axis='columns')
df = df.drop(['C', 'female'], axis='columns')

df['Age'] = df['Age'].fillna(math.floor(df.Age.median()))
# processed train data

# fitting models into LogisticRegression
model = LogisticRegression()
model.fit(df[['Pclass', 'Age', 'Fare', 'Family', 'Q', 'S', 'male']], df.Survived)
# fitting models into LogisticRegression
# can test with "print(model.predict([[3,22,22,2,1,0,0]]))"

# processed test data
family_content = []
df = pd.read_csv(r'C:\Users\User-PC\Downloads\titanic_machine_learning\test.csv')
df = df.drop(['Name', 'Ticket', 'Cabin'], axis='columns')

for i in range(len(df.PassengerId)):
    family_content.append(df.SibSp[i] + df.Parch[i])

dummies = pd.get_dummies(df.Embarked)
dummiesSex = pd.get_dummies(df.Sex)

df = pd.concat(
    [df.drop(['SibSp', 'Parch', 'Embarked', 'Sex'], axis='columns'), pd.DataFrame({'Family': family_content}), dummies,
     dummiesSex],
    axis='columns')
df = df.drop(['C', 'female'], axis='columns')

df['Age'] = df['Age'].fillna(math.floor(df.Age.median()))
# processed test data

#finding nan spaces in the datasheet(throws error)
#x=0
#for i in range(len(df.PassengerId)):
#    x=x+1
#    math.floor(model.predict([[df.Pclass[i], df.Age[i], df.Fare[i], df.Family[i], df.Q[i], df.S[i], df.male[i]]]))
#    print(x)
#result: when i=152 there's a nan
#print(df.Pclass[152], df.Age[152], df.Fare[152], df.Family[152], df.Q[152], df.S[152], df.male[152])
#df.Fare[152]=nan

df['Fare'] = df['Fare'].fillna(math.floor(df.Fare.median()))

# predicting results
resultId = []
resultSur = []
for i in range(len(df.PassengerId)):
    resultId.append(df.PassengerId[i])
    resultSur.append(
        math.floor(model.predict([[df.Pclass[i], df.Age[i], df.Fare[i], df.Family[i], df.Q[i], df.S[i], df.male[i]]])))

result = pd.DataFrame({'PassengerId': resultId, 'Survived': resultSur})
print(result)
result.to_csv(r'C:\Users\User-PC\Downloads\titanic_machine_learning\result.csv', index=False)
# predicting results
