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
# processed test data

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
