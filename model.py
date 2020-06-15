from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import math

#processed data
family_content=[]
df=pd.read_csv(r'C:\Users\User-PC\Downloads\train.csv')
df=df.drop(['Name','Ticket','Cabin'],axis='columns')

for i in range(len(df.PassengerId)):
    family_content.append(df.SibSp[i]+df.Parch[i])

dummies=pd.get_dummies(df.Embarked)

df=pd.concat([df.drop(['SibSp','Parch','Embarked'],axis='columns'),pd.DataFrame({'Family':family_content}),dummies],axis='columns')
df=df.drop(['C'],axis='columns')

df['Age'] = df['Age'].fillna(math.floor(df.Age.median()))
#processed data

#distributing male and female
IDmale=[]
Surmale=[]
Classmale=[]
Agemale=[]
Faremale=[]
Famale=[]
Qmale=[]
Smale=[]
IDfemale=[]
Surfemale=[]
Classfemale=[]
Agefemale=[]
Farefemale=[]
Fafemale=[]
Qfemale=[]
Sfemale=[]

for i in range(len(df.PassengerId)):
    if df.Sex[i]=='male':
        IDmale.append(df.PassengerId[i])
        Surmale.append(df.Survived[i])
        Classmale.append(df.Pclass[i])
        Agemale.append(df.Age[i])
        Faremale.append(df.Fare[i])
        Famale.append(df.Family[i])
        Qmale.append(df.Q[i])
        Smale.append(df.S[i])
    else:
        IDfemale.append(df.PassengerId[i])
        Surfemale.append(df.Survived[i])
        Classfemale.append(df.Pclass[i])
        Agefemale.append(df.Age[i])
        Farefemale.append(df.Fare[i])
        Fafemale.append(df.Family[i])
        Qfemale.append(df.Q[i])
        Sfemale.append(df.S[i])

maledf = pd.DataFrame(
            {'PassengerId': IDmale, 'Survived': Surmale,
             'Pclass': Classmale, 'Age': Agemale,
             'Fare': Faremale, 'Family': Famale, 'Q': Qmale,
             'S':Smale})
femaledf = pd.DataFrame(
            {'PassengerId': IDfemale, 'Survived': Surfemale,
             'Pclass': Classfemale, 'Age': Agefemale,
             'Fare': Farefemale, 'Family': Fafemale, 'Q': Qfemale,
             'S':Sfemale})
#done distributing to maledf and female df

#fitting models into LogisticRegression
modelmale=LogisticRegression()
modelmale.fit(maledf[['Pclass','Age','Fare','Family','Q','S']],maledf.Survived)
modelfemale=LogisticRegression()
modelfemale.fit(femaledf[['Pclass','Age','Fare','Family','Q','S']],femaledf.Survived)
#fitting models into LogisticRegression
# can test with "print(math.floor(modelmale.predict(df[[class,Age,Fare,Family,Q,S]])))"

#processed data
family_content=[]
df=pd.read_csv(r'C:\Users\User-PC\Downloads\test.csv')
df=df.drop(['Name','Ticket','Cabin'],axis='columns')

for i in range(len(df.PassengerId)):
    family_content.append(df.SibSp[i]+df.Parch[i])

dummies=pd.get_dummies(df.Embarked)

df=pd.concat([df.drop(['SibSp','Parch','Embarked'],axis='columns'),pd.DataFrame({'Family':family_content}),dummies],axis='columns')
df=df.drop(['C'],axis='columns')

df['Age'] = df['Age'].fillna(math.floor(df.Age.median()))
#processed data

#distributing male and female
IDmale=[]
Classmale=[]
Agemale=[]
Faremale=[]
Famale=[]
Qmale=[]
Smale=[]
IDfemale=[]
Classfemale=[]
Agefemale=[]
Farefemale=[]
Fafemale=[]
Qfemale=[]
Sfemale=[]

for i in range(len(df.PassengerId)):
    if df.Sex[i]=='male':
        IDmale.append(df.PassengerId[i])
        Classmale.append(df.Pclass[i])
        Agemale.append(df.Age[i])
        Faremale.append(df.Fare[i])
        Famale.append(df.Family[i])
        Qmale.append(df.Q[i])
        Smale.append(df.S[i])
    else:
        IDfemale.append(df.PassengerId[i])
        Classfemale.append(df.Pclass[i])
        Agefemale.append(df.Age[i])
        Farefemale.append(df.Fare[i])
        Fafemale.append(df.Family[i])
        Qfemale.append(df.Q[i])
        Sfemale.append(df.S[i])
#done distributing to maledf and female df
# can test with "print(math.floor(modelmale.predict(df[[class,Age,Fare,Family,Q,S]])))"

#Faremale[96]=nan
Faremaledf=pd.DataFrame({'Faremale':Faremale})
Faremaledf['Faremale'] = Faremaledf['Faremale'].fillna(math.floor(Faremaledf.Faremale.median()))
Farefemaledf=pd.DataFrame({'Farefemale':Farefemale})
Farefemaledf['Farefemale'] = Farefemaledf['Farefemale'].fillna(math.floor(Farefemaledf.Farefemale.median()))

#predicting result
testMaleSur=[]
testMaleId=[]
testFemaleSur=[]
testFemaleId=[]
for i in range(len(IDmale)):
    testMaleId.append(IDmale[i])
    testMaleSur.append(math.floor(modelmale.predict([[Classmale[i],Agemale[i],Faremaledf.Faremale[i],Famale[i],Qmale[i],Smale[i]]])))

for i in range(len(IDfemale)):
    testFemaleId.append(IDfemale[i])
    testFemaleSur.append(math.floor(modelfemale.predict([[Classfemale[i],Agefemale[i],Farefemaledf.Farefemale[i],Fafemale[i],Qfemale[i],Sfemale[i]]])))

resultId=testMaleId+testFemaleId
resultSur=testMaleSur+testFemaleSur

result=pd.DataFrame({'PassengerId':resultId,'Survived':resultSur})
print(result)
result.to_csv(r'C:\Users\User-PC\Downloads\result.csv')
