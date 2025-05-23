import pandas as pd
import matplotlib.pyplot as plt
titanic=pd.read_csv('Titanic-Dataset.csv')
titanic_clean=titanic.drop(columns=['PassengerId','Name','Ticket','Cabin'])
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.hist(titanic_clean['Age'].dropna(),bins=20,color='skyblue',edgecolor='black')
plt.title('age distribution')
plt.xlabel('age')
plt.ylabel('count')

plt.subplot(1,3,2)
survival_by_gender=titanic_clean.groupby(['Sex','Survived']).size().unstack()
survival_by_gender.plot(kind='bar',stacked=True,color=['red','green'])
plt.title('survival by gender')
plt.xlabel('gender')
plt.ylabel('count')
plt.xticks(rotation=0)
plt.legend(['Died','Survived'])

plt.subplot(1,3,3)
titanic_clean.boxplot(column='Fare',by='Pclass',patch_artist=True,boxprops=dict(facecolor='lightblue'))
plt.title('fare distribution by class')
plt.xlabel('passenger class')
plt.ylabel('fare')
plt.suptitle('')
plt.tight_layout()

plt.show()