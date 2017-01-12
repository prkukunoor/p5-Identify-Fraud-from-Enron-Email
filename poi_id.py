#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt

sys.path.append("../tools/")

### Selected features

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold,StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi',
'salary',
"to_poi_fraction",
"from_poi_fraction",
"deferral_payments",
"total_payments",
"exercised_stock_options",
'bonus',
"restricted_stock",
'shared_receipt_with_poi',
"restricted_stock_deferred",
'total_stock_value',
'expenses',
'loan_advances',
'other',
'director_fees',
'deferred_income',
'long_term_incentive'
]
#] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict =pickle.load(data_file)
    
### Explore data
print "sample data:",data_dict.values()[0]
print "Number of datapoints:",len(data_dict.values())
numberOfPOI = 0
numberOfNonPOI = 0

for sample in data_dict.values():
    if sample["poi"] == 1:
        numberOfPOI += 1
    elif sample["poi"] == 0:
        numberOfNonPOI += 1
print "Number of POI:",numberOfPOI
print "Number of non-POI",numberOfNonPOI
print "Number of features in dataset:",len(data_dict.values()[0].keys())

from collections import defaultdict
numberOfMissingValuesPerFeature = defaultdict(int)
for sample in data_dict.values():
    for feature in data_dict.values()[0].keys():
        if sample[feature] == "NaN":
            numberOfMissingValuesPerFeature[feature] += 1
import pprint
pp = pprint.PrettyPrinter()
print "Number of missing values per feature:"
pp.pprint(numberOfMissingValuesPerFeature.items())
        
#Task 2: Remove outliers

import matplotlib.pyplot
import math
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

### remove NAN's from dataset
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
### uncomment for printing top 4 salaries
print outliers_final

### plot features

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


### Task 3: Create new feature(s)

for sample in data_dict.values():
    if sample["from_messages"] != "NaN" and sample["from_messages"] != 0 and sample["from_this_person_to_poi"] != "NaN":
        sample["to_poi_fraction"] = sample["from_this_person_to_poi"] * 1.0/sample["from_messages"]
    else:
        sample["to_poi_fraction"] = 0           
        
    if sample["to_messages"] != "NaN" and sample["to_messages"] != 0 and sample["from_poi_to_this_person"] != "NaN":
        sample["from_poi_fraction"] = sample["from_poi_to_this_person"] * 1.0/sample["to_messages"]
    else:
        sample["from_poi_fraction"] = 0

my_dataset = data_dict

### plot new features

for point in data:
    from_poi = point[1]
    to_poi = point[0]
    plt.scatter( from_poi, to_poi )
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("fraction of emails this person gets from poi")
plt.ylabel("fraction of emails this person sends to poi")        
plt.show()

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

for numOfFeatures in range(1,18):
    ch2 = SelectKBest(k=numOfFeatures)
    features_temp = ch2.fit_transform(features, labels)
    selected_feature_names = [features_list[i+1] for i in ch2.get_support(indices=True)]
    
    
    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html
    
    print "Selected feature names:",selected_feature_names
    dtParams = {
    "clf__min_samples_split":[2,3,4,5,6,7,8,9,10,20,30,40],
    "clf__max_depth":range(1,17)}
    dtclf = Pipeline(steps=[("minmaxer",MinMaxScaler()),("clf",DecisionTreeClassifier(random_state=42))])
    cv = StratifiedShuffleSplit(labels,n_iter=10,test_size=0.3,random_state=60)
    dtGridSearch = GridSearchCV(dtclf, param_grid=dtParams, cv = cv, scoring='f1')
    dtGridSearch.fit(features_temp,labels)
    print "\nDTC with",numOfFeatures,"features:",dtGridSearch.best_params_,dtGridSearch.best_score_
    
    svcParams = {
    'clf__C': [1e-5, 1e-2, 1e-1, 1, 10, 1e2, 1e5],
    'clf__gamma': [0.0],
    'clf__kernel': ['rbf'],    
    'clf__tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],  
    'clf__class_weight': [{True: 12, False: 1},
        {True: 10, False: 1},
        {True: 8, False: 1},
        {True: 15, False: 1},
        {True: 4, False: 1},
        'auto', None]
    }
    svcclf = Pipeline(steps=[("minmaxer",MinMaxScaler()),("clf",SVC(random_state=42))])
    cv = StratifiedShuffleSplit(labels,n_iter=10,random_state=42)
    svcGridSearch = GridSearchCV(svcclf, param_grid=svcParams, cv = cv, scoring='f1')
    svcGridSearch.fit(features_temp,labels)
    print "\nSVC with",numOfFeatures,"features:",svcGridSearch.best_params_,svcGridSearch.best_score_
    
    # Provided to give you a starting point. Try a variety of classifiers.
    ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
    
    # Example starting point. Try investigating other evaluation techniques!
    # from sklearn.cross_validation import train_test_split
    # features_train, features_test, labels_train, labels_test = \
    #     train_test_split(features, labels, test_size=0.3, random_state=42)
    

#Comparing the tuned classifiers to make final choice between them
ch2 = SelectKBest(k=5)
features = ch2.fit_transform(features, labels)
print "feature scores:",ch2.scores_
selected_feature_names = [features_list[i+1] for i in ch2.get_support(indices=True)]
print "selected features for final classifier:",selected_feature_names
features_list = ["poi"]
for feature_name in selected_feature_names:
    features_list.append(feature_name)
tunedSVC = SVC(gamma=0.0, tol=0.01, C=10.0, class_weight= 'auto', kernel='rbf', random_state=42)
tunedDecisionTreeCLF = DecisionTreeClassifier(max_depth=7,min_samples_split=2, random_state=42)
for currentClf in [tunedSVC,tunedDecisionTreeCLF]:
    
    precisionScores = []
    recallScores = []
    f1Scores = []
    kf = KFold(n = len(features),n_folds=10,shuffle=True,random_state=42)
    for train_indices,test_indices in kf:
        features_train, features_test = [features[i] for i in train_indices],[features[i] for i in test_indices]
        labels_train, labels_test = [labels[i] for i in train_indices],[labels[i] for i in test_indices]   
        clf = Pipeline(steps=[("minmaxsclaer",MinMaxScaler()),("clf",currentClf)])
        clf.fit(features_train,labels_train)
        predictions = clf.predict(features_test)
        precisionScore = precision_score(y_true=labels_test,y_pred=predictions,pos_label=1)
        recallScore = recall_score(y_true=labels_test,y_pred=predictions,pos_label=1)
        f1Score = f1_score(y_true=labels_test,y_pred=predictions,pos_label=1)
        precisionScores.append(precisionScore)
        recallScores.append(recallScore)
        f1Scores.append(f1Score)
    print "\n\n",clf
    print "Average precision score:",sum(precisionScores)*1.0/len(precisionScores)
    print "Average recall score:",sum(recallScores)*1.0/len(recallScores)
    print "Average f1 score:",sum(f1Scores)*1.0/len(f1Scores)
    
#Final choice of the classifier
chosenCLF = SVC(gamma=0.0, tol=0.01, C=10.0,class_weight= 'auto', kernel='rbf', random_state=60)
clf = Pipeline(steps=[("minmaxsclaer",MinMaxScaler()),("clf",chosenCLF)])
clf.fit(features,labels)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.



# Print Results 
print "Classification report:" 
print " "

test_classifier(clf, my_dataset, features_list)

dump_classifier_and_data(clf, my_dataset, features_list)