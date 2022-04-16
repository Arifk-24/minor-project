from statistics import mode
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.tree import _tree
import csv
import pickle


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}


x = pd.read_csv('..//dataset//final_features.csv')
x = x.iloc[:,1:]
y = pd.read_csv('..//dataset//diseases.csv')
y = y['disease']
le = preprocessing.LabelEncoder()
le.fit(y.to_list())
training = pd.concat([x,y] , axis =1)
reduced_data = reduced_data = training.groupby(training['disease']).max()

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

def get_precaution_dict():
    global precautionDictionary
    with open('../dataset/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

def get_description():
    global description_list
    with open('../dataset/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

def get_info():
    # name=input("Name:")
    print("Your Name \n\t\t\t\t\t\t",end="->")
    name=input("")
    print("hello ",name)



def sec_predict(symptoms_exp,dtree,nbmodel,clf):
    symptoms_dict = {}

    for index, symptom in enumerate(x):
        symptoms_dict[symptom] = index

    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1
    result = mode([clf.predict([input_vector])[0] ,  dtree.predict([input_vector])[0], nbmodel.predict([input_vector])[0] ])
    return result

def print_disease(node):
    #print(node)
    node = node[0]
    #print(len(node))
    val  = node.nonzero() 
    #print(val)
    disease = le.inverse_transform(val[0])
    return disease

def tree_to_code(tree, feature_names,nbmodel, clf):
    tree_ = tree.tree_
    # print(tree_)
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    symptoms_present = []

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            inp = input(name + "?")
            if inp == "yes":
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            print( present_disease )
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            # dis_list=list(symptoms_present)
            # if len(dis_list)!=0:
            #     print("symptoms present  " + str(list(symptoms_present)))
            # print("symptoms given "  +  str(list(symptoms_given)) )
            print("Are you experiencing any ")
            symptoms_exp=[]
            for syms in list(symptoms_given):
                if syms in symptoms_present:
                    symptoms_exp.append(syms)
                    continue
                inp=""
                print(syms,"? : ",end='')
                while True:
                    inp=input("")
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ",end="")
                if(inp=="yes"):
                    symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp,dtree,nbmodel,clf)
            print(second_prediction)
            #calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction):
                print("You may have ", present_disease[0])

                print(description_list[present_disease[0]])

                # readn(f"You may have {present_disease[0]}")
                # readn(f"{description_list[present_disease[0]]}")

            else:
                print("You may have ", present_disease[0], "or ", second_prediction)
                print(description_list[present_disease[0]])
                print(description_list[second_prediction])

            # print(description_list[present_disease[0]])
            precution_list=precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)

            # confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
            # print("confidence level is " + str(confidence_level))

    recurse(0, 1)

if __name__ == '__main__':
    with open('..//models//dtree','rb') as f:
        dtree = pickle.load(f)
    with open('..//models//nb','rb') as f:
        nbmodel = pickle.load(f)    
    with open('..//models//random_forest','rb') as f:
        clf = pickle.load(f)

    get_description()
    get_precaution_dict()
    get_info()
    cols = training.columns
    tree_to_code(dtree,cols,nbmodel,clf)

