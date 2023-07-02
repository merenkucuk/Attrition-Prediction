from math import e, log
from numpy import array
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

# We do not use any ready made libraries to implement ID3. We implement own.
# Attention: we make the Attrition column place to last place.
# So the printed tree is printing according to this situation.

# read the csv file with pandas
# we use pandas only here
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
# get columns to change Attrition column place to last place
cols = df.columns.tolist()
cols = cols[0:1] + cols[2:] + cols[1:2]
df = df[cols]
# get columns names
column_names = list(df)
# get column ids (0,34)
column_id = np.array(range(0, len(column_names)))
# make dataframe to numpy array
df = df.to_numpy()
# shuffle dataframe
np.random.shuffle(df)


# For continuous features, you can simply extract minimum and maximum value of your colon and then create certain number of intervals
# between your range of minimum and maximum values for the discretization process. You can choose any number of intervals suitable for you.
# we take the split number 5
# The function makes continuous features to discrete
def cont2disc(data, col_idx):
    max_value = data[:, col_idx].max()
    min_value = data[:, col_idx].min()
    min_interval = data[:, col_idx].min()
    column_range = []
    while min_interval <= max_value:
        column_range.append(int(min_interval))
        min_interval += (max_value - min_value) / 5
    for k in range(len(data[:, col_idx])):
        for i in range(len(column_range) - 1):
            if column_range[i + 1] >= data[:, col_idx][k] >= column_range[i]:
                data[:, col_idx][k] = i + 1
    return data


# makes continuous features to discrete with their indexes
# We firstly make Age column which has index 0 to discrete.
df = cont2disc(df, 0)
df = cont2disc(df, 2)
df = cont2disc(df, 4)
df = cont2disc(df, 5)
df = cont2disc(df, 8)
df = cont2disc(df, 9)
df = cont2disc(df, 11)
df = cont2disc(df, 12)
df = cont2disc(df, 13)
df = cont2disc(df, 15)
df = cont2disc(df, 17)
df = cont2disc(df, 18)
df = cont2disc(df, 19)
df = cont2disc(df, 22)
df = cont2disc(df, 23)
df = cont2disc(df, 24)
df = cont2disc(df, 26)
df = cont2disc(df, 27)
df = cont2disc(df, 28)
df = cont2disc(df, 29)
df = cont2disc(df, 30)
df = cont2disc(df, 31)
df = cont2disc(df, 32)
df = cont2disc(df, 33)


# calculate entropy to use when we find best gain for ID3
# when the entropy is high. we can say that distribution is polarized
def entropy(data, base=None):
    attrition_column = data[:, -1]
    n_attrition = len(attrition_column)
    if n_attrition <= 1:
        return 0
    value, counts = np.unique(attrition_column, return_counts=True)
    probs = counts / n_attrition
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)
    return ent


# calculate information gain to use for ID3
# the function take dataset and col_id as parameter
# return best_gain_key as index for example 0 index 0 is Age
def calc_information_gain(data, col_ids):
    gains = {}
    for id in col_ids:
        values = [item for item, count in Counter(data[:, id]).items()]
        counts = [count for item, count in Counter(data[:, id]).items()]
        all_entropy = entropy(data)
        for i in range(len(values)):
            value_data = data[data[:, id] == values[i]]
            value_entropy = entropy(value_data)
            all_entropy -= (counts[i] / len(data[:, id])) * value_entropy
        gain = all_entropy
        gains[id] = gain
    best_gain_key = max(gains, key=gains.get)
    return best_gain_key


# A normal Node class
# which has children, values, rule, attribute_name and id features.
class Node:
    def __init__(self, attr_id, attr_name, values, rule):
        self.children = []
        self.values = values
        self.rule = rule
        self.attr_name = attr_name
        self.attr_id = attr_id


# An attribution dict
# The dict contains dataframe values and headers
# we make the attribution dict to apply to the model
attr_dict = {'Age': array([1, 2, 3, 4, 5, 60], dtype=object), 'BusinessTravel': array(['Non-Travel', 'Travel_Frequently', 'Travel_Rarely'],
      dtype=object), 'DailyRate': array([1, 2, 3, 4, 5], dtype=object), 'Department': array(['Human Resources', 'Research & Development', 'Sales'],
      dtype=object), 'DistanceFromHome': array([1, 2, 3, 4, 5], dtype=object), 'Education': array([5],
      dtype=object), 'EducationField': array(['Human Resources', 'Life Sciences', 'Marketing', 'Medical','Other', 'Technical Degree'],
      dtype=object), 'EmployeeCount': array([1], dtype=object), 'EmployeeNumber': array([1, 2, 3, 4, 5],
      dtype=object), 'EnvironmentSatisfaction': array([5], dtype=object), 'Gender': array(['Female', 'Male'], dtype=object), 'HourlyRate': array([1, 2, 3, 4, 5],
      dtype=object), 'JobInvolvement': array([5], dtype=object), 'JobLevel': array([5],
      dtype=object), 'JobRole': array(['Healthcare Representative', 'Human Resources','Laboratory Technician', 'Manager', 'Manufacturing Director','Research Director', 'Research Scientist', 'Sales Executive','Sales Representative'],
      dtype=object), 'JobSatisfaction': array([5], dtype=object), 'MaritalStatus': array(['Divorced', 'Married', 'Single'],
      dtype=object), 'MonthlyIncome': array([1, 2, 3, 4, 5], dtype=object), 'MonthlyRate': array([1, 2, 3, 4, 5], dtype=object), 'NumCompaniesWorked': array([2, 3, 4, 5],
      dtype=object), 'Over18': array(['Y'], dtype=object), 'OverTime': array(['No', 'Yes'], dtype=object), 'PercentSalaryHike': array([1, 2, 3, 4, 23, 24, 25],
      dtype=object), 'PerformanceRating': array([1, 4], dtype=object), 'RelationshipSatisfaction': array([5], dtype=object), 'StandardHours': array([80],
      dtype=object), 'StockOptionLevel': array([4, 5], dtype=object), 'TotalWorkingYears': array([1, 2, 3, 4, 5], dtype=object), 'TrainingTimesLastYear': array([5],
      dtype=object), 'WorkLifeBalance': array([5], dtype=object), 'YearsAtCompany': array([1, 2, 3, 4, 5], dtype=object), 'YearsInCurrentRole': array([1, 2, 3, 4, 5],
      dtype=object), 'YearsSinceLastPromotion': array([1, 2, 3, 4, 5], dtype=object), 'YearsWithCurrManager': array([1, 2, 3, 4, 5], dtype=object), 'Attrition': array(['No', 'Yes'],
      dtype=object)}


# ID3 decision tree algorithm
def ID3(data, attr_ids, rule):
    # All of the attributes in the attribute ids list have their GAIN values computed,
    # and the calculated ones are taken out of the list.
    # The variable with the greatest GAIN value is designated as the new LEAF when the list is entirely empty.
    if (len(attr_ids[:-1]) == 0):
        targets = [item for item, count in Counter(df[:, -1]).items()]
        counts = [count for item, count in Counter(df[:, -1]).items()]
        index, value = 0, counts[0]
        for i, v in enumerate(counts):
            if v > value:
                index, value = i, v
        rules = str(rule) + targets[index]
        print(rules)
        return Node(-1, column_names[-1], targets[index], rules)

    #  returns leaf node yes, no. when everything is ok.
    if (len(set(data[:, -1])) == 1):
        b = Counter(data[:, -1])
        common_check = b.most_common(1)[0][0]
        rules = str(rule) + common_check
        print(rules)
        return Node(-1, column_names[-1], common_check, rules)

    # best information gain values and best information gain value index
    best_val_idx = calc_information_gain(data, attr_ids[:-1])

    # the best attribute's potential values
    possible_vals = attr_dict[column_names[best_val_idx]]

    # creating node of tree ID3
    child_node = Node(best_val_idx, column_names[best_val_idx], possible_vals, rule)
    child_node.rule += column_names[best_val_idx] + " ==> "

    # print all combination of node and child nodes
    for values in possible_vals:
        val_data = data[np.where(data[:, best_val_idx] == values)[0]]
        if val_data.shape[0] != 0:
            attr_idx_list = []
            for i in attr_ids:
                if i == best_val_idx:
                    continue
                else:
                    attr_idx_list.append(i)
            rules = str(child_node.rule) + str(values) + " ^ "
            child_node.children.append(ID3(val_data, attr_idx_list, rules))
        else:
            targets = [item for item, count in Counter(df[:, -1]).items()]
            counts = [count for item, count in Counter(df[:, -1]).items()]
            index, value = 0, counts[0]
            for i, v in enumerate(counts):
                if v > value:
                    index, value = i, v
            rules = str(child_node.rule) + str(values) + " ^ " + targets[index]
            print(rules)
            child_node.children.append(Node(-1, column_names[-1], targets[index], rules))

    return child_node

# This function evaluates the row in with the decision tree and returns values of node
def predict(node, row):
    node.attr_name = ""
    while node.attr_name != "Attrition":
        vals_list = []
        for i in range(len(node.values)):
            if node.values[i] == row[node.attr_id]:
                vals_list.append(i)
        next_node = node.children[vals_list[0]]
        node = next_node
        if node.attr_name == "Attrition":
            return node.values


k_fold = KFold(n_splits=5)
fold_value = 0
mis_predict = []
mis_test_data = []
mis_train_data = []

for train_index, test_index in k_fold.split(df):
    train_data = df[train_index]
    test_data = df[test_index]
    mis_test_data = test_data
    mis_train_data = train_data
    tree = ID3(train_data, column_id, "")
    predicts = []
    for row in test_data:
        x = predict(tree, row)
        predicts.append(x)
    mis_predict = predicts
    fold_value += 1
    tn, fp, fn, tp = confusion_matrix(test_data[:, -1], predicts, labels=["Yes", "No"]).ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)  # accuracy
    prec = tp / (tp + fp)  # precision
    recall = tp / (tp + fn)  # recall
    f1score = 2 * (recall * prec) / (recall + prec)  # f1score
    print("Accuracy for fold ", fold_value, " is ", acc)
    print("Precision for fold ", fold_value, " is ", prec)
    print("Recall for fold ", fold_value, " is ", recall)
    print("F1 Score for fold ", fold_value, " is ", f1score)


print("--------- MISCLASSIFIED EXAMPLES ---------")

i = 0
while i < len(mis_test_data):
    if mis_predict[i] != mis_test_data[i, -1]:
        print(mis_test_data[i, :])
    i += 1
