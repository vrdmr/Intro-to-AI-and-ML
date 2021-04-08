from sklearn import tree

label_map = {1: "Apple", 0: "Orange"}

features = [[140, 1], # Apple
            [130, 1], # Apple
            [150, 0], # Orange
            [170, 0]] # Orange
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
print("Initialized")
clf = clf.fit(features, labels)
print("Trained")

# 150g "Smooth"
prediction = clf.predict([[150, 0]])[0]
print("Prediction", label_map[prediction])