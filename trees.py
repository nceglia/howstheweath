from sklearn import linear_model
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
import sklearn
import numpy
import sys
import matplotlib
matplotlib.use("Agg")
import pylab
import argparse
import operator

class EnsembleTreeModel(object):
    """
    Regression Model
        - Load Data
        - Trains with a regression algorithm of choice
        - Writes out a prediction file for uploading to Kaggle
    """
    def __init__(self):
        """
        Variables
        """
        self.header_f = "../data/kaggle.X1.names.txt"
        self.train_f = "../data/kaggle.X1.train.txt"
        self.test_f = "../data/kaggle.X1.test.txt"
        self.test_images_f = "../data/kaggle.X2.test.txt"
        self.target_f = "../data/kaggle.Y.train.txt"
        self.images_f = "../data/kaggle.X2.train.txt"
        self.scaler_funcs = {"standard": preprocessing.StandardScaler().fit,
                             "minmax": preprocessing.MinMaxScaler().fit}
        self.algorithms = {"rf": ensemble.RandomForestRegressor,
                           "et": ensemble.ExtraTreesRegressor,
                           "ada": ensemble.AdaBoostRegressor,
                           "gbt": ensemble.GradientBoostingRegressor}
        self.curve = None
        self.stepsize = None
        self.sorted_ranks = None

    def get_algorithms(selof):
        return self.algorithms.keys()

    def load(self,shuffled=True, load_images=False, images_only=False):
        mapper = lambda raw: [map(float,x.split(",")) for x in raw]
        self.header = open(self.header_f,"r").read().split(",")
        f = open(self.train_f,"r").read().split("\n")
        f = filter(lambda x:len(x.split(","))==len(self.header),f)
        self.train = mapper(f)
        f = open(self.test_f,"r").read().split("\n")
        f = filter(lambda x:len(x.split(","))==len(self.header),f)
        self.test = mapper(f)
        f = open(self.target_f,"r").read().split("\n")
        self.targets = numpy.array(map(float,filter(lambda x: x.strip()!="",f)))
        if load_images:
            f = open(self.images_f,"r").read().split("\n")
            f = filter(lambda x:len(x.split(","))>1,f)
            self.images = mapper(f)
            f = open(self.test_images_f,"r").read().split("\n")
            f = filter(lambda x:len(x.split(","))>1,f)
            self.test_images = mapper(f)
            if not images_only:
                for i in range(len(self.train)):
                    self.train[i] += self.images[i]
                for i in range(len(self.test)):
                    self.test[i] += self.test_images[i]
            else:
                self.train = self.images
                self.test = self.test_images
        assert len(self.train) == len(self.targets)
        self.train = numpy.array(self.train)
        self.test = numpy.array(self.test)
        if shuffled:
            self.targets, self.train = shuffle(self.targets,self.train)
        self.train_copy = []
        for val in self.train:
            self.train_copy.append(val)

    def preprocess(self,scaler_func="minmax"):
        if scaler_func:
            assert scaler_func in self.scaler_funcs.keys()
            self.scaler = self.scaler_funcs[scaler_func](self.train)
            self.train = self.scaler.transform(self.train)

    def fit(self,algorithm,n_folds=1,tr_range=None,loss='ls', n_estimators=10, max_depth=1,max_features="auto", reduce_features=None, min_samples_leaf=1, learning_rate=1.0):
        if not tr_range:
            tr_range = (0,len(self.train))
        train = self.train[tr_range[0]:tr_range[1]]
        targets = self.targets[tr_range[0]:tr_range[1]]
        if algorithm in ["et","rf"]:
            """
            Bagging Trees
            """
            self.model = self.algorithms[algorithm](n_estimators=n_estimators, max_features=max_features, n_jobs=-1, min_samples_leaf=1)
        if algorithm == "gbt":
            """
            Boosting Trees
            """
            self.model = self.algorithms[algorithm](loss=loss) 
        elif algorithm == "ada":
            #Base Estimator is Decision Tree Regressor
            estimator = sklearn.tree.DecisionTreeRegressor(max_depth=max_depth)
            self.model = self.algorithms[algorithm](base_estimator=estimator,n_estimators=n_estimators, learning_rate=learning_rate)
            
        if n_folds > 1:
            kf = cross_validation.KFold(len(train),n_folds=n_folds)
            error = 0
            for tr, val in kf:
                x_train, x_val = train[tr], train[val]
                y_train, y_val = targets[tr], targets[val]
                self.model = self.model.fit(x_train, y_train)
                y_pred = self.model.predict(x_val)
                error += mean_squared_error(y_val, y_pred)
            error /= float(n_folds)
            print "Mean MSE ({0}-fold): {1}".format(n_folds,error)
        else:
            self.model = self.model.fit(self.train, self.targets)
            return self.model.train_score_
            error = 0
            print "Model trained on full dataset..."
        if reduce_features:
            print "Reducing Features..."
            if not self.sorted_ranks:
                ranks = self.model.feature_importances_
                temp ={}
                for i, x in enumerate(ranks):
                    temp[i] = x
                ranks = temp
                self.sorted_ranks = sorted(ranks.items(), key=operator.itemgetter(1))
            temp_train = []
            for i, example in enumerate(self.train_copy):
                reduced_example = []
                for ranked_feat in self.sorted_ranks[:reduce_features]:
                    reduced_example.append(example[ranked_feat[1]])
                temp_train.append(reduced_example)
            self.train = numpy.array(temp_train)
        return error

    def feature_importance_plot(self,forest,algorithm):
        importances = forest.feature_importances_
        std = numpy.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = numpy.argsort(importances)[::-1]
        print indices
        labels = []
        for index in indices:
            labels.append(self.header[index])
        # Print the feature ranking
        print("Feature ranking:")

        for f in range(10):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        heights = []
        for i in range(10):
            heights.append(importances[indices[i]])
        pylab.figure()
        pylab.title("Feature importances")
        pylab.bar(range(10), heights, color="r", align="center")
        pylab.xticks(range(10), labels, rotation='vertical')
        pylab.xlim([-1, 10])
        pylab.margins(0.2)
        pylab.subplots_adjust(bottom=0.2)
        pylab.savefig("features_{0}.png".format(algorithm))
        pylab.close()


    def learn(self,algorithm,step_size=200,n_folds=5,alpha=1.0,normalize=True):
        curve = []
        for i in range(0,len(self.train),step_size):
            curve.append(1.0/self.fit(algorithm,tr_range=(0,i+step_size)))
        self.step_size = step_size
        self.curve = curve

    def vary(self,algorithm):
        self.curve = []
        x_range = [1,2,3,4,5,6,7,8,9,10]
        for i in x_range:
            self.curve.append(self.fit(algorithm,max_depth=i))
        self.publish(x_range,filename="max_depth_{0}.png".format(algorithm))

    def vary_boosting(self,algorithm,loss):
        pylab.figure()
        self.curve = self.fit(algorithm,loss=loss)
        x_range = xrange(len(self.curve))
        self.publish(x_range,filename="deviance_{0}.png".format(loss))

    def publish(self,x_range,filename="learning_curve.png"):
        pylab.figure()
        pylab.plot(x_range,self.curve)
        pylab.xlabel('Boosting Iteration')
        pylab.ylabel("Deviance")
        pylab.savefig(filename)
        pylab.close()

    def submission(self,filename="submission.csv"):
        predictions = self.model.predict(self.test)
        output = open(filename,"w")
        output.write("ID,Prediction\n")
        for i, pred in enumerate(predictions):
            output.write("{0},{1}\n".format(i+1,pred))
        output.close()
        
if __name__ == '__main__':
    scale = None
    algorithm ="gbt"

    rf = EnsembleTreeModel()
    print "Loading Data..."
    rf.load(shuffled=True, load_images=False)   
    print "Preprocessing..."

    rf.preprocess(scaler_func=scale)
    print "Training..."
    #rf.learn(algorithm)
    #rf.vary(algorithm)
    rf.vary_boosting(algorithm,"lad")
    #rf.submission()
    #f="{0}_learning_curve.png".format(algorithm)
    #rf.publish(filename=f)

