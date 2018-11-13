from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

class TextFeatures:
    
    def __init__(self, data, data_target):
        
        self.data = data # x_variables
        self.data_target = data_target # y_variable
        
        # The modeling pipeline
        
        self.pipeline = Pipeline([
                                ('vect', CountVectorizer()), 
                                ('tfidf', TfidfTransformer()), 
                                ('clf', SGDClassifier())], 
                                )
        
        # The parameters to test by grid search
        
        self.parameters = {
                        'vect__max_df': (0.5, 0.75, 1.0), # If word count percentage is higher than x, remove word.
                        'vect__max_features': (None, 5000, 10000, 50000), # Max features to consider.  None = all features
                        'vect__ngram_range': ((1, 1), (1, 2)),  # Unigrams or bigrams
                        'tfidf__use_idf': (True, False), # Yes/no to use inverse document frequency
                        'tfidf__norm': ('l1', 'l2'), # Choose normalization for tfidf
                        'clf__alpha': (0.00001, 0.000001), # Toggling the alpha for normalization in the classifier
                        'clf__penalty': ('l2', 'elasticnet'), # Norm type for classifier
                        'clf__n_iter': (10, 50, 80), # Number of iterations to try. 
                        }
        
    def get_best_features(self):
        
        if __name__ == "__main__":
            
            # Make the grid search.  n_jobs = -1 means use all processors
            
            grid_search = GridSearchCV(self.pipeline, self.parameters, n_jobs=-1) 
            
            # Fit your data
            
            grid_search.fit(self.data, self.data_target)
            
            # This result gives ALL the parameters even if they were not explicitly asked for.
            
            best_parameters = grid_search.best_estimator_.get_params() 
            
            # Now create a dict of just the parameters you are interested in.
            
            result = {}
            
            for param_name in sorted(self.parameters.keys()):
                result[param_name] = best_parameters[param_name]
                
            # Store the accuracy
                
            result['best_result'] = grid_search.best_score_
        
        # I outputed this as a dict so it would be easy to insert into a final pipeline.  
        # Could be outputed in various formats depending upon the use case.  The actual trained 
        # model could also be exported as well. 
            
        return result  