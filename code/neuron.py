
class Neuron:
    
    def __init__(self, id, file_path):
        self.id = id
        self.file_path = file_path
        self.features = {}
        self.dendrites = None

    def add_features(self, feats):
        self.features.update(feats)

    def add_dendrites(self, dendrites_df):
        self.dendrites = dendrites_df

    def get_feature_matrix(self, feature_list):
        # from a list of features, return a numpy array with all features
        return feature_list