
class Neuron:
    
    def __init__(self, id, file_path):
        self.id = id
        self.file_path = file_path
        self.features = {}
        self.dendrites = None
        self.dendrites_features = {}

    def add_features(self, feats):
        self.features.update(feats)

    def add_dendrites(self, dendrites_df):
        self.dendrites = dendrites_df

    def get_feature_dict(self, feature_list = []):
        # from a list of features, return a numpy array with all features
        if len(feature_list) == 0:
            feature_list = ['Depth', 'Level', 'Set 1', 'Dendrite Area', 'Dendrite Branching Angle', 'Dendrite Branching Angle B',
                            'Dendrite Length', 'Dendrite Orientation Angle', 'Dendrite Position X_x', 'Collection_x',
                            'Dendrite Position Y_x', 'Dendrite Position Z_x', 'Dendrite Straightness']

        self.get_dendrite_feature_statistics(feature_list)

        tmp_to_merge = self.features.copy()
        tmp_to_merge.update(self.dendrites_features)

        return tmp_to_merge

    def get_dendrite_feature_statistics(self, feature_list):

        # get selected features (except categorised data)
        is_not_categorised = [feat for feat in feature_list if feat not in ['Set 1']]
        feat_df = self.dendrites.loc[:, is_not_categorised]

        # - ugly but working -
        tmp_df = feat_df.mean(axis = 0)
        tmp_df.index = ['-'.join([idx, 'mean']) for idx in tmp_df.index]
        
        stats_dict = tmp_df.to_dict()

        tmp_df = feat_df.std(axis = 0)
        tmp_df.index = ['-'.join([idx, 'std']) for idx in tmp_df.index]

        stats_dict.update(tmp_df.to_dict())

        tmp_df = feat_df.min(axis = 0)
        tmp_df.index = ['-'.join([idx, 'min']) for idx in tmp_df.index]
        
        stats_dict.update(tmp_df.to_dict())

        tmp_df = feat_df.max(axis = 0)
        tmp_df.index = ['-'.join([idx, 'max']) for idx in tmp_df.index]
        
        stats_dict.update(tmp_df.to_dict())

        # computing number of dendrites
        stats_dict.update({'dendrites-count':self.dendrites.shape[0]})

        # computing number of Basal or Apical dendrites
        dendrite_type_count = self.dendrites.loc[:, 'Set 1'].value_counts()
        dendrite_type_count.index = ['-'.join([idx, 'count']) for idx in dendrite_type_count.index]
        stats_dict.update(dendrite_type_count.to_dict())

        # computing number of level 1 dendrites (coming out of the soma)
        stats_dict.update({'level_1-count':self.dendrites.loc[self.dendrites.Level == 1].shape[0]})

        self.dendrites_features = stats_dict


#def get_feature_matrix(neuron_list):
