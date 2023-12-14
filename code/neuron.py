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

    def get_feature_dict(self, feature_list=[], group_by_type=True):
        # from a list of features, return a numpy array with all features
        if len(feature_list) == 0:
            feature_list = [
                "Depth",
                "Level",
                "Set 1",
                "Dendrite Branching Angle",
                "Dendrite Branching Angle B",
                "Dendrite Length",
                "Dendrite Orientation Angle",
                "Dendrite Straightness",
            ]

        self.get_dendrite_feature_statistics(feature_list, group_by_type)

        tmp_to_merge = self.features.copy()
        tmp_to_merge.update(self.dendrites_features)

        return tmp_to_merge

    def get_statistics(self, grouped_by, features=None, aggregate_method="mean"):
        # Compute specified statistic for each dendrite type (Basal, Apical, Axon)
        stat_by_type = grouped_by.aggregate(aggregate_method)
        dendrite_types = stat_by_type.index.tolist()
        # Discard Axon dendrites
        dendrite_types_noaxons = [
            den_type for den_type in dendrite_types if "Axon" not in den_type
        ]

        # Handle if no feature is specified
        if features is None:
            features = stat_by_type.loc[dendrite_types_noaxons[0]].index.tolist()

        # Store average for each feature in the statistics dict
        stats_dict = {}
        for dend_type in dendrite_types_noaxons:
            # Get selected dendrite type and features
            type_specific_stat = stat_by_type.loc[dend_type].loc[features]
            # Rename features with selected dendrite type
            type_specific_stat.index = [
                "-".join([idx, dend_type.split(" ")[0], aggregate_method])
                for idx in type_specific_stat.index
            ]

            stats_dict.update(type_specific_stat.to_dict())
        return stats_dict

    def get_dendrite_feature_statistics(self, feature_list, group_by_type=True):
        # get selected features (except categorised data)
        # is_not_categorised = [feat for feat in feature_list if feat not in ['Set 1']]
        feat_df = self.dendrites.loc[:, feature_list]

        if not group_by_type:
            # Compute all statistics for each feature
            # - ugly but working -
            tmp_df = feat_df.mean(axis=0)
            tmp_df.index = ["-".join([idx, "mean"]) for idx in tmp_df.index]

            stats_dict = tmp_df.to_dict()

            tmp_df = feat_df.std(axis=0)
            tmp_df.index = ["-".join([idx, "std"]) for idx in tmp_df.index]

            stats_dict.update(tmp_df.to_dict())

            tmp_df = feat_df.min(axis=0)
            tmp_df.index = ["-".join([idx, "min"]) for idx in tmp_df.index]

            stats_dict.update(tmp_df.to_dict())

            tmp_df = feat_df.max(axis=0)
            tmp_df.index = ["-".join([idx, "max"]) for idx in tmp_df.index]

            stats_dict.update(tmp_df.to_dict())
        else:
            # Compute only selected statistics for each feature
            by_type = feat_df.groupby(by="Set 1")

            # Compute average of features for each dendrite type (Basal, Apical, Axon)
            stats_dict = self.get_statistics(
                grouped_by=by_type, aggregate_method="mean"
            )
            # Compute and store maximum value for Depth and Level features
            stats_dict.update(
                self.get_statistics(
                    grouped_by=by_type,
                    features=["Depth", "Level"],
                    aggregate_method="max",
                )
            )
            # Compute and store sum value for Length features
            stats_dict.update(
                self.get_statistics(
                    grouped_by=by_type,
                    features=["Dendrite Length"],
                    aggregate_method="sum",
                )
            )

        # computing number of dendrites
        stats_dict.update({"Dendrites Count": self.dendrites.shape[0]})

        # computing number of Basal or Apical dendrites
        dendrite_type_count = self.dendrites.loc[:, "Set 1"].value_counts()
        dendrite_type_count.index = [
            "-".join(["Dendrites Count", idx.split(" ")[0]])
            for idx in dendrite_type_count.index
        ]
        stats_dict.update(dendrite_type_count.to_dict())

        # computing number of level 1 dendrites (coming out of the soma)
        stats_dict.update(
            {
                "Dendrites Level_1_Count": self.dendrites.loc[
                    self.dendrites.Level == 1
                ].shape[0]
            }
        )

        self.dendrites_features = stats_dict
