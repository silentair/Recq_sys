'''the basic class of an RS'''
class Recommender_Base():
    def __init__(self,config,train_data,test_data,save_model=False,save_path=''):
        self.config = config
        self.train_data = train_data
        self.test_data = test_data
        self.save_model = save_model
        self.save_path = save_path