class NCA:
    '''
    Interface for NCAs
    '''

    def __init__(self):
        pass


    def train_step(self, name, prompt):
        '''
        Returns an NCA 

        Return Type: 
        '''
        raise NotImplementedError()

