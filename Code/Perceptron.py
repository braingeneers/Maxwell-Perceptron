

import numpy as np

class Perceptron:
    """
    This code implements the perceptron algorithim for a single perceptron.... write more
    Notes:
    * THe input, is a vector whose first value should always be 1
    * x can either be compared against a true output, y, or against the true weights, u
    """
    
    def __init__(self, x_len, w_var=0.1, learn_rate=0.1, use_u=False ):
        """
        We initialize weights and create arrays to hold previous data values for future analysis
        We initializing the class the user must decide whether to compare x against true output, y, or true weights, u
        """
        self.w = np.random.normal(loc=0.0, scale=w_var, size=x_len)     # Set weights
        self.r = learn_rate
        
        self.old_w =  np.empty((0,x_len))                  # Create arrays to hold history parameter values
        self.old_x =  np.empty((0,x_len))
        self.old_y =  np.empty((0))
        self.old_y_pred =  np.empty((0))
        
        self.use_u = use_u                                 # if user compares x to u, create an additional log for this history
        if use_u:
            self.old_u = np.empty((0,x_len))
        
    
    def predict( self, x ):
        return float( np.dot( x, self.w) > 0 )
    
    def update( self, x, u_or_y ):          
        y_pred = self.predict(x)                              # Get models prediction for output

        self.old_w = np.append( self.old_w, [self.w], 0 )     # Save old values for future analysis
        self.old_x = np.append( self.old_x, [x], 0 )
        self.old_y_pred = np.append( self.old_y_pred, y_pred )  
                          
        if self.use_u:
            self.old_u = np.append( self.old_u, [u_or_y], 0 )  
            y = float( np.dot( x, u_or_y ) > 0 )             
        else:
            y = float(u_or_y)
        self.old_y = np.append( self.old_y, y )  
        
        self.w = self.w = self.w + self.r*(y - y_pred) * np.array(x)     # Update weights based on x,y,y_pred
        
    def getHistory(self):
        if self.use_u:
            return {"W":self.old_w, "X":self.old_x, "U":self.old_u, "Y":self.old_y, "Y_pred":self.old_y_pred}
        else:                           
            return {"W":self.old_w, "X":self.old_x, "Y":self.old_y, "Y_pred":self.old_y_pred}
                                  
    def getW(self):
        return self.w
                                  
                                  
                                  