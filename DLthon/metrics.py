import tensorflow as tf

""" Loss Functions -------------------------------------- """
class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def call(self, y_true, y_pred):
        y_pred = tf.keras.activations.sigmoid(y_pred)

        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])

        intersection = tf.reduce_sum(y_true * y_pred)
        dice = (2.*intersection + 1)/(tf.reduce_sum(y_pred) + tf.reduce_sum(y_true) + 1)

        return 1 - dice

class DiceBCELoss(tf.keras.losses.Loss):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def call(self, y_true, y_pred):
        y_pred = tf.keras.activations.sigmoid(y_pred)

        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])

        intersection = tf.reduce_sum(y_true * y_pred)
        dice_loss = 1 - (2.*intersection + 1)/(tf.reduce_sum(y_pred) + tf.reduce_sum(y_true) + 1)
        BCE = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class MultiClassBCE(tf.keras.losses.Loss):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def call(self, y_true, y_pred):
        loss = []
        for i in range(tf.shape(y_pred)[1]):
            yp = y_pred[:, i]
            yt = y_true[:, i]
            BCE = tf.keras.losses.BinaryCrossentropy()(yt, yp)

            if i == 0:
                loss = BCE
            else:
                loss += BCE

        return loss

""" Metrics ------------------------------------------ """
def precision(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    return (intersection + 1e-15) / (tf.reduce_sum(y_pred) + 1e-15)

def recall(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    return (intersection + 1e-15) / (tf.reduce_sum(y_true) + 1e-15)

def F2(y_true, y_pred, beta=2):
    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * tf.reduce_sum(y_true * y_pred) + 1e-15) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-15)

def jac_score(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-15) / (union + 1e-15)
