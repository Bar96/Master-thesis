import numpy as np
import pandas

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from spektral.layers import GCNConv
from sklearn import metrics



def prepare_masks(num_nodes, train_perc=0.8, val_perc=0.2):
    '''Generate the boolean masks to divide the data set into train set,
    validation set and test set. Each mask is a boolean vector with true in the 
    index that belong to a certain set.

    Parameters
    ----------
    num_nodes : integer
        number of nodes in the network.
    train_perc : float, optional
        percentage of node to use for the train and the validation. 
        The remaining will be used as test set. The default is 0.8.
    val_perc : float, optional
        percentage of node of the train to use as validation. The default is 0.2.

    Returns
    -------
    train_mask : numpy array
        train mask.
    val_mask : numpy array
        validation mask.
    test_mask : numpy array
        test mask.

    '''
    num_train = int(num_nodes * train_perc) #number of elements in the train + validation set
    num_val = int(num_train * val_perc) #number of elements in the validation set

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[:(num_train-num_val)] = True
    val_mask[(num_train-num_val):num_train] = True
    test_mask[num_train:] = True
    
    return train_mask, val_mask, test_mask




def build_autoencoder(num_features, num_nodes, conv1_dim=64, conv2_dim=10, drop_perc=0.5, verbose=True):
    '''Build a graph convolutional autoencoder. Both the encoder and the decoder are made with 2
    convolutional layers and 2 dropout layers. The dimensions of the hidden layers are given
    as input parameters. Encoder and decoder architecture are mirrored

    Parameters
    ----------
    num_features : integer
        initial dimension of the input vectors.
    num_nodes : integer
        number of nodes in the graph.
    conv1_dim : integer, optional
        output dimension of the first hidden layer of the 
        encoder. The default is 64.
    conv2_dim : integer, optional
        output dimension the second hidden layer of the 
        encoder. This is also the dimension of the latent 
        space. The default is 10.
    drop_perc : float, optional
        drop percentage for the dropout layers. The default is 0.5.
    verbose : boolean, optional
        print the summary of the model structure. The default is True.

    Returns
    -------
    encoder : Model
        Model object of the encoder.
    autoencoder : Model
        Model object of the entire autoencoder.

    '''
    #encoder
    input_X_enc = layers.Input(shape=(num_features, ))
    input_A_enc = layers.Input(shape=(num_nodes, ), sparse=True)
    conv = GCNConv(conv1_dim, 'relu')([input_X_enc, input_A_enc])
    drop = layers.Dropout(drop_perc)(conv)
    encoder_output = GCNConv(conv2_dim, 'linear')([drop, input_A_enc])
    encoder = Model(inputs=[input_X_enc, input_A_enc], outputs=encoder_output, name='encoder')
    if verbose: encoder.summary()

    #decoder
    input_X_dec = layers.Input(shape=(conv2_dim, ))
    input_A_dec = layers.Input(shape=(num_nodes, ), sparse=True)
    conv = GCNConv(conv1_dim, 'relu')([input_X_dec, input_A_dec])
    drop = layers.Dropout(drop_perc)(conv)
    decoder_output = GCNConv(num_features, activation='linear')([drop, input_A_dec])
    decoder = Model(inputs=[input_X_dec, input_A_dec], outputs=decoder_output, name='decoder')
    if verbose: decoder.summary()

    #autoencoder
    input_X = layers.Input(shape=(num_features, ))
    input_A = layers.Input(shape=(num_nodes, ), sparse=True)
    enc = encoder([input_X, input_A])
    dec = decoder([enc, input_A])
    autoencoder = Model(inputs=[input_X, input_A], outputs=dec, name='autoencoder')
    if verbose: autoencoder.summary()
    return encoder, autoencoder




def build_autoencoder_3(num_features, num_nodes, conv1_dim=64, conv2_dim=32, 
                        conv3_dim=10, drop_perc=0.5, verbose=True):
    '''Build a graph convolutional autoencoder with 3 convolutional layers 
    both in the encoder and in the decoder. Encoder and decoder have mirrored
    architecture.

    Parameters
    ----------
    num_features : integer
        initial dimension of the input vectors.
    num_nodes : integer
        number of nodes in the graph.
    conv1_dim : integer, optional
        output dimension of the first hidden layer of the 
        encoder. The default is 64.
    conv2_dim : integer, optional
        output dimension of the second hidden layer of the 
        encoder. The default is 32.
    conv3_dim : integer, optional
        output dimension of the third hidden layer of the
        enoder. This is also the dimension of the latent space. 
        The default is 10.
    drop_perc : float, optional
        drop percentage for the dropout layers. The default is 0.5.
    verbose : boolean, optional
        print the summary of the model structure. The default is True.

    Returns
    -------
    encoder : Model
        Model object of the encoder.
    autoencoder : Model
        Model object of the decoder.

    '''
    #encoder
    input_X_enc = layers.Input(shape=(num_features, ))
    input_A_enc = layers.Input(shape=(num_nodes, ), sparse=True)
    conv = GCNConv(conv1_dim, 'relu')([input_X_enc, input_A_enc])
    drop = layers.Dropout(drop_perc)(conv)
    conv = GCNConv(conv2_dim, 'relu')([drop, input_A_enc])
    drop = layers.Dropout(drop_perc)(conv)
    encoder_output = GCNConv(conv3_dim, 'linear')([drop, input_A_enc])
    encoder = Model(inputs=[input_X_enc, input_A_enc], outputs=encoder_output, name='encoder')
    if verbose: encoder.summary()

    #decoder
    input_X_dec = layers.Input(shape=(conv3_dim, ))
    input_A_dec = layers.Input(shape=(num_nodes, ), sparse=True)
    conv = GCNConv(conv2_dim, 'relu')([input_X_dec, input_A_dec])
    drop = layers.Dropout(drop_perc)(conv)
    conv = GCNConv(conv1_dim, activation='relu')([drop, input_A_dec])
    drop = layers.Dropout(drop_perc)(conv)
    decoder_output = GCNConv(num_features, activation='linear')([drop, input_A_dec])
    decoder = Model(inputs=[input_X_dec, input_A_dec], outputs=decoder_output, name='decoder')
    if verbose: decoder.summary()

    #autoencoder
    input_X = layers.Input(shape=(num_features, ))
    input_A = layers.Input(shape=(num_nodes, ), sparse=True)
    enc = encoder([input_X, input_A])
    dec = decoder([enc, input_A])
    autoencoder = Model(inputs=[input_X, input_A], outputs=dec, name='autoencoder')
    if verbose: autoencoder.summary()
    return encoder, autoencoder




def build_autoencoder_4(num_features, num_nodes, conv1_dim=128, conv2_dim=64, conv3_dim=32,
                        conv4_dim=10, drop_perc=0.5, verbose=True):
    '''Build a graph convolutional autoencoder with 4 convolutional layers 
    both in the encoder and in the decoder. Encoder and decoder have mirrored
    architecture.

    Parameters
    ----------
    num_features : integer
        initial dimension of the input vectors.
    num_nodes : integer
        number of nodes in the graph.
    conv1_dim : integer, optional
        output dimension of the first hidden layer of the 
        encoder. The default is 128.
    conv2_dim : integer, optional
        output dimension of the second hidden layer of the 
        encoder. The default is 64.
    conv3_dim : integer, optional
        output dimension of the third hidden layer of the 
        encoder. The default is 32.
    conv4_dim : integer, optional
        output dimension of the fourth hidden layer of the
        enoder. This is also the dimension of the latent space. 
        The default is 10.
    drop_perc : float, optional
        drop percentage for the dropout layers. The default is 0.5.
    verbose : boolean, optional
        print the summary of the model structure. The default is True.

    Returns
    -------
     encoder : Model
        Model object of the encoder.
    autoencoder : Model
        Model object of the decoder.

    '''
    #encoder
    input_X_enc = layers.Input(shape=(num_features, ))
    input_A_enc = layers.Input(shape=(num_nodes, ), sparse=True)
    conv = GCNConv(conv1_dim, 'relu')([input_X_enc, input_A_enc])
    drop = layers.Dropout(drop_perc)(conv)
    conv = GCNConv(conv2_dim, 'relu')([drop, input_A_enc])
    drop = layers.Dropout(drop_perc)(conv)
    conv = GCNConv(conv3_dim, 'relu')([drop, input_A_enc])
    drop = layers.Dropout(drop_perc)(conv)
    encoder_output = GCNConv(conv4_dim, 'linear')([drop, input_A_enc])
    encoder = Model(inputs=[input_X_enc, input_A_enc], outputs=encoder_output, name='encoder')
    if verbose: encoder.summary()

    #decoder
    input_X_dec = layers.Input(shape=(conv4_dim, ))
    input_A_dec = layers.Input(shape=(num_nodes, ), sparse=True)
    conv = GCNConv(conv3_dim, 'relu')([input_X_dec, input_A_dec])
    drop = layers.Dropout(drop_perc)(conv)
    conv = GCNConv(conv2_dim, activation='relu')([drop, input_A_dec])
    drop = layers.Dropout(drop_perc)(conv)
    conv = GCNConv(conv1_dim, activation='relu')([drop, input_A_dec])
    drop = layers.Dropout(drop_perc)(conv)
    decoder_output = GCNConv(num_features, activation='linear')([drop, input_A_dec])
    decoder = Model(inputs=[input_X_dec, input_A_dec], outputs=decoder_output, name='decoder')
    if verbose: decoder.summary()

    #autoencoder
    input_X = layers.Input(shape=(num_features, ))
    input_A = layers.Input(shape=(num_nodes, ), sparse=True)
    enc = encoder([input_X, input_A])
    dec = decoder([enc, input_A])
    autoencoder = Model(inputs=[input_X, input_A], outputs=dec, name='autoencoder')
    if verbose: autoencoder.summary()
    return encoder, autoencoder




def compute_loss(X_true, X_pred, mask):
    '''

    Parameters
    ----------
    X_true : numpy array
        true values.
    X_pred : numpy array
        predicted values.
    mask : numpy array
        boolean mask to select which node consider to compute the error.

    Returns
    -------
    mse : float
        mean square error between the predicted X and the real one.

    '''
    mse = []
    for i in range(X.shape[0]):
        if mask[i]:
            mse.append(metrics.mean_squared_error(X_true[i], X_pred[i]))
    return mse
