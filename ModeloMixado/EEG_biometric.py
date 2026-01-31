# Arquivos que contem a arquitetura da rede neural
import models
import preprocessing
import utils
import data_manipulation
import loader

import argparse
import sys
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from numpy import savetxt, loadtxt

from tensorflow.keras.optimizers import Adam ##
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint ##

import zipfile
import os
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

folder_path = '/media/work/mariapaula/IC/ModeloMixado/Dataset_CSV/'
processed_data_path = '/media/work/mariapaula/IC/ModeloMixado/'

if not os.path.exists(os.path.join(processed_data_path, 'results')):
    os.makedirs(os.path.join(processed_data_path, 'results'), exist_ok=True)

# (Hiperparâmetros e Configuração de Dados)

# Numeros aleatorios sejam sempre os mesmo, tornando o experimento replicavel
random.seed(1051)
np.random.seed(1051)
tf.random.set_seed(1051)

# 132 amostras por vez e por 40 epocas e o learning rate 
batch_size = 8               
training_epochs = 40            
initial_learning_rate = 0.001

# Numeros de classes de classificação
num_classes = 2
# Sujeitos de treino
training_subjects_end = 70
# Total de sujeitos
total_subjects = 109  

train_tasks = [1, 2]
test_tasks = [1, 2] 

# (Processamento Digital dos sinais na functions.filter_data())
# Como os sinais do cerebros sao sujos, ele é limpado para ter apenas os sinais EEG
# Filtro passa-banda (Permite que frequencias dentro de um intervalo passem)

# Remove ruidos de baixissima frequencia (respiracao) e ruidos de alta frequencia (Envolve Delta, Theta, Alpha, Beta e Gamma)
band_pass_1 = [1, 50]
# Foca nos ritmos Alpha e Beta (Faixas que envolvem concentração, relaxamento ou imaginação motora)
band_pass_2 = [10, 30] 
# Foca na faixa Gamma (Processamento cognitivo de alto nivel e integração de informações
band_pass_3 = [30, 50]   
# 160 amostras por segundo
sample_frequency = 160 
# Rigor do filtro (Ordem alta, separa as frequencias de forma instantanea
filter_order = 12               
filter_type = 'filtfilt'        
# 'sosfilt' or 'filtfilt'

# Parametros para normalizacao dos dados
# Ajusta individualmente cada canal para que todos fiquem em uma escala comparavel
normalize_type = 'each_channel'

# Parametros para o janelamento
window_size = 1920
# Passo
offset = 35
# Divisao dos conjuntos de dados
split_ratio = 0.9

# Numero de canais
num_channels = 64

# Canais presentes no artigo de Yang et al
# Relacionado a decisões e atenção
frontal_lobe_yang = ['Af3.', 'Afz.', 'Af4.']
# Relacionado ao movimento físico ou imaginação de movimento
motor_cortex_yang = ['C1..', 'Cz..', 'C2..']
# Relacionamento ao processamento visual
occipital_lobe_yang = ['O1..', 'Oz..', 'O2..']
# Combinação dos 9 eletrodos dessas tres areas acimas
all_channels_yang = ['C1..', 'Cz..', 'C2..', 'Af3.', 'Afz.', 'Af4.', 'O1..', 'Oz..', 'O2..']

# Tudo que o codigo imprimir durante o treino sera gravado nesse arquivo txt
sys.stdout = utils.Logger(os.path.join(processed_data_path, 'results', 'log_script.txt'))
sys.stderr = sys.stdout

# Argparse
# Crio o objeto que vai guardar todas as regras e opções que voce definir
parser = argparse.ArgumentParser()
# Ativa o uso de Geradores de Dados (Dataset maior que a memoria RAM)
parser.add_argument('--datagen', action='store_true',
                    help='the model will use Data Generators to crop data on the fly')
# Ao usar isso o codigo nao treina, vai direto para a parte de teste carregando os pesos de um arquivo .h5 ja existente
parser.add_argument('--nofit', action='store_true',
                    help='model.fit will not be executed. The weights will be gathered from the file'+
                    ' \'model_weights.h5\', that is generated if you have ran the model in Identification mode'
                    ' at least once')
# Desativa o modo de identificação (classificação direta das classes)
parser.add_argument('--noimode', action='store_true',
                    help='the model won\'t run in Identification Mode')
# Desativa o modo de verificação (calculo de metricas biometricas como EER)
parser.add_argument('--novmode', action='store_true',
                    help='the model won\'t run in Verification Mode')

# Espera uma lista de numeros inteiros
parser.add_argument('-train', nargs="+", type=int, required=True, 
                    help='list of tasks used for training and validation. All specified tasks need to be higher than\n'+
                    ' 0 and lower than 15. This is a REQUIRED flag')
# Define quais tarefas serao usadas para testar a rede neural após o treino
parser.add_argument('-test', nargs="+", type=int, required=True, 
                    help='list of tasks used for testing. All specified tasks need to be higher than 0 and lower than\n'+
                    ' 15. This is a REQUIRED flag')
# Lê o que voce digitou e transforma em um objeto fácil de usar
args = parser.parse_args()

# Extrai as listas de tarefas que digitou para as variaveis locais
train_tasks = args.train
test_tasks = args.test

# Verifica se os numeros das tarefas entao entre 1 e 14
for task in train_tasks:
    if(task <= 0 or task >= 15):
        print('ERROR: All training/validation and testing tasks need to be higher than 0 and lower than 15.\n')
        sys.exit()

# Verifica se os numeoros das tarefas entao entre 1 e 14
for task in test_tasks:
    if(task <= 0 or task >= 15):
        print('ERROR: All training/validation and testing tasks need to be higher than 0 and lower than 15.\n')
        sys.exit()

# Define como a inteligencia artifical vai aprender e como o progresso sera salvo
# SGD ajusta os pesos da rede neural, o tamanho do peso e o momentum ajuda o otimizador a não ficar em buracos pequenos e a convergir mais rapido 
opt = Adam(learning_rate=0.001)# Cria callback, chama uma funcao que diminui a taxa de aprendizado
lr_scheduler = LearningRateScheduler(models.scheduler, verbose = 0)
# Salva os pesos a cada 5 epócas nesse arquivo
saver = models.SaveAtEpochEnd(5, '/media/work/mariapaula/IC/ModeloMixado/model_weights')
# apenas cria a variavel model
model = None

# Não usando o data generator (ou seja, nao digitei o --datagen, que é os casos atuais)
if(not args.datagen):
    # Carrega o conteudo de treino e teste, dentro do caminho e o numero 1 mostra o progresso
    train_content, test_content = loader.load_data(folder_path, train_tasks, test_tasks, 'csv', num_classes, 1)   

    # Filtrando o conteudo de teste e treino com passa banda 3
    train_content = preprocessing.filter_data(train_content, band_pass_2, sample_frequency, filter_order, filter_type, 1)
    test_content = preprocessing.filter_data(test_content, band_pass_2, sample_frequency, filter_order, filter_type, 1)

    # Normaliza os dados
    train_content = preprocessing.normalize_data(train_content, 'sun', 1)
    test_content = preprocessing.normalize_data(test_content, 'sun', 1)

    # Fazendo o janelamento, offset e a divisão de 90% para treino e 105 
    x_train, y_train, x_val, y_val = data_manipulation.crop_data(train_content, train_tasks, num_classes,
                                                                window_size, offset, split_ratio)
    x_test, y_test = data_manipulation.crop_data(test_content, test_tasks, num_classes, window_size, window_size)

    # Verifica se nao usou o nofit ou seja digitei --nofit (crio modelo novo) se tiver usado pula direto para os pesos ja pronto
    if(not args.nofit):

        # Criando o modelo mixado
        model = models.create_model_mixed(window_size, num_channels, num_classes)
        # Visualiza a rede
        model.summary()

        # Compilacao do modelo usando usando optimizador, a funcao de perda, e a metrica que eu quero
        model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Captura o horario exato antes de começar o treinamento
        fit_begin = time.time()
        
        # Pego o resultado do treinamento
        results = model.fit(x_train,
                            y_train,
                            batch_size = batch_size,
                            epochs = training_epochs,
                            callbacks = [lr_scheduler],
                            validation_data = (x_val, y_val)
                            )
        # Encerro o tempo de treinamento
        fit_end = time.time()
        print(f'Training time in seconds: {fit_end - fit_begin}')
        print(f'Training time in minutes: {(fit_end - fit_begin)/60.0}')
        print(f'Training time in hours: {(fit_end - fit_begin)/3600.0}\n')

        # Gráficos
        plt.subplot(211)
        plt.plot(results.history['accuracy'])
        plt.plot(results.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'])

        plt.subplot(212)
        plt.plot(results.history['loss'])
        plt.plot(results.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'])
        plt.tight_layout()
        plt.savefig(r'accuracy-loss.png', format='png')
        plt.show()
        
        # Perda minima e maxima e a diferença
        max_loss = np.max(results.history['loss'])
        min_loss = np.min(results.history['loss'])
        print("Maximum Loss : {:.4f}".format(max_loss))
        print("Minimum Loss : {:.4f}".format(min_loss))
        print("Loss difference : {:.4f}\n".format((max_loss - min_loss)))

        # Salvo o modelo no arquivo .h5
        model.save('/media/work/mariapaula/IC/ModeloMixado/model_weights.h5')
        print('model was saved to model_weights.h5.\n')

    # Executa o modo de identificação, ou seja, nao digitei --noimode para ver o quão bem a rede consegue classificar sinais EEG após o treinamento,
    if(not args.noimode):

        # Verifica se nao pulou o treino
        if(model is None):
            model = models.create_model_mixed(window_size, num_channels, num_classes)
            model.summary()
            model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
            model.load_weights('model_weights.h5', by_name=True)

        print('\nEvaluating on training set...')
        (loss, accuracy) = model.evaluate(x_train, y_train, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        print('Evaluating on validation set...')
        (loss, accuracy) = model.evaluate(x_val, y_val, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        print('Evaluating on testing set...')
        test_begin = time.time()

        (loss, accuracy) = model.evaluate(x_test, y_test, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        test_end = time.time()
        print(f'Evaluating on testing set time in miliseconds: {(test_end - test_begin) * 1000.0}')
        print(f'Evaluating on testing set time in seconds: {test_end - test_begin}')
        print(f'Evaluating on testing set time in minutes: {(test_end - test_begin)/60.0}\n')

    # Começa o modo de verificação, ou seja nao digitei --novmode
    if(not args.novmode):

        # Recria o modelo
        model_for_verification = models.create_model_mixed(window_size, num_channels, num_classes, True)
        model_for_verification.summary()
        model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        # Carrega os pesos treinados para esse novo modelo
        model_for_verification.load_weights('model_weights.h5', by_name=True)

        x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

        # Calcula as metricas biometricas
        y_test_classes = utils.one_hot_encoding_to_classes(y_test)
        d, eer, thresholds = utils.calc_metrics(x_pred, y_test_classes, x_pred, y_test_classes)
        print(f'EER: {eer*100.0} %')
        print(f'Decidability: {d}')

# Usando o data generators
else:

    # Carrego os dados
    train_content, test_content = loader.load_data(folder_path, train_tasks, test_tasks, 'csv', total_subjects)   

    # Filtro
    test_content = preprocessing.filter_data(test_content, band_pass_2, sample_frequency, filter_order, filter_type)

    # Normalizo
    test_content = preprocessing.normalize_data(test_content, 'each_channel')

    # Janelamento
    x_test, y_test = data_manipulation.crop_data(test_content, test_tasks, num_classes, window_size, window_size)

    # Sobre cada tarefa
    for task in train_tasks:
        
        # Verifica se nao ja processou
        if(not os.path.exists(processed_data_path + 'processed_data/task'+str(task))):
            folder = Path(processed_data_path + 'processed_data/task'+str(task))
            folder.mkdir(parents=True)

            # Carrega o dado bruto de apenas uma tarefa por vez
            train_content, test_content = loader.load_data(folder_path, [task], [], 'csv', num_classes)

            # Aplica os filtros
            train_content = preprocessing.filter_data(train_content, band_pass_2, sample_frequency, filter_order, filter_type)

            # Normalizo
            train_content = preprocessing.normalize_data(train_content, 'each_channel')
        
            # Percorro cada individuo da tarefa e transformo em arquivos fisicos
            list_csv = []
            for index in range(0, len(train_content)):
                data = train_content[index]
                string = 'x_subject_' + str(index+1)
                savetxt(processed_data_path + 'processed_data/task' + str(task) + '/' + string + '.csv', data, fmt='%f', delimiter=';')
                print(processed_data_path + 'processed_data/task' + str(task) + '/' + string + '.csv was saved.')
                list_csv.append(string+'.csv')
                
            savetxt(processed_data_path + 'processed_data/task' + str(task) + '/' + 'x_list.csv', [list_csv], delimiter=',', fmt='%s')
            print(f'file names were saved to processed_data/task{task}/x_list.csv')

    # Crio lista para armazenar o caminho de todos os arquivos de todas as tarefas
    x_train_list = []
    for subject_idx in range(1, training_subjects_end + 1):
        for task in train_tasks:
            x_train_list.append(f'x_subject_{subject_idx}.csv')

    # Defino o data generator de treino e validacao
    training_generator = data_manipulation.DataGenerator(x_train_list, batch_size, window_size, offset,
        num_channels, num_classes, train_tasks, 'train', split_ratio, processed_data_path, True, True)
    validation_generator = data_manipulation.DataGenerator(x_train_list, batch_size, window_size, offset,
        num_channels, num_classes, train_tasks, 'validation', split_ratio, processed_data_path, True, False)

    # Verifica se nao usou o nofit, ou seja se eu digitei --nofit vai usar o modelo com pesos prontos
    if(not args.nofit):
        model = models.create_model_mixed(window_size, num_channels, num_classes)
        # model = models.create_model_causal(window_size, num_channels, num_classes) ##
        model.summary()

        # model.load_weights('model_weights.h5', by_name=True) ###### When the connection breaks ######

        model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

        fit_begin = time.time()

        # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001) ##
        # model_checkpoint = ModelCheckpoint(filepath='resnet1d_best_model.hdf5', monitor='loss',
        #                                     save_best_only=True) ##
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        # Resultado
        results = model.fit(training_generator,
                            validation_data = validation_generator,
                            epochs = training_epochs,
                            callbacks = [lr_scheduler, saver, early_stop]
                            # callbacks = [reduce_lr, model_checkpoint] ##
                            )

        fit_end = time.time()
        print(f'Training time in seconds: {fit_end - fit_begin}')
        print(f'Training time in minutes: {(fit_end - fit_begin)/60.0}')
        print(f'Training time in hours: {(fit_end - fit_begin)/3600.0}\n')

        # Gráficos
        plt.subplot(211)
        plt.plot(results.history['accuracy'])
        plt.plot(results.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'])

        plt.subplot(212)
        plt.plot(results.history['loss'])
        plt.plot(results.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'])
        plt.tight_layout()
        plt.savefig(r'accuracy-loss.png', format='png')
        plt.show()

        max_loss = np.max(results.history['loss'])
        min_loss = np.min(results.history['loss'])
        print("Maximum Loss : {:.4f}".format(max_loss))
        print("Minimum Loss : {:.4f}".format(min_loss))
        print("Loss difference : {:.4f}\n".format((max_loss - min_loss)))
        
        model.save('model_weights.h5')
        print('model was saved to model_weights.h5.\n')

    # Executa o modo de identificação, ou seja se eu nao digitei --noimode eu estou usando
    if(not args.noimode):

        # Se for nenhum eu crio
        if(model is None):
            model = models.create_model_mixed(window_size, num_channels, num_classes) ##
            model.summary()
            model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
            model.load_weights('model_weights.h5', by_name=True)

        print('\nEvaluating on training set...')
        # Avalia o desempenho final nos geradores tabem
        (loss, accuracy) = model.evaluate(training_generator, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        print('Evaluating on validation set...')
        (loss, accuracy) = model.evaluate(validation_generator, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        print('Evaluating on testing set...')
        test_begin = time.time()

        (loss, accuracy) = model.evaluate(x_test, y_test, verbose = 0)
        print('loss={:.4f}, accuracy: {:.4f}%\n'.format(loss,accuracy * 100))

        test_end = time.time()
        print(f'Evaluating on testing set time in miliseconds: {(test_end - test_begin) * 1000.0}')
        print(f'Evaluating on testing set time in seconds: {test_end - test_begin}')
        print(f'Evaluating on testing set time in minutes: {(test_end - test_begin)/60.0}\n')
    
    # Executa o modo de verificacao, ou seja se eu nao digitei --novmode eu estou usando
    if(not args.novmode):

        model_for_verification = models.create_model_mixed(window_size, num_channels, num_classes, True) ##
        model_for_verification.summary()
        model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model_for_verification.load_weights('model_weights.h5', by_name=True)

        x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

        y_test_classes = utils.one_hot_encoding_to_classes(y_test)
        d, eer, thresholds = utils.calc_metrics(x_pred, y_test_classes, x_pred, y_test_classes)
        print(f'EER: {eer * 100.0} %')
        print(f'Decidability: {d}')
