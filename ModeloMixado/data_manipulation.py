import utils
import math
import random
import numpy as np
import tensorflow.keras as keras

#Janelamento e separacao dos dados em treino e validação de um sinal unico
def signal_cropping(x_data, y_data, content, window_size, offset, num_subject, num_classes, split_ratio=1.0, x_data_2=0, y_data_2=0):
    num_subject -= 1 # Converte para índice 0 ou 1

    if offset <= 0:
        print('ERROR: O parâmetro offset deve ser positivo.')
        return x_data, y_data
    elif split_ratio <= 0 or split_ratio > 1:
        print('ERROR: O parâmetro split_ratio deve estar no intervalo (0,1].')
        return x_data, y_data
    else:
        i = window_size
        # Processamento da primeira parte (Treino)
        while i <= content.shape[1] * split_ratio:
            arr = content[: , (i-window_size):i]
            noise = np.random.normal(0, 0.08, arr.shape) 
            arr = arr + noise
            x_data.append(arr)

            arr2 = np.zeros((1, num_classes))
            arr2[0, num_subject] = 1
            y_data.append(arr2)
            i += offset

        if split_ratio == 1.0:
            return x_data, y_data
        
        # Processamento da segunda parte (Validação)
        while i <= content.shape[1]:
            arr = content[: , (i-window_size):i]
            x_data_2.append(arr)

            arr2 = np.zeros((1, num_classes))
            arr2[0, num_subject] = 1
            y_data_2.append(arr2)
            i += offset

        return x_data, y_data, x_data_2, y_data_2

#Janelamento de varios sinais
def crop_data(data, data_tasks, num_classes, window_size, offset, split_ratio=1.0, verbose=0):
    x_dataL, x_dataL_2, y_dataL, y_dataL_2 = [], [], [], []

    if verbose == 1:
        print('Iniciando o recorte dos dados (Janelamento)...')

    # 'data' contém a sequência [S1R1, S1R2, S2R1, S2R2...]
    # Índices pares = R01 (Classe 1), Índices ímpares = R02 (Classe 2)
    for idx, signal in enumerate(data):
        current_class = 1 if (idx % 2 == 0) else 2
        
        if split_ratio == 1.0:
            x_dataL, y_dataL = signal_cropping(x_dataL, y_dataL, signal, 
                                               window_size, offset, current_class, num_classes)
        else:
            x_dataL, y_dataL, x_dataL_2, y_dataL_2 = signal_cropping(x_dataL, y_dataL, signal,
                                                                     window_size, offset, current_class, num_classes,
                                                                     split_ratio, x_dataL_2, y_dataL_2)

    # Conversão para Numpy Array e Reshape para o formato da Rede Neural (CNN/LSTM)
    x_data = np.asarray(x_dataL).astype('float32')
    y_data = np.asarray(y_dataL).astype('float32')
    
    # Reshape: (N, Canais, Janela) -> (N, Janela, Canais)
    x_data = x_data.reshape(x_data.shape[0], x_data.shape[2], x_data.shape[1])
    y_data = y_data.reshape(y_data.shape[0], y_data.shape[2])

    if split_ratio == 1.0:
        return x_data, y_data
    else:
        x_data_2 = np.asarray(x_dataL_2).astype('float32')
        y_data_2 = np.asarray(y_dataL_2).astype('float32')
        x_data_2 = x_data_2.reshape(x_data_2.shape[0], x_data_2.shape[2], x_data_2.shape[1])
        y_data_2 = y_data_2.reshape(y_data_2.shape[0], y_data_2.shape[2])
        return x_data, y_data, x_data_2, y_data_2

#Define em qual indice o conjunto de treinamento termina para que o conjunto de validação comece sem que haja sobreposicao
def first_validation_crop(signal_size, window_size, offset, split_ratio):
    i = window_size
    stop = signal_size * split_ratio
    while(i <= stop):
        i += offset
    return i

#Ao inves de cortar fisicamente, cria uma lista de coordenadas (índice do sinal, ponto de corte) que diz exatamente onde cada janela deve começar e terminar
def get_crop_positions(dataset_type, signal_sizes, window_size, offset, split_ratio):
    crop_positions = []
    signal_index = 0
    for size in signal_sizes:
        if(dataset_type == 'train' or dataset_type == 'test'):
            first_i = window_size
            stop = size * split_ratio
        elif(dataset_type == 'validation'):
            first_i = first_validation_crop(size, window_size, offset, split_ratio)
            stop = size
        i = first_i
        while(i <= stop):
            one_crop_position = (signal_index, i)
            crop_positions.append(one_crop_position)
            i += offset
        signal_index += 1
    return crop_positions

#Data generator (Entender melhor depois)
class DataGenerator(keras.utils.Sequence):
    #Configuração inicial
    def __init__(self, list_files, batch_size, dim, offset, n_channels,
                n_classes, tasks, dataset_type, split_ratio, processed_data_path, train=True, shuffle=False):
        #Salva os parametros para serem usados
        self.list_files = list_files
        self.batch_size = batch_size
        self.dim = dim
        self.offset = offset
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.tasks = tasks
        self.dataset_type = dataset_type
        self.split_ratio = split_ratio
        self.processed_data_path = processed_data_path
        self.shuffle = shuffle
        self.train = train

        data = []
        classes_list = [] # Armazena se é Classe 1 ou 2

        #Inicia um loop pelas tarefas
        for task_idx, task in enumerate(self.tasks):
            #Para cada tarefa percorre a lista de arquivos dos sujeitos
            for file_name in list_files:
                #Monta o caminho exato aonde o arquivo csv foi salvo
                path = f"{processed_data_path}processed_data/task{task}/{file_name}"
                #Lê o sinal EEG
                file_x = np.loadtxt(path, delimiter=';').astype('float32')
                #Adiciona o sinal bruto na lista dat
                data.append(file_x)
                
                # task_idx será 0 para a primeira tarefa (R01 - Olho Aberto) 
                # e 1 para a segunda (R02 - Olho Fechado)
                classes_list.append(task_idx + 1)

        #Calcula o comprimento total de cada sinal
        signal_sizes = [signal.shape[1] for signal in data]
        #Chama a funcao que cria uma lista de coordenadas
        self.crop_positions = get_crop_positions(self.dataset_type, signal_sizes, self.dim, self.offset, self.split_ratio)
        self.data = data
        self.classes_list = classes_list
        #Chama a funcao para embaralhar
        self.on_epoch_end()

    #Retorna os passos de uma epoca
    def __len__(self):
        return math.floor(len(self.crop_positions) / self.batch_size)
    
    #Toda vez que o GPU precisa de um novo lote
    def __getitem__(self, index):
        x, y = [], []
        #Pega as coordenadas das janelas que pertencem ao lote
        crop_positions = self.crop_positions[index*self.batch_size : (index+1)*self.batch_size]

        #Inicia o recorte de cada janela do lote
        for crop_position in crop_positions:
            file_index, crop_end = crop_position
            # Recorta a janela de sinal
            sample = self.data[file_index][:, (crop_end-self.dim):crop_end]
            #Ajusta o formato
            sample = sample.reshape(sample.shape[1], sample.shape[0])

            #Ruido gaussiano
            if self.train:
                # np.random.normal(média, desvio_padrão, formato)
                noise = np.random.normal(0, 0.01, sample.shape)
                sample = sample + noise
            # --------------------------------------

            x.append(sample)

            label = np.zeros((1, self.n_classes))
            label[0, self.classes_list[file_index]-1] = 1
            y.append(label)
            
        x = np.asarray(x).astype('float32')
        y = np.asarray(y).astype('float32').reshape(len(y), self.n_classes)
        return (x, y)

    #Ao final de cada epoca embaralha
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.crop_positions)
