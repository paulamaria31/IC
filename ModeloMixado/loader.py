from numpy import savetxt, loadtxt

def load_data(folder_path, train_tasks, test_tasks, file_type, num_classes, verbose=0):
    # Ignoramos train_tasks/test_tasks dos argumentos originais para seguir sua regra fixa
    train_content = []
    test_content = []

    # Sujeitos 1 a 70 -> Treino
    for i in range(1, 71):
        for r in [1, 2]: # R01 e R02
            path = f"{folder_path}S{i:03d}/S{i:03d}R{r:02d}.csv"
            train_content.append(loadtxt(path, delimiter=','))

    # Sujeitos 71 a 109 -> Teste
    for i in range(71, 110):
        for r in [1, 2]:
            path = f"{folder_path}S{i:03d}/S{i:03d}R{r:02d}.csv"
            test_content.append(loadtxt(path, delimiter=','))

    return train_content, test_content
