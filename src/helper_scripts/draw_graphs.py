import matplotlib.pyplot as plt

from src import LIBRARY_ROOT

font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16
        }


def plot_loss(all_lines):
    graph_b_x, graph_b_train = get_loss(all_lines)
    plt.plot(graph_b_x, graph_b_train, label='LOSS')
    # plt.plot(graph_b_x, graph_b_dev, label='Validation Accuracy')

    plt.xlabel('Iter', fontdict=font)
    plt.ylabel('Loss', fontdict=font)
    # plt.ylabel('Loss', fontdict=font)

    plt.title('Experiment', fontdict=font)
    # plt.xticks(np.arange(0, len(graph_b_x), step=30))
    plt.legend()
    plt.show()


def get_loss(all_lines):
    graph_x = list()
    graph_y_train = list()
    for i in range(len(all_lines)):
        line_split = all_lines[i].split(':')
        if line_split[0] == 'Dev-Acc':
            itr = all_lines[i - 1].split(':')[0]
            loss = all_lines[i - 1].split(':')[3]
            graph_x.append(itr)
            graph_y_train.append(float(loss))

    return graph_x, graph_y_train


def plot_accuracy_dev(all_lines):
    dev_x, dev_y, train_x, train_y = get_accuracy(all_lines)
    plt.plot(dev_x, dev_y, label='Dev Accuracy')
    plt.plot(train_x, train_y, label='Train Accuracy')

    plt.xlabel('Iter', fontdict=font)
    plt.ylabel('Accuracy', fontdict=font)

    plt.title('Experiment', fontdict=font)
    # plt.xticks(np.arange(0, len(graph_b_x), step=30))
    plt.legend()
    plt.show()


def get_accuracy(all_lines):
    dev_x = list()
    dev_y = list()
    train_x = list()
    train_y = list()
    for i in range(len(all_lines)):
        line_split = all_lines[i].split(':')
        if line_split[0] == 'Dev-Acc':
            itr = line_split[1]
            accuracy = line_split[3]
            dev_x.append(itr)
            dev_y.append(float(accuracy))
        if line_split[0] == 'Train-Acc':
            itr = line_split[1]
            accuracy = line_split[3]
            train_x.append(itr)
            train_y.append(float(accuracy))

    return dev_x, dev_y, train_x, train_y


def get_prf1(all_lines):
    itr_px = list()
    graph_py = list()
    graph_ry = list()
    graph_f1y = list()
    for i in range(len(all_lines)):
        line_split = all_lines[i].split(':')
        if line_split[0] == 'Dev-Acc':
            itr = line_split[1]
            p = line_split[5]
            r = line_split[7]
            f1 = line_split[9]
            itr_px.append(itr)
            graph_py.append(float(p))
            graph_ry.append(float(r))
            graph_f1y.append(float(f1))

    return itr_px, graph_py, graph_ry, graph_f1y


def plot_prf1(all_lines):
    x, p_y, r_y, f1_y = get_prf1(all_lines)
    plt.plot(x, p_y, label='Precision')
    plt.plot(x, r_y, label='Recall')
    plt.plot(x, f1_y, label='F1')

    plt.xlabel('Iter', fontdict=font)
    plt.ylabel('P/R/F1', fontdict=font)

    plt.title('Experiment', fontdict=font)
    # plt.xticks(np.arange(0, len(graph_b_x), step=30))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    file_name = str(LIBRARY_ROOT) + '/best_reports/ECB_lr1e-06_bs32_a7_itr20.txt'
    with open(file_name, 'r') as file_fs:
        all_lines = file_fs.readlines()
        # plot_loss(all_lines)
        # plot_accuracy_dev(all_lines)
        plot_prf1(all_lines)