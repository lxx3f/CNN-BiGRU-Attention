from SVM_SPAM import SVM_SPAM
from tqdm import tqdm
from CNN_BiGRU_Attention import CNN_BiGRU_SPAM


def test_SVM():
    svm = SVM_SPAM(isprinted=False)
    svm.load_data()
    for max_len in tqdm(range(50, 500, 10)):
        for max_features in tqdm(range(1000, 10000, 1000)):
            svm.set_max_len(max_len)
            svm.set_max_features(max_features)
            svm.train()
            svm.save_to_log()

def test_CNN_BiGRU():
    model = CNN_BiGRU_SPAM()


if __name__ == "__main__":
    # test_SVM()
    test_CNN_BiGRU()
