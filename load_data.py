import sys
import os

def data_loading():
    # 現在のスクリプトのディレクトリを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
        
    # 同じディレクトリにある 'deep-learning-from-scratch' ディレクトリを追加
    module_path = os.path.join(current_dir, "deep-learning-from-scratch")
    sys.path.append(module_path)
        
    # 正しいモジュールをインポート
    from dataset.mnist import load_mnist
        
    # load_mnist関数を使用してデータをロード
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    
    return x_train, t_train, x_test, t_test