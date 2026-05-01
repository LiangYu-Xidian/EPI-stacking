from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from evidential_random_forest import ERF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def split_positive_negative(X, y):
    """将特征集X和标签y按顺序分割为正负样本两部分
    
    参数:
        X (np.ndarray): 特征矩阵，形状为(n_samples, n_features)
        y (np.ndarray): 标签列向量，形状为(n_samples,)，前部分为1，后部分为0
    
    返回:
        X_pos (np.ndarray): 正样本特征集
        X_neg (np.ndarray): 负样本特征集
        y_pos (np.ndarray): 正样本标签（全1）
        y_neg (np.ndarray): 负样本标签（全0）
    """
    # 确保y是1D数组（若输入是列向量则展平）
    y = y.ravel()
    
    # 计算正样本数量（假设前部分全为1，后部分全为0）
    n_pos = np.sum(y == 1)
    
    # 检查数据是否符合假设
    assert np.all(y[:n_pos] == 1), "前部分标签不全为1"
    assert np.all(y[n_pos:] == 0), "后部分标签不全为0"
    
    # 分割X和y
    X_pos = X[:n_pos]
    X_neg = X[n_pos:]
    y_pos = y[:n_pos]
    y_neg = y[n_pos:]
    
    return X_pos, X_neg, y_pos, y_neg

def svm_train_predict(X_train, Y_label, X_predict):
    """使用RBF核优化 + 数据标准化"""

    # 标准化数据（SVM对尺度敏感）


    model = SVC(
        kernel='rbf',            # 保持RBF核以捕获非线性关系
        C=10.0,                  # 增大正则化强度，调优过拟合
        gamma='scale',           # 自动计算gamma，避免手动设置
        probability=True,        # 启用概率输出
        random_state=42,
        cache_size=1000          # 增大缓存加速计算（单位：MB）
    )
    model.fit(X_train, Y_label)
    print("SVM")
    return model.predict_proba(X_predict)[:, 1].reshape(-1, 1)

def rf_train_predict(X_train, Y_label, X_predict):
    """增加树深 + 特征采样优化"""
    model = RandomForestClassifier(
        n_estimators=200,        # 增加树的数量提升稳定性
        max_depth=None,          # 允许完全生长（大数据下可控制为50-100）
        min_samples_split=20,    # 防止过拟合的小样本分裂
        max_features='sqrt',     # 每棵树随机采样 sqrt(n_features) 个特征
        n_jobs=-1,              # 全核并行
        random_state=42,
        class_weight='balanced' # 处理类别不均衡
    )
    model.fit(X_train, Y_label)
    print("RF")
    return model.predict_proba(X_predict)[:, 1].reshape(-1, 1)

def lr_train_predict(X_train, Y_label, X_predict):
    """多项式特征 + 弹性网络正则化"""

    # 创建管道：多项式扩展 + 标准化 + 模型
    model = LogisticRegression(
        penalty='elasticnet',  # 弹性网络正则化
        solver='saga',         # 唯一支持 elasticnet 的求解器
        l1_ratio=0.5,          # L1/L2正则混合比例
        max_iter=100000,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, Y_label)
    print("LR")
    return model.predict_proba(X_predict)[:, 1].reshape(-1, 1)

def xgboost_train_predict(X_train, Y_label, X_predict):
    """深度调参 + 早停优化"""
    model = XGBClassifier(
        n_estimators=2000,        # 设置足够大的树数，依赖早停
        learning_rate=0.05,       # 降低学习率，提升泛化
        max_depth=8,              # 控制树复杂度
        subsample=0.8,            # 行采样
        colsample_bytree=0.8,     # 列采样
        gamma=0.1,                # 节点分裂最小损失减少
        reg_alpha=0.1,            # L1正则
        reg_lambda=1.0,           # L2正则
        n_jobs=-1,
        random_state=42,
        tree_method='hist'      # 内存友好
    )
    # 划分验证集
  
    model.fit(X_train, Y_label)
    print("XGBOOST")
    return model.predict_proba(X_predict)[:, 1].reshape(-1, 1)


def knn_train_predict(X_train, Y_label, X_predict):
    """PCA降维 + 近似搜索优化"""
    # 降维至50-100维（根据特征数调整）

    model = KNeighborsClassifier(
        n_neighbors=50,           # 增大邻居数平滑决策
        weights='distance',       # 距离加权投票
        algorithm='ball_tree',    # 适合高维数据
        leaf_size=40,
        p=2,                      # 欧氏距离
        n_jobs=-1
    )
    model.fit(X_train, Y_label)
    print("KNN")
    return model.predict_proba(X_predict)[:, 1].reshape(-1, 1)

def combine_predictions(X_train, Y_label, X_predict):
    """集成多个模型的预测概率，返回列拼接矩阵"""
    # 依次调用各个模型的预测函数
    svm_pred = svm_train_predict(X_train, Y_label, X_predict)
    rf_pred = rf_train_predict(X_train, Y_label, X_predict)
    lr_pred = lr_train_predict(X_train, Y_label, X_predict)
    xgb_pred = xgboost_train_predict(X_train, Y_label, X_predict)
    knn_pred = knn_train_predict(X_train, Y_label, X_predict)
    
    # 水平拼接所有预测结果
    print("*****************")
    return np.hstack((svm_pred, rf_pred, lr_pred, xgb_pred, knn_pred))

def one_main():
    base_path = './stackEPI/K5/'  # 修改变量名避免与标准库冲突
    files = ["kmer.csv", "csk.csv", "DPCP.csv", "TPCP.csv", "NAC.csv", "pseKNC.csv"]
    res = []
    
    for file in files:
        cell = "K5"
        X_train_file = base_path + cell + '_train_' + file
        X_test_file = base_path + cell + '_test_' + file
        Y_label = base_path + "K562_label.txt"
        
        X_train = np.loadtxt(X_train_file, delimiter=',') 
        X_test = np.loadtxt(X_test_file, delimiter=',') 
        y_labels = np.loadtxt(Y_label).reshape(-1, 1).ravel()
        print("数据准备完成！！！")
        y_prd = combine_predictions(X_train, y_labels, X_test)  # 确保该函数已定义
        
        res.append(y_prd)
    
    # 顶层缩进（0个空格）
    result_hstack = np.hstack(res)
    np.savetxt(base_path + "K5_test.csv", result_hstack, delimiter=',', fmt='%f')
    
def two_main():
    base_path = './stackEPI/NHEK/'  # 修改变量名避免与标准库冲突
    files = ["kmer.csv", "csk.csv", "DPCP.csv", "TPCP.csv", "NAC.csv", "pseKNC.csv"]
    n_folds = 5
    res = []
    
    for file in files:
        cell = "NHEK"
        X_file = base_path + cell + '_train_' + file
        Y_file = base_path + "NHEK_label.txt"
        
        X = np.loadtxt(X_file, delimiter=',')
        y = np.loadtxt(Y_file).reshape(-1, 1)
        
        X_pos, X_neg, y_pos, y_neg = split_positive_negative(X, y)
        
        X_pos_folds = np.array_split(X_pos, n_folds)  # 列表，包含5个子数组
        X_neg_folds = np.array_split(X_neg, n_folds)  # 列表，包含5个子数组
        
        y_pos_folds = np.array_split(y_pos, n_folds)
        y_neg_folds = np.array_split(y_neg, n_folds)
        
        y_pos_prd = []
        y_neg_prd = []
        print(X_file)
        for i in range(n_folds):
            X_pos_test = X_pos_folds[i]
            X_neg_test = X_neg_folds[i]
            
            X_test = np.vstack((X_pos_test, X_neg_test))
            
            y_pos_test = y_pos_folds[i]
            y_neg_test = y_neg_folds[i]
            
            Y_test = np.concatenate((y_pos_test, y_neg_test))
            
            X_pos_train = np.vstack([X_pos_folds[j] for j in range(n_folds) if j != i])
            X_neg_train = np.vstack([X_neg_folds[j] for j in range(n_folds) if j != i])
            
            X_train = np.vstack((X_pos_train, X_neg_train))
            
            y_pos_train = np.concatenate([y_pos_folds[j] for j in range(n_folds) if j != i])
            y_neg_train = np.concatenate([y_neg_folds[j] for j in range(n_folds) if j != i])
            
            y_train = np.concatenate((y_pos_train, y_neg_train))
            
            print("数据准备完毕:", i)
            y_prd_fold = combine_predictions(X_train, y_train, X_test)
            y_pos_fold, y_neg_fold, _, _ = split_positive_negative(y_prd_fold, Y_test)
            
            y_pos_prd.append(y_pos_fold)
            y_neg_prd.append(y_neg_fold)
            
        
        y_pos = np.vstack(y_pos_prd)
        y_neg = np.vstack(y_neg_prd)
        
        y_prd = np.vstack((y_pos, y_neg))
        res.append(y_prd)
        
    # 顶层缩进（0个空格）
    result_hstack = np.hstack(res)
    np.savetxt(base_path + "NHEK_train.csv", result_hstack, delimiter=',', fmt='%f')
    
def predict_by_ERT():
    classifier = ERF()
    
    base_path = './stackEPI/'  
    cell = "K5"
    X_train_file = base_path + cell + '/' + cell + '_train.csv'
    X_test_file = base_path + cell + '/' + cell + '_test.csv'
    Y_train = base_path + cell + '/' + "K562_label.txt"
    Y_test = base_path + cell + '/' + "K562_label_test.txt"
    
    
    X_train = np.loadtxt(X_train_file, delimiter=',') 
    X_test = np.loadtxt(X_test_file, delimiter=',') 
    Y_trains = np.loadtxt(Y_train).reshape(-1, 1)
    Y_tests = np.loadtxt(Y_test).reshape(-1, 1)
    
    classifier.fit(X_train, Y_trains)
    
    precision = classifier.score(X_test, Y_tests)

    print("Accuracy : ", precision)

def predict_by_RF():
    
    
    base_path = './stackEPI/GM/'  
    
    files = [""] 
   
    Y_train = base_path + "GM_label.txt"
    Y_test = base_path + "GM12878_label_test.txt"
    
    
   
    
    for file in files:
        cell = "GM"
        X_train_file = base_path + cell + '_train_' + file
        X_test_file = base_path + cell + '_test_' + file
       
        
        if file == "":
            X_train_file = base_path + cell + "_train.csv"
            X_test_file = base_path + cell + "_test.csv"
            
        X_train = np.loadtxt(X_train_file, delimiter=',') 
        X_test = np.loadtxt(X_test_file, delimiter=',')
        Y_trains = np.loadtxt(Y_train).reshape(-1, 1).ravel()
        Y_tests = np.loadtxt(Y_test).reshape(-1, 1).ravel()
        
        # RF
        
        model = RandomForestClassifier(
            n_estimators=1000,        # 增加树的数量提升稳定性
            max_depth=None,          # 允许完全生长（大数据下可控制为50-100）
            min_samples_split=20,    # 防止过拟合的小样本分裂
            max_features='sqrt',     # 每棵树随机采样 sqrt(n_features) 个特征
            n_jobs=-1,              # 全核并行
            random_state=42,
            class_weight='balanced' # 处理类别不均衡
        )
        
        
        '''
        model = SVC(
            kernel='rbf',            # 保持RBF核以捕获非线性关系
            C=10.0,                  # 增大正则化强度，调优过拟合
            gamma='scale',           # 自动计算gamma，避免手动设置
            probability=True,        # 启用概率输出
            random_state=42,
            cache_size=1000          # 增大缓存加速计算（单位：MB）
        )
        '''
        
        '''
        model = LogisticRegression(
            penalty='elasticnet',  # 弹性网络正则化
            solver='saga',         # 唯一支持 elasticnet 的求解器
            l1_ratio=0.5,          # L1/L2正则混合比例
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        '''
        
        '''
        model = XGBClassifier(
            n_estimators=2000,        # 设置足够大的树数，依赖早停
            learning_rate=0.05,       # 降低学习率，提升泛化
            max_depth=8,              # 控制树复杂度
            subsample=0.8,            # 行采样
            colsample_bytree=0.8,     # 列采样
            gamma=0.1,                # 节点分裂最小损失减少
            reg_alpha=0.1,            # L1正则
            reg_lambda=1.0,           # L2正则
            n_jobs=-1,
            random_state=42,
            tree_method='hist'      # 内存友好
        )
        '''
        
        '''
        model = KNeighborsClassifier(
            n_neighbors=50,           # 增大邻居数平滑决策
            weights='distance',       # 距离加权投票
            algorithm='ball_tree',    # 适合高维数据
            leaf_size=40,
            p=2,                      # 欧氏距离
            n_jobs=-1
        )
        '''
        
        
        model.fit(X_train, Y_trains)
        
        y_proba = model.predict_proba(X_test)[:, 1]  # 获取正类的概率

        # 计算评估指标
        auc = roc_auc_score(Y_tests, y_proba)
        aupr = average_precision_score(Y_tests, y_proba)
        
        print(file)
        print(f"AUC: {auc:.4f}", end=" ")
        print(f"AUPR: {aupr:.4f}")
    
def predict_cross():
    
    
    X_train = np.loadtxt('./stackEPI/NHEK/NHEK_train.csv', delimiter=',') 
    Y_train = np.loadtxt('./stackEPI/NHEK/NHEK_label.txt', delimiter=',').reshape(-1, 1).ravel()
    
    X_str = ["/GM/GM_test.csv", "/HeLa/HeLa_test.csv", "/HUVEC/HUVEC_test.csv", "/IMR/IMR_test.csv", "/K5/K5_test.csv", "/NHEK/NHEK_test.csv"]
    y_str = ["/GM/GM12878_label_test.txt", "/HeLa/HeLa_label_test.txt", "/HUVEC/HUVEC_label_test.txt", "/IMR/IMR90_label_test.txt", "/K5/K562_label_test.txt", "/NHEK/NHEK_label_test.txt"]
    
    model = RandomForestClassifier(
        n_estimators=10000,        # 增加树的数量提升稳定性
        max_depth=None,          # 允许完全生长（大数据下可控制为50-100）
        min_samples_split=20,    # 防止过拟合的小样本分裂
        max_features='sqrt',     # 每棵树随机采样 sqrt(n_features) 个特征
        n_jobs=-1,              # 全核并行
        random_state=42,
        class_weight='balanced' # 处理类别不均衡
    )
    
    model.fit(X_train, Y_train)
    
    auprs = []
    aucs = []
    for i in range(len(X_str)):
        X_test = np.loadtxt('./stackEPI' + X_str[i], delimiter=',') 
        Y_test = np.loadtxt('./stackEPI' + y_str[i], delimiter=',').reshape(-1, 1).ravel()
    
        
        
        y_proba = model.predict_proba(X_test)[:, 1]  # 获取正类的概率

        # 计算评估指标
        auc = roc_auc_score(Y_test, y_proba)
        aupr = average_precision_score(Y_test, y_proba)
        
        auprs.append(round(aupr, 3))  
        aucs.append(round(auc, 3))
        
    print(aucs)
    print(auprs)



def xgb_cross_val(X, y, r = 0):
    """
    使用XGBoost进行交叉验证并返回平均AUC和AUPR
    参数:
        X: 特征矩阵(numpy数组或DataFrame)
        y: 标签数组(numpy数组或Series)
        n_splits: 交叉验证折数(默认为5)
        **kwargs: 可传递XGBoost参数(如max_depth, learning_rate等)
    """
    # 初始化交叉验证
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=r)
    auc_scores = []
    aupr_scores = []
    acc_scores = []

  

    for train_index, val_index in skf.split(X, y):
        # 划分训练集/验证集
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

       

        # 训练模型
        model = XGBClassifier(
            n_estimators=2000,        # 设置足够大的树数，依赖早停
            learning_rate=0.01,       # 降低学习率，提升泛化
            max_depth=10,              # 控制树复杂度
            subsample=0.8,            # 行采样
            colsample_bytree=0.8,     # 列采样
            gamma=0.1,                # 节点分裂最小损失减少
            reg_alpha=0.15,            # L1正则
            reg_lambda=1.0,           # L2正则
            n_jobs=-1,
            
            tree_method='hist'      # 内存友好
        )


        model.fit(X_train, y_train)
       
        y_proba = model.predict_proba(X_val)[:, 1]  # 获取正类的概率
        y_pred = model.predict(X_val) 

        # 计算评估指标
        auc = roc_auc_score(y_val, y_proba)
        aupr = average_precision_score(y_val, y_proba)
        acc = accuracy_score(y_val, y_pred)
        
        acc_scores.append(acc)
        auc_scores.append(auc)
        aupr_scores.append(aupr)

    # 计算平均指标
    
    avg_acc = np.mean(acc_scores)
    
    mean_auc = np.mean(auc_scores)
    
    mean_aupr = np.mean(aupr_scores)
    

    # 打印结果
    print(f"{n_splits}-折交叉验证结果:")
    print("\n=== 平均结果 ===")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"平均 AUC: {mean_auc:.4f}")
    print(f"平均 AUPR: {mean_aupr:.4f}")
    return avg_acc


    
    
    
   
            
    

