'''
USE XGB to predict the hemoglobin value
dataset: EHR data from Stanford University
'''
from ALLlibrary import *
def get_dataset():
    '''
    function: get the feature and label

    "input": none
    "return": All_X_NEW: feature of demographic,lab test result/ meditation/ vitals
              Hemoglobin_value: label for regression
              anemia_flag: label for classification
    '''
    Path_dir = "../dataset/New_total_dataset.csv"
    All_X = pd.read_csv(Path_dir)
    #delete the patient with 0 hemoglobin value
    All_X_NEW = All_X[~All_X['hemoglobin:Value'].isin([0])]
    #delete the patient with 7<hemoglobin value <17 (too few samples)
    indexNames = All_X_NEW[All_X_NEW['hemoglobin:Value'] < 7.0].index
    All_X_NEW.drop(indexNames, inplace=True)
    indexNames = All_X_NEW[All_X_NEW['hemoglobin:Value'] > 17.0].index
    All_X_NEW.drop(indexNames, inplace=True)
    All_X_NEW = All_X_NEW.reset_index(drop=True)

    #determine the anemia flag according to the rule from WHO
    anemia_flag = []  #list for anemia flag for each patient
    for i in range(len(All_X_NEW)):
        if 0 <= All_X_NEW.loc[i, 'current_age_yrs'] < 5:
            if All_X.loc[i, 'hemoglobin:Value'] < 11:
                anemia_flag.append(1)
            else:
                anemia_flag.append(0)
        elif 5 <= All_X_NEW.loc[i, 'current_age_yrs'] < 12:
            if All_X_NEW.loc[i, 'hemoglobin:Value'] < 11.5:
                anemia_flag.append(1)
            else:
                anemia_flag.append(0)
        elif 12 <= All_X_NEW.loc[i, 'current_age_yrs'] < 15:
            if All_X_NEW.loc[i, 'hemoglobin:Value'] < 12:
                anemia_flag.append(1)
            else:
                anemia_flag.append(0)
        elif 15 <= All_X_NEW.loc[i, 'current_age_yrs']:
            if All_X_NEW.loc[i, 'Male'] == 1:  # 男的
                if All_X_NEW.loc[i, 'hemoglobin:Value'] < 13:
                    anemia_flag.append(1)
                else:
                    anemia_flag.append(0)
            elif All_X_NEW.loc[i, 'Male'] == 0:  # 女的
                if All_X_NEW.loc[i, 'hemoglobin:Value'] < 12:
                    anemia_flag.append(1)
                else:
                    anemia_flag.append(0)
    # extract hemoglobin:Value as the regression lanbel
    Hemoglobin_value = All_X_NEW['hemoglobin:Value']
    # delete the columns with all 0 and the columns of anemia diagnosis results
    All_X_NEW = All_X_NEW.drop(labels=None, axis=0, index=None,
                               columns=['Unnamed: 0',
                                        'hemoglobin:Value',
                                        'hemoglobin:Binary',
                                        'Acquired hemolytic anemias:frequency',
                                        'Aplastic anemia and other bone marrow failure syndromes:frequency',
                                        'Hereditary hemolytic anemias:frequency',
                                        'Iron deficiency anemias:frequency',
                                        'Other and unspecified anemias:frequency',
                                        'Other deficiency anemias:frequency',
                                        'Acquired hemolytic anemias:presence',
                                        'Aplastic anemia and other bone marrow failure syndromes:presence',
                                        'Hereditary hemolytic anemias:presence',
                                        'Iron deficiency anemias:presence',
                                        'Other and unspecified anemias:presence',
                                        'Other deficiency anemias:presence'],
                               inplace=False)
    All_X_NEW = All_X_NEW.loc[:, (All_X_NEW != 0).any(axis=0)]
    #delete the vitals feature
    All_X_NEW = All_X_NEW.iloc[:, 0:1258]
    #delete the outliers in each feature
    for i in range(len(All_X_NEW.iloc[0])):
        num_zero = 0
        sum = 0
        flag_outlier = 0

        data = np.array(All_X_NEW.iloc[:, i:i + 1])
        for n in range(len(data)):
            if data[n] == 0:
                num_zero = num_zero + 1
        aver = data.sum() / (len(data) - num_zero)  # calculate the mean without 0 nvalue

        # calculate the std
        for j in range(len(data)):
            if data[j] != 0:
                sum = sum + (data[j] - aver) * (data[j] - aver)
        std = sum / (len(data) - num_zero)  # 方差

        # Determine whether each point is an outlier
        for m in range(len(data)):
            if data[m] != 0:
                if pow((data[m] - aver), 2) > (3 * std):  # 3σ rule
                    All_X_NEW.iloc[m, i] = aver
                    flag_outlier = flag_outlier + 1
    return All_X_NEW, anemia_flag, Hemoglobin_value
def normalize_feature(dataset,method,chose_top_20feature_flag):
    '''
    normlize the feature and chose the feature to use

    :param dataset: the dataset needs to be normalized, the feature for predicting
    :param method: normalize method. 'StandardScaler': 0 mean with 1 var.
                                     'minmax_scale': [0,1]
    :param chose_top_20feature_flag: 1: chose the topest 20 feature
                                     0: use all the feature
    :return: dataset: normalized feature
    '''
    if method=='StandardScaler':
        if chose_top_20feature_flag==1:
            normalized_feature=scale(dataset)
            for i in range(len(dataset)):
                dataset.iloc[i] = normalized_feature[i, :]
            TOP_ALL_Stand = [
                    'alk:Binary',
                    'lactate:Binary',
                    'Other diseases of blood and blood-forming organs:presence',
                    'Male',
                    'LOCAL ANESTHETICS:Frequeny',
                    'ADRENERGIC AGENTS,CATECHOLAMINES:Frequeny',
                    'MULTIVITAMIN PREPARATIONS:Binary',
                    'SYMPATHOMIMETIC AGENTS:Frequeny',
                    'OTHER DISORDERS OF THE CENTRAL NERVOUS SYSTEM:presence',
                    'COMPLICATIONS OF SURGICAL AND MEDICAL CARE, NOT ELSEWHERE CLASSIFIED:frequency',
                    'ANTI-ANXIETY DRUGS:Binary',
                    'ACUTE RESPIRATORY INFECTIONS:frequency',
                    'Diseases of white blood cells:presence',
                    'albumin:Value',
                    'SMOKER_Y',
                    'HEREDITARY AND DEGENERATIVE DISEASES OF THE CENTRAL NERVOUS SYSTEM:presence',
                    ' PAIN:frequency',
                    'Female',
                    'OTHER DISEASES OF SKIN AND SUBCUTANEOUS TISSUE:frequency',
                    'NARCOTIC ANALGESIC AND NON-SALICYLATE ANALGESIC:Frequeny'
                ]
            dataset = dataset[TOP_ALL_Stand]
    elif method=='minmax_scale':
        if chose_top_20feature_flag==1:
            normalized_feature = minmax_scale(dataset)
            for i in range(len(dataset)):
                dataset.iloc[i] = normalized_feature[i, :]
            TOP_ALL_Zero_One = [
                'ANTIHISTAMINES - 1ST GENERATION:Binary',
                'Other diseases of blood and blood-forming organs:presence',
                'lactate:Binary',
                'SYMPATHOMIMETIC AGENTS:Frequeny',
                'bilirubin:Binary',
                'Male',
                'COMPLICATIONS OF SURGICAL AND MEDICAL CARE, NOT ELSEWHERE CLASSIFIED:frequency',
                'ADRENERGIC AGENTS,CATECHOLAMINES:Frequeny',
                'INFECTIONS OF SKIN AND SUBCUTANEOUS TISSUE:frequency',
                'ANTI-ANXIETY DRUGS:Binary',
                ' NUTRITIONAL DEFICIENCIES:frequency',
                'MULTIVITAMIN PREPARATIONS:Binary',
                'lactate:Value',

                'ACUTE RESPIRATORY INFECTIONS:frequency',
                'FRACTURES:frequency',
                'Diseases of white blood cells:presence',
                'MYCOSES:presence',
                'SEDATIVE-HYPNOTICS,NON-BARBITURATE:Frequeny',
                'NEOPLASMS:frequency',
                'NARCOTIC ANALGESIC AND NON-SALICYLATE ANALGESIC:Frequeny'
            ]
            dataset = dataset[TOP_ALL_Zero_One]
    return dataset
def performance_evaluate_regression(x_test, y_test):
    '''
    calculate evaluation metrix and plot the predict result:
                       evaluation metrix:  1. RMSE
                                           2. MAE
                                           3. R2 score

                                    plot:  1. the image of true value and predict value (sorted by order)
                                           2. the histogram of true value and predict value (distribution)

    :param x_test: x_test
    :param y_test: y_test
    :return: none
    '''
    #get the predict result
    y_pred = model.predict(x_test)
    #calculate MSE, RMSE and R2 score as the regression evaluation metrix
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = math.sqrt(MSE)
    print('RMSE is %.2f' % RMSE)
    MAE = mean_absolute_error(y_test, y_pred)
    print('MAE is %.2f' % MAE)
    r2score = r2_score(y_test, y_pred)
    print('r2score is %.2f' % r2score)

    # sort the predict and true value by order
    order = np.argsort(y_test.flatten())
    y_pred, y_test = y_pred[order], y_test[order]

    # plot the sorted predict and true value
    plt.figure()
    plt.plot(y_test, label='True', c='#FF4500')
    from matplotlib.pyplot import MultipleLocator
    x = range(0, len(y_pred), 1)
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(x_major_locator)  # x-axis is divided by 1
    plt.grid()
    plt.scatter(x, y_pred, label='Prediction')
    plt.legend()
    plt.savefig("../results/True_pre.jpg")

    #plot the histogram of true value and predict value
    plt.figure()
    bins = range(7, 18, 1)
    plt.hist(y_pred, bins, alpha=0.8, label='y_pre')
    plt.hist(y_test, bins, alpha=0.8, label='y_true')
    plt.legend()
    plt.savefig("../results/pre_hist.jpg")

if __name__ == "__main__":
    print('All started')
    #get　the feature and lable
    All_X_NEW, anemia_flag, Hemoglobin_value=get_dataset()
    #normalize the feature, use 'StandardScaler', chose all feature
    All_X_NEW=normalize_feature(dataset=All_X_NEW,method='minmax_scale',chose_top_20feature_flag=0)
    #Convert all data to array
    All_X_NEW = np.array(All_X_NEW)
    Hemoglobin_value = np.array(Hemoglobin_value)
    anemia_flag=np.array(anemia_flag)

    '''
    build the regression model
    '''
    model = XGBRegressor(
        # objective=fair_obj, #use the customized loss function
        # tree_method='gpu_hist', #if use the gpu, 'gpu_hist' can accelerate the speed
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.1,
        gamma=0.7,
        min_child_weight=5,
        max_delta_step=0.9,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1,
        reg_lambda=1,
        nthread=30)
    #split the dataset into training, validation and test

    x_train, x_test, y_train, y_test = train_test_split(All_X_NEW, Hemoglobin_value, test_size=0.2, random_state=1)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.1, random_state=1)

    #train the model
    eval_set = [(x_train, y_train), (x_validation, y_validation)]
    model.fit(x_train, y_train,
              early_stopping_rounds=10,
              eval_set=eval_set,
              eval_metric="rmse"
              )
    #evaluate the model
    performance_evaluate_regression(x_test, y_test)

    '''
       build the classification model
    '''
    model = XGBRFClassifier(
        objective='binary:logitraw',
        # tree_method='gpu_hist', #if use the gpu, 'gpu_hist' can accelerate the speed
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.1,
        gamma=0.7,
        min_child_weight=5,
        max_delta_step=0.9,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1,
        reg_lambda=1,
        nthread=30)
    # split the dataset into training, validation and test
    x_train, x_test, y_train, y_test = train_test_split(All_X_NEW, anemia_flag, test_size=0.2, random_state=1)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.1, random_state=1)

    # train the model
    eval_set = [(x_train, y_train), (x_validation, y_validation)]
    model.fit(x_train, y_train,
              early_stopping_rounds=10,
              eval_set=eval_set,
              eval_metric="auc"
              )
    # evaluate the model
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))








    print('All finished')