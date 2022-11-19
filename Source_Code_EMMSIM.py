import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from nltk.corpus import stopwords
import time
import xgboost as xgb

def split(dataset, train_size, test_size):
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    x_train, x_pool, y_train, y_pool = train_test_split(
        x, y, train_size = train_size)
    unlabel, x_test, label, y_test = train_test_split(
        x_pool, y_pool, test_size = test_size)
    return x_train, y_train, x_test, y_test, unlabel, label


##################################################### Read Data ###################################
train_data = pd.read_csv(r'G:\My Drive\\active_learning\data\training_data.csv')
train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed', na=False)]  
x_train_start = train_data.iloc[:,0]
y_train_start = train_data.iloc[:,-1]

test_data = pd.read_csv(r'G:\My Drive\My PhD Lab\1. Data\3_1_Churn_Intention\active_learning\data\test_data.csv')
test_data = test_data.loc[:, ~test_data.columns.str.contains('^Unnamed', na=False)]
x_test_start = test_data.iloc[:,0]
y_test_start = test_data.iloc[:,-1]

unlabel_data = pd.read_csv(r'G:\My Drive\My PhD Lab\1. Data\3_1_Churn_Intention\active_learning\data\unlabel_data.csv')
unlabel_data = unlabel_data.loc[:, ~unlabel_data.columns.str.contains('^Unnamed', na=False)]
unlabel_start = unlabel_data.iloc[:,0]
label_start = unlabel_data.iloc[:,-1]

################################ tfidf initialization ############################################
iter=1
run = 0
xgbacc_1, xgbf1_1, xgbprec_1, xgbrec_1 = [], [], [],  [] # arrays to store performance different models
xgbauc_1 = []
sample_size_xgb, t_xgb = [], []
max_sample = 1500 # Budget
batch_size = 25
mode1 = "TFIDF"
x_train = x_train_start.to_numpy()
y_train = y_train_start.to_numpy()

x_test = x_test_start.to_numpy()
y_test = y_test_start.to_numpy()

unlabel = unlabel_start.to_numpy()
label = label_start.to_numpy()


################################### LEAST CONFIDENT SAMPLING ######################################################

while(x_train.shape[0]<= max_sample and iter!=0):

    td = TfidfVectorizer(analyzer='word', stop_words=stopwords.words('french'), 
                                        token_pattern=r'\w{1,}')
    x_train_t = td.fit_transform(x_train).toarray()
    x_test_t = td.transform(x_test).toarray()
    unlabel_t = td.transform(unlabel).toarray()
    
    
    x_train_new = np.concatenate((x_train.reshape(-1, 1),x_train_t), axis = 1)
    x_test_new = np.concatenate((x_test.reshape(-1, 1),x_test_t), axis = 1)
    unlabel_new = np.concatenate((unlabel.reshape(-1, 1),unlabel_t), axis = 1)
    
    xgb1 = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    xgb1.fit(x_train_new[:,1:], y_train)
    
    y_pred = xgb1.predict(x_test_new[:,1:])
    cr2 = classification_report(y_test, y_pred, output_dict=True)
    xgbacc_1.append(cr2['accuracy'])
    xgbf1_1.append(cr2['1']['f1-score'])
    xgbprec_1.append(cr2['1']['precision'])
    xgbrec_1.append(cr2['1']['recall'])
    xgbauc_1.append(roc_auc_score(y_test, xgb1.predict_proba(x_test_new[:,1:])[:, 1]))
    sample_size_xgb.append(x_train_new.shape[0])
    start_xgb = time.perf_counter()
    t_xgb.append(time.perf_counter()-start_xgb)
    usage="vocab"
    mode = "Least Confidence Sampling"+mode1
    
    # Record Performance
    perf = pd.read_excel(r'G:\My Drive\performance.xlsx')
    perf = perf.loc[:, ~perf.columns.str.contains('^Unnamed')]
    
    perf = perf.append({'usage': usage,
                        'run':run,
                        'samp_size': sample_size_xgb[run],
                        'accuracy' : xgbacc_1[run],
                        'f1_score': xgbf1_1[run], 
                        'recall': xgbrec_1[run], 
                        'precision': xgbprec_1[run], 
                        'auc': xgbauc_1[run], 
                        'time': t_xgb[run], 
                        'mode': mode}, ignore_index = True)
    
    writer = pd.ExcelWriter(r'G:\My Drive\performance.xlsx',
                        engine='xlsxwriter',
                        engine_kwargs={'options': {'strings_to_urls': False}})
    perf.to_excel(writer)
    writer.close()
    y_probab = xgb1.predict_proba(unlabel_new[:,1:])[:, 0]
    
    ################ Calculate Least Confidence ##########################################
    lc = []
    for k in range(y_probab.shape[0]):
        if(y_probab[k] == 0.5):
            lc.append(0.5)
        elif (y_probab[k] < 0.5):
            lc.append(y_probab[k])
        else :
            lc.append(1-y_probab[k])
    lc = np.array(lc)
    sort_index = np.argsort(lc)[::-1]
    
    if x_train_new.shape[0]>max_sample:
        iter=0
        
    else:
        
        uncrt_pt_ind = [] # Stores all the selected samples
        uncrt_pt_ind = sort_index[0:batch_size]
        uncrt_pt_ind = uncrt_pt_ind.tolist()

        if  len(uncrt_pt_ind) == 0:
            iter=0
        else:
            if (len(uncrt_pt_ind)+x_train_new.shape[0]>max_sample):
                uncrt_pt_ind = uncrt_pt_ind[0:(max_sample-x_train_new.shape[0])]
                to_label = unlabel_new[uncrt_pt_ind, :]
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\query_'+str(run)+'_'+str(len(to_label))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                pd.DataFrame(to_label).to_excel(writer)
                writer.close()
                complete_label = pd.concat([pd.DataFrame(to_label),pd.DataFrame(label[uncrt_pt_ind])],axis = 1)
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\query_labelled_'+str(run)+'_'+str(len(to_label))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                complete_label.to_excel(writer)
                writer.close()
                
                x_train_new = np.append(unlabel_new[uncrt_pt_ind, :], x_train_new, axis = 0)
                y_train = np.append(label[uncrt_pt_ind], y_train)
                unlabel_new = np.delete(unlabel_new, uncrt_pt_ind, axis = 0)
                label = np.delete(label, uncrt_pt_ind)
                
                run=run+1
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\train_data_'+str(run)+'_'+str(len(x_train))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                pd.concat([pd.DataFrame(x_train_new),pd.DataFrame(y_train)], axis = 1).to_excel(writer)
                writer.close()
                
                xgbf1_1, xgbprec_1, xgbrec_1 = [], [],  [] # arrays to save performance of different models
                xgbauc_1 = []
                sample_size_xgb, t_xgb = [], []
                max_sample = 1500
    
                iter=1
    
                # Train model
                xgb1 = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
                xgb1.fit(x_train_new[:,1:], y_train)
                
                y_pred = xgb1.predict(x_test_new[:,1:])
                cr2 = classification_report(y_test, y_pred, output_dict=True)
                xgbf1_1.append(cr2['1']['f1-score'])
                xgbprec_1.append(cr2['1']['precision'])
                xgbrec_1.append(cr2['1']['recall'])
                xgbauc_1.append(roc_auc_score(y_test, xgb1.predict_proba(x_test_new[:,1:])[:, 1]))
                sample_size_xgb.append(x_train_new.shape[0])
                start_xgb = time.perf_counter()
                t_xgb.append(time.perf_counter()-start_xgb)
                
                # Record Performance
                perf = pd.read_excel(r'G:\My Drive\active_learning\performance.xlsx')
                perf = perf.loc[:, ~perf.columns.str.contains('^Unnamed')]
                perf = perf.append({'usage': usage,
                                    'run':run,
                                    'samp_size': sample_size_xgb[0],
                                    'accuracy' : xgbacc_1[0],
                                    'f1_score': xgbf1_1[0], 
                                    'recall': xgbrec_1[0], 
                                    'precision': xgbprec_1[0], 
                                    'auc': xgbauc_1[0], 
                                    'time': t_xgb[0], 
                                    'mode': mode}, ignore_index = True)
    
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\performance.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                perf.to_excel(writer)
                writer.close()
                iter=0
            else:
                to_label = unlabel_new[uncrt_pt_ind, :]
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\query_'+str(run)+'_'+str(len(to_label))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                pd.DataFrame(to_label).to_excel(writer)
                writer.close()
                
                complete_label = pd.concat([pd.DataFrame(to_label),pd.DataFrame(label[uncrt_pt_ind])],axis = 1)
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\query_labelled_'+str(run)+'_'+str(len(to_label))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                complete_label.to_excel(writer)
                writer.close()
                
                x_train_new = np.append(unlabel_new[uncrt_pt_ind, :], x_train_new, axis = 0)
                y_train = np.append(label[uncrt_pt_ind], y_train)
                unlabel_new = np.delete(unlabel_new, uncrt_pt_ind, axis = 0)
                label = np.delete(label, uncrt_pt_ind)
                
                run=run+1
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\train_data_'+str(run)+'_'+str(len(x_train))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                pd.concat([pd.DataFrame(x_train_new),pd.DataFrame(y_train)], axis = 1).to_excel(writer)
                writer.close()
                
                x_train = x_train_new[:,0]
                x_test = x_test_new[:,0]
                unlabel = unlabel_new[:,0]
                
                
                
######################### ENTROPY SAMPLING##################################################################

while(x_train.shape[0]<= max_sample and iter!=0):
     
    td = TfidfVectorizer(analyzer='word', stop_words=stopwords.words('french'), 
                                        token_pattern=r'\w{1,}')
    x_train_t = td.fit_transform(x_train).toarray()
    x_test_t = td.transform(x_test).toarray()
    unlabel_t = td.transform(unlabel).toarray()
    
    x_train_new = np.concatenate((x_train.reshape(-1, 1),x_train_t), axis = 1)
    x_test_new = np.concatenate((x_test.reshape(-1, 1),x_test_t), axis = 1)
    unlabel_new = np.concatenate((unlabel.reshape(-1, 1),unlabel_t), axis = 1)
    
    xgb1 = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    xgb1.fit(x_train_new[:,1:], y_train)
    
    y_pred = xgb1.predict(x_test_new[:,1:])
    cr2 = classification_report(y_test, y_pred, output_dict=True)
    xgbacc_1.append(cr2['accuracy'])
    xgbf1_1.append(cr2['1']['f1-score'])
    xgbprec_1.append(cr2['1']['precision'])
    xgbrec_1.append(cr2['1']['recall'])
    xgbauc_1.append(roc_auc_score(y_test, xgb1.predict_proba(x_test_new[:,1:])[:, 1]))
    sample_size_xgb.append(x_train_new.shape[0])
    start_xgb = time.perf_counter()
    t_xgb.append(time.perf_counter()-start_xgb)
    usage="vocab"
    mode = "Entropy Sampling"+mode1
    
    # Record Performance
    perf = pd.read_excel(r'G:\My Drive\active_learning\performance.xlsx')
    perf = perf.loc[:, ~perf.columns.str.contains('^Unnamed')]
    
    perf = perf.append({'usage': usage,
                        'run':run,
                        'samp_size': sample_size_xgb[run],
                        'accuracy' : xgbacc_1[run],
                        'f1_score': xgbf1_1[run], 
                        'recall': xgbrec_1[run], 
                        'precision': xgbprec_1[run], 
                        'auc': xgbauc_1[run], 
                        'time': t_xgb[run], 
                        'mode': mode}, ignore_index = True)
    
    writer = pd.ExcelWriter(r'G:\My Drive\active_learning\performance.xlsx',
                        engine='xlsxwriter',
                        engine_kwargs={'options': {'strings_to_urls': False}})
    perf.to_excel(writer)
    writer.close()
    y_probab = xgb1.predict_proba(unlabel_new[:,1:])[:, 0]
    y_probab_comp = 1-y_probab
    entropy = (-(y_probab*np.log2(y_probab))-(y_probab_comp*np.log2(y_probab_comp)))
    sort_index = np.argsort(entropy)[::-1]
    
    if x_train_new.shape[0]>max_sample:
        iter=0
    else:
        uncrt_pt_ind = [] # Stores all the selected samples
        uncrt_pt_ind = sort_index[0:batch_size]
        uncrt_pt_ind = uncrt_pt_ind.tolist()
        
        if  len(uncrt_pt_ind) == 0:
            iter=0
        else:
            if (len(uncrt_pt_ind)+x_train_new.shape[0]>max_sample):
                uncrt_pt_ind = uncrt_pt_ind[0:(max_sample-x_train_new.shape[0])]
                to_label = unlabel_new[uncrt_pt_ind, :]
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\query_'+str(run)+'_'+str(len(to_label))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                pd.DataFrame(to_label).to_excel(writer)
                writer.close()
                complete_label = pd.concat([pd.DataFrame(to_label),pd.DataFrame(label[uncrt_pt_ind])],axis = 1)
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\query_labelled_'+str(run)+'_'+str(len(to_label))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                complete_label.to_excel(writer)
                writer.close()
                
                x_train_new = np.append(unlabel_new[uncrt_pt_ind, :], x_train_new, axis = 0)
                y_train = np.append(label[uncrt_pt_ind], y_train)
                unlabel_new = np.delete(unlabel_new, uncrt_pt_ind, axis = 0)
                label = np.delete(label, uncrt_pt_ind)
                
                run=run+1
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\train_data_'+str(run)+'_'+str(len(x_train))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                pd.concat([pd.DataFrame(x_train_new),pd.DataFrame(y_train)], axis = 1).to_excel(writer)
                writer.close()
                
                xgbf1_1, xgbprec_1, xgbrec_1 = [], [],  [] # arrays to store performance different models
                xgbauc_1 = []
                sample_size_xgb, t_xgb = [], []
                max_sample = 1500
    
                iter=1
    
                # train model by active learning 
                xgb1 = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
                xgb1.fit(x_train_new[:,1:], y_train)
                
                y_pred = xgb1.predict(x_test_new[:,1:])
                cr2 = classification_report(y_test, y_pred, output_dict=True)
                xgbf1_1.append(cr2['1']['f1-score'])
                xgbprec_1.append(cr2['1']['precision'])
                xgbrec_1.append(cr2['1']['recall'])
                xgbauc_1.append(roc_auc_score(y_test, xgb1.predict_proba(x_test_new[:,1:])[:, 1]))
                sample_size_xgb.append(x_train_new.shape[0])
                start_xgb = time.perf_counter()
                t_xgb.append(time.perf_counter()-start_xgb)
                
                perf = pd.read_excel(r'G:\My Drive\active_learning\performance.xlsx')
                perf = perf.loc[:, ~perf.columns.str.contains('^Unnamed')]
                perf = perf.append({'usage': usage,
                                    'run':run,
                                    'samp_size': sample_size_xgb[0],
                                    'accuracy' : xgbacc_1[0],
                                    'f1_score': xgbf1_1[0], 
                                    'recall': xgbrec_1[0], 
                                    'precision': xgbprec_1[0], 
                                    'auc': xgbauc_1[0], 
                                    'time': t_xgb[run], 
                                    'mode': mode}, ignore_index = True)
    
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\performance.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                perf.to_excel(writer)
                writer.close()
                iter=0
            else:
                to_label = unlabel_new[uncrt_pt_ind, :]
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\query_'+str(run)+'_'+str(len(to_label))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                pd.DataFrame(to_label).to_excel(writer)
                writer.close()
                complete_label = pd.concat([pd.DataFrame(to_label),pd.DataFrame(label[uncrt_pt_ind])],axis = 1)
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\query_labelled_'+str(run)+'_'+str(len(to_label))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                complete_label.to_excel(writer)
                writer.close()
                
                x_train_new = np.append(unlabel_new[uncrt_pt_ind, :], x_train_new, axis = 0)
                y_train = np.append(label[uncrt_pt_ind], y_train)
                unlabel_new = np.delete(unlabel_new, uncrt_pt_ind, axis = 0)
                label = np.delete(label, uncrt_pt_ind)
                
                run=run+1
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\train_data_'+str(run)+'_'+str(len(x_train))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                pd.concat([pd.DataFrame(x_train_new),pd.DataFrame(y_train)], axis = 1).to_excel(writer)
                writer.close()
                
                x_train = x_train_new[:,0]
                x_test = x_test_new[:,0]
                unlabel = unlabel_new[:,0]

############################################################## RANDOM SAMPLING ##################################################################
import random 

# Active Learning Iteration Begins --need to repeat
while(x_train.shape[0]<= max_sample and iter!=0):
    
    td = TfidfVectorizer(analyzer='word', stop_words=stopwords.words('french'), 
                                        token_pattern=r'\w{1,}')
    x_train_t = td.fit_transform(x_train).toarray()
    x_test_t = td.transform(x_test).toarray()
    unlabel_t = td.transform(unlabel).toarray()
    
    x_train_new = np.concatenate((x_train.reshape(-1, 1),x_train_t), axis = 1)
    x_test_new = np.concatenate((x_test.reshape(-1, 1),x_test_t), axis = 1)
    unlabel_new = np.concatenate((unlabel.reshape(-1, 1),unlabel_t), axis = 1)
        
    xgb1 = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    xgb1.fit(x_train_new[:,1:], y_train)
    
    y_pred = xgb1.predict(x_test_new[:,1:])
    cr2 = classification_report(y_test, y_pred, output_dict=True)
    xgbacc_1.append(cr2['accuracy'])
    xgbf1_1.append(cr2['1']['f1-score'])
    xgbprec_1.append(cr2['1']['precision'])
    xgbrec_1.append(cr2['1']['recall'])
    xgbauc_1.append(roc_auc_score(y_test, xgb1.predict_proba(x_test_new[:,1:])[:, 1]))
    sample_size_xgb.append(x_train_new.shape[0])
    start_xgb = time.perf_counter()
    t_xgb.append(time.perf_counter()-start_xgb)
    usage="vocab"
    mode = "Random Sampling"+mode1
    
    # Record Performance
    perf = pd.read_excel(r'G:\My Drive\active_learning\performance.xlsx')
    perf = perf.loc[:, ~perf.columns.str.contains('^Unnamed')]
    
    perf = perf.append({'usage': usage,
                        'run':run,
                        'samp_size': sample_size_xgb[run],
                        'accuracy' : xgbacc_1[run],
                        'f1_score': xgbf1_1[run], 
                        'recall': xgbrec_1[run], 
                        'precision': xgbprec_1[run], 
                        'auc': xgbauc_1[run], 
                        'time': t_xgb[run], 
                        'mode': mode}, ignore_index = True)
    
    writer = pd.ExcelWriter(r'G:\My Drive\active_learning\performance.xlsx',
                        engine='xlsxwriter',
                        engine_kwargs={'options': {'strings_to_urls': False}})
    perf.to_excel(writer)
    writer.close()
    y_probab = xgb1.predict_proba(unlabel_new[:,1:])[:, 0]
    
    
    if x_train_new.shape[0]>max_sample:
        iter=0
    else:
        uncrt_pt_ind = [] # Stores all the selected samples
        uncrt_pt_ind = np.random.choice(y_probab.shape[0], batch_size, replace=False)
        uncrt_pt_ind = uncrt_pt_ind.tolist()
        
        if  len(uncrt_pt_ind) == 0:
            iter=0
        else:
            if (len(uncrt_pt_ind)+x_train_new.shape[0]>max_sample):
                uncrt_pt_ind = uncrt_pt_ind[0:(max_sample-x_train_new.shape[0])]
                to_label = unlabel_new[uncrt_pt_ind, :]
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\query_'+str(run)+'_'+str(len(to_label))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                pd.DataFrame(to_label).to_excel(writer)
                writer.close()
                complete_label = pd.concat([pd.DataFrame(to_label),pd.DataFrame(label[uncrt_pt_ind])],axis = 1)
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\query_labelled_'+str(run)+'_'+str(len(to_label))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                complete_label.to_excel(writer)
                writer.close()
                
                x_train_new = np.append(unlabel_new[uncrt_pt_ind, :], x_train_new, axis = 0)
                y_train = np.append(label[uncrt_pt_ind], y_train)
                unlabel_new = np.delete(unlabel_new, uncrt_pt_ind, axis = 0)
                label = np.delete(label, uncrt_pt_ind)
                
                run=run+1
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\train_data_'+str(run)+'_'+str(len(x_train))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                pd.concat([pd.DataFrame(x_train_new),pd.DataFrame(y_train)], axis = 1).to_excel(writer)
                writer.close()
                
                xgbf1_1, xgbprec_1, xgbrec_1 = [], [],  [] # arrays to store performance different models
                xgbauc_1 = []
                sample_size_xgb, t_xgb = [], []
                max_sample = 1500
    
                iter=1
    
                # train model by active learning 
                xgb1 = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
                xgb1.fit(x_train_new[:,1:], y_train)
                
                y_pred = xgb1.predict(x_test_new[:,1:])
                cr2 = classification_report(y_test, y_pred, output_dict=True)
                xgbf1_1.append(cr2['1']['f1-score'])
                xgbprec_1.append(cr2['1']['precision'])
                xgbrec_1.append(cr2['1']['recall'])
                xgbauc_1.append(roc_auc_score(y_test, xgb1.predict_proba(x_test_new[:,1:])[:, 1]))
                sample_size_xgb.append(x_train_new.shape[0])
                start_xgb = time.perf_counter()
                t_xgb.append(time.perf_counter()-start_xgb)
                
                perf = pd.read_excel(r'G:\My Drive\active_learning\performance.xlsx')
                perf = perf.loc[:, ~perf.columns.str.contains('^Unnamed')]
                perf = perf.append({'usage': usage,
                                    'run':run,
                                    'samp_size': sample_size_xgb[0],
                                    'accuracy' : xgbacc_1[0],
                                    'f1_score': xgbf1_1[0], 
                                    'recall': xgbrec_1[0], 
                                    'precision': xgbprec_1[0], 
                                    'auc': xgbauc_1[0], 
                                    'time': t_xgb[run], 
                                    'mode': mode}, ignore_index = True)
    
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\performance.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                perf.to_excel(writer)
                writer.close()
                iter=0
            else:
                to_label = unlabel_new[uncrt_pt_ind, :]
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\query_'+str(run)+'_'+str(len(to_label))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                pd.DataFrame(to_label).to_excel(writer)
                writer.close()
                complete_label = pd.concat([pd.DataFrame(to_label),pd.DataFrame(label[uncrt_pt_ind])],axis = 1)
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\query_labelled_'+str(run)+'_'+str(len(to_label))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                complete_label.to_excel(writer)
                writer.close()
                
                x_train_new = np.append(unlabel_new[uncrt_pt_ind, :], x_train_new, axis = 0)
                y_train = np.append(label[uncrt_pt_ind], y_train)
                unlabel_new = np.delete(unlabel_new, uncrt_pt_ind, axis = 0)
                label = np.delete(label, uncrt_pt_ind)
                
                run=run+1
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\train_data_'+str(run)+'_'+str(len(x_train))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                pd.concat([pd.DataFrame(x_train),pd.DataFrame(y_train)], axis = 1).to_excel(writer)
                writer.close()
                
                x_train = x_train_new[:,0]
                x_test = x_test_new[:,0]
                unlabel = unlabel_new[:,0]
                


################################### E-MMSIM Sampling ######################################################

from numpy.linalg import norm
min_similarity_th = 0.3
max_similarity_th = 0.6

# Active Learning Iteration Begins --need to repeat
while(x_train.shape[0]<= max_sample and iter!=0):
      
    td = TfidfVectorizer(analyzer='word', stop_words=stopwords.words('french'), 
                                        token_pattern=r'\w{1,}')

    x_train_t = td.fit_transform(x_train).toarray()
    x_test_t = td.transform(x_test).toarray()
    unlabel_t = td.transform(unlabel).toarray()
    
    x_train_new = np.concatenate((x_train.reshape(-1, 1),x_train_t), axis = 1)
    x_test_new = np.concatenate((x_test.reshape(-1, 1),x_test_t), axis = 1)
    unlabel_new = np.concatenate((unlabel.reshape(-1, 1),unlabel_t), axis = 1)
        
    xgb1 = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    xgb1.fit(x_train_new[:,1:], y_train)
    
    y_pred = xgb1.predict(x_test_new[:,1:])
    cr2 = classification_report(y_test, y_pred, output_dict=True)
    xgbacc_1.append(cr2['accuracy'])
    xgbf1_1.append(cr2['1']['f1-score'])
    xgbprec_1.append(cr2['1']['precision'])
    xgbrec_1.append(cr2['1']['recall'])
    xgbauc_1.append(roc_auc_score(y_test, xgb1.predict_proba(x_test_new[:,1:])[:, 1]))
    sample_size_xgb.append(x_train_new.shape[0])
    start_xgb = time.perf_counter()
    t_xgb.append(time.perf_counter()-start_xgb)
    usage="vocab"
    mode = "E-MMSIM Sampling"+mode1
    
    # Record Performance
    perf = pd.read_excel(r'G:\My Drive\active_learning\performance.xlsx')
    perf = perf.loc[:, ~perf.columns.str.contains('^Unnamed')]
    
    perf = perf.append({'usage': usage,
                        'run':run,
                        'samp_size': sample_size_xgb[run],
                        'accuracy' : xgbacc_1[run],
                        'f1_score': xgbf1_1[run], 
                        'recall': xgbrec_1[run], 
                        'precision': xgbprec_1[run], 
                        'auc': xgbauc_1[run], 
                        'time': t_xgb[run], 
                        'mode': mode}, ignore_index = True)
    
    writer = pd.ExcelWriter(r'G:\My Drive\active_learning\performance.xlsx',
                        engine='xlsxwriter',
                        engine_kwargs={'options': {'strings_to_urls': False}})
    perf.to_excel(writer)
    writer.close()
    y_probab = xgb1.predict_proba(unlabel_new[:,1:])[:, 0]
    
    ####### Calculate entropy ######################################
    y_probab_comp = 1-y_probab
    entropy = (-(y_probab*np.log2(y_probab))-(y_probab_comp*np.log2(y_probab_comp)))
    sort_index = np.argsort(entropy)[::-1]
    
    ########## Find misclassified samples #######################
    z = (y_test==1) & (y_pred==0)
    mc1 = [idx for idx, element in enumerate(z) if element==True]
    z = (y_test==0) & (y_pred==1)
    mc2 = [idx for idx, element in enumerate(z) if element==True]
    
    misclassified_list = mc1 + mc2
    
    if x_train_new.shape[0]>max_sample:
        iter=0
    else:
        uncrt_pt_ind = [] # Stores all the selected samples
        i = 0
        uncrt_pt_ind.append(sort_index[i])
        l = len(uncrt_pt_ind)
        
        while (l<batch_size):
            
            ########################### Hobohm & Sanders Algorithm ###########################
            for j in range(len(uncrt_pt_ind)):
                A = unlabel_new[sort_index[i],1:]       # pool vector
                B = unlabel_new[uncrt_pt_ind[j],1:]     # list vector
                cosine = np.dot(A,B)/(norm(A)*norm(B))
                if cosine >= min_similarity_th: # minimize cosine
                    select_for_rep = 0
                else:
                    select_for_rep = 1
                    
            ########################### Mis-classification Similarity ###########################
            select_for_mc = 0
            for k in range(len(misclassified_list)):
                C = x_test_new[misclassified_list[k],1:]     # list vector
                cosine = np.dot(A,C)/(norm(A)*norm(C))
                if cosine < max_similarity_th: # maximize cosine
                    select_for_mc += 0
                else:
                    select_for_mc += 1
                    break
                    
            ###################### Composite Score ##########################################        
            if ((select_for_rep == 1 & select_for_mc >= 1)):
                uncrt_pt_ind.append(sort_index[i])
            i += 1
            l = len(uncrt_pt_ind)

        
        if  len(uncrt_pt_ind) == 0:
            iter=0
        else:
            if (len(uncrt_pt_ind)+x_train_new.shape[0]>max_sample):
                uncrt_pt_ind = uncrt_pt_ind[0:(max_sample-x_train_new.shape[0])]
                to_label = unlabel_new[uncrt_pt_ind, :]
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\query_'+str(run)+'_'+str(len(to_label))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                pd.DataFrame(to_label).to_excel(writer)
                writer.close()
                complete_label = pd.concat([pd.DataFrame(to_label),pd.DataFrame(label[uncrt_pt_ind])],axis = 1)
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\query_labelled_'+str(run)+'_'+str(len(to_label))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                complete_label.to_excel(writer)
                writer.close()
                
                x_train_new = np.append(unlabel_new[uncrt_pt_ind, :], x_train_new, axis = 0)
                y_train = np.append(label[uncrt_pt_ind], y_train)
                unlabel_new = np.delete(unlabel_new, uncrt_pt_ind, axis = 0)
                label = np.delete(label, uncrt_pt_ind)
                
                run=run+1
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\train_data_'+str(run)+'_'+str(len(x_train))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                pd.concat([pd.DataFrame(x_train_new),pd.DataFrame(y_train)], axis = 1).to_excel(writer)
                writer.close()
                
                xgbf1_1, xgbprec_1, xgbrec_1 = [], [],  [] # arrays to store performance different models
                xgbauc_1 = []
                sample_size_xgb, t_xgb = [], []
                max_sample = 1500
    
                iter=1
    
                # train model by active learning 
                xgb1 = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
                xgb1.fit(x_train_new[:,1:], y_train)
                
                y_pred = xgb1.predict(x_test_new[:,1:])
                cr2 = classification_report(y_test, y_pred, output_dict=True)
                xgbf1_1.append(cr2['1']['f1-score'])
                xgbprec_1.append(cr2['1']['precision'])
                xgbrec_1.append(cr2['1']['recall'])
                xgbauc_1.append(roc_auc_score(y_test, xgb1.predict_proba(x_test_new[:,1:])[:, 1]))
                sample_size_xgb.append(x_train_new.shape[0])
                start_xgb = time.perf_counter()
                t_xgb.append(time.perf_counter()-start_xgb)
                
                perf = pd.read_excel(r'G:\My Drive\active_learning\performance.xlsx')
                perf = perf.loc[:, ~perf.columns.str.contains('^Unnamed')]
                perf = perf.append({'usage': usage,
                                    'run':run,
                                    'samp_size': sample_size_xgb[0],
                                    'accuracy' : xgbacc_1[0],
                                    'f1_score': xgbf1_1[0], 
                                    'recall': xgbrec_1[0], 
                                    'precision': xgbprec_1[0], 
                                    'auc': xgbauc_1[0], 
                                    'time': t_xgb[run], 
                                    'mode': mode}, ignore_index = True)
    
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\performance.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                perf.to_excel(writer)
                writer.close()
                iter=0
            else:
                to_label = unlabel_new[uncrt_pt_ind, :]
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\query_'+str(run)+'_'+str(len(to_label))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                pd.DataFrame(to_label).to_excel(writer)
                writer.close()
                complete_label = pd.concat([pd.DataFrame(to_label),pd.DataFrame(label[uncrt_pt_ind])],axis = 1)
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\query_labelled_'+str(run)+'_'+str(len(to_label))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                complete_label.to_excel(writer)
                writer.close()
                
                x_train_new = np.append(unlabel_new[uncrt_pt_ind, :], x_train_new, axis = 0)
                y_train = np.append(label[uncrt_pt_ind], y_train)
                unlabel_new = np.delete(unlabel_new, uncrt_pt_ind, axis = 0)
                label = np.delete(label, uncrt_pt_ind)
                
                run=run+1
                writer = pd.ExcelWriter(r'G:\My Drive\active_learning\run_'+str(run)+'\\train_data_'+str(run)+'_'+str(len(x_train))+'.xlsx',
                                    engine='xlsxwriter',
                                    engine_kwargs={'options': {'strings_to_urls': False}})
                pd.concat([pd.DataFrame(x_train_new),pd.DataFrame(y_train)], axis = 1).to_excel(writer)
                writer.close()
                
                x_train = x_train_new[:,0]
                x_test = x_test_new[:,0]
                unlabel = unlabel_new[:,0]