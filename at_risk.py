import numpy as np
import pandas as pd
import re




DATA_PATH_LIST = [ 
        'DR0008_activity_accumulator_2016_09.csv',
        'DR0008_activity_accumulator_2016-10.csv',
        'DR0008_activity_accumulator_2016-11.csv',
        'DR0008_activity_accumulator_2016-12.csv'
]

AT_RISK_LIST    = 'Std_list_atRist_2016_se1.csv'
STD_LIST        = 'Std_list_LibGate_2016_se1.csv'
LIB_GATE_RECORD = 'Gate_2016_deidentiy_filtered.csv'



def Get_Stat_Results(DATA_PATH_LIST, STD_LIST, LIB_GATE_RECORD):
    '''
        Input raw data
        Output statistical results of 2016 se1:

        Total students
        Normal students
        At_risk students
        Weekly learning behaivors comparison
    '''

    rows = [ 
        'se1_week_1', 'se1_week_2',
        'se1_week_3', 'se1_week_4',
        'se1_week_5', 'se1_week_6',
        'se1_week_7', 'se1_week_8',
        'se1_week_9', 'se1_week_10',
        'se1_week_11', 'se1_week_12',
        'se1_week_13','se1_week_14',
        'se1_week_15', 'se1_week_16'
    ]

    columns = [
        'Total_LMS',
        'Total_LIB',
    ]
    
    Data_statistical = pd.DataFrame(np.random.rand(16,2), index=rows, columns=columns).fillna(0)
    


    ### LMS statisticals
    total_LMS = []
    for path in DATA_PATH_LIST:
        print('Loading from:', path)
        df = pd.read_csv(path, sep='\t')
        dt = pd.to_datetime(df['timestamp'])
        
        test = dt.apply(lambda x: 1 if x.day >=1 and x.day <=7 else 0)
        total_LMS.append([test.sum()])
        
        test = dt.apply(lambda x: 1 if x.day >=8 and x.day <=15 else 0)
        total_LMS.append([test.sum()])
        
        test = dt.apply(lambda x: 1 if x.day >=16 and x.day <=23 else 0)
        total_LMS.append([test.sum()])
        
        test = dt.apply(lambda x: 1 if x.day >=24 and x.day <=32 else 0)
        total_LMS.append([test.sum()])
        
    total_LMS = np.array(total_LMS)
    Data_statistical['Total_LMS'] = total_LMS

    ### LIB statisticals
    total_lib = []
    df_2016_lib = pd.read_csv(LIB_GATE_RECORD)
    months = [9, 10, 11, 12]

    for mon in months:
        dt = pd.to_datetime(df_2016_lib['Column 1'])
        test = dt.apply(lambda x: 1 if x.month == mon and x.day >=1 and x.day <=7 else 0)
        total_lib.append([test.sum()])
        test = dt.apply(lambda x: 1 if x.month == mon and x.day >=8 and x.day <=15 else 0)
        total_lib.append([test.sum()])
        test = dt.apply(lambda x: 1 if x.month == mon and x.day >=16 and x.day <=23 else 0)
        total_lib.append([test.sum()])
        test = dt.apply(lambda x: 1 if x.month == mon and x.day >=24 and x.day <=32 else 0)
        total_lib.append([test.sum()])

    total_lib = np.array(total_lib)
    Data_statistical['Total_LIB'] = total_lib

    Std_list = pd.read_csv(STD_LIST)

    print('Total students: ', len(Std_list))

    Data_statistical_se1 = Data_statistical.head(13)
    Data_statistical_se1.to_csv('Demo_Data_statistical_se1.csv')
    print('Weekly learning behaivors comparison:')
    return Data_statistical_se1
    
def Draw_Co_occurence(Graph=None, threshold=5):
    import matplotlib.pyplot as plt
    import networkx as nx
    if not Graph:
        G = nx.read_weighted_edgelist("lib_co_occ_se1_week_1_5.edgelist", nodetype=int)
    else:
        G = Graph
    pos = nx.spring_layout(G, k=0.1)

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > threshold]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= threshold]


    plt.figure(figsize=(20,15))
    nx.draw_networkx_nodes(G, pos,node_size=10)
    nx.draw_networkx_edges(G, pos, edgelist=elarge,
                           width=2,edge_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=esmall,
                           width=2, alpha=0.5, edge_color='g', style='dashed')
    plt.axis('off')
    plt.show()


def Feature_Analysis(STAT_features = '2016_se1_stat_features.csv'):
    
    df_se1_stat = pd.read_csv(STAT_features, index_col=0)
    blacklist   = ['workday','weekend','workday_ExamMonth','weekend_ExamMonth','morning_ExamMonth',
        'afternoon_ExamMonth','evening_ExamMonth','overnight_ExamMonth','workday_notExamMonth',
        'weekend_notExamMonth','morning_notExamMonth','afternoon_notExamMonth','evening_notExamMonth',
        'overnight_notExamMonth','workday_firstMonth','weekend_firstMonth','morning_firstMonth',
        'afternoon_firstMonth','evening_firstMonth','overnight_firstMonth','notExamMonth'
    ]
    whitelist = ['label_atRist', 'morning','afternoon','evening', 
        'overnight', 'examMonth', 'firstMonth','total_checkin'
    ]

    df_se1_stat_new = df_se1_stat[[i for i in df_se1_stat.columns if i not in blacklist]]
    from utils import aggr

    df_se1_stat_new['Total_LOGIN_ATTEMPT']   = df_se1_stat_new.apply(aggr ,axis=1, pre_fix='LOGIN_ATTEMPT')
    df_se1_stat_new['Total_SESSION_TIMEOUT'] = df_se1_stat_new.apply(aggr ,axis=1, pre_fix='SESSION_TIMEOUT')
    df_se1_stat_new['Total_LOGOUT']          = df_se1_stat_new.apply(aggr ,axis=1, pre_fix='LOGOUT')
    df_se1_stat_new['Total_group']           = df_se1_stat_new.apply(aggr ,axis=1, pre_fix='group')
    df_se1_stat_new['Total_db']              = df_se1_stat_new.apply(aggr ,axis=1, pre_fix='db')
    df_se1_stat_new['Total_myinfo']          = df_se1_stat_new.apply(aggr ,axis=1, pre_fix='myinfo')
    df_se1_stat_new['Total_course']          = df_se1_stat_new.apply(aggr ,axis=1, pre_fix='course')
    df_se1_stat_new['Total_journal']         = df_se1_stat_new.apply(aggr ,axis=1, pre_fix='journal')
    df_se1_stat_new['Total_email']           = df_se1_stat_new.apply(aggr ,axis=1, pre_fix='email')
    df_se1_stat_new['Total_staff']           = df_se1_stat_new.apply(aggr ,axis=1, pre_fix='staff')
    df_se1_stat_new['Total_annoucements']    = df_se1_stat_new.apply(aggr ,axis=1, pre_fix='annoucements')
    df_se1_stat_new['Total_content']         = df_se1_stat_new.apply(aggr ,axis=1, pre_fix='content')
    df_se1_stat_new['Total_grade']           = df_se1_stat_new.apply(aggr ,axis=1, pre_fix='grade')
    

    df_total_stat = df_se1_stat_new[[i for i in df_se1_stat_new.columns if i.startswith('Total_') or i in whitelist]]
    
    At_risk_Mean = pd.DataFrame(df_total_stat[df_total_stat['label_atRist'] == 1].mean(), columns=['At_risk_Mean'])
    Normal_Mean = pd.DataFrame(df_total_stat[df_total_stat['label_atRist'] == 0].mean()
                               , columns=['Normal_Mean'])
    # P and F 
    import scipy.stats as stats
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    feas = ['morning', 'afternoon', 'evening', 'overnight', 'examMonth',
           'firstMonth', 'total_checkin', 'Total_LOGIN_ATTEMPT',
           'Total_SESSION_TIMEOUT', 'Total_LOGOUT', 'Total_group', 'Total_db',
           'Total_myinfo', 'Total_course', 'Total_journal', 'Total_email',
           'Total_staff', 'Total_annoucements', 'Total_content', 'Total_grade'
    ]

    Stat = pd.DataFrame(np.random.rand(len(feas), 2), index=feas, columns=['F-value', 'P-value']).fillna(0)
    for fea in feas:
        s1 = df_total_stat[df_total_stat['label_atRist'] == 1][fea]
        s2 = df_total_stat[df_total_stat['label_atRist'] == 0][fea]
        f,p = stats.f_oneway(s1, s2)
        Stat['F-value'][fea] = f
        Stat['P-value'][fea] = p

    Stat = Stat.join(At_risk_Mean, how='left')
    Stat = Stat.join(Normal_Mean, how='left')
    Stat.to_csv('Demo_Feature_analysis.csv')
    return Stat


def multiscale_reg_ext(seq, num_scale, stepsize, min_seq_length, overlap):
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)    
    seq[seq != 0] = 1;
    seq = seq.astype(int)  
    nz_index = np.argwhere(seq != 0)
    reg_sub_seq = []
    for i in range(0,num_scale):
        ## None zero sub_sequence extraction
        sub_seq = []        
        sub_seq_length = min_seq_length + i * stepsize
        buf_reg_feature = np.zeros(2 ** sub_seq_length - 1) ## will not count the number of zero subsequences.
        for j in nz_index:
            if overlap == 1:
                for k in range(int(j - sub_seq_length + 1), int(j + 1)):
                    buf_sub_seq = list(seq[k : k + sub_seq_length])
                    if len(buf_sub_seq) >= sub_seq_length:
                        sub_seq.append(list(seq[k : k + sub_seq_length]))
            else:
                k = int(j / sub_seq_length)
                sub_seq.append(list(seq[k * sub_seq_length : (k + 1) * sub_seq_length]))
        
        ## Removing unique sub_sequence
        for buf_sub_seq in sub_seq:
            num_seb_seq = sub_seq.count(buf_sub_seq)
            if num_seb_seq == 1:
                sub_seq.remove(buf_sub_seq)
            else:
                index = int(''.join(str(k) for k in buf_sub_seq),2) ## Will not be 0 because all zero subsequences have been removed.
                buf_reg_feature[index - 1] += 1
        
        reg_sub_seq.append(sub_seq)
                
        if i == 0:
            reg_feature = buf_reg_feature
        else:
            reg_feature = np.append(reg_feature,buf_reg_feature)
        
    reg_feature = list(reg_feature)    
                 
    return reg_feature


def Feature_Extraction(lib_features = 'Std_Lib_features_2016_se1.csv', DATA_PATH_LIST=None):

    lib_se1 = pd.read_csv(lib_features)
    df_se1  = lib_se1

    keys = ['first_week', 'sencond_week', 'third_week', 'forth_week',
            'fifth_week', 'sixth_week', 'seventh_week', 'eighth_week',
            'nineth_week', 'tenth_week', 'eleventh_week', 'twelfth_week',
            'thirteenth_week', 'fourteenth_week'
    ]

    from utils import extract_functions

    index = 0
    for path in DATA_PATH_LIST:
        df = pd.read_csv(path, sep='\t')
        df = df[['De-id', 'event_type', 'course_id', 'internal_handle', 'timestamp']]
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        df['first_week']  = df['timestamp'].apply(lambda x: 1 if x.day >=1 and x.day <=7 else 0)
        df['second_week'] = df['timestamp'].apply(lambda x: 1 if x.day >=8 and x.day <=15 else 0)
        df['third_week']  = df['timestamp'].apply(lambda x: 1 if x.day >=16 and x.day <=23 else 0)
        df['forth_week']  = df['timestamp'].apply(lambda x: 1 if x.day >=24 and x.day <=32 else 0)

        df_first_week  = df[df['first_week'] == 1]
        df_second_week = df[df['second_week'] == 1]
        df_third_week  = df[df['third_week'] == 1]
        df_forth_week  = df[df['forth_week'] == 1]


        df_tmp = extract_functions(df_first_week, keys[index*4 + 0])
        df_se1 = pd.merge(df_se1, df_tmp, on = ['MASKED_STUDENT_ID'], how='left').fillna(0)

        df_tmp = extract_functions(df_second_week, keys[index*4 + 1])
        df_se1 = pd.merge(df_se1, df_tmp, on = ['MASKED_STUDENT_ID'], how='left').fillna(0)

        if index == 3:
            pass
        else:
            df_tmp = extract_functions(df_third_week, keys[index*4 + 2])
            df_se1 = pd.merge(df_se1, df_tmp, on = ['MASKED_STUDENT_ID'], how='left').fillna(0)

            df_tmp = extract_functions(df_forth_week, keys[index*4 + 3])
            df_se1 = pd.merge(df_se1, df_tmp, on = ['MASKED_STUDENT_ID'], how='left').fillna(0)

        index +=1

    df_se1.to_csv('Demo_2016_se1_stat_features.csv')




def At_Risk_Prediction(first_n_weeks=7, mode='train', num_of_at_risk=500):
    df_se1_stat_reg_soho = pd.read_csv('Demo_STAT_Reg_SoH.csv', index_col=0)

    from utils import select_time

    tmp_df_se1_features = df_se1_stat_reg_soho[[i for i in df_se1_stat_reg_soho.columns if i != 'label_atRist' and i != 'MASKED_STUDENT_ID']]
    df_se1_features = tmp_df_se1_features[select_time(list(tmp_df_se1_features.columns), first_weeks=first_n_weeks)]
    df_se1_labels = df_se1_stat_reg_soho['label_atRist']
    df_se1_namelist = df_se1_stat_reg_soho['MASKED_STUDENT_ID']
    labels = df_se1_labels.apply(lambda x: str(x))


    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    from imblearn.over_sampling import SMOTE
    import pickle

    if mode == 'train':
        X_train, X_test, y_train, y_test = train_test_split(
            df_se1_features, df_se1_labels, test_size=0.2, stratify=df_se1_labels, random_state=42)
        X_resampled_smote, y_resampled_smote = SMOTE(
            random_state=0, sampling_strategy='auto',k_neighbors=10).fit_sample(X_train, y_train)
        clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42)
        clf.fit(X_resampled_smote, y_resampled_smote.ravel())
        
        ''' feature importance selection
        X_train = X_resampled_smote[:, clf.feature_importances_>0.001]
        X_test  = X_test.values[:, clf.feature_importances_>0.001]
        clf.fit(X_train, y_resampled_smote.ravel())
        '''
        
        y_predprob = clf.predict_proba(X_test)[:,1]
        with open('clf_first_'+str(first_n_weeks)+'_weeks.pickle', 'wb') as f:
            pickle.dump(clf, f)

    
    with open('clf_first_'+str(first_n_weeks)+'_weeks.pickle', 'rb') as f:
        clf = pickle.load(f)
        y_predprob = clf.predict_proba(df_se1_features)[:,1]
    
    df_test = pd.DataFrame(df_se1_namelist).join(pd.DataFrame(y_predprob, columns=['Probs']))
    df_test.sort_values("Probs",inplace=True)
    df_test.tail(num_of_at_risk)

    return df_test.tail(num_of_at_risk)



def Baseline_Features(STAT_Features_Path = '2016_se1_stat_features.csv'):
    df_se1_stat = pd.read_csv(STAT_Features_Path, index_col=0)
    
    ## add history gpa:
    at_risk_2015                     = pd.read_csv('Std_list_normal_2015_se1.csv')
    normal_2015                      = pd.read_csv('Std_list_atRist_2015_se1.csv')
    at_risk_2015.columns             = ['MASKED_STUDENT_ID', '2015_se1_CUM_GPA']
    normal_2015.columns              = ['MASKED_STUDENT_ID', '2015_se1_CUM_GPA']
    his_2015_se1                     = pd.concat([at_risk_2015, normal_2015])
    his_2015_se1['2015_at_risk_se1'] = his_2015_se1['2015_se1_CUM_GPA'].apply(lambda x: 1 if x < 2.0 else -1).fillna(0)

    at_risk_2015                     = pd.read_csv('Std_list_normal_2015_se2.csv')
    normal_2015                      = pd.read_csv('Std_list_atRist_2015_se2.csv')
    at_risk_2015.columns             = ['MASKED_STUDENT_ID', '2015_se2_CUM_GPA']
    normal_2015.columns              = ['MASKED_STUDENT_ID', '2015_se2_CUM_GPA']
    his_2015_se2                     = pd.concat([at_risk_2015, normal_2015])
    his_2015_se2['2015_at_risk_se2'] = his_2015_se2['2015_se2_CUM_GPA'].apply(lambda x: 1 if x < 2.0 else -1).fillna(0)
    his_2015                         = pd.merge(his_2015_se1, his_2015_se2, on='MASKED_STUDENT_ID', how='left').fillna(0)

    df_se1_stat = pd.merge(df_se1_stat, his_2015, on='MASKED_STUDENT_ID', how='left').fillna(0)
    df_se1_stat.to_csv('Demo_stat_totals.csv')
    return df_se1_stat


def Data_Augmentation(df_features=None):
    '''
        No need to augment data in the validation and test procedure
    '''
    pass



def Extract_Regularity(num_scale=4, data_path_list=DATA_PATH_LIST):
    df_se1 = pd.read_csv('2016_se1_lib_lms.csv')
    df_se1.head()
    df_se1_features = df_se1['MASKED_STUDENT_ID']
    print(df_se1_features.shape)

    lms_functions = ['COURSE_ACCESS', 'PAGE_ACCESS', 'LOGIN_ATTEMPT', 'SESSION_TIMEOUT', 'LOGOUT']
    index = 0

    from utils import get_funtions_seq
    for fun in lms_functions:
        print('getting seq of', fun, '...')
        df = get_funtions_seq(data_path_list=data_path_list, function=fun)
        features = [i for i in list(df.columns) if i.endswith(fun)]
        features.append('MASKED_STUDENT_ID')
        df_se1_features = pd.merge(df_se1_features, df[features], on='MASKED_STUDENT_ID', how='left').fillna(0)
        print(df_se1_features.shape)

    df_se1_features.to_csv('se1_LMS_function_seq.csv')
    df_se1_features_list = df_se1_features[[i for i in df_se1_features if not i.startswith('reg_')]]
    
    year = 2016
    semester = 1

    stepsize = 1
    min_seq_length = 1
    overlap = 0

    year_libData = year
    semester_startMonth = '09'
    semester_endMonth = '12'
    
    from utils import multiscale_reg_ext


    df_se1_features_list['reg_'+ str(num_scale) +'_COURSE_ACCESS'] = df_se1_features_list['total_list_COURSE_ACCESS'].apply(
        lambda x: multiscale_reg_ext(x, num_scale, stepsize, min_seq_length, overlap))

    df_se1_features_list['reg_'+ str(num_scale) +'_PAGE_ACCESS'] = df_se1_features_list['total_list_PAGE_ACCESS'].apply(
        lambda x: multiscale_reg_ext(x, num_scale, stepsize, min_seq_length, overlap))

    df_se1_features_list['reg_'+ str(num_scale) +'_COURSE_ACCESS'] = df_se1_features_list['total_list_COURSE_ACCESS'].apply(
        lambda x: multiscale_reg_ext(x, num_scale, stepsize, min_seq_length, overlap))

    df_se1_features_list['reg_'+ str(num_scale) +'_LOGIN_ATTEMPT'] = df_se1_features_list['total_list_LOGIN_ATTEMPT'].apply(
        lambda x: multiscale_reg_ext(x, num_scale, stepsize, min_seq_length, overlap))

    df_se1_features_list['reg_'+ str(num_scale) +'_SESSION_TIMEOUT'] = df_se1_features_list['total_list_SESSION_TIMEOUT'].apply(
        lambda x: multiscale_reg_ext(x, num_scale, stepsize, min_seq_length, overlap))


    df_lib_seq = pd.read_csv('Std_Lib_sequence_day_2016_se1.csv')
    df_lib_seq = df_lib_seq[['0','1']]
    df_lib_seq = df_lib_seq.rename(columns= {'0': 'MASKED_STUDENT_ID', '1':'lib_total_list'})
    
    from utils import split_str

    df_lib_seq['lib_total_list'] = df_lib_seq['lib_total_list'].apply(split_str)
    df_lib_seq['reg_'+ str(num_scale) +'_LIB'] = df_lib_seq['lib_total_list'].apply(
        lambda x: multiscale_reg_ext(x, num_scale, stepsize, min_seq_length, overlap))

    df_se1_features_all = pd.merge(df_se1_features_list, df_lib_seq, on='MASKED_STUDENT_ID', how='right')
    df_se1_features_all.to_csv('Demo_Se1_seq_feature_scale_'+str(num_scale)+'.csv')


def Combine_Regularity(regular_scale=4, df_se1_stat=None):
    df_se1_stat = pd.read_csv('Demo_stat_totals.csv', index_col=0)
    df_se1_reg = pd.read_csv('Demo_Se1_seq_feature_scale_'+str(regular_scale)+'.csv', index_col=0)
    
    from utils import str_to_list, select

    for col in df_se1_reg.columns:
        if col.startswith('reg_'):
            df_se1_reg[col] = df_se1_reg[col].apply(str_to_list)

    df_se1_reg = df_se1_reg[[i for i in df_se1_reg.columns if i.startswith('reg_') or i == 'MASKED_STUDENT_ID']]



    for col in df_se1_reg.columns:
        if col.startswith('reg_'):
            for i in range(len(df_se1_reg[col][0])):
                df_se1_reg[col + '_' + str(i)] = df_se1_reg[col].apply(lambda x: select(x, i))

    blacklist = ['reg_'+regular_scale+'_COURSE_ACCESS', 
                 'reg_'+regular_scale+'_PAGE_ACCESS',
                 'reg_'+regular_scale+'_LOGIN_ATTEMPT',
                 'reg_'+regular_scale+'_SESSION_TIMEOUT',
                 'reg_'+regular_scale+'_LIB']
    df_se1_reg = df_se1_reg[[i for i in df_se1_reg if i not in blacklist]]

    df_se1_stat_reg = pd.merge(df_se1_stat, df_se1_reg, on='MASKED_STUDENT_ID', how='left').fillna(0)
    df_se1_stat_reg.to_csv('Demo_baseline_Reg.csv')
    

def Extract_Social_Homophily():
    pass


def Merge_Social_Homophily(STAT_Reg_path='Demo_baseline_Reg.csv', SoH_path='se1_weekly_node_embeddings_30_sec_2s.csv'):
    df_se1_stat_reg = pd.read_csv(STAT_Reg_path, index_col=0)
    df_se1_soho = pd.read_csv(SoH_path, index_col=0)
    df_se1_soho = df_se1_soho[[i for i in df_se1_soho.columns if i != 'label_atRist']]

    from utils import str_to_list, select

    for col in df_se1_soho.columns:
        if col !='MASKED_STUDENT_ID':
            df_se1_soho[col] = df_se1_soho[col].apply(str_to_list)

            
    for col in df_se1_soho.columns:
        if col.startswith('week_'):
            for i in range(64):
                df_se1_soho[col + '_' + str(i)] = df_se1_soho[col].apply(lambda x: select(x, i))

    blacklist = ['week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6', 'week_7', 'week_8', 'week_9', 'week_10', 'week_11', 'week_12','week_13', 'week_14']
    df_se1_soho = df_se1_soho[[i for i in df_se1_soho.columns if i not in blacklist]]
    print(df_se1_soho.shape)

    df_se1_stat_reg_soho = pd.merge(df_se1_stat_reg, df_se1_soho, on='MASKED_STUDENT_ID', how='left').fillna(0)
    df_se1_stat_reg_soho.to_csv('Demo_STAT_Reg_SoH.csv')



def Draw_Regularity(DATA_PATH='Std_Lib_sequence_day_2016_se1.csv'):
    
    import matplotlib.pyplot as plt  
    import seaborn as sns  
    import matplotlib.patches as mpatches
    
    from utils import split_str
    
    df_lib_seq = pd.read_csv(DATA_PATH)
    df_lib_seq = df_lib_seq[['0','1']]
    df_lib_seq = df_lib_seq.rename(columns= {'0': 'MASKED_STUDENT_ID', '1':'lib_total_list'})
    
    num_scale=4
    year = 2016
    semester = 1

    stepsize = 1
    min_seq_length = 1
    overlap = 0

    year_libData = year
    semester_startMonth = '09'
    semester_endMonth = '12'

    df_lib_seq['lib_total_list'] = df_lib_seq['lib_total_list'].apply(split_str)
    df_lib_seq['reg_'+ str(num_scale) +'_LIB'] = df_lib_seq['lib_total_list'].apply(
            lambda x: multiscale_reg_ext(x, num_scale, stepsize, min_seq_length, overlap))
    
    at_risks = pd.read_csv('Std_list_atRist_2016_se1.csv')
    at_risks['Lable'] = 1
    new_data = pd.merge(df_lib_seq, at_risks, on='MASKED_STUDENT_ID', how='left').fillna(0)
    regs = new_data[['reg_4_LIB', 'Lable']]

    cnt = 0
    while(True):
        try:
            regs['reg_' + str(cnt+1)] = regs['reg_4_LIB'].map(lambda x: x[cnt])
            cnt += 1
        except:
            break
    regs = regs[[i for i in regs.columns if i != 'reg_4_LIB']]
    normal = pd.DataFrame(regs[regs['Lable']==0.0].mean())
    at_risk = pd.DataFrame(regs[regs['Lable']==1.0].mean())    
    
    plt.figure(figsize=(20,9))
    sns.set(font_scale=1.8, style='whitegrid', rc={'lines.markersize':10})
    p1 = sns.barplot(data = normal.T[[i for i in normal.index if i is not 'Lable']], color='#7CFC00')

    p2 = sns.barplot(data = at_risk.T[[i for i in at_risk.index if i is not 'Lable']], color='#FF0000') 
    plt.xlim((-1, 26))
    plt.xticks(list(range(26)), 
               list(['1', '01', '10', '11',
                     '001', '010', '011', '100', '101', '110', '111',
                     '0001', '0010', '0011', '0100', '0101', '0110', '0111', '1000',
                     '1001', '1010', '1011', '1100', '1101', '1110', '1111',
                    ]
               ), fontsize=20, rotation=300)

    labels = ['at_risk', 'normal']
    color = ['#FF0000', '#7CFC00']
    patches = [ mpatches.Patch(color=color[i], label="{:s}".format(labels[i]) ) for i in range(len(color)) ] 
    plt.legend(handles=patches)
    plt.ylabel('Avg Number of Patterns')
    plt.xlabel('Regularity Patterns')















