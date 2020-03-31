def int_handle_cnt(internel_handle_list, df_int_handle, name, PRE_FIX):
    df_temp = df_int_handle[df_int_handle['internal_handle'].isin(internel_handle_list)]
    df_temp = df_temp.groupby(['De-id']).count().reset_index('De-id')
    df_temp.columns = ['De-id', PRE_FIX + name]
    return df_temp


def extract_functions(df, PRE_FIX):
    
    import numpy as np
    import pandas as pd

    df_t = df[(df['event_type']=='PAGE_ACCESS') |
              (df['event_type']=='COURSE_ACCESS') |
              (df['event_type']=='LOGIN_ATTEMPT') |
              (df['event_type']=='SESSION_TIMEOUT') |
              (df['event_type']=='LOGOUT')]


    df_evt = df_t[['De-id', 'event_type']]
    df_login = df_evt[df_evt['event_type'] == 'LOGIN_ATTEMPT'].groupby(['De-id']).count().reset_index('De-id')
    df_login.columns = ['De-id', PRE_FIX + 'LOGIN_ATTEMPT']

    df_se_out = df_evt[df_evt['event_type'] == 'SESSION_TIMEOUT'].groupby(['De-id']).count().reset_index('De-id')
    df_se_out.columns = ['De-id', PRE_FIX + 'SESSION_TIMEOUT']

    df_logout = df_evt[df_evt['event_type'] == 'LOGOUT'].groupby(['De-id']).count().reset_index('De-id')
    df_logout.columns = ['De-id', PRE_FIX + 'LOGOUT']

    df_all = df_login
    df_all = pd.merge(df_all, df_se_out, on='De-id', how='left')
    df_all = pd.merge(df_all, df_logout, on='De-id', how='left')

    df_int_handle = df_t[['De-id', 'internal_handle']]

    group_list        = ['groups', 'cp_group_create_self_groupmem', 'group_file', 'group_file', 'group_forum', 'groups_sign_up', 'agroup', 'group_blogs','group_task_create', 'group_task_view','cp_group_edit_self_groupmem','group_file_add', 'group_email', 'cp_groups', 'cp_groups_settings','edit_group_blog_entry', 'db_forum_collection_group', 'group_tasks', 'group_journal','group_virtual_classroom', 'add_group_journal_entry','email_all_groups', 'edit_group_journal_entry', 'email_select_groups', 'add_group_blog_entry']
    db_list           = ['discussion_board_entry', 'db_thread_list_entry', 'discussion_board', 'db_thread_list','db_collection', 'db_collection_group', 'db_collection_entry', 'db_thread_list_group']
    myinfo_list       = ['my_inst_personal_info', 'my_inst_personal_settings','my_inst_personal_edit', 'my_inst_myplaces_settings','my_tasks', 'my_task_create', 'my_email_courses','my_task_view', 'my_announcements']
    course_list       = ['course_tools_area', 'course_task_view', 'enroll_course', 'classic_course_catalog']
    journal_list      = ['journal', 'journal_view', 'view_draft_journal_entry',  'add_journal_entry', 'edit_journal_entry']
    email_list        = ['send_email', 'email_all_instructors', 'email_all_students', 'email_select_students','email_all_users',  'email_select_groups','email_all_groups']
    staff_list        = ['staff_information', 'cp_staff_information']
    annoucements_list = ['my_announcements', 'announcements_entry', 'announcements', 'cp_announcements']
    content_list      = ['content', 'cp_content']
    grade_list        = ['check_grade']

    df_group        = int_handle_cnt(group_list, df_int_handle, 'group', PRE_FIX)
    df_db           = int_handle_cnt(db_list, df_int_handle, 'db', PRE_FIX)
    df_myinfo       = int_handle_cnt(myinfo_list, df_int_handle, 'myinfo', PRE_FIX)
    df_course       = int_handle_cnt(course_list, df_int_handle, 'course', PRE_FIX)
    df_journal      = int_handle_cnt(journal_list, df_int_handle, 'journal', PRE_FIX)
    df_email        = int_handle_cnt(email_list, df_int_handle, 'email', PRE_FIX)
    df_staff        = int_handle_cnt(staff_list, df_int_handle, 'staff', PRE_FIX)
    df_annoucements = int_handle_cnt(annoucements_list, df_int_handle, 'annoucements', PRE_FIX)
    df_content      = int_handle_cnt(content_list, df_int_handle, 'content', PRE_FIX)
    df_grade        = int_handle_cnt(grade_list, df_int_handle, 'grade', PRE_FIX)

    dfs = [df_group, df_db, df_myinfo, df_course, df_journal, df_email, df_staff, df_annoucements, df_content, df_grade]

    for df in dfs:
        df_all = pd.merge(df_all, df, on='De-id', how='left')   

    df_all = df_all.rename(columns={'De-id':'MASKED_STUDENT_ID'})
    return df_all

def get_funtions_seq(df_temp=None, function=None):
    if df_temp is None:
        df_temp = get_all_seq(data_path_list, function)
    
    df_temp = df_temp.rename(columns={'De-id':'MASKED_STUDENT_ID'})
    return df_temp

def aggr(row, pre_fix):
    ans = 0
    for i in row.index:
        if i.endswith(pre_fix):
            ans += row[i]
    return ans



def select_time(features, first_weeks):
    pre_fix_list = ['first_', 'sencond_', 'third_', 'forth_',
                    'fifth_', 'sixth_', 'seventh_', 'eighth_',
                    'nineth_', 'tenth_', 'eleventh_', 'twelfth_',
                    'thirteenth_', 'fourteenth_']
    blacklist = pre_fix_list[first_weeks:]
    feature_list = []
    for i in features:
        flag = 0
        for j in blacklist:
            if not i.startswith(j):
                pass
            else:
                flag = 1
        if flag == 0:
            feature_list.append(i)
    return feature_list

def select_feature(features, first_weeks):
    pre_fix_list_1 = ['first_', 'second_', 'third_', 'forth_',
                    'fifth_', 'sixth_', 'seventh_', 'eighth_',
                    'nineth_', 'tenth_', 'eleventh_', 'twelfth_','thirteenth_', 'forteenth_']
    pre_fix_list_2 = ['week_1_', 'week_2_', 'week_3_', 'week_4_',
                    'week_5_', 'week_6_', 'week_7_', 'week_8_',
                    'week_9_', 'week_10_', 'week_11_', 'week_12_','week_13_', 'week_14_']
    blacklist = pre_fix_list_1[first_weeks:]
    blacklist += pre_fix_list_2[first_weeks:]
#     print(blacklist)
    feature_list = []
    for i in features:
        flag = 0
        for j in blacklist:
            if not i.startswith(j):
                pass
            else:
                flag = 1
        if flag == 0:
            feature_list.append(i)
    return feature_list

def multiscale_reg_ext(seq, num_scale, stepsize, min_seq_length, overlap):

    import numpy as np
    import pandas as pd

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


def create_day_seq(days, length):

    tmp_dict = {}
    for day in days:
        try:
            tmp_dict[day] += 1
        except:
            tmp_dict[day] = 1
    res = [0]*(length+1)
    for k,v in tmp_dict.items():
        res[k] = v
    return res

def extract_function_seq(data_path, function, month='9', within_day=False):

    import numpy as np
    import pandas as pd
    
    df                   = pd.read_csv(data_path, sep='\t')
    df_temp              = df[df['event_type'] == function][['De-id', 'timestamp']]
    df_temp['timestamp'] = df_temp['timestamp'].apply(pd.to_datetime)
    df_temp['day']       = df_temp['timestamp'].apply(lambda x: x.day)
    df_day_list          = df_temp[['De-id', 'day']].groupby('De-id').agg(create_day_seq, length=df_temp['day'].nunique()).reset_index()
    df_day_list.columns  = ['De-id', month + '_day_list']
    return df_day_list

def get_all_seq(data_path_list, function):

    import numpy as np
    import pandas as pd

    first_flag = 1
    for data_path in data_path_list:
        df_day_list = extract_function_seq(data_path, function, data_path.split('.')[0][-2:])
        if first_flag:
            df_all = df_day_list.copy()
            first_flag = 0
        else:
            df_all = pd.merge(df_all, df_day_list, on='De-id', how='left')

    df_all['09_day_list'] = df_all['09_day_list'].fillna(0)
    df_all['10_day_list'] = df_all['10_day_list'].fillna(0)
    df_all['11_day_list'] = df_all['11_day_list'].fillna(0)
    df_all['12_day_list'] = df_all['12_day_list'].fillna(0)

    df_all['09_day_list'] = df_all['09_day_list'].apply(lambda x: [0]*31 if x == 0 else x)
    df_all['10_day_list'] = df_all['10_day_list'].apply(lambda x: [0]*32 if x == 0 else x)
    df_all['11_day_list'] = df_all['11_day_list'].apply(lambda x: [0]*31 if x == 0 else x)
    df_all['12_day_list'] = df_all['12_day_list'].apply(lambda x: [0]*32 if x == 0 else x)

    df_all['total_list_' + function]  = df_all.apply(lambda row: row['09_day_list'][1:] +  row['10_day_list'][1:]
                                           + row['11_day_list'][1:] +  row['12_day_list'][1:], axis=1)
    return df_all

def add_at_risk_label(df_all):

    import numpy as np
    import pandas as pd

    at_rsk_label            = pd.read_csv('Std_list_atRist_2016_se1.csv')
    at_rsk_label['at_risk'] = at_rsk_label['CUM_GPA'].apply(lambda x: '1' if x <= 2.0 else '0')
    at_rsk_label.columns    = ['De-id', 'CUM_GPA', 'at_risk']
    df_all                  = pd.merge(df_all, at_rsk_label, on='De-id', how='left')
    df_all['at_risk']       = df_all['at_risk'].fillna('0')
    return df_all

def get_funtions_seq(df_temp=None, function=None, data_path_list=None):
    if df_temp is None:
        df_temp = get_all_seq(data_path_list, function)
    
    df_temp = df_temp.rename(columns={'De-id':'MASKED_STUDENT_ID'})
    return df_temp

def select(x, i):
    try:
        return x[i]
    except:
        return -1

def str_to_list(s1):
    try:
        return [float(i) for i in s1[1:-1].split(',')]
    except:
        return 0.0

def split_str(s1):
    try:
        s = s1[2:-2].split()
        return [float(i) for i in s]
    except:
        return 0







