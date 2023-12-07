# 2_1 How to add an index field in python

subjects = ['Data Science', 'Discrete Structure', 'Operating System', 'Computer Graphics', 'Java']
indexed_subjects = list(enumerate(subjects, start=1))  # Starting index from 1
for index, subject in indexed_subjects:
    print(f'Index: {index}, Subject: {subject}')
