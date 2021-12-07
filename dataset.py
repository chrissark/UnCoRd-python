import random
import os
import json


def get_questions_and_answers(questions_dir, num_questions=None):

    questions_file = 'CLEVR_val_questions.json'
    with open(questions_file, 'r') as f:
        q_info = json.load(f)

    q_info = q_info['questions']
    answers = []
    with open('val_questions_none.graph') as f:
        graph_txt = f.read()
    graph_text_list = graph_txt.split('\n')
    questions = []
    for i, ques in enumerate(q_info):
        #if '<is_plural>' not in graph_text_list[i] and '<SuperNode>' not in graph_text_list[i]:
        if '<F>' in graph_text_list[i]:
            questions.append(ques)

    if not num_questions:
        num_questions = len(questions)
    sample = random.sample(questions, num_questions)
    for ques in sample:
        answers.append(ques['answer'])

    return sample, answers


def get_question_by_idx(ques_id):
    questions_file = 'CLEVR_val_questions.json'
    with open(questions_file, 'r') as f:
        q_info = json.load(f)

    q_info = q_info['questions']

    question = q_info[ques_id]
    answer = question['answer']

    return question, answer