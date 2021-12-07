import argparse
import json
from UnCoRd import UnCoRd
from dataset import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UnCoRd validation.')
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--questions_dir', type=str)
    parser.add_argument('--scenes_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--num_questions', type=int, default=20)
    args = parser.parse_args()
    questions, gT = get_questions_and_answers(args.questions_dir, args.num_questions)
    model = UnCoRd()
    accuracy = 0
    wrong_idx = []
    for i, question in enumerate(questions):
        answer = model.get_answer(args.image_dir, question)
        if answer != gT[i]:
            wrong_idx.append(question['question_index'])
        else:
            accuracy += 1
        print(f"Question: {question['question']}")
        print(f"Model answer: {answer}")
        print(f"Ground truth: {gT[i]}")
        print(f"Image id: {question['image_index']}")
        print('\n')

    accuracy = accuracy / len(questions)

    print(f'Accuracy: {accuracy}')

    with open('val_questions_none.graph') as f:
        graph_txt = f.read()
    graph_text_list = graph_txt.split('\n')

    for i in wrong_idx:
        print(f'{i}: {graph_text_list[i]}')
    '''
    question, gT = get_question_by_idx(32566)
    model = UnCoRd()
    answer = model.get_answer(args.image_dir, question)
    print(f"Question: {question['question']}")
    print(f"Model answer: {answer}")
    print(f"Ground truth: {gT}")
    '''



